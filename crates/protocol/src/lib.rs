//! Shared-memory contract between the three processes that make up dlssnr:
//!
//!   the Linux Vulkan layer   captures the frame, runs the composition, presents the result
//!   the Windows helper       owns the NGX model and runs it (today under Wine/Proton; a
//!                            future native-Linux NGX helper drops in here unchanged, since
//!                            this crate is the whole contract either side needs to agree on)
//!   the GTK4 GUI / CLI       write settings and read status
//!
//! Everything here is plain atomics in a file mapping, so no side needs the others'
//! toolchain and a process dying leaves the others reading a consistent — if stale —
//! picture. `ShmHeader` is `#[repr(C)]` and built entirely from `AtomicU32` and
//! `UnsafeCell<[u8; N]>` fields specifically so its layout matches what a C11/C++
//! `std::atomic<uint32_t>` of the same field, in the same position, would produce —
//! that's what makes a single mmap'd region a valid contract between two different
//! toolchains/processes in the first place.
//!
//! Layout of the mapping:
//!
//!   `[0, HEADER_BYTES)`                 `ShmHeader`
//!   `[HEADER_BYTES, +MAX_FRAME)`        the proxy the layer encoded, for the model
//!   `[HEADER_BYTES + MAX_FRAME, +MAX_FRAME)`  the model's answer, for the composition
//!
//! This is a from-scratch protocol for a from-scratch implementation — it is not wire
//! compatible with, and never attaches to, a mapping left behind by any other DLSS
//! neural-rendering project. `SHM_MAGIC` exists specifically so a stale mapping from
//! anything else is always rejected and reinitialized rather than half-read.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod enums;
mod header;
#[cfg(unix)]
pub mod mapping;
mod path;

pub use header::{load64, store64, PassControl, PassTuning, ShmHeader};
pub use path::{shm_default_path, shm_runtime_dir};

/// Identifies a dlssnr mapping. Bumped only if the protocol is ever forked into an
/// incompatible variant; a mismatch here means "not our mapping at all", not "an older
/// version of our mapping" — that distinction is `SHM_VERSION`'s job.
pub const SHM_MAGIC: u32 = u32::from_le_bytes(*b"DSN1");

/// The header layout version. A mismatch (matching magic, different version) means
/// another process in the chain is out of date; callers should log loudly and
/// reinitialize rather than half-read a header laid out differently than they expect.
pub const SHM_VERSION: u32 = 1;

pub const MAX_W: u32 = 7680;
pub const MAX_H: u32 = 4320;

/// Eight bytes a pixel: the float16 HDR proxy needs them, and the 8-bit path simply
/// uses the first half of each region. The mapping is file-backed and sparse, so an
/// SDR session never commits the second half.
pub const MAX_FRAME: usize = MAX_W as usize * MAX_H as usize * 8;

pub const HEADER_BYTES: usize = 65536;

/// The ceiling on how many times the model runs over one frame, and what the slider
/// offers unless the ceiling is lifted.
pub const MAX_PASSES: usize = 30;
pub const DEFAULT_MAX_PASSES: u32 = 5;

pub const REASON_BYTES: usize = 192;
pub const NAME_BYTES: usize = 128;

/// Total size of the mapping: the header plus both pixel regions.
pub const fn shm_total_bytes() -> usize {
    HEADER_BYTES + MAX_FRAME * 2
}

/// Byte offset of the proxy region (the frame the layer hands the model) within the
/// mapping. Both sides derive this the same way rather than hardcoding `HEADER_BYTES`
/// separately, so a future header resize can't silently desync them.
pub const fn proxy_offset() -> usize {
    HEADER_BYTES
}

/// Byte offset of the answer region (the model's raw output, for the composition pass)
/// within the mapping.
pub const fn answer_offset() -> usize {
    HEADER_BYTES + MAX_FRAME
}
