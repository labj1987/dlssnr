//! Opens (or creates) the mapping on Linux and hands back a live `&ShmHeader` — the
//! "just attach and read/write settings" case the GUI and CLI both need, as opposed
//! to `dlssnr_layer`'s own copy of this same open/create/mmap dance (kept separate
//! there because it's entangled with that crate's request/response round-trip state
//! machine, which the GUI/CLI have no reason to duplicate or depend on).
//!
//! Linux-only (`cfg(unix)`, though in practice only ever built for Linux in this
//! workspace) — the Windows-side equivalent is `dlssnr_helper::shm`, a different
//! enough set of Win32 APIs that sharing this module across the OS boundary would
//! cost more in `cfg` noise than it would save in shared logic.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use crate::{shm_default_path, shm_total_bytes, ShmHeader, HEADER_BYTES};

pub struct Mapping {
    _fd: OwnedFd,
    header: *mut ShmHeader,
}

// SAFETY: same reasoning as `ShmHeader` itself being `Sync` -- every access through
// `header` goes through its own atomics/seqlock-guarded accessors.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    pub fn header(&self) -> &ShmHeader {
        // SAFETY: `header` was mmap'd for at least `HEADER_BYTES` in `open()` and is
        // never unmapped before `self` is dropped (there is no `Drop` impl that
        // unmaps it -- deliberately: the mapping is meant to outlive this handle's
        // owner for as long as the process runs, matching a settings GUI/CLI's
        // lifetime expectations, so leaking the mapping until process exit is fine).
        unsafe { &*self.header }
    }
}

/// Opens the mapping at `$DLSSNR_SHM` (or the default runtime path), creating it if
/// necessary. Returns `None` on any I/O failure (permissions, disk full, etc.) — there
/// is nothing a caller can usefully do about those beyond reporting them.
pub fn open() -> Option<Mapping> {
    let path = std::env::var("DLSSNR_SHM").ok().filter(|s| !s.is_empty()).unwrap_or_else(shm_default_path);
    open_at(&path)
}

fn open_at(path: &str) -> Option<Mapping> {
    if let Some(slash) = path.rfind('/') {
        let dir = &path[..slash];
        if !dir.is_empty() {
            let _ = std::fs::create_dir_all(dir);
        }
    }

    let c_path = CString::new(path).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated C string for the call's duration.
    let raw_fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW, 0o600) };
    if raw_fd < 0 {
        return None;
    }
    // SAFETY: `raw_fd` was just returned by a successful `open()` and is not owned
    // anywhere else yet.
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

    let total = shm_total_bytes();
    // SAFETY: `st` is a plain out-parameter; zero-initializing it is always valid.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let needs_truncate = unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 || (st.st_size as usize) < total;
    if needs_truncate && unsafe { libc::ftruncate(fd.as_raw_fd(), total as libc::off_t) } != 0 {
        return None;
    }

    // SAFETY: `fd` is open and sized to at least `HEADER_BYTES`; mapping only that
    // many bytes is always in-bounds. The mapping is intentionally never unmapped
    // (see `Mapping::header`'s doc comment), so the returned pointer stays valid for
    // as long as anything derived from it is used.
    let map = unsafe {
        libc::mmap(std::ptr::null_mut(), HEADER_BYTES, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), 0)
    };
    if map == libc::MAP_FAILED {
        return None;
    }

    let header = map.cast::<ShmHeader>();
    // SAFETY: just mapped above, `HEADER_BYTES` is large enough for `ShmHeader`.
    let hdr = unsafe { &*header };
    if !hdr.is_valid() {
        hdr.init_defaults();
    }

    Some(Mapping { _fd: fd, header })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn scratch_path() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}/dlssnr-mapping-test-{}-{n}/shm.bin", std::env::temp_dir().display(), std::process::id())
    }

    #[test]
    fn open_at_creates_a_valid_mapping_with_real_defaults() {
        let path = scratch_path();
        let mapping = open_at(&path).expect("open_at should succeed");
        assert!(mapping.header().is_valid());
        assert!(mapping.header().neural_enabled());
        assert_eq!(f32::from_bits(mapping.header().intensity_bits.load(Ordering::Relaxed)), 1.0);
    }

    #[test]
    fn open_at_reattaches_to_an_existing_mapping_without_resetting_it() {
        let path = scratch_path();
        {
            let mapping = open_at(&path).unwrap();
            mapping.header().intensity_bits.store(0.42f32.to_bits(), Ordering::Relaxed);
        }
        let reattached = open_at(&path).expect("re-open should succeed");
        assert_eq!(f32::from_bits(reattached.header().intensity_bits.load(Ordering::Relaxed)), 0.42);
    }
}
