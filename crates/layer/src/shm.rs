//! The shared-memory round trip with the helper.
//!
//! This is the seam: everything here only ever touches `dlssnr_protocol::ShmHeader`'s
//! atomics, never anything Windows/NGX-specific. What answers on the other end of the
//! mapping — a Wine-wrapped helper today, a native one later — is none of this
//! module's business.
//!
//! Milestone 2 scope: the request/response sequence-number handshake and the fail-open
//! timing budget, ported for shape from upstream's `ShmOpen`/`ShmNeuralEnabled`/
//! `ShmProcessFrame`.
//!
//! Milestone 4 adds [`ShmClient::write_proxy`]/[`ShmClient::read_answer`]: the mapping
//! now covers the full `dlssnr_protocol::shm_total_bytes()` region (header plus both
//! `MAX_FRAME`-sized pixel buffers), not just the header, so the proxy/answer bytes
//! live in the same `mmap` this type already owns rather than a second one.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use dlssnr_protocol::{enums::helper_state, shm_default_path, MAX_FRAME, SHM_MAGIC};

/// The subset of `ShmHeader`'s composition fields `composition::apply::apply_rgba8`
/// needs, decoded once per frame from the raw atomics. See that field's own doc
/// comment in `dlssnr_protocol::header::ShmHeader` for what each one means.
pub struct CompositionSettings {
    pub colour_strength: f32,
    pub transfer_strength: f32,
    pub max_ratio: f32,
    pub debug_view: u32,
    pub apply_model: bool,
    pub neural_enabled: bool,
}

/// One process's connection to the mapping. Not `Clone` — there is exactly one of these
/// per device, guarded by a `Mutex` in [`crate::device::DlssnrDeviceInfo`].
pub struct ShmClient {
    fd: Option<OwnedFd>,
    header: *mut dlssnr_protocol::ShmHeader,
    path: String,
    timeouts: u32,
    ever_answered: bool,
    retry_after: Option<Instant>,
    last_control_seq: u32,
    last_heartbeat: u32,
    dead: bool,
    frames: u64,
    /// The request number and send time of a round trip issued via
    /// [`Self::begin_async_request`] that [`Self::poll_async_request`] hasn't yet
    /// resolved (answered or timed out). `None` means no request is in flight --
    /// callers use this to decide whether it's time to capture and send a new frame,
    /// per the same one-outstanding-request-at-a-time limit the wire protocol has
    /// always had (a single `seq_req`/`seq_resp` pair, not a queue).
    pending: Option<(u32, Instant)>,
}

// SAFETY: `header` points at a `MAP_SHARED` mapping that stays valid for the process's
// lifetime once opened (never unmapped or reallocated by this type), and every access
// through it goes through `ShmHeader`'s own atomics/seqlock-guarded accessors -- the
// same invariant that makes `ShmHeader` itself `Sync` (see `dlssnr_protocol::header`).
// `ShmClient` is always accessed from behind a `Mutex`, so only `Send` is needed, never
// concurrent access from two threads at once.
unsafe impl Send for ShmClient {}

impl Default for ShmClient {
    fn default() -> Self {
        Self {
            fd: None,
            header: std::ptr::null_mut(),
            path: String::new(),
            timeouts: 0,
            ever_answered: false,
            retry_after: None,
            last_control_seq: 0,
            last_heartbeat: 0,
            dead: false,
            frames: 0,
            pending: None,
        }
    }
}

impl ShmClient {
    /// Cross-module test access to the raw header pointer -- `capture::tests` needs
    /// to poke `helper_state`/`seq_resp` directly to stand in for a fake helper, the
    /// same way this module's own tests do, but `header` is private to this module
    /// and those tests live in a sibling one. Test-only; never called from real code.
    #[cfg(test)]
    pub(crate) fn test_header_ptr(&self) -> usize {
        self.header as usize
    }

    /// Cross-module test access to [`Self::open_at`], same reasoning as
    /// [`Self::test_header_ptr`]: `capture::tests` needs a scratch-path open, exactly
    /// like this module's own tests already do, but the real method is private.
    #[cfg(test)]
    pub(crate) fn test_open_at(&mut self, path: &str) -> bool {
        self.open_at(path)
    }

    fn header(&self) -> Option<&dlssnr_protocol::ShmHeader> {
        // SAFETY: non-null only after a successful `open()`, which mmaps
        // `dlssnr_protocol::shm_total_bytes()` at this address and never unmaps it for
        // the lifetime of the process.
        (!self.header.is_null()).then(|| unsafe { &*self.header })
    }

    /// Whether it's worth paying for a real capture this frame at all. `false` once
    /// the helper has reported the model permanently unavailable (see
    /// `dlssnr_helper::ngx::ensure_feature`'s own one-shot-then-disable design) --
    /// capturing and writing back a frame nobody will ever evaluate is pure overhead
    /// (a full image<->buffer round trip plus a `memcpy` of the whole frame, every
    /// single present call) for zero chance of a different outcome. Reads a single
    /// already-mapped atomic; never blocks and never opens the mapping itself, so it's
    /// always safe to check before deciding whether to call
    /// [`crate::capture::run`] at all.
    pub fn model_known_unavailable(&self) -> bool {
        let Some(hdr) = self.header() else { return false };
        hdr.helper_state.load(Ordering::Relaxed) == helper_state::MODEL_FAILED
    }

    /// Consumes a pending "dump one matched before/after frame pair" request (see
    /// `dlssnr_protocol::header::ShmHeader::capture_request`'s own doc comment) --
    /// `true` at most once per request, since this resets it to 0 in the same atomic
    /// operation, so the very next present doesn't dump again for a request that was
    /// already served.
    pub fn take_capture_request(&self) -> bool {
        let Some(hdr) = self.header() else { return false };
        hdr.capture_request.swap(0, Ordering::Relaxed) != 0
    }

    /// Non-consuming version of [`Self::take_capture_request`] -- lets a caller decide
    /// *how* to produce this frame's composited bytes (a fast, GPU-only path with no
    /// CPU-visible result, vs. a path that leaves the result somewhere
    /// [`Self::take_capture_request`]'s caller can dump) before committing to either,
    /// without losing/duplicating the actual one-shot request in the process.
    pub fn capture_request_pending(&self) -> bool {
        let Some(hdr) = self.header() else { return false };
        hdr.capture_request.load(Ordering::Relaxed) != 0
    }

    /// The settings `composition::apply::apply_rgba8` needs, read fresh every frame
    /// (each is a single atomic load) so a live GUI change takes effect on the very
    /// next present rather than needing a restart. `None` before the mapping is open.
    pub fn composition_settings(&self) -> Option<CompositionSettings> {
        let hdr = self.header()?;
        Some(CompositionSettings {
            colour_strength: f32::from_bits(hdr.colour_strength_bits.load(Ordering::Relaxed)),
            transfer_strength: f32::from_bits(hdr.transfer_strength_bits.load(Ordering::Relaxed)),
            max_ratio: f32::from_bits(hdr.max_ratio_bits.load(Ordering::Relaxed)),
            debug_view: hdr.debug_view.load(Ordering::Relaxed),
            apply_model: hdr.apply_model.load(Ordering::Relaxed) != 0,
            neural_enabled: hdr.neural_enabled(),
        })
    }

    /// Records what the proxy bytes about to be written actually are -- the helper
    /// (and, on the way back, this same layer reading the answer) needs `width`/
    /// `height`/`proxy_format` to know how many of the region's bytes are real for
    /// this frame, not the full `MAX_FRAME`-sized reservation. Call before
    /// [`Self::write_proxy`]/[`Self::try_round_trip`] so the helper never observes the
    /// `seq_req` bump before it can see what raster it describes.
    pub fn set_frame_info(&mut self, width: u32, height: u32, proxy_format: u32) {
        self.frames += 1;
        let frames = self.frames;
        let Some(hdr) = self.header() else { return };
        hdr.width.store(width, Ordering::Relaxed);
        hdr.height.store(height, Ordering::Relaxed);
        hdr.proxy_format.store(proxy_format, Ordering::Relaxed);

        // The layer's own "I am alive and capturing" telemetry -- mirrors what
        // `dlssnr_helper::main`'s loop already does for `hdr.helper_*`/`heartbeat`.
        // Nothing else in this crate ever wrote these fields before this (confirmed by
        // grep, 2026-09-10): `layer_attached` was declared, reset to 0 by
        // `ShmHeader::init_defaults`, and read by the GUI (`ui.rs`'s "not attached"
        // label) -- but never once set to 1 anywhere, so that label was always wrong,
        // regardless of whether the layer was actually attached. Confirmed on
        // `lordnikon` the same day: `/proc/<pid>/maps` and a live, advancing helper
        // frame counter both proved the real Vulkan layer was loaded and working the
        // whole time the GUI displayed "not attached". Setting this every frame (not
        // just once at `open()`) also survives a helper restart resetting the shared
        // header out from under an already-open, never-reconnecting layer -- exactly
        // what happened here: `open_at`'s own idempotent early return means a layer
        // that was already attached before the reset never calls it again to re-set a
        // one-shot flag.
        hdr.layer_attached.store(1, Ordering::Relaxed);
        hdr.layer_heartbeat.fetch_add(1, Ordering::Relaxed);
        hdr.layer_width.store(width, Ordering::Relaxed);
        hdr.layer_height.store(height, Ordering::Relaxed);
        hdr.layer_format.store(proxy_format, Ordering::Relaxed);
        dlssnr_protocol::store64(&hdr.layer_frames_lo, &hdr.layer_frames_hi, frames);
    }

    /// Writes `bytes` (truncated to `MAX_FRAME`, same discipline as the free-text
    /// fields in `ShmHeader`) into the proxy region -- the frame the layer is about to
    /// hand the model. Call before bumping `seq_req` (via [`Self::try_round_trip`]):
    /// the helper only starts reading once it observes that bump, so there is no
    /// concurrent-write hazard to guard against the way the header's atomics do.
    ///
    /// # Safety
    /// Must only be called after a successful [`Self::open`]/[`Self::try_round_trip`].
    pub fn write_proxy(&self, bytes: &[u8]) {
        let Some(base) = self.pixel_base() else { return };
        let n = bytes.len().min(MAX_FRAME);
        // SAFETY: `base` is the start of this process's own mapping of the full
        // `shm_total_bytes()` region (see `open_at`); `proxy_offset()..+n` is in bounds
        // for any `n <= MAX_FRAME` by that region's own definition.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                base.add(dlssnr_protocol::proxy_offset()),
                n,
            );
        }
    }

    /// Reads up to `out.len()` (capped at `MAX_FRAME`) bytes back from the answer
    /// region into `out`, returning the number of bytes copied. Meaningful only after
    /// [`Self::try_round_trip`] has returned `true` for the request this answer goes
    /// with -- reading it any earlier just observes whatever the helper last wrote
    /// (stale or all-zero), which is why this never blocks or checks sequence numbers
    /// itself; the caller already knows from the round trip's own return value whether
    /// there is a real answer to read.
    pub fn read_answer(&self, out: &mut [u8]) -> usize {
        let Some(base) = self.pixel_base() else { return 0 };
        let n = out.len().min(MAX_FRAME);
        // SAFETY: same reasoning as `write_proxy`, mirrored for the answer region.
        unsafe {
            std::ptr::copy_nonoverlapping(
                base.add(dlssnr_protocol::answer_offset()),
                out.as_mut_ptr(),
                n,
            );
        }
        n
    }

    fn pixel_base(&self) -> Option<*mut u8> {
        (!self.header.is_null()).then_some(self.header as *mut u8)
    }

    /// Opens (or creates) the mapping if not already attached. Idempotent.
    pub fn open(&mut self) -> bool {
        let path = std::env::var("DLSSNR_SHM")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(shm_default_path);
        self.open_at(&path)
    }

    /// The actual implementation, taking the path explicitly so tests can point it at a
    /// scratch directory instead of `$DLSSNR_SHM`/the real `/tmp/dlssnr-$UID/` -- mutating
    /// process-wide environment variables from parallel `#[test]`s would race.
    fn open_at(&mut self, path: &str) -> bool {
        if self.header().is_some() {
            return true;
        }
        if !ensure_private_parent_dir(path) {
            crate::log!("[shm] refusing {path}: parent directory is not private");
            return false;
        }
        let Ok(c_path) = CString::new(path) else {
            return false;
        };
        // SAFETY: `c_path` is a valid, NUL-terminated C string for the duration of the
        // call. `O_NOFOLLOW` refuses to open through a symlink -- this file lives under
        // a world-writable /tmp, so that refusal matters.
        let raw_fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW,
                0o600,
            )
        };
        if raw_fd < 0 {
            crate::log!("[shm] open {path} failed");
            return false;
        }
        // SAFETY: `raw_fd` was just returned by a successful `open()` above and is not
        // owned anywhere else yet.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        let total = dlssnr_protocol::shm_total_bytes();
        // SAFETY: `stat` is a plain out-parameter; zero-initializing it is always valid
        // and `fstat` either fully populates it or returns an error we check.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let needs_truncate = unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0
            || (st.st_size as usize) < total;
        if needs_truncate
            && unsafe { libc::ftruncate(fd.as_raw_fd(), total as libc::off_t) } != 0
        {
            crate::log!("[shm] ftruncate {path} failed");
            return false;
        }

        // SAFETY: `fd` is a valid, open file descriptor sized to at least `total` bytes
        // by the ftruncate above (or already that size); mapping the whole `total`
        // bytes (header plus both pixel regions) is always in-bounds. The mapping is
        // kept for the rest of the process's life, so the returned pointer stays valid
        // for as long as anything derived from it (`header()`'s `&ShmHeader`, or the
        // proxy/answer slices below) is used. A `MAP_SHARED` file mapping is a sparse,
        // page-cache-backed region -- reserving the full `MAX_FRAME*2` up front costs
        // no real memory beyond whatever pages an SDR session actually touches.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if map == libc::MAP_FAILED {
            crate::log!("[shm] mmap {path} failed");
            return false;
        }

        let header = map as *mut dlssnr_protocol::ShmHeader;
        // SAFETY: just mapped above, `HEADER_BYTES` is large enough for `ShmHeader`
        // (enforced at compile time in `dlssnr_protocol`).
        let hdr = unsafe { &*header };
        if hdr.magic.load(Ordering::Relaxed) != SHM_MAGIC || !hdr.is_valid() {
            // A magic mismatch is some other mapping entirely (or garbage); a version
            // mismatch is a stale build of ours. Both get the same answer: reinitialize
            // rather than half-read a layout we don't agree on.
            hdr.init_defaults();
        }
        self.last_heartbeat = hdr.heartbeat.load(Ordering::Relaxed);
        self.last_control_seq = hdr.control_seq.load(Ordering::Relaxed);
        self.fd = Some(fd);
        self.header = header;
        self.path = path.to_string();
        crate::log!(
            "[shm] attached {} seq_req={} seq_resp={}",
            self.path,
            hdr.seq_req.load(Ordering::Relaxed),
            hdr.seq_resp.load(Ordering::Relaxed)
        );
        true
    }

    /// One request/response cycle: bump `seq_req`, wait up to a budget for `seq_resp` to
    /// catch up. Returns whether the helper answered in time.
    ///
    /// This is a fail-open state machine, same as upstream: a helper that never answers
    /// four times in a row is marked dead and not retried for 5 seconds, so a missing
    /// helper costs one short wait per frame rather than the full budget forever. A
    /// live-but-busy helper (building its first feature) gets a much longer budget on
    /// its very first frame, since that is expected to be slow.
    pub fn try_round_trip(&mut self) -> bool {
        if !self.open() {
            self.dead = true;
            return false;
        }
        self.round_trip_after_open()
    }

    /// Same as [`Self::try_round_trip`], but against an explicit path rather than
    /// `$DLSSNR_SHM`/the real runtime dir -- so tests can point it at a scratch
    /// directory without racing on process-wide environment variables.
    #[cfg(test)]
    fn try_round_trip_at(&mut self, path: &str) -> bool {
        if !self.open_at(path) {
            self.dead = true;
            return false;
        }
        self.round_trip_after_open()
    }

    fn round_trip_after_open(&mut self) -> bool {
        // Re-check liveness before every attempt: a dead connection can come back
        // either because its retry timer elapsed, or because the helper's control_seq
        // or heartbeat moved, meaning something changed on the other end worth trying
        // again for.
        if self.dead && !self.should_retry() {
            return false;
        }
        self.dead = false;

        let hdr = self.header().expect("just opened above");
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            self.dead = true;
            return false;
        }

        let req = hdr.seq_req.load(Ordering::Relaxed) + 1;
        std::sync::atomic::fence(Ordering::Release);
        hdr.seq_req.store(req, Ordering::Relaxed);

        let helper_present = hdr.helper_state.load(Ordering::Relaxed) != helper_state::STOPPED;
        let warming_up = !self.ever_answered;
        let budget = if !helper_present {
            Duration::from_millis(20)
        } else if warming_up {
            Duration::from_secs(10)
        } else {
            Duration::from_secs(1)
        };

        let start = Instant::now();
        loop {
            if hdr.seq_resp.load(Ordering::Relaxed) >= req {
                std::sync::atomic::fence(Ordering::Acquire);
                self.timeouts = 0;
                self.ever_answered = true;
                return true;
            }
            if hdr.quit.load(Ordering::Relaxed) != 0 {
                self.dead = true;
                return false;
            }
            if start.elapsed() >= budget {
                break;
            }
            std::thread::sleep(Duration::from_micros(200));
        }

        self.timeouts += 1;
        if self.timeouts >= 4 {
            self.dead = true;
            self.retry_after = Some(Instant::now() + Duration::from_secs(5));
            crate::log!(
                "[shm] no answer in {:?} x4 (helper {}); passing frames through, retrying in 5s",
                budget,
                if helper_present { "is present but silent" } else { "not running" }
            );
        }
        false
    }

    /// Whether a round trip started by [`Self::begin_async_request`] is still
    /// in flight (sent, not yet resolved by [`Self::poll_async_request`]). Callers use
    /// this to decide whether it's worth capturing and sending a new frame this present
    /// call -- the wire protocol has only ever supported one outstanding request at a
    /// time (a single `seq_req`/`seq_resp` pair, not a queue), so starting a second one
    /// before the first resolves would just overwrite it.
    pub fn has_pending_request(&self) -> bool {
        self.pending.is_some()
    }

    /// Starts a round trip without waiting for it: bumps `seq_req` and records when,
    /// exactly like the first half of [`Self::round_trip_after_open`], but returns
    /// immediately instead of blocking. Pair with [`Self::poll_async_request`], called
    /// once per frame thereafter, to find out when (or whether) it resolves.
    ///
    /// Returns `false` (and starts nothing) on a dead connection whose retry timer
    /// hasn't elapsed, on `quit`, or if the mapping can't be opened -- the same
    /// conditions [`Self::try_round_trip`] fails open on. Only ever call this when
    /// [`Self::has_pending_request`] is `false`; calling it with a request already in
    /// flight would silently abandon that one (its `seq_req` gets overwritten before
    /// `poll_async_request` ever sees a matching `seq_resp`).
    pub fn begin_async_request(&mut self) -> bool {
        if !self.open() {
            self.dead = true;
            return false;
        }
        if self.dead && !self.should_retry() {
            return false;
        }
        self.dead = false;

        let hdr = self.header().expect("just opened above");
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            self.dead = true;
            return false;
        }

        let req = hdr.seq_req.load(Ordering::Relaxed) + 1;
        std::sync::atomic::fence(Ordering::Release);
        hdr.seq_req.store(req, Ordering::Relaxed);
        self.pending = Some((req, Instant::now()));
        true
    }

    /// Non-blocking: checks whether the request [`Self::begin_async_request`] started
    /// has answered yet. `Some(true)` once, the instant `seq_resp` catches up (clears
    /// the pending state, so [`Self::has_pending_request`] is `false` again
    /// afterward -- the caller is free to start a new one). `Some(false)` while still
    /// genuinely waiting, within budget. `None` once the budget is exceeded --
    /// also clears the pending state (same timeout/dead-connection bookkeeping
    /// [`Self::round_trip_after_open`] already does), so the caller knows to give up
    /// on this cycle and start fresh rather than keep polling a request that will
    /// never resolve. Returns `Some(false)` (never blocks, never panics) if called
    /// with nothing pending.
    pub fn poll_async_request(&mut self) -> Option<bool> {
        let Some((req, sent_at)) = self.pending else { return Some(false) };
        let Some(hdr) = self.header() else {
            self.pending = None;
            return None;
        };
        if hdr.seq_resp.load(Ordering::Relaxed) >= req {
            std::sync::atomic::fence(Ordering::Acquire);
            self.pending = None;
            self.timeouts = 0;
            self.ever_answered = true;
            return Some(true);
        }
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            self.pending = None;
            self.dead = true;
            return None;
        }
        let helper_present = hdr.helper_state.load(Ordering::Relaxed) != helper_state::STOPPED;
        let warming_up = !self.ever_answered;
        let budget = if !helper_present {
            Duration::from_millis(20)
        } else if warming_up {
            Duration::from_secs(10)
        } else {
            Duration::from_secs(1)
        };
        if sent_at.elapsed() < budget {
            return Some(false);
        }
        self.pending = None;
        self.timeouts += 1;
        if self.timeouts >= 4 {
            self.dead = true;
            self.retry_after = Some(Instant::now() + Duration::from_secs(5));
            crate::log!(
                "[shm] no answer in {:?} x4 (helper {}); passing frames through, retrying in 5s",
                budget,
                if helper_present { "is present but silent" } else { "not running" }
            );
        }
        None
    }

    fn should_retry(&mut self) -> bool {
        let Some(hdr) = self.header() else { return false };
        let control_seq = hdr.control_seq.load(Ordering::Relaxed);
        let heartbeat = hdr.heartbeat.load(Ordering::Relaxed);
        let changed = control_seq != self.last_control_seq || heartbeat != self.last_heartbeat;
        self.last_control_seq = control_seq;
        self.last_heartbeat = heartbeat;
        if !changed {
            return false;
        }
        // A heartbeat alone is not a reason to try again immediately -- the helper
        // ticks it while it sits idle, so a helper that is up but not answering would
        // otherwise re-enable the moment it had just given up, costing another full
        // wait every time. The retry timer is what actually paces retries; a change
        // just means it's worth checking whether that timer has elapsed yet.
        self.retry_after.is_none_or(|t| Instant::now() >= t)
    }
}

/// Refuses to create or use the mapping's directory unless it is private: a directory,
/// owned by this uid, with no group/other permission bits. It lives under the
/// world-writable `/tmp`, so this is the difference between "our socket" and "whatever
/// another local user left in our way."
fn ensure_private_parent_dir(path: &str) -> bool {
    let Some(dir) = path.rfind('/').map(|i| &path[..i]) else {
        return true;
    };
    if dir.is_empty() {
        return true;
    }

    // mkdir -p, ignoring EEXIST at each level -- the same tolerant, idempotent
    // create-if-missing upstream's shell version does.
    let mut built = String::new();
    for part in dir.split('/') {
        if part.is_empty() {
            continue;
        }
        built.push('/');
        built.push_str(part);
        if let Ok(c) = CString::new(built.as_str()) {
            // SAFETY: `c` is a valid NUL-terminated C string for the call's duration.
            // The return value is intentionally ignored: EEXIST (already there) and any
            // other failure are both handled uniformly by the `lstat` check below.
            unsafe {
                libc::mkdir(c.as_ptr(), 0o700);
            }
        }
    }

    let Ok(c_dir) = CString::new(dir) else { return false };
    // SAFETY: `c_dir` is valid for the call's duration; `st` is a plain out-parameter.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::lstat(c_dir.as_ptr(), &mut st) } != 0 {
        return false;
    }
    let is_dir = (st.st_mode & libc::S_IFMT) == libc::S_IFDIR;
    // SAFETY: getuid() takes no arguments and cannot fail.
    let owned_by_us = st.st_uid == unsafe { libc::getuid() };
    let no_group_other_perms = (st.st_mode & (libc::S_IRWXG | libc::S_IRWXO)) == 0;
    is_dir && owned_by_us && no_group_other_perms
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlssnr_protocol::ShmHeader;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::AtomicU64;

    /// A fresh, private scratch path per test -- never `$TMPDIR/dlssnr-*` or anything
    /// `open()`'s real env-var path would touch, so these can run in parallel with each
    /// other (and with a real layer, if one happened to be running) without colliding.
    fn scratch_path() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        format!("{}/dlssnr-shm-test-{pid}-{n}/shm.bin", std::env::temp_dir().display())
    }

    fn header_of<'a>(client: &'a ShmClient) -> &'a ShmHeader {
        client.header().expect("open_at should have attached")
    }

    #[test]
    fn open_creates_a_valid_mapping() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        assert!(std::path::Path::new(&path).exists());
        assert!(header_of(&client).is_valid());
        // Idempotent: a second open on the same client is a no-op, not a re-create.
        assert!(client.open_at(&path));
    }

    #[test]
    fn round_trip_with_no_helper_times_out_then_marks_dead() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        // helper_state defaults to STOPPED, so each attempt's budget is the short
        // "nobody's listening" one -- four of them should complete quickly.
        for _ in 0..3 {
            assert!(!client.try_round_trip_at(&path));
            assert!(!client.dead, "should not give up before the fourth timeout");
        }
        assert!(!client.try_round_trip_at(&path));
        assert!(client.dead, "four consecutive timeouts should mark the connection dead");
        assert!(client.retry_after.is_some());

        // Dead, and the retry timer hasn't elapsed yet, and nothing on the other end
        // has changed control_seq/heartbeat -- so this call must not even attempt a
        // round trip.
        let resp_before = header_of(&client).seq_resp.load(Ordering::Relaxed);
        assert!(!client.try_round_trip_at(&path));
        assert_eq!(header_of(&client).seq_resp.load(Ordering::Relaxed), resp_before);
    }

    #[test]
    fn round_trip_succeeds_when_something_answers() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        let hdr_ptr = client.header as usize;

        // A minimal stand-in for the helper: echo every seq_req into seq_resp as soon
        // as it changes. Exactly the contract `try_round_trip` waits on -- nothing
        // upstream-specific about it.
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = std::sync::Arc::clone(&stop);
        let echo = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends,
            // and never unmapped by `ShmClient` regardless).
            let hdr = unsafe { &*(hdr_ptr as *mut ShmHeader) };
            while !stop_clone.load(Ordering::Relaxed) {
                let req = hdr.seq_req.load(Ordering::Relaxed);
                if hdr.seq_resp.load(Ordering::Relaxed) != req {
                    hdr.seq_resp.store(req, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        });

        assert!(client.try_round_trip_at(&path));
        assert!(client.ever_answered);
        assert!(!client.dead);

        stop.store(true, Ordering::Relaxed);
        echo.join().unwrap();
    }

    #[test]
    fn async_request_resolves_without_blocking_when_something_answers() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        let hdr_ptr = client.header as usize;
        // A live helper, so `poll_async_request`'s budget is the long "steady state"
        // one rather than the short "nobody's listening" one -- doesn't matter here
        // since the echo thread answers almost immediately either way, but matches
        // what a real run looks like.
        header_of(&client).helper_state.store(dlssnr_protocol::enums::helper_state::RUNNING, Ordering::Relaxed);

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = std::sync::Arc::clone(&stop);
        let echo = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends).
            let hdr = unsafe { &*(hdr_ptr as *mut ShmHeader) };
            while !stop_clone.load(Ordering::Relaxed) {
                let req = hdr.seq_req.load(Ordering::Relaxed);
                if req != 0 && hdr.seq_resp.load(Ordering::Relaxed) != req {
                    hdr.seq_resp.store(req, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        });

        assert!(client.begin_async_request(), "should start a request against an already-open mapping");
        assert!(client.has_pending_request());
        // Real non-blocking behavior: a call arriving before the echo thread has had a
        // chance to run must not hang waiting -- it either sees `Some(false)` (not
        // answered yet) or, if the thread was fast enough, `Some(true)` -- either way
        // this call itself returns immediately.
        let mut resolved = client.poll_async_request() == Some(true);
        let deadline = Instant::now() + Duration::from_secs(2);
        while !resolved {
            assert!(Instant::now() < deadline, "poll_async_request never resolved true");
            resolved = client.poll_async_request() == Some(true);
        }
        assert!(!client.has_pending_request(), "a resolved request must clear pending state");
        assert!(client.ever_answered);
        assert!(!client.dead);

        stop.store(true, Ordering::Relaxed);
        echo.join().unwrap();
    }

    #[test]
    fn async_request_with_no_helper_times_out_without_blocking_and_clears_pending() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        // helper_state defaults to STOPPED -- short "nobody's listening" budget (20ms),
        // so this test still runs fast despite exercising a real timeout.
        assert!(client.begin_async_request());
        assert!(client.has_pending_request());

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut result = client.poll_async_request();
        while result == Some(false) {
            assert!(Instant::now() < deadline, "poll_async_request never gave up");
            result = client.poll_async_request();
        }
        assert_eq!(result, None, "an unanswered request past budget must resolve to None, not Some(true)");
        assert!(!client.has_pending_request(), "a timed-out request must clear pending state too");
    }

    #[test]
    fn private_parent_dir_is_created_when_missing() {
        let path = scratch_path();
        assert!(ensure_private_parent_dir(&path));
        let dir = &path[..path.rfind('/').unwrap()];
        let meta = std::fs::metadata(dir).unwrap();
        assert!(meta.is_dir());
        assert_eq!(meta.permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn world_writable_existing_dir_is_rejected() {
        let dir = format!("{}-{}", scratch_path().trim_end_matches("/shm.bin"), "world-writable");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(!ensure_private_parent_dir(&format!("{dir}/shm.bin")));
    }
}
