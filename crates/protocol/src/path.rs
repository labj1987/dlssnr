//! Where the mapping lives.
//!
//! It has to name the same file in every process that touches it, and a Steam game
//! does not share a mount namespace with the helper: pressure-vessel gives the
//! container a private tmpfs at `$XDG_RUNTIME_DIR`, so a mapping put there is simply
//! absent inside the game. `/tmp` is bind-mounted from the host into the container, so
//! both sides land on one file; it is also what a Wine prefix exposes as `Z:\tmp\...`,
//! which is how the helper opens it.

/// The directory the mapping (and the PID file, log, etc.) live under:
/// `/tmp/dlssnr-<uid>/`.
///
/// `DLSSNR_UID` overrides the detected uid — the helper is always handed it by the
/// launcher (it runs under Wine, which has no native concept of a Linux uid), so this
/// is the one thing that lets both sides agree on the path without either one having to
/// ask the other.
pub fn shm_runtime_dir() -> String {
    if let Ok(uid) = std::env::var("DLSSNR_UID") {
        if !uid.is_empty() {
            return format!("/tmp/dlssnr-{uid}");
        }
    }
    fallback_runtime_dir()
}

pub fn shm_default_path() -> String {
    format!("{}/shm.bin", shm_runtime_dir())
}

#[cfg(unix)]
fn fallback_runtime_dir() -> String {
    // SAFETY: getuid() takes no arguments and cannot fail.
    let uid = unsafe { libc::getuid() };
    format!("/tmp/dlssnr-{uid}")
}

#[cfg(not(unix))]
fn fallback_runtime_dir() -> String {
    // The helper is always handed DLSSNR_UID by the launcher (see above), since it runs
    // under Wine and has no native concept of a Linux uid — this is only ever a last resort.
    "/tmp/dlssnr".to_string()
}
