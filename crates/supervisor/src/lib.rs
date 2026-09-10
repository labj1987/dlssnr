//! Shared helper-process supervision: config, XDG paths, and start/stop, used by
//! both `dlssnr-cli` and `dlssnr-gui` so "how to launch the helper" (runner type,
//! environment variables, the Proton-vs-Wine command line) exists in exactly one
//! place instead of being duplicated and risking drift between the two front ends.
//! Linux-only: spawns child processes, reads XDG env vars.

pub mod config;
pub mod install_dir;
pub mod paths;
mod process;

pub use config::Config;

use std::time::Duration;

pub fn pid_file() -> String {
    format!("{}/helper.pid", dlssnr_protocol::shm_runtime_dir())
}

/// The helper's PID if a process is actually alive at the PID in the pid file.
pub fn is_running() -> Option<i32> {
    process::running_pid(&pid_file())
}

/// Graceful-then-forced stop of the whole helper process group.
pub fn stop(timeout: Duration) -> std::io::Result<()> {
    process::stop(&pid_file(), timeout)
}

#[derive(Debug)]
pub enum StartError {
    AlreadyRunning(i32),
    HelperNotFound,
    NoRunnerConfigured,
    Spawn(std::io::Error),
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartError::AlreadyRunning(pid) => write!(f, "helper already running (pid {pid})"),
            StartError::HelperNotFound => write!(f, "dlssnr_helper.exe not found"),
            StartError::NoRunnerConfigured => write!(f, "no runner configured (run `dlssnr-cli init` first)"),
            StartError::Spawn(e) => write!(f, "failed to start helper: {e}"),
        }
    }
}

pub struct StartedHelper {
    pub pid: i32,
    pub runner_type: String,
    pub runner_path: String,
    pub log: String,
}

/// Starts the helper under `cfg`'s configured runner (Proton or plain Wine),
/// building the same environment variables (`WINEPREFIX`, `DLSSNR_SHM`, `DLSSNR_UID`,
/// the Proton-specific NVAPI/compat-data ones) either front end needs -- this is the
/// one place that construction happens.
pub fn start(cfg: &Config) -> Result<StartedHelper, StartError> {
    if let Some(pid) = is_running() {
        return Err(StartError::AlreadyRunning(pid));
    }
    let Some(helper) = install_dir::helper_exe() else {
        return Err(StartError::HelperNotFound);
    };
    if cfg.runner_path.is_empty() {
        return Err(StartError::NoRunnerConfigured);
    }

    let mut envs = vec![
        ("WINEPREFIX".to_string(), paths::prefix_dir()),
        ("DLSSNR_SHM".to_string(), cfg.shm.clone()),
        ("DLSSNR_LOG".to_string(), cfg.log.clone()),
        ("DLSSNR_BIN_DIR".to_string(), format!("Z:{}", cfg.binaries)),
        ("WINEDEBUG".to_string(), "-all".to_string()),
    ];
    // SAFETY-relevant only in the "matches a real deployment" sense, not memory
    // safety: DLSSNR_UID has to be the same value the layer computes
    // `shm_runtime_dir()` from, which reads it from the environment too -- passing it
    // explicitly here is what keeps both sides pointed at the same file.
    // SAFETY: getuid() takes no arguments and cannot fail.
    let uid = unsafe { libc::getuid() };
    envs.push(("DLSSNR_UID".to_string(), uid.to_string()));

    let (program, args): (String, Vec<String>) = if cfg.runner_type == "proton" {
        envs.push(("PROTON_ENABLE_NVAPI".to_string(), "1".to_string()));
        envs.push(("DLSSNR_SKIP_NVAPI".to_string(), "1".to_string()));
        envs.push(("STEAM_COMPAT_DATA_PATH".to_string(), paths::prefix_dir()));
        // Proton's own launch script reads this directly out of the environment
        // (`os.environ["STEAM_COMPAT_CLIENT_INSTALL_PATH"]`, no fallback) during
        // prefix setup, before it ever gets to running the helper .exe -- omitting it
        // is a real, confirmed `KeyError` crash on *every* start attempt, found
        // 2026-09-10 running this against a real game session on `lordnikon`: the
        // helper never got further than Proton's own setup_prefix() step. Every
        // manual SSH test this project's own history has ever done set this by hand
        // without that fix ever making it back into this function -- this is that fix.
        if let Some(steam_dir) = paths::steam_install_dir() {
            envs.push(("STEAM_COMPAT_CLIENT_INSTALL_PATH".to_string(), steam_dir));
        }
        (cfg.runner_path.clone(), vec!["run".to_string(), helper.display().to_string()])
    } else {
        (cfg.runner_path.clone(), vec![helper.display().to_string()])
    };

    process::start_detached(&program, &args, &envs, &cfg.log, &pid_file())
        .map(|pid| StartedHelper {
            pid,
            runner_type: cfg.runner_type.clone(),
            runner_path: cfg.runner_path.clone(),
            log: cfg.log.clone(),
        })
        .map_err(StartError::Spawn)
}
