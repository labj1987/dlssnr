//! XDG-aware paths, matching upstream's own reasoning (see the plan's `layer`
//! section and upstream's README for why the runtime/SHM path specifically lives
//! under `/tmp`, not `$XDG_RUNTIME_DIR` — that one's `dlssnr_protocol::shm_runtime_dir`,
//! already shared code; everything else here is config/data/state, which has no
//! Steam-container wrinkle to work around.

fn home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/".to_string())
}

fn xdg(var: &str, fallback_under_home: &str) -> String {
    std::env::var(var).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| format!("{}/{fallback_under_home}", home()))
}

pub fn config_dir() -> String {
    format!("{}/dlssnr", xdg("XDG_CONFIG_HOME", ".config"))
}

pub fn config_file() -> String {
    format!("{}/config.ini", config_dir())
}

pub fn data_dir() -> String {
    format!("{}/dlssnr", xdg("XDG_DATA_HOME", ".local/share"))
}

pub fn state_dir() -> String {
    format!("{}/dlssnr", xdg("XDG_STATE_HOME", ".local/state"))
}

pub fn log_file() -> String {
    format!("{}/helper.log", state_dir())
}

pub fn binaries_dir() -> String {
    format!("{}/binaries", data_dir())
}

/// The managed Wine/Proton prefix `dlssnr` creates and owns, distinct from any
/// prefix a game or Steam manages -- so importing NGX DLLs into it, or a bad prefix
/// state, never touches anything else.
pub fn prefix_dir() -> String {
    format!("{}/dlssnr/prefix", xdg("XDG_DATA_HOME", ".local/share"))
}

pub fn ensure_dirs() -> std::io::Result<()> {
    for dir in [config_dir(), data_dir(), state_dir(), binaries_dir()] {
        std::fs::create_dir_all(dir)?;
    }
    Ok(())
}
