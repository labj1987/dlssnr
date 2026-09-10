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

/// The real Steam client install root (the directory containing `steamapps/`,
/// `compatibilitytools.d/`, etc.) -- what `STEAM_COMPAT_CLIENT_INSTALL_PATH` needs to
/// point at for Proton's own launch script to run at all (`start()`'s own doc comment
/// explains why this has to be set). Same candidate list `dlssnr-cli`'s own Proton
/// discovery (`runners.rs::candidate_dirs`) already scans for
/// `compatibilitytools.d` -- this just checks the *parent* of each and returns the
/// first that's a real directory, since a real Steam install is what actually creates
/// these paths, in this same order of likelihood (native package first, then the
/// sandboxed variants).
pub fn steam_install_dir() -> Option<String> {
    let xdg_data_home = xdg("XDG_DATA_HOME", ".local/share");
    let home = home();
    [
        format!("{xdg_data_home}/Steam"),
        format!("{home}/.var/app/com.valvesoftware.Steam/data/Steam"),
        format!("{home}/snap/steam/common/.local/share/Steam"),
    ]
    .into_iter()
    .find(|dir| std::path::Path::new(dir).is_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real filesystem, real env var override -- confirms `steam_install_dir` actually
    /// finds a directory that exists (not just "returns some string unconditionally"),
    /// and returns `None` when nothing does. Found and fixed as a real bug 2026-09-10:
    /// `start()` never set `STEAM_COMPAT_CLIENT_INSTALL_PATH` at all before this
    /// existed, causing a real `KeyError` crash in Proton's own script on every single
    /// start attempt with `runner_type = "proton"`.
    #[test]
    fn finds_a_real_steam_install_under_xdg_data_home() {
        let scratch = std::env::temp_dir().join(format!("dlssnr-steam-detect-test-{}", std::process::id()));
        let steam_dir = scratch.join("Steam");
        std::fs::create_dir_all(&steam_dir).unwrap();

        let prev = std::env::var("XDG_DATA_HOME").ok();
        std::env::set_var("XDG_DATA_HOME", &scratch);

        let found = steam_install_dir();
        assert_eq!(found.as_deref(), steam_dir.to_str());

        match prev {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
        std::fs::remove_dir_all(&scratch).ok();
    }

    #[test]
    fn returns_none_when_no_candidate_exists() {
        let scratch = std::env::temp_dir().join(format!("dlssnr-steam-detect-test-none-{}", std::process::id()));
        // Deliberately do not create `scratch` itself -- every candidate under it is
        // real-but-nonexistent, the case this function must fail open on.

        let prev = std::env::var("XDG_DATA_HOME").ok();
        std::env::set_var("XDG_DATA_HOME", &scratch);

        assert_eq!(steam_install_dir(), None);

        match prev {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
    }
}
