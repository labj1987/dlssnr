//! Where `dlssnr-cli import-binaries` puts NVIDIA's NGX DLLs, and the same copy the
//! CLI does -- duplicated rather than shared because it's four lines and the two
//! crates otherwise share nothing filesystem-related (the GUI only ever talks to the
//! helper over the SHM mapping, never touches paths, apart from this one exception).

const NGX_FILES: [&str; 3] = ["nvngx_dlssnr.dll", "nvngx.dll", "nvapi64.dll"];

pub fn dir() -> std::path::PathBuf {
    let data_home = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
            std::path::PathBuf::from(home).join(".local/share")
        });
    data_home.join("dlssnr").join("binaries")
}

/// Copies whichever of the known NGX DLLs are present in `src` into [`dir`]. Returns
/// how many were copied.
pub fn import_from(src: &std::path::Path) -> std::io::Result<usize> {
    let dest = dir();
    std::fs::create_dir_all(&dest)?;
    let mut copied = 0;
    for name in NGX_FILES {
        let from = src.join(name);
        if from.is_file() {
            std::fs::copy(&from, dest.join(name))?;
            copied += 1;
        }
    }
    Ok(copied)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_from_copies_known_files_and_skips_unknown_ones() {
        let src = std::env::temp_dir().join(format!("dlssnr-binaries-test-src-{}", std::process::id()));
        let dest_home = std::env::temp_dir().join(format!("dlssnr-binaries-test-home-{}", std::process::id()));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("nvngx_dlssnr.dll"), b"model").unwrap();
        std::fs::write(src.join("nvapi64.dll"), b"nvapi").unwrap();
        std::fs::write(src.join("unrelated.txt"), b"ignore me").unwrap();

        let prev_xdg = std::env::var("XDG_DATA_HOME").ok();
        std::env::set_var("XDG_DATA_HOME", &dest_home);

        let copied = import_from(&src).unwrap();
        assert_eq!(copied, 2);
        assert!(dir().join("nvngx_dlssnr.dll").is_file());
        assert!(dir().join("nvapi64.dll").is_file());
        assert!(!dir().join("unrelated.txt").exists());
        assert!(!dir().join("nvngx.dll").exists());

        match prev_xdg {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&dest_home).ok();
    }
}
