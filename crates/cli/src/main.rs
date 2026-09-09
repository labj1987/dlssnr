//! `dlssnr-cli` — the helper-manager: init/setup/start/stop/restart/status/doctor/
//! config/runners/detect-gpu/import-binaries. Replaces upstream's ~900-line bash
//! `dlssnr-helper` script with the same command surface (a CLI contract, not
//! upstream's expression of it) reimplemented in Rust, sharing logic with the GUI
//! through `dlssnr_protocol` instead of duplicating it in shell.

mod config;
mod gpu;
mod install_dir;
mod paths;
mod process;
mod runners;

use std::process::ExitCode;
use std::time::Duration;

use config::Config;

fn usage() {
    eprintln!(
        "usage: dlssnr-cli <command>\n\n\
         commands:\n\
         \x20 init                 create default config\n\
         \x20 setup                init config, dxvk config, and managed prefix if needed\n\
         \x20 start                start helper using configured runner\n\
         \x20 stop                 stop helper process tree\n\
         \x20 restart              stop then start\n\
         \x20 status               show helper status\n\
         \x20 doctor               check runner, prefix, dxvk, binaries, and paths\n\
         \x20 config               print effective config\n\
         \x20 runners              list discovered custom compatibility tool runners\n\
         \x20 detect-gpu           print detected NVIDIA PCI vendor/device\n\
         \x20 import-binaries DIR  copy NVIDIA NGX DLLs into user data dir"
    );
}

fn pid_file() -> String {
    format!("{}/helper.pid", dlssnr_protocol::shm_runtime_dir())
}

fn default_config() -> Config {
    let mut cfg = Config::default();
    if let Some(proton) = runners::best_proton() {
        cfg.runner_type = "proton".to_string();
        cfg.runner_path = proton.path.to_string_lossy().into_owned();
    } else if let Some(wine) = runners::find_wine() {
        cfg.runner_type = "wine".to_string();
        cfg.runner_path = wine.to_string_lossy().into_owned();
    } else {
        cfg.runner_type = "custom".to_string();
    }
    cfg.binaries = paths::binaries_dir();
    if let Some((vendor, device)) = gpu::detect_nvidia_gpu() {
        cfg.dxvk_vendor = format!("{vendor:04x}");
        cfg.dxvk_device = format!("{device:04x}");
    }
    cfg.shm = dlssnr_protocol::shm_default_path();
    cfg.log = paths::log_file();
    cfg
}

fn cmd_init() -> ExitCode {
    if std::path::Path::new(&paths::config_file()).exists() {
        println!("config already exists: {}", paths::config_file());
        return ExitCode::SUCCESS;
    }
    let cfg = default_config();
    match cfg.save() {
        Ok(()) => {
            println!("wrote {}", paths::config_file());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to write config: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_config() -> ExitCode {
    let cfg = Config::load();
    println!("config_file={}", paths::config_file());
    println!("data_dir={}", paths::data_dir());
    println!("state_dir={}", paths::state_dir());
    println!("prefix_dir={}", paths::prefix_dir());
    println!("runner_type={}", cfg.runner_type);
    println!("runner_path={}", cfg.runner_path);
    println!("binaries={}", cfg.binaries);
    println!("shm={}", cfg.shm);
    println!("log={}", cfg.log);
    println!("dxvk_vendor={}", cfg.dxvk_vendor);
    println!("dxvk_device={}", cfg.dxvk_device);
    println!("helper_exe={}", install_dir::helper_exe().map(|p| p.display().to_string()).unwrap_or_else(|| "missing".to_string()));
    ExitCode::SUCCESS
}

fn cmd_runners() -> ExitCode {
    let found = runners::discover_proton();
    if found.is_empty() {
        if let Some(wine) = runners::find_wine() {
            println!("wine\t{}", wine.display());
            return ExitCode::SUCCESS;
        }
        eprintln!("no custom compatibility tools or system wine found");
        return ExitCode::FAILURE;
    }
    for runner in found {
        println!("{}\t{}", runner.name, runner.path.display());
    }
    ExitCode::SUCCESS
}

fn cmd_detect_gpu() -> ExitCode {
    match gpu::detect_nvidia_gpu() {
        Some((vendor, device)) => {
            println!("{vendor:04x}:{device:04x}");
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("no NVIDIA GPU detected");
            ExitCode::FAILURE
        }
    }
}

fn cmd_status() -> ExitCode {
    match process::running_pid(&pid_file()) {
        Some(pid) => println!("helper running (pid {pid})"),
        None => println!("helper not running"),
    }
    println!("  config: {}", paths::config_file());
    println!("  runtime: {}", dlssnr_protocol::shm_runtime_dir());
    println!("  state: {}", paths::state_dir());
    ExitCode::SUCCESS
}

fn cmd_doctor() -> ExitCode {
    let mut ok = true;
    let cfg = Config::load();

    print!("config: {}\n  ", paths::config_file());
    if std::path::Path::new(&paths::config_file()).exists() {
        println!("ok");
    } else {
        println!("missing (run `dlssnr-cli init`)");
        ok = false;
    }

    let helper = install_dir::helper_exe();
    print!("helper exe: {}\n  ", helper.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "missing".to_string()));
    if helper.is_some() {
        println!("ok");
    } else {
        println!("missing");
        ok = false;
    }

    let (runner_type, runner_path) = if !cfg.runner_path.is_empty() {
        (cfg.runner_type.clone(), cfg.runner_path.clone())
    } else if let Some(proton) = runners::best_proton() {
        ("proton".to_string(), proton.path.display().to_string())
    } else if let Some(wine) = runners::find_wine() {
        ("wine".to_string(), wine.display().to_string())
    } else {
        ("none".to_string(), String::new())
    };
    print!("runner: {runner_type} {runner_path}\n  ");
    if !runner_path.is_empty() && std::path::Path::new(&runner_path).exists() {
        println!("ok");
    } else {
        println!("missing/not executable");
        ok = false;
    }

    let binaries = if cfg.binaries.is_empty() { paths::binaries_dir() } else { cfg.binaries.clone() };
    let ngx_dll = std::path::Path::new(&binaries).join("nvngx_dlssnr.dll");
    print!("binaries: {binaries}\n  nvngx_dlssnr.dll: ");
    if ngx_dll.exists() {
        println!("ok");
    } else {
        println!("error -- missing (required; see `dlssnr-cli import-binaries DIR`)");
        ok = false;
    }

    print!("vendored dxvk dll: {}\n  ", install_dir::dxvk_dll().as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "missing".to_string()));
    if install_dir::dxvk_dll().is_some() {
        println!("ok");
    } else {
        println!("missing (only needed for the system-Wine fallback runner)");
    }

    print!("runtime dir: {}\n  ", dlssnr_protocol::shm_runtime_dir());
    match paths::ensure_dirs() {
        Ok(()) => println!("ok"),
        Err(e) => {
            println!("not writable: {e}");
            ok = false;
        }
    }

    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn cmd_setup() -> ExitCode {
    let init_result = cmd_init();
    if init_result != ExitCode::SUCCESS {
        return init_result;
    }
    println!("setup complete");
    ExitCode::SUCCESS
}

fn cmd_start() -> ExitCode {
    if process::running_pid(&pid_file()).is_some() {
        println!("helper already running");
        return ExitCode::SUCCESS;
    }
    let cfg = Config::load();
    let Some(helper) = install_dir::helper_exe() else {
        eprintln!("error: dlssnr_helper.exe not found");
        return ExitCode::FAILURE;
    };
    if cfg.runner_path.is_empty() {
        eprintln!("error: no runner configured (run `dlssnr-cli init` first)");
        return ExitCode::FAILURE;
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
        (cfg.runner_path.clone(), vec!["run".to_string(), helper.display().to_string()])
    } else {
        (cfg.runner_path.clone(), vec![helper.display().to_string()])
    };

    match process::start_detached(&program, &args, &envs, &cfg.log, &pid_file()) {
        Ok(pid) => {
            println!("helper started (pid {pid})");
            println!("  runner: {} {}", cfg.runner_type, cfg.runner_path);
            println!("  log: {}", cfg.log);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to start helper: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_stop() -> ExitCode {
    match process::stop(&pid_file(), Duration::from_secs(5)) {
        Ok(()) => {
            println!("helper stopped");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to stop helper: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_import_binaries(dir: Option<&String>) -> ExitCode {
    let Some(dir) = dir else {
        eprintln!("usage: dlssnr-cli import-binaries DIR");
        return ExitCode::FAILURE;
    };
    let src = std::path::Path::new(dir);
    if !src.is_dir() {
        eprintln!("not a directory: {dir}");
        return ExitCode::FAILURE;
    }
    let dest = paths::binaries_dir();
    if let Err(e) = std::fs::create_dir_all(&dest) {
        eprintln!("failed to create {dest}: {e}");
        return ExitCode::FAILURE;
    }
    let mut copied = 0;
    for name in ["nvngx_dlssnr.dll", "nvngx.dll", "nvapi64.dll"] {
        let from = src.join(name);
        if from.is_file() {
            if let Err(e) = std::fs::copy(&from, std::path::Path::new(&dest).join(name)) {
                eprintln!("failed to copy {name}: {e}");
                return ExitCode::FAILURE;
            }
            copied += 1;
        }
    }
    println!("imported {copied} file(s) to {dest}");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(command) = args.get(1) else {
        usage();
        return ExitCode::FAILURE;
    };

    match command.as_str() {
        "init" => cmd_init(),
        "setup" => cmd_setup(),
        "start" => cmd_start(),
        "stop" => cmd_stop(),
        "restart" => {
            cmd_stop();
            cmd_start()
        }
        "status" => cmd_status(),
        "doctor" => cmd_doctor(),
        "config" => cmd_config(),
        "runners" => cmd_runners(),
        "detect-gpu" => cmd_detect_gpu(),
        "import-binaries" => cmd_import_binaries(args.get(2)),
        "help" | "--help" | "-h" => {
            usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {other}");
            usage();
            ExitCode::FAILURE
        }
    }
}
