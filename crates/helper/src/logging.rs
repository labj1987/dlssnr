//! A minimal logging sink: `DLSSNR_LOG` names a file to append to, otherwise stderr.
//! One handle for the process, not one per call site. Same shape as
//! `dlssnr_layer::logging` — kept as a separate copy rather than a shared crate since
//! it's this small and the two crates otherwise share nothing OS-specific here
//! (`std::env`/`std::fs`/`std::io` are already portable).

use std::fmt::Arguments;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};

enum Sink {
    File(File),
    Stderr,
}

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Sink::File(f) => f.write(buf),
            Sink::Stderr => std::io::stderr().write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Sink::File(f) => f.flush(),
            Sink::Stderr => std::io::stderr().flush(),
        }
    }
}

static SINK: OnceLock<Mutex<Sink>> = OnceLock::new();

fn sink() -> &'static Mutex<Sink> {
    SINK.get_or_init(|| {
        let sink = std::env::var("DLSSNR_LOG")
            .ok()
            .filter(|p| !p.is_empty())
            .and_then(|path| OpenOptions::new().create(true).append(true).open(path).ok())
            .map(Sink::File)
            .unwrap_or(Sink::Stderr);
        Mutex::new(sink)
    })
}

/// Writes one `[dlssnr-helper] ...` line. Never called directly -- use the
/// [`crate::log!`] macro so every call site gets the same prefix and newline handling.
pub fn log(args: Arguments<'_>) {
    let Ok(mut sink) = sink().lock() else { return };
    let _ = writeln!(sink, "[dlssnr-helper] {args}");
    let _ = sink.flush();
}

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        $crate::logging::log(format_args!($($arg)*))
    };
}
