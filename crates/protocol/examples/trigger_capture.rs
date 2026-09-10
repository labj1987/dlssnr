//! Sets `ShmHeader::capture_request` on an already-running instance's mapping, so the
//! layer's very next present writes a matched before/after PNG pair to
//! `$XDG_DATA_HOME/dlssnr/captures` (see `dlssnr_layer::dump`). A manual trigger for
//! the same real feature the GUI/a future `dlssnr-shmctl` would expose as a button --
//! useful stand-alone for exactly this: confirming what the layer just wrote to the
//! screen actually looks right, from the outside, with no GUI needed.
//!
//! Respects `$DLSSNR_SHM`/`$DLSSNR_UID` the same way every other tool in this
//! workspace does -- point it at a specific instance's mapping the same way you'd
//! point `vkcube`/the layer at it.

fn main() {
    let Some(mapping) = dlssnr_protocol::mapping::open() else {
        eprintln!("trigger_capture: failed to open the SHM mapping (see $DLSSNR_SHM/$DLSSNR_UID)");
        std::process::exit(1);
    };
    // Optional `debug_view` override (0-3, see ShmHeader::debug_view's doc comment) --
    // set before the request so the very next present both shows and dumps that mode.
    if let Some(mode) = std::env::args().nth(1).and_then(|s| s.parse::<u32>().ok()) {
        mapping.header().debug_view.store(mode, std::sync::atomic::Ordering::Relaxed);
        println!("trigger_capture: debug_view set to {mode}");
    }
    mapping.header().capture_request.store(1, std::sync::atomic::Ordering::Relaxed);
    println!("trigger_capture: capture_request set -- check $XDG_DATA_HOME/dlssnr/captures on the layer's next present");
}
