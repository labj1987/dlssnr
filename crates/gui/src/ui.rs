//! The settings UI: one row per `ShmHeader` setting, grouped the way upstream's Qt
//! GUI grouped them (Model, Motion, Composition, Status) — used as a checklist of
//! what has to exist, not as layout code to port; a per-field `QCheckBox`/`QSpinBox`
//! binder has no logic worth transliterating either way.

use std::sync::atomic::Ordering;

use dlssnr_protocol::enums::{colour_mode, downscaler, mvec_quality, mvec_scale_mode, reversible_mode};
use gtk4::prelude::*;
use libadwaita as adw;
use libadwaita::prelude::*;

use crate::shm::{bind_bool, bind_float, bind_u32, Shm};

fn spin_row(title: &str, subtitle: &str, value: f32, lower: f64, upper: f64, step: f64, setter: impl Fn(f32) + 'static) -> adw::SpinRow {
    let adjustment = gtk4::Adjustment::new(value as f64, lower, upper, step, step * 10.0, 0.0);
    let row = adw::SpinRow::new(Some(&adjustment), step, 2);
    row.set_title(title);
    row.set_subtitle(subtitle);
    adjustment.connect_value_changed(move |adj| setter(adj.value() as f32));
    row
}

fn switch_row(title: &str, subtitle: &str, active: bool, setter: impl Fn(bool) + 'static) -> adw::SwitchRow {
    let row = adw::SwitchRow::new();
    row.set_title(title);
    row.set_subtitle(subtitle);
    row.set_active(active);
    row.connect_active_notify(move |row| setter(row.is_active()));
    row
}

fn combo_row(title: &str, options: &[&str], selected: u32, setter: impl Fn(u32) + 'static) -> adw::ComboRow {
    let model = gtk4::StringList::new(options);
    let row = adw::ComboRow::new();
    row.set_title(title);
    row.set_model(Some(&model));
    row.set_selected(selected.min(options.len() as u32 - 1));
    row.connect_selected_notify(move |row| setter(row.selected()));
    row
}

/// GDK on Linux X11/Wayland uses XKB hardware codes (evdev + 8).
fn evdev_keycode(hardware: u32) -> Option<u32> {
    hardware.checked_sub(8).filter(|&code| code > 0 && code <= 767)
}

fn hotkey_row(initial: u32, setter: impl Fn(u32) + 'static) -> adw::ActionRow {
    let row = adw::ActionRow::builder().title("Toggle key")
        .subtitle("Save a single key binding; in-game hotkey polling is not available yet") .build();
    let label = |code| if code == 0 { "Unbound".to_owned() } else { format!("Key {code}") };
    let button = gtk4::Button::with_label(&label(initial));
    button.set_valign(gtk4::Align::Center);
    let clear = gtk4::Button::with_label("Clear");
    clear.set_valign(gtk4::Align::Center);
    let value = std::rc::Rc::new(std::cell::Cell::new(initial));
    let capturing = std::rc::Rc::new(std::cell::Cell::new(false));
    let setter = std::rc::Rc::new(setter);
    let controller = gtk4::EventControllerKey::new();
    controller.set_propagation_phase(gtk4::PropagationPhase::Capture);
    {
        let capturing = capturing.clone();
        button.connect_clicked(move |button| {
            capturing.set(true);
            button.set_label("Press a key (Esc cancels)");
            button.grab_focus();
        });
    }
    {
        let button = button.downgrade();
        let capturing = capturing.clone();
        let value = value.clone();
        let setter = setter.clone();
        controller.connect_key_pressed(move |_, key, hardware, _| {
            if !capturing.get() { return glib::Propagation::Proceed; }
            if let Some(button) = button.upgrade() {
                if key != gtk4::gdk::Key::Escape {
                    if let Some(code) = evdev_keycode(hardware) {
                        setter(code);
                        value.set(code);
                    }
                }
                capturing.set(false);
                button.set_label(&label(value.get()));
            }
            glib::Propagation::Stop
        });
    }
    {
        let capturing = capturing.clone();
        let value = value.clone();
        button.connect_has_focus_notify(move |button| {
            if !button.has_focus() && capturing.replace(false) {
                button.set_label(&label(value.get()));
            }
        });
    }
    {
        let button = button.clone();
        clear.connect_clicked(move |_| {
            capturing.set(false);
            value.set(0);
            setter(0);
            button.set_label("Unbound");
        });
    }
    button.add_controller(controller);
    row.add_suffix(&button);
    row.add_suffix(&clear);
    row
}

pub fn build_ui(app: &adw::Application) {
    let Some(shm) = Shm::open() else {
        build_error_window(app);
        return;
    };
    let shm = shm.0;

    let page = adw::PreferencesPage::new();

    // --- Model -------------------------------------------------------------------
    let model_group = adw::PreferencesGroup::new();
    model_group.set_title("Model");

    let (enabled, set_enabled) = bind_bool(&shm, Some("enabled"), |h| &h.enabled);
    model_group.add(&switch_row("Neural rendering", "Off keeps the layer running but presents the original frame", enabled, set_enabled));

    let (style, set_style) = bind_u32(&shm, Some("style"), |h| &h.style);
    model_group.add(&combo_row("Style", &["Default", "Natural", "Cinematic"], style, set_style));

    let (preset, set_preset) = bind_u32(&shm, Some("preset"), |h| &h.preset);
    model_group.add(&spin_row("Preset", "0-3, model-defined presets", preset as f32, 0.0, 3.0, 1.0, move |v| set_preset(v as u32)));

    let (intensity, set_intensity) = bind_float(&shm, Some("intensity"), |h| &h.intensity_bits);
    model_group.add(&spin_row("Intensity", "", intensity, 0.0, 2.0, 0.05, set_intensity));

    let (local_tone, set_local_tone) = bind_float(&shm, Some("local_tone"), |h| &h.local_tone_bits);
    model_group.add(&spin_row("Local tone", "", local_tone, 0.0, 2.0, 0.05, set_local_tone));

    let (local_structure, set_local_structure) = bind_float(&shm, Some("local_structure"), |h| &h.local_structure_bits);
    model_group.add(&spin_row("Local structure", "", local_structure, 0.0, 2.0, 0.05, set_local_structure));

    let (skin_structure, set_skin_structure) = bind_float(&shm, Some("skin_structure"), |h| &h.skin_structure_bits);
    model_group.add(&spin_row("Skin structure", "-1 follows local structure", skin_structure, -1.0, 2.0, 0.05, set_skin_structure));

    let (sharpness, set_sharpness) = bind_float(&shm, Some("sharpness"), |h| &h.sharpness_bits);
    model_group.add(&spin_row("Sharpness", "", sharpness, 0.0, 1.0, 0.05, set_sharpness));

    let (auto_mask, set_auto_mask) = bind_bool(&shm, Some("auto_mask"), |h| &h.auto_mask);
    model_group.add(&switch_row("Auto mask", "Automatic skin/detail masking", auto_mask, set_auto_mask));

    let (passes, set_passes) = bind_u32(&shm, Some("passes"), |h| &h.passes);
    model_group.add(&spin_row("Passes", "How many times the model runs over one frame", passes as f32, 1.0, 30.0, 1.0, move |v| set_passes(v as u32)));

    let (toggle_key, set_toggle_key) = bind_u32(&shm, Some("toggle_key"), |h| &h.toggle_key);
    model_group.add(&hotkey_row(toggle_key, set_toggle_key));

    // --- Motion --------------------------------------------------------------------
    let motion_group = adw::PreferencesGroup::new();
    motion_group.set_title("Motion");

    let (mvec_enabled, set_mvec_enabled) = bind_bool(&shm, Some("mvec_enabled"), |h| &h.mvec_enabled);
    motion_group.add(&switch_row("Estimate motion vectors", "On by default", mvec_enabled, set_mvec_enabled));

    let (mvec_scale, set_mvec_scale) = bind_u32(&shm, Some("mvec_scale_mode"), |h| &h.mvec_scale_mode);
    motion_group.add(&combo_row("Motion units", &["Normalised", "Pixels", "UV 0..1"], mvec_scale, set_mvec_scale));
    debug_assert_eq!(mvec_scale_mode::PIXELS, 1);

    let (mvec_quality, set_mvec_quality) = bind_u32(&shm, Some("mvec_quality"), |h| &h.mvec_quality);
    motion_group.add(&combo_row("Motion quality", &["Fast", "Balanced", "Quality"], mvec_quality, set_mvec_quality));
    debug_assert_eq!(mvec_quality::BALANCED, 1);

    // --- Composition -----------------------------------------------------------------
    let comp_group = adw::PreferencesGroup::new();
    comp_group.set_title("Composition");

    let (bypass, set_bypass) = bind_bool(&shm, Some("composition_bypass"), |h| &h.composition_bypass);
    comp_group.add(&switch_row("Bypass composition", "On: the model's raw answer is the presented frame", bypass, set_bypass));

    let (transfer_strength, set_transfer_strength) = bind_float(&shm, Some("transfer_strength"), |h| &h.transfer_strength_bits);
    comp_group.add(&spin_row("Transfer strength", "How much of the model's edit reaches the frame", transfer_strength, 0.0, 1.0, 0.05, set_transfer_strength));

    let (colour_strength, set_colour_strength) = bind_float(&shm, Some("colour_strength"), |h| &h.colour_strength_bits);
    comp_group.add(&spin_row("Colour strength", "How much of the transfer is allowed to be colour, not just luminance", colour_strength, 0.0, 1.0, 0.05, set_colour_strength));

    let (max_ratio, set_max_ratio) = bind_float(&shm, Some("max_ratio"), |h| &h.max_ratio_bits);
    comp_group.add(&spin_row("Max ratio", "The most the pass may multiply/divide a pixel by", max_ratio, 1.0, 8.0, 0.1, set_max_ratio));

    let (working_scale, set_working_scale) = bind_float(&shm, Some("working_scale"), |h| &h.working_scale_bits);
    comp_group.add(&spin_row("Working scale", "Above 1.0 is supersampling", working_scale, 0.25, 2.0, 0.05, set_working_scale));

    let (downscaler_v, set_downscaler) = bind_u32(&shm, Some("scaling_downscaler"), |h| &h.scaling_downscaler);
    comp_group.add(&combo_row(
        "Supersampling filter",
        &["FSR1 (unsupported)", "Bicubic", "Catmull-Rom", "Lanczos2", "Lanczos3", "Kaiser2", "Kaiser3", "Magic"],
        downscaler_v,
        set_downscaler,
    ));
    debug_assert_eq!(downscaler::LANCZOS3, 4);

    let (reversible, set_reversible) = bind_u32(&shm, Some("reversible_mode"), |h| &h.reversible_mode);
    comp_group.add(&combo_row(
        "Reversible mode",
        &["Knee", "Neutwo", "Neutwo replace", "Hybrid", "Hybrid replace"],
        reversible,
        set_reversible,
    ));
    debug_assert_eq!(reversible_mode::KNEE, 0);

    let (hdr_mode, set_hdr_mode) = bind_u32(&shm, Some("hdr_mode"), |h| &h.hdr_mode);
    comp_group.add(&combo_row("HDR input", &["Auto", "Off", "Force float16"], hdr_mode, set_hdr_mode));

    let (colour_mode, set_colour_mode) = bind_u32(&shm, Some("colour_mode"), |h| &h.colour_mode);
    comp_group.add(&combo_row("Colour mode", &["Auto", "Force display-referred", "Force linear HDR"], colour_mode, set_colour_mode));
    debug_assert_eq!(colour_mode::AUTO, 0);

    let hdr_group = adw::PreferencesGroup::new();
    hdr_group.set_title("HDR white point");
    hdr_group.set_description(Some("Saved for HDR processing. The current capture pipeline does not yet apply these controls."));
    let (source, set_source) = bind_u32(&shm, Some("white_point_source"), |h| &h.white_point_source);
    hdr_group.add(&combo_row("White point source", &["Manual", "Measured"], source, set_source));
    let (white, set_white) = bind_float(&shm, Some("white_point"), |h| &h.white_point_bits);
    hdr_group.add(&spin_row("Manual white point", "Linear-light reference", white, 0.01, 10000.0, 0.1, set_white));
    let (scale, set_scale) = bind_float(&shm, Some("white_point_scale"), |h| &h.white_point_scale_bits);
    hdr_group.add(&spin_row("White point scale", "Multiplier", scale, 0.01, 100.0, 0.05, set_scale));
    let (trim, set_trim) = bind_float(&shm, Some("white_point_trim"), |h| &h.white_point_trim_bits);
    hdr_group.add(&spin_row("White point trim", "Calibration multiplier", trim, 0.01, 100.0, 0.05, set_trim));

    let (transfer, set_transfer) = bind_u32(&shm, Some("transfer"), |h| &h.transfer);
    comp_group.add(&combo_row("Transfer mode", &["Classic", "Matched residual", "Native + edit"], transfer, set_transfer));

    let (unlock_passes, set_unlock_passes) = bind_bool(&shm, Some("unlock_passes"), |h| &h.unlock_passes);
    comp_group.add(&switch_row("Unlock pass limit", "Allow more passes than the normal ceiling", unlock_passes, set_unlock_passes));

    let (apply_model, set_apply_model) = bind_bool(&shm, Some("apply_model"), |h| &h.apply_model);
    comp_group.add(&switch_row("Apply model edit", "Off presents the clean frame — capture/transport/round-trip still run, for an honest A/B", apply_model, set_apply_model));

    let (hold_frame, set_hold_frame) = bind_bool(&shm, Some("hold_frame"), |h| &h.hold_frame);
    comp_group.add(&switch_row("Hold frame", "Freeze the frame the pass works on, to re-run composition over the same picture", hold_frame, set_hold_frame));

    // --- Compare and debug ----------------------------------------------------------
    let debug_group = adw::PreferencesGroup::new();
    // Not "Compare & debug" -- `AdwPreferencesGroup::title` is parsed as Pango markup,
    // and a bare `&` breaks it (confirmed via a real run: "Failed to set text ...
    // Entity did not end with a semicolon").
    debug_group.set_title("Compare and debug");

    let (compare_mode, set_compare_mode) = bind_u32(&shm, Some("compare_mode"), |h| &h.compare_mode);
    debug_group.add(&combo_row("Compare mode", &["Off", "Side by side", "Wipe"], compare_mode, set_compare_mode));

    let (compare_split, set_compare_split) = bind_float(&shm, Some("compare_split"), |h| &h.compare_split_bits);
    debug_group.add(&spin_row("Compare split", "Wipe position, 0=left edge, 1=right edge", compare_split, 0.0, 1.0, 0.05, set_compare_split));

    let (compare_zoom, set_compare_zoom) = bind_float(&shm, Some("compare_zoom"), |h| &h.compare_zoom_bits);
    debug_group.add(&spin_row("Compare zoom", "", compare_zoom, 0.1, 8.0, 0.1, set_compare_zoom));

    let (compare_swap, set_compare_swap) = bind_bool(&shm, Some("compare_swap"), |h| &h.compare_swap);
    debug_group.add(&switch_row("Swap compare sides", "", compare_swap, set_compare_swap));

    let (debug_view, set_debug_view) = bind_u32(&shm, Some("debug_view"), |h| &h.debug_view);
    debug_group.add(&combo_row(
        "Debug view",
        &["Off", "Original / proxy", "Model's raw answer", "Amplified diff"],
        debug_view,
        set_debug_view,
    ));

    let toasts = adw::ToastOverlay::new();

    page.add(&model_group);
    page.add(&motion_group);
    page.add(&comp_group);
    page.add(&hdr_group);
    page.add(&debug_group);
    page.add(&build_status_group(&shm, &toasts));

    let header = adw::HeaderBar::new();
    let about_btn = gtk4::Button::builder().icon_name("help-about-symbolic").tooltip_text("About").build();
    header.pack_end(&about_btn);
    let content = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    content.append(&header);
    content.append(&page);
    toasts.set_child(Some(&content));

    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("dlssnr")
        .default_width(620)
        .default_height(760)
        .content(&toasts)
        .build();

    {
        let window = window.clone();
        about_btn.connect_clicked(move |_| {
            let dialog = adw::AboutDialog::builder()
                .application_name("dlssnr")
                .version(env!("CARGO_PKG_VERSION"))
                .developers(vec!["Linnard Alex Brown Jr."])
                .comments("Vulkan layer and settings GUI for running NVIDIA DLSS 5 Neural Rendering on Linux/Proton games.")
                .build();
            dialog.add_acknowledgement_section(Some("Built with"), &["Claude Code (Anthropic)"]);
            dialog.present(Some(&window));
        });
    }

    window.present();
}

/// A read-only status group, refreshed on a timer -- helper/layer liveness, frame
/// counts. Nothing here is a setting; it only ever reads.
///
/// The one exception is the "NGX binaries" row's Import button: unlike everything
/// else here, it's an action, not a live readout, because it's the only place besides
/// `dlssnr-cli import-binaries` to get NVIDIA's DLLs into `binaries_dir()` -- there's
/// no separate menu for it.
fn build_status_group(shm: &std::sync::Arc<dlssnr_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Status");

    let helper_row = adw::ActionRow::new();
    helper_row.set_title("Helper");
    let start_stop_button = gtk4::Button::with_label("Start");
    start_stop_button.set_valign(gtk4::Align::Center);
    helper_row.add_suffix(&start_stop_button);
    let layer_row = adw::ActionRow::new();
    layer_row.set_title("Layer");
    group.add(&helper_row);
    group.add(&layer_row);

    let shm_for_timer = std::sync::Arc::clone(shm);
    let start_stop_button_for_timer = start_stop_button.clone();
    glib::timeout_add_seconds_local(1, move || {
        let hdr = shm_for_timer.header();
        let helper_state = hdr.helper_state.load(Ordering::Relaxed);
        helper_row.set_subtitle(helper_state_label(helper_state));
        let attached = hdr.layer_attached.load(Ordering::Relaxed) != 0;
        layer_row.set_subtitle(if attached { "attached" } else { "not attached" });
        // Keyed off the actual OS-level pid-file check (what start/stop manage), not
        // the SHM helper_state above -- those can briefly disagree right after a
        // start/stop (e.g. STARTING vs. the process not existing yet) and the button
        // should reflect what clicking it will actually do, not the helper's own
        // self-reported state.
        if start_stop_button_for_timer.is_sensitive() {
            start_stop_button_for_timer.set_label(if dlssnr_supervisor::is_running().is_some() { "Stop" } else { "Start" });
        }
        glib::ControlFlow::Continue
    });

    {
        let toasts = toasts.clone();
        start_stop_button.connect_clicked(move |button| {
            let toasts = toasts.clone();
            if dlssnr_supervisor::is_running().is_some() {
                // Stopping waits up to 5s for a graceful exit before escalating to
                // SIGKILL (see dlssnr_supervisor::stop) -- a brief, bounded main-thread
                // block on an explicit user click, not worth the async plumbing this
                // small a GUI doesn't otherwise need.
                button.set_sensitive(false);
                button.set_label("Stopping…");
                match dlssnr_supervisor::stop(std::time::Duration::from_secs(5)) {
                    Ok(()) => toasts.add_toast(adw::Toast::new("Helper stopped")),
                    Err(e) => toasts.add_toast(adw::Toast::new(&format!("Stop failed: {e}"))),
                }
                button.set_sensitive(true);
            } else {
                let cfg = dlssnr_supervisor::Config::load();
                match dlssnr_supervisor::start(&cfg) {
                    Ok(started) => toasts.add_toast(adw::Toast::new(&format!("Helper started (pid {})", started.pid))),
                    Err(e) => toasts.add_toast(adw::Toast::new(&format!("Start failed: {e}"))),
                }
            }
        });
    }

    let binaries_row = adw::ActionRow::new();
    binaries_row.set_title("NGX binaries");
    binaries_row.set_subtitle(&binaries_status_subtitle());
    let import_button = gtk4::Button::with_label("Import…");
    import_button.set_valign(gtk4::Align::Center);
    binaries_row.add_suffix(&import_button);
    group.add(&binaries_row);

    let toasts = toasts.clone();
    import_button.connect_clicked(move |button| {
        let toasts = toasts.clone();
        let binaries_row = binaries_row.clone();
        let parent = button.root().and_downcast::<gtk4::Window>();
        let dialog = gtk4::FileDialog::builder().title("Select folder containing NVIDIA NGX DLLs").build();
        dialog.select_folder(parent.as_ref(), None::<&gio::Cancellable>, move |result| {
            let Ok(folder) = result else { return };
            let Some(path) = folder.path() else { return };
            match crate::binaries::import_from(&path) {
                Ok(0) => toasts.add_toast(adw::Toast::new("No matching DLLs found in that folder")),
                Ok(n) => {
                    toasts.add_toast(adw::Toast::new(&format!("Imported {n} file(s) -- restart the helper to load them")));
                    binaries_row.set_subtitle(&binaries_status_subtitle());
                }
                Err(e) => toasts.add_toast(adw::Toast::new(&format!("Import failed: {e}"))),
            }
        });
    });

    group
}

fn binaries_status_subtitle() -> String {
    if crate::binaries::dir().join("nvngx_dlssnr.dll").is_file() {
        "nvngx_dlssnr.dll present".to_string()
    } else {
        "nvngx_dlssnr.dll missing".to_string()
    }
}

fn helper_state_label(state: u32) -> &'static str {
    use dlssnr_protocol::enums::helper_state::*;
    match state {
        STARTING => "starting",
        NO_VULKAN => "no NVIDIA Vulkan device",
        NO_BINARIES => "NGX binaries missing",
        MODEL_FAILED => "model failed to load",
        RUNNING => "running",
        STOPPED => "stopped",
        _ => "unknown",
    }
}

fn build_error_window(app: &adw::Application) {
    let status = adw::StatusPage::builder()
        .icon_name("dialog-error-symbolic")
        .title("Couldn't open the shared-memory mapping")
        .description("Check the helper's log; dlssnr-cli doctor may also help.")
        .build();
    let window = adw::ApplicationWindow::builder().application(app).title("dlssnr").content(&status).build();
    window.present();
}

#[cfg(test)] mod hotkey_tests {
    use super::*;
    #[test] fn hardware_codes_are_converted_without_underflow() {
        assert_eq!(evdev_keycode(95),Some(87)); // F11: XKB -> Linux evdev.
        assert_eq!(evdev_keycode(38),Some(30)); // physical A key on evdev.
        assert_eq!(evdev_keycode(0),None);
        assert_eq!(evdev_keycode(8),None);
        assert_eq!(evdev_keycode(u32::MAX),None);
    }
}
