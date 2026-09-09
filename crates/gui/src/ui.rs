//! The settings UI: one row per `ShmHeader` setting, grouped the way upstream's Qt
//! GUI grouped them (Model, Motion, Composition, Status) — used as a checklist of
//! what has to exist, not as layout code to port; a per-field `QCheckBox`/`QSpinBox`
//! binder has no logic worth transliterating either way.

use std::sync::atomic::Ordering;

use dlssnr_protocol::enums::{downscaler, mvec_quality, mvec_scale_mode, reversible_mode};
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

    let (enabled, set_enabled) = bind_bool(&shm, |h| &h.enabled);
    model_group.add(&switch_row("Neural rendering", "Off keeps the layer running but presents the original frame", enabled, set_enabled));

    let (style, set_style) = bind_u32(&shm, |h| &h.style);
    model_group.add(&combo_row("Style", &["Default", "Natural", "Cinematic"], style, set_style));

    let (preset, set_preset) = bind_u32(&shm, |h| &h.preset);
    model_group.add(&spin_row("Preset", "0-3, model-defined presets", preset as f32, 0.0, 3.0, 1.0, move |v| set_preset(v as u32)));

    let (intensity, set_intensity) = bind_float(&shm, |h| &h.intensity_bits);
    model_group.add(&spin_row("Intensity", "", intensity, 0.0, 2.0, 0.05, set_intensity));

    let (local_tone, set_local_tone) = bind_float(&shm, |h| &h.local_tone_bits);
    model_group.add(&spin_row("Local tone", "", local_tone, 0.0, 2.0, 0.05, set_local_tone));

    let (local_structure, set_local_structure) = bind_float(&shm, |h| &h.local_structure_bits);
    model_group.add(&spin_row("Local structure", "", local_structure, 0.0, 2.0, 0.05, set_local_structure));

    let (skin_structure, set_skin_structure) = bind_float(&shm, |h| &h.skin_structure_bits);
    model_group.add(&spin_row("Skin structure", "-1 follows local structure", skin_structure, -1.0, 2.0, 0.05, set_skin_structure));

    let (sharpness, set_sharpness) = bind_float(&shm, |h| &h.sharpness_bits);
    model_group.add(&spin_row("Sharpness", "", sharpness, 0.0, 1.0, 0.05, set_sharpness));

    let (auto_mask, set_auto_mask) = bind_bool(&shm, |h| &h.auto_mask);
    model_group.add(&switch_row("Auto mask", "Automatic skin/detail masking", auto_mask, set_auto_mask));

    let (passes, set_passes) = bind_u32(&shm, |h| &h.passes);
    model_group.add(&spin_row("Passes", "How many times the model runs over one frame", passes as f32, 1.0, 30.0, 1.0, move |v| set_passes(v as u32)));

    // --- Motion --------------------------------------------------------------------
    let motion_group = adw::PreferencesGroup::new();
    motion_group.set_title("Motion");

    let (mvec_enabled, set_mvec_enabled) = bind_bool(&shm, |h| &h.mvec_enabled);
    motion_group.add(&switch_row("Estimate motion vectors", "On by default", mvec_enabled, set_mvec_enabled));

    let (mvec_scale, set_mvec_scale) = bind_u32(&shm, |h| &h.mvec_scale_mode);
    motion_group.add(&combo_row("Motion units", &["Normalised", "Pixels", "UV 0..1"], mvec_scale, set_mvec_scale));
    debug_assert_eq!(mvec_scale_mode::PIXELS, 1);

    let (mvec_quality, set_mvec_quality) = bind_u32(&shm, |h| &h.mvec_quality);
    motion_group.add(&combo_row("Motion quality", &["Fast", "Balanced", "Quality"], mvec_quality, set_mvec_quality));
    debug_assert_eq!(mvec_quality::BALANCED, 1);

    // --- Composition -----------------------------------------------------------------
    let comp_group = adw::PreferencesGroup::new();
    comp_group.set_title("Composition");

    let (bypass, set_bypass) = bind_bool(&shm, |h| &h.composition_bypass);
    comp_group.add(&switch_row("Bypass composition", "On: the model's raw answer is the presented frame", bypass, set_bypass));

    let (transfer_strength, set_transfer_strength) = bind_float(&shm, |h| &h.transfer_strength_bits);
    comp_group.add(&spin_row("Transfer strength", "How much of the model's edit reaches the frame", transfer_strength, 0.0, 1.0, 0.05, set_transfer_strength));

    let (colour_strength, set_colour_strength) = bind_float(&shm, |h| &h.colour_strength_bits);
    comp_group.add(&spin_row("Colour strength", "How much of the transfer is allowed to be colour, not just luminance", colour_strength, 0.0, 1.0, 0.05, set_colour_strength));

    let (max_ratio, set_max_ratio) = bind_float(&shm, |h| &h.max_ratio_bits);
    comp_group.add(&spin_row("Max ratio", "The most the pass may multiply/divide a pixel by", max_ratio, 1.0, 8.0, 0.1, set_max_ratio));

    let (working_scale, set_working_scale) = bind_float(&shm, |h| &h.working_scale_bits);
    comp_group.add(&spin_row("Working scale", "Above 1.0 is supersampling", working_scale, 0.25, 2.0, 0.05, set_working_scale));

    let (downscaler_v, set_downscaler) = bind_u32(&shm, |h| &h.scaling_downscaler);
    comp_group.add(&combo_row(
        "Supersampling filter",
        &["FSR1 (unsupported)", "Bicubic", "Catmull-Rom", "Lanczos2", "Lanczos3", "Kaiser2", "Kaiser3", "Magic"],
        downscaler_v,
        set_downscaler,
    ));
    debug_assert_eq!(downscaler::LANCZOS3, 4);

    let (reversible, set_reversible) = bind_u32(&shm, |h| &h.reversible_mode);
    comp_group.add(&combo_row(
        "Reversible mode",
        &["Knee", "Neutwo", "Neutwo replace", "Hybrid", "Hybrid replace"],
        reversible,
        set_reversible,
    ));
    debug_assert_eq!(reversible_mode::KNEE, 0);

    let (hdr_mode, set_hdr_mode) = bind_u32(&shm, |h| &h.hdr_mode);
    comp_group.add(&combo_row("HDR input", &["Auto", "Off", "Force float16"], hdr_mode, set_hdr_mode));

    let toasts = adw::ToastOverlay::new();

    page.add(&model_group);
    page.add(&motion_group);
    page.add(&comp_group);
    page.add(&build_status_group(&shm, &toasts));

    let header = adw::HeaderBar::new();
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
    let layer_row = adw::ActionRow::new();
    layer_row.set_title("Layer");
    group.add(&helper_row);
    group.add(&layer_row);

    let shm_for_timer = std::sync::Arc::clone(shm);
    glib::timeout_add_seconds_local(1, move || {
        let hdr = shm_for_timer.header();
        let helper_state = hdr.helper_state.load(Ordering::Relaxed);
        helper_row.set_subtitle(helper_state_label(helper_state));
        let attached = hdr.layer_attached.load(Ordering::Relaxed) != 0;
        layer_row.set_subtitle(if attached { "attached" } else { "not attached" });
        glib::ControlFlow::Continue
    });

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
