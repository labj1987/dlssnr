use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::enums::{mvec_quality, mvec_scale_mode};
use crate::{DEFAULT_MAX_PASSES, HEADER_BYTES, MAX_PASSES, NAME_BYTES, REASON_BYTES, SHM_MAGIC, SHM_VERSION};

/// One pass's overrides. Every field is present; `override_mask` says which of them
/// mean anything — see [`crate::enums::pass_override`].
///
/// `Default` gives the all-zero value a fresh (or `ftruncate`d, zero-filled) mapping
/// naturally starts with; [`PassControl::reset_to_defaults`] is the separate,
/// explicit "what a pass should actually mean until told otherwise" values.
#[repr(C)]
#[derive(Default)]
pub struct PassControl {
    pub override_mask: AtomicU32,
    pub intensity_bits: AtomicU32,
    pub local_tone_bits: AtomicU32,
    pub local_structure_bits: AtomicU32,
    pub skin_structure_bits: AtomicU32,
    pub sharpness_bits: AtomicU32,
    pub style: AtomicU32,
    pub preset: AtomicU32,
    pub auto_mask: AtomicU32,
}

impl PassControl {
    fn reset_to_defaults(&self) {
        self.override_mask.store(0, Ordering::Relaxed);
        self.intensity_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.local_tone_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.local_structure_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        // -1 follows local structure; it is not a strength of zero.
        self.skin_structure_bits.store((-1.0f32).to_bits(), Ordering::Relaxed);
        self.sharpness_bits.store(0.0f32.to_bits(), Ordering::Relaxed);
        // Inert until override_mask names them, but initialized to the global defaults
        // so a pass that is switched on later starts from what the rest of the frame is
        // already doing.
        self.style.store(0, Ordering::Relaxed);
        self.preset.store(0, Ordering::Relaxed);
        self.auto_mask.store(1, Ordering::Relaxed);
    }
}

/// A pass's settings after the global values and its own overrides have been merged.
/// Plain floats: this is the resolved answer, not shared state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassTuning {
    pub intensity: f32,
    pub local_tone: f32,
    pub local_structure: f32,
    /// -1 follows local structure; it is not a strength of zero.
    pub skin_structure: f32,
    pub sharpness: f32,
    pub style: u32,
    pub preset: u32,
    pub auto_mask: u32,
}

impl Default for PassTuning {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            local_tone: 1.0,
            local_structure: 1.0,
            skin_structure: -1.0,
            sharpness: 0.0,
            style: 0,
            preset: 0,
            auto_mask: 1,
        }
    }
}

impl PassTuning {
    /// Everything the model latches when its feature is built. Sharpness is absent
    /// because it is read at evaluate, and so is the only one of these a running
    /// feature will actually follow.
    pub fn same_create_params(&self, other: &PassTuning) -> bool {
        self.intensity == other.intensity
            && self.local_tone == other.local_tone
            && self.local_structure == other.local_structure
            && self.skin_structure == other.skin_structure
            && self.style == other.style
            && self.preset == other.preset
            && self.auto_mask == other.auto_mask
    }
}

/// The shared-memory header. See the crate-level docs for the mapping this sits at the
/// front of.
///
/// `#[repr(C)]` and built entirely from `AtomicU32` and `UnsafeCell<[u8; N]>` fields:
/// every field here has to land at the offset a same-order, same-width C struct would
/// put it at, because the whole point is that two different toolchains (this crate on
/// Linux, and whatever builds the Windows helper) agree on the layout without either
/// one reading the other's header file.
///
/// `Default` gives the all-zero value that a fresh, `ftruncate`d mapping already holds
/// before anyone touches it — useful for tests and for constructing one off the heap.
/// It is deliberately not "the real defaults": [`ShmHeader::init_defaults`] is the
/// explicit call (mirroring the original `ShmInitDefaults`) that sets the values this
/// protocol actually wants a fresh session to start at (`enabled = 1`,
/// `intensity = 1.0`, and so on).
#[repr(C)]
pub struct ShmHeader {
    pub magic: AtomicU32,
    pub version: AtomicU32,

    /// The frame handshake. The layer bumps `seq_req` after writing a proxy; the helper
    /// answers by storing the same number into `seq_resp` once the model's answer is in
    /// the output region.
    pub seq_req: AtomicU32,
    pub seq_resp: AtomicU32,
    pub width: AtomicU32,
    pub height: AtomicU32,
    /// Always 1 (RGBA byte order); kept so a mismatched-version helper is not silently
    /// wrong.
    pub format: AtomicU32,
    pub quit: AtomicU32,
    pub heartbeat: AtomicU32,

    /// Bumped by whoever writes a setting. The layer and the helper watch this rather
    /// than re-reading every field every frame.
    pub control_seq: AtomicU32,
    /// Bumped only when something the model latches at feature creation changes. The
    /// helper rebuilds its features on this and debounces the rebuild — bumping it
    /// every frame exhausts the driver's latches and the model stops responding until
    /// the process restarts.
    pub tuning_seq: AtomicU32,

    // --- the model ----------------------------------------------------------------
    pub enabled: AtomicU32,
    pub passes: AtomicU32,
    pub unlock_passes: AtomicU32,
    pub preset: AtomicU32,
    pub style: AtomicU32,
    pub auto_mask: AtomicU32,
    pub intensity_bits: AtomicU32,
    pub local_tone_bits: AtomicU32,
    pub local_structure_bits: AtomicU32,
    pub skin_structure_bits: AtomicU32,
    pub sharpness_bits: AtomicU32,

    // --- the composition ------------------------------------------------------------
    /// How much of the model's edit reaches the frame, and how much of it is allowed to
    /// be colour rather than luminance. Separating the two is what keeps saturated
    /// highlights from shifting hue.
    pub transfer_strength_bits: AtomicU32,
    pub colour_strength_bits: AtomicU32,
    /// The most the pass may multiply or divide a pixel by. The transfer is a ratio,
    /// and a ratio against a near-black proxy pixel is unbounded without one.
    pub max_ratio_bits: AtomicU32,
    /// How a model that worked below the frame's size is brought back. 0 classic, 1
    /// matched residual, 2 native + edit.
    pub transfer: AtomicU32,
    /// 0 off, 1 the picture the model was shown, 2 its raw answer, 3 what it changed, amplified.
    pub debug_view: AtomicU32,
    pub debug_scale_bits: AtomicU32,
    pub white_point_bits: AtomicU32,
    pub white_point_scale_bits: AtomicU32,
    pub white_point_source: AtomicU32,
    pub white_point_trim_bits: AtomicU32,
    /// What fraction of the frame's resolution the model works at. The frame itself is
    /// never reduced: only the model's contribution is computed at this scale and
    /// resized. Above 1.0 is supersampling; below 1.0 also cuts what crosses the shared
    /// memory, quadratically.
    pub working_scale_bits: AtomicU32,
    /// 0 off, 1 side by side, 2 a wipe.
    pub compare_mode: AtomicU32,
    pub compare_split_bits: AtomicU32,
    pub compare_zoom_bits: AtomicU32,
    pub compare_swap: AtomicU32,
    pub colour_mode: AtomicU32,
    /// Writes one set of matched before/after frames per session when the layer next presents.
    pub capture_request: AtomicU32,
    /// A Linux key code the layer watches to toggle the pass, or 0 for none.
    pub toggle_key: AtomicU32,
    /// Which proxy the model is shown, and whether its answer is composed or
    /// substituted. See [`crate::enums::reversible_mode`].
    pub reversible_mode: AtomicU32,
    /// Whether the model's edit is applied at all. Off keeps the whole pass running —
    /// the capture, the round trip, the encode — and simply presents the clean frame,
    /// which is what makes an honest A/B possible.
    pub apply_model: AtomicU32,
    /// Freeze the frame the pass works on, so changing a setting re-runs the
    /// composition over the same picture instead of over whatever the game has drawn
    /// since.
    pub hold_frame: AtomicU32,
    /// The filter for the supersampling down-leg. See [`crate::enums::downscaler`]; only
    /// read when `working_scale > 1`.
    pub scaling_downscaler: AtomicU32,

    // --- status, written by the helper ----------------------------------------------
    pub helper_state: AtomicU32,
    pub model_up: AtomicU32,
    pub helper_frames_lo: AtomicU32,
    pub helper_frames_hi: AtomicU32,
    pub helper_eval_ms_bits: AtomicU32,
    pub helper_upload_ms_bits: AtomicU32,
    pub helper_readback_ms_bits: AtomicU32,
    pub helper_vram_mb: AtomicU32,
    /// How many NGX features are actually built.
    pub helper_features: AtomicU32,
    /// What the VRAM budget currently allows.
    pub helper_pass_ceiling: AtomicU32,

    // --- status, written by the layer -------------------------------------------------
    pub layer_attached: AtomicU32,
    pub layer_frames_lo: AtomicU32,
    pub layer_frames_hi: AtomicU32,
    pub layer_width: AtomicU32,
    pub layer_height: AtomicU32,
    pub layer_format: AtomicU32,
    pub layer_composition_up: AtomicU32,
    pub layer_ms_bits: AtomicU32,
    pub layer_measured_white_bits: AtomicU32,
    pub layer_heartbeat: AtomicU32,

    // --- free text, each guarded by its own sequence number --------------------------
    // Bumped after the bytes are written, so a reader that sees an unchanged number is
    // looking at a whole string. See `store_seq_guarded`/`load_seq_guarded` below.
    pub helper_reason_seq: AtomicU32,
    helper_reason: UnsafeCell<[u8; REASON_BYTES]>,
    pub layer_reason_seq: AtomicU32,
    layer_reason: UnsafeCell<[u8; REASON_BYTES]>,
    pub game_name_seq: AtomicU32,
    game_name: UnsafeCell<[u8; NAME_BYTES]>,

    pub pass: [PassControl; MAX_PASSES],

    // Appended after the pass array on purpose: everything before it has a pinned
    // offset, and a new field inserted higher up would move all of them.
    pub mvec_enabled: AtomicU32,
    pub mvec_scale_mode: AtomicU32,
    pub mvec_quality: AtomicU32,
    /// How far the helper has answered *successfully*. `seq_resp` says a frame came
    /// back; this says it was worth using, so the layer can present the game's own
    /// frame when it was not.
    pub seq_ok: AtomicU32,

    /// 0: the composition blends the model's edit onto the frame under the strength and
    /// guard limits. 1: no composition at all — the model's raw answer IS the presented
    /// frame. Default 1: the composition is off until the user turns it on.
    pub composition_bypass: AtomicU32,
    /// Wall-clock milliseconds the helper waits after the last tuning change before it
    /// rebuilds a feature, and between one rebuild and the next.
    pub rebuild_settle_ms: AtomicU32,

    /// The raster the helper actually answered, echoed before `seq_resp`. Without this
    /// echo a swapchain waiting on its own request could be satisfied by another
    /// swapchain's answer and copy the wrong number of bytes.
    pub answered_w: AtomicU32,
    pub answered_h: AtomicU32,

    // --- the dma-buf exchange, carried entirely through this header ------------------
    /// Bumped when the fd or the image behind it changes; the importer re-opens on a
    /// new sequence.
    pub proxy_export_seq: AtomicU32,
    /// The helper's Linux pid.
    pub proxy_pid: AtomicU32,
    /// Its fd number for the proxy image.
    pub proxy_fd: AtomicU32,
    /// Rebuilt (size or channel) since.
    pub proxy_gen: AtomicU32,
    pub answer_export_seq: AtomicU32,
    pub answer_pid: AtomicU32,
    pub answer_fd: AtomicU32,
    pub answer_gen: AtomicU32,
    /// The importer's echo: the export sequence each side has taken a reference at,
    /// restated every frame, 0 for none.
    pub layer_proxy_seq: AtomicU32,
    pub layer_answer_seq: AtomicU32,

    // --- the HDR input path ----------------------------------------------------------
    pub hdr_mode: AtomicU32,
    pub hdr_detected: AtomicU32,
    pub hdr_active: AtomicU32,
    pub proxy_format: AtomicU32,
    /// What the proxy bytes in the shared region actually are for the request being
    /// made: the layer writes this immediately before `seq_req`, so the helper reads
    /// the width from the same statement that announced the pixels.
    pub hdr_encode: AtomicU32,
    /// Motion payload is valid for this request; published before seq_req.
    pub frame_mvec_valid: AtomicU32,
    /// Snapshot of units used to encode this request, independent of GUI changes.
    pub frame_mvec_scale_mode: AtomicU32,
}

// The whole point of a shared, memory-mapped struct like this is that every writer
// synchronizes through its atomics (or, for the free-text fields, through the
// sequence-number-guarded protocol in `store_seq_guarded`/`load_seq_guarded` below) —
// never through Rust's own aliasing rules, which don't apply to memory another process
// can write at any time regardless of what this process believes about it. That is
// exactly the invariant that makes it safe to treat `&ShmHeader` as shareable across
// threads within this process too.
unsafe impl Sync for ShmHeader {}

impl Default for ShmHeader {
    fn default() -> Self {
        // SAFETY: every field is either an `AtomicU32` (valid for any `u32` bit
        // pattern, including all-zero) or an `UnsafeCell<[u8; N]>` (valid for any byte
        // pattern) or an array of `PassControl`, itself made only of `AtomicU32`s — so
        // the all-zero bit pattern `zeroed()` produces is a valid value of every field,
        // and therefore of the whole struct. This is exactly the value a fresh,
        // `ftruncate`d (zero-filled) mapping already holds before anyone touches it.
        unsafe { std::mem::zeroed() }
    }
}

const _: () = assert!(std::mem::size_of::<ShmHeader>() <= HEADER_BYTES, "ShmHeader outgrew its region");

// The layout, pinned. Every process that maps this file agrees on where each field is
// only because they were compiled from the same header. A field inserted anywhere but
// the end silently moves everything after it, and a build that has not caught up then
// reads its neighbor's value — which is not a crash, it is a status display quietly
// reporting a nonsensical number for a flag that is 0 or 1. If any of these fire, the
// layout changed: bump `SHM_VERSION` in the same commit, then update these numbers.
const _: () = assert!(std::mem::size_of::<ShmHeader>() == 1960, "the header layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, enabled) == 44, "layout changed -- bump SHM_VERSION");
const _: () = assert!(
    std::mem::offset_of!(ShmHeader, transfer_strength_bits) == 88,
    "layout changed -- bump SHM_VERSION"
);
const _: () = assert!(std::mem::offset_of!(ShmHeader, helper_state) == 176, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, pass) == 780, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, mvec_enabled) == 1860, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, hdr_mode) == 1932, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, frame_mvec_valid) == 1952, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, frame_mvec_scale_mode) == 1956, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::size_of::<PassControl>() == 36, "layout changed -- bump SHM_VERSION");

impl ShmHeader {
    /// Resets every field to the defaults a freshly created mapping should hold. Takes
    /// `&self` rather than `&mut self`: this is called on a live mapping another
    /// process may be reading, same as everything else here, so it goes through the
    /// atomics rather than a bulk memory write.
    pub fn init_defaults(&self) {
        self.magic.store(SHM_MAGIC, Ordering::Relaxed);
        self.version.store(SHM_VERSION, Ordering::Relaxed);
        self.seq_req.store(0, Ordering::Relaxed);
        self.seq_resp.store(0, Ordering::Relaxed);
        self.width.store(0, Ordering::Relaxed);
        self.height.store(0, Ordering::Relaxed);
        self.format.store(1, Ordering::Relaxed);
        self.quit.store(0, Ordering::Relaxed);
        self.heartbeat.store(0, Ordering::Relaxed);
        self.control_seq.store(0, Ordering::Relaxed);
        self.tuning_seq.store(0, Ordering::Relaxed);

        self.enabled.store(1, Ordering::Relaxed);
        self.passes.store(1, Ordering::Relaxed);
        self.unlock_passes.store(0, Ordering::Relaxed);
        self.preset.store(0, Ordering::Relaxed);
        self.style.store(0, Ordering::Relaxed);
        self.auto_mask.store(1, Ordering::Relaxed);
        self.intensity_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.local_tone_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.local_structure_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.skin_structure_bits.store((-1.0f32).to_bits(), Ordering::Relaxed);
        self.sharpness_bits.store(0.0f32.to_bits(), Ordering::Relaxed);

        self.transfer_strength_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.colour_strength_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.max_ratio_bits.store(2.0f32.to_bits(), Ordering::Relaxed);
        self.transfer.store(1, Ordering::Relaxed);
        self.debug_view.store(0, Ordering::Relaxed);
        self.debug_scale_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.white_point_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.white_point_scale_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.white_point_source.store(crate::enums::white_point_source::MANUAL, Ordering::Relaxed);
        self.white_point_trim_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.working_scale_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.compare_mode.store(0, Ordering::Relaxed);
        self.compare_split_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        self.compare_zoom_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.compare_swap.store(0, Ordering::Relaxed);
        self.colour_mode.store(crate::enums::colour_mode::AUTO, Ordering::Relaxed);
        self.capture_request.store(0, Ordering::Relaxed);
        self.toggle_key.store(0, Ordering::Relaxed);
        self.reversible_mode.store(crate::enums::reversible_mode::KNEE, Ordering::Relaxed);
        self.apply_model.store(1, Ordering::Relaxed);
        self.hold_frame.store(0, Ordering::Relaxed);
        self.scaling_downscaler.store(crate::enums::downscaler::LANCZOS3, Ordering::Relaxed);

        self.helper_state.store(crate::enums::helper_state::STOPPED, Ordering::Relaxed);
        self.model_up.store(0, Ordering::Relaxed);
        self.helper_frames_lo.store(0, Ordering::Relaxed);
        self.helper_frames_hi.store(0, Ordering::Relaxed);
        self.helper_eval_ms_bits.store(0, Ordering::Relaxed);
        self.helper_upload_ms_bits.store(0, Ordering::Relaxed);
        self.helper_readback_ms_bits.store(0, Ordering::Relaxed);
        self.helper_vram_mb.store(0, Ordering::Relaxed);
        self.helper_features.store(0, Ordering::Relaxed);
        self.helper_pass_ceiling.store(DEFAULT_MAX_PASSES, Ordering::Relaxed);

        self.layer_attached.store(0, Ordering::Relaxed);
        self.layer_frames_lo.store(0, Ordering::Relaxed);
        self.layer_frames_hi.store(0, Ordering::Relaxed);
        self.layer_width.store(0, Ordering::Relaxed);
        self.layer_height.store(0, Ordering::Relaxed);
        self.layer_format.store(0, Ordering::Relaxed);
        self.layer_composition_up.store(0, Ordering::Relaxed);
        self.layer_ms_bits.store(0, Ordering::Relaxed);
        self.layer_measured_white_bits.store(0, Ordering::Relaxed);
        self.layer_heartbeat.store(0, Ordering::Relaxed);

        self.set_helper_reason("");
        self.set_layer_reason("");
        self.set_game_name("");

        for p in &self.pass {
            p.reset_to_defaults();
        }

        self.mvec_enabled.store(1, Ordering::Relaxed);
        self.mvec_scale_mode.store(mvec_scale_mode::PIXELS, Ordering::Relaxed);
        self.mvec_quality.store(mvec_quality::BALANCED, Ordering::Relaxed);
        self.seq_ok.store(0, Ordering::Relaxed);
        // Neural rendering should affect the presented image by default. A
        // bypassed composition is an explicit debug choice, not the normal mode.
        self.composition_bypass.store(0, Ordering::Relaxed);
        self.rebuild_settle_ms.store(250, Ordering::Relaxed);
        self.answered_w.store(0, Ordering::Relaxed);
        self.answered_h.store(0, Ordering::Relaxed);

        self.proxy_export_seq.store(0, Ordering::Relaxed);
        self.proxy_pid.store(0, Ordering::Relaxed);
        self.proxy_fd.store(0, Ordering::Relaxed);
        self.proxy_gen.store(0, Ordering::Relaxed);
        self.answer_export_seq.store(0, Ordering::Relaxed);
        self.answer_pid.store(0, Ordering::Relaxed);
        self.answer_fd.store(0, Ordering::Relaxed);
        self.answer_gen.store(0, Ordering::Relaxed);
        self.layer_proxy_seq.store(0, Ordering::Relaxed);
        self.layer_answer_seq.store(0, Ordering::Relaxed);

        self.hdr_mode.store(crate::enums::hdr_mode::AUTO, Ordering::Relaxed);
        self.hdr_detected.store(crate::enums::hdr_kind::NONE, Ordering::Relaxed);
        self.hdr_active.store(0, Ordering::Relaxed);
        self.proxy_format.store(crate::enums::proxy_format::RGBA8, Ordering::Relaxed);
        self.hdr_encode.store(0, Ordering::Relaxed);
        self.frame_mvec_valid.store(0, Ordering::Relaxed);
        self.frame_mvec_scale_mode.store(mvec_scale_mode::PIXELS, Ordering::Relaxed);
    }

    /// Whether this mapping is one of ours and laid out the way this build expects.
    pub fn is_valid(&self) -> bool {
        self.magic.load(Ordering::Relaxed) == SHM_MAGIC && self.version.load(Ordering::Relaxed) == SHM_VERSION
    }

    /// Every setting a user can change from the GUI, as `("name", current bits)`
    /// pairs -- what [`crate::persist::snapshot`]/[`crate::persist::apply`] round-trip
    /// through `config.ini` so tuning survives a reboot (the SHM mapping itself lives
    /// under `/tmp` and does not). Add here, not just to the GUI, whenever a new
    /// tunable needs to survive a restart -- this is the one list that decides it.
    pub fn persisted_settings(&self) -> [(&'static str, bool, u32); 36] {
        [
            ("white_point", true, self.white_point_bits.load(Ordering::Relaxed)),
            ("white_point_scale", true, self.white_point_scale_bits.load(Ordering::Relaxed)),
            ("white_point_trim", true, self.white_point_trim_bits.load(Ordering::Relaxed)),
            ("white_point_source", false, self.white_point_source.load(Ordering::Relaxed)),
            ("toggle_key", false, self.toggle_key.load(Ordering::Relaxed)),
            ("enabled", false, self.enabled.load(Ordering::Relaxed)),
            ("style", false, self.style.load(Ordering::Relaxed)),
            ("preset", false, self.preset.load(Ordering::Relaxed)),
            ("intensity", true, self.intensity_bits.load(Ordering::Relaxed)),
            ("local_tone", true, self.local_tone_bits.load(Ordering::Relaxed)),
            ("local_structure", true, self.local_structure_bits.load(Ordering::Relaxed)),
            ("skin_structure", true, self.skin_structure_bits.load(Ordering::Relaxed)),
            ("sharpness", true, self.sharpness_bits.load(Ordering::Relaxed)),
            ("auto_mask", false, self.auto_mask.load(Ordering::Relaxed)),
            ("passes", false, self.passes.load(Ordering::Relaxed)),
            ("mvec_enabled", false, self.mvec_enabled.load(Ordering::Relaxed)),
            ("mvec_scale_mode", false, self.mvec_scale_mode.load(Ordering::Relaxed)),
            ("mvec_quality", false, self.mvec_quality.load(Ordering::Relaxed)),
            ("composition_bypass", false, self.composition_bypass.load(Ordering::Relaxed)),
            ("transfer_strength", true, self.transfer_strength_bits.load(Ordering::Relaxed)),
            ("colour_strength", true, self.colour_strength_bits.load(Ordering::Relaxed)),
            ("max_ratio", true, self.max_ratio_bits.load(Ordering::Relaxed)),
            ("working_scale", true, self.working_scale_bits.load(Ordering::Relaxed)),
            ("scaling_downscaler", false, self.scaling_downscaler.load(Ordering::Relaxed)),
            ("reversible_mode", false, self.reversible_mode.load(Ordering::Relaxed)),
            ("hdr_mode", false, self.hdr_mode.load(Ordering::Relaxed)),
            // Added 2026-09-10 alongside the GUI rows that expose them -- see this
            // function's own doc comment on why both have to change together.
            ("transfer", false, self.transfer.load(Ordering::Relaxed)),
            ("compare_mode", false, self.compare_mode.load(Ordering::Relaxed)),
            ("compare_split", true, self.compare_split_bits.load(Ordering::Relaxed)),
            ("compare_zoom", true, self.compare_zoom_bits.load(Ordering::Relaxed)),
            ("compare_swap", false, self.compare_swap.load(Ordering::Relaxed)),
            ("colour_mode", false, self.colour_mode.load(Ordering::Relaxed)),
            ("hold_frame", false, self.hold_frame.load(Ordering::Relaxed)),
            ("unlock_passes", false, self.unlock_passes.load(Ordering::Relaxed)),
            ("apply_model", false, self.apply_model.load(Ordering::Relaxed)),
            ("debug_view", false, self.debug_view.load(Ordering::Relaxed)),
        ]
    }

    /// Stores one persisted setting back by name (as looked up in a `config.ini`
    /// `set_<name>=<value>` line) -- `bits` is already the right representation
    /// (`f32::to_bits()` for the float-valued ones, per [`Self::persisted_settings`]'s
    /// second field).
    pub fn apply_persisted_setting(&self, name: &str, bits: u32) {
        let field = match name {
            "white_point" => &self.white_point_bits,
            "white_point_scale" => &self.white_point_scale_bits,
            "white_point_trim" => &self.white_point_trim_bits,
            "white_point_source" => &self.white_point_source,
            "toggle_key" => &self.toggle_key,
            "enabled" => &self.enabled,
            "style" => &self.style,
            "preset" => &self.preset,
            "intensity" => &self.intensity_bits,
            "local_tone" => &self.local_tone_bits,
            "local_structure" => &self.local_structure_bits,
            "skin_structure" => &self.skin_structure_bits,
            "sharpness" => &self.sharpness_bits,
            "auto_mask" => &self.auto_mask,
            "passes" => &self.passes,
            "mvec_enabled" => &self.mvec_enabled,
            "mvec_scale_mode" => &self.mvec_scale_mode,
            "mvec_quality" => &self.mvec_quality,
            "composition_bypass" => &self.composition_bypass,
            "transfer_strength" => &self.transfer_strength_bits,
            "colour_strength" => &self.colour_strength_bits,
            "max_ratio" => &self.max_ratio_bits,
            "working_scale" => &self.working_scale_bits,
            "scaling_downscaler" => &self.scaling_downscaler,
            "reversible_mode" => &self.reversible_mode,
            "hdr_mode" => &self.hdr_mode,
            "transfer" => &self.transfer,
            "compare_mode" => &self.compare_mode,
            "compare_split" => &self.compare_split_bits,
            "compare_zoom" => &self.compare_zoom_bits,
            "compare_swap" => &self.compare_swap,
            "colour_mode" => &self.colour_mode,
            "hold_frame" => &self.hold_frame,
            "unlock_passes" => &self.unlock_passes,
            "apply_model" => &self.apply_model,
            "debug_view" => &self.debug_view,
            _ => return,
        };
        field.store(bits, Ordering::Relaxed);
    }

    pub fn pass_ceiling(&self) -> u32 {
        if self.unlock_passes.load(Ordering::Relaxed) != 0 {
            MAX_PASSES as u32
        } else {
            DEFAULT_MAX_PASSES
        }
    }

    pub fn resolved_passes(&self) -> u32 {
        let p = self.passes.load(Ordering::Relaxed);
        let ceiling = self.pass_ceiling();
        if p == 0 {
            1
        } else {
            p.min(ceiling)
        }
    }

    pub fn neural_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) != 0
    }

    pub fn mvec_enabled(&self) -> bool {
        self.mvec_enabled.load(Ordering::Relaxed) != 0
    }

    pub fn mvec_scale_mode(&self) -> u32 {
        let m = self.mvec_scale_mode.load(Ordering::Relaxed);
        if m <= mvec_scale_mode::UV01 {
            m
        } else {
            mvec_scale_mode::NORMALIZED
        }
    }

    pub fn mvec_quality(&self) -> u32 {
        let q = self.mvec_quality.load(Ordering::Relaxed);
        if q <= mvec_quality::QUALITY {
            q
        } else {
            mvec_quality::BALANCED
        }
    }

    /// The global settings with one pass's overrides applied. A field the pass does not
    /// name follows the global value, which is what keeps a sparse override sparse.
    pub fn resolve_pass(&self, pass: usize) -> PassTuning {
        let mut t = PassTuning {
            intensity: f32::from_bits(self.intensity_bits.load(Ordering::Relaxed)),
            local_tone: f32::from_bits(self.local_tone_bits.load(Ordering::Relaxed)),
            local_structure: f32::from_bits(self.local_structure_bits.load(Ordering::Relaxed)),
            skin_structure: f32::from_bits(self.skin_structure_bits.load(Ordering::Relaxed)),
            sharpness: f32::from_bits(self.sharpness_bits.load(Ordering::Relaxed)),
            style: self.style.load(Ordering::Relaxed),
            preset: self.preset.load(Ordering::Relaxed),
            auto_mask: self.auto_mask.load(Ordering::Relaxed),
        };

        let Some(p) = self.pass.get(pass) else { return t };
        let mask = p.override_mask.load(Ordering::Relaxed);
        if mask == 0 {
            return t;
        }
        use crate::enums::pass_override::*;
        if mask & INTENSITY != 0 {
            t.intensity = f32::from_bits(p.intensity_bits.load(Ordering::Relaxed));
        }
        if mask & LOCAL_TONE != 0 {
            t.local_tone = f32::from_bits(p.local_tone_bits.load(Ordering::Relaxed));
        }
        if mask & LOCAL_STRUCTURE != 0 {
            t.local_structure = f32::from_bits(p.local_structure_bits.load(Ordering::Relaxed));
        }
        if mask & SKIN_STRUCTURE != 0 {
            t.skin_structure = f32::from_bits(p.skin_structure_bits.load(Ordering::Relaxed));
        }
        if mask & SHARPNESS != 0 {
            t.sharpness = f32::from_bits(p.sharpness_bits.load(Ordering::Relaxed));
        }
        if mask & STYLE != 0 {
            t.style = p.style.load(Ordering::Relaxed);
        }
        if mask & PRESET != 0 {
            t.preset = p.preset.load(Ordering::Relaxed);
        }
        if mask & AUTO_MASK != 0 {
            t.auto_mask = p.auto_mask.load(Ordering::Relaxed);
        }
        t
    }

    pub fn set_helper_reason(&self, s: &str) {
        store_seq_guarded(&self.helper_reason_seq, &self.helper_reason, s);
    }
    pub fn helper_reason(&self) -> String {
        load_seq_guarded(&self.helper_reason_seq, &self.helper_reason)
    }
    pub fn set_layer_reason(&self, s: &str) {
        store_seq_guarded(&self.layer_reason_seq, &self.layer_reason, s);
    }
    pub fn layer_reason(&self) -> String {
        load_seq_guarded(&self.layer_reason_seq, &self.layer_reason)
    }
    pub fn set_game_name(&self, s: &str) {
        store_seq_guarded(&self.game_name_seq, &self.game_name, s);
    }
    pub fn game_name(&self) -> String {
        load_seq_guarded(&self.game_name_seq, &self.game_name)
    }
}

/// Reads a lo/hi `AtomicU32` pair as one `u64` frame counter.
pub fn load64(lo: &AtomicU32, hi: &AtomicU32) -> u64 {
    (u64::from(hi.load(Ordering::Relaxed)) << 32) | u64::from(lo.load(Ordering::Relaxed))
}

pub fn store64(lo: &AtomicU32, hi: &AtomicU32, v: u64) {
    hi.store((v >> 32) as u32, Ordering::Relaxed);
    lo.store(v as u32, Ordering::Relaxed);
}

/// Writes a fixed-size text field guarded by a sequence number: the field is zeroed,
/// the bytes are copied in (truncated to fit, always NUL-terminated), then the sequence
/// is bumped — a reader that observes an unchanged sequence before and after its own
/// read saw one consistent string, never half of an old one and half of a new one.
fn store_seq_guarded<const N: usize>(seq: &AtomicU32, cell: &UnsafeCell<[u8; N]>, s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(N - 1);
    // SAFETY: this is the only place that writes through `cell`, and every reader goes
    // through `load_seq_guarded`, which never dereferences the cell's contents as a str
    // without having first confirmed (via the sequence number) that no write is/was in
    // progress across its read.
    unsafe {
        let buf = &mut *cell.get();
        buf.fill(0);
        buf[..n].copy_from_slice(&bytes[..n]);
    }
    seq.fetch_add(1, Ordering::Release);
}

fn load_seq_guarded<const N: usize>(seq: &AtomicU32, cell: &UnsafeCell<[u8; N]>) -> String {
    for _ in 0..4 {
        let before = seq.load(Ordering::Acquire);
        // SAFETY: see `store_seq_guarded`. The bytes read here are only trusted below,
        // after confirming the sequence number did not change across the read.
        let snapshot = unsafe { *cell.get() };
        let after = seq.load(Ordering::Acquire);
        if before == after {
            let end = snapshot.iter().position(|&b| b == 0).unwrap_or(N);
            return String::from_utf8_lossy(&snapshot[..end]).into_owned();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::pass_override;

    #[test]
    fn default_is_all_zero() {
        let h = ShmHeader::default();
        assert_eq!(h.magic.load(Ordering::Relaxed), 0);
        assert_eq!(h.enabled.load(Ordering::Relaxed), 0);
        assert_eq!(h.helper_reason(), "");
    }

    #[test]
    fn init_defaults_sets_real_values() {
        let h = ShmHeader::default();
        h.init_defaults();
        assert!(h.is_valid());
        assert!(h.neural_enabled());
        assert_eq!(h.resolved_passes(), 1);
        assert_eq!(f32::from_bits(h.intensity_bits.load(Ordering::Relaxed)), 1.0);
        // Bypass is on until the user turns composition on.
        assert_eq!(h.composition_bypass.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn seq_guarded_string_round_trips() {
        let h = ShmHeader::default();
        h.init_defaults();
        h.set_helper_reason("no NGX binaries");
        assert_eq!(h.helper_reason(), "no NGX binaries");
        // A field longer than its buffer is truncated, not rejected.
        let long = "x".repeat(REASON_BYTES + 50);
        h.set_helper_reason(&long);
        assert_eq!(h.helper_reason().len(), REASON_BYTES - 1);
    }

    #[test]
    fn resolve_pass_follows_global_when_no_override() {
        let h = ShmHeader::default();
        h.init_defaults();
        h.intensity_bits.store(0.75f32.to_bits(), Ordering::Relaxed);
        let t = h.resolve_pass(0);
        assert_eq!(t.intensity, 0.75);
        assert_eq!(t.style, 0);
    }

    #[test]
    fn resolve_pass_applies_named_overrides_only() {
        let h = ShmHeader::default();
        h.init_defaults();
        h.intensity_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        h.style.store(0, Ordering::Relaxed);

        let p = &h.pass[3];
        p.intensity_bits.store(0.4f32.to_bits(), Ordering::Relaxed);
        p.style.store(2, Ordering::Relaxed);
        // Only intensity is named as overridden; style must still follow the global
        // value even though the pass's own `style` field holds something else.
        p.override_mask.store(pass_override::INTENSITY, Ordering::Relaxed);

        let t = h.resolve_pass(3);
        assert_eq!(t.intensity, 0.4);
        assert_eq!(t.style, 0);
    }

    #[test]
    fn mvec_scale_mode_clamps_out_of_range() {
        let h = ShmHeader::default();
        h.init_defaults();
        h.mvec_scale_mode.store(99, Ordering::Relaxed);
        assert_eq!(h.mvec_scale_mode(), crate::enums::mvec_scale_mode::NORMALIZED);
    }

    #[test]
    fn load_store_64_round_trips() {
        let lo = AtomicU32::new(0);
        let hi = AtomicU32::new(0);
        store64(&lo, &hi, 0x0001_0203_0405_0607);
        assert_eq!(load64(&lo, &hi), 0x0001_0203_0405_0607);
    }
}
