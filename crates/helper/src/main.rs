//! `dlssnr_helper.exe` — Windows-side NGX service.
//!
//! Owns its own Vulkan device and the `nvngx_dlssnr.dll` model, waits on the
//! shared-memory frame queue the layer writes to, runs the neural pass, and returns
//! processed frames. Built for `x86_64-pc-windows-gnu` and run under Wine/Proton —
//! see `CLAUDE.md` for what has and hasn't actually been verified on this dev machine.
//!
//! Milestone 4: watches `seq_req` for a change, ensures the NGX feature exists at
//! that frame's real size (`ngx::ensure_feature`, deferred from startup since there's
//! no real size before the layer's first capture), runs `EvaluateFeature` through
//! `frame::FrameResources`, and writes the result into the answer region -- or, if the
//! feature isn't ready (still building, or the guarded `CreateFeature`/
//! `EvaluateFeature` call failed), echoes the proxy bytes straight through so the
//! transport itself still proves out even when the model side doesn't.
//!
//! `frame_resources` is rebuilt whenever the observed width/height changes (e.g. two
//! separate Vulkan processes -- the game and the Steam overlay -- both driving this
//! same helper before `swapchain::is_plausible_game_size` existed to filter the
//! overlay's own swapchain out on the layer side) -- the *old* `FrameResources` is
//! always explicitly destroyed first via `FrameResources::destroy`, since letting the
//! `Option` just get overwritten would leak its images/memory/command pool every time,
//! not free them (`ash` handles are not `Drop`).
//!
//! This is a thin wrapper around the `dlssnr_helper` library crate (see `lib.rs`) --
//! that split exists so `examples/` can exercise individual modules directly.

use std::sync::atomic::Ordering;
use std::time::Duration;

use ash::vk;
use dlssnr_helper::{frame, guard, ngx, shm};

fn main() {
    guard::install();

    let Some(shm) = shm::open() else {
        dlssnr_helper::log!("[helper] failed to open the shared-memory mapping");
        return;
    };
    // SAFETY: `shm.header` was just validated by `shm::open`.
    let hdr = unsafe { &*shm.header };
    hdr.helper_state.store(dlssnr_protocol::enums::helper_state::STARTING, Ordering::Relaxed);
    hdr.control_seq.fetch_add(1, Ordering::Relaxed);
    hdr.heartbeat.fetch_add(1, Ordering::Relaxed);
    dlssnr_helper::log!("[helper] shm attached");
    dlssnr_helper::logging::flush();

    let Some((entry, instance, physical_device, device, queue)) = create_vulkan_context() else {
        dlssnr_helper::log!("[helper] failed to create a Vulkan context");
        dlssnr_helper::logging::flush();
        hdr.helper_state.store(dlssnr_protocol::enums::helper_state::NO_VULKAN, Ordering::Relaxed);
        // SAFETY: nothing else references `shm` after this; it owns its own handles.
        unsafe { shm.close() };
        return;
    };
    dlssnr_helper::log!("[helper] Vulkan context created, loading NGX next");
    dlssnr_helper::logging::flush();

    let mut snippet = ngx::load_and_init(instance.handle(), physical_device, device.handle());
    hdr.helper_state.store(
        if snippet.disabled {
            dlssnr_protocol::enums::helper_state::MODEL_FAILED
        } else {
            dlssnr_protocol::enums::helper_state::RUNNING
        },
        Ordering::Relaxed,
    );
    dlssnr_helper::log!("[helper] NGX snippet disabled={}", snippet.disabled);
    dlssnr_helper::logging::flush();

    let mut frame_resources: Option<frame::FrameResources> = None;
    // Resized (not reallocated fresh every frame) to whatever the current frame's
    // real byte count is -- never the full `MAX_FRAME` reservation, which is sized for
    // the protocol's absolute ceiling (7680x4320 float16), not a typical frame.
    let mut proxy_buf: Vec<u8> = Vec::new();
    let mut answer_buf: Vec<u8> = Vec::new();
    let mut last_seq_req = hdr.seq_req.load(Ordering::Relaxed);
    let mut frames: u64 = 0;

    loop {
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            break;
        }
        let seq_req = hdr.seq_req.load(Ordering::Relaxed);
        if seq_req != last_seq_req {
            last_seq_req = seq_req;
            let width = hdr.width.load(Ordering::Relaxed);
            let height = hdr.height.load(Ordering::Relaxed);
            let proxy_format = hdr.proxy_format.load(Ordering::Relaxed);
            let bytes = (dlssnr_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) * (width as usize) * (height as usize))
                .min(dlssnr_protocol::MAX_FRAME);
            proxy_buf.resize(bytes, 0);
            answer_buf.resize(bytes, 0);
            let n = shm.read_proxy(&mut proxy_buf);

            let ready = ngx::ensure_feature(&mut snippet, &device, queue, width, height);
            if ready {
                hdr.model_up.store(1, Ordering::Relaxed);
            } else if snippet.disabled {
                // `ensure_feature` only ever disables the snippet after a real,
                // one-shot `CreateFeature` attempt (see its own doc comment) -- worth
                // surfacing in status immediately rather than leaving `RUNNING`
                // displayed forever after the model is permanently unavailable.
                hdr.helper_state.store(dlssnr_protocol::enums::helper_state::MODEL_FAILED, Ordering::Relaxed);
            }
            let evaluated = ready
                && proxy_format == dlssnr_protocol::enums::proxy_format::RGBA8
                && (|| {
                    if !frame_resources.as_ref().is_some_and(|f| f.matches(0, width, height)) {
                        // SAFETY: any previous resources are no longer referenced by
                        // in-flight work -- `FrameResources::evaluate` always waits on
                        // its own fences before returning, so by the time we're back
                        // here (a later loop iteration) nothing is still submitted.
                        if let Some(old) = frame_resources.take() {
                            unsafe { old.destroy(&device) };
                        }
                        frame_resources = frame::FrameResources::new(&device, &instance, physical_device, 0, width, height);
                    }
                    let f = frame_resources.as_ref()?;
                    let (Some(eval_fn), params) = (snippet.evaluate_feature_fn(), snippet.params()) else { return None };
                    f.evaluate(&device, queue, eval_fn, snippet.feature, params, &proxy_buf[..n], &mut answer_buf[..n]).then_some(())
                })()
                .is_some();
            if !evaluated {
                // Fail open: no real answer yet (feature still warming up, wrong
                // proxy format, or a guarded `EvaluateFeature` failure) -- echo the
                // proxy straight through so the transport round trip still completes
                // with *something* rather than stale/all-zero bytes.
                answer_buf[..n].copy_from_slice(&proxy_buf[..n]);
            }
            shm.write_answer(&answer_buf[..n]);
            hdr.seq_ok.store(seq_req, Ordering::Relaxed);
            hdr.seq_resp.store(seq_req, Ordering::Relaxed);
            frames += 1;
            dlssnr_protocol::store64(&hdr.helper_frames_lo, &hdr.helper_frames_hi, frames);
            dlssnr_helper::log!("[helper] frame {frames}: {width}x{height} evaluated={evaluated}");
        }
        hdr.heartbeat.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_micros(200));
    }

    // SAFETY: process is tearing down; nothing else can still be submitting work
    // against `frame_resources`'s handles.
    if let Some(f) = frame_resources {
        unsafe { f.destroy(&device) };
    }
    ngx::teardown(snippet);
    hdr.helper_state.store(dlssnr_protocol::enums::helper_state::STOPPED, Ordering::Relaxed);
    // SAFETY: destroyed in the reverse order of creation; nothing else holds a
    // reference to `device`/`instance` past this point.
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
        shm.close();
    }
    drop(entry);
}

/// A minimal Vulkan instance + device — just enough to hand NGX a live
/// `VkInstance`/`VkPhysicalDevice`/`VkDevice`, plus the one queue (family 0, index 0 --
/// every device has at least one family, and this is the same family `ngx`'s device
/// was created against) real per-frame work submits against.
/// Extensions a real, working reference implementation's own compiled helper
/// references (confirmed via `strings` on its binary, never its source -- run side
/// by side with ours this session, which is how the actual gap this closes was
/// found: our device previously enabled *none* of these). Requested only when
/// actually present in `vkEnumerateDeviceExtensionProperties` -- never assumed.
/// `VK_NVX_binary_import`/`VK_NVX_image_view_handle`/`VK_KHR_buffer_device_address`
/// in particular are the kind of NVIDIA-internal plumbing NGX's own shader/resource
/// management is plausibly built on; calling into it from a device that never
/// enabled them is the leading suspect for why `CreateFeature` behaved inconsistently
/// (a clean reject one run, an actual C++ exception the next) even after every
/// parameter this session could confirm from the same binary was added.
///
/// **2026-09-10 correction**: the reference binary `strings` was originally run
/// against was upstream's *native Linux* helper (per this project's own architecture,
/// upstream's own eventual roadmap target -- see the crate-level doc comment) --
/// `VK_EXT_external_memory_dma_buf`/`VK_KHR_external_memory_fd` are POSIX-specific
/// external-memory handle types that a real Linux Vulkan ICD legitimately exposes and
/// that reference binary legitimately used. `dlssnr_helper.exe` is not that: it is a
/// Windows binary running under Wine/Proton (this crate's current, documented, interim
/// architecture), and Wine's Vulkan implementation for Windows guest apps exposes the
/// Windows-shaped `VK_KHR_external_memory_win32` handle type, never the Linux `_fd`
/// ones -- requesting the Linux-specific pair here could never succeed no matter what
/// the real driver supports, a platform mismatch inherited from copying the reference
/// list without adjusting for which binary actually needed which handles. Found while
/// investigating a real, confirmed-white `EvaluateFeature` answer (see `CLAUDE.md`) --
/// this DLL performs real CUDA-Vulkan interop internally (`cuSurfObjectGetResourceDesc`
/// et al., confirmed via `strings` on the real DLL itself), which is exactly the kind
/// of external-memory-dependent operation a missing win32 handle type would degrade.
const WANTED_DEVICE_EXTENSIONS: &[&str] = &[
    "VK_EXT_debug_utils",
    "VK_EXT_external_memory_host",
    "VK_KHR_buffer_device_address",
    "VK_KHR_external_memory",
    "VK_KHR_external_memory_win32",
    "VK_KHR_push_descriptor",
    "VK_NV_optical_flow",
    "VK_NVX_binary_import",
    "VK_NVX_image_view_handle",
];

fn create_vulkan_context() -> Option<(ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue)> {
    // SAFETY: dynamically loads `vulkan-1.dll` via the `loaded` feature; the usual
    // caveats of loading an arbitrary shared library apply and are accepted here the
    // same way every other `ash` consumer accepts them.
    let entry = unsafe { ash::Entry::load() }.ok()?;

    let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    // SAFETY: `create_info` is a valid, fully-populated `VkInstanceCreateInfo`.
    let instance = unsafe { entry.create_instance(&create_info, None) }.ok()?;

    // SAFETY: `instance` was just created above.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.ok()?;
    let physical_device = *physical_devices.iter().find(|&&pd| {
        // SAFETY: `pd` is one of the handles just enumerated.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        props.vendor_id == 0x10DE // NVIDIA -- the model only ever runs on its own hardware.
    }).or(physical_devices.first())?;

    // SAFETY: `physical_device` is one of the handles just enumerated above.
    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }.unwrap_or_default();
    let available_names: std::collections::HashSet<String> = available
        .iter()
        .filter_map(|e| {
            // SAFETY: `extension_name` is a NUL-terminated C string the driver itself
            // populated; reading it as a CStr is exactly what every other `ash`
            // consumer does with this same field.
            unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }.to_str().ok().map(str::to_owned)
        })
        .collect();
    let enabled: Vec<&str> = WANTED_DEVICE_EXTENSIONS.iter().copied().filter(|e| available_names.contains(*e)).collect();
    dlssnr_helper::log!(
        "[helper] device extensions: {}/{} of the wanted set available: {:?}",
        enabled.len(),
        WANTED_DEVICE_EXTENSIONS.len(),
        enabled
    );
    let enabled_c: Vec<std::ffi::CString> = enabled.iter().map(|e| std::ffi::CString::new(*e).unwrap()).collect();
    let enabled_ptrs: Vec<*const std::ffi::c_char> = enabled_c.iter().map(|c| c.as_ptr()).collect();

    let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
    let device_create_info =
        vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info).enabled_extension_names(&enabled_ptrs);
    // SAFETY: `device_create_info` is valid; queue family 0 exists on every physical
    // device (the Vulkan spec guarantees at least one queue family); `enabled_ptrs`
    // point at only extensions just confirmed present in `available_names`, and
    // `enabled_c` (which owns the bytes they point into) outlives this call.
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.ok()?;
    // SAFETY: `device` was just created with exactly one queue on family 0, index 0.
    let queue = unsafe { device.get_device_queue(0, 0) };

    Some((entry, instance, physical_device, device, queue))
}
