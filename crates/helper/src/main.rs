//! `dlssnr_helper.exe` — Windows-side NGX service.
//!
//! Owns its own Vulkan device and the `nvngx_dlssnr.dll` model, waits on the
//! shared-memory frame queue the layer writes to, runs the neural pass, and returns
//! processed frames. Built for `x86_64-pc-windows-gnu` and run under Wine/Proton —
//! see `CLAUDE.md` for what has and hasn't actually been verified on this dev machine.
//!
//! Milestone 3 scope (see the plan): everything up through `NgxSnippet::load_and_init`
//! — the caller-identity spoof, the SEH guard, `CreateFeature(18)` — runs for real.
//! There is no per-frame evaluate loop yet (`EvaluateFeature` needs bound Vulkan image
//! resources, which arrive with milestone 4's composition pass); this binary loads,
//! initializes, logs its outcome, and idles ticking a heartbeat, matching how upstream
//! itself behaves before a frame ever arrives.
//!
//! This is a thin wrapper around the `dlssnr_helper` library crate (see `lib.rs`) --
//! that split exists so `examples/` can exercise individual modules directly.

use std::sync::atomic::Ordering;

use ash::vk;
use dlssnr_helper::{guard, ngx, shm};

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

    let Some((entry, instance, physical_device, device)) = create_vulkan_context() else {
        dlssnr_helper::log!("[helper] failed to create a Vulkan context");
        hdr.helper_state.store(dlssnr_protocol::enums::helper_state::NO_VULKAN, Ordering::Relaxed);
        // SAFETY: nothing else references `shm` after this; it owns its own handles.
        unsafe { shm.close() };
        return;
    };

    let snippet = ngx::load_and_init(instance.handle(), physical_device, device.handle());
    hdr.helper_state.store(
        if snippet.disabled {
            dlssnr_protocol::enums::helper_state::MODEL_FAILED
        } else {
            dlssnr_protocol::enums::helper_state::RUNNING
        },
        Ordering::Relaxed,
    );
    dlssnr_helper::log!("[helper] NGX snippet disabled={}", snippet.disabled);

    // No per-frame loop yet (see the module doc comment) -- idle, ticking the
    // heartbeat so the layer's liveness check (`ShmClient::should_retry` on the Linux
    // side) has something to observe, exactly like a real, otherwise-idle helper.
    loop {
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            break;
        }
        hdr.heartbeat.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(500));
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
/// `VkInstance`/`VkPhysicalDevice`/`VkDevice`. Real per-frame resources (the imported
/// dma-buf/staging buffers, the optical-flow images) are milestone 4's job.
fn create_vulkan_context() -> Option<(ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device)> {
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

    let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
    let device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);
    // SAFETY: `device_create_info` is valid; queue family 0 exists on every physical
    // device (the Vulkan spec guarantees at least one queue family).
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.ok()?;

    Some((entry, instance, physical_device, device))
}
