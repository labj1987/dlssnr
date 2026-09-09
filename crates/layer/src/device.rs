//! Per-device state and the [`DeviceHooks`] implementation: `vkCreateSwapchainKHR`/
//! `vkDestroySwapchainKHR` track swapchains, `vkQueuePresentKHR` is where the
//! shared-memory round trip happens.
//!
//! Milestone 2 scope (see `/home/alex/.claude/plans/breezy-napping-waffle.md`): the
//! round trip runs for real (exercising `crate::shm::ShmClient`'s state machine against
//! a real or stub helper), but nothing here touches swapchain image contents yet --
//! `queue_present_khr` always calls through with the original, unmodified present info.
//! Actually capturing/composing frames needs its own GPU resources (command pool,
//! fences, staging/imported buffers) that milestone 4 (the composition pass) adds here;
//! `get_device_queue`/`get_device_queue2` (remembering which queue family to build that
//! command pool on) and the device-extension injection for the dma-buf transport are
//! deferred to that same milestone, since neither means anything before there's a real
//! GPU pass to serve.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;
use vulkan_layer::{DeviceHooks, DeviceInfo, LayerResult, LayerVulkanCommand as VulkanCommand};

use crate::shm::ShmClient;
use crate::swapchain::{self, SwapchainState};

/// The one swapchain (across every device in this process) allowed to drive the
/// shared-memory channel. A process can present more than one swapchain -- the game
/// window and the Steam overlay, or, mid-resize, the old and new windows at once.
/// Routing all of them through one channel would make the helper rebuild its feature on
/// every size switch, and could hand one swapchain another's answer; the largest by
/// area is assumed to be the game, and the rest present untouched.
struct Primary {
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    area: u64,
}

static PRIMARY: Mutex<Option<Primary>> = Mutex::new(None);

fn claim_primary(device: vk::Device, swapchain: vk::SwapchainKHR, width: u32, height: u32) -> bool {
    let mut guard = PRIMARY.lock().unwrap();
    let area = u64::from(width) * u64::from(height);
    match &*guard {
        Some(p) if p.device == device && p.swapchain == swapchain => true,
        Some(p) if area <= p.area => false,
        _ => {
            *guard = Some(Primary { device, swapchain, area });
            true
        }
    }
}

fn release_primary(device: vk::Device, swapchain: vk::SwapchainKHR) {
    let mut guard = PRIMARY.lock().unwrap();
    if matches!(&*guard, Some(p) if p.device == device && p.swapchain == swapchain) {
        *guard = None;
    }
}

/// Resolves one function pointer through the next layer/driver's `vkGetDeviceProcAddr`,
/// or `None` if it isn't there. That's not a failure worth panicking over: a device
/// that never enabled `VK_KHR_swapchain` (a compute-only device, or any device an app
/// simply never presents from) legitimately has no `vkCreateSwapchainKHR` to resolve,
/// and such a device will also never have an app call it -- so a missing pointer here
/// just means this device's hooks quietly do nothing, not that anything is wrong. This
/// was caught by `examples/smoke.rs` creating a device with no extensions enabled at
/// all: the first version of this function panicked on exactly that, which would have
/// crashed every plain compute app the layer got loaded into.
///
/// Transmuting the result to `F` is sound exactly as far as the caller names the right
/// `PFN_vk*` type for `name` -- the same contract the equivalent C cast upstream's own
/// `next_dpa` calls carry.
///
/// # Safety
/// `get_proc` must be a valid `vkGetDeviceProcAddr` for `device`, and `F` must be the
/// PFN type matching `name`.
unsafe fn resolve<F: Copy>(get_proc: vk::PFN_vkGetDeviceProcAddr, device: vk::Device, name: &CStr) -> Option<F> {
    let p = unsafe { get_proc(device, name.as_ptr()) }?;
    // SAFETY: forwarded from the caller's own safety contract.
    Some(unsafe { std::mem::transmute_copy::<_, F>(&p) })
}

pub struct DlssnrDeviceInfo {
    #[allow(dead_code)] // kept alive so `next_*` fn pointers stay valid; not yet called through directly
    device: Arc<ash::Device>,
    next_create_swapchain_khr: Option<vk::PFN_vkCreateSwapchainKHR>,
    next_destroy_swapchain_khr: Option<vk::PFN_vkDestroySwapchainKHR>,
    next_queue_present_khr: Option<vk::PFN_vkQueuePresentKHR>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    swapchains: HashMap<vk::SwapchainKHR, SwapchainState>,
    shm: ShmClient,
}

impl DlssnrDeviceInfo {
    pub fn new(
        device: Arc<ash::Device>,
        next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    ) -> Self {
        let handle = device.handle();
        // SAFETY: `next_get_device_proc_addr` is the next layer/driver's own
        // `vkGetDeviceProcAddr`, handed to us by the layer framework for exactly this
        // device; each name below matches the `PFN_vk*` type requested.
        //
        // `vkGetSwapchainImagesKHR` isn't resolved here (yet): nothing reads the image
        // list until milestone 4 needs to know which swapchain image `vkQueuePresentKHR`
        // is about to present, to capture from it.
        let (create, destroy, present) = unsafe {
            (
                resolve::<vk::PFN_vkCreateSwapchainKHR>(
                    next_get_device_proc_addr,
                    handle,
                    c"vkCreateSwapchainKHR",
                ),
                resolve::<vk::PFN_vkDestroySwapchainKHR>(
                    next_get_device_proc_addr,
                    handle,
                    c"vkDestroySwapchainKHR",
                ),
                resolve::<vk::PFN_vkQueuePresentKHR>(
                    next_get_device_proc_addr,
                    handle,
                    c"vkQueuePresentKHR",
                ),
            )
        };
        crate::log!(
            "[layer] hooked device {:?} (swapchain support: {})",
            handle,
            create.is_some() && destroy.is_some() && present.is_some()
        );
        Self {
            device,
            next_create_swapchain_khr: create,
            next_destroy_swapchain_khr: destroy,
            next_queue_present_khr: present,
            state: Mutex::default(),
        }
    }
}

impl DeviceInfo for DlssnrDeviceInfo {
    type HooksType = Self;
    type HooksRefType<'a> = &'a Self;

    fn hooked_commands() -> &'static [VulkanCommand] {
        &[
            VulkanCommand::CreateSwapchainKhr,
            VulkanCommand::DestroySwapchainKhr,
            VulkanCommand::QueuePresentKhr,
        ]
    }

    fn hooks(&self) -> Self::HooksRefType<'_> {
        self
    }
}

impl DeviceHooks for DlssnrDeviceInfo {
    fn create_swapchain_khr(
        &self,
        create_info: &vk::SwapchainCreateInfoKHR,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> LayerResult<ash::prelude::VkResult<vk::SwapchainKHR>> {
        // No `VK_KHR_swapchain` on this device -- see `resolve()`'s doc comment. An app
        // that enabled the extension would never let this be `None`; let the framework's
        // own next-in-chain dispatch handle it exactly as if we weren't here.
        let Some(next_create) = self.next_create_swapchain_khr else {
            return LayerResult::Unhandled;
        };
        let mut swapchain = vk::SwapchainKHR::null();
        let alloc_ptr = allocator.map_or(std::ptr::null(), std::ptr::from_ref);
        // SAFETY: `create_info`/`allocator` are valid for the duration of this call
        // (handed to us by the loader for exactly this call, per the Vulkan spec);
        // `next_create` was resolved from the next layer/driver's own proc-addr table
        // in `new()` above.
        let result = unsafe { next_create(self.device.handle(), create_info, alloc_ptr, &mut swapchain) };
        if result != vk::Result::SUCCESS {
            return LayerResult::Handled(Err(result));
        }

        let hdr_kind = swapchain::detect_hdr_kind(create_info.image_format, create_info.image_color_space);
        let pass_through = !swapchain::is_supported_format(create_info.image_format)
            || create_info.image_extent.width > dlssnr_protocol::MAX_W
            || create_info.image_extent.height > dlssnr_protocol::MAX_H;
        let state = SwapchainState {
            format: create_info.image_format,
            width: create_info.image_extent.width,
            height: create_info.image_extent.height,
            hdr_kind,
            pass_through,
        };
        crate::log!(
            "[layer] swapchain {:?} {}x{} fmt={:?} hdr={} pass_through={}",
            swapchain,
            state.width,
            state.height,
            state.format,
            state.hdr_kind,
            state.pass_through
        );
        self.state.lock().unwrap().swapchains.insert(swapchain, state);
        LayerResult::Handled(Ok(swapchain))
    }

    fn destroy_swapchain_khr(
        &self,
        swapchain: vk::SwapchainKHR,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> LayerResult<()> {
        let Some(next_destroy) = self.next_destroy_swapchain_khr else {
            return LayerResult::Unhandled;
        };
        release_primary(self.device.handle(), swapchain);
        self.state.lock().unwrap().swapchains.remove(&swapchain);
        let alloc_ptr = allocator.map_or(std::ptr::null(), std::ptr::from_ref);
        // SAFETY: same contract as `create_swapchain_khr` above.
        unsafe { next_destroy(self.device.handle(), swapchain, alloc_ptr) };
        LayerResult::Handled(())
    }

    fn queue_present_khr(
        &self,
        queue: vk::Queue,
        present_info: &vk::PresentInfoKHR,
    ) -> LayerResult<ash::prelude::VkResult<()>> {
        let Some(next_present) = self.next_queue_present_khr else {
            return LayerResult::Unhandled;
        };
        if crate::layer_enabled() {
            // SAFETY: `p_swapchains`/`swapchain_count` are a valid slice for the
            // duration of this call -- part of the `VkPresentInfoKHR` the loader just
            // handed us.
            let swapchains = unsafe {
                std::slice::from_raw_parts(present_info.p_swapchains, present_info.swapchain_count as usize)
            };
            let mut state = self.state.lock().unwrap();
            for &sc in swapchains {
                let Some(sw) = state.swapchains.get(&sc) else { continue };
                if sw.pass_through {
                    continue;
                }
                if claim_primary(self.device.handle(), sc, sw.width, sw.height) {
                    // Exercises the SHM state machine end to end -- the fail-open
                    // budget, the dead/retry timer -- without touching any pixels yet.
                    // See the module doc comment: real capture/compose is milestone 4.
                    state.shm.try_round_trip();
                }
                break;
            }
        }

        // SAFETY: `present_info` is valid for the duration of this call; `next_present`
        // was resolved from the next layer/driver's own proc-addr table.
        let result = unsafe { next_present(queue, present_info) };
        LayerResult::Handled(result.result())
    }
}
