//! Per-device state and the [`DeviceHooks`] implementation: `vkCreateSwapchainKHR`/
//! `vkDestroySwapchainKHR` track swapchains, `vkGetDeviceQueue`/`vkGetDeviceQueue2`
//! learn which queue family a queue belongs to, `vkQueuePresentKHR` is where the real
//! capture/transport/write-back round trip (`crate::capture::run`) happens now.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;
use vulkan_layer::{DeviceHooks, DeviceInfo, LayerResult, LayerVulkanCommand as VulkanCommand};

use crate::capture;
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
    device: Arc<ash::Device>,
    /// `None` only in the hypothetical case `create_device_info`'s own doc comment
    /// notes (a device created against an instance from before this layer loaded,
    /// which never happens for an implicit layer) -- capture is simply skipped
    /// (present passes through unmodified) whenever it is, rather than panicking.
    instance: Option<Arc<ash::Instance>>,
    physical_device: vk::PhysicalDevice,
    next_create_swapchain_khr: Option<vk::PFN_vkCreateSwapchainKHR>,
    next_destroy_swapchain_khr: Option<vk::PFN_vkDestroySwapchainKHR>,
    next_queue_present_khr: Option<vk::PFN_vkQueuePresentKHR>,
    next_get_swapchain_images_khr: Option<vk::PFN_vkGetSwapchainImagesKHR>,
    next_get_device_queue: Option<vk::PFN_vkGetDeviceQueue>,
    next_get_device_queue2: Option<vk::PFN_vkGetDeviceQueue2>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    swapchains: HashMap<vk::SwapchainKHR, SwapchainState>,
    shm: ShmClient,
    /// Which queue family a `VkQueue` handle belongs to -- learned by observing the
    /// app's own `vkGetDeviceQueue`/`vkGetDeviceQueue2` calls (see those hooks below),
    /// since Vulkan has no query that answers this for a handle after the fact. Needed
    /// to build a command pool for whatever queue `queue_present_khr` hands us.
    queue_families: HashMap<vk::Queue, u32>,
    capture: Option<capture::CaptureResources>,
    gpu_compose: Option<crate::composition::gpu::GpuCompose>,
}

impl DlssnrDeviceInfo {
    pub fn new(
        instance: Option<Arc<ash::Instance>>,
        physical_device: vk::PhysicalDevice,
        device: Arc<ash::Device>,
        next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    ) -> Self {
        let handle = device.handle();
        // SAFETY: `next_get_device_proc_addr` is the next layer/driver's own
        // `vkGetDeviceProcAddr`, handed to us by the layer framework for exactly this
        // device; each name below matches the `PFN_vk*` type requested.
        let (create, destroy, present, get_images, get_queue, get_queue2) = unsafe {
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
                resolve::<vk::PFN_vkGetSwapchainImagesKHR>(
                    next_get_device_proc_addr,
                    handle,
                    c"vkGetSwapchainImagesKHR",
                ),
                resolve::<vk::PFN_vkGetDeviceQueue>(next_get_device_proc_addr, handle, c"vkGetDeviceQueue"),
                resolve::<vk::PFN_vkGetDeviceQueue2>(next_get_device_proc_addr, handle, c"vkGetDeviceQueue2"),
            )
        };
        crate::log!(
            "[layer] hooked device {:?} (swapchain support: {})",
            handle,
            create.is_some() && destroy.is_some() && present.is_some()
        );
        Self {
            device,
            instance,
            physical_device,
            next_create_swapchain_khr: create,
            next_destroy_swapchain_khr: destroy,
            next_queue_present_khr: present,
            next_get_swapchain_images_khr: get_images,
            next_get_device_queue: get_queue,
            next_get_device_queue2: get_queue2,
            state: Mutex::default(),
        }
    }

    /// The images backing `swapchain`, in the order the loader hands out indices for
    /// `VkPresentInfoKHR::pImageIndices` -- cached once at creation (see
    /// `create_swapchain_khr`) since the list never changes for a swapchain's lifetime.
    fn fetch_swapchain_images(&self, swapchain: vk::SwapchainKHR) -> Vec<vk::Image> {
        let Some(get_images) = self.next_get_swapchain_images_khr else { return Vec::new() };
        let handle = self.device.handle();
        let mut count = 0u32;
        // SAFETY: `get_images` was resolved from the next layer/driver's own proc-addr
        // table; the two-call enumeration pattern (count, then fill) is exactly what
        // the Vulkan spec requires for this function.
        if unsafe { get_images(handle, swapchain, &mut count, std::ptr::null_mut()) } != vk::Result::SUCCESS {
            return Vec::new();
        }
        let mut images = vec![vk::Image::null(); count as usize];
        // SAFETY: `images` has exactly `count` elements, matching what the first call
        // just reported.
        if unsafe { get_images(handle, swapchain, &mut count, images.as_mut_ptr()) } != vk::Result::SUCCESS {
            return Vec::new();
        }
        images
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
            VulkanCommand::GetDeviceQueue,
            VulkanCommand::GetDeviceQueue2,
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
            || create_info.image_extent.height > dlssnr_protocol::MAX_H
            || !swapchain::is_plausible_game_size(create_info.image_extent.width, create_info.image_extent.height);
        let images = if pass_through { Vec::new() } else { self.fetch_swapchain_images(swapchain) };
        let state = SwapchainState {
            format: create_info.image_format,
            width: create_info.image_extent.width,
            height: create_info.image_extent.height,
            hdr_kind,
            pass_through,
            images,
        };
        crate::log!(
            "[layer] swapchain {:?} {}x{} fmt={:?} hdr={} pass_through={} images={}",
            swapchain,
            state.width,
            state.height,
            state.format,
            state.hdr_kind,
            state.pass_through,
            state.images.len()
        );
        self.state.lock().unwrap().swapchains.insert(swapchain, state);
        LayerResult::Handled(Ok(swapchain))
    }

    fn get_device_queue(&self, queue_family_index: u32, queue_index: u32) -> LayerResult<vk::Queue> {
        let Some(next) = self.next_get_device_queue else { return LayerResult::Unhandled };
        let mut queue = vk::Queue::null();
        // SAFETY: `next` was resolved from the next layer/driver's own proc-addr
        // table; `queue_family_index`/`queue_index` are the caller's own, forwarded
        // unchanged.
        unsafe { next(self.device.handle(), queue_family_index, queue_index, &mut queue) };
        self.state.lock().unwrap().queue_families.insert(queue, queue_family_index);
        LayerResult::Handled(queue)
    }

    fn get_device_queue2(&self, queue_info: &vk::DeviceQueueInfo2) -> LayerResult<vk::Queue> {
        let Some(next) = self.next_get_device_queue2 else { return LayerResult::Unhandled };
        let mut queue = vk::Queue::null();
        // SAFETY: `next` was resolved from the next layer/driver's own proc-addr
        // table; `queue_info` is valid for the duration of this call (handed to us by
        // the loader for exactly this call).
        unsafe { next(self.device.handle(), queue_info, &mut queue) };
        self.state.lock().unwrap().queue_families.insert(queue, queue_info.queue_family_index);
        LayerResult::Handled(queue)
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
            // SAFETY: `p_swapchains`/`p_image_indices`/`swapchain_count` are a valid,
            // parallel pair of slices for the duration of this call -- part of the
            // `VkPresentInfoKHR` the loader just handed us.
            let (swapchains, image_indices) = unsafe {
                (
                    std::slice::from_raw_parts(present_info.p_swapchains, present_info.swapchain_count as usize),
                    std::slice::from_raw_parts(present_info.p_image_indices, present_info.swapchain_count as usize),
                )
            };
            let mut state = self.state.lock().unwrap();
            for (&sc, &image_index) in swapchains.iter().zip(image_indices) {
                let Some(sw) = state.swapchains.get(&sc) else { continue };
                if sw.pass_through {
                    continue;
                }
                if !claim_primary(self.device.handle(), sc, sw.width, sw.height) {
                    break;
                }
                let Some(&image) = sw.images.get(image_index as usize) else { break };
                let Some(&queue_family) = state.queue_families.get(&queue) else {
                    // We've never seen this queue via a hooked `vkGetDeviceQueue`/
                    // `vkGetDeviceQueue2` call (e.g. an app using `VK_KHR_synchronization2`
                    // queue submission paths this layer doesn't intercept) -- no family
                    // to build a command pool on, so fail open rather than guess one.
                    break;
                };
                let width = sw.width;
                let height = sw.height;
                let proxy_format = swapchain::proxy_format_for(sw.format);
                let State { shm, capture, gpu_compose, .. } = &mut *state;
                if shm.model_known_unavailable() {
                    // The helper has permanently disabled itself for this session
                    // (see `ngx::ensure_feature`'s one-shot design) -- nothing will
                    // ever evaluate a captured frame, so paying for the capture
                    // itself (a full image<->buffer round trip plus a whole-frame
                    // `memcpy`, every single present call) is pure waste. Skip
                    // straight to a real no-op present, matching what "fail-open"
                    // should actually cost: nothing.
                    break;
                }
                if let Some(instance) = &self.instance {
                    // SAFETY: `queue` is the same queue this present call was made on,
                    // externally synchronized for its duration by the same Vulkan rule
                    // that lets the caller call `vkQueuePresentKHR` on it at all right
                    // after this returns -- exactly this function's own safety
                    // contract. `image` is one of `sc`'s own images, currently
                    // `PRESENT_SRC_KHR` per `vkQueuePresentKHR`'s precondition on every
                    // image it's about to present.
                    unsafe {
                        capture::run(
                            &self.device,
                            instance,
                            self.physical_device,
                            queue,
                            queue_family,
                            image,
                            width,
                            height,
                            proxy_format,
                            capture,
                            gpu_compose,
                            shm,
                        );
                    }
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
