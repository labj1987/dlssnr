//! `VK_LAYER_dlssnr_neural` — the Linux-side Vulkan implicit layer.
//!
//! Hooks the swapchain lifecycle (`vkCreateSwapchainKHR`/`vkDestroySwapchainKHR`/
//! `vkQueuePresentKHR`) and exchanges frames with the helper over the shared-memory
//! transport defined in `dlssnr_protocol`. This crate never knows or cares whether the
//! helper on the other end of that mapping is running under Wine/Proton today or a
//! native Linux process later — that's the whole point of the seam.
//!
//! Built on Google's [`vulkan_layer`](https://github.com/google/vk-layer-for-rust)
//! crate, which supplies the actual `vkGetInstanceProcAddr`/`vkGetDeviceProcAddr`
//! dispatch machinery, the `VkLayerInstanceCreateInfo`/`VkLayerDeviceCreateInfo`
//! chain-walk, and the loader-negotiation entry points — this crate only implements
//! [`vulkan_layer::DeviceHooks`] for the handful of functions it actually cares about;
//! everything else falls through to the next layer/driver automatically.
//!
//! Milestone 4 status: `queue_present_khr` now really captures the primary swapchain's
//! image into the shared-memory proxy region and writes a result back (`capture.rs`)
//! instead of always presenting the frame unmodified. Still open: the real GPU compute
//! composition (`composition/`'s math is real and tested but not yet dispatched --
//! `capture.rs` currently writes the captured bytes straight back, an identity
//! passthrough through the transport rather than a real edit), the
//! inert-on-non-NVIDIA-device check, `pidfd_getfd` descriptor adoption (an optional
//! future zero-copy optimization, not required for the transport that exists today),
//! hotkey polling. Also open: real cross-process arbitration for which swapchain gets
//! to be "primary" -- today's `device::PRIMARY` only elects one per *process*, so a
//! second Vulkan process (the Steam overlay, observed in testing) independently elects
//! its own small swapchain as primary too; `swapchain::is_plausible_game_size` is a
//! blunt, documented-as-such filter for that, not a real fix.

mod capture;
mod composition;
mod device;
mod logging;
mod shm;
mod swapchain;

use std::ops::Deref;
use std::sync::{Arc, Mutex};

use ash::vk;
use once_cell::sync::Lazy;
use vulkan_layer::{
    auto_globalhooksinfo_impl, declare_introspection_queries, Global, GlobalHooks, Layer, LayerManifest,
    LayerResult, StubInstanceInfo, VkLayerInstanceLink,
};

use device::DlssnrDeviceInfo;

/// The most recently created `VkInstance`, so `create_device_info` (which the
/// `vulkan_layer` framework calls with no way to reach whatever `create_instance_info`
/// returned -- see that method's own doc comment) can still get an `ash::Instance` to
/// query physical-device memory properties from when it builds capture resources.
/// Games overwhelmingly create exactly one `VkInstance`; a plain "last one wins" slot
/// is the same simplification `device::PRIMARY` already makes for the analogous
/// one-swapchain-at-a-time assumption.
static CURRENT_INSTANCE: Mutex<Option<Arc<ash::Instance>>> = Mutex::new(None);

pub const LAYER_NAME: &str = "VK_LAYER_dlssnr_neural";

/// Whether the pass should do anything at all. Off by default (`VKLayer_DLSS5` unset)
/// so the layer is a true no-op for every game that hasn't opted in via its launch
/// options — checked once and cached, same as upstream, since it can't change for the
/// life of the process.
pub(crate) fn layer_enabled() -> bool {
    static ENABLED: Lazy<bool> = Lazy::new(|| {
        env_flag("VKLayer_DLSS5") || env_flag("DLSSNR_ENABLE")
    });
    *ENABLED
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// Works around a crash inside Mesa's `device_select` implicit layer, confirmed
/// reproducible even with `vulkan-layer`'s own pristine `hello-world` example under
/// implicit activation on this machine's Mesa build (see `CLAUDE.md`'s "CRITICAL,
/// confirmed" section for the full bisection) -- so this is a bug in the interaction
/// between the pinned `vulkan-layer` commit and this Mesa build, not anything specific
/// to this crate. `vulkan_layer::Global::create_instance`'s default fallback path
/// (taken whenever `GlobalHooks::create_instance` is `Unhandled`) eagerly resolves all
/// three Vulkan 1.0 global entry points -- `vkCreateInstance`,
/// `vkEnumerateInstanceExtensionProperties`, `vkEnumerateInstanceLayerProperties` --
/// through the chained, `VK_NULL_HANDLE`-instance `vkGetInstanceProcAddr`
/// (`ash::vk::EntryFnV1_0::load`), even though only `vkCreateInstance` is ever actually
/// called afterward. Resolving `vkEnumerateInstanceExtensionProperties` that way
/// segfaults inside `libVkLayer_MESA_device_select.so` 100% of the time (confirmed via
/// `gdb`: `vkCreateInstance` resolves fine through the exact same chained pointer,
/// `vkEnumerateInstanceExtensionProperties` right after it does not). Hooking
/// `create_instance` ourselves and resolving only the one entry point this layer
/// actually needs avoids ever making the query that crashes.
#[derive(Default)]
struct DlssnrGlobalHooks;

#[auto_globalhooksinfo_impl]
impl GlobalHooks for DlssnrGlobalHooks {
    fn create_instance(
        &self,
        create_info: &vk::InstanceCreateInfo,
        layer_instance_link: &VkLayerInstanceLink,
        allocator: Option<&vk::AllocationCallbacks>,
        p_instance: *mut vk::Instance,
    ) -> LayerResult<ash::prelude::VkResult<()>> {
        // SAFETY: `layer_instance_link.pfnNextGetInstanceProcAddr` is the loader- or
        // next-layer-supplied chained `vkGetInstanceProcAddr`, valid for the duration of
        // this call; `VK_NULL_HANDLE` + a global-command name is the spec-mandated way
        // to query it before an instance exists.
        let create_instance = unsafe {
            (layer_instance_link.pfnNextGetInstanceProcAddr)(vk::Instance::null(), c"vkCreateInstance".as_ptr())
        };
        let create_instance: vk::PFN_vkCreateInstance = match create_instance {
            // SAFETY: a non-null `vkGetInstanceProcAddr(NULL, "vkCreateInstance")` result
            // is guaranteed by the Vulkan spec to have this exact signature.
            Some(fp) => unsafe { std::mem::transmute(fp) },
            None => return LayerResult::Handled(Err(vk::Result::ERROR_INITIALIZATION_FAILED)),
        };
        let allocator = allocator.map_or(std::ptr::null(), |allocator| allocator as *const _);
        // SAFETY: `create_info`/`p_instance` are the same, still-valid pointers the
        // framework was called with; `allocator` is either null or that same valid
        // pointer.
        LayerResult::Handled(unsafe { create_instance(create_info, allocator, p_instance) }.result())
    }
}

#[derive(Default)]
struct DlssnrLayer(DlssnrGlobalHooks);

impl Layer for DlssnrLayer {
    type GlobalHooksInfo = DlssnrGlobalHooks;
    type InstanceInfo = StubInstanceInfo;
    type DeviceInfo = DlssnrDeviceInfo;
    type InstanceInfoContainer = StubInstanceInfo;
    type DeviceInfoContainer = DlssnrDeviceInfo;

    fn global_instance() -> impl Deref<Target = Global<Self>> + 'static {
        static GLOBAL: Lazy<Global<DlssnrLayer>> = Lazy::new(Default::default);
        &*GLOBAL
    }

    fn manifest() -> LayerManifest {
        let mut manifest = LayerManifest::default();
        manifest.name = LAYER_NAME;
        manifest.spec_version = vk::API_VERSION_1_1;
        manifest.implementation_version = 1;
        manifest.description = "DLSS 5 Neural Rendering injection layer (Linux side)";
        manifest
    }

    fn global_hooks_info(&self) -> &Self::GlobalHooksInfo {
        &self.0
    }

    fn create_instance_info(
        &self,
        _create_info: &vk::InstanceCreateInfo,
        _allocator: Option<&vk::AllocationCallbacks>,
        instance: Arc<ash::Instance>,
        _next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    ) -> Self::InstanceInfoContainer {
        *CURRENT_INSTANCE.lock().unwrap() = Some(instance);
        Default::default()
    }

    fn create_device_info(
        &self,
        physical_device: vk::PhysicalDevice,
        _create_info: &vk::DeviceCreateInfo,
        _allocator: Option<&vk::AllocationCallbacks>,
        device: Arc<ash::Device>,
        next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    ) -> Self::DeviceInfoContainer {
        // `create_instance_info` always runs before `create_device_info` for the
        // instance a device is created against (the app must call `vkCreateInstance`
        // before `vkCreateDevice`), so this is always `Some` in practice; `unwrap_or`
        // only matters for a hypothetical device created against an instance from
        // before this layer was loaded, which never happens for an implicit layer.
        let instance = CURRENT_INSTANCE.lock().unwrap().clone();
        DlssnrDeviceInfo::new(instance, physical_device, device, next_get_device_proc_addr)
    }
}

declare_introspection_queries!(Global::<DlssnrLayer>);
