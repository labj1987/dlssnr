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
//! TODO(milestone 4, see /home/alex/.claude/plans/breezy-napping-waffle.md): real
//! capture/compose (currently `queue_present_khr` always presents the original,
//! unmodified frame), device-extension injection for the dma-buf transport, the
//! inert-on-non-NVIDIA-device check, `pidfd_getfd` descriptor adoption, hotkey polling.

mod composition;
mod device;
mod logging;
mod shm;
mod swapchain;

use std::ops::Deref;
use std::sync::Arc;

use ash::vk;
use once_cell::sync::Lazy;
use vulkan_layer::{declare_introspection_queries, Global, Layer, LayerManifest, StubGlobalHooks, StubInstanceInfo};

use device::DlssnrDeviceInfo;

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

#[derive(Default)]
struct DlssnrLayer(StubGlobalHooks);

impl Layer for DlssnrLayer {
    type GlobalHooksInfo = StubGlobalHooks;
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
        manifest.spec_version = vk::API_VERSION_1_3;
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
        _instance: Arc<ash::Instance>,
        _next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    ) -> Self::InstanceInfoContainer {
        Default::default()
    }

    fn create_device_info(
        &self,
        _physical_device: vk::PhysicalDevice,
        _create_info: &vk::DeviceCreateInfo,
        _allocator: Option<&vk::AllocationCallbacks>,
        device: Arc<ash::Device>,
        next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    ) -> Self::DeviceInfoContainer {
        DlssnrDeviceInfo::new(device, next_get_device_proc_addr)
    }
}

declare_introspection_queries!(Global::<DlssnrLayer>);
