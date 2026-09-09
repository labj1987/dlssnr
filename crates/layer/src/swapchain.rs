//! Per-swapchain state, and what a swapchain's format/colour space say about the light
//! it carries.

use ash::vk;
use dlssnr_protocol::enums::hdr_kind;

pub struct SwapchainState {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub hdr_kind: u32,
    /// True when this swapchain's format isn't one the pass can work with, or it's
    /// larger than the protocol's ceiling (`dlssnr_protocol::{MAX_W,MAX_H}`) -- it
    /// presents untouched either way.
    pub pass_through: bool,
    /// The swapchain's own images, in `vkGetSwapchainImagesKHR` order -- index `i`
    /// here is exactly what `VkPresentInfoKHR::pImageIndices[i]` refers to. Fetched
    /// once at creation (see `device::DlssnrDeviceInfo::fetch_swapchain_images`).
    pub images: Vec<vk::Image>,
}

/// A blunt filter against known-small compositor/overlay swapchains (the Steam
/// overlay's own render target showed up at 1262x598 in testing) that would otherwise
/// each separately decide they're "primary" -- `device::PRIMARY`'s claim is per
/// *process*, and the overlay runs as its own separate process with its own copy of
/// every static in this crate, so it can't see that the game already claimed primary
/// in a different address space. Not a real fix for multi-process arbitration (there
/// isn't one yet -- see the milestone-4 plan's known-gaps note); just enough to stop
/// the overlay's own tiny swapchain from fighting the game's real one over the same
/// shared-memory transport.
pub fn is_plausible_game_size(width: u32, height: u32) -> bool {
    u64::from(width) * u64::from(height) >= 1280 * 720
}

/// Whether this is a format the composition pass can work in. Every one has an 8-bit
/// UNORM twin the pass can normalize to; the 10-bit and float entries are what let an
/// HDR game reach the model at all.
///
/// This is a starting list covering the common presentable formats, not a claim of
/// completeness -- widen it if a real game surfaces a swapchain format outside this
/// set (the layer already fails safely closed: an unrecognized format means
/// `pass_through`, never a wrong read).
pub fn is_supported_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::B8G8R8A8_UNORM
            | vk::Format::B8G8R8A8_SRGB
            | vk::Format::R8G8B8A8_UNORM
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::A2B10G10R10_UNORM_PACK32
            | vk::Format::A2R10G10B10_UNORM_PACK32
            | vk::Format::R16G16B16A16_SFLOAT
    )
}

/// The `dlssnr_protocol::enums::proxy_format` this swapchain's raw
/// `vkCmdCopyImageToBuffer` dump actually is -- only meaningful for formats
/// [`is_supported_format`] already accepted. `proxy_format::bytes_per_pixel` is the
/// single source of truth for the byte count that goes with this; nothing here
/// duplicates it.
pub fn proxy_format_for(format: vk::Format) -> u32 {
    match format {
        vk::Format::R16G16B16A16_SFLOAT => dlssnr_protocol::enums::proxy_format::RGBA16F,
        _ => dlssnr_protocol::enums::proxy_format::RGBA8,
    }
}

/// What a swapchain's format and colour space together say about the light in the
/// frame. A float swapchain hands over linear light directly. A ten-bit swapchain in an
/// HDR10/PQ colour space carries ST 2084 code -- absolute nits, not just more precision
/// on an already-tone-mapped picture, which is what the same ten-bit format in an SDR
/// colour space would be.
pub fn detect_hdr_kind(format: vk::Format, color_space: vk::ColorSpaceKHR) -> u32 {
    if format == vk::Format::R16G16B16A16_SFLOAT {
        return hdr_kind::LINEAR_FP16;
    }
    let ten_bit = matches!(
        format,
        vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32
    );
    let pq = color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT;
    if ten_bit && pq {
        hdr_kind::PQ10
    } else {
        hdr_kind::NONE
    }
}
