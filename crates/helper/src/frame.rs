//! Milestone 4 phase B: the real per-frame resources `EvaluateFeature` needs -- Color
//! (the proxy the layer captured), Output (where the model writes its answer), MVec
//! (motion vectors), and Depth -- and the upload/evaluate/download sequence that binds
//! them.
//!
//! MVec scope: no real optical flow yet (`VK_NV_optical_flow`, mentioned in the
//! README, isn't wired up anywhere in this crate) -- this always hands the model an
//! all-zero MVec image, i.e. "no motion", a safe default rather than nothing at all.
//! Wiring up real synthetic motion vectors is separate follow-up work, not required
//! for the create/evaluate/read-back plumbing this module proves.
//!
//! Depth scope, added 2026-09-10: `DLSSNR.Depth`/`DLSSNR.DepthInverted` are real,
//! confirmed-present parameters (`strings` against the real `nvngx_dlssnr.dll` turns up
//! `DLSSNR: EvaluateFeature Color=%p MVec=%p Depth=%p Output=%p ...`, naming exactly
//! four resources) this crate never bound before -- a real, plausible cause of a first
//! real visual check (see `CLAUDE.md`) turning up a solid-white `EvaluateFeature`
//! answer despite a `0x1` success code. Like MVec, there is no real depth buffer to
//! give it yet (`dlssnr_layer::capture` only ever captures the presented color image),
//! so this hands the model a constant, synthetic "far plane, no real depth" value --
//! an honest stand-in, not a real per-pixel depth buffer, same spirit as MVec's
//! all-zero placeholder.
//!
//! Same staging-copy discipline as `dlssnr_layer::capture`: images are populated via
//! an explicit host-visible-buffer upload/download, not a zero-copy import.

use ash::vk;

use crate::abi::{self, NgxImageViewInfoVk, NgxResourceVk};

pub struct FrameResources {
    width: u32,
    height: u32,
    queue_family: u32,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,

    color_image: vk::Image,
    color_view: vk::ImageView,
    color_memory: vk::DeviceMemory,
    output_image: vk::Image,
    output_view: vk::ImageView,
    output_memory: vk::DeviceMemory,
    mvec_image: vk::Image,
    mvec_view: vk::ImageView,
    mvec_memory: vk::DeviceMemory,
    depth_image: vk::Image,
    depth_view: vk::ImageView,
    depth_memory: vk::DeviceMemory,

    /// Host-visible staging, sized to the larger of upload (Color/MVec) or download
    /// (Output) -- one buffer, reused sequentially, same simplification
    /// `dlssnr_layer::capture` makes for its own single staging buffer.
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_ptr: *mut u8,

    /// Whether `evaluate` has run at least once yet -- see `DLSSNR.Reset`'s own
    /// handling in `evaluate` for why this matters. `Cell`, not a plain `bool`:
    /// `evaluate` takes `&self` (this whole struct is a fixed, per-size-class set of
    /// GPU resources shared across every frame at that size, not something that needs
    /// `&mut` to use), so this is the one piece of real per-frame state that needs
    /// interior mutability to track from there.
    reset_done: std::cell::Cell<bool>,
}

// SAFETY: every field is a plain Vulkan handle or a `vkMapMemory` pointer into memory
// this struct owns exclusively -- never aliased outside the single-threaded frame loop
// in `main.rs` that owns this value.
unsafe impl Send for FrameResources {}

const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const MVEC_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;
// `DLSSNR.Depth`/`DLSSNR.DepthInverted` exist in the real DLL's own string table
// (`EvaluateFeature Color=%p MVec=%p Depth=%p Output=%p` -- confirmed present via
// `strings` against the real binary, 2026-09-10) but were never bound here before --
// this crate had no depth buffer to give it and the real capture path
// (`dlssnr_layer::capture`) only ever captures the presented color image, never a
// depth attachment. A color-aspect (not a real `D32_SFLOAT` depth-aspect image, to
// avoid the different layout/aspect-mask rules those need) constant-far-plane image is
// a synthetic stand-in -- "no usable depth" as honestly as this crate can currently
// say it, not a real per-pixel depth buffer. See this module's own doc comment.
const DEPTH_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

fn find_memory_type(props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32, wanted: vk::MemoryPropertyFlags) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| (type_bits & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(wanted))
}

fn create_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Option<(vk::Image, vk::ImageView, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `info` is a valid `VkImageCreateInfo`.
    let image = unsafe { device.create_image(&info, None) }.ok()?;
    // SAFETY: `image` was just created and has no memory bound yet.
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let type_index = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::empty()))?;
    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
    // SAFETY: `alloc` is valid; `type_index` satisfies `reqs`.
    let memory = unsafe { device.allocate_memory(&alloc, None) }.ok().or_else(|| {
        // SAFETY: `image` has no memory bound; freeing it here is sound.
        unsafe { device.destroy_image(image, None) };
        None
    })?;
    // SAFETY: `image`/`memory` were each just created above, sized/typed for each other.
    if unsafe { device.bind_image_memory(image, memory, 0) }.is_err() {
        // SAFETY: neither is aliased anywhere else yet.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
        }
        return None;
    }
    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        );
    // SAFETY: `image` is bound to memory; `view_info` matches it.
    let view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(v) => v,
        Err(_) => {
            // SAFETY: nothing else references `image`/`memory` yet.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
            }
            return None;
        }
    };
    Some((image, view, memory))
}

impl FrameResources {
    /// Builds every resource `EvaluateFeature` needs for a `width`x`height` frame.
    /// `None` on any failure -- callers treat that as "skip evaluate this frame",
    /// mirroring `dlssnr_layer::capture`'s own fail-open discipline.
    pub fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        // SAFETY: `physical_device` is the device everything below is built against.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let (color_image, color_view, color_memory) = create_image(
            device,
            &mem_props,
            width,
            height,
            COLOR_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let (output_image, output_view, output_memory) = create_image(
            device,
            &mem_props,
            width,
            height,
            COLOR_FORMAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let (mvec_image, mvec_view, mvec_memory) = create_image(
            device,
            &mem_props,
            width,
            height,
            MVEC_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let (depth_image, depth_view, depth_memory) = create_image(
            device,
            &mem_props,
            width,
            height,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;

        let pool_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `pool_info` is valid.
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.ok()?;
        let alloc_info =
            vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        // SAFETY: `pool` was just created above.
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.ok()?[0];
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: `fence_info` is valid.
        let fence = unsafe { device.create_fence(&fence_info, None) }.ok()?;

        // Sized for the largest single-plane 8-bit transfer this ever does (Color or
        // Output, both 4 bytes/pixel); MVec (2 bytes/pixel) always fits the same
        // allocation.
        let staging_size = u64::from(width) * u64::from(height) * 4;
        let buf_info = vk::BufferCreateInfo::builder()
            .size(staging_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `buf_info` is valid.
        let staging_buffer = unsafe { device.create_buffer(&buf_info, None) }.ok()?;
        // SAFETY: `staging_buffer` was just created, not yet bound to memory.
        let reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let type_index = find_memory_type(
            &mem_props,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
        // SAFETY: `alloc` is valid; `type_index` satisfies `reqs`.
        let staging_memory = unsafe { device.allocate_memory(&alloc, None) }.ok()?;
        // SAFETY: `staging_buffer`/`staging_memory` were each just created, sized/typed
        // for each other.
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }.ok()?;
        // SAFETY: `staging_memory` is `HOST_VISIBLE`; mapping the whole allocation is
        // always in bounds.
        let staging_ptr = unsafe { device.map_memory(staging_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.ok()?.cast::<u8>();

        Some(Self {
            width,
            height,
            queue_family,
            pool,
            cmd,
            fence,
            color_image,
            color_view,
            color_memory,
            output_image,
            output_view,
            output_memory,
            mvec_image,
            mvec_view,
            mvec_memory,
            depth_image,
            depth_view,
            depth_memory,
            staging_buffer,
            staging_memory,
            staging_ptr,
            reset_done: std::cell::Cell::new(false),
        })
    }

    pub fn matches(&self, queue_family: u32, width: u32, height: u32) -> bool {
        self.queue_family == queue_family && self.width == width && self.height == height
    }

    /// Uploads `proxy` into Color, zeroes MVec, runs `EvaluateFeature`, downloads
    /// Output into `answer_out`. Returns `false` (leaving `answer_out` untouched) on
    /// any failure, including a guarded fault inside `EvaluateFeature` itself.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        evaluate_feature: abi::FnVkEvaluateFeature,
        feature: abi::NgxHandle,
        params: abi::NgxParameter,
        proxy: &[u8],
        answer_out: &mut [u8],
    ) -> bool {
        let pixel_count = (self.width as usize) * (self.height as usize);
        if proxy.len() < pixel_count * 4 || answer_out.len() < pixel_count * 4 {
            return false;
        }

        // Stage 1: upload proxy -> Color, zero -> MVec.
        // SAFETY: `staging_ptr` is a live mapping of at least `pixel_count * 4` bytes
        // (this type's own construction sized it to exactly that).
        unsafe { std::ptr::copy_nonoverlapping(proxy.as_ptr(), self.staging_ptr, pixel_count * 4) };
        if !self.run_transfer(device, queue, TransferKind::Upload) {
            return false;
        }

        // Stage 2: the real NGX call, guarded the same way every other DLL call in
        // this crate already is.
        let color_info = self.resource_info(self.color_view, self.color_image, COLOR_FORMAT);
        let output_info = self.resource_info(self.output_view, self.output_image, COLOR_FORMAT);
        let mvec_info = self.resource_info(self.mvec_view, self.mvec_image, MVEC_FORMAT);
        let depth_info = self.resource_info(self.depth_view, self.depth_image, DEPTH_FORMAT);
        // Parameter names guessed "for shape" from the same `DLSSNR.*` convention
        // `crates/helper/src/ngx.rs::create_feature_at` already uses for the scalar
        // parameters -- no public spec exists for this fictional feature's resource
        // bindings any more than for its scalars. Wrong names/slots here fail via the
        // guard below, not a crash, exactly like a wrong `CreateFeature` parameter did
        // before the v0.1.2 fix. `Depth`/`DepthInverted` and every `*Subrect*` name
        // below are confirmed present in the real DLL's own string table (`strings`,
        // 2026-09-10) -- not new guesses, the first ones checked against real evidence.
        let name = |n: &str| std::ffi::CString::new(n).unwrap();
        // SAFETY: `params` was allocated and validated by the caller (`ngx::load_and_init`).
        unsafe {
            let mut color = NgxResourceVk::from_image_view(color_info, false);
            abi::ngx_set_ptr(params, name("DLSSNR.Color").as_ptr(), std::ptr::from_mut(&mut color).cast());
            let mut output = NgxResourceVk::from_image_view(output_info, true);
            abi::ngx_set_ptr(params, name("DLSSNR.Output").as_ptr(), std::ptr::from_mut(&mut output).cast());
            let mut mvec = NgxResourceVk::from_image_view(mvec_info, false);
            abi::ngx_set_ptr(params, name("DLSSNR.MVec").as_ptr(), std::ptr::from_mut(&mut mvec).cast());
            let mut depth = NgxResourceVk::from_image_view(depth_info, false);
            abi::ngx_set_ptr(params, name("DLSSNR.Depth").as_ptr(), std::ptr::from_mut(&mut depth).cast());
            // Standard, non-reversed-Z convention (near=0, far=1) -- matches the
            // constant 1.0 ("far") the depth image is cleared to in `run_transfer`.
            abi::ngx_set_u32(params, name("DLSSNR.DepthInverted").as_ptr(), 0);

            // Every resource is the full frame at (0,0) -- no sub-rect windowing is
            // used anywhere in this crate yet.
            for resource in ["Color", "Output", "MVec", "Depth"] {
                abi::ngx_set_u32(params, name(&format!("DLSSNR.{resource}SubrectBaseX")).as_ptr(), 0);
                abi::ngx_set_u32(params, name(&format!("DLSSNR.{resource}SubrectBaseY")).as_ptr(), 0);
                abi::ngx_set_u32(params, name(&format!("DLSSNR.{resource}SubrectWidth")).as_ptr(), self.width);
                abi::ngx_set_u32(params, name(&format!("DLSSNR.{resource}SubrectHeight")).as_ptr(), self.height);
            }
            // `ngx::create_feature_at` sets `DLSSNR.Reset = 1` once, at creation, and
            // this crate never touched it again before now -- every single evaluate
            // call therefore told the model "no valid history, this is frame one" for
            // the life of the feature, a real, plausible cause of a first real visual
            // check (see CLAUDE.md) finding every frame produces the exact same
            // (solid white) output regardless of input: a temporal model's real,
            // history-dependent answer would only ever appear from the second frame
            // set to `Reset = 0` onward, which never happened before this. `1` only on
            // this feature's actual first `evaluate` call, `0` on every one after.
            let reset = u32::from(!self.reset_done.replace(true));
            abi::ngx_set_u32(params, name("DLSSNR.Reset").as_ptr(), reset);

            let (result, seh) = crate::guard::guarded(
                || {
                    // SAFETY: `evaluate_feature` was resolved from the live snippet
                    // module; `feature`/`params` were validated at creation; a null
                    // command buffer matches this crate's own create-time convention
                    // (no active command buffer recording -- see `ngx.rs`'s doc
                    // comment on `create_feature_at`'s identical choice) and a null
                    // progress callback is always accepted per the NGX contract.
                    evaluate_feature(vk::CommandBuffer::null(), feature, params, std::ptr::null())
                },
                abi::result::FAIL_SEH,
            );
            crate::log!("[ngx] EvaluateFeature -> {:#x} seh={:#x}", result as u32, seh);
            if !abi::succeeded(result) {
                return false;
            }
        }

        // Stage 3: download Output -> answer_out.
        if !self.run_transfer(device, queue, TransferKind::Download) {
            return false;
        }
        // SAFETY: `staging_ptr` is a live mapping of at least `pixel_count * 4` bytes.
        unsafe { std::ptr::copy_nonoverlapping(self.staging_ptr, answer_out.as_mut_ptr(), pixel_count * 4) };
        true
    }

    fn resource_info(&self, view: vk::ImageView, image: vk::Image, format: vk::Format) -> NgxImageViewInfoVk {
        NgxImageViewInfoVk {
            image_view: view,
            image,
            subresource_range: vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
            format,
            width: self.width,
            height: self.height,
        }
    }

    fn run_transfer(&self, device: &ash::Device, queue: vk::Queue, kind: TransferKind) -> bool {
        // SAFETY: `self.cmd` was allocated from `self.pool`, created with
        // `RESET_COMMAND_BUFFER`.
        if unsafe { device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
            return false;
        }
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `self.cmd` was just reset above.
        if unsafe { device.begin_command_buffer(self.cmd, &begin_info) }.is_err() {
            return false;
        }
        let region = |width: u32, height: u32| {
            vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_offset(vk::Offset3D::default())
                .image_extent(vk::Extent3D { width, height, depth: 1 })
                .build()
        };
        let sub = |aspect| {
            vk::ImageSubresourceRange::builder().aspect_mask(aspect).base_mip_level(0).level_count(1).base_array_layer(0).layer_count(1).build()
        };
        let img_barrier = |image, old, new, src, dst| {
            vk::ImageMemoryBarrier::builder()
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(sub(vk::ImageAspectFlags::COLOR))
                .src_access_mask(src)
                .dst_access_mask(dst)
                .build()
        };
        match kind {
            TransferKind::Upload => {
                let to_dst_color = img_barrier(
                    self.color_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                let to_dst_mvec = img_barrier(
                    self.mvec_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                let to_dst_depth = img_barrier(
                    self.depth_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                // SAFETY: `self.cmd` is recording; all three images were just created
                // (`UNDEFINED` matches their real, never-yet-transitioned layout).
                unsafe {
                    device.cmd_pipeline_barrier(
                        self.cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_dst_color, to_dst_mvec, to_dst_depth],
                    );
                    device.cmd_copy_buffer_to_image(
                        self.cmd,
                        self.staging_buffer,
                        self.color_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region(self.width, self.height)],
                    );
                    // MVec: no real motion vectors yet (see module doc comment) --
                    // clear to zero ("no motion") rather than leave it undefined.
                    device.cmd_clear_color_image(
                        self.cmd,
                        self.mvec_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue { float32: [0.0; 4] },
                        &[sub(vk::ImageAspectFlags::COLOR)],
                    );
                    // Depth: no real depth buffer captured yet either (see module doc
                    // comment) -- a constant 1.0 ("far plane", standard non-reversed-Z
                    // convention, matching `DLSSNR.DepthInverted = 0` below) rather
                    // than leaving it undefined.
                    device.cmd_clear_color_image(
                        self.cmd,
                        self.depth_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue { float32: [1.0, 1.0, 1.0, 1.0] },
                        &[sub(vk::ImageAspectFlags::COLOR)],
                    );
                    let to_shader = img_barrier(
                        self.color_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    );
                    let mvec_to_shader = img_barrier(
                        self.mvec_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    );
                    let depth_to_shader = img_barrier(
                        self.depth_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    );
                    let output_to_general = img_barrier(
                        self.output_image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::SHADER_WRITE,
                    );
                    device.cmd_pipeline_barrier(
                        self.cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_shader, mvec_to_shader, depth_to_shader, output_to_general],
                    );
                }
            }
            TransferKind::Download => {
                let to_src = img_barrier(
                    self.output_image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                );
                // SAFETY: `self.cmd` is recording; `output_image` was left `GENERAL`
                // by the upload stage's own final barrier, matching what
                // `EvaluateFeature` (run on the CPU-side call in between, not this
                // command buffer) was told to expect as the storage image's layout.
                unsafe {
                    device.cmd_pipeline_barrier(
                        self.cmd,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_src],
                    );
                    device.cmd_copy_image_to_buffer(
                        self.cmd,
                        self.output_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.staging_buffer,
                        &[region(self.width, self.height)],
                    );
                }
            }
        }
        if unsafe { device.end_command_buffer(self.cmd) }.is_err() {
            return false;
        }
        // SAFETY: `self.fence` starts signaled or was reset+waited-on by this same
        // function's previous call.
        if unsafe { device.reset_fences(&[self.fence]) }.is_err() {
            return false;
        }
        let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&self.cmd)).build();
        // SAFETY: `self.cmd` was just recorded and ended above.
        if unsafe { device.queue_submit(queue, &[submit], self.fence) }.is_err() {
            return false;
        }
        // SAFETY: `self.fence` was just submitted against above.
        unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX) }.is_ok()
    }

    /// # Safety
    /// Must only be called at process-teardown time, with no submitted work
    /// referencing these handles still in flight.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_buffer(self.staging_buffer, None);
            device.free_memory(self.staging_memory, None);
            device.destroy_image_view(self.color_view, None);
            device.destroy_image(self.color_image, None);
            device.free_memory(self.color_memory, None);
            device.destroy_image_view(self.output_view, None);
            device.destroy_image(self.output_image, None);
            device.free_memory(self.output_memory, None);
            device.destroy_image_view(self.mvec_view, None);
            device.destroy_image(self.mvec_image, None);
            device.free_memory(self.mvec_memory, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_memory, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

enum TransferKind {
    Upload,
    Download,
}
