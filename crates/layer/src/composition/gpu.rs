//! GPU dispatch of `shaders/compose.comp`, wired into `capture.rs`'s write-back as of
//! 2026-09-10 -- the real fix for [`super::apply`]'s CPU path's real, measured
//! performance cost (see that module's own doc comment, and the crate's `CLAUDE.md`
//! entry, for the actual numbers: ~800ms/frame single-threaded, ~97/10s even
//! multi-threaded on a 16-core machine, both far short of the no-composition
//! baseline). Same algorithm (`compose.comp` is the hand-translated GPU twin of the
//! exact `color.rs` functions `apply.rs` calls directly), same real-hardware
//! verification target -- this is a different, faster execution strategy for
//! identical math, not a rewrite of it.
//!
//! Three storage images bound as inputs (`u_original`, `u_proxy`, `u_model_answer`)
//! and one as output (`u_output`), all `R8G8B8A8_UNORM` -- matching the real, only-
//! currently-supported `RGBA8` proxy format (`RGBA16F` still falls back to
//! [`super::apply`]'s CPU path, same gap that path already has, unchanged by this
//! module). `u_original` and `u_proxy` are bound to the *same* image view: no separate
//! downscaled proxy exists yet (`capture.rs` sends the full captured frame as the
//! proxy), so uploading it twice would be pure waste, on the GPU exactly as it already
//! was on the CPU path.
//!
//! One command buffer, one fence, fully synchronous (submit, then wait) -- the same
//! discipline `capture.rs`'s own two-stage transfer already uses, and for the same
//! reason: correctness first, ahead of double-buffering/pipelining this too.

use ash::vk;

const SPV: &[u8] = include_bytes!("../../shaders/compose.spv");
const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

#[repr(C)]
struct PushConstants {
    colour_strength: f32,
    transfer_strength: f32,
    max_ratio: f32,
}

struct Image {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

impl Image {
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract (device-destruction
        // time, or a same-size-class rebuild with no in-flight GPU work referencing
        // these handles -- both callers below only ever call this after a completed
        // `wait_for_fences`).
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn find_memory_type(props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32, wanted: vk::MemoryPropertyFlags) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| (type_bits & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(wanted))
}

fn create_storage_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
) -> Option<Image> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(FORMAT)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage | vk::ImageUsageFlags::STORAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `info` is a valid `VkImageCreateInfo`.
    let image = unsafe { device.create_image(&info, None) }.ok()?;
    // SAFETY: `image` was just created, no memory bound yet.
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let Some(type_index) = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::empty()))
    else {
        // SAFETY: `image` has no memory bound; nothing else references it.
        unsafe { device.destroy_image(image, None) };
        return None;
    };
    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
    // SAFETY: `alloc` is valid; `type_index` satisfies `reqs`.
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(_) => {
            // SAFETY: `image` has no memory bound.
            unsafe { device.destroy_image(image, None) };
            return None;
        }
    };
    // SAFETY: `image`/`memory` were each just created, sized/typed for each other.
    if unsafe { device.bind_image_memory(image, memory, 0) }.is_err() {
        // SAFETY: neither is aliased anywhere else yet.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
        }
        return None;
    }
    let view_info = vk::ImageViewCreateInfo::builder().image(image).view_type(vk::ImageViewType::TYPE_2D).format(FORMAT).subresource_range(
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
    Some(Image { image, view, memory })
}

/// The size-independent pipeline state: shader module, descriptor/pipeline layouts,
/// the compute pipeline itself. Built once and reused across every resolution --
/// `Sizeable` below is what actually changes on a resize.
pub struct GpuCompose {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    sized: Option<Sized_>,
}

// SAFETY: every field is either a plain Vulkan handle or (inside `Sized_`) a
// `vkMapMemory` pointer into memory this struct owns exclusively -- never aliased
// outside the `Mutex<State>` this always lives behind in `DlssnrDeviceInfo`, same
// reasoning as `capture::CaptureResources`.
unsafe impl Send for GpuCompose {}

struct Sized_ {
    width: u32,
    height: u32,
    original: Image,
    model_answer: Image,
    output: Image,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_ptr: *mut u8,
    staging_capacity: vk::DeviceSize,
}

impl GpuCompose {
    /// Builds the size-independent pipeline state. `None` on any failure -- callers
    /// treat that as "the GPU compose path isn't available", falling back to
    /// [`super::apply`]'s CPU path, never a reason to stop presenting frames.
    pub fn new(device: &ash::Device, queue_family: u32) -> Option<Self> {
        let bindings: Vec<_> = (0..4)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build()
            })
            .collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        // SAFETY: `layout_info` is valid.
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }.ok()?;

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .build();
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        // SAFETY: `pipeline_layout_info` is valid; `descriptor_set_layout` was just
        // created above.
        let pipeline_layout = match unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) } {
            Ok(l) => l,
            Err(_) => {
                // SAFETY: nothing else references `descriptor_set_layout` yet.
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return None;
            }
        };

        let Ok(code) = ash::util::read_spv(&mut std::io::Cursor::new(SPV)) else {
            // SAFETY: neither `pipeline_layout` nor `descriptor_set_layout` is
            // referenced anywhere else yet.
            unsafe {
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            return None;
        };
        let module_info = vk::ShaderModuleCreateInfo::builder().code(&code);
        // SAFETY: `module_info.code` is a valid SPIR-V module (`compose.spv`, compiled
        // from `shaders/compose.comp` via `glslangValidator -V`, validated with
        // `spirv-val` before being committed).
        let shader_module = match unsafe { device.create_shader_module(&module_info, None) } {
            Ok(m) => m,
            Err(_) => {
                // SAFETY: same reasoning as the branch above.
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return None;
            }
        };
        let entry_point = c"main";
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(*stage_info).layout(pipeline_layout);
        // SAFETY: `pipeline_info` is valid; `shader_module`/`pipeline_layout` were
        // just created above and outlive this call.
        let pipeline_result = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None) };
        // SAFETY: the module is never referenced again after pipeline creation
        // (successful or not) -- the Vulkan spec allows destroying it immediately.
        unsafe { device.destroy_shader_module(shader_module, None) };
        let pipeline = match pipeline_result {
            Ok(pipelines) => pipelines[0],
            Err(_) => {
                // SAFETY: same reasoning as the branches above.
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return None;
            }
        };

        let pool_size = vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(4).build();
        let pool_info =
            vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(std::slice::from_ref(&pool_size));
        // SAFETY: `pool_info` is valid.
        let descriptor_pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(_) => {
                // SAFETY: nothing else references these yet.
                unsafe {
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return None;
            }
        };
        let alloc_info =
            vk::DescriptorSetAllocateInfo::builder().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&descriptor_set_layout));
        // SAFETY: `descriptor_pool` was just created with room for exactly this one set.
        let descriptor_set = match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets[0],
            Err(_) => {
                // SAFETY: nothing else references these yet.
                unsafe {
                    device.destroy_descriptor_pool(descriptor_pool, None);
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return None;
            }
        };

        let pool_create_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `pool_create_info` is valid.
        let Ok(cmd_pool) = (unsafe { device.create_command_pool(&pool_create_info, None) }) else {
            // SAFETY: nothing else references these yet.
            unsafe {
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            return None;
        };
        let cmd_alloc_info =
            vk::CommandBufferAllocateInfo::builder().command_pool(cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        // SAFETY: `cmd_pool` was just created above.
        let Ok(cmd) = (unsafe { device.allocate_command_buffers(&cmd_alloc_info) }) else {
            // SAFETY: `cmd_pool` owns no other resources yet.
            unsafe {
                device.destroy_command_pool(cmd_pool, None);
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            return None;
        };
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: starting signaled means the first frame's own wait never blocks on a
        // fence nothing has submitted work against yet.
        let Ok(fence) = (unsafe { device.create_fence(&fence_info, None) }) else {
            // SAFETY: `cmd_pool` owns `cmd`; freeing the pool frees it too.
            unsafe {
                device.destroy_command_pool(cmd_pool, None);
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            return None;
        };

        Some(Self {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_set,
            pool: cmd_pool,
            cmd: cmd[0],
            fence,
            sized: None,
        })
    }

    fn ensure_sized(&mut self, device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, width: u32, height: u32) -> bool {
        if let Some(s) = &self.sized {
            if s.width == width && s.height == height {
                return true;
            }
            // SAFETY: called only between frames, after this struct's own `fence` has
            // been waited on by the previous `dispatch` call (or never submitted
            // against yet) -- never while GPU work might still reference these images.
            unsafe {
                s.original.destroy(device);
                s.model_answer.destroy(device);
                s.output.destroy(device);
                device.destroy_buffer(s.staging_buffer, None);
                device.free_memory(s.staging_memory, None);
            }
            self.sized = None;
        }

        let usage_in = vk::ImageUsageFlags::TRANSFER_DST;
        let usage_out = vk::ImageUsageFlags::TRANSFER_SRC;
        let (Some(original), Some(model_answer), Some(output)) = (
            create_storage_image(device, mem_props, width, height, usage_in),
            create_storage_image(device, mem_props, width, height, usage_in),
            create_storage_image(device, mem_props, width, height, usage_out),
        ) else {
            return false;
        };

        let frame_bytes = u64::from(width) * u64::from(height) * 4;
        let staging_capacity = frame_bytes * 2; // original + model_answer, uploaded together.
        let buf_info = vk::BufferCreateInfo::builder()
            .size(staging_capacity)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `buf_info` is valid.
        let Ok(staging_buffer) = (unsafe { device.create_buffer(&buf_info, None) }) else { return false };
        // SAFETY: `staging_buffer` was just created, not yet bound to memory.
        let reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let Some(type_index) =
            find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
        else {
            // SAFETY: `staging_buffer` has no memory bound.
            unsafe { device.destroy_buffer(staging_buffer, None) };
            return false;
        };
        let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
        // SAFETY: `alloc` is valid; `type_index` satisfies `reqs`.
        let Ok(staging_memory) = (unsafe { device.allocate_memory(&alloc, None) }) else {
            // SAFETY: same reasoning as above.
            unsafe { device.destroy_buffer(staging_buffer, None) };
            return false;
        };
        // SAFETY: `staging_buffer`/`staging_memory` were each just created, sized/typed
        // for each other.
        if unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }.is_err() {
            // SAFETY: neither is aliased anywhere else.
            unsafe {
                device.free_memory(staging_memory, None);
                device.destroy_buffer(staging_buffer, None);
            }
            return false;
        }
        // SAFETY: `staging_memory` is `HOST_VISIBLE`; mapping the whole allocation is
        // always in bounds.
        let Ok(staging_ptr) = (unsafe { device.map_memory(staging_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }) else {
            // SAFETY: same reasoning as above.
            unsafe {
                device.free_memory(staging_memory, None);
                device.destroy_buffer(staging_buffer, None);
            }
            return false;
        };

        let image_info = |view: vk::ImageView| {
            vk::DescriptorImageInfo::builder().image_view(view).image_layout(vk::ImageLayout::GENERAL).build()
        };
        let infos = [image_info(original.view), image_info(original.view), image_info(model_answer.view), image_info(output.view)];
        let writes: Vec<_> = (0..4u32)
            .map(|i| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set)
                    .dst_binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&infos[i as usize]))
                    .build()
            })
            .collect();
        // SAFETY: `descriptor_set` was allocated in `new()`, matches `writes`' layout
        // (4 storage-image bindings); every image view is live for at least as long as
        // `self.sized` holds its owning `Image`.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        self.sized = Some(Sized_ { width, height, original, model_answer, output, staging_buffer, staging_memory, staging_ptr: staging_ptr.cast(), staging_capacity });
        true
    }

    /// Runs `shaders/compose.comp` against `original`/`model_answer` (both `RGBA8`,
    /// `width`x`height`), writing the composited result back into `model_answer` in
    /// place -- the same signature and in-place-overwrite convention
    /// [`super::apply::apply_rgba8`] uses, so `capture.rs` can call either
    /// interchangeably. `false` (leaving `model_answer` untouched) on any failure,
    /// fails open exactly like every other stage of the capture path.
    #[allow(clippy::too_many_arguments)]
    fn image_copy_region(width: u32, height: u32, offset: u64) -> vk::BufferImageCopy {
        vk::BufferImageCopy::builder()
            .buffer_offset(offset)
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
    }

    fn full_subresource() -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build()
    }

    fn image_barrier(image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src: vk::AccessFlags, dst: vk::AccessFlags) -> vk::ImageMemoryBarrier {
        vk::ImageMemoryBarrier::builder()
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(Self::full_subresource())
            .src_access_mask(src)
            .dst_access_mask(dst)
            .build()
    }

    /// Records everything shared by [`Self::dispatch`] and
    /// [`Self::dispatch_into_image`] onto `self.cmd` (not yet begun): upload
    /// `original`/`model_answer` into `s.original`/`s.model_answer`, run the compute
    /// shader, and copy its `s.output` image back into `s.staging_buffer` at offset 0.
    /// Callers begin/end the command buffer and submit themselves, since what happens
    /// *after* this (download to a CPU slice, vs. straight into another image) is the
    /// one real difference between the two public methods.
    ///
    /// # Safety
    /// `self.cmd` must not already be recording (fresh reset or never begun).
    unsafe fn record_upload_and_compute(&self, device: &ash::Device, width: u32, height: u32, frame_bytes: u64, colour_strength: f32, transfer_strength: f32, max_ratio: f32) {
        let s = self.sized.as_ref().expect("caller already ensured this");
        let region = |offset| Self::image_copy_region(width, height, offset);
        // SAFETY: `self.cmd` is recording (forwarded from this function's own
        // contract); every image below was just (re)created by `ensure_sized` and is
        // still `UNDEFINED` (or is being deliberately discarded via `UNDEFINED` as
        // `oldLayout`, spec-legal and exactly what a fresh per-frame result needs --
        // see the crash-fix writeup in `CLAUDE.md` for why this specific pattern is
        // safe, unlike blindly assuming a *different* real prior layout).
        unsafe {
            let to_dst = [
                Self::image_barrier(s.original.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE),
                Self::image_barrier(s.model_answer.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE),
            ];
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_dst);
            device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, s.original.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(0)]);
            device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, s.model_answer.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(frame_bytes)]);

            let to_general = [
                Self::image_barrier(s.original.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ),
                Self::image_barrier(s.model_answer.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ),
                Self::image_barrier(s.output.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::AccessFlags::empty(), vk::AccessFlags::SHADER_WRITE),
            ];
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &to_general);

            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&self.descriptor_set), &[]);
            let push = PushConstants { colour_strength, transfer_strength, max_ratio };
            let push_bytes = std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), std::mem::size_of::<PushConstants>());
            device.cmd_push_constants(self.cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
            device.cmd_dispatch(self.cmd, width.div_ceil(8), height.div_ceil(8), 1);

            let to_src = Self::image_barrier(s.output.image, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
            device.cmd_copy_image_to_buffer(self.cmd, s.output.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, s.staging_buffer, &[region(0)]);
        }
    }

    fn begin_ensured(&mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, width: u32, height: u32, original: &[u8], model_answer: &[u8]) -> Option<u64> {
        let frame_bytes = (u64::from(width) * u64::from(height) * 4) as usize;
        if original.len() < frame_bytes || model_answer.len() < frame_bytes {
            return None;
        }
        // SAFETY: `physical_device` is the device this instance was created against.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        if !self.ensure_sized(device, &mem_props, width, height) {
            return None;
        }
        let s = self.sized.as_ref().expect("just ensured above");
        if (s.staging_capacity as usize) < frame_bytes * 2 {
            return None;
        }
        // SAFETY: `s.staging_ptr` is a live mapping of at least `frame_bytes * 2`
        // bytes; `original`/`model_answer` were just confirmed at least `frame_bytes`.
        unsafe {
            std::ptr::copy_nonoverlapping(original.as_ptr(), s.staging_ptr, frame_bytes);
            std::ptr::copy_nonoverlapping(model_answer.as_ptr(), s.staging_ptr.add(frame_bytes), frame_bytes);
        }
        Some(frame_bytes as u64)
    }

    /// Runs `shaders/compose.comp` against `original`/`model_answer` (both `RGBA8`,
    /// `width`x`height`), writing the composited result back into `model_answer` in
    /// place -- the same signature and in-place-overwrite convention
    /// [`super::apply::apply_rgba8`] uses, so `capture.rs` can call either
    /// interchangeably. `false` (leaving `model_answer` untouched) on any failure,
    /// fails open exactly like every other stage of the capture path.
    ///
    /// Downloads the result to a CPU-visible slice -- use this when something on the
    /// CPU actually needs to see the bytes (a pending `capture_request` dump in
    /// particular). [`Self::dispatch_into_image`] is the faster, no-CPU-round-trip
    /// path for the common case where nothing does.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        width: u32,
        height: u32,
        original: &[u8],
        model_answer: &mut [u8],
        colour_strength: f32,
        transfer_strength: f32,
        max_ratio: f32,
    ) -> bool {
        let Some(frame_bytes) = self.begin_ensured(device, instance, physical_device, width, height, original, model_answer) else { return false };

        // SAFETY: `self.cmd` was allocated from `self.pool`, created with
        // `RESET_COMMAND_BUFFER`.
        if unsafe { device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
            return false;
        }
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `self.cmd` was just reset.
        if unsafe { device.begin_command_buffer(self.cmd, &begin_info) }.is_err() {
            return false;
        }
        // SAFETY: `self.cmd` was just begun above.
        unsafe { self.record_upload_and_compute(device, width, height, frame_bytes, colour_strength, transfer_strength, max_ratio) };
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
        if unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX) }.is_err() {
            return false;
        }

        let s = self.sized.as_ref().expect("ensured by begin_ensured above");
        // SAFETY: the fence wait above guarantees the download copy has completed;
        // `s.staging_ptr` is host-coherent (no explicit invalidate needed).
        unsafe { std::ptr::copy_nonoverlapping(s.staging_ptr, model_answer.as_mut_ptr(), frame_bytes as usize) };
        true
    }

    /// Same composition as [`Self::dispatch`], but writes the result directly into
    /// `target_image` (assumed already `TRANSFER_DST_OPTIMAL` -- exactly the layout
    /// `capture::run`'s own stage 1 already leaves the real swapchain image in) instead
    /// of downloading it to a CPU-visible slice, and in the *same* command
    /// buffer/submission as the compute dispatch itself, restoring `target_image` to
    /// `PRESENT_SRC_KHR` before returning.
    ///
    /// This is the real point of this method, not just "one fewer copy": it lets
    /// `capture::run` skip its own separate stage-2 submission entirely for the common
    /// case (GPU compose succeeds, no debug dump pending) -- two total GPU
    /// submissions/fence-waits per frame instead of three, and zero CPU round trips
    /// for the composited bytes at all (the intermediate still passes through
    /// `s.staging_buffer`, but purely as a device-side buffer -- copying through a
    /// *buffer* rather than image-to-image straight from `s.output` is deliberate: a
    /// raw `vkCmdCopyImage` between two images of different formats is a byte-for-byte
    /// copy with no channel-swizzle, so it would silently corrupt colors if
    /// `target_image`'s real format ever differs from this struct's own hardcoded
    /// `FORMAT` -- e.g. a `B8G8R8A8` swapchain vs. this module's `R8G8B8A8`. A
    /// buffer-to-image copy has no such format attached to the source, so it is always
    /// correct regardless of what `target_image`'s real format turns out to be, the
    /// same reasoning `capture.rs`'s own existing stage 2 already relies on.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into_image(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        width: u32,
        height: u32,
        original: &[u8],
        model_answer: &[u8],
        colour_strength: f32,
        transfer_strength: f32,
        max_ratio: f32,
        target_image: vk::Image,
    ) -> bool {
        let Some(frame_bytes) = self.begin_ensured(device, instance, physical_device, width, height, original, model_answer) else { return false };

        // SAFETY: `self.cmd` was allocated from `self.pool`, created with
        // `RESET_COMMAND_BUFFER`.
        if unsafe { device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
            return false;
        }
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `self.cmd` was just reset.
        if unsafe { device.begin_command_buffer(self.cmd, &begin_info) }.is_err() {
            return false;
        }
        // SAFETY: `self.cmd` was just begun above.
        unsafe { self.record_upload_and_compute(device, width, height, frame_bytes, colour_strength, transfer_strength, max_ratio) };

        let s = self.sized.as_ref().expect("just ensured by begin_ensured above");
        // SAFETY: `self.cmd` is still recording. `s.staging_buffer` offset 0 was just
        // written by `record_upload_and_compute`'s own final `cmd_copy_image_to_buffer`
        // -- the buffer memory barrier makes that write visible to this read.
        // `target_image` is `TRANSFER_DST_OPTIMAL` per this function's own contract
        // (`capture::run`'s stage 1 guarantees this for the real swapchain image).
        unsafe {
            let buffer_barrier = vk::BufferMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(s.staging_buffer)
                .offset(0)
                .size(frame_bytes)
                .build();
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[buffer_barrier], &[]);
            device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, target_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[Self::image_copy_region(width, height, 0)]);
            let to_present = Self::image_barrier(target_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
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
    /// Must only be called at device-destruction time, with no submitted work
    /// referencing these handles still in flight.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            if let Some(s) = &self.sized {
                s.original.destroy(device);
                s.model_answer.destroy(device);
                s.output.destroy(device);
                device.destroy_buffer(s.staging_buffer, None);
                device.free_memory(s.staging_memory, None);
            }
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// A real `VkInstance`/`VkDevice`/queue through the system Vulkan loader --
/// `None` if no loader/ICD is available in this environment (this crate's own
/// `examples/smoke.rs` doc comment already establishes lavapipe is enough, no real
/// GPU needed, for exactly this kind of check). Shared by every `#[test]` below
/// rather than each standing up its own instance/device.
#[cfg(test)]
fn test_device() -> Option<(ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue, u32)> {
    // SAFETY: same reasoning as `examples/smoke.rs`'s identical call.
    let entry = unsafe { ash::Entry::load() }.ok()?;
    let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    // SAFETY: `create_info` is valid.
    let instance = unsafe { entry.create_instance(&create_info, None) }.ok()?;
    // SAFETY: `instance` was just created and outlives every use of `physical_device`.
    let physical_device = *unsafe { instance.enumerate_physical_devices() }.ok()?.first()?;
    let queue_family = 0;
    let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(queue_family).queue_priorities(&[1.0]).build()];
    let device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);
    // SAFETY: `device_create_info` is valid; every physical device has a family 0
    // (the Vulkan spec guarantees at least one queue family).
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.ok()?;
    // SAFETY: `device`/family/index 0 match what `device_create_info` just requested.
    let queue = unsafe { device.get_device_queue(queue_family, 0) };
    Some((entry, instance, physical_device, device, queue, queue_family))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real correctness check for this whole module: dispatches
    /// `shaders/compose.comp` on a real (if software) Vulkan device and confirms
    /// its output matches [`super::super::apply::apply_rgba8`] -- the same
    /// already-real-hardware-verified CPU reference (see the crate's `CLAUDE.md`
    /// entry) -- to within a small per-channel tolerance (GPU and CPU `pow`/`cbrt`
    /// implementations are never bit-identical, only close). A real, non-uniform
    /// test image (not one flat color) so the tone-mapping/OkLab branches this
    /// algorithm actually has are exercised, not just the identity case.
    #[test]
    fn gpu_dispatch_matches_the_cpu_reference() {
        let Some((_entry, instance, physical_device, device, queue, _queue_family)) = test_device() else {
            eprintln!("gpu_dispatch_matches_the_cpu_reference: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, 0) else {
            eprintln!("gpu_dispatch_matches_the_cpu_reference: GpuCompose::new failed (e.g. no compute-capable queue), skipping");
            // SAFETY: nothing was created past the device/instance.
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return;
        };

        let (width, height) = (8u32, 8u32);
        let pixel_count = (width * height) as usize;
        // A real gradient plus a few deliberately out-of-band pixels, not a flat
        // color -- exercises the headroom/no-headroom branches, the OkLab hue
        // correction, and the gamut-compression path all at once.
        let original: Vec<u8> = (0..pixel_count)
            .flat_map(|i| {
                let t = (i * 37 % 256) as u8;
                [t, t.wrapping_add(64), t.wrapping_add(128), 255]
            })
            .collect();
        let model_answer: Vec<u8> = (0..pixel_count)
            .flat_map(|i| {
                let t = (i * 53 % 256) as u8;
                [t.wrapping_add(20), t, t.wrapping_add(200), 255]
            })
            .collect();

        let (colour_strength, transfer_strength, max_ratio) = (0.7, 0.8, 2.0);

        let mut gpu_result = model_answer.clone();
        let ok = gpu.dispatch(&device, &instance, physical_device, queue, width, height, &original, &mut gpu_result, colour_strength, transfer_strength, max_ratio);
        assert!(ok, "GpuCompose::dispatch returned false");

        let mut cpu_result = model_answer.clone();
        super::super::apply::apply_rgba8(&original, &mut cpu_result, colour_strength, transfer_strength, max_ratio, 0);

        let mut max_diff = 0i32;
        for (g, c) in gpu_result.chunks_exact(4).zip(cpu_result.chunks_exact(4)) {
            for ch in 0..3 {
                max_diff = max_diff.max((i32::from(g[ch]) - i32::from(c[ch])).abs());
            }
        }
        assert!(max_diff <= 3, "GPU and CPU composition diverge by up to {max_diff} (expected <= 3): gpu={gpu_result:?} cpu={cpu_result:?}");

        // SAFETY: `gpu`'s own fence wait inside `dispatch` guarantees no GPU work
        // is in flight; nothing else references `device`/`instance`.
        unsafe {
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// A second real dispatch at a *different* size than the first test used,
    /// through the *same* `GpuCompose` instance-creation path -- confirms
    /// `ensure_sized`'s rebuild-on-resize path (not just its happy-path "already
    /// the right size" branch) actually works, real device included.
    #[test]
    fn gpu_dispatch_handles_a_resize() {
        let Some((_entry, instance, physical_device, device, queue, _queue_family)) = test_device() else {
            eprintln!("gpu_dispatch_handles_a_resize: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, 0) else {
            eprintln!("gpu_dispatch_handles_a_resize: GpuCompose::new failed, skipping");
            // SAFETY: nothing was created past the device/instance.
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return;
        };

        for &(width, height) in &[(4u32, 4u32), (16u32, 12u32)] {
            let pixel_count = (width * height) as usize;
            let original = vec![128u8; pixel_count * 4];
            let mut answer = vec![100u8; pixel_count * 4];
            let ok = gpu.dispatch(&device, &instance, physical_device, queue, width, height, &original, &mut answer, 1.0, 1.0, 2.0);
            assert!(ok, "dispatch failed at {width}x{height}");
        }

        // SAFETY: same reasoning as the test above.
        unsafe {
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// The real point of `dispatch_into_image`: confirms it produces the *same*
    /// composited result as [`GpuCompose::dispatch`] when writing directly into a
    /// target image instead of a CPU slice -- real device, real image transitions,
    /// real merged submission, not just "doesn't return false."
    #[test]
    fn dispatch_into_image_matches_dispatch() {
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("dispatch_into_image_matches_dispatch: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, queue_family) else {
            eprintln!("dispatch_into_image_matches_dispatch: GpuCompose::new failed, skipping");
            // SAFETY: nothing was created past the device/instance.
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return;
        };

        let (width, height) = (8u32, 8u32);
        let pixel_count = (width * height) as usize;
        let original: Vec<u8> = (0..pixel_count).flat_map(|i| { let t = (i * 41 % 256) as u8; [t, t.wrapping_add(90), t.wrapping_add(30), 255] }).collect();
        let model_answer: Vec<u8> = (0..pixel_count).flat_map(|i| { let t = (i * 61 % 256) as u8; [t.wrapping_add(10), t, t.wrapping_add(180), 255] }).collect();
        let (colour_strength, transfer_strength, max_ratio) = (0.6, 0.9, 2.0);

        // The reference: `dispatch`'s already-verified CPU-visible path.
        let mut expected = model_answer.clone();
        assert!(gpu.dispatch(&device, &instance, physical_device, queue, width, height, &original, &mut expected, colour_strength, transfer_strength, max_ratio));

        // A standalone target image, standing in for a real swapchain image --
        // `dispatch_into_image`'s own contract only requires `TRANSFER_DST_OPTIMAL`,
        // which `capture::run`'s stage 1 already guarantees for the real one.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let target = create_storage_image(&device, &mem_props, width, height, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .expect("failed to create the test's own target image");

        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("failed to create the test's own command pool");
        let alloc_info = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.expect("failed to allocate the test's own command buffer")[0];
        let fence_info = vk::FenceCreateInfo::builder();
        let fence = unsafe { device.create_fence(&fence_info, None) }.expect("failed to create the test's own fence");

        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(cmd, &begin_info).unwrap();
            let to_dst = GpuCompose::image_barrier(target.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            device.end_command_buffer(cmd).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        }

        let mut model_answer_for_direct = model_answer.clone();
        let ok = gpu.dispatch_into_image(
            &device, &instance, physical_device, queue, width, height, &original, &mut model_answer_for_direct,
            colour_strength, transfer_strength, max_ratio, target.image,
        );
        assert!(ok, "dispatch_into_image returned false");

        // Read `target` back (it's `PRESENT_SRC_KHR` now, per the function's own
        // contract) purely to verify the test's own expectations -- production code
        // never needs to do this for the real swapchain image.
        let frame_bytes = (width * height * 4) as u64;
        let readback_buf_info = vk::BufferCreateInfo::builder().size(frame_bytes).usage(vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE);
        let readback_buffer = unsafe { device.create_buffer(&readback_buf_info, None) }.unwrap();
        let reqs = unsafe { device.get_buffer_memory_requirements(readback_buffer) };
        let type_index = find_memory_type(&mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();
        let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
        let readback_memory = unsafe { device.allocate_memory(&alloc, None) }.unwrap();
        unsafe { device.bind_buffer_memory(readback_buffer, readback_memory, 0) }.unwrap();
        let readback_ptr = unsafe { device.map_memory(readback_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.unwrap().cast::<u8>();

        unsafe {
            device.reset_fences(&[fence]).unwrap();
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            device.begin_command_buffer(cmd, &begin_info).unwrap();
            let to_src = GpuCompose::image_barrier(target.image, vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
            device.cmd_copy_image_to_buffer(cmd, target.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, readback_buffer, &[GpuCompose::image_copy_region(width, height, 0)]);
            device.end_command_buffer(cmd).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        }

        let actual = unsafe { std::slice::from_raw_parts(readback_ptr, frame_bytes as usize) };
        assert_eq!(actual, expected.as_slice(), "dispatch_into_image's target image content must match dispatch's CPU-visible result exactly (same command sequence, same inputs)");

        // SAFETY: the fence wait above guarantees no GPU work is in flight.
        unsafe {
            device.unmap_memory(readback_memory);
            device.free_memory(readback_memory, None);
            device.destroy_buffer(readback_buffer, None);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
            target.destroy(&device);
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

}
