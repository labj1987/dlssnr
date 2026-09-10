//! Milestone 4 phase A/B: capture the image `queue_present_khr` is about to present
//! into the shared-memory proxy region, run the round trip, and copy a result back
//! before the real present call.
//!
//! No `VK_EXT_external_memory_host` import yet -- every byte crosses an explicit CPU
//! `memcpy` between a host-visible/host-coherent staging buffer and the mapping
//! `ShmClient` owns. That is exactly the "staging copy" fallback
//! `crates/helper/src/shm.rs`'s own doc comment already describes as always-correct,
//! just not zero-copy; importing the mapping directly as device memory is a later
//! optimization on top of this, not a prerequisite for it working.
//!
//! One command buffer + one fence, reused every frame and waited on synchronously in
//! two stages (capture, then write-back) so the CPU can get at the bytes in between.
//! This blocks `queue_present_khr` for as long as the two copies (plus the round trip)
//! take -- correct, and the right first target given how much about the real
//! frame-time budget is still unknown, but not yet double-buffered/pipelined; that is
//! a follow-up once this path is proven to work at all.

use ash::vk;

use crate::shm::ShmClient;

pub struct CaptureResources {
    queue_family: u32,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    capacity: vk::DeviceSize,
}

// SAFETY: every field is either a plain Vulkan handle (as `Send`-safe as `ash::Device`
// itself already assumes) or `ptr`, a `vkMapMemory` pointer into memory this struct
// owns exclusively -- never aliased outside the `Mutex<State>` this always lives behind
// in `DlssnrDeviceInfo`.
unsafe impl Send for CaptureResources {}

impl CaptureResources {
    /// # Safety
    /// Must not be called while any submitted work referencing these handles might
    /// still be in flight -- callers only ever call this right after a successful
    /// `vkWaitForFences` on `self.fence`, or at device-destruction time.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Builds (or rebuilds, if the queue family changed or `bytes` grew past what's
/// already allocated) the resources capture needs. `existing` is left `None` on any
/// failure -- every caller treats that as "skip capture this frame, present
/// unmodified", never a reason to stop trying on a later frame.
fn ensure(
    existing: &mut Option<CaptureResources>,
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    bytes: vk::DeviceSize,
) -> bool {
    if let Some(r) = existing {
        if r.queue_family == queue_family && r.capacity >= bytes {
            return true;
        }
        // SAFETY: called between frames, never while `r.fence` might still be
        // unsignaled from an in-flight submission -- `queue_present_khr` only reaches
        // here after the previous frame's own capture fully completed.
        unsafe { r.destroy(device) };
        *existing = None;
    }

    let pool_info =
        vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    // SAFETY: `device` is the live device this capture serves; `pool_info` is valid.
    let Ok(pool) = (unsafe { device.create_command_pool(&pool_info, None) }) else { return false };

    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: `pool` was just created above.
    let cmd = match unsafe { device.allocate_command_buffers(&alloc_info) } {
        Ok(bufs) => bufs[0],
        Err(_) => {
            // SAFETY: `pool` owns no other resources yet.
            unsafe { device.destroy_command_pool(pool, None) };
            return false;
        }
    };

    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
    // SAFETY: starting signaled means the first frame's own wait (see `run` below)
    // never blocks on a fence nothing has submitted work against yet.
    let fence = match unsafe { device.create_fence(&fence_info, None) } {
        Ok(f) => f,
        Err(_) => {
            // SAFETY: `pool` owns no other resources yet; freeing it also frees `cmd`.
            unsafe { device.destroy_command_pool(pool, None) };
            return false;
        }
    };

    let buf_info = vk::BufferCreateInfo::builder()
        .size(bytes)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `buf_info` is valid.
    let buffer = match unsafe { device.create_buffer(&buf_info, None) } {
        Ok(b) => b,
        Err(_) => {
            // SAFETY: neither `fence` nor `pool` owns `buffer` (it doesn't exist).
            unsafe {
                device.destroy_fence(fence, None);
                device.destroy_command_pool(pool, None);
            }
            return false;
        }
    };
    // SAFETY: `buffer` was just created and is not yet bound to memory.
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    // SAFETY: `physical_device` is the device this capture serves; `instance` is its
    // owning instance (stored once at `vkCreateInstance`, see `crate::CURRENT_INSTANCE`).
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    // The Vulkan spec guarantees at least one memory type with both bits set, so this
    // failing would mean a spec-non-compliant driver, not a real device limitation --
    // still handled as a plain "skip capture" rather than assumed impossible.
    let Some(type_index) = (0..mem_props.memory_type_count)
        .find(|&i| (reqs.memory_type_bits & (1 << i)) != 0 && mem_props.memory_types[i as usize].property_flags.contains(wanted))
    else {
        // SAFETY: `buffer` has no memory bound yet; nothing else owns `fence`/`pool`.
        unsafe {
            device.destroy_buffer(buffer, None);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
        return false;
    };

    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
    // SAFETY: `alloc` is valid; `type_index` was just confirmed to satisfy `reqs`.
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(_) => {
            // SAFETY: same reasoning as the branch above.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.destroy_fence(fence, None);
                device.destroy_command_pool(pool, None);
            }
            return false;
        }
    };
    // SAFETY: `buffer`/`memory` were each just created above, sized/typed to satisfy
    // each other by construction.
    if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        // SAFETY: `memory` is not yet bound to anything that would make freeing it
        // unsound; `buffer` has no memory bound.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
        return false;
    }
    // SAFETY: `memory` is `HOST_VISIBLE` by the type selection above; mapping the
    // whole allocation is always in bounds.
    let ptr = match unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
        Ok(p) => p.cast::<u8>(),
        Err(_) => {
            // SAFETY: same reasoning as the bind-failure branch above.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
                device.destroy_fence(fence, None);
                device.destroy_command_pool(pool, None);
            }
            return false;
        }
    };

    *existing = Some(CaptureResources { queue_family, pool, cmd, fence, buffer, memory, ptr, capacity: reqs.size });
    true
}

fn subresource() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build()
}

fn barrier(image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src: vk::AccessFlags, dst: vk::AccessFlags) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource())
        .src_access_mask(src)
        .dst_access_mask(dst)
        .build()
}

/// Captures `image` into the proxy region, runs the shared-memory round trip, and
/// copies a result back into `image` before the caller's own present call. `resources`
/// is the per-device slot `queue_present_khr` owns (lazily built/rebuilt here).
///
/// Fails open on any error: returns without having touched `image` at all (still in
/// whatever layout the caller found it in, `PRESENT_SRC_KHR`) if anything along the way
/// doesn't work, so the caller can always fall back to presenting unmodified.
///
/// # Safety
/// `queue` must be the same queue `image`'s presentation was requested on, with no
/// concurrent use of it from another thread for the duration of this call (the same
/// external-synchronization requirement `vkQueuePresentKHR` itself already places on
/// its own `queue` argument, which is what makes submitting here, from inside the
/// present hook, sound without any additional locking).
pub unsafe fn run(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    image: vk::Image,
    width: u32,
    height: u32,
    proxy_format: u32,
    resources: &mut Option<CaptureResources>,
    gpu_compose: &mut Option<crate::composition::gpu::GpuCompose>,
    shm: &mut ShmClient,
) -> bool {
    let bytes_per_pixel = dlssnr_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
    let frame_bytes = u64::from(width) * u64::from(height) * bytes_per_pixel;
    if frame_bytes == 0 || frame_bytes as usize > dlssnr_protocol::MAX_FRAME {
        return false;
    }
    if !ensure(resources, device, instance, physical_device, queue_family, frame_bytes) {
        return false;
    }
    let r = resources.as_ref().expect("just ensured above");

    // Stage 1: image -> staging buffer.
    // SAFETY: `r.cmd` was allocated from `r.pool`, created with
    // `RESET_COMMAND_BUFFER`; resetting before every `begin_command_buffer` is exactly
    // what that flag exists to allow.
    if unsafe { device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
        return false;
    }
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `r.cmd` was just reset above.
    if unsafe { device.begin_command_buffer(r.cmd, &begin_info) }.is_err() {
        return false;
    }
    let to_transfer_src = barrier(
        image,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_READ,
    );
    // SAFETY: `r.cmd` is in the recording state; `image` is the caller's own,
    // currently-`PRESENT_SRC_KHR` swapchain image per `vkQueuePresentKHR`'s contract.
    unsafe {
        device.cmd_pipeline_barrier(
            r.cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer_src],
        );
    }
    let copy_out = vk::BufferImageCopy::builder()
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
        .build();
    // SAFETY: `image` was just transitioned to `TRANSFER_SRC_OPTIMAL` above; `r.buffer`
    // was sized to at least `frame_bytes` by `ensure`.
    unsafe {
        device.cmd_copy_image_to_buffer(r.cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, r.buffer, &[copy_out]);
    }
    let to_transfer_dst = barrier(
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::TRANSFER_WRITE,
    );
    // SAFETY: same reasoning as the first barrier above, transitioning for the
    // write-back this same command buffer will record in stage 2.
    unsafe {
        device.cmd_pipeline_barrier(
            r.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer_dst],
        );
    }
    if unsafe { device.end_command_buffer(r.cmd) }.is_err() {
        return false;
    }
    // SAFETY: `r.fence` starts signaled (see `ensure`) or was reset+waited-on by the
    // previous call to this function; `queue` is the caller's, externally synchronized
    // for the duration of this call per this function's own safety contract.
    if unsafe { device.reset_fences(&[r.fence]) }.is_err() {
        return false;
    }
    let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&r.cmd)).build();
    // SAFETY: `r.cmd` was just recorded and ended above.
    if unsafe { device.queue_submit(queue, &[submit], r.fence) }.is_err() {
        return false;
    }
    // SAFETY: `r.fence` was just submitted against above.
    if unsafe { device.wait_for_fences(&[r.fence], true, u64::MAX) }.is_err() {
        return false;
    }

    // CPU side: the captured bytes are now in `r.ptr` (host-coherent, no explicit
    // flush/invalidate needed). Hand them to the helper, then, if it actually
    // answered, overwrite `r.ptr` in place with that answer -- stage 2 below copies
    // whatever is sitting in `r.ptr` back into `image`, so this is what makes the
    // helper's answer (a real NGX evaluation, or the helper's own proxy-echo fallback
    // when the model isn't ready -- `dlssnr_helper::main`'s per-frame loop guarantees
    // the answer region is always the same size/format as the proxy either way)
    // actually reach the screen. A helper that never answers (not running, or the
    // round trip timed out) leaves `r.ptr` untouched -- it still holds the bytes
    // stage 1 just captured, so stage 2 below presents those unmodified, same fail-open
    // behavior as every other error path in this function.
    // SAFETY: `r.ptr` is a live mapping of at least `frame_bytes` bytes (the memory
    // type/size `ensure` just built or confirmed already satisfies this call's own
    // `frame_bytes`).
    let captured = unsafe { std::slice::from_raw_parts(r.ptr, frame_bytes as usize) };
    // Composition (below) needs the pre-edit frame after `read_answer` has already
    // overwritten `r.ptr` in place, so it has to be copied out now, before that
    // happens -- one extra `frame_bytes`-sized allocation/copy per frame, on top of
    // the two Vulkan transfers this function already does; not yet worth avoiding
    // ahead of proving the composition path correct at all.
    let original = captured.to_vec();
    shm.set_frame_info(width, height, proxy_format);
    shm.write_proxy(captured);
    let answered = shm.try_round_trip();
    // `true` only when `composition::gpu::GpuCompose::dispatch_into_image` already
    // wrote the fully composited result straight into `image` (and restored its
    // `PRESENT_SRC_KHR` layout) itself -- skips the capture_request dump (nothing
    // useful to dump: `r.ptr` still holds the *raw* answer, not the composited
    // result, on this path) and stage 2 (there is nothing left for it to do) below,
    // returning early instead. See `dispatch_into_image`'s own doc comment for why
    // this is worth a whole separate path rather than just "one fewer copy".
    let mut composed_directly = false;
    if answered {
        // SAFETY: same reasoning as the read above; `ShmClient::read_answer` never
        // writes past the slice's length, which is exactly `frame_bytes` here.
        let answer_dst = unsafe { std::slice::from_raw_parts_mut(r.ptr, frame_bytes as usize) };
        shm.read_answer(answer_dst);
        // Only `RGBA8` is handled -- `RGBA16F` still passes the helper's raw answer
        // through untouched (see `composition::apply`'s own doc comment for why, and
        // `dlssnr_protocol::enums::proxy_format` for the format codes).
        if proxy_format == dlssnr_protocol::enums::proxy_format::RGBA8 {
            if let Some(settings) = shm.composition_settings() {
                if settings.apply_model {
                    // GPU dispatch (`composition::gpu`) only implements the normal
                    // composited case (`compose.comp` has no concept of `debug_view`
                    // at all) -- fails open to the CPU reference
                    // (`composition::apply::apply_rgba8`, which every mode already
                    // handles) whenever the GPU path isn't applicable, isn't
                    // available, or fails, same fail-open discipline as every other
                    // stage in this function.
                    let mut composed = false;
                    if settings.debug_view == 0 {
                        if gpu_compose.is_none() {
                            *gpu_compose = crate::composition::gpu::GpuCompose::new(device, queue_family);
                        }
                        // Try the fast, no-CPU-round-trip path first -- but only when
                        // nothing on the CPU needs to see the result afterward. A
                        // pending `capture_request` does (its dump needs real bytes
                        // in `r.ptr`), so that specific, rare, deliberately-triggered
                        // case still goes through the slower CPU-visible `dispatch`
                        // below, same as before this path existed.
                        if !shm.capture_request_pending() {
                            if let Some(gpu) = gpu_compose {
                                composed_directly = gpu.dispatch_into_image(
                                    device,
                                    instance,
                                    physical_device,
                                    queue,
                                    width,
                                    height,
                                    &original,
                                    answer_dst,
                                    settings.colour_strength,
                                    settings.transfer_strength,
                                    settings.max_ratio,
                                    image,
                                );
                            }
                        }
                        if !composed_directly {
                            if let Some(gpu) = gpu_compose {
                                composed = gpu.dispatch(
                                    device,
                                    instance,
                                    physical_device,
                                    queue,
                                    width,
                                    height,
                                    &original,
                                    answer_dst,
                                    settings.colour_strength,
                                    settings.transfer_strength,
                                    settings.max_ratio,
                                );
                            }
                        }
                    }
                    if !composed_directly && !composed {
                        crate::composition::apply::apply_rgba8(
                            &original,
                            answer_dst,
                            settings.colour_strength,
                            settings.transfer_strength,
                            settings.max_ratio,
                            settings.debug_view,
                        );
                    }
                } else {
                    // "Off keeps the whole pass running... and simply presents the
                    // clean frame" -- ShmHeader::apply_model's own doc comment.
                    answer_dst.copy_from_slice(&original);
                }
            }
        }
    }
    crate::log!(
        "[capture] {}x{} {} bytes -> proxy; round trip answered={} composed_directly={}",
        width,
        height,
        frame_bytes,
        answered,
        composed_directly
    );
    if composed_directly {
        // `image` is already fully written and back in `PRESENT_SRC_KHR` --
        // `dispatch_into_image` did stage 2's whole job itself, in the same
        // submission as the compute dispatch. Nothing left to do this frame.
        return true;
    }

    // Real `ShmHeader::capture_request` support: dump this frame's original and
    // final (post-composition, if any ran above) bytes to disk. Checked regardless of
    // `answered`/`proxy_format` so a request during a fail-open frame still produces a
    // (identical) matched pair rather than silently doing nothing -- `write_pair`
    // itself is the only place that would need to special-case a format it can't
    // encode, and today it always gets `RGBA8` bytes either way.
    if shm.take_capture_request() && proxy_format == dlssnr_protocol::enums::proxy_format::RGBA8 {
        // SAFETY: same reasoning as every other read of `r.ptr` in this function --
        // still a live mapping of at least `frame_bytes` bytes, and stage 2 below
        // hasn't started overwriting it yet.
        let current = unsafe { std::slice::from_raw_parts(r.ptr, frame_bytes as usize) };
        crate::dump::write_pair(&original, current, width, height);
    }

    // Stage 2: staging buffer (now holding the answer, if there was one -- otherwise
    // still the captured bytes) -> image.
    // SAFETY: `r.cmd` was ended above; the pool it came from allows re-recording.
    if unsafe { device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
        return false;
    }
    // SAFETY: `r.cmd` was just reset.
    if unsafe { device.begin_command_buffer(r.cmd, &begin_info) }.is_err() {
        return false;
    }
    let copy_in = copy_out;
    // SAFETY: `image` is currently `TRANSFER_DST_OPTIMAL` from stage 1's own final
    // barrier; `r.buffer` (same host-coherent memory as `r.ptr`, which the CPU-side
    // block above may have just overwritten with the answer) holds exactly
    // `frame_bytes` valid bytes either way, matching `copy_in`'s own extent.
    unsafe {
        device.cmd_copy_buffer_to_image(r.cmd, r.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy_in]);
    }
    let to_present = barrier(
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::empty(),
    );
    // SAFETY: restores the layout `vkQueuePresentKHR` requires before the caller's own
    // (real) present call runs right after this function returns.
    unsafe {
        device.cmd_pipeline_barrier(
            r.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_present],
        );
    }
    if unsafe { device.end_command_buffer(r.cmd) }.is_err() {
        return false;
    }
    // SAFETY: same reasoning as stage 1's own fence reset/submit/wait.
    if unsafe { device.reset_fences(&[r.fence]) }.is_err() {
        return false;
    }
    let submit2 = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&r.cmd)).build();
    // SAFETY: `r.cmd` was just recorded and ended above.
    if unsafe { device.queue_submit(queue, &[submit2], r.fence) }.is_err() {
        return false;
    }
    // SAFETY: `r.fence` was just submitted against above. Waiting here (rather than
    // deferring to the next frame) keeps `image` fully write-back-complete and back in
    // `PRESENT_SRC_KHR` before this function returns, which is what the caller's own
    // immediately-following real present call requires.
    if unsafe { device.wait_for_fences(&[r.fence], true, u64::MAX) }.is_err() {
        return false;
    }

    true
}

/// # Safety
/// Must only be called at device-destruction time, with no submitted work referencing
/// these handles still in flight.
pub unsafe fn destroy(resources: Option<CaptureResources>, device: &ash::Device) {
    if let Some(r) = resources {
        // SAFETY: forwarded from this function's own contract.
        unsafe { r.destroy(device) };
    }
}
