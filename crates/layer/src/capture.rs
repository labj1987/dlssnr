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
//! Stage 1 (capture into a staging buffer) is one command buffer + one fence,
//! synchronous -- the CPU needs those bytes before it can even start the SHM round
//! trip, so there's no way around blocking on it. What happens after the round trip
//! depends on the settings and what's available: the common case (real GPU compose,
//! no debug dump pending) is `composition::gpu::GpuCompose::dispatch_into_image_async`
//! (2026-09-10) -- non-blocking, its own doc comment covers why that's sound. Every
//! other case (CPU compose, a pending `capture_request`, `RGBA16F`, no GPU available)
//! still falls back to the original synchronous stage-2 write-back below, one more
//! command buffer + fence wait, same as this whole function used to always do.

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

/// What [`run`] is carrying forward from the round trip it most recently *sent*,
/// across as many present calls as the helper takes to answer it. `original` holds
/// the exact pixels captured at send time -- needed again once the answer finally
/// arrives, since composition combines the two -- alongside the dimensions/format
/// that capture was taken at, so a resolution change mid-flight is detected (and the
/// stale pair discarded) rather than composited against a mismatched frame size.
#[derive(Default)]
pub struct Inflight {
    original: Vec<u8>,
    dims: Option<(u32, u32, u32)>,
}

/// Real per-frame NR compute (a helper round trip through a Wine-hosted process, plus
/// whatever GPU work either side does) does not run at anywhere close to swapchain
/// present rate -- measured on real hardware (`lordnikon`, 2026-09-10, see
/// `CLAUDE.md`) at roughly 100-150ms end to end even once every other bottleneck
/// found that same session was fixed. [`run_sync`] (this crate's entire capture path
/// before this) called that round trip, and blocked waiting for it, from *inside*
/// every single present call -- meaning the game's own presentation rate could never
/// exceed the round trip's, even though the actual GPU compute involved is only a
/// few milliseconds. That coupling, not any single slow operation, was the real
/// cause of a reported ~2.8 fps at 4K with NR on, confirmed by removing this
/// project's layer entirely and watching the same game return to 99% GPU utilization
/// and a normal framerate.
///
/// This function decouples the two: it captures and sends a new frame only when no
/// round trip is currently in flight, checks on any in-flight one *without blocking*
/// (see [`ShmClient::poll_async_request`]), and applies whatever answer arrives to
/// whichever frame happens to be current at that moment -- not necessarily the one
/// that was captured alongside it. Every other frame (which, once the pipeline is
/// running, is most of them) touches `image` not at all and returns `None`
/// immediately, at effectively zero cost. The tradeoff this accepts, deliberately,
/// per Alex's own explicit authorization ("do it if it gives us the most frames when
/// NR is on"): the visible NR enhancement updates at whatever rate the round trip
/// actually achieves, not every frame, and is very occasionally composited against a
/// slightly newer frame than the one it was computed from (a few frames of temporal
/// staleness at most, bounded by the round trip's own duration) -- a real quality
/// tradeoff, not a free lunch, but one that keeps the game's own rendering and
/// presentation running at its true native rate instead of being held hostage by a
/// cross-process IPC round trip on every single frame.
///
/// Ordering inside a single call matters and is deliberate: capturing a new frame
/// (when due) always happens *before* compositing an answer that arrived this same
/// frame, because compositing overwrites `image` -- capturing after that would
/// capture this function's own composited output instead of the game's real
/// rendering, feeding a corrupted "original" into the next cycle.
///
/// # Safety
/// Same contract as [`run_sync`]: `queue` must be the same queue `image`'s
/// presentation was requested on, with no concurrent use of it from another thread
/// for the duration of this call.
#[allow(clippy::too_many_arguments)]
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
    bgr_order: bool,
    resources: &mut Option<CaptureResources>,
    gpu_compose: &mut Option<crate::composition::gpu::GpuCompose>,
    shm: &mut ShmClient,
    original_scratch: &mut Vec<u8>,
    inflight: &mut Inflight,
    answer_scratch: &mut Vec<u8>,
    last_answer: &mut Vec<u8>,
) -> Option<vk::Semaphore> {
    // `composition_settings()` (and everything else below) only ever reads through an
    // already-open mapping -- nothing about it opens one. Every real path that DOES
    // open the mapping (`try_round_trip`/`begin_async_request`) lives later in this
    // same function, gated behind the `composition_settings()` check right below.
    // Real bug, found and fixed 2026-09-11 via a live `vkcube` bisection on
    // `lordnikon`: on a brand-new process the mapping is never open yet, so this used
    // to return `None` here on literally every single frame, forever -- this function
    // was being called every present call (confirmed real, not theoretical) but never
    // actually captured or sent a single frame, because it always bailed out before
    // ever reaching the code that would open the mapping in the first place.
    // `ShmClient::open` is cheap to call unconditionally (an immediate no-op once
    // already open, see its own early return), so there's no real cost to calling it
    // here up front instead of leaving each caller to remember to.
    shm.open();
    let Some(settings) = shm.composition_settings() else { return None };
    // `debug_view`'s compare/split views and a pending `capture_request`'s dump both
    // need *this* frame's own original and answer, not whatever the async pipeline
    // below happens to have on hand -- same-frame correctness matters more than
    // throughput for either, and both are rare, deliberately-triggered cases (a
    // developer toggling a debug view, or a one-shot dump request), not the normal
    // per-frame path this function otherwise replaces.
    if settings.debug_view != 0 || shm.capture_request_pending() {
        return unsafe {
            run_sync(
                device,
                instance,
                physical_device,
                queue,
                queue_family,
                image,
                width,
                height,
                proxy_format,
                bgr_order,
                resources,
                gpu_compose,
                shm,
                original_scratch,
                last_answer,
            )
        };
    }
    // "Off keeps the whole pass running... and simply presents the clean frame" --
    // `ShmHeader::apply_model`'s own doc comment -- and the model being permanently
    // unavailable is the same "nothing will ever consume a captured frame" case
    // `ShmClient::model_known_unavailable`'s own doc comment already covers. Either
    // way, paying for a capture+round-trip cycle nobody will use is pure waste;
    // skip the whole pipeline and let the caller present `image` untouched.
    // `neural_enabled` (the GUI's own "Enabled" toggle, `ShmHeader::enabled`) is
    // included here too -- a real bug, found 2026-09-11: `ShmHeader::neural_enabled()`
    // existed and the GUI wrote to it, but nothing in this crate ever read it back,
    // so turning "Enabled" off in the GUI had no effect on anything real at all.
    if !settings.apply_model || !settings.neural_enabled || shm.model_known_unavailable() {
        return None;
    }

    let bytes_per_pixel = dlssnr_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
    let frame_bytes = u64::from(width) * u64::from(height) * bytes_per_pixel;
    if frame_bytes == 0 || frame_bytes as usize > dlssnr_protocol::MAX_FRAME {
        return None;
    }

    // Poll whatever was sent on some earlier frame *before* touching anything else --
    // `inflight`'s current contents correspond to it, and must be read (below) before
    // a new capture this same frame (if one happens) is allowed to replace them.
    let mut have_answer = false;
    if shm.has_pending_request() {
        if shm.poll_async_request() == Some(true) {
            answer_scratch.resize(frame_bytes as usize, 0);
            shm.read_answer(answer_scratch);
            have_answer = true;
        }
    }

    // Capture and send a new frame if (and only if) nothing is currently in flight --
    // the wire protocol has only ever supported one outstanding request at a time.
    // Deliberately *before* compositing below: compositing overwrites `image`, and
    // this capture needs the game's real, unmodified rendering for this frame, not
    // whatever this same call is about to paint over it.
    if !shm.has_pending_request() {
        if !ensure(resources, device, instance, physical_device, queue_family, frame_bytes) {
            return None;
        }
        let r = resources.as_ref().expect("just ensured above");
        if capture_pristine(device, r, queue, image, width, height, frame_bytes, original_scratch) {
            shm.set_frame_info(width, height, proxy_format);
            shm.write_proxy(original_scratch);
            shm.prepare_motion(instance, physical_device, width, height, proxy_format, original_scratch);
            if shm.begin_async_request() {
                std::mem::swap(&mut inflight.original, original_scratch);
                inflight.dims = Some((width, height, proxy_format));
            }
        }
    }

    if !have_answer {
        return None;
    }
    // A resolution (or format) change between when `inflight` was captured and now
    // means its bytes describe a differently-sized frame -- compositing them against
    // `image` at today's dimensions would read/write out of step with reality.
    // Discard rather than risk it; `inflight` gets overwritten by the next successful
    // capture above regardless.
    if inflight.dims != Some((width, height, proxy_format)) {
        return None;
    }
    if !dlssnr_protocol::enums::proxy_format::is_8bit(proxy_format) {
        // `RGBA16F` has no composition path at all yet (see `composition::apply`'s own
        // doc comment) -- nothing to do with a fresh answer for it here.
        return None;
    }

    if gpu_compose.is_none() {
        *gpu_compose = crate::composition::gpu::GpuCompose::new(device, queue_family);
    }
    if let Some(gpu) = gpu_compose {
        if let Some(sem) = gpu.dispatch_into_image_async(
            device,
            instance,
            physical_device,
            queue,
            width,
            height,
            &inflight.original,
            answer_scratch,
            settings.colour_strength,
            settings.transfer_strength,
            settings.max_ratio,
            bgr_order,
            image,
        ) {
            return Some(sem);
        }
        // Async slot busy or failed -- fall back to the same dispatch, synchronously,
        // still cheaper and simpler than standing up a whole second (buffer-mediated,
        // CPU-visible) write-back path for what should be a rare case.
        if gpu.dispatch_into_image(
            device,
            instance,
            physical_device,
            queue,
            width,
            height,
            &inflight.original,
            answer_scratch,
            settings.colour_strength,
            settings.transfer_strength,
            settings.max_ratio,
            bgr_order,
            image,
        ) {
            return None;
        }
    }
    // No GPU compose available at all (`GpuCompose::new` failed) -- last resort: the
    // CPU reference implementation, then a plain buffer-mediated write-back using the
    // same `CaptureResources` staging buffer `capture_pristine` above already ensured
    // exists.
    crate::composition::apply::apply_rgba8(
        &inflight.original,
        answer_scratch,
        settings.colour_strength,
        settings.transfer_strength,
        settings.max_ratio,
        0,
        bgr_order,
    );
    if !ensure(resources, device, instance, physical_device, queue_family, frame_bytes) {
        return None;
    }
    let r = resources.as_ref().expect("just ensured above");
    write_bytes_to_image(device, r, queue, image, width, height, answer_scratch);
    None
}

/// Reads `image` (assumed `PRESENT_SRC_KHR`, exactly what any image
/// `vkQueuePresentKHR`'s own contract hasn't already been violated on satisfies) into
/// `out`, restoring `image` to `PRESENT_SRC_KHR` before returning -- a self-contained
/// "read pixels, leave everything as I found it" operation, deliberately not sharing
/// [`run_sync`]'s stage-1 barrier sequence (which ends in `TRANSFER_DST_OPTIMAL`,
/// correct only when a stage 2 write-back on the very same image immediately
/// follows). [`run`] calls this for a capture that will send its bytes off for
/// evaluation and not touch `image` again until (if ever) a composited answer for a
/// *different*, later frame arrives.
///
/// Fully synchronous (submits and waits) -- real, measured cost on `lordnikon`
/// (2026-09-10) is only a few milliseconds, and it now runs once per round-trip
/// cycle rather than once per frame, not on the hot path this exists to unblock.
///
/// `false` on any failure, leaving `out` unchanged and `image` in whatever layout the
/// failure happened in -- callers already fail open on this exactly like every other
/// stage in this module.
#[allow(clippy::too_many_arguments)]
fn capture_pristine(
    device: &ash::Device,
    r: &CaptureResources,
    queue: vk::Queue,
    image: vk::Image,
    width: u32,
    height: u32,
    frame_bytes: u64,
    out: &mut Vec<u8>,
) -> bool {
    // SAFETY: `r.cmd` was allocated from `r.pool`, created with
    // `RESET_COMMAND_BUFFER`.
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
    let region = vk::BufferImageCopy::builder()
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
        device.cmd_copy_image_to_buffer(r.cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, r.buffer, &[region]);
    }
    // Restore `image` to exactly the layout this function found it in -- unlike
    // `run_sync`'s stage 1, nothing is guaranteed to touch `image` again this same
    // frame, so leaving it in `TRANSFER_DST_OPTIMAL` (a layout only valid mid-way
    // through an image<->buffer round trip) would be a real bug the moment the real
    // present call ran against it instead.
    let to_present = barrier(
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::empty(),
    );
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
    // SAFETY: `r.fence` starts signaled (see `ensure`) or was reset+waited-on by
    // whichever of `capture_pristine`/`run_sync` last used it.
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
    // SAFETY: `r.ptr` is a live mapping of at least `frame_bytes` bytes (the memory
    // type/size `ensure` just built or confirmed already satisfies this call's own
    // `frame_bytes`).
    let captured = unsafe { std::slice::from_raw_parts(r.ptr, frame_bytes as usize) };
    out.clear();
    out.extend_from_slice(captured);
    true
}

/// Writes `bytes` (exactly `width*height*4` `RGBA8` bytes) into `image` via
/// `r`'s own staging buffer -- the CPU-composited last resort when no GPU compose
/// path is available at all. Fully synchronous; `image` assumed/left `PRESENT_SRC_KHR`
/// exactly like [`capture_pristine`]. Best-effort: does nothing observable on failure
/// beyond leaving `image` unpresented-to this frame, same fail-open discipline as
/// every other stage in this module.
fn write_bytes_to_image(device: &ash::Device, r: &CaptureResources, queue: vk::Queue, image: vk::Image, width: u32, height: u32, bytes: &[u8]) {
    let frame_bytes = u64::from(width) * u64::from(height) * 4;
    if bytes.len() as u64 != frame_bytes {
        return;
    }
    // SAFETY: `r.ptr` is a live mapping of at least `frame_bytes` bytes -- the same
    // invariant `capture_pristine`/`run_sync` already rely on.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), r.ptr, bytes.len()) };
    // SAFETY: `r.cmd` was allocated with `RESET_COMMAND_BUFFER`.
    if unsafe { device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
        return;
    }
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    if unsafe { device.begin_command_buffer(r.cmd, &begin_info) }.is_err() {
        return;
    }
    let to_dst = barrier(
        image,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
    );
    unsafe {
        device.cmd_pipeline_barrier(r.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
    }
    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(0).layer_count(1).build())
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D { width, height, depth: 1 })
        .build();
    unsafe {
        device.cmd_copy_buffer_to_image(r.cmd, r.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
    }
    let to_present = barrier(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());
    unsafe {
        device.cmd_pipeline_barrier(r.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
    }
    if unsafe { device.end_command_buffer(r.cmd) }.is_err() {
        return;
    }
    if unsafe { device.reset_fences(&[r.fence]) }.is_err() {
        return;
    }
    let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&r.cmd)).build();
    if unsafe { device.queue_submit(queue, &[submit], r.fence) }.is_err() {
        return;
    }
    let _ = unsafe { device.wait_for_fences(&[r.fence], true, u64::MAX) };
}

/// Captures `image` into the proxy region, runs the shared-memory round trip, and
/// copies a result back into `image` before the caller's own present call. `resources`
/// is the per-device slot `queue_present_khr` owns (lazily built/rebuilt here).
///
/// Fails open on any error: returns without having touched `image` at all (still in
/// whatever layout the caller found it in, `PRESENT_SRC_KHR`) if anything along the way
/// doesn't work, so the caller can always fall back to presenting unmodified.
///
/// Returns `Some(semaphore)` when (and only when)
/// `composition::gpu::GpuCompose::dispatch_into_image_async` was used: `image` is
/// already fully written with the composited result, but the GPU work that wrote it
/// is not guaranteed *complete* yet (that is the entire point of the "async" in its
/// name -- this function never blocks on it). The caller **must** add that semaphore
/// to the real present call's own wait-semaphore list before presenting `image` --
/// otherwise the presentation engine could display `image` before the compute work
/// finishes writing it, a real, visible corruption/tearing bug, not merely a style
/// preference. `None` in every other case means `image` is already fully complete and
/// correctly laid out (`PRESENT_SRC_KHR`) -- safe to present with no extra wait.
///
/// # Safety
/// `queue` must be the same queue `image`'s presentation was requested on, with no
/// concurrent use of it from another thread for the duration of this call (the same
/// external-synchronization requirement `vkQueuePresentKHR` itself already places on
/// its own `queue` argument, which is what makes submitting here, from inside the
/// present hook, sound without any additional locking).
/// The old, fully-synchronous, one-frame-at-a-time path: capture *this* frame, block
/// on the helper round trip for *this* frame's own answer (up to a real timeout
/// budget), composite, write back -- all within the same present call. Kept
/// unchanged and still used for the two cases that genuinely need same-frame
/// correctness: a pending `capture_request` (its dump must show *this* frame's real
/// before/after, not some other frame's) and any non-zero `debug_view` (the
/// compare/split views are meaningless if original and answer come from different
/// moments). See [`run`]'s own doc comment for why every other case no longer goes
/// through here.
#[allow(clippy::too_many_arguments)]
unsafe fn run_sync(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    image: vk::Image,
    width: u32,
    height: u32,
    proxy_format: u32,
    bgr_order: bool,
    resources: &mut Option<CaptureResources>,
    gpu_compose: &mut Option<crate::composition::gpu::GpuCompose>,
    shm: &mut ShmClient,
    original_scratch: &mut Vec<u8>,
    last_answer: &mut Vec<u8>,
) -> Option<vk::Semaphore> {
    let bytes_per_pixel = dlssnr_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
    let frame_bytes = u64::from(width) * u64::from(height) * bytes_per_pixel;
    if frame_bytes == 0 || frame_bytes as usize > dlssnr_protocol::MAX_FRAME {
        return None;
    }
    if !ensure(resources, device, instance, physical_device, queue_family, frame_bytes) {
        return None;
    }
    let r = resources.as_ref().expect("just ensured above");

    // Stage 1: image -> staging buffer.
    // SAFETY: `r.cmd` was allocated from `r.pool`, created with
    // `RESET_COMMAND_BUFFER`; resetting before every `begin_command_buffer` is exactly
    // what that flag exists to allow.
    if unsafe { device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
        return None;
    }
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `r.cmd` was just reset above.
    if unsafe { device.begin_command_buffer(r.cmd, &begin_info) }.is_err() {
        return None;
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
        return None;
    }
    // SAFETY: `r.fence` starts signaled (see `ensure`) or was reset+waited-on by the
    // previous call to this function; `queue` is the caller's, externally synchronized
    // for the duration of this call per this function's own safety contract.
    if unsafe { device.reset_fences(&[r.fence]) }.is_err() {
        return None;
    }
    let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&r.cmd)).build();
    let t_stage1_start = std::time::Instant::now();
    // SAFETY: `r.cmd` was just recorded and ended above.
    if unsafe { device.queue_submit(queue, &[submit], r.fence) }.is_err() {
        return None;
    }
    // SAFETY: `r.fence` was just submitted against above.
    if unsafe { device.wait_for_fences(&[r.fence], true, u64::MAX) }.is_err() {
        return None;
    }
    let t_stage1 = t_stage1_start.elapsed();

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
    let t_snapshot_start = std::time::Instant::now();
    original_scratch.clear();
    original_scratch.extend_from_slice(captured);
    let original: &[u8] = original_scratch.as_slice();
    let t_snapshot = t_snapshot_start.elapsed();
    let t_write_proxy_start = std::time::Instant::now();
    shm.set_frame_info(width, height, proxy_format);
    shm.write_proxy(captured);
    shm.prepare_motion(instance, physical_device, width, height, proxy_format, captured);
    let t_write_proxy = t_write_proxy_start.elapsed();
    let t_roundtrip_start = std::time::Instant::now();
    let answered = shm.try_round_trip();
    let t_roundtrip = t_roundtrip_start.elapsed();
    let t_compose_start = std::time::Instant::now();
    // `Some(sem)` only when `composition::gpu::GpuCompose::dispatch_into_image_async`
    // already wrote the fully composited result straight into `image` itself, on the
    // GPU's own timeline -- skips the capture_request dump (nothing useful to dump:
    // `r.ptr` still holds the *raw* answer, not the composited result, on this path)
    // and stage 2 (there is nothing left for it to do) below, returning early instead.
    // The caller (`device.rs`) must chain `sem` into the real present call -- see this
    // function's own doc comment and `dispatch_into_image_async`'s for why.
    let mut composed_async: Option<vk::Semaphore> = None;
    if answered {
        // SAFETY: same reasoning as the read above; `ShmClient::read_answer` never
        // writes past the slice's length, which is exactly `frame_bytes` here.
        let answer_dst = unsafe { std::slice::from_raw_parts_mut(r.ptr, frame_bytes as usize) };
        shm.read_answer(answer_dst);
        last_answer.clear();
        last_answer.extend_from_slice(answer_dst);
        // Only `RGBA8` is handled -- `RGBA16F` still passes the helper's raw answer
        // through untouched (see `composition::apply`'s own doc comment for why, and
        // `dlssnr_protocol::enums::proxy_format` for the format codes).
        if dlssnr_protocol::enums::proxy_format::is_8bit(proxy_format) {
            if let Some(settings) = shm.composition_settings() {
                if settings.apply_model && settings.neural_enabled {
                    // GPU dispatch (`composition::gpu`) only implements the normal
                    // composited case (`compose.comp` has no concept of `debug_view`
                    // at all) -- fails open to the CPU reference
                    // (`composition::apply::apply_rgba8`, which every mode already
                    // handles) whenever the GPU path isn't applicable, isn't
                    // available, or fails, same fail-open discipline as every other
                    // stage in this function.
                    let mut composed_sync = false;
                    if settings.debug_view == 0 {
                        if gpu_compose.is_none() {
                            *gpu_compose = crate::composition::gpu::GpuCompose::new(device, queue_family);
                        }
                        // Try the fast, non-blocking path first -- but only when
                        // nothing on the CPU needs to see the result afterward. A
                        // pending `capture_request` does (its dump needs real bytes
                        // in `r.ptr`), so that specific, rare, deliberately-triggered
                        // case still goes through the slower, fully-synchronous
                        // CPU-visible `dispatch` below, same as before this path
                        // existed.
                        // The first neural frame after enabling the feature can
                        // race the application's present transition on NVIDIA
                        // drivers. Keep composition CPU-visible until the
                        // async handoff is proven safe for live games.
                        if false && !shm.capture_request_pending() {
                            if let Some(gpu) = gpu_compose {
                                composed_async = gpu.dispatch_into_image_async(
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
                                    bgr_order,
                                    image,
                                );
                            }
                        }
                        if composed_async.is_none() {
                            if let Some(gpu) = gpu_compose {
                                composed_sync = gpu.dispatch(
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
                                    bgr_order,
                                );
                            }
                        }
                    }
                    if composed_async.is_none() && !composed_sync {
                        crate::composition::apply::apply_rgba8(
                            &original,
                            answer_dst,
                            settings.colour_strength,
                            settings.transfer_strength,
                            settings.max_ratio,
                            settings.debug_view,
                            bgr_order,
                        );
                    }
                } else {
                    // "Off keeps the whole pass running... and simply presents the
                    // clean frame" -- ShmHeader::apply_model's own doc comment.
                    answer_dst.copy_from_slice(&original);
                }
            }
        }
    } else if !last_answer.is_empty() && last_answer.len() == frame_bytes as usize {
        // Keep the presentation mode stable while the asynchronous helper is
        // processing the next frame. Reusing the last model answer prevents an
        // untouched original frame from flashing between composited frames.
        let answer_dst = unsafe { std::slice::from_raw_parts_mut(r.ptr, frame_bytes as usize) };
        answer_dst.copy_from_slice(last_answer);
        if let Some(settings) = shm.composition_settings() {
            if settings.apply_model && settings.neural_enabled && dlssnr_protocol::enums::proxy_format::is_8bit(proxy_format) {
                crate::composition::apply::apply_rgba8(&original, answer_dst, settings.colour_strength,
                    settings.transfer_strength, settings.max_ratio, settings.debug_view, bgr_order);
            }
        }
    }
    crate::log!(
        "[capture] {}x{} {} bytes -> proxy; round trip answered={} composed_async={}",
        width,
        height,
        frame_bytes,
        answered,
        composed_async.is_some()
    );
    if let Some(sem) = composed_async {
        // `image` is already fully written (on the GPU's own timeline -- not
        // necessarily *complete* yet, that's the entire point) and back in
        // `PRESENT_SRC_KHR`. Nothing left to do this frame except hand `sem` up to
        // the caller so the real present call waits on it.
        crate::log!(
            "[capture] timing stage1={:?} snapshot={:?} write_proxy={:?} roundtrip={:?} compose(async-dispatch-only)={:?} stage2=skipped total={:?}",
            t_stage1,
            t_snapshot,
            t_write_proxy,
            t_roundtrip,
            t_compose_start.elapsed(),
            t_stage1_start.elapsed(),
        );
        return Some(sem);
    }

    // Real `ShmHeader::capture_request` support: dump this frame's original and
    // final (post-composition, if any ran above) bytes to disk. Checked regardless of
    // `answered`/`proxy_format` so a request during a fail-open frame still produces a
    // (identical) matched pair rather than silently doing nothing -- `write_pair`
    // itself is the only place that would need to special-case a format it can't
    // encode, and today it always gets `RGBA8` bytes either way.
    if shm.take_capture_request() && dlssnr_protocol::enums::proxy_format::is_8bit(proxy_format) {
        // SAFETY: same reasoning as every other read of `r.ptr` in this function --
        // still a live mapping of at least `frame_bytes` bytes, and stage 2 below
        // hasn't started overwriting it yet.
        let current = unsafe { std::slice::from_raw_parts(r.ptr, frame_bytes as usize) };
        crate::dump::write_pair(&original, current, width, height, bgr_order);
    }

    // Stage 2: staging buffer (now holding the answer, if there was one -- otherwise
    // still the captured bytes) -> image.
    // SAFETY: `r.cmd` was ended above; the pool it came from allows re-recording.
    if unsafe { device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
        return None;
    }
    // SAFETY: `r.cmd` was just reset.
    if unsafe { device.begin_command_buffer(r.cmd, &begin_info) }.is_err() {
        return None;
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
        return None;
    }
    // SAFETY: same reasoning as stage 1's own fence reset/submit/wait.
    if unsafe { device.reset_fences(&[r.fence]) }.is_err() {
        return None;
    }
    let submit2 = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&r.cmd)).build();
    let t_stage2_start = std::time::Instant::now();
    // SAFETY: `r.cmd` was just recorded and ended above.
    if unsafe { device.queue_submit(queue, &[submit2], r.fence) }.is_err() {
        return None;
    }
    // SAFETY: `r.fence` was just submitted against above. Waiting here (rather than
    // deferring to the next frame) keeps `image` fully write-back-complete and back in
    // `PRESENT_SRC_KHR` before this function returns, which is what the caller's own
    // immediately-following real present call requires.
    if unsafe { device.wait_for_fences(&[r.fence], true, u64::MAX) }.is_err() {
        return None;
    }
    let t_stage2 = t_stage2_start.elapsed();
    crate::log!(
        "[capture] timing stage1={:?} snapshot={:?} write_proxy={:?} roundtrip={:?} compose={:?} stage2={:?} total={:?}",
        t_stage1,
        t_snapshot,
        t_write_proxy,
        t_roundtrip,
        // `t_compose_start` was captured right after the round trip; `t_stage2_start`
        // right before stage 2's own submit -- the gap between them is exactly the
        // composition work (CPU reference or GPU dispatch), with no double-counting
        // against `t_stage2` below.
        t_stage2_start.duration_since(t_compose_start),
        t_stage2,
        t_stage1_start.elapsed(),
    );

    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Same shape as `composition::gpu::tests::test_device` -- a real (if software)
    /// Vulkan device via whatever loader/ICD is on this machine, `None` if there
    /// isn't one. Not shared with that module (private to it, and this crate has no
    /// shared test-support module yet); small enough that duplicating it costs less
    /// than inventing one.
    fn test_device() -> Option<(ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue, u32)> {
        // SAFETY: loads the system Vulkan loader; the usual caveats of loading an
        // arbitrary shared library apply and are accepted here the same way every
        // other `ash` consumer in this crate already does.
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
        // SAFETY: `device_create_info` is valid; every physical device has a family 0.
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.ok()?;
        // SAFETY: `device`/family/index 0 match what `device_create_info` just requested.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        Some((entry, instance, physical_device, device, queue, queue_family))
    }

    /// A standalone image standing in for a real swapchain image, already in
    /// `PRESENT_SRC_KHR` -- what `run`'s own contract requires of `image` on entry,
    /// same as any image `vkQueuePresentKHR`'s own precondition hasn't been violated
    /// on.
    fn make_present_src_image(device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, queue: vk::Queue, pool: vk::CommandPool, width: u32, height: u32) -> (vk::Image, vk::DeviceMemory) {
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None) }.expect("failed to create the test's own target image");
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let type_index = (0..mem_props.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && mem_props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
            .expect("no suitable memory type for the test's own target image");
        let memory = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None) }.unwrap();
        unsafe { device.bind_image_memory(image, memory, 0) }.unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(cmd, &begin_info).unwrap();
            let to_present = barrier(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::empty(), vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
            device.end_command_buffer(cmd).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            device.destroy_fence(fence, None);
            device.free_command_buffers(pool, &[cmd]);
        }
        (image, memory)
    }

    fn scratch_path(tag: &str) -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        format!("{}/dlssnr-capture-test-{}-{tag}-{n}/shm.bin", std::env::temp_dir().display(), std::process::id())
    }

    /// The real point of the pipelined redesign, exercised end to end against a real
    /// (if software) Vulkan device: `run` must never block a present call waiting on
    /// the helper, even when the helper genuinely takes far longer than one frame to
    /// answer -- and once it does answer, the result must actually reach `image` via
    /// a real, verifiable composited write (not just "a semaphore came back").
    #[test]
    fn run_never_blocks_on_a_slow_helper_and_eventually_composites() {
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("run_never_blocks_on_a_slow_helper_and_eventually_composites: no Vulkan loader/ICD, skipping");
            return;
        };

        let path = scratch_path("blocks");
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path), "test-only open_at should always succeed against a scratch path");
        let hdr_ptr = shm.test_header_ptr();
        // A live helper (matters for `poll_async_request`'s timeout budget: the long
        // "steady state" one, not the short "nobody's listening" one, since this test
        // deliberately answers slower than that short budget).
        unsafe { &*(hdr_ptr as *mut dlssnr_protocol::ShmHeader) }.helper_state.store(dlssnr_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);

        // A fake helper that only answers `HELPER_DELAY` after it sees a new request --
        // long enough that if `run` ever blocked waiting for it, a handful of calls
        // spaced much closer together than that would visibly take just as long.
        const HELPER_DELAY: Duration = Duration::from_millis(250);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let helper = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends).
            let hdr = unsafe { &*(hdr_ptr as *mut dlssnr_protocol::ShmHeader) };
            let mut last_seen = 0u32;
            while !stop_clone.load(AtomicOrdering::Relaxed) {
                let req = hdr.seq_req.load(AtomicOrdering::Relaxed);
                if req != 0 && req != last_seen {
                    last_seen = req;
                    std::thread::sleep(HELPER_DELAY);
                    hdr.seq_resp.store(req, AtomicOrdering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let (width, height) = (8u32, 8u32);
        let proxy_format = dlssnr_protocol::enums::proxy_format::RGBA8;
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("failed to create the test's own command pool");
        let (image, image_memory) = make_present_src_image(&device, &mem_props, queue, pool, width, height);

        let mut resources: Option<CaptureResources> = None;
        let mut gpu_compose: Option<crate::composition::gpu::GpuCompose> = None;
        let mut original_scratch = Vec::new();
        let mut answer_scratch = Vec::new();
        let mut last_answer = Vec::new();
        let mut inflight = Inflight::default();

        let mut got_semaphore = false;
        let deadline = Instant::now() + Duration::from_secs(5);
        // Real usage calls this once per present, indefinitely -- loop until either a
        // real composited result shows up or the deadline (comfortably several
        // `HELPER_DELAY`-long round trips) is exhausted, not a fixed iteration count,
        // so this can't spuriously fail just because a scratch VM's first Vulkan call
        // of the test happened to be slow.
        while Instant::now() < deadline {
            let call_start = Instant::now();
            // SAFETY: `image` is this test's own, currently `PRESENT_SRC_KHR`; `queue`
            // is used from this one thread only, exactly like `run`'s own contract
            // requires of the real present hook.
            let sem = unsafe {
                run(
                    &device,
                    &instance,
                    physical_device,
                    queue,
                    queue_family,
                    image,
                    width,
                    height,
                    proxy_format,
                    false,
                    &mut resources,
                    &mut gpu_compose,
                    &mut shm,
                    &mut original_scratch,
                    &mut inflight,
                    &mut answer_scratch,
                    &mut last_answer,
                )
            };
            let call_time = call_start.elapsed();
            assert!(
                call_time < HELPER_DELAY / 2,
                "a single run() call took {call_time:?} -- must never approach the helper's own {HELPER_DELAY:?} answer delay"
            );
            if let Some(sem) = sem {
                got_semaphore = true;
                // Stand in for what the real present call does: wait on the semaphore
                // before the image is considered final, exactly like
                // `composition::gpu::tests`' own async tests already establish.
                let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
                let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
                let submit = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&wait_stage)).build();
                unsafe {
                    device.queue_submit(queue, &[submit], wait_fence).unwrap();
                    device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
                    device.destroy_fence(wait_fence, None);
                }
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(got_semaphore, "the pipeline must eventually composite a real answer within 5s of real time, not just avoid blocking forever");

        stop.store(true, AtomicOrdering::Relaxed);
        helper.join().unwrap();

        // SAFETY: every semaphore this test waited on has a completed, waited-for
        // fence behind it (the explicit wait above); nothing else touched `image`.
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            destroy(resources, &device);
            if let Some(gpu) = gpu_compose {
                gpu.destroy(&device);
            }
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }
}
