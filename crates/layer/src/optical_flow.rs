//! Native optical flow between consecutive captured (model-input) frames.
//! A private device on the game's physical GPU owns an optical-flow + transfer
//! queue. This avoids changing the game's requested device features or queues.
//! Reference: https://docs.vulkan.org/spec/latest/chapters/VK_NV_optical_flow/optical_flow.html
use ash::vk;
use std::ffi::CStr;

pub struct OpticalFlow {
    device: ash::Device,
    api: vk::NvOpticalFlowFn,
    sync: ash::extensions::khr::Synchronization2,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    session: vk::OpticalFlowSessionNV,
    images: Vec<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    pub width: u32,
    pub height: u32,
    pub quality: u32,
    current: usize,
    previous: bool,
    grid: u32,
}
// Access is serialized by the layer's device-state mutex.
unsafe impl Send for OpticalFlow {}

impl OpticalFlow {
    pub fn new(instance: &ash::Instance, pd: vk::PhysicalDevice, width: u32, height: u32, quality: u32) -> Result<Self, String> {
        unsafe { Self::create(instance, pd, width, height, quality) }.map_err(|e| format!("{e:?}"))
    }

    unsafe fn create(instance: &ash::Instance, pd: vk::PhysicalDevice, width: u32, height: u32, quality: u32) -> Result<Self, vk::Result> {
        let available = instance.enumerate_device_extension_properties(pd)?;
        let wanted = [c"VK_NV_optical_flow", c"VK_KHR_synchronization2", c"VK_KHR_format_feature_flags2"];
        if !wanted.iter().all(|name| available.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == *name)) {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        }
        let family = instance.get_physical_device_queue_family_properties(pd).iter().position(|p|
            p.queue_flags.contains(vk::QueueFlags::OPTICAL_FLOW_NV | vk::QueueFlags::TRANSFER)
        ).ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)? as u32;
        let mut optical_features = vk::PhysicalDeviceOpticalFlowFeaturesNV::default();
        let mut sync_features = vk::PhysicalDeviceSynchronization2Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::builder().push_next(&mut optical_features).push_next(&mut sync_features);
        instance.get_physical_device_features2(pd, &mut features);
        if optical_features.optical_flow == 0 || sync_features.synchronization2 == 0 { return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT); }
        let mut props = vk::PhysicalDeviceOpticalFlowPropertiesNV::default();
        instance.get_physical_device_properties2(pd, &mut vk::PhysicalDeviceProperties2::builder().push_next(&mut props));
        if width < props.min_width || width > props.max_width || height < props.min_height || height > props.max_height {
            return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
        }
        let grid = [4, 2, 1, 8].into_iter().find(|&g| props.supported_output_grid_sizes.as_raw() & g != 0)
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        // Query builders link these structs; clear their links before reusing them
        // in the device chain, otherwise pushing both again creates a cycle.
        optical_features.p_next = std::ptr::null_mut();
        sync_features.p_next = std::ptr::null_mut();
        let queues = [vk::DeviceQueueCreateInfo::builder().queue_family_index(family).queue_priorities(&[1.0]).build()];
        let extensions: Vec<_> = wanted.iter().map(|n| n.as_ptr()).collect();
        let device = instance.create_device(pd, &vk::DeviceCreateInfo::builder().queue_create_infos(&queues)
            .enabled_extension_names(&extensions).push_next(&mut optical_features).push_next(&mut sync_features), None)?;
        let api = vk::NvOpticalFlowFn::load(|name| std::mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr())));
        let sync = ash::extensions::khr::Synchronization2::new(instance, &device);
        let queue = device.get_device_queue(family, 0);
        // Own the device immediately, so all subsequent failures clean up.
        let mut flow = Self { device, api, sync, queue, pool: vk::CommandPool::null(), cmd: vk::CommandBuffer::null(),
            session: vk::OpticalFlowSessionNV::null(), images: vec![], buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(), width, height, quality, current: 0, previous: false, grid };
        let level = match quality { 0 => vk::OpticalFlowPerformanceLevelNV::FAST, 2 => vk::OpticalFlowPerformanceLevelNV::SLOW, _ => vk::OpticalFlowPerformanceLevelNV::MEDIUM };
        let info = vk::OpticalFlowSessionCreateInfoNV::builder().width(width).height(height)
            .image_format(vk::Format::B8G8R8A8_UNORM).flow_vector_format(vk::Format::R16G16_S10_5_NV)
            .output_grid_size(vk::OpticalFlowGridSizeFlagsNV::from_raw(grid)).performance_level(level);
        (flow.api.create_optical_flow_session_nv)(flow.device.handle(), &*info, std::ptr::null(), &mut flow.session).result()?;
        let mem = instance.get_physical_device_memory_properties(pd);
        for i in 0..3 {
            let output = i == 2;
            let (w,h) = if output { (width.div_ceil(grid), height.div_ceil(grid)) } else { (width,height) };
            let format = if output { vk::Format::R16G16_S10_5_NV } else { vk::Format::B8G8R8A8_UNORM };
            let mut usage = vk::OpticalFlowImageFormatInfoNV::builder().usage(if output { vk::OpticalFlowUsageFlagsNV::OUTPUT } else { vk::OpticalFlowUsageFlagsNV::INPUT });
            let info = vk::ImageCreateInfo::builder().image_type(vk::ImageType::TYPE_2D).format(format)
                .extent(vk::Extent3D {width:w,height:h,depth:1}).mip_levels(1).array_layers(1).samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL).usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE).push_next(&mut usage);
            let image = flow.device.create_image(&info, None)?;
            flow.images.push((image, vk::DeviceMemory::null(), vk::ImageView::null()));
            let req = flow.device.get_image_memory_requirements(image);
            let memory = flow.device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(req.size)
                .memory_type_index(memory_type(&mem, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?), None)?;
            flow.images[i].1 = memory;
            flow.device.bind_image_memory(image, memory, 0)?;
            let view = flow.device.create_image_view(&vk::ImageViewCreateInfo::builder().image(image)
                .view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(subresource()), None)?;
            flow.images[i].2 = view;
        }
        flow.pool = flow.device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)?;
        flow.cmd = flow.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(flow.pool)
            .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
        flow.buffer = flow.device.create_buffer(&vk::BufferCreateInfo::builder().size(u64::from(width)*u64::from(height)*4)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST), None)?;
        let req = flow.device.get_buffer_memory_requirements(flow.buffer);
        flow.memory = flow.device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(req.size)
            .memory_type_index(memory_type(&mem, req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?), None)?;
        flow.device.bind_buffer_memory(flow.buffer, flow.memory, 0)?;
        flow.mapped = flow.device.map_memory(flow.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?.cast();
        Ok(flow)
    }

    /// Returns full-resolution signed pixel displacements, current -> previous.
    /// The first capture seeds history and returns no vectors.
    pub fn estimate(&mut self, bytes: &[u8], bgr: bool) -> Result<Option<Vec<[f32;2]>>, String> {
        unsafe { self.run(bytes,bgr) }.map_err(|e| format!("{e:?}"))
    }
    unsafe fn run(&mut self, bytes: &[u8], bgr: bool) -> Result<Option<Vec<[f32;2]>>, vk::Result> {
        let n = self.width as usize * self.height as usize * 4;
        if bytes.len() != n { return Err(vk::Result::ERROR_INITIALIZATION_FAILED); }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped, n);
        if !bgr { for p in std::slice::from_raw_parts_mut(self.mapped,n).chunks_exact_mut(4) { p.swap(0,2); } }
        self.device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
        self.device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let input = self.images[self.current].0;
        self.barrier(input, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        self.device.cmd_copy_buffer_to_image(self.cmd,self.buffer,input,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[region(self.width,self.height)]);
        self.barrier(input,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::GENERAL);
        if self.previous {
            for (binding, view) in [(vk::OpticalFlowSessionBindingPointNV::INPUT,self.images[self.current].2),
                (vk::OpticalFlowSessionBindingPointNV::REFERENCE,self.images[1-self.current].2),
                (vk::OpticalFlowSessionBindingPointNV::FLOW_VECTOR,self.images[2].2)] {
                (self.api.bind_optical_flow_session_image_nv)(self.device.handle(),self.session,binding,view,vk::ImageLayout::GENERAL).result()?;
            }
            self.barrier(self.images[2].0,vk::ImageLayout::UNDEFINED,vk::ImageLayout::GENERAL);
            // Captures may be several presents apart: disable implicit temporal hints.
            let execute = vk::OpticalFlowExecuteInfoNV::builder().flags(vk::OpticalFlowExecuteFlagsNV::DISABLE_TEMPORAL_HINTS);
            (self.api.cmd_optical_flow_execute_nv)(self.cmd,self.session,&*execute);
            self.barrier(self.images[2].0,vk::ImageLayout::GENERAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            self.device.cmd_copy_image_to_buffer(self.cmd,self.images[2].0,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,self.buffer,
                &[region(self.width.div_ceil(self.grid),self.height.div_ceil(self.grid))]);
            let barrier = [vk::MemoryBarrier2::builder().src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE).dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ).build()];
            self.sync.cmd_pipeline_barrier2(self.cmd,&vk::DependencyInfo::builder().memory_barriers(&barrier));
        }
        self.device.end_command_buffer(self.cmd)?;
        let cmds = [self.cmd];
        self.device.queue_submit(self.queue,&[vk::SubmitInfo::builder().command_buffers(&cmds).build()],vk::Fence::null())?;
        self.device.queue_wait_idle(self.queue)?;
        let output = if self.previous {
            let grid_w = self.width.div_ceil(self.grid) as usize;
            let raw = std::slice::from_raw_parts(self.mapped, grid_w * self.height.div_ceil(self.grid) as usize * 4);
            let mut output = vec![[0.0;2];self.width as usize*self.height as usize];
            for y in 0..self.height as usize { for x in 0..self.width as usize {
                let offset = ((y/self.grid as usize)*grid_w+x/self.grid as usize)*4;
                output[y*self.width as usize+x] = [i16::from_le_bytes([raw[offset],raw[offset+1]]) as f32/32.0,
                    i16::from_le_bytes([raw[offset+2],raw[offset+3]]) as f32/32.0];
            }}
            Some(output)
        } else {None};
        self.previous = true;
        self.current = 1-self.current;
        Ok(output)
    }
    unsafe fn barrier(&self,image:vk::Image,old:vk::ImageLayout,new:vk::ImageLayout) {
        let barriers = [vk::ImageMemoryBarrier2::builder().image(image).old_layout(old).new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(subresource()).src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE).build()];
        self.sync.cmd_pipeline_barrier2(self.cmd,&vk::DependencyInfo::builder().image_memory_barriers(&barriers));
    }
}
fn memory_type(mem:&vk::PhysicalDeviceMemoryProperties,bits:u32,flags:vk::MemoryPropertyFlags)->Result<u32,vk::Result> {
    (0..mem.memory_type_count).find(|&i| bits&(1<<i)!=0 && mem.memory_types[i as usize].property_flags.contains(flags))
        .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)
}
fn subresource()->vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1).build()
}
fn region(width:u32,height:u32)->vk::BufferImageCopy {
    vk::BufferImageCopy::builder().image_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).build())
        .image_extent(vk::Extent3D{width,height,depth:1}).build()
}
impl Drop for OpticalFlow {
    fn drop(&mut self) { unsafe {
        let _ = self.device.device_wait_idle();
        if self.session != vk::OpticalFlowSessionNV::null() { (self.api.destroy_optical_flow_session_nv)(self.device.handle(),self.session,std::ptr::null()); }
        for &(image,memory,view) in &self.images {
            self.device.destroy_image_view(view,None); self.device.destroy_image(image,None); self.device.free_memory(memory,None);
        }
        if !self.mapped.is_null() { self.device.unmap_memory(self.memory); }
        self.device.destroy_buffer(self.buffer,None); self.device.free_memory(self.memory,None);
        self.device.destroy_command_pool(self.pool,None); self.device.destroy_device(None);
    }}
}
