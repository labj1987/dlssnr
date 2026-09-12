#[path = "../src/optical_flow.rs"]
mod optical_flow;
use ash::vk;
fn main() {
    let entry = unsafe { ash::Entry::load() }.unwrap();
    let app = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::builder().application_info(&app), None) }.unwrap();
    let pd = unsafe { instance.enumerate_physical_devices() }.unwrap().into_iter().find(|&p| unsafe {instance.get_physical_device_properties(p)}.vendor_id == 0x10de).expect("NVIDIA GPU required");
    let (w,h) = (512u32,512u32);
    eprintln!("creating flow");
    let mut flow = optical_flow::OpticalFlow::new(&instance,pd,w,h,1).expect("optical flow creation");
    eprintln!("created flow quality={}",flow.quality);
    let mut first = vec![0u8;(w*h*4) as usize];
    for y in 0..h {for x in 0..w {
        let i = ((y*w+x)*4) as usize;
        let noise = ((x/8).wrapping_mul(747796405) ^ (y/8).wrapping_mul(2891336453)).wrapping_mul(277803737);
        let value = (noise >> 24) as u8;
        first[i..i+4].copy_from_slice(&[value,value,value,255]);
    }}
    eprintln!("seed");
    assert!(flow.estimate(&first,true).unwrap().is_none());
    eprintln!("stationary");
    let stationary=flow.estimate(&first,true).unwrap().unwrap();
    let static_mean: f32 = stationary.iter().map(|v|v[0].abs()+v[1].abs()).sum::<f32>()/stationary.len() as f32;
    assert!(static_mean < 0.25,"stationary flow {static_mean}");
    let mut second = first.clone();
    for y in 0..h {for x in 8..w {let i=((y*w+x)*4) as usize;second[i..i+4].copy_from_slice(&first[i-32..i-28]);}}
    eprintln!("translation");
    let vectors = flow.estimate(&second,true).unwrap().unwrap();
    let mut xs=Vec::new(); let mut ys=Vec::new();
    for y in 64..h-64 {for x in 64..w-64 {let v=vectors[(y*w+x) as usize];xs.push(v[0]);ys.push(v[1]);}}
    xs.sort_by(f32::total_cmp);ys.sort_by(f32::total_cmp);
    let (dx,dy)=(xs[xs.len()/2],ys[ys.len()/2]);
    println!("stationary mean={static_mean}; translated median=({dx},{dy}), expected (-8,0)");
    assert!((dx+8.0).abs()<1.0 && dy.abs()<1.0);
    drop(flow);
    unsafe {instance.destroy_instance(None)};
}
