//! Owns the `nvngx_dlssnr.dll` ("the signed snippet") lifecycle: load, install the
//! caller-identity spoof from `crate::spoof`, initialize NGX, create Feature 18.
//!
//! Ported for shape from `core/ngx_snippet.cpp`'s `NgxLoadAndInit`/`NgxCreatePass`/
//! `NgxTeardown` — the call sequence (which exports to resolve, which parameters to
//! set, in which order) is NVIDIA's own NGX contract, not upstream's expression of it;
//! this is a fresh Rust implementation of walking that contract.
//!
//! Milestone 3 scope (see the plan): load, spoof, init, and `CreateFeature(18)` — the
//! path that actually exercises the caller-identity spoof (NGX checks its caller
//! during init/create, not evaluate) and the SEH guard around a real call into
//! NVIDIA's DLL. `EvaluateFeature` needs bound `DLSSNR.Color`/`Output`/`MVec` Vulkan
//! image resources to mean anything, and building those is naturally paired with
//! milestone 4's composition pass (the same resources that pass reads/writes) — so
//! evaluate's parameter-binding (`NgxSetResources` upstream) is deferred there. The
//! call mechanics are structurally identical to create's (same guarded-call pattern,
//! same vtable parameter setting), so getting create working through the spoof
//! de-risks evaluate too, even unwired.

use std::ffi::{c_void, CString};

use ash::vk;

use crate::abi::{self, NgxParameter};
use crate::guard::guarded;
use crate::spoof::{self, InstalledSpoof};

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryExW(filename: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
}

const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn resolve_bin_dir() -> Option<String> {
    // `std::env::var` already goes through the real `GetEnvironmentVariableW` on this
    // target -- no reason to hand-rolled that call the way `spoof.rs`'s PE parsing
    // genuinely needs to hand-roll PE-specific things.
    let dir = std::env::var("DLSSNR_BIN_DIR").ok()?;
    if dir.is_empty() {
        return None;
    }
    Some(dir)
}

pub struct NgxSnippet {
    snippet: *mut c_void,
    core: *mut c_void,
    snippet_spoof: Option<InstalledSpoof>,
    core_spoof: Option<InstalledSpoof>,

    init_ext: Option<abi::FnVkInitExt>,
    create_feature: Option<abi::FnVkCreateFeature>,
    evaluate_feature: Option<abi::FnVkEvaluateFeature>,
    release_feature: Option<abi::FnVkReleaseFeature>,
    shutdown1: Option<abi::FnVkShutdown1>,

    params: NgxParameter,
    params_destroy: Option<abi::FnVkDestroyParameters>,

    /// The Vulkan device NGX was initialized against — `Shutdown1` must be called with
    /// this exact device, not a null placeholder.
    device: vk::Device,

    pub feature: abi::NgxHandle,
    pub disabled: bool,
}

// SAFETY: every raw pointer/handle field here is either an opaque DLL/NGX handle
// (never dereferenced by this crate as anything other than an opaque token passed
// back to the same DLL) or a function pointer resolved once and never mutated after —
// nothing here assumes exclusive access beyond what `Mutex<NgxSnippet>` at the call
// site already guarantees.
unsafe impl Send for NgxSnippet {}

impl Default for NgxSnippet {
    fn default() -> Self {
        Self {
            snippet: std::ptr::null_mut(),
            core: std::ptr::null_mut(),
            snippet_spoof: None,
            core_spoof: None,
            init_ext: None,
            create_feature: None,
            evaluate_feature: None,
            release_feature: None,
            shutdown1: None,
            params: std::ptr::null_mut(),
            params_destroy: None,
            device: vk::Device::null(),
            feature: std::ptr::null_mut(),
            disabled: false,
        }
    }
}

/// # Safety
/// `module`/`name` must be exactly what [`crate::spoof::find_imported_function_slot`]
/// and `GetProcAddress` require.
unsafe fn resolve_export<F: Copy>(module: *mut c_void, name: &str) -> Option<F> {
    let c_name = CString::new(name).ok()?;
    // SAFETY: `module` is a valid, loaded module handle (the caller's contract).
    let p = unsafe { GetProcAddress(module, c_name.as_ptr()) };
    if p.is_null() {
        return None;
    }
    // SAFETY: forwarded from this function's own contract -- `F` must be the type
    // matching `name`'s real signature.
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) })
}

/// Loads `nvngx_dlssnr.dll`, installs the caller-identity spoof, initializes NGX, and
/// creates Feature 18. Every DLL call is wrapped in [`guarded`] — a fault anywhere in
/// here latches `disabled` rather than taking the whole helper down with it.
pub fn load_and_init(instance: vk::Instance, physical_device: vk::PhysicalDevice, device: vk::Device) -> NgxSnippet {
    let mut s = NgxSnippet::default();
    s.device = device;

    let Some(bin_dir) = resolve_bin_dir() else {
        crate::log!("[ngx] DLSSNR_BIN_DIR not set or nvngx_dlssnr.dll not found there");
        s.disabled = true;
        return s;
    };
    let dll_path = utf16(&format!("{bin_dir}\\nvngx_dlssnr.dll"));
    // SAFETY: `dll_path` is a valid NUL-terminated UTF-16 string.
    s.snippet = unsafe {
        LoadLibraryExW(
            dll_path.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    };
    if s.snippet.is_null() {
        crate::log!("[ngx] LoadLibraryExW nvngx_dlssnr.dll failed");
        s.disabled = true;
        return s;
    }
    crate::log!("[ngx] nvngx_dlssnr.dll loaded at {:?}", s.snippet);

    // SAFETY: `s.snippet` was just confirmed non-null and loaded above.
    unsafe {
        s.init_ext = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_Init_Ext");
        s.create_feature = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_CreateFeature");
        s.evaluate_feature = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_EvaluateFeature");
        s.release_feature = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_ReleaseFeature");
        s.shutdown1 = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_Shutdown1");
    }
    if s.create_feature.is_none() || s.evaluate_feature.is_none() || s.release_feature.is_none() || s.shutdown1.is_none()
    {
        crate::log!("[ngx] snippet Vulkan exports incomplete");
        unsafe { FreeLibrary(s.snippet) };
        s.snippet = std::ptr::null_mut();
        s.disabled = true;
        return s;
    }

    // SAFETY: `s.snippet` is a valid, currently-loaded module.
    s.snippet_spoof = unsafe { spoof::install(s.snippet) };
    if s.snippet_spoof.is_none() {
        crate::log!("[ngx] failed to install the caller-identity spoof on the snippet");
        s.disabled = true;
        return s;
    }

    // Core (nvngx.dll) is the preferred parameter allocator; optional, same as
    // upstream -- a missing or faulting core degrades to the snippet's own allocator.
    let core_path = utf16(&format!("{bin_dir}\\nvngx.dll"));
    // SAFETY: `core_path` is a valid NUL-terminated UTF-16 string.
    s.core = unsafe {
        LoadLibraryExW(
            core_path.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    };
    if !s.core.is_null() {
        // SAFETY: `s.core` just confirmed non-null.
        s.core_spoof = unsafe { spoof::install(s.core) };
    }

    let alloc: Option<abi::FnVkAllocateParameters> = if !s.core.is_null() {
        // SAFETY: `s.core` is a valid, loaded module.
        unsafe { resolve_export(s.core, "NVSDK_NGX_VULKAN_AllocateParameters") }
    } else {
        None
    };
    let alloc = alloc.or_else(|| unsafe { resolve_export(s.snippet, "NVSDK_NGX_VULKAN_AllocateParameters") });
    s.params_destroy = if !s.core.is_null() {
        unsafe { resolve_export(s.core, "NVSDK_NGX_VULKAN_DestroyParameters") }
    } else {
        None
    }
    .or_else(|| unsafe { resolve_export(s.snippet, "NVSDK_NGX_VULKAN_DestroyParameters") });

    let Some(alloc) = alloc else {
        crate::log!("[ngx] no AllocateParameters export found on core or snippet");
        s.disabled = true;
        return s;
    };
    let (alloc_result, seh) = guarded(
        || {
            let mut params: NgxParameter = std::ptr::null_mut();
            // SAFETY: `alloc` resolved above from a live module; `&mut params` is a
            // valid out-pointer for the call's duration.
            let r = unsafe { alloc(&mut params) };
            (r, params)
        },
        (abi::result::FAIL_SEH, std::ptr::null_mut()),
    );
    let (alloc_code, params) = alloc_result;
    crate::log!("[ngx] AllocateParameters -> {:#x} seh={:#x}", alloc_code as u32, seh);
    if !abi::succeeded(alloc_code) || params.is_null() {
        s.disabled = true;
        return s;
    }
    s.params = params;

    let Some(init_ext) = s.init_ext else {
        crate::log!("[ngx] snippet has no VULKAN_Init_Ext export");
        s.disabled = true;
        return s;
    };
    let app_data_path = utf16(&bin_dir);
    let ((init_result,), seh) = guarded(
        || {
            // SAFETY: `init_ext` resolved above; `instance`/`physical_device`/`device`
            // are the caller's own, live Vulkan handles; `app_data_path` is a valid
            // NUL-terminated UTF-16 string kept alive for this call's duration.
            let r = unsafe {
                init_ext(
                    abi::SIGNED_SNIPPET_APPLICATION_ID,
                    app_data_path.as_ptr(),
                    instance,
                    physical_device,
                    device,
                    abi::VERSION_API_14,
                    std::ptr::null(),
                )
            };
            (r,)
        },
        (abi::result::FAIL_SEH,),
    );
    crate::log!("[ngx] VULKAN_Init_Ext -> {:#x} seh={:#x}", init_result as u32, seh);
    if !abi::succeeded(init_result) {
        s.disabled = true;
        return s;
    }

    // A 1x1 placeholder: this proves the create call chain (and therefore the
    // caller-identity spoof and the SEH guard around a real DLL call) end to end. Real
    // sizing arrives with milestone 4, once there's an actual frame to build the
    // feature at the size of.
    if !create_feature_at(&mut s, 1, 1) {
        s.disabled = true;
    }
    s
}

fn create_feature_at(s: &mut NgxSnippet, width: u32, height: u32) -> bool {
    let Some(create_feature) = s.create_feature else { return false };
    let name = |n: &str| CString::new(n).unwrap();
    let params = s.params;
    // Guarded like every other real call into the DLL below: `params`'s vtable is a
    // hand-ported ABI shape for a feature this crate has never had real hardware/DLL to
    // test against (see the module doc comment and CLAUDE.md's `ngx.rs` gotchas) --  if
    // a slot is misaligned relative to what the real driver's `nvngx.dll`/
    // `nvngx_dlssnr.dll` actually expects, calling through it faults, and an unguarded
    // fault here takes the whole helper down with no log line at all rather than
    // latching `disabled` the way every other DLL call in this file already does.
    let ((), seh) = guarded(
        || {
            // SAFETY: `params` was allocated and validated in `load_and_init` above.
            unsafe {
                abi::ngx_set_u32(params, name("DLSSNR.Width").as_ptr(), width);
                abi::ngx_set_u32(params, name("DLSSNR.Height").as_ptr(), height);
                abi::ngx_set_u32(params, name("DLSSNR.InputWidth").as_ptr(), width);
                abi::ngx_set_u32(params, name("DLSSNR.InputHeight").as_ptr(), height);
                abi::ngx_set_u32(params, name("DLSSNR.OutputWidth").as_ptr(), width);
                abi::ngx_set_u32(params, name("DLSSNR.OutputHeight").as_ptr(), height);
                abi::ngx_set_u32(params, name("DLSSNR.Upscaling").as_ptr(), 0);
                abi::ngx_set_f32(params, name("DLSSNR.Scale").as_ptr(), 1.0);
                abi::ngx_set_f32(params, name("DLSSNR.ScalingRatio").as_ptr(), 1.0);
                abi::ngx_set_u32(params, name("Width").as_ptr(), width);
                abi::ngx_set_u32(params, name("Height").as_ptr(), height);
                abi::ngx_set_u32(params, name("CreationNodeMask").as_ptr(), 1);
                abi::ngx_set_u32(params, name("VisibilityNodeMask").as_ptr(), 1);
                let flags = abi::feature_flags::DO_SHARPENING | abi::feature_flags::AUTO_EXPOSURE;
                abi::ngx_set_u32(params, name("Feature_Flags").as_ptr(), flags);
            }
        },
        (),
    );
    crate::log!("[ngx] set DLSSNR parameters seh={:#x}", seh);
    if seh != 0 {
        return false;
    }

    let ((result, handle), seh) = guarded(
        || {
            let mut handle: abi::NgxHandle = std::ptr::null_mut();
            // SAFETY: `create_feature` resolved and validated in `load_and_init`;
            // `vk::CommandBuffer::null()` matches upstream's own create-time contract
            // (create happens outside a recording command buffer in this milestone --
            // no frame exists yet to record into); `params`/`&mut handle` are valid.
            let r = unsafe { create_feature(vk::CommandBuffer::null(), abi::FEATURE_DLSSNR, params, &mut handle) };
            (r, handle)
        },
        (abi::result::FAIL_SEH, std::ptr::null_mut()),
    );
    crate::log!(
        "[ngx] VULKAN_CreateFeature(18) -> {:#x} seh={:#x} handle={:?} size={width}x{height}",
        result as u32,
        seh,
        handle
    );
    if !abi::succeeded(result) || handle.is_null() {
        return false;
    }
    s.feature = handle;
    true
}

/// Release the feature, shut down the snippet, restore the caller-identity spoof, and
/// unload both modules — in that order, matching upstream's verified teardown
/// sequence.
pub fn teardown(mut s: NgxSnippet) {
    if !s.feature.is_null() {
        if let Some(release) = s.release_feature {
            let feature = s.feature;
            let (result, seh) = guarded(|| unsafe { release(feature) }, abi::result::FAIL_SEH);
            crate::log!("[ngx] ReleaseFeature -> {:#x} seh={:#x}", result as u32, seh);
        }
        s.feature = std::ptr::null_mut();
    }
    if let Some(shutdown1) = s.shutdown1 {
        let device = s.device;
        let (result, seh) = guarded(|| unsafe { shutdown1(device) }, abi::result::FAIL_SEH);
        crate::log!("[ngx] Shutdown1 -> {:#x} seh={:#x}", result as u32, seh);
    }
    if !s.params.is_null() {
        if let Some(destroy) = s.params_destroy {
            let params = s.params;
            let (result, seh) = guarded(|| unsafe { destroy(params) }, abi::result::FAIL_SEH);
            crate::log!("[ngx] DestroyParameters -> {:#x} seh={:#x}", result as u32, seh);
        }
        s.params = std::ptr::null_mut();
    }
    if let Some(spoofed) = s.snippet_spoof.take() {
        // SAFETY: the snippet module is still loaded at this point.
        unsafe { spoof::remove(spoofed) };
    }
    if let Some(spoofed) = s.core_spoof.take() {
        unsafe { spoof::remove(spoofed) };
    }
    if !s.core.is_null() {
        unsafe { FreeLibrary(s.core) };
    }
    if !s.snippet.is_null() {
        unsafe { FreeLibrary(s.snippet) };
    }
}
