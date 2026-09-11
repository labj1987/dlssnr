//! A self-implemented, in-process `NVSDK_NGX_Parameter` object -- used only when
//! neither `nvngx_dlssnr.dll` (the snippet) nor `nvngx.dll` (Core) hand back a
//! working parameter block of their own.
//!
//! Real evidence this exists for, not speculation: a real side-by-side comparison on
//! `lordnikon` (2026-09-11) against upstream's own compiled helper binary (recovered
//! from its official GitHub release, never its source -- same "shape not expression"
//! rule as everywhere else in this project) showed upstream hitting the *exact same*
//! `0xbad00002` (`FAIL_PLATFORM_ERROR`) from Core's `VULKAN_Init_with_ProjectID`/
//! `VULKAN_Init_Ext`/`AllocateParameters` that this crate does, on this same machine,
//! with this same real `nvngx.dll`. Upstream's own log then shows: `AllocateParameters`
//! failing on Core, the snippet not exporting it at all, and immediately after --
//! `[params] using own NVSDK_NGX_Parameter implementation`, followed by a passing
//! round-trip self-test and a real, successful `VULKAN_CreateFeature(18) -> 0x1`.
//! NGX's own `VULKAN_Init_Ext`/`CreateFeature` never actually require a parameter
//! block *allocated by the DLL itself* -- they just need a pointer matching the real
//! `NVSDK_NGX_Parameter` vtable shape, which anyone can construct. Rejecting Core's own
//! allocator is apparently an expected, survivable condition in this exact environment,
//! not something either implementation is meant to treat as fatal.
//!
//! The object layout matches [`abi::NgxParameterObj`]: a vtable pointer as the first
//! (and, for this implementation, only C++-visible) field, backed by a real
//! `HashMap` for storage -- safe because C++ virtual dispatch only ever touches the
//! `this` pointer opaquely through the vtable; nothing on the DLL side assumes
//! anything about this object's layout beyond that first pointer.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_void, CStr};

use crate::abi::{self, NgxParameter, NgxParameterObj, NgxParameterVtable, NgxResult};

#[derive(Clone, Copy)]
enum Value {
    Ptr(*mut c_void),
    U64(u64),
    F32(f32),
    F64(f64),
    U32(u32),
    I32(i32),
}

// SAFETY: every `Value` variant is either a plain number or a pointer this object
// never dereferences itself -- it only ever hands pointers back out to the same
// single-threaded NGX call sequence that stored them, exactly as `NgxSnippet` already
// assumes for its own raw-pointer fields.
unsafe impl Send for Value {}

#[repr(C)]
struct SelfParam {
    vtable: *const NgxParameterVtable,
    store: RefCell<HashMap<String, Value>>,
}

/// # Safety
/// `name` must be a valid, NUL-terminated C string for the duration of this call —
/// guaranteed by every real NGX vtable call's own contract (the string is only read
/// synchronously, never retained past the call).
unsafe fn key(name: *const i8) -> String {
    // SAFETY: contract above.
    unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
}

/// # Safety
/// `this` must be a live `*mut SelfParam` -- guaranteed by every call here originating
/// from a vtable slot invoked against a pointer this module itself allocated.
unsafe fn store<'a>(this: *mut c_void) -> &'a RefCell<HashMap<String, Value>> {
    // SAFETY: contract above; `this` is exactly the `SelfParam` this module allocated,
    // cast back to what it really is.
    unsafe { &(*this.cast::<SelfParam>()).store }
}

unsafe extern "system" fn set_ptr(this: *mut c_void, name: *const i8, value: *mut c_void) {
    // SAFETY: contracts of `key`/`store` above.
    unsafe { store(this).borrow_mut().insert(key(name), Value::Ptr(value)) };
}
unsafe extern "system" fn set_u64(this: *mut c_void, name: *const i8, value: u64) {
    unsafe { store(this).borrow_mut().insert(key(name), Value::U64(value)) };
}
unsafe extern "system" fn set_f32(this: *mut c_void, name: *const i8, value: f32) {
    unsafe { store(this).borrow_mut().insert(key(name), Value::F32(value)) };
}
unsafe extern "system" fn set_f64(this: *mut c_void, name: *const i8, value: f64) {
    unsafe { store(this).borrow_mut().insert(key(name), Value::F64(value)) };
}
unsafe extern "system" fn set_u32(this: *mut c_void, name: *const i8, value: u32) {
    unsafe { store(this).borrow_mut().insert(key(name), Value::U32(value)) };
}
unsafe extern "system" fn set_i32(this: *mut c_void, name: *const i8, value: i32) {
    unsafe { store(this).borrow_mut().insert(key(name), Value::I32(value)) };
}

unsafe extern "system" fn get_ptr(this: *mut c_void, name: *const i8, out: *mut *mut c_void) -> NgxResult {
    // SAFETY: `out` is a valid out-pointer per every real NGX getter's own contract.
    match unsafe { store(this).borrow().get(&key(name)).copied() } {
        Some(Value::Ptr(v)) => {
            unsafe { *out = v };
            abi::result::SUCCESS
        }
        Some(_) => abi::result::FAIL_INCOMPATIBLE_TYPES,
        None => abi::result::FAIL_INVALID_PARAMETER,
    }
}
unsafe extern "system" fn get_u64(this: *mut c_void, name: *const i8, out: *mut u64) -> NgxResult {
    match unsafe { store(this).borrow().get(&key(name)).copied() } {
        Some(Value::U64(v)) => {
            unsafe { *out = v };
            abi::result::SUCCESS
        }
        Some(_) => abi::result::FAIL_INCOMPATIBLE_TYPES,
        None => abi::result::FAIL_INVALID_PARAMETER,
    }
}
unsafe extern "system" fn get_f32(this: *mut c_void, name: *const i8, out: *mut f32) -> NgxResult {
    match unsafe { store(this).borrow().get(&key(name)).copied() } {
        Some(Value::F32(v)) => {
            unsafe { *out = v };
            abi::result::SUCCESS
        }
        Some(_) => abi::result::FAIL_INCOMPATIBLE_TYPES,
        None => abi::result::FAIL_INVALID_PARAMETER,
    }
}
unsafe extern "system" fn get_f64(this: *mut c_void, name: *const i8, out: *mut f64) -> NgxResult {
    match unsafe { store(this).borrow().get(&key(name)).copied() } {
        Some(Value::F64(v)) => {
            unsafe { *out = v };
            abi::result::SUCCESS
        }
        Some(_) => abi::result::FAIL_INCOMPATIBLE_TYPES,
        None => abi::result::FAIL_INVALID_PARAMETER,
    }
}
unsafe extern "system" fn get_u32(this: *mut c_void, name: *const i8, out: *mut u32) -> NgxResult {
    match unsafe { store(this).borrow().get(&key(name)).copied() } {
        Some(Value::U32(v)) => {
            unsafe { *out = v };
            abi::result::SUCCESS
        }
        Some(_) => abi::result::FAIL_INCOMPATIBLE_TYPES,
        None => abi::result::FAIL_INVALID_PARAMETER,
    }
}
unsafe extern "system" fn get_i32(this: *mut c_void, name: *const i8, out: *mut i32) -> NgxResult {
    match unsafe { store(this).borrow().get(&key(name)).copied() } {
        Some(Value::I32(v)) => {
            unsafe { *out = v };
            abi::result::SUCCESS
        }
        Some(_) => abi::result::FAIL_INCOMPATIBLE_TYPES,
        None => abi::result::FAIL_INVALID_PARAMETER,
    }
}
/// Slots 9/10/13 in the real vtable are reserved padding — never called for real, but
/// present so every later slot's offset matches the real interface.
unsafe extern "system" fn get_reserved(_this: *mut c_void, _name: *const i8, _out: *mut c_void) -> NgxResult {
    abi::result::FAIL_NOT_IMPLEMENTED
}
unsafe extern "system" fn reset(this: *mut c_void) {
    unsafe { store(this).borrow_mut().clear() };
}

static VTABLE: NgxParameterVtable = NgxParameterVtable {
    set_ptr,
    set_u64,
    set_f32,
    set_f64,
    set_u32,
    set_i32,
    get_f64,
    get_u64,
    get_ptr,
    get_reserved9: get_reserved,
    get_reserved10: get_reserved,
    get_i32,
    get_u32,
    get_reserved13: get_reserved,
    get_f32,
    reset,
};

/// Allocates a new self-implemented parameter block, matching the real
/// `NVSDK_NGX_Parameter` vtable shape closely enough that `nvngx_dlssnr.dll` accepts a
/// pointer to it wherever a real `NgxParameter` is expected. Pair with [`destroy`],
/// never with a DLL's own `DestroyParameters` export -- this object isn't one of its
/// allocations.
pub fn allocate() -> NgxParameter {
    let boxed = Box::new(SelfParam { vtable: &VTABLE, store: RefCell::new(HashMap::new()) });
    Box::into_raw(boxed).cast::<NgxParameterObj>()
}

/// # Safety
/// `params` must be a pointer previously returned by [`allocate`] and not already
/// destroyed.
pub unsafe fn destroy(params: NgxParameter) {
    // SAFETY: contract above -- reconstructs exactly the `Box<SelfParam>` `allocate`
    // leaked, as the same concrete type it was allocated as.
    drop(unsafe { Box::from_raw(params.cast::<SelfParam>()) });
}
