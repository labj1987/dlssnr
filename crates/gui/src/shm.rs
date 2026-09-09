//! Binds GTK widgets to `dlssnr_protocol::ShmHeader` fields — this crate's equivalent
//! of upstream's `shm_binder.h/.cpp`, written fresh against the protocol crate's Rust
//! types (there's no logic in a per-field binder worth porting either way, just a
//! mechanical widget<->field mapping).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use dlssnr_protocol::mapping::Mapping;

/// Wraps the open mapping so the UI module can pass one `Arc` around to every
/// callback instead of re-opening or re-threading raw pointers everywhere.
pub struct Shm(pub Arc<Mapping>);

impl Shm {
    pub fn open() -> Option<Self> {
        dlssnr_protocol::mapping::open().map(Arc::new).map(Shm)
    }
}

/// Binds a GTK `Scale`/`SpinButton`-shaped float control to one `f32`-bits field:
/// reads the current value to initialize the widget, and writes back (bumping
/// `control_seq` so the layer/helper notice) whenever the widget changes.
pub fn bind_float(
    shm: &Arc<Mapping>,
    get: impl Fn(&dlssnr_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> (f32, impl Fn(f32) + 'static) {
    let initial = f32::from_bits(get(shm.header()).load(Ordering::Relaxed));
    let shm = Arc::clone(shm);
    let setter = move |value: f32| {
        get(shm.header()).store(value.to_bits(), Ordering::Relaxed);
        shm.header().control_seq.fetch_add(1, Ordering::Relaxed);
    };
    (initial, setter)
}

/// Same shape as [`bind_float`], for a plain `u32`-valued field (enable flags,
/// enum-valued settings, etc).
pub fn bind_u32(
    shm: &Arc<Mapping>,
    get: impl Fn(&dlssnr_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> (u32, impl Fn(u32) + 'static) {
    let initial = get(shm.header()).load(Ordering::Relaxed);
    let shm = Arc::clone(shm);
    let setter = move |value: u32| {
        get(shm.header()).store(value, Ordering::Relaxed);
        shm.header().control_seq.fetch_add(1, Ordering::Relaxed);
    };
    (initial, setter)
}

/// Same shape again, for a boolean flag stored as `0`/`1` in a `u32` field.
pub fn bind_bool(
    shm: &Arc<Mapping>,
    get: impl Fn(&dlssnr_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> (bool, impl Fn(bool) + 'static) {
    let (initial, setter) = bind_u32(shm, get);
    (initial != 0, move |value: bool| setter(if value { 1 } else { 0 }))
}
