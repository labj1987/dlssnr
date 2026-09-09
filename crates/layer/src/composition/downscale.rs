//! The supersampling down-leg's resampling filters: standard, well-published kernels
//! (Lanczos, Catmull-Rom, Mitchell-Netravali/"bicubic", Kaiser-windowed sinc), each
//! implemented directly from its mathematical definition — none of this needs, or
//! reads, upstream's `.spv`/`.h` pairs. Matches
//! [`dlssnr_protocol::enums::downscaler`]'s numbering (kept because it's a settings
//! contract, not an algorithm — see that module's doc comment).
//!
//! Each filter here is a 1D kernel; a 2D resample separably applies it along each
//! axis (standard practice for separable kernels — all of these are), which the GPU
//! compute shader that actually runs this does as two passes. This module is the
//! reference the shader is translated from and the thing `#[test]`s below check
//! against known closed-form values.

use dlssnr_protocol::enums::downscaler;

/// `sinc(x) = sin(pi*x) / (pi*x)`, with the removable singularity at 0 filled in.
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let px = std::f32::consts::PI * x;
        px.sin() / px
    }
}

/// The Lanczos kernel with `lobes` lobes on each side: a windowed sinc, zero outside
/// `[-lobes, lobes]`.
fn lanczos(x: f32, lobes: f32) -> f32 {
    if x.abs() >= lobes {
        0.0
    } else {
        sinc(x) * sinc(x / lobes)
    }
}

/// Catmull-Rom: the Mitchell-Netravali family member with `b=0, c=0.5` — interpolating
/// (passes through the input samples exactly), unlike Mitchell-Netravali's own default.
fn catmull_rom(x: f32) -> f32 {
    mitchell_netravali(x, 0.0, 0.5)
}

/// The Mitchell-Netravali two-parameter cubic filter family (Mitchell & Netravali,
/// *Reconstruction Filters in Computer Graphics*, 1988) — `b=1/3, c=1/3` is their own
/// recommended default ("bicubic" in the protocol's naming); `b=0, c=0.5` is
/// Catmull-Rom (see [`catmull_rom`]).
fn mitchell_netravali(x: f32, b: f32, c: f32) -> f32 {
    let ax = x.abs();
    if ax < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * ax.powi(3) + (-18.0 + 12.0 * b + 6.0 * c) * ax.powi(2) + (6.0 - 2.0 * b)) / 6.0
    } else if ax < 2.0 {
        ((-b - 6.0 * c) * ax.powi(3)
            + (6.0 * b + 30.0 * c) * ax.powi(2)
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

fn bicubic(x: f32) -> f32 {
    mitchell_netravali(x, 1.0 / 3.0, 1.0 / 3.0)
}

/// A Kaiser-windowed sinc with the given `beta` (higher = narrower main lobe, more
/// sidelobe suppression) and support radius `lobes`.
fn kaiser(x: f32, lobes: f32, beta: f32) -> f32 {
    if x.abs() >= lobes {
        return 0.0;
    }
    sinc(x) * kaiser_window(x / lobes, beta)
}

/// The Kaiser window itself, using the modified Bessel function `I0` (series
/// expansion — standard, e.g. Oppenheim & Schafer, *Discrete-Time Signal Processing*).
fn kaiser_window(t: f32, beta: f32) -> f32 {
    let arg = beta * (1.0 - t * t).max(0.0).sqrt();
    bessel_i0(arg) / bessel_i0(beta)
}

fn bessel_i0(x: f32) -> f32 {
    // I0(x) = sum_{k=0}^inf ( (x/2)^(2k) / (k!)^2 ). Converges quickly for the small
    // `beta` values (a handful) any reasonable Kaiser window uses; 24 terms is
    // overkill headroom, not a tuned minimum.
    let half_x_sq = (x * 0.5) * (x * 0.5);
    let mut term = 1.0f32;
    let mut sum = 1.0f32;
    for k in 1..24 {
        term *= half_x_sq / (k as f32 * k as f32);
        sum += term;
    }
    sum
}

/// The 1D kernel value at offset `x` (in samples) for the given
/// [`dlssnr_protocol::enums::downscaler`] value. Unsupported/unknown values (`FSR1`,
/// or anything out of range) fall back to `LANCZOS3`, matching the protocol's own
/// documented fallback behavior for a value this pipeline can't run.
pub fn kernel(x: f32, downscaler_kind: u32) -> f32 {
    match downscaler_kind {
        downscaler::BICUBIC => bicubic(x),
        downscaler::CATMULL_ROM => catmull_rom(x),
        downscaler::LANCZOS2 => lanczos(x, 2.0),
        downscaler::KAISER2 => kaiser(x, 2.0, 6.0),
        downscaler::KAISER3 => kaiser(x, 3.0, 8.0),
        downscaler::MAGIC => mitchell_netravali(x, 0.0, 0.6),
        _ => lanczos(x, 3.0), // LANCZOS3 and any unrecognized value both fall back here
    }
}

/// The kernel's support radius in samples — how far out a resample needs to gather
/// input taps from for this filter.
pub fn support_radius(downscaler_kind: u32) -> f32 {
    match downscaler_kind {
        downscaler::BICUBIC | downscaler::CATMULL_ROM | downscaler::MAGIC => 2.0,
        downscaler::LANCZOS2 => 2.0,
        downscaler::KAISER2 => 2.0,
        downscaler::KAISER3 => 3.0,
        _ => 3.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolating_kernels_are_one_at_zero_and_zero_at_their_own_integer_taps() {
        // The defining property of an *interpolating* kernel: it reproduces the sample
        // it sits on exactly (1 at x=0) and doesn't leak into neighboring integer
        // sample positions within its support (0 at x=+-1, +-2, ... up to its radius).
        // Every sinc-based filter here (Lanczos, Kaiser: sinc(0)=1, sinc(integer)=0)
        // and Catmull-Rom (Mitchell-Netravali B=0,C=0.5) has this property.
        // `BICUBIC` (Mitchell-Netravali's own recommended B=1/3,C=1/3) and `MAGIC`
        // (B=0,C=0.6) are deliberately excluded: Mitchell-Netravali's defining
        // trade-off is trading exact interpolation for a smoother reconstruction, so
        // `f(0) = 1 - B/3 != 1` whenever `B != 0` is correct behavior, not a bug.
        for &d in &[downscaler::CATMULL_ROM, downscaler::LANCZOS2, downscaler::LANCZOS3, downscaler::KAISER2, downscaler::KAISER3]
        {
            assert!((kernel(0.0, d) - 1.0).abs() < 1e-5, "kernel {d} at 0 should be 1");
            let radius = support_radius(d) as i32;
            for i in 1..radius {
                let v = kernel(i as f32, d);
                assert!(v.abs() < 1e-4, "kernel {d} at integer tap {i} should be ~0, got {v}");
            }
        }
    }

    #[test]
    fn mitchell_netravali_default_is_deliberately_non_interpolating() {
        // f(0) = (6 - 2B) / 6 = 1 - B/3 for the general B,C filter -- confirms the
        // "trades interpolation for smoothness" property the test above relies on to
        // justify excluding BICUBIC/MAGIC, rather than just asserting a number.
        let b_bicubic = 1.0 / 3.0;
        assert!((kernel(0.0, downscaler::BICUBIC) - (1.0 - b_bicubic / 3.0)).abs() < 1e-5);
        assert!(kernel(0.0, downscaler::BICUBIC) < 1.0, "B=1/3 must not be interpolating");
    }

    #[test]
    fn catmull_rom_matches_known_closed_form_at_half_sample() {
        // Catmull-Rom's cardinal-spline blending weights at t=0.5 are the well-known
        // (-1/16, 9/16, 9/16, -1/16); this kernel's value at x=0.5 (distance from the
        // nearest sample to the query point) is that 9/16 = 0.5625, not 0.5 -- a
        // previous version of this test asserted 0.5 from a since-corrected
        // hand-computation and would have hidden a real bug in the other direction.
        assert!((catmull_rom(0.5) - 0.5625).abs() < 1e-5, "got {}", catmull_rom(0.5));
    }

    #[test]
    fn kernels_are_symmetric() {
        for &d in &[downscaler::BICUBIC, downscaler::CATMULL_ROM, downscaler::LANCZOS3, downscaler::KAISER3] {
            for x in [0.3f32, 0.7, 1.4, 2.1] {
                assert!(
                    (kernel(x, d) - kernel(-x, d)).abs() < 1e-5,
                    "kernel {d} should be symmetric, differs at x={x}"
                );
            }
        }
    }

    #[test]
    fn kernels_vanish_beyond_their_support_radius() {
        for &d in
            &[downscaler::BICUBIC, downscaler::CATMULL_ROM, downscaler::LANCZOS2, downscaler::LANCZOS3, downscaler::KAISER3]
        {
            let r = support_radius(d);
            assert_eq!(kernel(r + 0.5, d), 0.0, "kernel {d} should be exactly 0 beyond its support radius");
        }
    }

    #[test]
    fn unknown_downscaler_falls_back_to_lanczos3() {
        assert_eq!(kernel(0.5, downscaler::FSR1), kernel(0.5, downscaler::LANCZOS3));
        assert_eq!(kernel(0.5, 999), kernel(0.5, downscaler::LANCZOS3));
    }
}
