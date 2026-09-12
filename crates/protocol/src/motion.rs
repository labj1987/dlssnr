//! Motion payload: full-resolution little-endian R16G16_SFLOAT, current -> previous.
use crate::enums::mvec_scale_mode;
pub fn scales(mode: u32, width: u32, height: u32) -> [f32; 2] {
    match mode {
        mvec_scale_mode::NORMALIZED => [width as f32 / 2.0, height as f32 / 2.0],
        mvec_scale_mode::UV01 => [width as f32, height as f32],
        _ => [1.0, 1.0],
    }
}
/// IEEE binary16 round-to-nearest-even. Nonfinite motion is discarded.
pub fn half(value: f32) -> u16 {
    if !value.is_finite() { return 0; }
    let bits = value.clamp(-65504.0,65504.0).to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 255) as i32 - 127 + 15;
    let mantissa = bits & 0x7fffff;
    if exponent < -10 { return sign; }
    if exponent <= 0 {
        let mantissa = mantissa | 0x800000;
        let shift = (14-exponent) as u32;
        let rounded = (mantissa + ((1 << (shift-1))-1) + ((mantissa >> shift)&1)) >> shift;
        return sign | rounded as u16;
    }
    let rounded = mantissa + 0xfff + ((mantissa >> 13)&1);
    sign | (((exponent as u32) << 10) + (rounded >> 13)) as u16
}
pub fn encode(vectors: &[[f32;2]], scale: [f32;2]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vectors.len()*4);
    for vector in vectors { for axis in 0..2 {
        let v = if vector[axis].abs() < 0.5 {0.0} else {vector[axis] / scale[axis].max(1.0)};
        out.extend_from_slice(&half(v).to_le_bytes());
    }}
    out
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn known_half_values_and_rounding() {
        for (value,bits) in [(0.0,0), (1.0,0x3c00),(-8.0,0xc800),(0.5,0x3800),(65504.0,0x7bff),(2.0f32.powi(-24),1)] { assert_eq!(half(value),bits); }
        assert_eq!(half(1.0+2.0f32.powi(-11)),0x3c00);
        assert_eq!(half(f32::NAN),0);
    }
    #[test] fn pixel_direction_units_and_deadzone() {
        assert_eq!(encode(&[[-8.0,0.25]], [1.0,1.0]), [0,0xc8,0,0]);
        assert_eq!(scales(mvec_scale_mode::UV01,512,256),[512.0,256.0]);
        assert_eq!(scales(mvec_scale_mode::NORMALIZED,512,256),[256.0,128.0]);
    }
}
