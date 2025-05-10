#![allow(unused)]

// use wide::u32x4;

use std::arch::x86_64::__m512;
use std::arch::x86_64::_mm512_shuffle_f32x4;
use std::arch::x86_64::_mm512_shuffle_ps;
use std::arch::x86_64::_mm512_unpackhi_ps;
use std::arch::x86_64::_mm512_unpacklo_ps;
use std::mem::transmute;

use crate::L;
use crate::S;

/// Transpose a matrix of 8 SIMD vectors.
/// <https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2>
// TODO: Investigate other transpose functions mentioned there?
pub fn transpose(m: [S; L]) -> [S; L] {
    _transpose512(m)
}

/// A utility function for creating masks to use with Intel shuffle and
/// permute intrinsics.
///
/// Copied from the standard library, since it is unstable.
const fn _mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
fn _transpose512(m: [S; L]) -> [S; L] {
    unsafe {
        let m: [__m512; 16] = transmute(m);
        let _tmp0 = _mm512_unpacklo_ps(m[0], m[1]);
        let _tmp1 = _mm512_unpackhi_ps(m[0], m[1]);
        let _tmp2 = _mm512_unpacklo_ps(m[2], m[3]);
        let _tmp3 = _mm512_unpackhi_ps(m[2], m[3]);
        let _tmp4 = _mm512_unpacklo_ps(m[4], m[5]);
        let _tmp5 = _mm512_unpackhi_ps(m[4], m[5]);
        let _tmp6 = _mm512_unpacklo_ps(m[6], m[7]);
        let _tmp7 = _mm512_unpackhi_ps(m[6], m[7]);
        let _tmp8 = _mm512_unpacklo_ps(m[8], m[9]);
        let _tmp9 = _mm512_unpackhi_ps(m[8], m[9]);
        let _tmpa = _mm512_unpacklo_ps(m[10], m[11]);
        let _tmpb = _mm512_unpackhi_ps(m[10], m[11]);
        let _tmpc = _mm512_unpacklo_ps(m[12], m[13]);
        let _tmpd = _mm512_unpackhi_ps(m[12], m[13]);
        let _tmpe = _mm512_unpacklo_ps(m[14], m[15]);
        let _tmpf = _mm512_unpackhi_ps(m[14], m[15]);

        let _tmpg = _mm512_shuffle_ps(_tmp0, _tmp2, _mm_shuffle(1, 0, 1, 0));
        let _tmph = _mm512_shuffle_ps(_tmp0, _tmp2, _mm_shuffle(3, 2, 3, 2));
        let _tmpi = _mm512_shuffle_ps(_tmp1, _tmp3, _mm_shuffle(1, 0, 1, 0));
        let _tmpj = _mm512_shuffle_ps(_tmp1, _tmp3, _mm_shuffle(3, 2, 3, 2));
        let _tmpk = _mm512_shuffle_ps(_tmp4, _tmp6, _mm_shuffle(1, 0, 1, 0));
        let _tmpl = _mm512_shuffle_ps(_tmp4, _tmp6, _mm_shuffle(3, 2, 3, 2));
        let _tmpm = _mm512_shuffle_ps(_tmp5, _tmp7, _mm_shuffle(1, 0, 1, 0));
        let _tmpn = _mm512_shuffle_ps(_tmp5, _tmp7, _mm_shuffle(3, 2, 3, 2));
        let _tmpo = _mm512_shuffle_ps(_tmp8, _tmpa, _mm_shuffle(1, 0, 1, 0));
        let _tmpp = _mm512_shuffle_ps(_tmp8, _tmpa, _mm_shuffle(3, 2, 3, 2));
        let _tmpq = _mm512_shuffle_ps(_tmp9, _tmpb, _mm_shuffle(1, 0, 1, 0));
        let _tmpr = _mm512_shuffle_ps(_tmp9, _tmpb, _mm_shuffle(3, 2, 3, 2));
        let _tmps = _mm512_shuffle_ps(_tmpc, _tmpe, _mm_shuffle(1, 0, 1, 0));
        let _tmpt = _mm512_shuffle_ps(_tmpc, _tmpe, _mm_shuffle(3, 2, 3, 2));
        let _tmpu = _mm512_shuffle_ps(_tmpd, _tmpf, _mm_shuffle(1, 0, 1, 0));
        let _tmpv = _mm512_shuffle_ps(_tmpd, _tmpf, _mm_shuffle(3, 2, 3, 2));

        let _tmp0 = _mm512_shuffle_f32x4(_tmpg, _tmpk, _mm_shuffle(2, 0, 2, 0));
        let _tmp1 = _mm512_shuffle_f32x4(_tmpo, _tmps, _mm_shuffle(2, 0, 2, 0));
        let _tmp2 = _mm512_shuffle_f32x4(_tmph, _tmpl, _mm_shuffle(2, 0, 2, 0));
        let _tmp3 = _mm512_shuffle_f32x4(_tmpp, _tmpt, _mm_shuffle(2, 0, 2, 0));
        let _tmp4 = _mm512_shuffle_f32x4(_tmpi, _tmpm, _mm_shuffle(2, 0, 2, 0));
        let _tmp5 = _mm512_shuffle_f32x4(_tmpq, _tmpu, _mm_shuffle(2, 0, 2, 0));
        let _tmp6 = _mm512_shuffle_f32x4(_tmpj, _tmpn, _mm_shuffle(2, 0, 2, 0));
        let _tmp7 = _mm512_shuffle_f32x4(_tmpr, _tmpv, _mm_shuffle(2, 0, 2, 0));
        let _tmp8 = _mm512_shuffle_f32x4(_tmpg, _tmpk, _mm_shuffle(3, 1, 3, 1));
        let _tmp9 = _mm512_shuffle_f32x4(_tmpo, _tmps, _mm_shuffle(3, 1, 3, 1));
        let _tmpa = _mm512_shuffle_f32x4(_tmph, _tmpl, _mm_shuffle(3, 1, 3, 1));
        let _tmpb = _mm512_shuffle_f32x4(_tmpp, _tmpt, _mm_shuffle(3, 1, 3, 1));
        let _tmpc = _mm512_shuffle_f32x4(_tmpi, _tmpm, _mm_shuffle(3, 1, 3, 1));
        let _tmpd = _mm512_shuffle_f32x4(_tmpq, _tmpu, _mm_shuffle(3, 1, 3, 1));
        let _tmpe = _mm512_shuffle_f32x4(_tmpj, _tmpn, _mm_shuffle(3, 1, 3, 1));
        let _tmpf = _mm512_shuffle_f32x4(_tmpr, _tmpv, _mm_shuffle(3, 1, 3, 1));

        let mut t: [__m512; 16] = [transmute([0; 16]); 16];
        t[0] = _mm512_shuffle_f32x4(_tmp0, _tmp1, _mm_shuffle(2, 0, 2, 0));
        t[1] = _mm512_shuffle_f32x4(_tmp2, _tmp3, _mm_shuffle(2, 0, 2, 0));
        t[2] = _mm512_shuffle_f32x4(_tmp4, _tmp5, _mm_shuffle(2, 0, 2, 0));
        t[3] = _mm512_shuffle_f32x4(_tmp6, _tmp7, _mm_shuffle(2, 0, 2, 0));
        t[4] = _mm512_shuffle_f32x4(_tmp8, _tmp9, _mm_shuffle(2, 0, 2, 0));
        t[5] = _mm512_shuffle_f32x4(_tmpa, _tmpb, _mm_shuffle(2, 0, 2, 0));
        t[6] = _mm512_shuffle_f32x4(_tmpc, _tmpd, _mm_shuffle(2, 0, 2, 0));
        t[7] = _mm512_shuffle_f32x4(_tmpe, _tmpf, _mm_shuffle(2, 0, 2, 0));
        t[8] = _mm512_shuffle_f32x4(_tmp0, _tmp1, _mm_shuffle(3, 1, 3, 1));
        t[9] = _mm512_shuffle_f32x4(_tmp2, _tmp3, _mm_shuffle(3, 1, 3, 1));
        t[10] = _mm512_shuffle_f32x4(_tmp4, _tmp5, _mm_shuffle(3, 1, 3, 1));
        t[11] = _mm512_shuffle_f32x4(_tmp6, _tmp7, _mm_shuffle(3, 1, 3, 1));
        t[12] = _mm512_shuffle_f32x4(_tmp8, _tmp9, _mm_shuffle(3, 1, 3, 1));
        t[13] = _mm512_shuffle_f32x4(_tmpa, _tmpb, _mm_shuffle(3, 1, 3, 1));
        t[14] = _mm512_shuffle_f32x4(_tmpc, _tmpd, _mm_shuffle(3, 1, 3, 1));
        t[15] = _mm512_shuffle_f32x4(_tmpe, _tmpf, _mm_shuffle(3, 1, 3, 1));
        transmute(t)
    }
}

// // NOTE: AVX is sufficient here. AVX2 is not needed.
// #[inline(always)]
// #[cfg(target_feature = "avx")]
// fn _transpose(m: [S; 8]) -> [S; 8] {
//     unsafe {
//         #[cfg(target_arch = "x86")]
//         use core::arch::x86::*;
//         #[cfg(target_arch = "x86_64")]
//         use core::arch::x86_64::*;
//         use core::mem::transmute;

//         let m: [__m256; 8] = transmute(m);
//         let x0 = _mm256_unpacklo_ps(m[0], m[1]);
//         let x1 = _mm256_unpackhi_ps(m[0], m[1]);
//         let x2 = _mm256_unpacklo_ps(m[2], m[3]);
//         let x3 = _mm256_unpackhi_ps(m[2], m[3]);
//         let x4 = _mm256_unpacklo_ps(m[4], m[5]);
//         let x5 = _mm256_unpackhi_ps(m[4], m[5]);
//         let x6 = _mm256_unpacklo_ps(m[6], m[7]);
//         let x7 = _mm256_unpackhi_ps(m[6], m[7]);
//         let y0 = _mm256_shuffle_ps(x0, x2, _mm_shuffle(1, 0, 1, 0));
//         let y1 = _mm256_shuffle_ps(x0, x2, _mm_shuffle(3, 2, 3, 2));
//         let y2 = _mm256_shuffle_ps(x1, x3, _mm_shuffle(1, 0, 1, 0));
//         let y3 = _mm256_shuffle_ps(x1, x3, _mm_shuffle(3, 2, 3, 2));
//         let y4 = _mm256_shuffle_ps(x4, x6, _mm_shuffle(1, 0, 1, 0));
//         let y5 = _mm256_shuffle_ps(x4, x6, _mm_shuffle(3, 2, 3, 2));
//         let y6 = _mm256_shuffle_ps(x5, x7, _mm_shuffle(1, 0, 1, 0));
//         let y7 = _mm256_shuffle_ps(x5, x7, _mm_shuffle(3, 2, 3, 2));
//         let mut t: [__m256; 8] = [transmute([0; 8]); 8];
//         t[0] = _mm256_permute2f128_ps(y0, y4, 0x20);
//         t[1] = _mm256_permute2f128_ps(y1, y5, 0x20);
//         t[2] = _mm256_permute2f128_ps(y2, y6, 0x20);
//         t[3] = _mm256_permute2f128_ps(y3, y7, 0x20);
//         t[4] = _mm256_permute2f128_ps(y0, y4, 0x31);
//         t[5] = _mm256_permute2f128_ps(y1, y5, 0x31);
//         t[6] = _mm256_permute2f128_ps(y2, y6, 0x31);
//         t[7] = _mm256_permute2f128_ps(y3, y7, 0x31);
//         transmute(t)
//     }
// }

// #[inline(always)]
// #[cfg(target_feature = "neon")]
// fn _transpose(m: [S; 8]) -> [S; 8] {
//     unsafe {
//         use core::mem::transmute;

//         let m: [u32x4; 16] = transmute(m);
//         let t11 = transpose_4x4_neon(
//             *m.get_unchecked(0),
//             *m.get_unchecked(2),
//             *m.get_unchecked(4),
//             *m.get_unchecked(6),
//         );
//         let t21 = transpose_4x4_neon(
//             *m.get_unchecked(1),
//             *m.get_unchecked(3),
//             *m.get_unchecked(5),
//             *m.get_unchecked(7),
//         );
//         let t12 = transpose_4x4_neon(
//             *m.get_unchecked(8),
//             *m.get_unchecked(10),
//             *m.get_unchecked(12),
//             *m.get_unchecked(14),
//         );
//         let t22 = transpose_4x4_neon(
//             *m.get_unchecked(9),
//             *m.get_unchecked(11),
//             *m.get_unchecked(13),
//             *m.get_unchecked(15),
//         );

//         transmute((
//             *t11.get_unchecked(0),
//             *t12.get_unchecked(0),
//             *t11.get_unchecked(1),
//             *t12.get_unchecked(1),
//             *t11.get_unchecked(2),
//             *t12.get_unchecked(2),
//             *t11.get_unchecked(3),
//             *t12.get_unchecked(3),
//             *t21.get_unchecked(0),
//             *t22.get_unchecked(0),
//             *t21.get_unchecked(1),
//             *t22.get_unchecked(1),
//             *t21.get_unchecked(2),
//             *t22.get_unchecked(2),
//             *t21.get_unchecked(3),
//             *t22.get_unchecked(3),
//         ))
//     }
// }

// #[inline(always)]
// #[cfg(target_feature = "neon")]
// fn transpose_4x4_neon(m0: u32x4, m1: u32x4, m2: u32x4, m3: u32x4) -> [u32x4; 4] {
//     unsafe {
//         use core::arch::aarch64::vzipq_u32;
//         use core::mem::transmute;

//         let x = vzipq_u32(transmute(m0), transmute(m2));
//         let y = vzipq_u32(transmute(m1), transmute(m3));
//         transmute((vzipq_u32(x.0, y.0), vzipq_u32(x.1, y.1)))
//     }
// }

// #[inline(always)]
// #[cfg(not(any(target_feature = "avx", target_feature = "neon")))]
// fn _transpose(m: [S; 8]) -> [S; 8] {
//     unsafe {
//         let m = m.map(|v| v.to_array());
//         [0, 1, 2, 3, 4, 5, 6, 7].map(|i| S::new(m.map(|v| *v.get_unchecked(i))))
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_transpose() {
//         let m = [
//             S::new([0, 1, 2, 3, 4, 5, 6, 7]),
//             S::new([10, 11, 12, 13, 14, 15, 16, 17]),
//             S::new([20, 21, 22, 23, 24, 25, 26, 27]),
//             S::new([30, 31, 32, 33, 34, 35, 36, 37]),
//             S::new([40, 41, 42, 43, 44, 45, 46, 47]),
//             S::new([50, 51, 52, 53, 54, 55, 56, 57]),
//             S::new([60, 61, 62, 63, 64, 65, 66, 67]),
//             S::new([70, 71, 72, 73, 74, 75, 76, 77]),
//         ];

//         let mt = [
//             S::new([0, 10, 20, 30, 40, 50, 60, 70]),
//             S::new([1, 11, 21, 31, 41, 51, 61, 71]),
//             S::new([2, 12, 22, 32, 42, 52, 62, 72]),
//             S::new([3, 13, 23, 33, 43, 53, 63, 73]),
//             S::new([4, 14, 24, 34, 44, 54, 64, 74]),
//             S::new([5, 15, 25, 35, 45, 55, 65, 75]),
//             S::new([6, 16, 26, 36, 46, 56, 66, 76]),
//             S::new([7, 17, 27, 37, 47, 57, 67, 77]),
//         ];

//         assert_eq!(transpose(m), mt);
//     }
// }
