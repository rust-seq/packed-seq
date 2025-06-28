#[cfg(target_feature = "avx2")]
mod simd {
    /// SIMD vector of `u8`
    pub type VU8 = wide::u8x32;
    /// SIMD vector of `i8`
    pub type VI8 = wide::i8x32;
    /// Number of 8-bit SIMD lanes
    pub const L8: usize = 32;
    /// SIMD vector of `u16`
    pub type VU16 = wide::u16x16;
    /// SIMD vector of `i16`
    pub type VI16 = wide::i16x16;
    /// Number of 16-bit SIMD lanes
    pub const L16: usize = 16;
    /// SIMD vector of `u32`
    pub type VU32 = wide::u32x8;
    /// SIMD vector of `i32`
    pub type VI32 = wide::i32x8;
    /// Number of 32-bit SIMD lanes
    pub const L32: usize = 8;
    /// SIMD vector of `u64`
    pub type VU64 = wide::u64x4;
    /// SIMD vector of `i64`
    pub type VI64 = wide::i64x4;
    /// Number of 64-bit SIMD lanes
    pub const L64: usize = 4;
}

#[cfg(target_feature = "neon")]
mod simd {
    /// SIMD vector of `u8`
    pub type VU8 = wide::u8x16;
    /// SIMD vector of `i8`
    pub type VI8 = wide::i8x16;
    /// Number of 8-bit SIMD lanes
    pub const L8: usize = 16;
    /// SIMD vector of `u16`
    pub type VU16 = wide::u16x8;
    /// SIMD vector of `i16`
    pub type VI16 = wide::i16x8;
    /// Number of 16-bit SIMD lanes
    pub const L16: usize = 8;
    /// SIMD vector of `u32`
    pub type VU32 = wide::u32x4;
    /// SIMD vector of `i32`
    pub type VI32 = wide::i32x4;
    /// Number of 32-bit SIMD lanes
    pub const L32: usize = 4;
    /// SIMD vector of `u64`
    pub type VU64 = wide::u64x2;
    /// SIMD vector of `i64`
    pub type VI64 = wide::i64x2;
    /// Number of 64-bit SIMD lanes
    pub const L64: usize = 2;
}

#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
mod simd {
    /// SIMD vector of `u8`
    pub type VU8 = wide::u8x16;
    /// SIMD vector of `i8`
    pub type VI8 = wide::i8x16;
    /// Number of 8-bit SIMD lanes
    pub const L8: usize = 16;
    /// SIMD vector of `u16`
    pub type VU16 = wide::u16x8;
    /// SIMD vector of `i16`
    pub type VI16 = wide::i16x8;
    /// Number of 16-bit SIMD lanes
    pub const L16: usize = 8;
    /// SIMD vector of `u32`
    pub type VU32 = wide::u32x4;
    /// SIMD vector of `i32`
    pub type VI32 = wide::i32x4;
    /// Number of 32-bit SIMD lanes
    pub const L32: usize = 4;
    /// SIMD vector of `u64`
    pub type VU64 = wide::u64x2;
    /// SIMD vector of `i64`
    pub type VI64 = wide::i64x2;
    /// Number of 64-bit SIMD lanes
    pub const L64: usize = 2;
}

pub use simd::*;
