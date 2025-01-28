//! Types and traits for 2-bit packed size-4 (DNA) alphabets.
//!
//! What this library does:
//! - Allowing APIs to take an `impl Seq` type that provides a `iter_bp` function to iterate all bases in the sequence as `u8` values in `0..4`.
//! - Providing `PackedSeqVec` storage of packed sequence and `PackedSeq` slices over packed data.
//! - Efficiently converting from ASCII representation to `PackedSeqVec`.
//!
//! What this library does not:
//! - Handling non-ACGT characters.
//!
//! TODO: Currently this relies on the `pext` instruction for good performance on `x86`.
//!       Alternatively, the multiplication trick from [1] should be implemented.
//!
//! 1: https://github.com/Daniel-Liu-c0deb0t/cute-nucleotides/blob/master/src/n_to_bits.rs#L213

mod intrinsics {
    mod deinterleave;
    mod gather;

    pub use deinterleave::deinterleave;
    pub use gather::gather;
}

mod traits;

mod ascii;
mod ascii_seq;
mod packed_seq;

#[cfg(test)]
mod test;

use core::{array::from_fn, mem::transmute};
use mem_dbg::{MemDbg, MemSize};
use rand::Rng;
use std::{hint::assert_unchecked, ops::Range};
use wide::{u32x8, u64x4};

/// A SIMD vector containing 8 u32s.
pub use wide::u32x8 as S;
/// The number of lanes in `S`.
pub const L: usize = 8;

pub use ascii_seq::{AsciiSeq, AsciiSeqVec};
pub use packed_seq::{
    complement_base, complement_base_simd, complement_char, pack_char, unpack_base,
};
pub use packed_seq::{PackedSeq, PackedSeqVec};
pub use traits::{Seq, SeqVec};
