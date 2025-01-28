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

pub use traits::{Seq, SeqVec};

// ---------------- STRUCTS ----------------

/// A `Vec<u8>` representing an ASCII-encoded DNA sequence of `ACGTacgt`.
///
/// Other characters will be mapped into `[0, 4)` via `(c>>1)&3`, or may cause panics.
#[derive(Clone, Debug, Default, MemSize, MemDbg)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
pub struct AsciiVec {
    pub seq: Vec<u8>,
}

/// A `&[u8]` representing an ASCII-encoded DNA sequence of `ACGTacgt`.
///
/// Other characters will be mapped into `[0, 4)` via `(c>>1)&3`, or may cause panics.
#[derive(Copy, Clone, Debug, MemSize, MemDbg, PartialEq, Eq, PartialOrd, Ord)]
pub struct AsciiSeq<'s>(pub &'s [u8]);

/// A `Vec<u8>` representing an ASCII-encoded DNA sequence of `ACGTacgt`.
///
/// Other characters will be mapped into `[0, 4)` via `(c>>1)&3`, or may cause panics.
#[derive(Clone, Debug, Default, MemSize, MemDbg)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
pub struct AsciiSeqVec {
    pub seq: Vec<u8>,
}

/// A 2-bit packed sequence representation.
#[derive(Copy, Clone, Debug, MemSize, MemDbg)]
pub struct PackedSeq<'s> {
    /// Packed data.
    pub seq: &'s [u8],
    /// Offset in bp from the start of the `seq`.
    pub offset: usize,
    /// Length of the sequence in bp, starting at `offset` from the start of `seq`.
    pub len: usize,
}

#[derive(Clone, Debug, Default, MemSize, MemDbg)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
pub struct PackedSeqVec {
    pub seq: Vec<u8>,
    pub len: usize,
}

// ============================= PACKED ================================

pub fn pack_char(base: u8) -> u8 {
    match base {
        b'a' | b'A' => 0,
        b'c' | b'C' => 1,
        b'g' | b'G' => 3,
        b't' | b'T' => 2,
        _ => panic!(
            "Unexpected character '{}' with ASCII value {base}. Expected one of ACTGactg.",
            base as char
        ),
    }
}

pub fn unpack(base: u8) -> u8 {
    debug_assert!(base < 4, "Base {base} is not <4.");
    b"ACTG"[base as usize]
}

pub const fn complement_char(base: u8) -> u8 {
    match base {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        _ => panic!("Unexpected character. Expected one of ACTGactg.",),
    }
}

pub const fn complement_base(base: u8) -> u8 {
    base ^ 2
}

pub fn complement_base_simd(base: u32x8) -> u32x8 {
    base ^ u32x8::splat(2)
}
