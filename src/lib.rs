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

mod ascii;
mod ascii_seq;
mod intrinsics;
mod packed_seq;

#[cfg(test)]
mod test;

use core::{array::from_fn, mem::transmute};
use mem_dbg::{MemDbg, MemSize};
use rand::Rng;
use std::{hint::assert_unchecked, ops::Range};
use wide::u64x4;

/// A SIMD vector containing 8 u32s.
pub use wide::u32x8 as S;
/// The number of lanes in `S`.
pub const L: usize = 8;

// ---------------- TRAITS ----------------

/// Interface to sequences over a 2-bit alphabet.
///
/// Currently supports `&[u8]`, where each `u8` must be in `0..4`, and the
/// `PackedSeq` type that contains packed sequences.
pub trait Seq<'s>: Copy + Eq + Ord {
    const BASES_PER_BYTE: usize;
    type SeqVec: SeqVec;

    /// The length of the sequence in bp.
    fn len(&self) -> usize;

    /// Get the packed character at the given index.
    fn get(&self, _index: usize) -> u8;

    /// Get the ASCII character at the given index, _without_ packing it.
    /// Not implemented for packed data.
    fn get_ascii(&self, _index: usize) -> u8;

    /// Convert a short sequence (kmer) to a packed word.
    /// Panics if `self` is longer than 29 characters.
    fn to_word(&self) -> usize;

    /// Convert to an owned version.
    fn to_vec(&self) -> Self::SeqVec;

    /// Get a sub-slice of the sequence.
    fn slice(&self, range: Range<usize>) -> Self;

    /// A simple iterator over characters.
    /// Returns u8 values in [0, 4).
    fn iter_bp(self) -> impl ExactSizeIterator<Item = u8> + Clone;

    /// An iterator that splits the input into 8 chunks and streams over them in parallel.
    /// Returns a separate `tail` iterator over the remaining characters.
    /// The context can be e.g. the k-mer size being iterated. When `context>1`, consecutive chunk overlap by `context-1` bases.
    fn par_iter_bp(self, context: usize) -> (impl ExactSizeIterator<Item = S> + Clone, Self);

    fn par_iter_bp_delayed(
        self,
        _context: usize,
        _delay: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S)> + Clone, Self);

    fn par_iter_bp_delayed_2(
        self,
        _context: usize,
        _delay1: usize,
        _delay2: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S, S)> + Clone, Self);

    /// Compare and return the LCP of the two sequences.
    fn cmp_lcp(&self, other: &Self) -> (std::cmp::Ordering, usize);
}

// Some hacky stuff to make conditional supertraits.
cfg_if::cfg_if! {
    if #[cfg(feature = "epserde")] {
        pub use serde::{DeserializeInner, SerializeInner};
    } else {
        pub trait SerializeInner {}
        pub trait DeserializeInner {}

        impl SerializeInner for Vec<u8> {}
        impl DeserializeInner for Vec<u8> {}
        impl SerializeInner for AsciiSeqVec {}
        impl DeserializeInner for AsciiSeqVec {}
        impl SerializeInner for PackedSeqVec {}
        impl DeserializeInner for PackedSeqVec {}
    }
}

pub trait SeqVec: Default + Sync + SerializeInner + DeserializeInner + MemSize + MemDbg {
    type Seq<'s>: Seq<'s>;

    fn as_slice(&self) -> Self::Seq<'_>;

    /// Get a sub-slice of the sequence.
    fn slice(&self, range: Range<usize>) -> Self::Seq<'_> {
        self.as_slice().slice(range)
    }

    /// The length of the sequence in bp.
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Convert into the underlying raw representation.
    fn into_raw(self) -> Vec<u8>;

    /// Append the given sequence to the underlying storage.
    /// This may leave gaps (padding) between consecutively pushed sequences to avoid re-aligning the pushed data.
    /// Returns the range of indices corresponding to the pushed sequence.
    /// Use `self.as_slice()[range]` to get the corresponding slice.
    fn push_seq(&mut self, seq: Self::Seq<'_>) -> Range<usize>;

    /// Append the given ASCII sequence to the underlying storage.
    /// This may leave gaps (padding) between consecutively pushed sequences to avoid re-aligning the pushed data.
    /// Returns the range of indices corresponding to the pushed sequence.
    /// Use `self.as_slice()[range]` to get the corresponding slice.
    fn push_ascii(&mut self, seq: &[u8]) -> Range<usize>;

    /// Create an `PackedSeqVec` from an ASCII sequence. See `push_ascii` for details.
    fn from_ascii(seq: &[u8]) -> Self {
        let mut packed_vec = Self::default();
        packed_vec.push_ascii(seq);
        packed_vec
    }

    fn from_seqs<'a>(input_seqs: impl Iterator<Item = Self::Seq<'a>>) -> Self {
        let mut seq = Self::default();
        input_seqs.for_each(|slice| {
            seq.push_seq(slice);
        });
        seq
    }

    fn random(n: usize) -> Self;
}

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
