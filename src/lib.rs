#![feature(slice_as_array)]
//! Types and traits to iterate over (packed) input data.
//!
//! The main type is [`PackedSeqVec`], that holds a sequence of 2-bit packed DNA bases. [`PackedSeq`] is a non-owned slice of packed data.
//!
//! To make libraries depending on this crate more generic, logic is encapsulated in the [`Seq`] and [`SeqVec`] traits.
//! [`Seq`] is a non-owned slice of characters that can be iterated, while [`SeqVec`] is the corresponding owned type.
//!
//! These traits serve two purposes:
//! 1. They encapsulate the packing/unpacking of characters between ASCII and the possibly different in-memory format.
//! 2. They allow efficiently iterating over 8 _chunks_ of a sequence in parallel using SIMD instructions.
//!
//! ## Sequence types
//!
//! The traits are implemented for three types.
//!
//! #### Plain ASCII sequences
//!
//! With `&[u8]: Seq` and `Vec<u8>: SeqVec`, the ASCII characters (or arbitrary
//! `u8` values, really) of any input slice can be iterated.
//!
//! #### ASCII-encoded DNA sequences
//!
//! The [`AsciiSeq: Seq`] and [`AsciiSeqVec: SeqVec`] types store a DNA sequence of `ACTGactg` characters.
//! When iterated, these are returned as `0123` values, with the mapping `A=0`, `C=1`, `T=2`, `G=3`.
//!
//! Any other characters are silently mapped to `0123` using `(c>>1) & 3`, but this should not be relied upon.
//!
//! #### Packed DNA
//!
//! The [`PackedSeq: Seq`] and [`PackedSeqVec: SeqVec`] types store a packed DNA sequence, encoded as 4 bases per byte.
//! Each `ACTG` base is stored as `0123` as above and four of these 2-bit values fill a byte.
//!
//! Use [`PackedSeqVec::from_ascii`] to construct a [`PackedSeqVec`].
//! Currently this relies on the `pext` instruction for good performance on `x86`.
//!
//! ## Parallel iterators
//!
//! This library enables iterating 8 chunks of a sequence at the same time using SIMD instructions.
//! The [`Seq::par_iter_bp`] functions return a `wide::u32x8` that contains the 2-bit or 8-bit values of the next character in each chunk in a `u32` for 8 SIMD lanes.
//!
//! This is used in the `simd-minimizers` crate, and explained in more detail in the corresponding [preprint](https://www.biorxiv.org/content/10.1101/2025.01.27.634998v1).
//!
//! #### Context
//!
//! The `context` parameter determines how much adjacent chunks overlap. When `context=1`, they are disjoint.
//! When `context=k`, adjacent chunks overlap by `k-1` characters, so that each k-mer is present in exactly one chunk.
//! Thus, this can be used to iterate all k-mers, where the first `k-1` characters in each chunk are used to initialize the first k-mer.
//!
//! #### Delayed iteration
//!
//! This crate also provides [`Seq::par_iter_bp_delayed`] and [`Seq::par_iter_bp_delayed_2`] functions. Like [`Seq::par_iter_bp`], these split the input into 8 chunks and stream over the chunks in parallel.
//! But instead of just returning a single character, they also return a second (and third) character, that is `delay` positions _behind_ the new character (at index `idx - delay`).
//! This way, k-mers can be enumerated by setting `delay=k` and then mapping e.g. `|(add, remove)| kmer = (kmer<<2) ^ add ^ (remove << (2*k))`.
//!
//! ## Example
//!
//! ```
//! use packed_seq::{SeqVec, Seq, AsciiSeqVec, PackedSeqVec, pack_char};
//! // Plain ASCII sequence.
//! let seq = b"ACTGCAGCGCATATGTAGT";
//! // ASCII DNA sequence.
//! let ascii_seq = AsciiSeqVec::from_ascii(seq);
//! // Packed DNA sequence.
//! let packed_seq = PackedSeqVec::from_ascii(seq);
//! assert_eq!(ascii_seq.len(), packed_seq.len());
//! // Iterate the ASCII characters.
//! let characters: Vec<u8> = seq.iter_bp().collect();
//! assert_eq!(characters, seq);
//!
//! // Iterate the bases with 0..4 values.
//! let bases: Vec<u8> = seq.iter().copied().map(pack_char).collect();
//! assert_eq!(bases, vec![0,1,2,3,1,0,3,1,3,1,0,2,0,2,3,2,0,3,2]);
//! let ascii_bases: Vec<u8> = ascii_seq.as_slice().iter_bp().collect();
//! assert_eq!(ascii_bases, bases);
//! let packed_bases: Vec<u8> = ascii_seq.as_slice().iter_bp().collect();
//! assert_eq!(packed_bases, bases);
//! ```
//!
//! ## Feature flags
//! - `epserde` enables `derive(epserde::Epserde)` for `PackedSeqVec` and `AsciiSeqVec`, and adds its `SerializeInner` and `DeserializeInner` traits to `SeqVec`.
//! - `pyo3` enables `derive(pyo3::pyclass)` for `PackedSeqVec` and `AsciiSeqVec`.

/// Functions with architecture-specific implementations.
mod intrinsics {
    mod transpose;
    pub use transpose::transpose;
}

mod traits;

mod ascii;
mod ascii_seq;
mod packed_seq;

#[cfg(test)]
mod test;

/// A SIMD vector containing 8 u32s.
pub use wide::u32x8;
/// The number of lanes in a `u32x8`.
pub const L: usize = 8;

pub use ascii_seq::{AsciiSeq, AsciiSeqVec};
pub use packed_seq::{
    complement_base, complement_base_simd, complement_char, pack_char, unpack_base,
};
pub use packed_seq::{PackedSeq, PackedSeqVec};
pub use traits::{Seq, SeqVec};

// For internal use only.
use core::array::from_fn;
use mem_dbg::{MemDbg, MemSize};
use rand::{RngCore, SeedableRng};
use std::{hint::assert_unchecked, ops::Range};
use wide::u32x8 as S;
