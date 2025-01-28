//! Traits and types to generically iterate over possibly packed input data.
//!
//! Main traits are `Seq` and `SeqVec`.
//! `Seq` is a non-owned slice of characters that can be iterated, while `SeqVec` is the corresponding owned type.
//!
//! These traits serve two purposes:
//! 1. They encapsulate the packing/unpacking of characters between ASCII and the possibly different in-memory format.
//! 2. They allow efficiently iterating over 8 _chunks_ of a sequence in parallel using SIMD instructions.
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
//! The `AsciiSeq: Seq` and `AsciiSeqVec: SeqVec` types store a DNA sequence of `ACTGactg` characters.
//! When iterated, these are returned as `0123` values, with the mapping `A=0`, `C=1`, `T=2`, `G=3`.
//!
//! Any other characters are silently mapped to `0123` using `(c>>1) & 3`, but this should not be relied upon.
//!
//! #### Packed DNA
//!
//! The `PackedSeq: Seq` and `PackedSeqVec: SeqVec` types store a packed DNA sequence, encoded as 4 bases per byte.
//! Each `ACTG` base is stored as `0123` as above and four of these 2-bit values fill a byte.
//!
//! Use `PackedSeqVec::from_ascii` to construct a `PackedSeqVec`.
//! Currently this relies on the `pext` instruction for good performance on `x86`.
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
//!
//! ```

/// Functions with architecture-specific implementations.
#[allow(unused)]
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
