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

mod intrinsics;

use core::{array::from_fn, mem::transmute};
use epserde::{deser::DeserializeInner, ser::SerializeInner, Epserde};
use mem_dbg::{MemDbg, MemSize};
use std::ops::Range;
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
pub trait Seq: Copy {
    const BASES_PER_BYTE: usize;
    type SeqVec: SeqVec;

    /// The length of the sequence in bp.
    fn len(&self) -> usize;

    /// Convert a short sequence (kmer) to a packed word.
    /// Panics if `self` is longer than 29 characters.
    fn to_word(&self) -> usize;

    /// Convert to an owned version.
    fn to_vec(&self) -> Self::SeqVec;

    /// Get a sub-slice of the sequence.
    fn slice(&self, range: Range<usize>) -> Self;

    /// A simple iterator over characters.
    /// Returns u8 values in [0, 4).
    fn iter_bp(self) -> impl ExactSizeIterator<Item = u8>;

    /// An iterator that splits the input into 8 chunks and streams over them in parallel.
    /// Returns a separate `tail` iterator over the remaining characters.
    /// The context can be e.g. the k-mer size being iterated. When `context>1`, consecutive chunk overlap by `context-1` bases.
    fn par_iter_bp(self, context: usize) -> (impl ExactSizeIterator<Item = S>, Self);
}

pub trait SeqVec: Default + Sync + SerializeInner + DeserializeInner {
    type Seq<'s>: Seq;

    fn as_slice(&self) -> Self::Seq<'_>;

    /// Get a sub-slice of the sequence.
    fn slice(&self, range: Range<usize>) -> Self::Seq<'_> {
        self.as_slice().slice(range)
    }

    /// The length of the sequence in bp.
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Append the given sequence to the underlying storage.
    /// This may leave gaps (padding) between consecutively pushed sequences to avoid re-aligning the pushed data.
    /// Returns the range of indices corresponding to the pushed sequence.
    /// Use `self.as_slice()[range]` to get the corresponding slice.
    fn push_seq(&mut self, seq: Self::Seq<'_>) -> Range<usize>;

    fn from_seqs<'a>(input_seqs: impl Iterator<Item = Self::Seq<'a>>) -> (Self, Vec<Range<usize>>) {
        let mut seq = Self::default();
        let ranges = input_seqs.map(|slice| seq.push_seq(slice)).collect();
        (seq, ranges)
    }

    fn random(n: usize) -> Self;
}

// ---------------- STRUCTS ----------------

/// A `&[u8]` representing an ASCII sequence.
/// Only supported characters are `ACGTacgt`.
/// Other characters will be silently mapped into `[0, 4)`, or may cause panics.
#[derive(Copy, Clone, Debug, MemSize, MemDbg, PartialEq, Eq)]
pub struct AsciiSeq<'s>(pub &'s [u8]);

/// An owned ASCII sequence.
/// Only supported characters are `ACGTacgt`.
/// Other characters will be silently mapped into `[0, 4)`, or may cause panics.
///
/// TODO: Should this be a strong type instead?
#[derive(Clone, Debug, Default, Epserde, MemSize, MemDbg)]
#[cfg_attr(feature = "python-bindings", pyo3::pyclass)]
pub struct AsciiSeqVec {
    pub seq: Vec<u8>,
    pub ranges: Vec<(usize, usize)>,
}

/// A 2-bit packed sequence representation.
#[derive(Copy, Clone)]
pub struct PackedSeq<'s> {
    /// Packed data.
    pub seq: &'s [u8],
    /// Offset in bp from the start of the `seq`.
    pub offset: usize,
    /// Length of the sequence in bp, starting at `offset` from the start of `seq`.
    pub len: usize,
}

#[derive(Debug, Default, Epserde)]
pub struct PackedSeqVec {
    pub seq: Vec<u8>,
    pub len: usize,
    pub ranges: Vec<(usize, usize)>,
}

// ---------------- IMPLEMENTATIONS ----------------

/// Maps ASCII to `[0, 4)` on the fly.
/// Prefer first packing into a `PackedSeqVec` for storage.
impl<'s> Seq for AsciiSeq<'s> {
    const BASES_PER_BYTE: usize = 1;
    type SeqVec = AsciiSeqVec;

    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn to_word(&self) -> usize {
        let len = self.len();
        assert!(len <= usize::BITS as usize / 2);

        let mut val = 0u64;

        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            for i in (0..len).step_by(8) {
                let packed_bytes = if i + 8 <= self.len() {
                    let chunk: &[u8; 8] = &self.0[i..i + 8].try_into().unwrap();
                    let ascii = u64::from_ne_bytes(*chunk);
                    unsafe { std::arch::x86_64::_pext_u64(ascii, 0x0606060606060606) }
                } else {
                    let mut chunk: [u8; 8] = [0; 8];
                    // Copy only part of the slice to avoid out-of-bounds indexing.
                    chunk[..self.len() - i].copy_from_slice(self.0[i..].try_into().unwrap());
                    let ascii = u64::from_ne_bytes(chunk);
                    unsafe { std::arch::x86_64::_pext_u64(ascii, 0x0606060606060606) }
                };
                val |= packed_bytes << (i * 2);
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            for (i, &base) in self.0.iter().enumerate().skip(head) {
                val |= match base {
                    b'a' | b'A' => 0,
                    b'c' | b'C' => 1,
                    b'g' | b'G' => 3,
                    b't' | b'T' => 2,
                    _ => panic!(
                    "Unexpected character '{}' with ASCII value {base}. Expected one of ACTGactg.",
                    base as char
                ),
                } << (i * 2);
            }
        }

        val as usize
    }

    /// Convert to an owned version.
    fn to_vec(&self) -> AsciiSeqVec {
        AsciiSeqVec {
            seq: self.0.to_vec(),
            ranges: vec![(0, self.len())],
        }
    }

    #[inline(always)]
    fn slice(&self, range: Range<usize>) -> Self {
        Self(&self.0[range])
    }

    /// Iterate the basepairs in the sequence, assuming values in `0..4`.
    ///
    /// NOTE: This is only efficient on x86_64 with `BMI2` support for `pext`.
    #[inline(always)]
    fn iter_bp(self) -> impl ExactSizeIterator<Item = u8> {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            let mut cache = 0;
            (0..self.len()).map(move |i| {
                if i % 8 == 0 {
                    if i + 8 <= self.len() {
                        let chunk: &[u8; 8] = &self.0[i..i + 8].try_into().unwrap();
                        let ascii = u64::from_ne_bytes(*chunk);
                        cache = unsafe { std::arch::x86_64::_pext_u64(ascii, 0x0606060606060606) };
                    } else {
                        let mut chunk: [u8; 8] = [0; 8];
                        // Copy only part of the slice to avoid out-of-bounds indexing.
                        chunk[..self.len() - i].copy_from_slice(self.0[i..].try_into().unwrap());
                        let ascii = u64::from_ne_bytes(chunk);
                        cache = unsafe { std::arch::x86_64::_pext_u64(ascii, 0x0606060606060606) };
                    }
                }
                let base = cache & 0x03;
                cache >>= 2;
                base as u8
            })
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        self.iter().map(|base| match base {
            b'a' | b'A' => 0,
            b'c' | b'C' => 1,
            b'g' | b'G' => 3,
            b't' | b'T' => 2,
            _ => panic!(),
        })
    }

    /// Iterate the basepairs in the sequence in 8 parallel streams, assuming values in `0..4`.
    #[inline(always)]
    fn par_iter_bp(self, context: usize) -> (impl ExactSizeIterator<Item = S>, Self) {
        let num_kmers = self.len().saturating_sub(context - 1);
        let n = num_kmers / L;

        let base_ptr = self.0.as_ptr();
        let offsets_lanes_0_4: u64x4 = from_fn(|l| (l * n) as u64).into();
        let offsets_lanes_4_8: u64x4 = from_fn(|l| ((4 + l) * n) as u64).into();
        let mut upcoming_1 = S::ZERO;
        let mut upcoming_2 = S::ZERO;

        let it = (0..if num_kmers == 0 { 0 } else { n + context - 1 }).map(move |i| {
            if i % 4 == 0 {
                if i % 8 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat(i as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat(i as u64);
                    let u64_0_4: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_0_4)) };
                    let u64_4_8: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_4_8)) };
                    // Split into two vecs containing a u32 of 4 characters each.
                    (upcoming_1, upcoming_2) = intrinsics::deinterleave(u64_0_4, u64_4_8);
                    // TODO: Use packing to avoid needing two vectors?
                    let mask = 0x06060606;
                    upcoming_1 &= S::splat(mask);
                    upcoming_2 &= S::splat(mask);
                    upcoming_1 = upcoming_1 >> S::splat(1);
                    upcoming_2 = upcoming_2 >> S::splat(1);
                } else {
                    // Move on to the next u32 containing 4 buffered characters.
                    upcoming_1 = upcoming_2;
                }
            }
            // Extract the last 2 bits of each character.
            let chars = upcoming_1 & S::splat(0x03);
            // Shift remaining characters to the right.
            upcoming_1 = upcoming_1 >> S::splat(8);
            chars
        });

        (it, Self(&self.0[L * n..]))
    }
}

impl<'s> PackedSeq<'s> {
    /// Shrink `seq` to only just cover the data.
    #[inline(always)]
    pub fn normalize(&self) -> Self {
        let start = self.offset / 4;
        let end = (self.offset + self.len).div_ceil(4);
        Self {
            seq: &self.seq[start..end],
            offset: self.offset % 4,
            len: self.len,
        }
    }
}

impl<'s> Seq for PackedSeq<'s> {
    const BASES_PER_BYTE: usize = 4;
    type SeqVec = PackedSeqVec;

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn to_word(&self) -> usize {
        assert!(self.len() <= usize::BITS as usize / 2 - 3);
        let mask = usize::MAX >> (64 - 2 * self.len());
        unsafe {
            ((self.seq.as_ptr() as *const usize).read_unaligned() >> (2 * self.offset)) & mask
        }
    }

    /// Convert to an owned version.
    fn to_vec(&self) -> PackedSeqVec {
        assert_eq!(self.offset, 0);
        PackedSeqVec {
            seq: self.seq.to_vec(),
            len: self.len,
            ranges: vec![(0, self.len)],
        }
    }

    #[inline(always)]
    fn slice(&self, range: Range<usize>) -> Self {
        assert!(range.end <= self.len);
        PackedSeq {
            seq: self.seq,
            offset: self.offset + range.start,
            len: range.end - range.start,
        }
        .normalize()
    }

    #[inline(always)]
    fn iter_bp(self) -> impl ExactSizeIterator<Item = u8> {
        assert!(self.len <= self.seq.len() * 4);

        let this = self.normalize();

        // read u64 at a time?
        let mut byte = 0;
        let mut it = (0..this.len + this.offset).map(move |i| {
            if i % 4 == 0 {
                byte = this.seq[i / 4];
            }
            // Shift byte instead of i?
            (byte >> (2 * (i % 4))) & 0b11
        });
        it.by_ref().take(this.offset).for_each(drop);
        it
    }

    #[inline(always)]
    fn par_iter_bp(self, context: usize) -> (impl ExactSizeIterator<Item = S>, Self) {
        #[cfg(target_endian = "big")]
        panic!("Big endian architectures are not supported.");

        let this = self.normalize();
        assert_eq!(
            this.offset % 4,
            0,
            "Non-byte offsets are not yet supported."
        );

        let num_kmers = this.len.saturating_sub(context - 1);
        let n = (num_kmers / L) / 4 * 4;
        let bytes_per_chunk = n / 4;

        let base_ptr = this.seq.as_ptr();
        let offsets_lanes_0_4: u64x4 = from_fn(|l| (l * bytes_per_chunk) as u64).into();
        let offsets_lanes_4_8: u64x4 = from_fn(|l| ((4 + l) * bytes_per_chunk) as u64).into();
        let mut upcoming_1 = S::ZERO;
        let mut upcoming_2 = S::ZERO;

        let it = (0..if num_kmers == 0 { 0 } else { n + context - 1 }).map(move |i| {
            if i % 16 == 0 {
                if i % 32 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat((i / 4) as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat((i / 4) as u64);
                    let u64_0_4: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_0_4)) };
                    let u64_4_8: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_4_8)) };
                    // Split into two vecs containing a u32 of 4 characters each.
                    (upcoming_1, upcoming_2) = intrinsics::deinterleave(u64_0_4, u64_4_8);
                } else {
                    // Move on to the next u32 containing 4 buffered characters.
                    upcoming_1 = upcoming_2;
                }
            }
            // Extract the last 2 bits of each character.
            let chars = upcoming_1 & S::splat(0x03);
            // Shift remaining characters to the right.
            upcoming_1 = upcoming_1 >> S::splat(2);
            chars
        });

        (
            it,
            PackedSeq {
                seq: &self.seq[L * bytes_per_chunk..],
                offset: 0,
                len: self.len - L * n,
            },
        )
    }
}

impl PackedSeqVec {
    /// Create an `PackedSeqVec` from an ASCII sequence. See `push_ascii` for details.
    pub fn from_ascii(seq: &[u8]) -> Self {
        let mut packed_vec = Self::default();
        packed_vec.push_ascii(seq);
        packed_vec
    }

    /// Push an ASCII sequence to an `PackedSeqVec`.
    /// `Aa` map to `0`, `Cc` to `1`, `Gg` to `3`, and `Tt` to `2`.
    /// Other characters may be silently mapped into `[0, 4)` or panick.
    /// (TODO: Explicitly support different conversions.)
    ///
    /// Uses the BMI2 `pext` instruction when available, based on the
    /// `n_to_bits_pext` method described at
    /// https://github.com/Daniel-Liu-c0deb0t/cute-nucleotides.
    ///
    /// TODO: Optimize for non-BMI2 platforms.
    /// TODO: Support multiple ways of dealing with non-`ACGT` characters.
    #[cfg(target_endian = "little")]
    pub fn push_ascii(&mut self, seq: &[u8]) -> Range<usize> {
        let start = 4 * self.seq.len();
        let len = seq.len();

        #[allow(unused)]
        let mut last = 0;

        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            last = seq.len() / 8 * 8;

            for i in (0..last).step_by(8) {
                let chunk = &seq[i..i + 8].try_into().unwrap();
                let ascii = u64::from_ne_bytes(*chunk);
                let packed_bytes =
                    unsafe { std::arch::x86_64::_pext_u64(ascii, 0x0606060606060606) };
                self.seq.push(packed_bytes as u8);
                self.seq.push((packed_bytes >> 8) as u8);
                self.len += 8;
            }
        }

        let mut packed_byte = 0;
        for &base in &seq[last..] {
            packed_byte |= match base {
                b'a' | b'A' => 0,
                b'c' | b'C' => 1,
                b'g' | b'G' => 3,
                b't' | b'T' => 2,
                _ => panic!(),
            } << ((self.len % 4) * 2);
            self.len += 1;
            if self.len % 4 == 0 {
                self.seq.push(packed_byte);
                packed_byte = 0;
            }
        }
        if self.len % 4 != 0 {
            self.seq.push(packed_byte);
        }
        start..start + len
    }
}

impl AsciiSeqVec {
    pub fn from_vec(seq: Vec<u8>) -> Self {
        Self {
            ranges: vec![(0, seq.len())],
            seq,
        }
    }
}

impl SeqVec for AsciiSeqVec {
    type Seq<'s> = AsciiSeq<'s>;

    fn as_slice(&self) -> Self::Seq<'_> {
        AsciiSeq(self.seq.as_slice())
    }

    fn push_seq(&mut self, seq: AsciiSeq) -> Range<usize> {
        let start = seq.len();
        let end = start + seq.len();
        let range = start..end;
        self.seq.extend(seq.0);
        self.ranges.push((start, end));
        range
    }

    fn random(n: usize) -> Self {
        Self {
            seq: (0..n)
                .map(|_| b"ACGT"[rand::random::<u8>() as usize % 4])
                .collect(),
            ranges: vec![(0, n)],
        }
    }
}

impl SeqVec for PackedSeqVec {
    type Seq<'s> = PackedSeq<'s>;

    fn as_slice(&self) -> Self::Seq<'_> {
        PackedSeq {
            seq: &self.seq,
            offset: 0,
            len: self.len,
        }
    }

    fn push_seq<'a>(&mut self, seq: PackedSeq<'_>) -> Range<usize> {
        let start = 4 * self.seq.len() + seq.offset;
        let end = start + seq.len();
        self.seq.extend(seq.seq);
        self.len = 4 * self.seq.len();
        start..end
    }

    fn random(n: usize) -> Self {
        let seq = (0..n.div_ceil(4)).map(|_| rand::random::<u8>()).collect();
        PackedSeqVec {
            seq,
            len: n,
            ranges: vec![(0, n)],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn pack_naive(seq: &[u8]) -> (Vec<u8>, usize) {
        let mut packed_byte = 0;
        let mut packed_len = 0;
        let mut packed = vec![];
        for &base in seq {
            packed_byte |= match base {
                b'a' | b'A' => 0,
                b'c' | b'C' => 1,
                b'g' | b'G' => 3,
                b't' | b'T' => 2,
                _ => panic!(),
            } << ((packed_len % 4) * 2);
            packed_len += 1;
            if packed_len % 4 == 0 {
                packed.push(packed_byte);
                packed_byte = 0;
            }
        }
        if packed_len % 4 != 0 {
            packed.push(packed_byte);
        }
        (packed, packed_len)
    }

    #[test]
    fn pack() {
        for n in 0..=128 {
            let seq: Vec<_> = (0..n)
                .map(|_| b"ACGTacgt"[rand::random::<u8>() as usize % 8])
                .collect();
            let (packed_1, len1) = pack_naive(&seq);
            let packed_2 = PackedSeqVec::from_ascii(&seq);
            assert_eq!(len1, packed_2.len);
            assert_eq!(packed_1, packed_2.seq);
        }
    }

    #[test]
    fn pack_word() {
        let packed = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGT");
        let slice = packed.slice(0..4);
        assert_eq!(slice.to_word(), 0b10110100);
        let slice = packed.slice(0..8);
        assert_eq!(slice.to_word(), 0b1011010010110100);
        let slice = packed.slice(0..16);
        assert_eq!(slice.to_word(), 0b10110100101101001011010010110100);
        let slice = packed.slice(0..28);
        assert_eq!(
            slice.to_word(),
            0b10110100101101001011010010110100101101001011010010110100
        );
    }
}
