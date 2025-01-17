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

    fn ranges(&mut self) -> &mut Vec<(usize, usize)>;

    fn random(n: usize) -> Self;
}

// ---------------- STRUCTS ----------------

/// A `&[u8]` representing an ASCII sequence.
/// Only supported characters are `ACGTacgt`.
/// Other characters will be silently mapped into `[0, 4)`, or may cause panics.
#[derive(Copy, Clone, Debug, MemSize, MemDbg, PartialEq, Eq, PartialOrd, Ord)]
pub struct AsciiSeq<'s>(pub &'s [u8]);

/// An owned ASCII sequence.
/// Only supported characters are `ACGTacgt`.
/// Other characters will be silently mapped into `[0, 4)`, or may cause panics.
///
/// TODO: Should this be a strong type instead?
#[derive(Clone, Debug, Default, MemSize, MemDbg)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
pub struct AsciiSeqVec {
    pub seq: Vec<u8>,
    pub ranges: Vec<(usize, usize)>,
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
    pub ranges: Vec<(usize, usize)>,
}

// ---------------- IMPLEMENTATIONS ----------------

/// Maps ASCII to `[0, 4)` on the fly.
/// Prefer first packing into a `PackedSeqVec` for storage.
impl<'s> Seq<'s> for AsciiSeq<'s> {
    const BASES_PER_BYTE: usize = 1;
    type SeqVec = AsciiSeqVec;

    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn get(&self, index: usize) -> u8 {
        pack_char(self.0[index])
    }

    #[inline(always)]
    fn get_ascii(&self, index: usize) -> u8 {
        self.0[index]
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
            for (i, &base) in self.0.iter().enumerate() {
                val |= (pack_char(base) as u64) << (i * 2);
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
    fn iter_bp(self) -> impl ExactSizeIterator<Item = u8> + Clone {
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
        self.0.iter().copied().map(pack_char)
    }

    /// Iterate the basepairs in the sequence in 8 parallel streams, assuming values in `0..4`.
    #[inline(always)]
    fn par_iter_bp(self, context: usize) -> (impl ExactSizeIterator<Item = S> + Clone, Self) {
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

                    // Mask out the unneeded bits.
                    let mask = 0x06060606;
                    upcoming_1 &= S::splat(mask);
                    upcoming_2 &= S::splat(mask);
                    // Shift down everything by 1, so the needed bits are in the lowest 2 positions.
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

    #[inline(always)]
    fn par_iter_bp_delayed(
        self,
        context: usize,
        delay: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S)> + Clone, Self) {
        let num_kmers = self.len().saturating_sub(context - 1);
        let n = num_kmers / L;

        let base_ptr = self.0.as_ptr();
        let offsets_lanes_0_4: u64x4 = from_fn(|l| (l * n) as u64).into();
        let offsets_lanes_4_8: u64x4 = from_fn(|l| ((4 + l) * n) as u64).into();

        let mut upcoming = S::ZERO;
        let mut upcoming_d = S::ZERO;

        // Even buf_len is nice to only have the write==buf_len check once.
        // We also make it the next power of 2, for faster modulo operations.
        // delay/4: number of bp in a u32.
        let buf_len = (delay / 4 + 2).next_power_of_two();
        let buf_mask = buf_len - 1;
        let mut buf = vec![S::ZERO; buf_len];
        let mut write_idx = 0;
        // We compensate for the first delay/16 triggers of the check below that
        // happen before the delay is actually reached.
        let mut read_idx = (buf_len - delay / 4) % buf_len;

        let it = (0..if num_kmers == 0 { 0 } else { n + context - 1 }).map(move |i| {
            if i % 4 == 0 {
                if i % 8 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat(i as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat(i as u64);
                    let u64_0_4: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_0_4)) };
                    let u64_4_8: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_4_8)) };
                    // Split into two vecs containing a u32 of 4 characters each.
                    let (mut a, mut b) = intrinsics::deinterleave(u64_0_4, u64_4_8);
                    // Mask out the unneeded bits.
                    let mask = 0x06060606;
                    a &= S::splat(mask);
                    b &= S::splat(mask);
                    // Shift down everything by 1, so the needed bits are in the lowest 2 positions.
                    a = a >> S::splat(1);
                    b = b >> S::splat(1);
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    buf[write_idx] = a;
                    upcoming = a;
                    write_idx += 1;
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    buf[write_idx] = b;
                    // write_idx will be incremented one more in the `else` below.
                } else {
                    // Move on to the next u32 containing 4 buffered characters.
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    upcoming = buf[write_idx];
                    write_idx += 1;
                    write_idx &= buf_mask;
                }
            }
            if i % 4 == delay % 4 {
                unsafe { assert_unchecked(read_idx < buf.len()) };
                upcoming_d = buf[read_idx];
                read_idx += 1;
                read_idx &= buf_mask;
            }
            // Extract the last 2 bits of each character.
            let chars = upcoming & S::splat(0x03);
            let chars_d = upcoming_d & S::splat(0x03);
            // Shift remaining characters to the right.
            upcoming = upcoming >> S::splat(8);
            upcoming_d = upcoming_d >> S::splat(8);
            (chars, chars_d)
        });

        (it, Self(&self.0[L * n..]))
    }

    #[inline(always)]
    fn par_iter_bp_delayed_2(
        self,
        context: usize,
        delay1: usize,
        delay2: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S, S)> + Clone, Self) {
        assert!(delay1 <= delay2, "Delay1 must be at most delay2.");

        let num_kmers = self.len().saturating_sub(context - 1);
        let n = num_kmers / L;

        let base_ptr = self.0.as_ptr();
        let offsets_lanes_0_4: u64x4 = from_fn(|l| (l * n) as u64).into();
        let offsets_lanes_4_8: u64x4 = from_fn(|l| ((4 + l) * n) as u64).into();

        let mut upcoming = S::ZERO;
        let mut upcoming_d1 = S::ZERO;
        let mut upcoming_d2 = S::ZERO;

        // Even buf_len is nice to only have the write==buf_len check once.
        // We also make it the next power of 2, for faster modulo operations.
        // delay/4: number of bp in a u32.
        let buf_len = (delay2 / 4 + 2).next_power_of_two();
        let buf_mask = buf_len - 1;
        let mut buf = vec![S::ZERO; buf_len];
        let mut write_idx = 0;
        // We compensate for the first delay/16 triggers of the check below that
        // happen before the delay is actually reached.
        let mut read_idx1 = (buf_len - delay1 / 4) % buf_len;
        let mut read_idx2 = (buf_len - delay2 / 4) % buf_len;

        let it = (0..if num_kmers == 0 { 0 } else { n + context - 1 }).map(move |i| {
            if i % 4 == 0 {
                if i % 8 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat(i as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat(i as u64);
                    let u64_0_4: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_0_4)) };
                    let u64_4_8: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_4_8)) };
                    // Split into two vecs containing a u32 of 4 characters each.
                    let (mut a, mut b) = intrinsics::deinterleave(u64_0_4, u64_4_8);
                    // Mask out the unneeded bits.
                    let mask = 0x06060606;
                    a &= S::splat(mask);
                    b &= S::splat(mask);
                    // Shift down everything by 1, so the needed bits are in the lowest 2 positions.
                    a = a >> S::splat(1);
                    b = b >> S::splat(1);
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    buf[write_idx] = a;
                    upcoming = a;
                    write_idx += 1;
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    buf[write_idx] = b;
                    // write_idx will be incremented one more in the `else` below.
                } else {
                    // Move on to the next u32 containing 4 buffered characters.
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    upcoming = buf[write_idx];
                    write_idx += 1;
                    write_idx &= buf_mask;
                }
            }
            if i % 4 == delay1 % 4 {
                unsafe { assert_unchecked(read_idx1 < buf.len()) };
                upcoming_d1 = buf[read_idx1];
                read_idx1 += 1;
                read_idx1 &= buf_mask;
            }
            if i % 4 == delay2 % 4 {
                unsafe { assert_unchecked(read_idx2 < buf.len()) };
                upcoming_d2 = buf[read_idx2];
                read_idx2 += 1;
                read_idx2 &= buf_mask;
            }
            // Extract the last 2 bits of each character.
            let chars = upcoming & S::splat(0x03);
            let chars_d1 = upcoming_d1 & S::splat(0x03);
            let chars_d2 = upcoming_d2 & S::splat(0x03);
            // Shift remaining characters to the right.
            upcoming = upcoming >> S::splat(8);
            upcoming_d1 = upcoming_d1 >> S::splat(8);
            upcoming_d2 = upcoming_d2 >> S::splat(8);
            (chars, chars_d1, chars_d2)
        });

        (it, Self(&self.0[L * n..]))
    }

    fn cmp_lcp(&self, other: &Self) -> (std::cmp::Ordering, usize) {
        for i in 0..self.len().min(other.len()) {
            if self.0[i] != other.0[i] {
                return (self.0[i].cmp(&other.0[i]), i);
            }
        }
        (self.len().cmp(&other.len()), self.len().min(other.len()))
    }
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

    pub fn unpack(&self) -> Vec<u8> {
        self.iter_bp().map(unpack).collect()
    }
}

impl<'s> Seq<'s> for PackedSeq<'s> {
    const BASES_PER_BYTE: usize = 4;
    type SeqVec = PackedSeqVec;

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn get(&self, index: usize) -> u8 {
        let offset = self.offset + index;
        let idx = offset / 4;
        let offset = offset % 4;
        unsafe { (*self.seq.get_unchecked(idx) >> (2 * offset)) & 3 }
    }

    #[inline(always)]
    fn get_ascii(&self, index: usize) -> u8 {
        unpack(self.get(index))
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
        assert!(
            range.end <= self.len,
            "Slice index out of bounds: {} > {}",
            range.end,
            self.len
        );
        PackedSeq {
            seq: self.seq,
            offset: self.offset + range.start,
            len: range.end - range.start,
        }
        .normalize()
    }

    #[inline(always)]
    fn iter_bp(self) -> impl ExactSizeIterator<Item = u8> + Clone {
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
    fn par_iter_bp(self, context: usize) -> (impl ExactSizeIterator<Item = S> + Clone, Self) {
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
        let mut cur = S::ZERO;
        let mut buf = S::ZERO;

        let it = (0..if num_kmers == 0 { 0 } else { n + context - 1 }).map(move |i| {
            if i % 16 == 0 {
                if i % 32 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat((i / 4) as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat((i / 4) as u64);
                    let u64_0_4: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_0_4)) };
                    let u64_4_8: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_4_8)) };
                    // Split into two vecs containing a u32 of 4 characters each.
                    (cur, buf) = intrinsics::deinterleave(u64_0_4, u64_4_8);
                } else {
                    // Move on to the next u32 containing 4 buffered characters.
                    cur = buf;
                }
            }
            // Extract the last 2 bits of each character.
            let chars = cur & S::splat(0x03);
            // Shift remaining characters to the right.
            cur = cur >> S::splat(2);
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

    /// Iterate the basepairs in the sequence in 8 parallel streams, assuming values in `0..4`.
    /// This version returns two streams, with one `delay` steps behind the other.
    ///
    /// The first `delay` iterations of the delayed character will return 0.
    #[inline(always)]
    fn par_iter_bp_delayed(
        self,
        context: usize,
        delay: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S)> + Clone, Self) {
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
        let mut upcoming = S::ZERO;
        let mut upcoming_d = S::ZERO;

        // Even buf_len is nice to only have the write==buf_len check once.
        // We also make it the next power of 2, for faster modulo operations.
        // delay/16: number of bp in a u32.
        let buf_len = (delay / 16 + 2).next_power_of_two();
        let buf_mask = buf_len - 1;
        let mut buf = vec![S::ZERO; buf_len];
        let mut write_idx = 0;
        // We compensate for the first delay/16 triggers of the check below that
        // happen before the delay is actually reached.
        let mut read_idx = (buf_len - delay / 16) % buf_len;

        let it = (0..if num_kmers == 0 { 0 } else { n + context - 1 }).map(move |i| {
            if i % 16 == 0 {
                if i % 32 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat((i / 4) as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat((i / 4) as u64);
                    let u64_0_4: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_0_4)) };
                    let u64_4_8: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_4_8)) };
                    // Split into two vecs containing a u32 of 4 characters each.
                    let (a, b) = intrinsics::deinterleave(u64_0_4, u64_4_8);
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    buf[write_idx] = a;
                    upcoming = a;
                    write_idx += 1;
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    buf[write_idx] = b;
                    // write_idx will be incremented one more in the `else` below.
                } else {
                    // Move on to the next u32 containing 4 buffered characters.
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    upcoming = buf[write_idx];
                    write_idx += 1;
                    write_idx &= buf_mask;
                }
            }
            if i % 16 == delay % 16 {
                unsafe { assert_unchecked(read_idx < buf.len()) };
                upcoming_d = buf[read_idx];
                read_idx += 1;
                read_idx &= buf_mask;
            }
            // Extract the last 2 bits of each character.
            let chars = upcoming & S::splat(0x03);
            let chars_d = upcoming_d & S::splat(0x03);
            // Shift remaining characters to the right.
            upcoming = upcoming >> S::splat(2);
            upcoming_d = upcoming_d >> S::splat(2);
            (chars, chars_d)
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

    /// Delay1 must be smaller than delay2.
    #[inline(always)]
    fn par_iter_bp_delayed_2(
        self,
        context: usize,
        delay1: usize,
        delay2: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S, S)> + Clone, Self) {
        #[cfg(target_endian = "big")]
        panic!("Big endian architectures are not supported.");

        let this = self.normalize();
        assert_eq!(
            this.offset % 4,
            0,
            "Non-byte offsets are not yet supported."
        );
        assert!(delay1 <= delay2, "Delay1 must be at most delay2.");

        let num_kmers = this.len.saturating_sub(context - 1);
        let n = (num_kmers / L) / 4 * 4;
        let bytes_per_chunk = n / 4;

        let base_ptr = this.seq.as_ptr();
        let offsets_lanes_0_4: u64x4 = from_fn(|l| (l * bytes_per_chunk) as u64).into();
        let offsets_lanes_4_8: u64x4 = from_fn(|l| ((4 + l) * bytes_per_chunk) as u64).into();
        let mut upcoming = S::ZERO;
        let mut upcoming_d1 = S::ZERO;
        let mut upcoming_d2 = S::ZERO;

        // Even buf_len is nice to only have the write==buf_len check once.
        let buf_len = (delay2 / 16 + 2).next_power_of_two();
        let buf_mask = buf_len - 1;
        let mut buf = vec![S::ZERO; buf_len];
        let mut write_idx = 0;
        // We compensate for the first delay/16 triggers of the check below that
        // happen before the delay is actually reached.
        let mut read_idx1 = (buf_len - delay1 / 16) % buf_len;
        let mut read_idx2 = (buf_len - delay2 / 16) % buf_len;

        let it = (0..if num_kmers == 0 { 0 } else { n + context - 1 }).map(move |i| {
            if i % 16 == 0 {
                if i % 32 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat((i / 4) as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat((i / 4) as u64);
                    let u64_0_4: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_0_4)) };
                    let u64_4_8: S = unsafe { transmute(intrinsics::gather(base_ptr, idx_4_8)) };
                    // Split into two vecs containing a u32 of 4 characters each.
                    let (a, b) = intrinsics::deinterleave(u64_0_4, u64_4_8);
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    buf[write_idx] = a;
                    upcoming = a;
                    write_idx += 1;
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    buf[write_idx] = b;
                    // write_idx will be incremented one more in the `else` below.
                } else {
                    // Move on to the next u32 containing 4 buffered characters.
                    unsafe { assert_unchecked(write_idx < buf.len()) };
                    upcoming = buf[write_idx];
                    write_idx += 1;
                    write_idx &= buf_mask;
                }
            }
            if i % 16 == delay1 % 16 {
                unsafe { assert_unchecked(read_idx1 < buf.len()) };
                upcoming_d1 = buf[read_idx1];
                read_idx1 += 1;
                read_idx1 &= buf_mask;
            }
            if i % 16 == delay2 % 16 {
                unsafe { assert_unchecked(read_idx2 < buf.len()) };
                upcoming_d2 = buf[read_idx2];
                read_idx2 += 1;
                read_idx2 &= buf_mask;
            }
            // Extract the last 2 bits of each character.
            let chars = upcoming & S::splat(0x03);
            let chars_d1 = upcoming_d1 & S::splat(0x03);
            let chars_d2 = upcoming_d2 & S::splat(0x03);
            // Shift remaining characters to the right.
            upcoming = upcoming >> S::splat(2);
            upcoming_d1 = upcoming_d1 >> S::splat(2);
            upcoming_d2 = upcoming_d2 >> S::splat(2);
            (chars, chars_d1, chars_d2)
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

    fn cmp_lcp(&self, other: &Self) -> (std::cmp::Ordering, usize) {
        // Compare 29 characters at a time by converting them to a word.
        let mut lcp = 0;
        let min_len = self.len.min(other.len);
        for i in (0..min_len).step_by(29) {
            let len = (min_len - i).min(29);
            let this = self.slice(i..i + len);
            let other = other.slice(i..i + len);
            let this_word = this.to_word();
            let other_word = other.to_word();
            if this_word != other_word {
                // Unfortunately, bases are packed in little endian order, so the default order is reversed.
                let eq = this_word ^ other_word;
                let t = eq.trailing_zeros() / 2 * 2;
                lcp += t as usize / 2;
                let mask = 0b11 << t;
                return ((this_word & mask).cmp(&(other_word & mask)), lcp);
            }
            lcp += len;
        }
        (self.len.cmp(&other.len), lcp)
    }
}

impl PartialEq for PackedSeq<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        // Compare 29 characters at a time by converting them to a word.
        for i in (0..self.len).step_by(29) {
            let len = (self.len - i).min(29);
            let this = self.slice(i..i + len);
            let that = other.slice(i..i + len);
            if this.to_word() != that.to_word() {
                return false;
            }
        }
        return true;
    }
}

impl Eq for PackedSeq<'_> {}

impl PartialOrd for PackedSeq<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PackedSeq<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare 29 characters at a time by converting them to a word.
        let min_len = self.len.min(other.len);
        for i in (0..min_len).step_by(29) {
            let len = (min_len - i).min(29);
            let this = self.slice(i..i + len);
            let other = other.slice(i..i + len);
            let this_word = this.to_word();
            let other_word = other.to_word();
            if this_word != other_word {
                // Unfortunately, bases are packed in little endian order, so the default order is reversed.
                let eq = this_word ^ other_word;
                let t = eq.trailing_zeros() / 2 * 2;
                let mask = 0b11 << t;
                return (this_word & mask).cmp(&(other_word & mask));
            }
        }
        self.len.cmp(&other.len)
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

    /// Get the underlying ASCII text.
    fn into_raw(self) -> Vec<u8> {
        self.seq
    }

    fn as_slice(&self) -> Self::Seq<'_> {
        AsciiSeq(self.seq.as_slice())
    }

    fn push_seq(&mut self, seq: AsciiSeq) -> Range<usize> {
        let start = self.seq.len();
        let end = start + seq.len();
        let range = start..end;
        self.seq.extend(seq.0);
        self.ranges.push((start, end));
        range
    }

    fn push_ascii(&mut self, seq: &[u8]) -> Range<usize> {
        self.push_seq(AsciiSeq(seq))
    }

    fn ranges(&mut self) -> &mut Vec<(usize, usize)> {
        &mut self.ranges
    }

    fn random(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            seq: (0..n)
                .map(|_| b"ACGT"[rng.gen::<u8>() as usize % 4])
                .collect(),
            ranges: vec![(0, n)],
        }
    }
}

impl SeqVec for PackedSeqVec {
    type Seq<'s> = PackedSeq<'s>;

    fn into_raw(self) -> Vec<u8> {
        self.seq
    }

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
        self.ranges.push((start, end));
        start..end
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
    fn push_ascii(&mut self, seq: &[u8]) -> Range<usize> {
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
            packed_byte |= pack_char(base) << ((self.len % 4) * 2);
            self.len += 1;
            if self.len % 4 == 0 {
                self.seq.push(packed_byte);
                packed_byte = 0;
            }
        }
        if self.len % 4 != 0 {
            self.seq.push(packed_byte);
        }
        self.ranges.push((start, start + len));
        start..start + len
    }

    fn ranges(&mut self) -> &mut Vec<(usize, usize)> {
        &mut self.ranges
    }

    fn random(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        let seq = (0..n.div_ceil(4)).map(|_| rng.gen::<u8>()).collect();
        PackedSeqVec {
            seq,
            len: n,
            ranges: vec![(0, n)],
        }
    }
}

#[cfg(test)]
mod test {
    use wide::u32x8;

    use super::*;

    fn pack_naive(seq: &[u8]) -> (Vec<u8>, usize) {
        let mut packed_byte = 0;
        let mut packed_len = 0;
        let mut packed = vec![];
        for &base in seq {
            packed_byte |= pack_char(base) << ((packed_len % 4) * 2);
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
            let mut rng = rand::thread_rng();
            let seq: Vec<_> = (0..n)
                .map(|_| b"ACGTacgt"[rng.gen::<u8>() as usize % 8])
                .collect();
            let (packed_1, len1) = pack_naive(&seq);
            let packed_2 = PackedSeqVec::from_ascii(&seq);
            assert_eq!(len1, packed_2.len);
            assert_eq!(packed_1, packed_2.seq);
        }
    }

    #[test]
    fn pack_via_ascii() {
        for n in 0..=128 {
            let mut rng = rand::thread_rng();
            let seq: Vec<_> = (0..n)
                .map(|_| b"ACGTacgt"[rng.gen::<u8>() as usize % 8])
                .collect();
            let ascii_seq = AsciiSeqVec::from_ascii(&seq);
            let (packed_1, len1) = pack_naive(&seq);
            let packed_2 = PackedSeqVec::from_ascii(&ascii_seq.seq);
            assert_eq!(len1, packed_2.len);
            assert_eq!(packed_1, packed_2.seq);
        }
    }

    #[test]
    fn pack_word() {
        let packed = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGT");
        let slice = packed.slice(0..1);
        assert_eq!(slice.to_word(), 0b00000000);
        let slice = packed.slice(0..2);
        assert_eq!(slice.to_word(), 0b00000100);
        let slice = packed.slice(0..3);
        assert_eq!(slice.to_word(), 0b00110100);
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

    #[test]
    fn packed_ord() {
        let ascii_seq = b"ACGTACGTACGTACGTACGTACGTACGT";
        let packed_seq = ascii_seq
            .iter()
            .map(|c| match c {
                // Swap G and T values since they are encoded in opposite order.
                b'G' => b'T',
                b'T' => b'G',
                c => *c,
            })
            .collect::<Vec<_>>();
        let ascii = AsciiSeqVec::from_ascii(ascii_seq);
        let packed = PackedSeqVec::from_ascii(&packed_seq);
        for i in 0..ascii.len() {
            for j in i..ascii.len() {
                for k in 0..ascii.len() {
                    for l in k..ascii.len() {
                        let a0 = ascii.as_slice().slice(i..j);
                        let a1 = ascii.as_slice().slice(k..l);
                        let b0 = packed.as_slice().slice(i..j);
                        let b1 = packed.as_slice().slice(k..l);
                        assert_eq!(
                            a0.cmp(&a1),
                            b0.cmp(&b1),
                            "Failed at ({}, {})={:?}, ({}, {})={:?}",
                            i,
                            j,
                            &ascii_seq[i..j],
                            k,
                            l,
                            &ascii_seq[k..l]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn iter_bp() {
        let seq = b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT";
        for len in 0..=seq.len() {
            let ascii = AsciiSeqVec::from_ascii(&seq[..len]);
            let packed = PackedSeqVec::from_ascii(&seq[..len]);
            eprintln!("ascii {ascii:?}");
            eprintln!("packed {packed:?}");
            let ascii = ascii.as_slice().iter_bp().collect::<Vec<_>>();
            let packed = packed.as_slice().iter_bp().collect::<Vec<_>>();
            assert_eq!(ascii, packed);
        }
    }

    #[test]
    fn par_iter_bp_delayed0() {
        let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
        let (head, tail) = s.as_slice().par_iter_bp_delayed(1, 0);
        let head = head.collect::<Vec<_>>();
        let tail = tail.iter_bp().collect::<Vec<_>>();
        fn f(x: &[u8; 8], y: &[u8; 8]) -> (u32x8, u32x8) {
            let x = x.map(|x| pack_char(x) as u32);
            let y = y.map(|x| pack_char(x) as u32);
            (u32x8::from(x), u32x8::from(y))
        }
        assert_eq!(
            head,
            vec![
                f(b"AAGACGAA", b"AAGACGAA"),
                f(b"CAGACTAA", b"CAGACTAA"),
                f(b"GCTAGTAA", b"GCTAGTAA"),
                f(b"TCTCGTAA", b"TCTCGTAA"),
            ]
        );
        assert_eq!(tail, vec![0, 1, 3, 2]);
    }

    #[test]
    fn par_iter_bp_delayed1() {
        let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
        let (head, tail) = s.as_slice().par_iter_bp_delayed(1, 1);
        let head = head.collect::<Vec<_>>();
        let tail = tail.iter_bp().collect::<Vec<_>>();
        fn f(x: &[u8; 8], y: &[u8; 8]) -> (u32x8, u32x8) {
            let x = x.map(|x| pack_char(x) as u32);
            let y = y.map(|x| pack_char(x) as u32);
            (u32x8::from(x), u32x8::from(y))
        }
        assert_eq!(
            head,
            vec![
                f(b"AAGACGAA", b"AAAAAAAA"),
                f(b"CAGACTAA", b"AAGACGAA"),
                f(b"GCTAGTAA", b"CAGACTAA"),
                f(b"TCTCGTAA", b"GCTAGTAA"),
            ]
        );
        assert_eq!(tail, vec![0, 1, 3, 2]);
    }

    #[test]
    fn par_iter_bp_delayed_large() {
        let seq = AsciiSeqVec::random(48);
        eprintln!("SEQ: {:?}", seq.seq);
        let s = PackedSeqVec::from_ascii(&seq.seq);
        let delay = 16;
        let (head, _tail) = s.as_slice().par_iter_bp_delayed(17, delay);
        let head = head.collect::<Vec<_>>();
        fn f(x: &[u8; 8], y: &[u8; 8]) -> (u32x8, u32x8) {
            let x = x.map(|x| pack_char(x) as u32);
            let y = y.map(|x| pack_char(x) as u32);
            (u32x8::from(x), u32x8::from(y))
        }
        let stride = 4;
        let len = head.len();
        assert_eq!(
            head,
            (0..len)
                .map(|i| {
                    f(
                        &from_fn(|j| seq.seq[i + stride * j]),
                        &from_fn(|j| {
                            if i < delay {
                                b'A'
                            } else {
                                seq.seq[i + stride * j - delay]
                            }
                        }),
                    )
                })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn par_iter_bp_delayed01() {
        let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
        let (head, tail) = s.as_slice().par_iter_bp_delayed_2(1, 0, 1);
        let head = head.collect::<Vec<_>>();
        let tail = tail.iter_bp().collect::<Vec<_>>();
        fn f(x: &[u8; 8], y: &[u8; 8], z: &[u8; 8]) -> (u32x8, u32x8, u32x8) {
            let x = x.map(|x| pack_char(x) as u32);
            let y = y.map(|x| pack_char(x) as u32);
            let z = z.map(|x| pack_char(x) as u32);
            (u32x8::from(x), u32x8::from(y), u32x8::from(z))
        }
        assert_eq!(
            head,
            vec![
                f(b"AAGACGAA", b"AAGACGAA", b"AAAAAAAA"),
                f(b"CAGACTAA", b"CAGACTAA", b"AAGACGAA"),
                f(b"GCTAGTAA", b"GCTAGTAA", b"CAGACTAA"),
                f(b"TCTCGTAA", b"TCTCGTAA", b"GCTAGTAA"),
            ]
        );
        assert_eq!(tail, vec![0, 1, 3, 2]);
    }

    #[test]
    fn get() {
        let n = 1000;
        let s = PackedSeqVec::random(n);
        let iter_bp = s.as_slice().iter_bp().collect::<Vec<_>>();
        let get = (0..n).map(|i| s.as_slice().get(i)).collect::<Vec<_>>();
        assert_eq!(iter_bp, get);
    }
}
