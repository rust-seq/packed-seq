mod intrinsics;

use core::{array::from_fn, mem::transmute};
use epserde::{deser::DeserializeInner, ser::SerializeInner, Epserde};
use std::ops::Range;
use wide::u64x4;

pub(crate) use wide::u32x8 as S;
pub(crate) const L: usize = 8;

/// Interface to sequences over a 2-bit alphabet.
///
/// Currently supports `&[u8]`, where each `u8` must be in `0..4`, and the
/// `PackedSeq` type that contains packed sequences.
pub trait Seq: Copy {
    const BASES_PER_BYTE: usize;

    /// The length of the sequence in bp.
    fn len(&self) -> usize;

    /// Convert a short sequence (kmer) to a single underlying word.
    /// Note that this does no additional packing, so for `&[u8]` it can only contain up to 8 characters.
    /// Panics if the sequence is too long.
    fn to_word(&self) -> usize;

    /// Get a sub-slice of the sequence.
    fn sub_slice(&self, idx: usize, len: usize) -> Self;

    /// A simple iterator over characters.
    /// Returns u8 values in [0, 4).
    fn iter_bp(&self) -> impl ExactSizeIterator<Item = u8>;

    /// An iterator that splits the input into 8 chunks and streams over them in parallel.
    /// Returns a separate `tail` iterator over the remaining characters.
    /// k is the k-mer size being iterated. When k>1, consecutive chunk overlap by k-1 characters.
    fn par_iter_bp(&self, k: usize) -> (impl ExactSizeIterator<Item = S>, Self);
}

impl<'s> Seq for &'s [u8] {
    const BASES_PER_BYTE: usize = 1;

    #[inline(always)]
    fn len(&self) -> usize {
        (self as &[u8]).len()
    }

    #[inline(always)]
    fn to_word(&self) -> usize {
        assert!(self.len() <= usize::BITS as usize / 8);
        let mask = usize::MAX >> (64 - 8 * self.len());
        unsafe { *(self.as_ptr() as *const usize) & mask }
    }

    #[inline(always)]
    fn sub_slice(&self, idx: usize, len: usize) -> Self {
        &self[idx..idx + len]
    }

    #[inline(always)]
    fn iter_bp(&self) -> impl ExactSizeIterator<Item = u8> {
        self.iter().copied()
    }

    #[inline(always)]
    fn par_iter_bp(&self, k: usize) -> (impl ExactSizeIterator<Item = S>, Self) {
        let num_kmers = self.len().saturating_sub(k - 1);
        let n = num_kmers / L;

        let base_ptr = self.as_ptr();
        let offsets_lanes_0_4: u64x4 = from_fn(|l| (l * n) as u64).into();
        let offsets_lanes_4_8: u64x4 = from_fn(|l| ((4 + l) * n) as u64).into();
        let mut upcoming_1 = S::ZERO;
        let mut upcoming_2 = S::ZERO;

        let it = (0..if num_kmers == 0 { 0 } else { n + k - 1 }).map(move |i| {
            if i % 4 == 0 {
                if i % 8 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat(i as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat(i as u64);
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
            upcoming_1 = upcoming_1 >> S::splat(8);
            chars
        });

        (it, &self[L * n..])
    }
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

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn to_word(&self) -> usize {
        assert!(self.len() <= usize::BITS as usize / 2 - 3);
        let mask = usize::MAX >> (64 - 2 * self.len());
        unsafe { (*(self.seq.as_ptr() as *const usize) >> (2 * self.offset)) & mask }
    }

    #[inline(always)]
    fn sub_slice(&self, idx: usize, len: usize) -> Self {
        assert!(idx + len <= self.len);
        PackedSeq {
            seq: self.seq,
            offset: self.offset + idx,
            len,
        }
        .normalize()
    }

    #[inline(always)]
    fn iter_bp(&self) -> impl ExactSizeIterator<Item = u8> {
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
    fn par_iter_bp(&self, k: usize) -> (impl ExactSizeIterator<Item = S>, Self) {
        #[cfg(target_endian = "big")]
        panic!("Big endian architectures are not supported.");

        let this = self.normalize();
        assert_eq!(
            this.offset % 4,
            0,
            "Non-byte offsets are not yet supported."
        );

        let num_kmers = this.len.saturating_sub(k - 1);
        let n = (num_kmers / L) / 4 * 4;
        let bytes_per_chunk = n / 4;

        let base_ptr = this.seq.as_ptr();
        let offsets_lanes_0_4: u64x4 = from_fn(|l| (l * bytes_per_chunk) as u64).into();
        let offsets_lanes_4_8: u64x4 = from_fn(|l| ((4 + l) * bytes_per_chunk) as u64).into();
        let mut upcoming_1 = S::ZERO;
        let mut upcoming_2 = S::ZERO;

        let it = (0..if num_kmers == 0 { 0 } else { n + k - 1 }).map(move |i| {
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

impl OwnedPackedSeq {
    pub fn new(seq: &[u8]) -> Self {
        let mut packed_vec = Self {
            seq: vec![],
            len: 0,
        };
        let last = seq.len() / 8 * 8;
        for i in (0..last).step_by(8) {
            let chunk = &seq[i..i + 8].try_into().unwrap();
            let word = u64::from_ne_bytes(*chunk);
            let packed_byte = unsafe { std::arch::x86_64::_pext_u64(word, 0x0606060606060606) };
            packed_vec.seq.push(packed_byte as u8);
            packed_vec.seq.push((packed_byte >> 8) as u8);
            packed_vec.len += 8;
        }

        let mut packed_byte = 0;
        for &base in &seq[last..] {
            packed_byte |= match base {
                b'a' | b'A' => 0,
                b'c' | b'C' => 1,
                b'g' | b'G' => 3,
                b't' | b'T' => 2,
                b'\r' | b'\n' => continue,
                _ => panic!(),
            } << (packed_vec.len * 2);
            packed_vec.len += 1;
            if packed_vec.len % 4 == 0 {
                packed_vec.seq.push(packed_byte);
                packed_byte = 0;
            }
        }
        if packed_vec.len % 4 != 0 {
            packed_vec.seq.push(packed_byte);
        }
        packed_vec
    }
}

pub trait OwnedSeq: Default + Sync + SerializeInner + DeserializeInner {
    type BpSlice<'a>: Seq
    where
        Self: 'a;

    fn push(&mut self, seq: Self::BpSlice<'_>) -> Range<usize>;
    fn concat<'a>(input_seqs: impl Iterator<Item = Self::BpSlice<'a>>) -> (Self, Vec<Range<usize>>)
    where
        Self: Sized + 'a;
    fn get(&self) -> Self::BpSlice<'_>;
    fn slice(&self, range: Range<usize>) -> Self::BpSlice<'_>;
    fn size(&self) -> usize;
    #[cfg(test)]
    fn random(n: usize, alphabet: usize) -> Self;
}

impl OwnedSeq for Vec<u8> {
    type BpSlice<'a> = &'a [u8];

    fn push(&mut self, seq: Self::BpSlice<'_>) -> Range<usize> {
        let start = seq.len();
        let end = start + Seq::len(&seq);
        let range = start..end;
        self.extend(seq);
        range
    }

    fn concat<'a>(
        input_seqs: impl Iterator<Item = Self::BpSlice<'a>>,
    ) -> (Self, Vec<Range<usize>>) {
        let mut seq = vec![];
        let ranges = input_seqs
            .map(|slice| {
                let start = seq.len();
                let end = start + Seq::len(&slice);
                let range = start..end;
                seq.extend(slice);
                range
            })
            .collect();
        (seq, ranges)
    }
    fn get(&self) -> Self::BpSlice<'_> {
        &*self
    }
    fn slice(&self, range: Range<usize>) -> Self::BpSlice<'_> {
        &self[range]
    }
    fn size(&self) -> usize {
        size_of_val(self.as_slice())
    }
    #[cfg(test)]
    fn random(n: usize, alphabet: usize) -> Self {
        (0..n)
            .map(|_| ((rand::random::<u8>() as usize) % alphabet) as u8)
            .collect()
    }
}

#[derive(Debug, Default, Epserde)]
pub struct OwnedPackedSeq {
    pub seq: Vec<u8>,
    pub len: usize,
}

impl OwnedSeq for OwnedPackedSeq {
    type BpSlice<'a> = PackedSeq<'a>;

    fn push<'a>(&mut self, seq: Self::BpSlice<'a>) -> Range<usize> {
        let start = 4 * self.seq.len() + seq.offset;
        let end = start + Seq::len(&seq);
        let range = start..end;
        self.seq.extend(seq.seq);
        self.len = 4 * self.seq.len();
        range
    }
    fn concat<'a>(
        input_seqs: impl Iterator<Item = Self::BpSlice<'a>>,
    ) -> (Self, Vec<Range<usize>>) {
        let mut packed_vec = OwnedPackedSeq {
            len: 0,
            seq: vec![],
        };
        let ranges = input_seqs.map(|slice| packed_vec.push(slice)).collect();
        (packed_vec, ranges)
    }
    fn get(&self) -> Self::BpSlice<'_> {
        PackedSeq {
            seq: &self.seq,
            offset: 0,
            len: self.len,
        }
    }
    fn slice(&self, range: Range<usize>) -> Self::BpSlice<'_> {
        PackedSeq {
            seq: &self.seq,
            offset: range.start,
            len: range.len(),
        }
        .normalize()
    }
    fn size(&self) -> usize {
        size_of_val(self.seq.as_slice())
    }
    #[cfg(test)]
    fn random(n: usize, alphabet: usize) -> Self {
        assert!(alphabet == 4);
        let seq = (0..n.div_ceil(4)).map(|_| rand::random::<u8>()).collect();
        OwnedPackedSeq { seq, len: n }
    }
}
