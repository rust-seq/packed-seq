use super::u32x8;
use mem_dbg::{MemDbg, MemSize};
use std::ops::Range;

/// A non-owned slice of characters.
///
/// The represented character values are expected to be in `[0, 2^b)`,
/// but they can be encoded in various ways. E.g.:
/// - A `&[u8]` of ASCII characters, returning 8-bit values.
/// - An `AsciiSeq` of DNA characters `ACGT`, interpreted 2-bit values.
/// - A `PackedSeq` of packed DNA characters (4 per byte), returning 2-bit values.
///
/// Each character is assumed to fit in 8 bits. Some functions take or return
/// this 'unpacked' (ASCII) character.
pub trait Seq<'s>: Copy + Eq + Ord {
    /// Number of encoded characters per byte of memory of the `Seq`.
    const BASES_PER_BYTE: usize;
    /// Number of bits `b` to represent each character returned by `iter_bp` and variants..
    const BITS_PER_CHAR: usize;

    /// The corresponding owned sequence type.
    type SeqVec: SeqVec;

    /// Convenience function that returns `b=Self::BITS_PER_CHAR`.
    fn bits_per_char(&self) -> usize {
        Self::BITS_PER_CHAR
    }

    /// The length of the sequence in characters.
    fn len(&self) -> usize;

    /// Get the character at the given index.
    fn get(&self, _index: usize) -> u8;

    /// Get the ASCII character at the given index, _without_ mapping to `b`-bit values.
    fn get_ascii(&self, _index: usize) -> u8;

    /// Convert a short sequence (kmer) to a packed representation as `usize`.
    fn to_word(&self) -> usize;

    /// Convert a short sequence (kmer) to a packed representation of its reverse complement as `usize`.
    fn to_word_revcomp(&self) -> usize;

    /// Compute the reverse complement of a short sequence packed in a `usize`.
    #[inline(always)]
    fn revcomp_word(word: usize, len: usize) -> usize {
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        {
            let mut res = word.reverse_bits(); // ARM can reverse bits in a single instruction
            res = ((res >> 1) & 0x5555_5555_5555_5555) | ((res & 0x5555_5555_5555_5555) << 1);
            res ^= 0xAAAA_AAAA_AAAA_AAAA;
            res >> (usize::BITS as usize - 2 * len)
        }

        #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
        {
            let mut res = word.swap_bytes();
            res = ((res >> 4) & 0x0F0F_0F0F_0F0F_0F0F) | ((res & 0x0F0F_0F0F_0F0F_0F0F) << 4);
            res = ((res >> 2) & 0x3333_3333_3333_3333) | ((res & 0x3333_3333_3333_3333) << 2);
            res ^= 0xAAAA_AAAA_AAAA_AAAA;
            res >> (usize::BITS as usize - 2 * len)
        }
    }

    /// Convert to an owned version.
    fn to_vec(&self) -> Self::SeqVec;

    /// Get a sub-slice of the sequence.
    /// `range` indicates character indices.
    fn slice(&self, range: Range<usize>) -> Self;

    /// Iterate over the `b`-bit characters of the sequence.
    fn iter_bp(self) -> impl ExactSizeIterator<Item = u8> + Clone;

    /// Iterate over 8 chunks of `b`-bit characters of the sequence in parallel.
    ///
    /// This splits the input into 8 chunks and streams over them in parallel.
    /// The second output returns the number of 'padding' characters that was added to get a full number of SIMD lanes.
    /// Thus, the last `padding` number of returned elements (from the last lane(s)) should be ignored.
    /// The context can be e.g. the k-mer size being iterated.
    /// When `context>1`, consecutive chunks overlap by `context-1` bases.
    ///
    /// Expected to be implemented using SIMD instructions.
    fn par_iter_bp(self, context: usize) -> (impl ExactSizeIterator<Item = u32x8> + Clone, usize);

    /// Iterate over 8 chunks of the sequence in parallel, returning two characters offset by `delay` positions.
    ///
    /// Returned pairs are `(add, remove)`, and the first `delay` 'remove' characters are always `0`.
    ///
    /// For example, when the sequence starts as `ABCDEF...`, and `delay=2`,
    /// the first returned tuples in the first lane are:
    /// `(b'A', 0)`, `(b'B', 0)`, `(b'C', b'A')`, `(b'D', b'B')`.
    ///
    /// When `context>1`, consecutive chunks overlap by `context-1` bases:
    /// the first `context-1` 'added' characters of the second chunk overlap
    /// with the last `context-1` 'added' characters of the first chunk.
    fn par_iter_bp_delayed(
        self,
        context: usize,
        delay: usize,
    ) -> (impl ExactSizeIterator<Item = (u32x8, u32x8)> + Clone, usize);

    /// Iterate over 8 chunks of the sequence in parallel, returning three characters:
    /// the char added, the one `delay` positions before, and the one `delay2` positions before.
    ///
    /// Requires `delay1 <= delay2`.
    ///
    /// Returned pairs are `(add, d1, d2)`. The first `delay1` `d1` characters and first `delay2` `d2` are always `0`.
    ///
    /// For example, when the sequence starts as `ABCDEF...`, and `delay1=2` and `delay2=3`,
    /// the first returned tuples in the first lane are:
    /// `(b'A', 0, 0)`, `(b'B', 0, 0)`, `(b'C', b'A', 0)`, `(b'D', b'B', b'A')`.
    ///
    /// When `context>1`, consecutive chunks overlap by `context-1` bases:
    /// the first `context-1` 'added' characters of the second chunk overlap
    /// with the last `context-1` 'added' characters of the first chunk.
    fn par_iter_bp_delayed_2(
        self,
        context: usize,
        delay1: usize,
        delay2: usize,
    ) -> (
        impl ExactSizeIterator<Item = (u32x8, u32x8, u32x8)> + Clone,
        usize,
    );

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
        impl SerializeInner for crate::AsciiSeqVec {}
        impl DeserializeInner for crate::AsciiSeqVec {}
        impl SerializeInner for crate::PackedSeqVec {}
        impl DeserializeInner for crate::PackedSeqVec {}
    }
}

/// An owned sequence.
/// Can be constructed from either ASCII input or the underlying non-owning `Seq` type.
///
/// Implemented for:
/// - A `Vec<u8>` of ASCII characters, returning 8-bit values.
/// - An `AsciiSeqVec` of DNA characters `ACGT`, interpreted as 2-bit values.
/// - A `PackedSeqVec` of packed DNA characters (4 per byte), returning 2-bit values.
pub trait SeqVec:
    Default + Sync + SerializeInner + DeserializeInner + MemSize + MemDbg + Clone + 'static
{
    type Seq<'s>: Seq<'s>;

    /// Get a non-owning slice to the underlying sequence.
    ///
    /// Unfortunately, `Deref` into a `Seq` can not be supported.
    fn as_slice(&self) -> Self::Seq<'_>;

    /// Get a sub-slice of the sequence. Indices are character offsets.
    #[inline(always)]
    fn slice(&self, range: Range<usize>) -> Self::Seq<'_> {
        self.as_slice().slice(range)
    }

    /// The length of the sequence in characters.
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Convert into the underlying raw representation.
    fn into_raw(self) -> Vec<u8>;

    /// Generate a random sequence with the given number of characters.
    fn random(n: usize) -> Self;

    /// Create a `SeqVec` from ASCII input.
    fn from_ascii(seq: &[u8]) -> Self {
        let mut packed_vec = Self::default();
        packed_vec.push_ascii(seq);
        packed_vec
    }

    /// Append the given sequence to the underlying storage.
    ///
    /// This may leave gaps (padding) between consecutively pushed sequences to avoid re-aligning the pushed data.
    /// Returns the range of indices corresponding to the pushed sequence.
    /// Use `self.slice(range)` to get the corresponding slice.
    fn push_seq(&mut self, seq: Self::Seq<'_>) -> Range<usize>;

    /// Append the given ASCII sequence to the underlying storage.
    ///
    /// This may leave gaps (padding) between consecutively pushed sequences to avoid re-aligning the pushed data.
    /// Returns the range of indices corresponding to the pushed sequence.
    /// Use `self.slice(range)` to get the corresponding slice.
    fn push_ascii(&mut self, seq: &[u8]) -> Range<usize>;
}
