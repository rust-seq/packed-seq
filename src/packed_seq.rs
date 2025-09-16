use traits::Seq;

use crate::intrinsics::transpose;

use super::*;

/// A 2-bit packed non-owned slice of DNA bases.
#[derive(Copy, Clone, Debug, MemSize, MemDbg)]
pub struct PackedSeq<'s> {
    /// Packed data.
    seq: &'s [u8],
    /// Offset in bp from the start of the `seq`.
    offset: usize,
    /// Length of the sequence in bp, starting at `offset` from the start of `seq`.
    len: usize,
}

/// Number of padding bytes at the end of `PackedSeqVec::seq`.
const PADDING: usize = 16;

/// A 2-bit packed owned sequence of DNA bases.
#[derive(Clone, Debug, MemSize, MemDbg)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
pub struct PackedSeqVec {
    /// NOTE: We maintain the invariant that this has at least 16 bytes padding
    /// at the end after `len` finishes.
    /// This ensures that `read_unaligned` in `as_64` works OK.
    pub(crate) seq: Vec<u8>,

    /// The length, in bp, of the underlying sequence. See `.len()`.
    len: usize,
}

impl Default for PackedSeqVec {
    fn default() -> Self {
        Self {
            seq: vec![0; PADDING],
            len: 0,
        }
    }
}

/// Pack an ASCII `ACTGactg` character into its 2-bit representation, and panic for anything else.
#[inline(always)]
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

/// Pack an ASCII `ACTGactg` character into its 2-bit representation, and silently convert other characters into 0..4 as well.
#[inline(always)]
pub fn pack_char_lossy(base: u8) -> u8 {
    (base >> 1) & 3
}

/// Unpack a 2-bit DNA base into the corresponding `ACTG` character.
#[inline(always)]
pub fn unpack_base(base: u8) -> u8 {
    debug_assert!(base < 4, "Base {base} is not <4.");
    b"ACTG"[base as usize]
}

/// Complement an ASCII character: `A<>T` and `C<>G`.
#[inline(always)]
pub const fn complement_char(base: u8) -> u8 {
    match base {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        _ => panic!("Unexpected character. Expected one of ACTGactg.",),
    }
}

/// Complement a 2-bit base: `0<>2` and `1<>3`.
#[inline(always)]
pub const fn complement_base(base: u8) -> u8 {
    base ^ 2
}

/// Complement 8 lanes of 2-bit bases: `0<>2` and `1<>3`.
#[inline(always)]
pub fn complement_base_simd(base: u32x8) -> u32x8 {
    base ^ u32x8::splat(2)
}

/// Reverse complement the 2-bit pairs in the input.
#[inline(always)]
const fn revcomp_raw(word: u64) -> u64 {
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        let mut res = word.reverse_bits(); // ARM can reverse bits in a single instruction
        res = ((res >> 1) & 0x5555_5555_5555_5555) | ((res & 0x5555_5555_5555_5555) << 1);
        res ^ 0xAAAA_AAAA_AAAA_AAAA
    }

    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    {
        let mut res = word.swap_bytes();
        res = ((res >> 4) & 0x0F0F_0F0F_0F0F_0F0F) | ((res & 0x0F0F_0F0F_0F0F_0F0F) << 4);
        res = ((res >> 2) & 0x3333_3333_3333_3333) | ((res & 0x3333_3333_3333_3333) << 2);
        res ^ 0xAAAA_AAAA_AAAA_AAAA
    }
}

/// Compute the reverse complement of a short sequence packed in a `u64`.
#[inline(always)]
pub const fn revcomp_u64(word: u64, len: usize) -> u64 {
    revcomp_raw(word) >> (usize::BITS as usize - 2 * len)
}

#[inline(always)]
pub const fn revcomp_u128(word: u128, len: usize) -> u128 {
    let low = word as u64;
    let high = (word >> 64) as u64;
    let rlow = revcomp_raw(low);
    let rhigh = revcomp_raw(high);
    let out = ((rlow as u128) << 64) | rhigh as u128;
    out >> (u128::BITS as usize - 2 * len)
}

impl PackedSeq<'_> {
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

    /// Return a `Vec<u8>` of ASCII `ACTG` characters.
    #[inline(always)]
    pub fn unpack(&self) -> Vec<u8> {
        self.iter_bp().map(unpack_base).collect()
    }
}

#[inline(always)]
pub(crate) fn read_slice(seq: &[u8], idx: usize) -> u32x8 {
    // assert!(idx <= seq.len());
    let mut result = [0u8; 32];
    let num_bytes = 32.min(seq.len().saturating_sub(idx));
    unsafe {
        let src = seq.as_ptr().add(idx);
        std::ptr::copy_nonoverlapping(src, result.as_mut_ptr(), num_bytes);
        std::mem::transmute(result)
    }
}

impl<'s> Seq<'s> for PackedSeq<'s> {
    const BASES_PER_BYTE: usize = 4;
    const BITS_PER_CHAR: usize = 2;
    type SeqVec = PackedSeqVec;

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    fn get(&self, index: usize) -> u8 {
        let offset = self.offset + index;
        let idx = offset / 4;
        let offset = offset % 4;
        (self.seq[idx] >> (2 * offset)) & 3
    }

    #[inline(always)]
    fn get_ascii(&self, index: usize) -> u8 {
        unpack_base(self.get(index))
    }

    /// Convert a short sequence (kmer) to a packed representation as `u64`.
    /// Panics if `self` is longer than 32 characters.
    #[inline(always)]
    fn as_u64(&self) -> u64 {
        assert!(self.len() <= 32);
        debug_assert!(self.seq.len() <= 9);

        let mask = u64::MAX >> (64 - 2 * self.len());

        // The unaligned read is OK, because we ensure that the underlying `PackedSeqVec::seq` always
        // has at least 16 bytes (the size of a u128) of padding at the end.
        if self.len() <= 29 {
            let x = unsafe { (self.seq.as_ptr() as *const u64).read_unaligned() };
            (x >> (2 * self.offset)) & mask
        } else {
            let x = unsafe { (self.seq.as_ptr() as *const u128).read_unaligned() };
            (x >> (2 * self.offset)) as u64 & mask
        }
    }

    /// Convert a short sequence (kmer) to a packed representation as `u128`.
    /// Panics if `self` is longer than 64 characters.
    #[inline(always)]
    fn as_u128(&self) -> u128 {
        assert!(self.len() <= 61, "Sequences >61 long cannot be read with a single unaligned u128 read.");
        debug_assert!(self.seq.len() <= 17);

        let mask = u128::MAX >> (128 - 2 * self.len());

        // The unaligned read is OK, because we ensure that the underlying `PackedSeqVec::seq` always
        // has at least 16 bytes (the size of a u128) of padding at the end.
        let x = unsafe { (self.seq.as_ptr() as *const u128).read_unaligned() };
        (x >> (2 * self.offset)) & mask
    }

    /// Convert a short sequence (kmer) to a packed representation of its reverse complement as `usize`.
    /// Panics if `self` is longer than 32 characters.
    #[inline(always)]
    fn revcomp_as_u64(&self) -> u64 {
        revcomp_u64(self.as_u64(), self.len())
    }

    /// Convert a short sequence (kmer) to a packed representation of its reverse complement as `usize`.
    /// Panics if `self` is longer than 64 characters.
    #[inline(always)]
    fn revcomp_as_u128(&self) -> u128 {
        revcomp_u128(self.as_u128(), self.len())
    }

    #[inline(always)]
    fn to_vec(&self) -> PackedSeqVec {
        assert_eq!(self.offset, 0);
        PackedSeqVec {
            seq: self
                .seq
                .iter()
                .copied()
                .chain(std::iter::repeat_n(0u8, PADDING))
                .collect(),
            len: self.len,
        }
    }

    fn to_revcomp(&self) -> PackedSeqVec {
        let mut seq = self.seq[..(self.offset + self.len).div_ceil(4)]
            .iter()
            // 1. reverse the bytes
            .rev()
            .copied()
            .map(|mut res| {
                // 2. swap the bases in the byte
                // This is auto-vectorized.
                res = ((res >> 4) & 0x0F) | ((res & 0x0F) << 4);
                res = ((res >> 2) & 0x33) | ((res & 0x33) << 2);
                res ^ 0xAA
            })
            .chain(std::iter::repeat_n(0u8, PADDING))
            .collect::<Vec<u8>>();

        // 3. Shift away the offset.
        let new_offset = (4 - (self.offset + self.len) % 4) % 4;

        if new_offset > 0 {
            // Shift everything left by `2*new_offset` bits.
            let shift = 2 * new_offset;
            *seq.last_mut().unwrap() >>= shift;
            // This loop is also auto-vectorized.
            for i in 0..seq.len() - 1 {
                seq[i] = (seq[i] >> shift) | (seq[i + 1] << (8 - shift));
            }
        }

        PackedSeqVec { seq, len: self.len }
    }

    #[inline(always)]
    fn slice(&self, range: Range<usize>) -> Self {
        debug_assert!(
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
    fn iter_bp(self) -> impl ExactSizeIterator<Item = u8> {
        assert!(self.len <= self.seq.len() * 4);

        let this = self.normalize();

        // read u64 at a time?
        let mut byte = 0;
        let mut it = (0..this.len + this.offset).map(
            #[inline(always)]
            move |i| {
                if i % 4 == 0 {
                    byte = this.seq[i / 4];
                }
                // Shift byte instead of i?
                (byte >> (2 * (i % 4))) & 0b11
            },
        );
        it.by_ref().take(this.offset).for_each(drop);
        it
    }

    #[inline(always)]
    fn par_iter_bp(self, context: usize) -> (impl ExactSizeIterator<Item = S> + Clone, usize) {
        #[cfg(target_endian = "big")]
        panic!("Big endian architectures are not supported.");

        let this = self.normalize();
        let o = this.offset;
        assert!(o < 4);

        let num_kmers = if this.len == 0 {
            0
        } else {
            (this.len + o).saturating_sub(context - 1)
        };
        // without +o, since we don't need them in the stride.
        let num_kmers_stride = this.len.saturating_sub(context - 1);
        let n = num_kmers_stride.div_ceil(L).next_multiple_of(4);
        let bytes_per_chunk = n / 4;
        let padding = 4 * L * bytes_per_chunk - num_kmers_stride;

        let offsets: [usize; 8] = from_fn(|l| l * bytes_per_chunk);
        let mut cur = S::ZERO;

        // Boxed, so it doesn't consume precious registers.
        // Without this, cur is not always inlined into a register.
        let mut buf = Box::new([S::ZERO; 8]);

        // We skip the first o iterations.
        let par_len = if num_kmers == 0 {
            0
        } else {
            n + context + o - 1
        };
        let mut it = (0..par_len).map(
            #[inline(always)]
            move |i| {
                if i % 16 == 0 {
                    if i % 128 == 0 {
                        // Read a u256 for each lane containing the next 128 characters.
                        let data: [u32x8; 8] = from_fn(
                            #[inline(always)]
                            |lane| read_slice(this.seq, offsets[lane] + (i / 4)),
                        );
                        *buf = transpose(data);
                    }
                    cur = buf[(i % 128) / 16];
                }
                // Extract the last 2 bits of each character.
                let chars = cur & S::splat(0x03);
                // Shift remaining characters to the right.
                cur = cur >> S::splat(2);
                chars
            },
        );
        // Drop the first few chars.
        it.by_ref().take(o).for_each(drop);

        (it, padding)
    }

    /// NOTE: When `self` starts does not start at a byte boundary, the
    /// 'delayed' character is not guaranteed to be `0`.
    #[inline(always)]
    fn par_iter_bp_delayed(
        self,
        context: usize,
        delay: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S)> + Clone, usize) {
        #[cfg(target_endian = "big")]
        panic!("Big endian architectures are not supported.");

        assert!(
            delay < usize::MAX / 2,
            "Delay={} should be >=0.",
            delay as isize
        );

        let this = self.normalize();
        let o = this.offset;
        assert!(o < 4);

        let num_kmers = if this.len == 0 {
            0
        } else {
            (this.len + o).saturating_sub(context - 1)
        };
        // without +o, since we don't need them in the stride.
        let num_kmers_stride = this.len.saturating_sub(context - 1);
        let n = num_kmers_stride.div_ceil(L).next_multiple_of(4);
        let bytes_per_chunk = n / 4;
        let padding = 4 * L * bytes_per_chunk - num_kmers_stride;

        let offsets: [usize; 8] = from_fn(|l| l * bytes_per_chunk);
        let mut upcoming = S::ZERO;
        let mut upcoming_d = S::ZERO;

        // Even buf_len is nice to only have the write==buf_len check once.
        // We also make it the next power of 2, for faster modulo operations.
        // delay/16: number of bp in a u32.
        let buf_len = (delay / 16 + 8).next_power_of_two();
        let buf_mask = buf_len - 1;
        let mut buf = vec![S::ZERO; buf_len];
        let mut write_idx = 0;
        // We compensate for the first delay/16 triggers of the check below that
        // happen before the delay is actually reached.
        let mut read_idx = (buf_len - delay / 16) % buf_len;

        let par_len = if num_kmers == 0 {
            0
        } else {
            n + context + o - 1
        };
        let mut it = (0..par_len).map(
            #[inline(always)]
            move |i| {
                if i % 16 == 0 {
                    if i % 128 == 0 {
                        // Read a u256 for each lane containing the next 128 characters.
                        let data: [u32x8; 8] = from_fn(
                            #[inline(always)]
                            |lane| read_slice(this.seq, offsets[lane] + (i / 4)),
                        );
                        unsafe {
                            *TryInto::<&mut [u32x8; 8]>::try_into(
                                buf.get_unchecked_mut(write_idx..write_idx + 8),
                            )
                            .unwrap_unchecked() = transpose(data);
                        }
                        if i == 0 {
                            // Mask out chars before the offset.
                            let elem = !((1u32 << (2 * o)) - 1);
                            let mask = S::splat(elem);
                            buf[write_idx] &= mask;
                        }
                    }
                    upcoming = buf[write_idx];
                    write_idx += 1;
                    write_idx &= buf_mask;
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
            },
        );
        it.by_ref().take(o).for_each(drop);

        (it, padding)
    }

    /// NOTE: When `self` starts does not start at a byte boundary, the
    /// 'delayed' character is not guaranteed to be `0`.
    #[inline(always)]
    fn par_iter_bp_delayed_2(
        self,
        context: usize,
        delay1: usize,
        delay2: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S, S)> + Clone, usize) {
        #[cfg(target_endian = "big")]
        panic!("Big endian architectures are not supported.");

        let this = self.normalize();
        let o = this.offset;
        assert!(o < 4);
        assert!(delay1 <= delay2, "Delay1 must be at most delay2.");

        let num_kmers = if this.len == 0 {
            0
        } else {
            (this.len + o).saturating_sub(context - 1)
        };
        // without +o, since we don't need them in the stride.
        let num_kmers_stride = this.len.saturating_sub(context - 1);
        let n = num_kmers_stride.div_ceil(L).next_multiple_of(4);
        let bytes_per_chunk = n / 4;
        let padding = 4 * L * bytes_per_chunk - num_kmers_stride;

        let offsets: [usize; 8] = from_fn(|l| l * bytes_per_chunk);
        let mut upcoming = S::ZERO;
        let mut upcoming_d1 = S::ZERO;
        let mut upcoming_d2 = S::ZERO;

        // Even buf_len is nice to only have the write==buf_len check once.
        let buf_len = (delay2 / 16 + 8).next_power_of_two();
        let buf_mask = buf_len - 1;
        let mut buf = vec![S::ZERO; buf_len];
        let mut write_idx = 0;
        // We compensate for the first delay/16 triggers of the check below that
        // happen before the delay is actually reached.
        let mut read_idx1 = (buf_len - delay1 / 16) % buf_len;
        let mut read_idx2 = (buf_len - delay2 / 16) % buf_len;

        let par_len = if num_kmers == 0 {
            0
        } else {
            n + context + o - 1
        };
        let mut it = (0..par_len).map(
            #[inline(always)]
            move |i| {
                if i % 16 == 0 {
                    if i % 128 == 0 {
                        // Read a u256 for each lane containing the next 128 characters.
                        let data: [u32x8; 8] = from_fn(
                            #[inline(always)]
                            |lane| read_slice(this.seq, offsets[lane] + (i / 4)),
                        );
                        unsafe {
                            *TryInto::<&mut [u32x8; 8]>::try_into(
                                buf.get_unchecked_mut(write_idx..write_idx + 8),
                            )
                            .unwrap_unchecked() = transpose(data);
                        }
                        if i == 0 {
                            // Mask out chars before the offset.
                            let elem = !((1u32 << (2 * o)) - 1);
                            let mask = S::splat(elem);
                            buf[write_idx] &= mask;
                        }
                    }
                    upcoming = buf[write_idx];
                    write_idx += 1;
                    write_idx &= buf_mask;
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
            },
        );
        it.by_ref().take(o).for_each(drop);

        (it, padding)
    }

    /// Compares 29 characters at a time.
    fn cmp_lcp(&self, other: &Self) -> (std::cmp::Ordering, usize) {
        let mut lcp = 0;
        let min_len = self.len.min(other.len);
        for i in (0..min_len).step_by(29) {
            let len = (min_len - i).min(29);
            let this = self.slice(i..i + len);
            let other = other.slice(i..i + len);
            let this_word = this.as_u64();
            let other_word = other.as_u64();
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
    /// Compares 29 characters at a time.
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        for i in (0..self.len).step_by(29) {
            let len = (self.len - i).min(29);
            let this = self.slice(i..i + len);
            let that = other.slice(i..i + len);
            if this.as_u64() != that.as_u64() {
                return false;
            }
        }
        true
    }
}

impl Eq for PackedSeq<'_> {}

impl PartialOrd for PackedSeq<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PackedSeq<'_> {
    /// Compares 29 characters at a time.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let min_len = self.len.min(other.len);
        for i in (0..min_len).step_by(29) {
            let len = (min_len - i).min(29);
            let this = self.slice(i..i + len);
            let other = other.slice(i..i + len);
            let this_word = this.as_u64();
            let other_word = other.as_u64();
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

impl SeqVec for PackedSeqVec {
    type Seq<'s> = PackedSeq<'s>;

    #[inline(always)]
    fn into_raw(mut self) -> Vec<u8> {
        self.seq.resize(self.len.div_ceil(4), 0);
        self.seq
    }

    #[inline(always)]
    fn as_slice(&self) -> Self::Seq<'_> {
        PackedSeq {
            seq: &self.seq[..self.len.div_ceil(4)],
            offset: 0,
            len: self.len,
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.seq.clear();
        self.len = 0;
    }

    fn push_seq<'a>(&mut self, seq: PackedSeq<'_>) -> Range<usize> {
        let start = self.len.next_multiple_of(4) + seq.offset;
        let end = start + seq.len();
        // Reserve *additional* capacity.
        self.seq.reserve(seq.seq.len());
        // Shrink away the padding.
        self.seq.resize(self.len.div_ceil(4), 0);
        // Extend.
        self.seq.extend(seq.seq);
        // Push padding.
        self.seq.extend(std::iter::repeat_n(0u8, PADDING));
        self.len = end;
        start..end
    }

    /// Push an ASCII sequence to an `PackedSeqVec`.
    /// `Aa` map to `0`, `Cc` to `1`, `Gg` to `3`, and `Tt` to `2`.
    /// Other characters are silently mapped into `0..4`.
    ///
    /// Uses the BMI2 `pext` instruction when available, based on the
    /// `n_to_bits_pext` method described at
    /// <https://github.com/Daniel-Liu-c0deb0t/cute-nucleotides>.
    ///
    /// TODO: Support multiple ways of dealing with non-`ACTG` characters:
    /// - panic on non-`ACGT`,
    /// - filter out non-`ACGT`.
    fn push_ascii(&mut self, seq: &[u8]) -> Range<usize> {
        self.seq
            .resize((self.len + seq.len()).div_ceil(4) + PADDING, 0);
        let start_aligned = self.len.next_multiple_of(4);
        let start = self.len;
        let len = seq.len();
        let mut idx = self.len / 4;

        let unaligned = core::cmp::min(start_aligned - start, len);
        if unaligned > 0 {
            let mut packed_byte = self.seq[idx];
            for &base in &seq[..unaligned] {
                packed_byte |= pack_char_lossy(base) << ((self.len % 4) * 2);
                self.len += 1;
            }
            self.seq[idx] = packed_byte;
            idx += 1;
        }

        #[allow(unused)]
        let mut last = unaligned;

        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            last = unaligned + (len - unaligned) / 8 * 8;

            for i in (unaligned..last).step_by(8) {
                let chunk = &seq[i..i + 8].try_into().unwrap();
                let ascii = u64::from_ne_bytes(*chunk);
                let packed_bytes =
                    unsafe { std::arch::x86_64::_pext_u64(ascii, 0x0606060606060606) };
                self.seq[idx] = packed_bytes as u8;
                idx += 1;
                self.seq[idx] = (packed_bytes >> 8) as u8;
                idx += 1;
                self.len += 8;
            }
        }

        #[cfg(target_feature = "neon")]
        {
            use core::arch::aarch64::{vandq_u8, vdup_n_u8, vld1q_u8, vpadd_u8, vshlq_u8, vst1_u8};
            use core::mem::transmute;

            last = unaligned + (len - unaligned) / 16 * 16;

            for i in (unaligned..last).step_by(16) {
                unsafe {
                    let ascii = vld1q_u8(seq.as_ptr().add(i));
                    let masked_bits = vandq_u8(ascii, transmute([6i8; 16]));
                    let (bits_0, bits_1) = transmute(vshlq_u8(
                        masked_bits,
                        transmute([-1i8, 1, 3, 5, -1, 1, 3, 5, -1, 1, 3, 5, -1, 1, 3, 5]),
                    ));
                    let half_packed = vpadd_u8(bits_0, bits_1);
                    let packed = vpadd_u8(half_packed, vdup_n_u8(0));
                    vst1_u8(self.seq.as_mut_ptr().add(idx), packed);
                    idx += 4;
                    self.len += 16;
                }
            }
        }

        let mut packed_byte = 0;
        for &base in &seq[last..] {
            packed_byte |= pack_char_lossy(base) << ((self.len % 4) * 2);
            self.len += 1;
            if self.len % 4 == 0 {
                self.seq[idx] = packed_byte;
                idx += 1;
                packed_byte = 0;
            }
        }
        if self.len % 4 != 0 && last < len {
            self.seq[idx] = packed_byte;
            idx += 1;
        }
        assert_eq!(idx + PADDING, self.seq.len());
        start..start + len
    }

    #[cfg(feature = "rand")]
    fn random(n: usize) -> Self {
        use rand::{RngCore, SeedableRng};

        let byte_len = n.div_ceil(4);
        let mut seq = vec![0; byte_len + PADDING];
        rand::rngs::SmallRng::from_os_rng().fill_bytes(&mut seq[..byte_len]);
        // Ensure that the last byte is padded with zeros.
        if n % 4 != 0 {
            seq[byte_len - 1] &= (1 << (2 * (n % 4))) - 1;
        }

        Self { seq, len: n }
    }
}
