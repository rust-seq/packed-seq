use traits::Seq;

use super::*;

/// A 2-bit packed non-owned slice of DNA bases.
#[derive(Copy, Clone, Debug, MemSize, MemDbg)]
pub struct PackedSeq<'s> {
    /// Packed data.
    pub seq: &'s [u8],
    /// Offset in bp from the start of the `seq`.
    pub offset: usize,
    /// Length of the sequence in bp, starting at `offset` from the start of `seq`.
    pub len: usize,
}

/// A 2-bit packed owned sequence of DNA bases.
#[derive(Clone, Debug, Default, MemSize, MemDbg)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
pub struct PackedSeqVec {
    pub seq: Vec<u8>,
    pub len: usize,
}

/// Pack an ASCII `ACTGactg` character into its 2-bit representation.
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

/// Unpack a 2-bit DNA base into the corresponding `ACTG` character.
pub fn unpack_base(base: u8) -> u8 {
    debug_assert!(base < 4, "Base {base} is not <4.");
    b"ACTG"[base as usize]
}

/// Complement an ASCII character: `A<>T` and `C<>G`.
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
pub const fn complement_base(base: u8) -> u8 {
    base ^ 2
}

/// Complement 8 lanes of 2-bit bases: `0<>2` and `1<>3`.
pub fn complement_base_simd(base: u32x8) -> u32x8 {
    base ^ u32x8::splat(2)
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

    /// Return a `Vec<u8>` of ASCII `ACTG` characters.
    pub fn unpack(&self) -> Vec<u8> {
        self.iter_bp().map(unpack_base).collect()
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
    fn get(&self, index: usize) -> u8 {
        let offset = self.offset + index;
        let idx = offset / 4;
        let offset = offset % 4;
        unsafe { (*self.seq.get_unchecked(idx) >> (2 * offset)) & 3 }
    }

    #[inline(always)]
    fn get_ascii(&self, index: usize) -> u8 {
        unpack_base(self.get(index))
    }

    /// Panics if `self` is longer than 29 characters.
    #[inline(always)]
    fn to_word(&self) -> usize {
        debug_assert!(self.len() <= usize::BITS as usize / 2 - 3);
        let mask = usize::MAX >> (64 - 2 * self.len());
        unsafe {
            ((self.seq.as_ptr() as *const usize).read_unaligned() >> (2 * self.offset)) & mask
        }
    }

    fn to_vec(&self) -> PackedSeqVec {
        assert_eq!(self.offset, 0);
        PackedSeqVec {
            seq: self.seq.to_vec(),
            len: self.len,
        }
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

        let par_len = if num_kmers == 0 { 0 } else { n + context - 1 };
        let it = (0..par_len).map(move |i| {
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

    #[inline(always)]
    fn par_iter_bp_delayed(
        self,
        context: usize,
        delay: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S)> + Clone, Self) {
        #[cfg(target_endian = "big")]
        panic!("Big endian architectures are not supported.");

        assert!(
            delay < usize::MAX / 2,
            "Delay={} should be >=0.",
            delay as isize
        );

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

        let par_len = if num_kmers == 0 { 0 } else { n + context - 1 };
        let it = (0..par_len).map(move |i| {
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

        let par_len = if num_kmers == 0 { 0 } else { n + context - 1 };
        let it = (0..par_len).map(move |i| {
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

    /// Compares 29 characters at a time.
    fn cmp_lcp(&self, other: &Self) -> (std::cmp::Ordering, usize) {
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
    /// Compares 29 characters at a time.
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
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
    /// Compares 29 characters at a time.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
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

impl SeqVec for PackedSeqVec {
    type Seq<'s> = PackedSeq<'s>;

    fn into_raw(self) -> Vec<u8> {
        self.seq
    }

    #[inline(always)]
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

    /// Push an ASCII sequence to an `PackedSeqVec`.
    /// `Aa` map to `0`, `Cc` to `1`, `Gg` to `3`, and `Tt` to `2`.
    /// Other characters may be silently mapped into `[0, 4)` or panic.
    /// (TODO: Explicitly support different conversions.)
    ///
    /// Uses the BMI2 `pext` instruction when available, based on the
    /// `n_to_bits_pext` method described at
    /// <https://github.com/Daniel-Liu-c0deb0t/cute-nucleotides>.
    ///
    /// TODO: Optimize for non-BMI2 platforms.
    /// TODO: Support multiple ways of dealing with non-`ACTG` characters.
    fn push_ascii(&mut self, seq: &[u8]) -> Range<usize> {
        let start_aligned = 4 * self.seq.len();
        let start = self.len;
        let len = seq.len();

        let unaligned = core::cmp::min(start_aligned - start, len);
        if unaligned > 0 {
            let mut packed_byte = *self.seq.last().unwrap();
            for &base in &seq[..unaligned] {
                packed_byte |= pack_char(base) << ((self.len % 4) * 2);
                self.len += 1;
            }
            *self.seq.last_mut().unwrap() = packed_byte;
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
        if self.len % 4 != 0 && last < len {
            self.seq.push(packed_byte);
        }
        start..start + len
    }

    fn random(n: usize) -> Self {
        let mut seq = vec![0; n.div_ceil(4)];
        rand::rngs::SmallRng::from_os_rng().fill_bytes(&mut seq);
        PackedSeqVec { seq, len: n }
    }
}
