use super::*;

/// Maps ASCII to `[0, 4)` on the fly.
/// Prefer first packing into a `PackedSeqVec` for storage.
impl<'s> Seq<'s> for &[u8] {
    const BASES_PER_BYTE: usize = 1;
    const BITS_PER_CHAR: usize = 8;
    type SeqVec = Vec<u8>;

    #[inline(always)]
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }

    #[inline(always)]
    fn get(&self, index: usize) -> u8 {
        self[index]
    }

    #[inline(always)]
    fn get_ascii(&self, index: usize) -> u8 {
        self[index]
    }

    #[inline(always)]
    fn to_word(&self) -> usize {
        assert!(self.len() <= usize::BITS as usize / 8);
        let mask = usize::MAX >> (64 - 8 * self.len());
        unsafe { (self.as_ptr() as *const usize).read_unaligned() & mask }
    }

    /// Convert to an owned version.
    fn to_vec(&self) -> Vec<u8> {
        <[u8]>::to_vec(self)
    }

    #[inline(always)]
    fn slice(&self, range: Range<usize>) -> Self {
        &self[range]
    }

    /// Iter the ASCII characters.
    #[inline(always)]
    fn iter_bp(self) -> impl ExactSizeIterator<Item = u8> + Clone {
        self.iter().copied()
    }

    /// Iter the ASCII characters in parallel.
    #[inline(always)]
    fn par_iter_bp(self, context: usize) -> (impl ExactSizeIterator<Item = S> + Clone, Self) {
        let num_kmers = self.len().saturating_sub(context - 1);
        let n = num_kmers / L;

        let base_ptr = self.as_ptr();
        let offsets_lanes_0_4: u64x4 = from_fn(|l| (l * n) as u64).into();
        let offsets_lanes_4_8: u64x4 = from_fn(|l| ((4 + l) * n) as u64).into();
        let mut upcoming_1 = S::ZERO;
        let mut upcoming_2 = S::ZERO;

        let par_len = if num_kmers == 0 { 0 } else { n + context - 1 };
        let it = (0..par_len).map(move |i| {
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
            // Extract the last 1-byte character.
            let chars = upcoming_1 & S::splat(0xff);
            // Shift remaining characters to the right.
            upcoming_1 = upcoming_1 >> S::splat(8);
            chars
        });

        (it, &self[L * n..])
    }

    #[inline(always)]
    fn par_iter_bp_delayed(
        self,
        context: usize,
        delay: usize,
    ) -> (impl ExactSizeIterator<Item = (S, S)> + Clone, Self) {
        let num_kmers = self.len().saturating_sub(context - 1);
        let n = num_kmers / L;

        let base_ptr = self.as_ptr();
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

        let par_len = if num_kmers == 0 { 0 } else { n + context - 1 };
        let it = (0..par_len).map(move |i| {
            if i % 4 == 0 {
                if i % 8 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat(i as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat(i as u64);
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
            if i % 4 == delay % 4 {
                unsafe { assert_unchecked(read_idx < buf.len()) };
                upcoming_d = buf[read_idx];
                read_idx += 1;
                read_idx &= buf_mask;
            }
            // Extract the last 2 bits of each character.
            let chars = upcoming & S::splat(0xff);
            let chars_d = upcoming_d & S::splat(0xff);
            // Shift remaining characters to the right.
            upcoming = upcoming >> S::splat(8);
            upcoming_d = upcoming_d >> S::splat(8);
            (chars, chars_d)
        });

        (it, &self[L * n..])
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

        let base_ptr = self.as_ptr();
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

        let par_len = if num_kmers == 0 { 0 } else { n + context - 1 };
        let it = (0..par_len).map(move |i| {
            if i % 4 == 0 {
                if i % 8 == 0 {
                    // Read a u64 containing the next 8 characters.
                    let idx_0_4 = offsets_lanes_0_4 + u64x4::splat(i as u64);
                    let idx_4_8 = offsets_lanes_4_8 + u64x4::splat(i as u64);
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
            let chars = upcoming & S::splat(0xff);
            let chars_d1 = upcoming_d1 & S::splat(0xff);
            let chars_d2 = upcoming_d2 & S::splat(0xff);
            // Shift remaining characters to the right.
            upcoming = upcoming >> S::splat(8);
            upcoming_d1 = upcoming_d1 >> S::splat(8);
            upcoming_d2 = upcoming_d2 >> S::splat(8);
            (chars, chars_d1, chars_d2)
        });

        (it, &self[L * n..])
    }

    // TODO: This is not very optimized.
    fn cmp_lcp(&self, other: &Self) -> (std::cmp::Ordering, usize) {
        for i in 0..self.len().min(other.len()) {
            if self[i] != other[i] {
                return (self[i].cmp(&other[i]), i);
            }
        }
        (self.len().cmp(&other.len()), self.len().min(other.len()))
    }
}

impl SeqVec for Vec<u8> {
    type Seq<'s> = &'s [u8];

    /// Get the underlying ASCII text.
    fn into_raw(self) -> Vec<u8> {
        self
    }

    fn as_slice(&self) -> Self::Seq<'_> {
        self.as_slice()
    }

    fn push_seq(&mut self, seq: &[u8]) -> Range<usize> {
        let start = self.len();
        let end = start + seq.len();
        let range = start..end;
        self.extend(seq);
        range
    }

    fn push_ascii(&mut self, seq: &[u8]) -> Range<usize> {
        self.push_seq(seq)
    }

    fn random(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u8>()).collect()
    }
}
