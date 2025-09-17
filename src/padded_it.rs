//! FIXME Collect SIMD-iterator values into a flat `Vec<u32>`.

/// Trait alias for iterators over multiple chunks in parallel, typically over `u32x8`.
pub trait ChunkIt<T>: ExactSizeIterator<Item = T> {}
impl<T, I: ExactSizeIterator<Item = T>> ChunkIt<T> for I {}
use crate::intrinsics::transpose;
use std::mem::transmute;
use wide::u32x8;

/// An iterator over multiple lanes, with a given amount of padding at the end of the last lane(s).
pub struct PaddedIt<I> {
    pub it: I,
    pub padding: usize,
}

pub trait Advance {
    fn advance(self, n: usize) -> Self;
}
impl<I: ExactSizeIterator> Advance for I {
    #[inline(always)]
    fn advance(mut self, n: usize) -> Self {
        self.by_ref().take(n).for_each(drop);
        self
    }
}

impl<I> PaddedIt<I> {
    #[inline(always)]
    pub fn map<T, T2>(self, f: impl FnMut(T) -> T2) -> PaddedIt<impl ChunkIt<T2>>
    where
        I: ChunkIt<T>,
    {
        PaddedIt {
            it: self.it.map(f),
            padding: self.padding,
        }
    }

    #[inline(always)]
    pub fn advance<T>(mut self, n: usize) -> PaddedIt<impl ChunkIt<T>>
    where
        I: ChunkIt<T>,
    {
        self.it = self.it.advance(n);
        self
    }
}

impl<I: ChunkIt<u32x8>> PaddedIt<I> {
    /// Convenience wrapper around `collect_into`.
    pub fn collect(self) -> Vec<u32> {
        let mut v = vec![];
        self.collect_into(&mut v);
        v
    }

    /// Collect a SIMD-iterator into a single flat vector.
    /// Works by taking 8 elements from each stream, and transposing this SIMD-matrix before writing out the results.
    /// The `tail` is appended at the end.
    #[inline(always)]
    pub fn collect_into(self, out_vec: &mut Vec<u32>) {
        let PaddedIt { it, padding } = self;
        let len = it.len();
        out_vec.resize(len * 8, 0);

        let mut m = [unsafe { transmute([0; 8]) }; 8];
        let mut i = 0;
        it.for_each(|x| {
            m[i % 8] = x;
            if i % 8 == 7 {
                let t = transpose(m);
                for j in 0..8 {
                    unsafe {
                        *out_vec
                            .get_unchecked_mut(j * len + 8 * (i / 8)..)
                            .split_first_chunk_mut::<8>()
                            .unwrap()
                            .0 = transmute(t[j]);
                    }
                }
            }
            i += 1;
        });

        // Manually write the unfinished parts of length k=i%8.
        let t = transpose(m);
        let k = i % 8;
        for j in 0..8 {
            unsafe {
                out_vec[j * len + 8 * (i / 8)..j * len + 8 * (i / 8) + k]
                    .copy_from_slice(&transmute::<_, [u32; 8]>(t[j])[..k]);
            }
        }

        out_vec.resize(out_vec.len() - padding, 0);
    }
}
