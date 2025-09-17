/// Trait alias for iterators over multiple chunks in parallel, typically over `u32x8`.
pub trait ChunkIt<T>: ExactSizeIterator<Item = T> {}
impl<T, I: ExactSizeIterator<Item = T>> ChunkIt<T> for I {}

/// An iterator over multiple lanes, with a given amount of padding at the end of the last lane(s).
pub struct PaddedIt<I> {
    pub it: I,
    pub padding: usize,
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
