use super::*;

#[derive(Clone, Copy, Debug, MemSize, MemDbg)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
pub struct PackedNSeq<'s> {
    pub seq: PackedSeq<'s>,
    pub ambiguous: BitSeq<'s>,
}

#[derive(Clone, Debug, MemSize, MemDbg, Default)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
pub struct PackedNSeqVec {
    pub seq: PackedSeqVec,
    pub ambiguous: BitSeqVec,
}

/// Implement a subset of `Seq` for `PackedNSeq`.
impl<'s> PackedNSeq<'s> {
    pub fn to_revcomp(&self) -> PackedNSeqVec {
        PackedNSeqVec {
            seq: self.seq.to_revcomp(),
            ambiguous: self.ambiguous.to_revcomp(),
        }
    }

    pub fn slice(&self, range: Range<usize>) -> PackedNSeq<'s> {
        PackedNSeq {
            seq: self.seq.slice(range.clone()),
            ambiguous: self.ambiguous.slice(range),
        }
    }
}

/// Implement a subset of `SeqVec` for `PackedNSeqVec`.
impl PackedNSeqVec {
    pub fn clear(&mut self) {
        self.seq.clear();
        self.ambiguous.clear();
    }

    pub fn random(len: usize, n_frac: f32) -> Self {
        let seq = PackedSeqVec::random(len);
        let ambiguous = BitSeqVec::random(len, n_frac);
        Self { seq, ambiguous }
    }

    pub fn as_slice(&self) -> PackedNSeq<'_> {
        PackedNSeq {
            seq: self.seq.as_slice(),
            ambiguous: self.ambiguous.as_slice(),
        }
    }

    pub fn slice(&self, range: Range<usize>) -> PackedNSeq<'_> {
        self.as_slice().slice(range).to_owned()
    }

    pub fn push_ascii(&mut self, seq: &[u8]) -> Range<usize> {
        let r1 = self.seq.push_ascii(seq);
        let r2 = self.ambiguous.push_ascii(seq);
        assert_eq!(r1, r2);
        r1
    }

    pub fn from_ascii(seq: &[u8]) -> Self {
        Self {
            seq: PackedSeqVec::from_ascii(seq),
            ambiguous: BitSeqVec::from_ascii(seq),
        }
    }
}
