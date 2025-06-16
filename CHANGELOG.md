# Changelog

## Git (3.0)
- Add `Seq::to_revcomp()`.
- Add `PackedSeqVec::clear()`.
- `PackedSeq::as_u64` now supports k up to 32, instead of only up to 29. (Fixes #2.)
- `PackedSeqVec::get` now uses checked indexing, to prevent unsafe out-of-bounds access.
- **Breaking**: `PackedSeqVec` and `PackedSeq` fields are now private, to uphold internal
  padding guarantees.
- Add convenience methods `{Seq, SeqVec}::{read_kmer, read_revcomp_kmer}` as
  shorthands for `.slice(pos..pos+k).as_u64()`.
- Deprecate `to_word` and `to_word_revcomp`. Instead `as_u64` and
  `revcomp_as_u64` are added. ('word' is not very clear, and `u64` makes more
  sense for kmers than `usize`.)
- Support `par_iter_bp` for sequences that do not start at byte offsets.
- Add `packed_seq::revcomp_u64()` and `Seq::revcomp_as_u64()` for reverse
  complementing kmers.
- Make `random` dependency optional but enabled by default (so it can be
  disabled for wasm backends).

## 2.0
- Change from tuples of (simd iterator, scalar tail iterator) to returning only a
  simd iterator and `padding: usize` that indicates the number of padded elements
  at the back that should be ignored.

## 1.0
- Initial release.
