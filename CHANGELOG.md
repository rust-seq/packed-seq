# Changelog

## 4.0.1
- feature: optimize `BitSeq::from_ascii` for NEON.
- bugfix: 1-bit `reverse` was also doing 'complement'.

## 4.0.0
- **Feature**: Add `BitSeq` and `BitSeqVec` as 1-bit encoded 'sequences' that by default
  indicate the positions of ambiguous (non-ACTG) characters.
- **Feature**: Add `BitSeq::par_iter_kmer_ambiguity` that returns for each kmer
  whether it contains an ambiguous base.
- **Breaking**: Encapsulate parallel iterators in new `PaddedIt { it, padding: usize }` type with `.map`, `.advance`, `zip`, and `.collect_into` functions.
- **Breaking**: Make `delay` passed into `par_iter_bp_delayed` a strong type `Delay(pub usize)` to
  reduce potential for bugs.
- Make `intrinsics::transpose` public for use in `collect_and_dedup` in `simd_minimizers`.

## 3.2.1
- Add `Seq::read_{revcomp}_kmer_u128` with more tests
- Fix bug in `revcomp_u128`

## 3.2.0: yanked
- Add `Seq::as_u128` and `Seq::revcomp_as_u128`
- `revcomp_u128` had a bug; fixed in 3.2.1

## 3.1
- `PackedSeqVec::{from,push}_ascii` now silently convert non-ACGT characters to
  values in `0..4`, instead of inconsistently only panicking for non-ACGT near
  the sequence start and end.

## 3.0
- Speed up `PackedSeqVec::push_ascii()` on ARM
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
