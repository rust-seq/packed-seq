# Changelog

## Git
- Support `par_iter_bp` for sequences that do not start at byte offsets.
- Add `Seq::revcomp_word()` and `Seq::to_word_revcomp()` for reverse
  complementing kmers.
- Make `random` dependency optional but enabled by default (so it can be
  disabled for wasm backends).

## 2.0
- Change from tuples of (simd iterator, scalar tail iterator) to returning only a
  simd iterator and `padding: usize` that indicates the number of padded elements
  at the back that should be ignored.

## 1.0
- Initial release.
