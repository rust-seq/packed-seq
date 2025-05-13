# Changelog

## 2.0
Change from tuples of (simd iterator, scalar tail iterator) to returning only a
simd iterator and `padding: usize` that indicates the number of padded elements
at the back that should be ignored.

## 1.0
Initial release.
