# packed-seq

[![crates.io](https://img.shields.io/crates/v/packed-seq)](https://crates.io/crates/packed-seq)
[![docs](https://img.shields.io/docsrs/packed-seq)](https://docs.rs/packed-seq)

A library for constructing and iterating packed `PackedSeq` DNA sequences that
handles the encoding, decoding, and complements of packed bases.

ASCII `ACTG` representations of DNA and general ASCII text are also supported via the `Seq` trait.

Fast SIMD-based iteration over sequences is supported by splitting the sequence
into 8 (slightly overlapping) chunks and iterating those in parallel in a
memory-efficient way.

The underlying algorithm is described in the following
[preprint](https://doi.org/10.1101/2025.01.27.634998) on
[`simd-minimizers`](https://github.com/rust-seq/simd-minimizers), for which this
crate was developed:

-   SimdMinimizers: Computing random minimizers, fast.
    Ragnar Groot Koerkamp, Igor Martayan
    bioRxiv 2025.01.27 [doi.org/10.1101/2025.01.27.634998](https://doi.org/10.1101/2025.01.27.634998)


## Requirements

This library supports AVX2 and NEON instruction sets.
Make sure to set `RUSTFLAGS="-C target-cpu=native"` when compiling to use the instruction sets available on your architecture.

    RUSTFLAGS="-C target-cpu=native" cargo run --release


## Usage example

Full documentation can be found on [docs.rs](https://docs.rs/packed-seq).

```rust
use packed_seq::{SeqVec, Seq, AsciiSeqVec, PackedSeqVec, pack_char};
// Plain ASCII sequence.
let seq = b"ACTGCAGCGCATATGTAGT";
// ASCII DNA sequence.
let ascii_seq = AsciiSeqVec::from_ascii(seq);
// Packed DNA sequence.
let packed_seq = PackedSeqVec::from_ascii(seq);
assert_eq!(ascii_seq.len(), packed_seq.len());
// Iterate the ASCII characters.
let characters: Vec<u8> = seq.iter_bp().collect();
assert_eq!(characters, seq);

// Iterate the bases with 0..4 values.
let bases: Vec<u8> = seq.iter().copied().map(pack_char).collect();
assert_eq!(bases, vec![0,1,2,3,1,0,3,1,3,1,0,2,0,2,3,2,0,3,2]);
let ascii_bases: Vec<u8> = ascii_seq.as_slice().iter_bp().collect();
assert_eq!(ascii_bases, bases);
let packed_bases: Vec<u8> = ascii_seq.as_slice().iter_bp().collect();
assert_eq!(packed_bases, bases);
```
