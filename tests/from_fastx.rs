#![cfg(feature = "file_io")]

use packed_seq::{PackedNSeqVec, PackedSeqVec, Seq, SeqVec, pack_char};
use std::{
    ops::Range,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Input(PathBuf);

impl Input {
    fn new(extension: &str, contents: &[u8]) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "packed-seq-from-fastx-{}-{}.{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed),
            extension,
        ));
        std::fs::write(&path, contents).unwrap();
        Self(path)
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn check(input: &Input, records: &[&[u8]], expected_ranges: &[Range<usize>]) {
    let sequential = PackedSeqVec::from_fastx(&input.0);
    assert_eq!(sequential.1, expected_ranges);
    check_output(sequential, records);
}

fn check_output((packed, ranges): (PackedSeqVec, Vec<Range<usize>>), records: &[&[u8]]) {
    assert_eq!(ranges.len(), records.len());
    assert_eq!(ranges.first().map(|r| r.start).unwrap_or(0), 0);
    assert_eq!(ranges.last().map(|r| r.end).unwrap_or(0), packed.len());
    let mut previous_end = 0;
    for (i, (range, record)) in ranges.iter().zip(records).enumerate() {
        assert_eq!(range.len(), record.len(), "record {i}");
        assert!(range.start >= previous_end);
        assert!(range.start - previous_end < 4);
        if range.start != previous_end {
            assert_eq!(range.start % 4, 0);
        }
        previous_end = range.end;
        assert_eq!(
            packed
                .as_slice()
                .slice(range.clone())
                .iter_bp()
                .collect::<Vec<_>>(),
            record.iter().copied().map(pack_char).collect::<Vec<_>>(),
            "record {i} at {range:?}",
        );
    }
}

#[test]
fn fasta_ranges_exclude_headers_and_line_endings() {
    let input = Input::new(
        "fa",
        b">first description\r\nA\r\n>second\r\nCG\r\nTAC\r\n>third\r\nTG\r\n>last\r\nGACTA",
    );
    check(
        &input,
        &[b"A", b"CGTAC", b"TG", b"GACTA"],
        &[0..1, 1..6, 6..8, 8..13],
    );
}

#[test]
fn fastq_ranges_exclude_headers_and_quality() {
    let input = Input::new(
        "fq",
        b"@first\nTGC\n+\n!!!\n@second description\nA\n+\nI\n@third\nCGTAC\n+\n@+!II\n",
    );
    check(&input, &[b"TGC", b"A", b"CGTAC"], &[0..3, 3..4, 4..9]);
}

#[test]
fn fasta_ranges_preserve_many_records_across_input_buffers() {
    check_many_records(false);
}

#[test]
fn fastq_ranges_preserve_many_records_across_input_buffers() {
    check_many_records(true);
}

#[test]
fn packed_n_fastx_preserves_ambiguity_and_record_separators() {
    for (extension, contents) in [
        ("fa", &b">first\r\nAcN\r\ngTR\r\n>second\r\nNYta"[..]),
        (
            "fq",
            &b"@first\nAcNgTR\n+\nIIIIII\n@second\nNYta\n+\nIIII\n"[..],
        ),
    ] {
        let input = Input::new(extension, contents);
        let actual = PackedNSeqVec::from_fastx(&input.0);
        let expected = PackedNSeqVec::from_ascii(b"AcNgTRNNYtaN");
        assert_eq!(actual.seq.len(), expected.seq.len());
        assert_eq!(actual.seq.into_raw(), expected.seq.into_raw());
        assert_eq!(actual.ambiguous.into_raw(), expected.ambiguous.into_raw());
    }
}

#[test]
fn appended_quality_masks_handle_all_bit_offsets() {
    for offset in 0..8 {
        for len in 0..70 {
            let mut packed = PackedNSeqVec::from_ascii(&b"ACGTACGT"[..offset]);
            let bases: Vec<_> = (0..len).map(|i| b"ACGTN"[i % 5]).collect();
            let quality: Vec<_> = (0..len).map(|i| b"456!I"[i % 5]).collect();
            packed.push_from_ascii_and_quality(&bases, &quality, 20);
            // A later append must not inherit any mask bits from SIMD padding.
            packed.push_from_ascii_and_quality(b"ACGT", b"IIII", 20);
            let expected: Vec<_> = std::iter::repeat_n(0, offset)
                .chain(
                    (0..len).map(|i| (!b"ACGTacgt".contains(&bases[i]) || quality[i] < b'5') as u8),
                )
                .chain([0; 4])
                .collect();
            assert_eq!(
                packed.ambiguous.as_slice().iter_bp().collect::<Vec<_>>(),
                expected,
                "offset {offset}, length {len}"
            );
        }
    }
}

#[test]
fn quality_mask_is_one_below_threshold() {
    let bases = b"ACGTN".repeat(7);
    // Phred 19, 20, 21, 0, 40: only 19 and 0 are below 20.
    let quality = b"456!I".repeat(7);
    let expected = [1, 0, 0, 1, 1].repeat(7);
    let constructed = PackedNSeqVec::from_ascii_and_quality(&bases, &quality, 20);
    let mut appended = PackedNSeqVec::default();
    appended.push_from_ascii_and_quality(&bases, &quality, 20);
    for packed in [constructed, appended] {
        assert_eq!(
            packed.ambiguous.as_slice().iter_bp().collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn packed_n_fastq_masks_low_quality_across_batches() {
    let mut fastq = Vec::new();
    let mut expected_ascii = Vec::new();
    let mut expected_mask = Vec::new();
    for i in 0..4000 {
        let bases = &b"AcGTNRYtacg"[..1 + i % 11];
        fastq.extend_from_slice(format!("@record-{i}\n").as_bytes());
        fastq.extend_from_slice(bases);
        fastq.extend_from_slice(b"\n+\n");
        for (j, &base) in bases.iter().enumerate() {
            // Phred 19, 20 and 21 straddle the threshold of 20.
            let quality = b'!' + 19 + (j % 3) as u8;
            fastq.push(quality);
            expected_ascii.push(base);
            expected_mask.push((!b"ACGTacgt".contains(&base) || quality < b'!' + 20) as u8);
        }
        fastq.push(b'\n');
        expected_ascii.push(b'N');
        expected_mask.push(1);
    }
    let input = Input::new("fq", &fastq);
    let actual = PackedNSeqVec::from_fastq_with_quality(&input.0, 20);
    assert_eq!(actual.seq.len(), expected_ascii.len());
    assert_eq!(
        actual.seq.into_raw(),
        PackedSeqVec::from_ascii(&expected_ascii).into_raw()
    );
    assert_eq!(actual.ambiguous.len(), expected_mask.len());
    for (i, (actual, expected)) in actual
        .ambiguous
        .as_slice()
        .iter_bp()
        .zip(expected_mask)
        .enumerate()
    {
        assert_eq!(actual, expected, "ambiguity at base {i}");
    }
}

fn check_many_records(fastq: bool) {
    let records: Vec<Vec<u8>> = (0..4096)
        .map(|i| (0..(1 + i % 257)).map(|j| b"ACGT"[(i + j) % 4]).collect())
        .collect();
    let mut fasta = Vec::new();
    let mut ranges = Vec::new();
    let mut offset = 0;
    for (i, record) in records.iter().enumerate() {
        let marker = if fastq { '@' } else { '>' };
        fasta.extend_from_slice(format!("{marker}record-{i}\n").as_bytes());
        fasta.extend_from_slice(record);
        fasta.push(b'\n');
        if fastq {
            fasta.extend_from_slice(b"+\n");
            fasta.extend(std::iter::repeat_n(b'I', record.len()));
            fasta.push(b'\n');
        }
        ranges.push(offset..offset + record.len());
        offset += record.len();
    }
    let input = Input::new(if fastq { "fq" } else { "fa" }, &fasta);
    let records: Vec<&[u8]> = records.iter().map(Vec::as_slice).collect();
    check(&input, &records, &ranges);
}

#[test]
fn multithreaded_fasta_collects_all_records() {
    check_multithreaded(false);
}

#[test]
fn multithreaded_fastq_collects_all_records_and_quality_masks() {
    check_multithreaded(true);
}

fn check_multithreaded(fastq: bool) {
    let mut contents = Vec::new();
    let mut expected = Vec::new();
    let mut expected_quality = Vec::new();
    for i in 0..2048usize {
        // A unique prefix detects missing or duplicated records, independent of order.
        let mut bases: Vec<_> = (0..8).map(|j| b"ACTG"[(i >> (2 * j)) & 3]).collect();
        bases.extend((0..(65 + i % 13)).map(|j| b"ACGT"[(i + j) % 4]));
        contents
            .extend_from_slice(format!("{}record-{i}\n", if fastq { '@' } else { '>' }).as_bytes());
        contents.extend_from_slice(&bases);
        contents.push(b'\n');
        // Keep G high-quality so (packed G, ambiguous) uniquely denotes separator N.
        let quality: Vec<_> = bases
            .iter()
            .enumerate()
            .map(|(j, &base)| {
                if j % 3 == 0 && base != b'G' {
                    b'4'
                } else {
                    b'5'
                }
            })
            .collect();
        if fastq {
            contents.extend_from_slice(b"+\n");
            contents.extend_from_slice(&quality);
            contents.push(b'\n');
        }
        expected.push(bases.iter().copied().map(pack_char).collect::<Vec<_>>());
        expected_quality.push(
            bases
                .iter()
                .zip(&quality)
                .map(|(&base, &q)| (pack_char(base), (q < b'5') as u8))
                .collect::<Vec<_>>(),
        );
    }
    expected.sort();
    expected_quality.sort();
    let input = Input::new(if fastq { "fq" } else { "fa" }, &contents);
    for threads in [2, 4] {
        let (packed, ranges) = PackedSeqVec::from_fastx_parallel(&input.0, threads);
        let mut previous_end = 0;
        let mut records = Vec::new();
        for range in ranges {
            assert!(range.start >= previous_end);
            assert!(range.start - previous_end < 4);
            assert!(range.end <= packed.len());
            previous_end = range.end;
            records.push(packed.slice(range).iter_bp().collect::<Vec<_>>());
        }
        assert!(packed.len() - previous_end < 4);
        records.sort();
        assert_eq!(records, expected);

        let plain = PackedNSeqVec::from_fastx_parallel(&input.0, threads);
        let records = split_parallel_n_records(&plain);
        let expected_plain: Vec<_> = expected
            .iter()
            .map(|record| record.iter().map(|&base| (base, 0)).collect::<Vec<_>>())
            .collect();
        assert_eq!(records, expected_plain);
        if fastq {
            let masked = PackedNSeqVec::from_fastq_with_quality_parallel(&input.0, 20, threads);
            assert_eq!(split_parallel_n_records(&masked), expected_quality);
        }
    }
}

fn split_parallel_n_records(packed: &PackedNSeqVec) -> Vec<Vec<(u8, u8)>> {
    assert_eq!(packed.seq.len(), packed.ambiguous.len());
    let bases: Vec<_> = packed
        .seq
        .as_slice()
        .iter_bp()
        .zip(packed.ambiguous.as_slice().iter_bp())
        .collect();
    let separator = (pack_char(b'G'), 1);
    assert_eq!(bases.last(), Some(&separator));
    let mut records: Vec<_> = bases
        .split(|&base| base == separator)
        .filter(|record| !record.is_empty())
        .map(<[_]>::to_vec)
        .collect();
    records.sort();
    records
}
