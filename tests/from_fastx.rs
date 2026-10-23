#![cfg(feature = "file_io")]

use packed_seq::{PackedSeqVec, Seq, SeqVec, pack_char};
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
    check_output(PackedSeqVec::from_fastx(&input.0), records);
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
