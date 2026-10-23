use wide::i8x32;

use crate::packed_seq::read_slice_32;

use super::*;

#[derive(Clone, Copy, Debug, MemSize, MemDbg)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
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
    /// Given a bunch of `PackedNSeqVec` chunks, assemble them into a single `PackedNSeqVec` in parallel.
    #[cfg(feature = "file_io")]
    fn assemble_parallel_chunks(chunks: Vec<Self>, num_threads: usize) -> Self {
        let len = chunks
            .iter()
            .map(|chunk| chunk.seq.len().next_multiple_of(8))
            .sum();
        let mut output = Self {
            seq: PackedSeqVec::from_raw_parts(vec![0; len / 4], len),
            ambiguous: BitSeqVec::with_len(len),
        };
        let mut seq_out = &mut output.seq.seq[..len / 4];
        let mut mask_out = &mut output.ambiguous.seq[..len / 8];
        let mut jobs = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let aligned_len = chunk.seq.len().next_multiple_of(8);
            let (seq_dest, rest) = seq_out.split_at_mut(aligned_len / 4);
            seq_out = rest;
            let (mask_dest, rest) = mask_out.split_at_mut(aligned_len / 8);
            mask_out = rest;
            jobs.push((chunk, seq_dest, mask_dest));
        }
        let jobs = std::sync::Mutex::new(jobs.into_iter());
        std::thread::scope(|scope| {
            for _ in 0..num_threads.max(1) {
                let jobs = &jobs;
                scope.spawn(move || {
                    loop {
                        let job = jobs.lock().unwrap().next();
                        let Some((chunk, seq_dest, mask_dest)) = job else {
                            break;
                        };
                        let chunk_len = chunk.seq.len();
                        let seq_bytes = chunk_len.div_ceil(4);
                        let mask_bytes = chunk_len.div_ceil(8);
                        seq_dest[..seq_bytes].copy_from_slice(&chunk.seq.seq[..seq_bytes]);
                        mask_dest[..mask_bytes].copy_from_slice(&chunk.ambiguous.seq[..mask_bytes]);
                        // Fill alignment padding with N in both representations.
                        seq_dest[seq_bytes..].fill(u8::MAX);
                        if chunk_len % 4 != 0 {
                            seq_dest[seq_bytes - 1] |= u8::MAX << (2 * (chunk_len % 4));
                        }
                        if chunk_len % 8 != 0 {
                            mask_dest[mask_bytes - 1] |= u8::MAX << (chunk_len % 8);
                        }
                    }
                });
            }
        });
        output
    }

    pub fn clear(&mut self) {
        self.seq.clear();
        self.ambiguous.clear();
    }

    #[cfg(feature = "rand")]
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

    /// Create a mask that is 1 for all non-ACGT bases and for all low-quality bases with quality `<threshold`.
    pub fn from_ascii_and_quality(seq: &[u8], quality: &[u8], threshold: usize) -> Self {
        assert_eq!(seq.len(), quality.len());

        let mut ambiguous = BitSeqVec::from_ascii(seq);

        // Low-quality bases are also ambiguous.
        {
            let t = (b'!' + threshold as u8) as i8;
            let t_simd = i8x32::splat(t);
            let ambiguous = ambiguous.seq.as_chunks_mut::<4>().0;
            for i in (0..quality.len()).step_by(32) {
                let chunk = i8x32::from(unsafe {
                    std::mem::transmute::<_, i8x32>(read_slice_32(quality, i))
                });

                let mut mask = chunk.simd_lt(t_simd).to_bitmask() as u32;
                let remaining = quality.len() - i;
                if remaining < 32 {
                    mask &= (1u32 << remaining) - 1;
                }
                let ambi = &mut ambiguous[i / 32];
                *ambi = (u32::from_ne_bytes(*ambi) | mask).to_ne_bytes();
            }
        }

        Self {
            seq: PackedSeqVec::from_ascii(seq),
            ambiguous,
        }
    }

    /// Create a mask that is 1 for all non-ACGT bases and for all low-quality bases with quality `<threshold`.
    pub fn push_from_ascii_and_quality(&mut self, seq: &[u8], quality: &[u8], min_qual: u8) {
        let r = self.seq.push_ascii(seq);
        let r2 = self.ambiguous.push_ascii(seq);
        assert_eq!(r, r2);

        assert_eq!(seq.len(), quality.len());

        // Low-quality bases are also ambiguous.
        let t = b'!' + min_qual;
        let t_simd = i8x32::splat(t as i8);

        let shift = r2.start % 8;
        let ambiguous = &mut self.ambiguous.seq[r2.start / 8..];
        for i in (0..quality.len()).step_by(32) {
            let chunk =
                i8x32::from(unsafe { std::mem::transmute::<_, i8x32>(read_slice_32(quality, i)) });

            let mut mask = chunk.simd_lt(t_simd).to_bitmask() as u32;
            let remaining = quality.len() - i;
            if remaining < 32 {
                mask &= (1u32 << remaining) - 1;
            }
            let mask = (mask as u64) << shift;
            let offset = i / 8;
            let word = u32::from_le_bytes(ambiguous[offset..offset + 4].try_into().unwrap());
            ambiguous[offset..offset + 4].copy_from_slice(&(word | mask as u32).to_le_bytes());
            ambiguous[offset + 4] |= (mask >> 32) as u8;
        }
    }

    /// Read a fasta/fastq file at the given path into a `PackedNSeq`, separating records with `N`.
    #[cfg(feature = "file_io")]
    pub fn from_fastx(path: &std::path::Path) -> Self {
        use helicase::{HelicaseParser, input::FromFile};

        const CONFIG: helicase::Config = helicase::ParserOptions::default()
            .ignore_headers()
            .dna_string()
            .config();
        let mut ascii = Vec::with_capacity(16000);

        let mut seq = Self::default();

        let mut reader = helicase::FastxParser::<CONFIG>::from_file(path).unwrap();
        while let Some(event) = reader.next() {
            event.expect("Invalid FASTA/Q record");
            ascii.extend_from_slice(reader.get_dna_string());
            ascii.push(b'N');

            // Convert from ascii dna to packed dna+mask in batches of 16kbp.
            if ascii.len() > 16000 {
                seq.push_ascii(&ascii);
                ascii.clear();
            }
        }

        if ascii.len() > 0 {
            seq.push_ascii(&ascii);
        }

        seq
    }

    /// Read FASTA/FASTQ using Helicase's parallel reader, separating records with `N`.
    ///
    /// Does *not* preserve the order or records!
    ///
    /// Chunk boundaries may contain extra `N`s for byte alignment.
    #[cfg(feature = "file_io")]
    pub fn from_fastx_parallel(path: &std::path::Path, num_threads: usize) -> Self {
        use helicase::HelicaseParser;

        const CONFIG: helicase::Config = helicase::ParserOptions::default()
            .ignore_headers()
            .ignore_quality()
            .dna_string()
            .config();
        let chunks = std::sync::Mutex::new(Vec::new());
        helicase::ParallelFastxParser::<CONFIG>::from_file(path)
            .unwrap()
            .threads(num_threads)
            .chunks(num_threads.max(1))
            .for_each(|mut parser| {
                let mut seq = Self::default();
                let mut ascii = Vec::with_capacity(16000);
                while let Some(event) = parser.next() {
                    event?;
                    ascii.extend_from_slice(parser.get_dna_string());
                    ascii.push(b'N');
                    if ascii.len() >= 16000 {
                        seq.push_ascii(&ascii);
                        ascii.clear();
                    }
                }
                if !ascii.is_empty() {
                    seq.push_ascii(&ascii);
                }
                chunks.lock().unwrap().push(seq);
                Ok(())
            })
            .unwrap();
        let chunks = chunks.into_inner().unwrap();
        Self::assemble_parallel_chunks(chunks, num_threads)
    }

    /// Read a fasta/fastq file at the given path into a `PackedNSeq`, separating records with `N`.
    /// Low-quality bases are masked out.
    #[cfg(feature = "file_io")]
    pub fn from_fastq_with_quality(path: &std::path::Path, min_qual: u8) -> Self {
        use helicase::{HelicaseParser, input::FromFile};

        const CONFIG: helicase::Config = helicase::ParserOptions::default()
            .ignore_headers()
            .dna_string()
            .compute_quality()
            .config();
        let mut ascii = vec![];
        let mut qual = vec![];

        let mut seq = Self::default();

        let mut reader = helicase::FastxParser::<CONFIG>::from_file(path).unwrap();
        while let Some(event) = reader.next() {
            event.expect("Invalid FASTA/Q record");
            ascii.extend_from_slice(reader.get_dna_string());
            qual.extend_from_slice(reader.get_quality().expect("Expected FASTQ quality scores"));
            ascii.push(b'N');
            qual.push(0);

            // Convert from ascii dna+qual to packed dna+mask in batches of 16kbp.
            if ascii.len() > 16000 {
                seq.push_from_ascii_and_quality(&ascii, &qual, min_qual);
                ascii.clear();
                qual.clear();
            }
        }

        if ascii.len() > 0 {
            seq.push_from_ascii_and_quality(&ascii, &qual, min_qual);
        }

        seq
    }

    /// Read FASTQ using Helicase's parallel reader, separating records with `N`.
    ///
    /// Masks non-ACGT bases and qualities below `min_qual`.
    /// Does *not* preserve the order or records!
    /// Chunk boundaries may contain extra `N`s for byte alignment.
    #[cfg(feature = "file_io")]
    pub fn from_fastq_with_quality_parallel(
        path: &std::path::Path,
        min_qual: u8,
        num_threads: usize,
    ) -> Self {
        use helicase::HelicaseParser;

        const CONFIG: helicase::Config = helicase::ParserOptions::default()
            .ignore_headers()
            .dna_string()
            .compute_quality()
            .config();
        let chunks = std::sync::Mutex::new(Vec::new());
        helicase::ParallelFastxParser::<CONFIG>::from_file(path)
            .unwrap()
            .threads(num_threads)
            .chunks(num_threads.max(1))
            .for_each(|mut parser| {
                let mut seq = Self::default();
                let mut ascii = Vec::with_capacity(16000);
                let mut quality = Vec::with_capacity(16000);
                while let Some(event) = parser.next() {
                    event?;
                    ascii.extend_from_slice(parser.get_dna_string());
                    ascii.push(b'N');
                    quality.extend_from_slice(
                        parser.get_quality().expect("Expected FASTQ quality scores"),
                    );
                    quality.push(0);
                    if ascii.len() >= 16000 {
                        seq.push_from_ascii_and_quality(&ascii, &quality, min_qual);
                        ascii.clear();
                        quality.clear();
                    }
                }
                if !ascii.is_empty() {
                    seq.push_from_ascii_and_quality(&ascii, &quality, min_qual);
                }
                chunks.lock().unwrap().push(seq);
                Ok(())
            })
            .unwrap();
        let chunks = chunks.into_inner().unwrap();
        Self::assemble_parallel_chunks(chunks, num_threads)
    }
}
