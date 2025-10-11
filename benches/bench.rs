#![feature(test)]

use std::hint::black_box;

use packed_seq::*;
use rand::random_range;
use test::Bencher;

extern crate test;

// #[bench]
// fn read_kmer_bench(b: &mut Bencher) {
//     eprintln!("\nBench PackedSeq::read_kmer");
//     for len in [100, 150, 200, 1000, 1_000_000] {
//         let x = vec![b'A'; len];
//         let packed = PackedSeqVec::from_ascii(&x);
//         let mut out = vec![];
//         for k in 1..=32 {
//             let start = std::time::Instant::now();
//             let mut sum = 0;
//             for i in 0..=(x.len() - k) {
//                 let word = packed.read_kmer(k, i);
//                 sum += word;
//             }
//             std::hint::black_box(sum);
//             let d = start.elapsed();
//             let ns_each = d.as_nanos() as f32 / (x.len() - k + 1) as f32;
//             out.push(ns_each);
//         }
//         let mut s = String::new();
//         s += &format!("Len {len:>7} => ");
//         for x in out {
//             s += &format!("{x:>4.2} ");
//         }
//         s += &format!("ns/kmer");
//         eprintln!("{s}");
//     }
// }

// #[bench]
// fn push_ascii_1bit_bench() {
//     eprintln!("\nBench BitSeqVec::from_ascii");

//     let mut packed = BitSeqVec::default();

//     for len in [100, 150, 200, 1000, 1_000_000] {
//         // 1Gbp input.
//         let rep = 1_000_000_000 / len;
//         let mut ascii = AsciiSeqVec::random(len);
//         // set 1% to N
//         for _ in 0..len / 100 {
//             ascii.seq[random_range(0..len)] = b'N';
//         }

//         let start = std::time::Instant::now();
//         for _ in 0..rep {
//             packed.push_ascii(&ascii.seq);
//             core::hint::black_box(&packed);
//             packed.clear();
//         }
//         eprintln!(
//             "Len {len:>7} => {:.03} Gbp/s",
//             start.elapsed().as_secs_f64().recip()
//         );
//     }
// }

// #[bench]
// fn push_ascii_2bit_bench() {
//     eprintln!("\nBench PackedSeqVec::from_ascii");
//     let mut packed = PackedSeqVec::default();

//     for len in [100, 150, 200, 1000, 1_000_000] {
//         // 1Gbp input.
//         let rep = 1_000_000_000 / len;
//         let mut ascii = AsciiSeqVec::random(len);
//         // set 1% to N
//         for _ in 0..len / 100 {
//             ascii.seq[random_range(0..len)] = b'N';
//         }

//         let start = std::time::Instant::now();
//         for _ in 0..rep {
//             packed.push_ascii(&ascii.seq);
//             core::hint::black_box(&packed);
//             packed.clear();
//         }
//         eprintln!(
//             "Len {len:>7} => {:.03} Gbp/s",
//             start.elapsed().as_secs_f64().recip()
//         );
//     }
// }

fn par_iter_bp(b: &mut Bencher, len: usize) {
    let len = black_box(len);
    // 1Gbp input.
    let rep = 1_000_000 / len;
    let seq = black_box(PackedSeqVec::random(len));

    b.iter(|| {
        for _ in 0..rep {
            let PaddedIt { it, .. } = seq.as_slice().par_iter_bp(1);
            it.for_each(
                #[inline(always)]
                |y| {
                    black_box(&y);
                },
            );
        }
    });
}

#[bench]
fn par_iter_bp_150(b: &mut Bencher) {
    par_iter_bp(b, 150);
}
#[bench]
fn par_iter_bp_1K(b: &mut Bencher) {
    par_iter_bp(b, 1000);
}
#[bench]
fn par_iter_bp_1M(b: &mut Bencher) {
    par_iter_bp(b, 1_000_000);
}

// #[bench]
// fn par_iter_bp_buf_bench() {
//     eprintln!("\nBench PackedSeq::par_iter_bp_buf");

//     let mut buf = [S::ZERO; 8];

//     for len in [100, 150, 200, 1000, 1_000_000] {
//         // 1Gbp input.
//         let rep = 1_000_000_000 / len;
//         let seq = PackedSeqVec::random(len);

//         let start = std::time::Instant::now();
//         for _ in 0..rep {
//             let PaddedIt { it, .. } = seq.as_slice().par_iter_bp_with_buf(1, &mut buf);
//             it.for_each(
//                 #[inline(always)]
//                 |y| {
//                     core::hint::black_box(&y);
//                 },
//             );
//         }
//         eprintln!(
//             "Len {len:>7} => {:.03} Gbp/s",
//             start.elapsed().as_secs_f64().recip()
//         );
//     }
// }

// #[bench]
// fn par_iter_bp_delayed_bench() {
//     eprintln!("\nBench PackedSeq::par_iter_bp_delayed");
//     for len in [100, 1000, 1_000_000] {
//         // 1Gbp input.
//         let rep = 4_000_000_000 / len;
//         let seq = PackedSeqVec::random(len);

//         let start = std::time::Instant::now();
//         let ss = tick_counter::start();
//         for _ in 0..rep {
//             let PaddedIt { it, .. } = seq.as_slice().par_iter_bp_delayed(1, Delay(27));
//             it.for_each(
//                 #[inline(always)]
//                 |y| {
//                     core::hint::black_box(&y);
//                 },
//             );
//         }
//         let ee = tick_counter::stop() - ss;
//         eprintln!(
//             "Len {len:>7} => {:.03} Gbp/s",
//             start.elapsed().as_secs_f64().recip()
//         );
//         eprintln!("ee {:?}", ee);
//     }
// }

// #[bench]
// fn par_iter_kmer_ambiguity_bench() {
//     eprintln!("\nBench PackedSeq::par_iter_kmer_ambiguity");
//     let k = 31;
//     for len in [100, 150, 200, 1000, 1_000_000] {
//         // 1Gbp input.
//         let rep = 1_000_000_000 / len;
//         let seq = BitSeqVec::random(len, 0.01);

//         let start = std::time::Instant::now();
//         for _ in 0..rep {
//             let PaddedIt { it, .. } = seq.as_slice().par_iter_kmer_ambiguity(k, k - 1, 0);
//             it.for_each(
//                 #[inline(always)]
//                 |y| {
//                     core::hint::black_box(&y);
//                 },
//             );
//         }
//         eprintln!(
//             "Len {len:>7} => {:.03} Gbp/s",
//             start.elapsed().as_secs_f64().recip()
//         );
//     }
// }
