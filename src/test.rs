use rand::random_range;

use super::*;

fn pack_naive(seq: &[u8]) -> (Vec<u8>, usize) {
    let mut packed_byte = 0;
    let mut packed_len = 0;
    let mut packed = vec![];
    for &base in seq {
        packed_byte |= pack_char(base) << ((packed_len % 4) * 2);
        packed_len += 1;
        if packed_len % 4 == 0 {
            packed.push(packed_byte);
            packed_byte = 0;
        }
    }
    if packed_len % 4 != 0 {
        packed.push(packed_byte);
    }
    (packed, packed_len)
}

#[cfg(test)]
use rand::Rng;

#[test]
fn pack() {
    for n in 0..=128 {
        let mut rng = rand::rng();
        let seq: Vec<_> = (0..n)
            .map(|_| b"ACGTacgt"[rng.random::<u8>() as usize % 8])
            .collect();
        let (packed_1, len1) = pack_naive(&seq);
        let packed_2 = PackedSeqVec::from_ascii(&seq);
        assert_eq!(len1, packed_2.len);
        assert_eq!(packed_1, packed_2.seq);
    }
}

#[test]
fn pack_via_ascii() {
    for n in 0..=128 {
        let mut rng = rand::rng();
        let seq: Vec<_> = (0..n)
            .map(|_| b"ACGTacgt"[rng.random::<u8>() as usize % 8])
            .collect();
        let ascii_seq = AsciiSeqVec::from_ascii(&seq);
        let (packed_1, len1) = pack_naive(&seq);
        let packed_2 = PackedSeqVec::from_ascii(&ascii_seq.seq);
        assert_eq!(len1, packed_2.len);
        assert_eq!(packed_1, packed_2.seq);
    }
}

#[test]
fn pack_word() {
    let packed = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGT");
    let slice = packed.slice(0..1);
    assert_eq!(slice.to_word(), 0b00000000);
    let slice = packed.slice(0..2);
    assert_eq!(slice.to_word(), 0b00000100);
    let slice = packed.slice(0..3);
    assert_eq!(slice.to_word(), 0b00110100);
    let slice = packed.slice(0..4);
    assert_eq!(slice.to_word(), 0b10110100);
    let slice = packed.slice(0..8);
    assert_eq!(slice.to_word(), 0b1011010010110100);
    let slice = packed.slice(0..16);
    assert_eq!(slice.to_word(), 0b10110100101101001011010010110100);
    let slice = packed.slice(0..28);
    assert_eq!(
        slice.to_word(),
        0b10110100101101001011010010110100101101001011010010110100
    );
}

#[test]
fn packed_ord() {
    let ascii_seq = b"ACGTACGTACGTACGTACGTACGTACGT";
    let packed_seq = ascii_seq
        .iter()
        .map(|c| match c {
            // Swap G and T values since they are encoded in opposite order.
            b'G' => b'T',
            b'T' => b'G',
            c => *c,
        })
        .collect::<Vec<_>>();
    let ascii = AsciiSeqVec::from_ascii(ascii_seq);
    let packed = PackedSeqVec::from_ascii(&packed_seq);
    for i in 0..ascii.len() {
        for j in i..ascii.len() {
            for k in 0..ascii.len() {
                for l in k..ascii.len() {
                    let a0 = ascii.as_slice().slice(i..j);
                    let a1 = ascii.as_slice().slice(k..l);
                    let b0 = packed.as_slice().slice(i..j);
                    let b1 = packed.as_slice().slice(k..l);
                    assert_eq!(
                        a0.cmp(&a1),
                        b0.cmp(&b1),
                        "Failed at ({}, {})={:?}, ({}, {})={:?}",
                        i,
                        j,
                        &ascii_seq[i..j],
                        k,
                        l,
                        &ascii_seq[k..l]
                    );
                }
            }
        }
    }
}

#[test]
fn push_ascii_unaligned() {
    let seq = b"TCGGCTCGTTCC";
    let mut packed = PackedSeqVec::default();
    let range = packed.push_ascii(&seq[..3]);
    assert_eq!(range, (0..3));
    let range = packed.push_ascii(&seq[3..9]);
    assert_eq!(range, (3..9));
    let range = packed.push_ascii(&seq[9..]);
    assert_eq!(range, (9..12));
    assert_eq!(packed.len, seq.len());
    let slice = packed.as_slice();
    for (i, &c) in seq.iter().enumerate() {
        assert_eq!(slice.get_ascii(i), c);
    }
    let packed2 = PackedSeqVec::from_ascii(seq);
    let expected = packed2.as_slice();
    assert!(slice.eq(&expected));
}

#[test]
fn push_ascii_random() {
    for _ in 0..20 {
        let seq = AsciiSeqVec::random(110).into_raw();
        let mut packed = PackedSeqVec::default();
        let mut rng = rand::rng();
        while packed.len < 100 {
            let push_len = rng.random_range(1..=8);
            let range = packed.len..(packed.len + push_len);
            let range2 = packed.push_ascii(&seq[range.clone()]);
            assert_eq!(range, range2);
        }
        let slice = packed.as_slice();
        for (i, &c) in seq.iter().take(100).enumerate() {
            assert_eq!(slice.get_ascii(i), c);
        }
        let packed2 = PackedSeqVec::from_ascii(&seq[..packed.len]);
        let expected = packed2.as_slice();
        assert!(slice.eq(&expected));
    }
}

#[test]
fn iter_bp() {
    let seq = b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT";
    for len in 0..=seq.len() {
        let ascii = AsciiSeqVec::from_ascii(&seq[..len]);
        let packed = PackedSeqVec::from_ascii(&seq[..len]);
        eprintln!("ascii {ascii:?}");
        eprintln!("packed {packed:?}");
        let ascii = ascii.as_slice().iter_bp().collect::<Vec<_>>();
        let packed = packed.as_slice().iter_bp().collect::<Vec<_>>();
        assert_eq!(ascii, packed);
    }
    let ascii = AsciiSeqVec::from_ascii(seq);
    let packed = PackedSeqVec::from_ascii(seq);
    let len = seq.len();
    for offset in 0..len {
        let ascii = ascii.slice(offset..len).iter_bp().collect::<Vec<_>>();
        let packed = packed.slice(offset..len).iter_bp().collect::<Vec<_>>();
        assert_eq!(ascii, packed);
    }
}

// #[test]
// fn par_iter_bp() {
//     let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
//     let (head, padding) = s.as_slice().par_iter_bp(1);
//     let head = head.collect::<Vec<_>>();
//     fn f(x: &[u8; 16]) -> S {
//         let x = x.map(|x| pack_char(x) as u32);
//         S::from(x)
//     }
//     assert_eq!(padding, 8 * 8 - s.len());
//     assert_eq!(
//         head,
//         vec![
//             f(b"AGCAAAAA"),
//             f(b"CGCACAAA"),
//             f(b"GTGAGAAA"),
//             f(b"TTGATAAA"),
//             f(b"AAGAAAAA"),
//             f(b"AATAAAAA"),
//             f(b"CATAAAAA"),
//             f(b"CCTAAAAA"),
//         ]
//     );
// }

// #[test]
// fn par_iter_bp_delayed0() {
//     let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
//     let (head, padding) = s.as_slice().par_iter_bp_delayed(1, 0);
//     let head = head.collect::<Vec<_>>();
//     fn f(x: &[u8; 16], y: &[u8; 16]) -> (S, S) {
//         let x = x.map(|x| pack_char(x) as u32);
//         let y = y.map(|x| pack_char(x) as u32);
//         (S::from(x), S::from(y))
//     }
//     assert_eq!(padding, 8 * 8 - s.len());
//     assert_eq!(
//         head,
//         vec![
//             f(b"AGCAAAAA", b"AGCAAAAA"),
//             f(b"CGCACAAA", b"CGCACAAA"),
//             f(b"GTGAGAAA", b"GTGAGAAA"),
//             f(b"TTGATAAA", b"TTGATAAA"),
//             f(b"AAGAAAAA", b"AAGAAAAA"),
//             f(b"AATAAAAA", b"AATAAAAA"),
//             f(b"CATAAAAA", b"CATAAAAA"),
//             f(b"CCTAAAAA", b"CCTAAAAA"),
//         ]
//     );
// }

// #[test]
// fn par_iter_bp_delayed1() {
//     let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
//     let (head, padding) = s.as_slice().par_iter_bp_delayed(1, 1);
//     let head = head.collect::<Vec<_>>();
//     fn f(x: &[u8; 16], y: &[u8; 16]) -> (S, S) {
//         let x = x.map(|x| pack_char(x) as u32);
//         let y = y.map(|x| pack_char(x) as u32);
//         (S::from(x), S::from(y))
//     }
//     assert_eq!(padding, 8 * 8 - s.len());
//     assert_eq!(
//         head,
//         vec![
//             f(b"AGCAAAAA", b"AAAAAAAA"),
//             f(b"CGCACAAA", b"AGCAAAAA"),
//             f(b"GTGAGAAA", b"CGCACAAA"),
//             f(b"TTGATAAA", b"GTGAGAAA"),
//             f(b"AAGAAAAA", b"TTGATAAA"),
//             f(b"AATAAAAA", b"AAGAAAAA"),
//             f(b"CATAAAAA", b"AATAAAAA"),
//             f(b"CCTAAAAA", b"CATAAAAA"),
//         ]
//     );
// }

fn get(slice: &[u8], i: usize) -> u8 {
    if i < slice.len() {
        slice[i]
    } else {
        b'A'
    }
}

#[test]
fn par_iter_bp_fuzz() {
    let lens = (0..10)
        .map(|_| random_range(0..10))
        .chain((0..10).map(|_| random_range(10..100)))
        .chain((0..10).map(|_| random_range(100..1000)))
        .chain((0..10).map(|_| random_range(1000..10000)));
    for len in lens {
        let seq = AsciiSeqVec::random(len);
        eprintln!("SEQ: {:?}", seq.seq);
        let s = PackedSeqVec::from_ascii(&seq.seq);
        let context = random_range(1..512);
        let (head, padding) = s.as_slice().par_iter_bp(context);
        eprintln!("padding: {padding}");
        let head = head.collect::<Vec<_>>();
        fn f(x: &[u8; 16]) -> S {
            let x = x.map(|x| pack_char(x) as u32);
            S::from(x)
        }

        let stride = {
            let num_kmers = len.saturating_sub(context - 1);
            num_kmers.div_ceil(L).next_multiple_of(4)
        };

        eprintln!("stride {stride}");
        let len = head.len();
        eprintln!("len {len}");
        assert_eq!(
            head,
            (0..len)
                .map(|i| { f(&from_fn(|j| get(&seq.seq, i + stride * j))) })
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn par_iter_bp_delayed_fuzz() {
    let lens = (0..10)
        .map(|_| random_range(0..10))
        .chain((0..10).map(|_| random_range(10..100)))
        .chain((0..10).map(|_| random_range(100..1000)))
        .chain((0..10).map(|_| random_range(1000..10000)));
    for len in lens {
        let seq = AsciiSeqVec::random(len);
        eprintln!("SEQ: {:?}", seq.seq);
        let s = PackedSeqVec::from_ascii(&seq.seq);
        let context = random_range(1..512);
        let delay = random_range(0..512);
        eprintln!("LEN {len} CONTEXT {context} DELAY {delay}");
        let (head, padding) = s.as_slice().par_iter_bp_delayed(context, delay);
        eprintln!("padding: {padding}");
        let head = head.collect::<Vec<_>>();
        fn f(x: &[u8; 16], y: &[u8; 16]) -> (S, S) {
            let x = x.map(|x| pack_char(x) as u32);
            let y = y.map(|x| pack_char(x) as u32);
            (S::from(x), S::from(y))
        }

        let stride = {
            let num_kmers = len.saturating_sub(context - 1);
            num_kmers.div_ceil(L).next_multiple_of(4)
        };

        eprintln!("stride {stride}");
        let len = head.len();
        eprintln!("len {len}");
        let ans = (0..len)
            .map(|i| {
                f(
                    &from_fn(|j| get(&seq.seq, i + stride * j)),
                    &from_fn(|j| {
                        if i < delay {
                            b'A'
                        } else {
                            get(&seq.seq, (i + stride * j).wrapping_sub(delay))
                        }
                    }),
                )
            })
            .collect::<Vec<_>>();
        if head != ans {
            for (i, (x, y)) in head.iter().zip(ans.iter()).enumerate() {
                if x != y {
                    eprintln!("head {i} {x:?} != {y:?}");
                }
            }
        }
        assert!(head == ans);
    }
}

#[test]
fn par_iter_bp_delayed2_fuzz() {
    let lens = (0..10)
        .map(|_| random_range(0..10))
        .chain((0..10).map(|_| random_range(10..100)))
        .chain((0..10).map(|_| random_range(100..1000)))
        .chain((0..10).map(|_| random_range(1000..10000)));
    for len in lens {
        let seq = AsciiSeqVec::random(len);
        eprintln!("SEQ: {:?}", seq.seq);
        let s = PackedSeqVec::from_ascii(&seq.seq);
        let context = random_range(1..512);
        let delay = random_range(0..512);
        let delay2 = random_range(delay..=512);
        eprintln!("LEN {len} CONTEXT {context} DELAY {delay}");
        let (head, padding) = s.as_slice().par_iter_bp_delayed_2(context, delay, delay2);
        eprintln!("padding: {padding}");
        let head = head.collect::<Vec<_>>();
        fn f(x: &[u8; 16], y: &[u8; 16], z: &[u8; 16]) -> (S, S, S) {
            let x = x.map(|x| pack_char(x) as u32);
            let y = y.map(|x| pack_char(x) as u32);
            let z = z.map(|x| pack_char(x) as u32);
            (S::from(x), S::from(y), S::from(z))
        }

        let stride = {
            let num_kmers = len.saturating_sub(context - 1);
            num_kmers.div_ceil(L).next_multiple_of(4)
        };

        eprintln!("stride {stride}");
        let len = head.len();
        eprintln!("len {len}");
        assert_eq!(
            head,
            (0..len)
                .map(|i| {
                    f(
                        &from_fn(|j| get(&seq.seq, i + stride * j)),
                        &from_fn(|j| {
                            if i < delay {
                                b'A'
                            } else {
                                get(&seq.seq, i + stride * j - delay)
                            }
                        }),
                        &from_fn(|j| {
                            if i < delay2 {
                                b'A'
                            } else {
                                get(&seq.seq, i + stride * j - delay2)
                            }
                        }),
                    )
                })
                .collect::<Vec<_>>()
        );
    }
}

// #[test]
// fn par_iter_bp_delayed01() {
//     let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
//     let (head, padding) = s.as_slice().par_iter_bp_delayed_2(1, 0, 1);
//     let head = head.collect::<Vec<_>>();
//     fn f(x: &[u8; 16], y: &[u8; 16], z: &[u8; 16]) -> (S, S, S) {
//         let x = x.map(|x| pack_char(x) as u32);
//         let y = y.map(|x| pack_char(x) as u32);
//         let z = z.map(|x| pack_char(x) as u32);
//         (S::from(x), S::from(y), S::from(z))
//     }
//     assert_eq!(padding, 8 * 8 - s.len());
//     assert_eq!(
//         head,
//         vec![
//             f(b"AGCAAAAA", b"AGCAAAAA", b"AAAAAAAA"),
//             f(b"CGCACAAA", b"CGCACAAA", b"AGCAAAAA"),
//             f(b"GTGAGAAA", b"GTGAGAAA", b"CGCACAAA"),
//             f(b"TTGATAAA", b"TTGATAAA", b"GTGAGAAA"),
//             f(b"AAGAAAAA", b"AAGAAAAA", b"TTGATAAA"),
//             f(b"AATAAAAA", b"AATAAAAA", b"AAGAAAAA"),
//             f(b"CATAAAAA", b"CATAAAAA", b"AATAAAAA"),
//             f(b"CCTAAAAA", b"CCTAAAAA", b"CATAAAAA"),
//         ]
//     );
// }

#[test]
fn slice_get() {
    let n = 1000;
    let s = PackedSeqVec::random(n);
    let iter_bp = s.as_slice().iter_bp().collect::<Vec<_>>();
    let get = (0..n).map(|i| s.as_slice().get(i)).collect::<Vec<_>>();
    assert_eq!(iter_bp, get);
}
