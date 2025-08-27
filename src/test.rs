use rand::{random_range, Rng};
use wide::u32x8;

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

#[test]
fn pack() {
    for n in 0..=128 {
        let mut rng = rand::rng();
        let seq: Vec<_> = (0..n)
            .map(|_| b"ACGTacgt"[rng.random::<u8>() as usize % 8])
            .collect();
        let (packed_1, len1) = pack_naive(&seq);
        let packed_2 = PackedSeqVec::from_ascii(&seq);
        assert_eq!(len1, packed_2.len());
        assert_eq!(packed_1, packed_2.seq[..packed_2.len().div_ceil(4)]);
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
        assert_eq!(len1, packed_2.len());
        assert_eq!(packed_1, packed_2.seq[..packed_2.len().div_ceil(4)]);
    }
}

#[test]
#[ignore = "This is a benchmark, not a test"]
fn pack_word_bench() {
    let n = 10000000;
    let x = vec![b'A'; n];
    let packed = PackedSeqVec::from_ascii(&x);
    for k in 1..=32 {
        let start = std::time::Instant::now();
        let mut sum = 0;
        for i in 0..=(x.len() - k) {
            let word = packed.read_kmer(k, i);
            sum += word;
        }
        std::hint::black_box(sum);
        let d = start.elapsed();
        eprintln!(
            "PackedSeqVec::read_kmer({k}) took {} ms or {} ns per k-mer",
            d.as_millis(),
            d.as_nanos() as f32 / (x.len() - k + 1) as f32
        );
    }
}

#[test]
fn pack_word() {
    let packed = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGT");
    let slice = packed.slice(0..1);
    assert_eq!(slice.as_u64(), 0b00000000);
    let slice = packed.slice(1..2);
    assert_eq!(slice.as_u64(), 0b00000001);
    let slice = packed.slice(2..3);
    assert_eq!(slice.as_u64(), 0b00000011);
    let slice = packed.slice(3..4);
    assert_eq!(slice.as_u64(), 0b00000010);
    let slice = packed.slice(0..2);
    assert_eq!(slice.as_u64(), 0b00000100);
    let slice = packed.slice(0..3);
    assert_eq!(slice.as_u64(), 0b00110100);
    let slice = packed.slice(0..4);
    assert_eq!(slice.as_u64(), 0b10110100);
    let slice = packed.slice(0..8);
    assert_eq!(slice.as_u64(), 0b1011010010110100);
    let slice = packed.slice(0..16);
    assert_eq!(slice.as_u64(), 0b10110100101101001011010010110100);
    let slice = packed.slice(0..28);
    assert_eq!(
        slice.as_u64(),
        0b10110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(0..29).as_u64(),
        0b0010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(1..30).as_u64(),
        0b0100101101001011010010110100101101001011010010110100101101
    );
    assert_eq!(
        packed.slice(2..31).as_u64(),
        0b1101001011010010110100101101001011010010110100101101001011
    );
    assert_eq!(
        packed.slice(3..32).as_u64(),
        0b1011010010110100101101001011010010110100101101001011010010
    );
    assert_eq!(
        packed.slice(0..30).as_u64(),
        0b010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(1..31).as_u64(),
        0b110100101101001011010010110100101101001011010010110100101101
    );
    assert_eq!(
        packed.slice(2..32).as_u64(),
        0b101101001011010010110100101101001011010010110100101101001011
    );
    assert_eq!(
        packed.slice(3..33).as_u64(),
        0b001011010010110100101101001011010010110100101101001011010010
    );
    assert_eq!(
        packed.slice(0..31).as_u64(),
        0b11010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(1..32).as_u64(),
        0b10110100101101001011010010110100101101001011010010110100101101
    );
    assert_eq!(
        packed.slice(2..33).as_u64(),
        0b00101101001011010010110100101101001011010010110100101101001011
    );
    assert_eq!(
        packed.slice(3..34).as_u64(),
        0b01001011010010110100101101001011010010110100101101001011010010
    );
    assert_eq!(
        packed.slice(0..32).as_u64(),
        0b1011010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(1..33).as_u64(),
        0b0010110100101101001011010010110100101101001011010010110100101101
    );
    assert_eq!(
        packed.slice(2..34).as_u64(),
        0b0100101101001011010010110100101101001011010010110100101101001011
    );
    assert_eq!(
        packed.slice(3..35).as_u64(),
        0b1101001011010010110100101101001011010010110100101101001011010010
    );
}

#[test]
fn pack_u128() {
    let packed = PackedSeqVec::from_ascii(b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
    let slice = packed.slice(0..1);
    assert_eq!(slice.as_u128(), 0b0000000000000000);
    let slice = packed.slice(1..2);
    assert_eq!(slice.as_u128(), 0b0000000000000001);
    let slice = packed.slice(2..3);
    assert_eq!(slice.as_u128(), 0b0000000000000011);
    let slice = packed.slice(3..4);
    assert_eq!(slice.as_u128(), 0b0000000000000010);
    let slice = packed.slice(0..2);
    assert_eq!(slice.as_u128(), 0b0000000000000100);
    let slice = packed.slice(0..3);
    assert_eq!(slice.as_u128(), 0b0000000000110100);
    assert_eq!(
        packed.slice(0..30).as_u128(),
        0b010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(0..31).as_u128(),
        0b11010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(0..32).as_u128(),
        0b1011010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(0..33).as_u128(),
        0b001011010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(0..34).as_u128(),
        0b01001011010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(0..35).as_u128(),
        0b1101001011010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(0..36).as_u128(),
        0b101101001011010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(1..36).as_u128(),
        0b1011010010110100101101001011010010110100101101001011010010110100101101
    );
    assert_eq!(
        packed.slice(2..36).as_u128(),
        0b10110100101101001011010010110100101101001011010010110100101101001011
    );
    assert_eq!(
        packed.slice(0..61).as_u128(),
        0b00101101001011010010110100101101001011010010110100101101001011010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(0..60).as_u128(),
        0b101101001011010010110100101101001011010010110100101101001011010010110100101101001011010010110100101101001011010010110100
    );
    assert_eq!(
        packed.slice(1..61).as_u128(),
        0b001011010010110100101101001011010010110100101101001011010010110100101101001011010010110100101101001011010010110100101101
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
    assert_eq!(packed.len(), seq.len());
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
        while packed.len() < 100 {
            let push_len = rng.random_range(1..=8);
            let range = packed.len()..(packed.len() + push_len);
            let range2 = packed.push_ascii(&seq[range.clone()]);
            assert_eq!(range, range2);
        }
        let slice = packed.as_slice();
        for (i, &c) in seq.iter().take(100).enumerate() {
            assert_eq!(slice.get_ascii(i), c);
        }
        let packed2 = PackedSeqVec::from_ascii(&seq[..packed.len()]);
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

#[test]
fn par_iter_bp() {
    let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
    let (head, padding) = s.as_slice().par_iter_bp(1);
    let head = head.collect::<Vec<_>>();
    fn f(x: &[u8; 8]) -> u32x8 {
        let x = x.map(|x| pack_char(x) as u32);
        u32x8::from(x)
    }
    assert_eq!(padding, 8 * 8 - s.len());
    assert_eq!(
        head,
        vec![
            f(b"AGCAAAAA"),
            f(b"CGCACAAA"),
            f(b"GTGAGAAA"),
            f(b"TTGATAAA"),
            f(b"AAGAAAAA"),
            f(b"AATAAAAA"),
            f(b"CATAAAAA"),
            f(b"CCTAAAAA"),
        ]
    );
}

#[test]
fn par_iter_bp_delayed0() {
    let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
    let (head, padding) = s.as_slice().par_iter_bp_delayed(1, 0);
    let head = head.collect::<Vec<_>>();
    fn f(x: &[u8; 8], y: &[u8; 8]) -> (u32x8, u32x8) {
        let x = x.map(|x| pack_char(x) as u32);
        let y = y.map(|x| pack_char(x) as u32);
        (u32x8::from(x), u32x8::from(y))
    }
    assert_eq!(padding, 8 * 8 - s.len());
    assert_eq!(
        head,
        vec![
            f(b"AGCAAAAA", b"AGCAAAAA"),
            f(b"CGCACAAA", b"CGCACAAA"),
            f(b"GTGAGAAA", b"GTGAGAAA"),
            f(b"TTGATAAA", b"TTGATAAA"),
            f(b"AAGAAAAA", b"AAGAAAAA"),
            f(b"AATAAAAA", b"AATAAAAA"),
            f(b"CATAAAAA", b"CATAAAAA"),
            f(b"CCTAAAAA", b"CCTAAAAA"),
        ]
    );
}

#[test]
fn par_iter_bp_delayed1() {
    let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
    let (head, padding) = s.as_slice().par_iter_bp_delayed(1, 1);
    let head = head.collect::<Vec<_>>();
    fn f(x: &[u8; 8], y: &[u8; 8]) -> (u32x8, u32x8) {
        let x = x.map(|x| pack_char(x) as u32);
        let y = y.map(|x| pack_char(x) as u32);
        (u32x8::from(x), u32x8::from(y))
    }
    assert_eq!(padding, 8 * 8 - s.len());
    assert_eq!(
        head,
        vec![
            f(b"AGCAAAAA", b"AAAAAAAA"),
            f(b"CGCACAAA", b"AGCAAAAA"),
            f(b"GTGAGAAA", b"CGCACAAA"),
            f(b"TTGATAAA", b"GTGAGAAA"),
            f(b"AAGAAAAA", b"TTGATAAA"),
            f(b"AATAAAAA", b"AAGAAAAA"),
            f(b"CATAAAAA", b"AATAAAAA"),
            f(b"CCTAAAAA", b"CATAAAAA"),
        ]
    );
}

fn get(slice: &[u8], i: usize) -> u8 {
    if i < slice.len() {
        slice[i]
    } else {
        b'A'
    }
}

#[test]
fn par_iter_bp_fuzz() {
    let lens = (0..100)
        .map(|_| random_range(0..10))
        .chain((0..100).map(|_| random_range(10..100)))
        .chain((0..10).map(|_| random_range(100..1000)))
        .chain((0..10).map(|_| random_range(1000..10000)));
    for mut len in lens {
        eprintln!();
        let seq = AsciiSeqVec::random(len);
        eprintln!("SEQ: {:?}", seq.seq);
        let s = PackedSeqVec::from_ascii(&seq.seq);

        let offset = random_range(0..=8.min(len));
        eprintln!("OFFSET: {offset:?}");
        let seq = seq.slice(offset..len);
        let s = s.slice(offset..len);
        len -= offset;

        let context = random_range(1..=512.min(len).max(1));
        eprintln!("CONTEXT: {context:?}");
        let (head, padding) = s.par_iter_bp(context);
        let head = head.collect::<Vec<_>>();
        fn f(x: &[u8; 8]) -> u32x8 {
            let x = x.map(|x| pack_char(x) as u32);
            u32x8::from(x)
        }

        let head_len = head.len();
        eprintln!("par it len {head_len}");
        eprintln!("padding: {padding}");

        // Test padding len.
        assert_eq!(8 * head_len, len + 7 * (context - 1) + padding);
        assert!(padding < 32);

        // Test context overlap.
        for i in 0..7 {
            for j in 0..context - 1 {
                assert_eq!(
                    head[head_len - (context - 1) + j].as_array_ref()[i],
                    head[j].as_array_ref()[i + 1],
                    "Context check failed at {i} {j}"
                );
            }
        }

        let stride = head_len - (context - 1);
        eprintln!("stride {stride}");

        assert_eq!(
            head,
            (0..head_len)
                .map(|i| { f(&from_fn(|j| get(seq.0, i + stride * j))) })
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn par_iter_bp_delayed_fuzz() {
    let lens = (0..100)
        .map(|_| random_range(0..10))
        .chain((0..100).map(|_| random_range(10..100)))
        .chain((0..10).map(|_| random_range(100..1000)))
        .chain((0..10).map(|_| random_range(1000..10000)));
    for mut len in lens {
        eprintln!();
        let seq = AsciiSeqVec::random(len);
        eprintln!("SEQ: {:?}", seq.seq);
        let s = PackedSeqVec::from_ascii(&seq.seq);

        let offset = random_range(0..=8.min(len));
        // let offset = 0;
        eprintln!("OFFSET: {offset:?}");
        let seq = seq.slice(offset..len);
        let s = s.slice(offset..len);
        len -= offset;

        // let context = random_range(1..=512.min(len).max(1));
        let context = 1;
        let delay = random_range(0..512);
        eprintln!("LEN {len} CONTEXT {context} DELAY {delay}");
        let (head, padding) = s.par_iter_bp_delayed(context, delay);
        eprintln!("padding: {padding}");
        let head = head.collect::<Vec<_>>();
        fn f(x: &[u8; 8], y: &[u8; 8]) -> (u32x8, u32x8) {
            let x = x.map(|x| pack_char(x) as u32);
            let y = y.map(|x| pack_char(x) as u32);
            (u32x8::from(x), u32x8::from(y))
        }

        let head_len = head.len();
        eprintln!("par it len {head_len}");
        eprintln!("padding: {padding}");

        // Test padding len.
        assert_eq!(8 * head_len, len + 7 * (context - 1) + padding);
        assert!(padding < 32);

        // Test context overlap.
        for i in 0..7 {
            for j in 0..context - 1 {
                assert_eq!(
                    head[head_len - (context - 1) + j].0.as_array_ref()[i],
                    head[j].0.as_array_ref()[i + 1],
                    "Context check failed at {i} {j}"
                );
            }
        }

        let stride = head_len - (context - 1);
        eprintln!("stride {stride}");

        let ans = (0..head_len)
            .map(|i| {
                f(
                    &from_fn(|j| get(seq.0, i + stride * j)),
                    &from_fn(|j| {
                        if i < delay {
                            b'A'
                        } else {
                            get(seq.0, (i + stride * j).wrapping_sub(delay))
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
    let lens = (0..100)
        .map(|_| random_range(0..10))
        .chain((0..100).map(|_| random_range(10..100)))
        .chain((0..10).map(|_| random_range(100..1000)))
        .chain((0..10).map(|_| random_range(1000..10000)));
    for mut len in lens {
        let seq = AsciiSeqVec::random(len);
        eprintln!("SEQ: {:?}", seq.seq);
        let s = PackedSeqVec::from_ascii(&seq.seq);

        let offset = random_range(0..=8.min(len));
        eprintln!("OFFSET: {offset:?}");
        let seq = seq.slice(offset..len);
        let s = s.slice(offset..len);
        len -= offset;

        let context = random_range(1..=512.min(len).max(1));
        let delay = random_range(0..512);
        let delay2 = random_range(delay..=512);
        eprintln!("LEN {len} CONTEXT {context} DELAY {delay}");
        let (head, padding) = s.par_iter_bp_delayed_2(context, delay, delay2);
        eprintln!("padding: {padding}");
        let head = head.collect::<Vec<_>>();
        fn f(x: &[u8; 8], y: &[u8; 8], z: &[u8; 8]) -> (u32x8, u32x8, u32x8) {
            let x = x.map(|x| pack_char(x) as u32);
            let y = y.map(|x| pack_char(x) as u32);
            let z = z.map(|x| pack_char(x) as u32);
            (u32x8::from(x), u32x8::from(y), u32x8::from(z))
        }

        let head_len = head.len();
        eprintln!("par it len {head_len}");
        eprintln!("padding: {padding}");

        // Test padding len.
        assert_eq!(8 * head_len, len + 7 * (context - 1) + padding);
        assert!(padding < 32);

        // Test context overlap.
        for i in 0..7 {
            for j in 0..context - 1 {
                assert_eq!(
                    head[head_len - (context - 1) + j].0.as_array_ref()[i],
                    head[j].0.as_array_ref()[i + 1],
                    "Context check failed at {i} {j}"
                );
            }
        }

        let stride = head_len - (context - 1);
        eprintln!("stride {stride}");

        let ans = (0..head_len)
            .map(|i| {
                f(
                    &from_fn(|j| get(seq.0, i + stride * j)),
                    &from_fn(|j| {
                        if i < delay {
                            b'A'
                        } else {
                            get(seq.0, i + stride * j - delay)
                        }
                    }),
                    &from_fn(|j| {
                        if i < delay2 {
                            b'A'
                        } else {
                            get(seq.0, i + stride * j - delay2)
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
fn par_iter_bp_delayed01() {
    let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
    let (head, padding) = s.as_slice().par_iter_bp_delayed_2(1, 0, 1);
    let head = head.collect::<Vec<_>>();
    fn f(x: &[u8; 8], y: &[u8; 8], z: &[u8; 8]) -> (u32x8, u32x8, u32x8) {
        let x = x.map(|x| pack_char(x) as u32);
        let y = y.map(|x| pack_char(x) as u32);
        let z = z.map(|x| pack_char(x) as u32);
        (u32x8::from(x), u32x8::from(y), u32x8::from(z))
    }
    assert_eq!(padding, 8 * 8 - s.len());
    assert_eq!(
        head,
        vec![
            f(b"AGCAAAAA", b"AGCAAAAA", b"AAAAAAAA"),
            f(b"CGCACAAA", b"CGCACAAA", b"AGCAAAAA"),
            f(b"GTGAGAAA", b"GTGAGAAA", b"CGCACAAA"),
            f(b"TTGATAAA", b"TTGATAAA", b"GTGAGAAA"),
            f(b"AAGAAAAA", b"AAGAAAAA", b"TTGATAAA"),
            f(b"AATAAAAA", b"AATAAAAA", b"AAGAAAAA"),
            f(b"CATAAAAA", b"CATAAAAA", b"AATAAAAA"),
            f(b"CCTAAAAA", b"CCTAAAAA", b"CATAAAAA"),
        ]
    );
}

#[test]
fn slice_get() {
    let n = 1000;
    let s = PackedSeqVec::random(n);
    let iter_bp = s.as_slice().iter_bp().collect::<Vec<_>>();
    let get = (0..n).map(|i| s.as_slice().get(i)).collect::<Vec<_>>();
    assert_eq!(iter_bp, get);
}

#[test]
fn rc_rc_u64() {
    let n = 10000;
    let seq = PackedSeqVec::random(n);
    for k in 1..=29 {
        for i in 0..=(n - k) {
            let word = seq.read_kmer(k, i);
            let rc = seq.read_revcomp_kmer(k, i);
            assert_eq!(packed_seq::revcomp_u64(word, k), rc, "k={k} i={i}");
            assert_eq!(packed_seq::revcomp_u64(rc, k), word, "k={k} i={i}");
        }
    }
    let seq = AsciiSeqVec::random(n);
    for k in 1..=32 {
        for i in 0..=(n - k) {
            let word = seq.read_kmer(k, i);
            let rc = seq.read_revcomp_kmer(k, i);
            assert_eq!(packed_seq::revcomp_u64(word, k), rc, "k={k} i={i}");
            assert_eq!(packed_seq::revcomp_u64(rc, k), word, "k={k} i={i}");
        }
    }
}

#[test]
fn rc_rc_seq() {
    for n in (0..100).chain(100000..100000 + 10) {
        {
            let seq = AsciiSeqVec::random(n);
            let rc = seq.as_slice().to_revcomp();
            let rc_rc = rc.as_slice().to_revcomp();
            assert_eq!(seq.seq, rc_rc.seq);
        }

        {
            let seq = PackedSeqVec::random(n);
            let rc = seq.as_slice().to_revcomp();
            let rc_rc = rc.as_slice().to_revcomp();
            assert_eq!(seq.seq, rc_rc.seq);
        }
    }
}

// NOTE: For me, this requires jemalloc to actually trigger a segfault.
// System malloc seems to consistently include some padding bytes at the end, so
// that unaligned reads are never actually out of bounds.
#[test]
#[ignore = "requires jemalloc, and flaky anyway"]
fn as_u64_read_unaligned_does_not_segfault() {
    // #[global_allocator]
    // static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

    let mut vs = vec![];
    loop {
        let len = rand::random_range(4..130 * 4);
        let ascii_seq = vec![b'A'; len];
        let seq = PackedSeqVec::from_ascii(&ascii_seq);
        drop(ascii_seq);
        let kmer = seq.read_kmer(4, len - 4);
        std::hint::black_box(kmer);
        vs.push(seq);
    }
}

#[test]
fn packed_seq_push_seq() {
    let mut packed = PackedSeqVec::default();
    let mut rng = rand::rng();
    let mut total_len_in_bp = 0;
    let num_seqs = 100;
    let mut last_end = 0;
    for _ in 0..num_seqs {
        let len = rng.random_range(0..100);
        let new = PackedSeqVec::random(len);
        eprintln!("new len {len} seq len {}", new.seq.len());
        let range = packed.push_seq(new.as_slice());
        assert!(range.start <= last_end + 3);
        assert_eq!(range.len(), len);
        last_end = range.end;
        eprintln!(
            "len {len} range {range:?} cur len bp {} bytes {} capacity {}",
            packed.len(),
            packed.seq.len(),
            packed.seq.capacity()
        );
        total_len_in_bp += len.next_multiple_of(4);
    }
    assert_eq!(packed.seq.len(), total_len_in_bp.div_ceil(4) + 16,);
}
