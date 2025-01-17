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
        let mut rng = rand::thread_rng();
        let seq: Vec<_> = (0..n)
            .map(|_| b"ACGTacgt"[rng.gen::<u8>() as usize % 8])
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
        let mut rng = rand::thread_rng();
        let seq: Vec<_> = (0..n)
            .map(|_| b"ACGTacgt"[rng.gen::<u8>() as usize % 8])
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
}

#[test]
fn par_iter_bp_delayed0() {
    let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
    let (head, tail) = s.as_slice().par_iter_bp_delayed(1, 0);
    let head = head.collect::<Vec<_>>();
    let tail = tail.iter_bp().collect::<Vec<_>>();
    fn f(x: &[u8; 8], y: &[u8; 8]) -> (u32x8, u32x8) {
        let x = x.map(|x| pack_char(x) as u32);
        let y = y.map(|x| pack_char(x) as u32);
        (u32x8::from(x), u32x8::from(y))
    }
    assert_eq!(
        head,
        vec![
            f(b"AAGACGAA", b"AAGACGAA"),
            f(b"CAGACTAA", b"CAGACTAA"),
            f(b"GCTAGTAA", b"GCTAGTAA"),
            f(b"TCTCGTAA", b"TCTCGTAA"),
        ]
    );
    assert_eq!(tail, vec![0, 1, 3, 2]);
}

#[test]
fn par_iter_bp_delayed1() {
    let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
    let (head, tail) = s.as_slice().par_iter_bp_delayed(1, 1);
    let head = head.collect::<Vec<_>>();
    let tail = tail.iter_bp().collect::<Vec<_>>();
    fn f(x: &[u8; 8], y: &[u8; 8]) -> (u32x8, u32x8) {
        let x = x.map(|x| pack_char(x) as u32);
        let y = y.map(|x| pack_char(x) as u32);
        (u32x8::from(x), u32x8::from(y))
    }
    assert_eq!(
        head,
        vec![
            f(b"AAGACGAA", b"AAAAAAAA"),
            f(b"CAGACTAA", b"AAGACGAA"),
            f(b"GCTAGTAA", b"CAGACTAA"),
            f(b"TCTCGTAA", b"GCTAGTAA"),
        ]
    );
    assert_eq!(tail, vec![0, 1, 3, 2]);
}

#[test]
fn par_iter_bp_delayed_large() {
    let seq = AsciiSeqVec::random(48);
    eprintln!("SEQ: {:?}", seq.seq);
    let s = PackedSeqVec::from_ascii(&seq.seq);
    let delay = 16;
    let (head, _tail) = s.as_slice().par_iter_bp_delayed(17, delay);
    let head = head.collect::<Vec<_>>();
    fn f(x: &[u8; 8], y: &[u8; 8]) -> (u32x8, u32x8) {
        let x = x.map(|x| pack_char(x) as u32);
        let y = y.map(|x| pack_char(x) as u32);
        (u32x8::from(x), u32x8::from(y))
    }
    let stride = 4;
    let len = head.len();
    assert_eq!(
        head,
        (0..len)
            .map(|i| {
                f(
                    &from_fn(|j| seq.seq[i + stride * j]),
                    &from_fn(|j| {
                        if i < delay {
                            b'A'
                        } else {
                            seq.seq[i + stride * j - delay]
                        }
                    }),
                )
            })
            .collect::<Vec<_>>()
    );
}

#[test]
fn par_iter_bp_delayed01() {
    let s = PackedSeqVec::from_ascii(b"ACGTAACCGGTTAAACCCGGGTTTAAAAAAAAACGT");
    let (head, tail) = s.as_slice().par_iter_bp_delayed_2(1, 0, 1);
    let head = head.collect::<Vec<_>>();
    let tail = tail.iter_bp().collect::<Vec<_>>();
    fn f(x: &[u8; 8], y: &[u8; 8], z: &[u8; 8]) -> (u32x8, u32x8, u32x8) {
        let x = x.map(|x| pack_char(x) as u32);
        let y = y.map(|x| pack_char(x) as u32);
        let z = z.map(|x| pack_char(x) as u32);
        (u32x8::from(x), u32x8::from(y), u32x8::from(z))
    }
    assert_eq!(
        head,
        vec![
            f(b"AAGACGAA", b"AAGACGAA", b"AAAAAAAA"),
            f(b"CAGACTAA", b"CAGACTAA", b"AAGACGAA"),
            f(b"GCTAGTAA", b"GCTAGTAA", b"CAGACTAA"),
            f(b"TCTCGTAA", b"TCTCGTAA", b"GCTAGTAA"),
        ]
    );
    assert_eq!(tail, vec![0, 1, 3, 2]);
}

#[test]
fn get() {
    let n = 1000;
    let s = PackedSeqVec::random(n);
    let iter_bp = s.as_slice().iter_bp().collect::<Vec<_>>();
    let get = (0..n).map(|i| s.as_slice().get(i)).collect::<Vec<_>>();
    assert_eq!(iter_bp, get);
}