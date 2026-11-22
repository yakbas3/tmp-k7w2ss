//! One-pass CSV reader for event and click lines: the bit masks of one line
//! give its field bounds directly, with no delimiter list in memory.
//!
//! Per line: compare the bytes from the line start for commas, line breaks
//! and quotes, find the line break, check the comma count, take the comma
//! positions from the mask words with `tzcnt`/`blsr`, and check and strip
//! the quotes with `pext`. A line that does not have the fast shape (wrong
//! field count, a misplaced quote, too long, no line break in reach) is
//! refused, and the caller hands it to the two-stage reader in `scan.rs`,
//! which is exact for every input. The fast shape only accepts: whatever it
//! accepts, it splits exactly as `split_line` does.
//!
//! An event line is 56 to 95 bytes, so it is read as 96 bytes: the first
//! three commas come from the first 64 bytes, the rest of the line from the
//! 64 bytes starting at the third comma. A click line fits in 64 bytes.
//!
//! A line with a long ID does not fit those reads (an ID may be a letter
//! and 32 hex digits, and a line then up to 145 bytes). `wide_line` reads
//! such a line from the masks of 160 bytes and the bytes around its
//! delimiters: the same accept-only contract, tried when the narrow reader
//! refuses.
//!
//! The caller must make the 192 bytes from the line start readable.

use crate::scan::Bounds;

/// Not a line of the fast shape.
pub const REFUSED: u32 = u32::MAX;

/// Masks of up to three 32-byte blocks: commas, line breaks, quotes, and
/// whether every one of the first 64 bytes is in 1..=127.
struct Masks {
    c: [u32; 3],
    n: [u32; 3],
    q: [u32; 3],
    clean: bool,
}

#[inline(always)]
unsafe fn masks<const BLOCKS: usize>(p: *const u8) -> Masks {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let comma = _mm256_set1_epi8(b',' as i8);
        let nl = _mm256_set1_epi8(b'\n' as i8);
        let quote = _mm256_set1_epi8(b'"' as i8);
        let mut out = Masks { c: [0; 3], n: [0; 3], q: [0; 3], clean: false };
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        out.clean = _mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_min_epi8(a, b), _mm256_setzero_si256())) == -1;
        let m = |x: __m256i, k: __m256i| _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, k)) as u32;
        out.c[0] = m(a, comma);
        out.n[0] = m(a, nl);
        out.q[0] = m(a, quote);
        out.c[1] = m(b, comma);
        out.n[1] = m(b, nl);
        out.q[1] = m(b, quote);
        if BLOCKS == 3 {
            let x = _mm256_loadu_si256(p.add(64) as *const __m256i);
            out.c[2] = m(x, comma);
            out.n[2] = m(x, nl);
            out.q[2] = m(x, quote);
        }
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        masks_scalar::<BLOCKS>(p)
    }
}

#[allow(dead_code)]
unsafe fn masks_scalar<const BLOCKS: usize>(p: *const u8) -> Masks {
    let mut out = Masks { c: [0; 3], n: [0; 3], q: [0; 3], clean: true };
    for i in 0..32 * BLOCKS {
        let b = *p.add(i);
        let (w, bit) = (i / 32, 1u32 << (i % 32));
        if b == b',' {
            out.c[w] |= bit;
        }
        if b == b'\n' {
            out.n[w] |= bit;
        }
        if b == b'"' {
            out.q[w] |= bit;
        }
        if i < 64 && !(1..=127).contains(&b) {
            out.clean = false;
        }
    }
    out
}

/// The bits of `x` selected by `m`, packed to the bottom.
#[inline(always)]
fn pext(x: u64, m: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        core::arch::x86_64::_pext_u64(x, m)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let (mut m, mut out, mut k) = (m, 0u64, 0);
        while m != 0 {
            let b = m.trailing_zeros();
            out |= ((x >> b) & 1) << k;
            k += 1;
            m &= m - 1;
        }
        out
    }
}

/// `x` with the bits from `n` up cleared; `x` itself for `n >= 64`.
#[inline(always)]
fn bzhi(x: u64, n: u32) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        core::arch::x86_64::_bzhi_u64(x, n)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let n = n & 0xFF;
        if n >= 64 {
            x
        } else {
            x & ((1u64 << n) - 1)
        }
    }
}

/// Lowest set bit position, 64 for zero; one `tzcnt` with no zeroing
/// `xor` in front.
#[inline(always)]
fn tz(x: u64) -> u32 {
    crate::scan::lowest_bit(x)
}

#[inline(always)]
fn blsr(x: u64) -> u64 {
    crate::scan::clear_lowest(x)
}

/// Starts and lengths of the fields from the delimiter positions (`d[0]`
/// is -1, the line break before the line; `d[k + 1]` ends field `k`), given
/// as `front` and `back` plus `add` (two halves of four lanes, built in
/// registers), and the quoted fields `quoted` (bit `k`: field `k` is
/// wrapped in quotes). Returns the fields of the first `N` that end before
/// they start: a field that is one lone quote.
#[inline(always)]
unsafe fn bounds<const N: usize>(front: [u32; 4], back: [u32; 4], add: u32, quoted: u64, out: &mut Bounds) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
    {
        use core::arch::x86_64::*;
        let lo = _mm_setr_epi32(front[0] as i32, front[1] as i32, front[2] as i32, front[3] as i32);
        let hi = _mm_add_epi32(
            _mm_setr_epi32(back[0] as i32, back[1] as i32, back[2] as i32, back[3] as i32),
            _mm_set1_epi32(add as i32),
        );
        let v = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        let q8 = _pdep_u64(quoted, 0x0101_0101_0101_0101);
        let q = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(q8 as i64));
        let start = _mm256_sub_epi32(_mm256_add_epi32(v, q), _mm256_set1_epi32(-1));
        let next = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7));
        let len = _mm256_sub_epi32(_mm256_sub_epi32(next, start), q);
        _mm256_store_si256(out.s.as_mut_ptr() as *mut __m256i, start);
        _mm256_store_si256(out.n.as_mut_ptr() as *mut __m256i, len);
        _mm256_movemask_ps(_mm256_castsi256_ps(len)) as u32 & ((1 << N) - 1)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
    {
        let mut d = [0u32; 8];
        for k in 0..4 {
            d[k] = front[k];
            d[k + 4] = back[k].wrapping_add(add);
        }
        let mut lone = 0u32;
        for k in 0..8 {
            let q = ((quoted >> k) & 1) as u32;
            let start = d[k].wrapping_add(1).wrapping_add(q);
            let next = d[(k + 1).min(7)];
            let len = next.wrapping_sub(start).wrapping_sub(q);
            out.s[k] = start;
            out.n[k] = len;
            if k < N && (len as i32) < 0 {
                lone |= 1 << k;
            }
        }
        lone
    }
}

/// The event line (7 fields) at `p`: fills `out` with the field bounds,
/// relative to `p`, quotes stripped, and returns the offset of its line
/// break, or `REFUSED`. `clean` is set when the first 64 bytes from `p`
/// are all in 1..=127 (so both ID fields are).
#[inline(always)]
pub unsafe fn event_line(p: *const u8, out: &mut Bounds, clean: &mut bool) -> u32 {
    let m = masks::<3>(p);
    *clean = m.clean;
    let cf = u64::from(m.c[0]) | (u64::from(m.c[1]) << 32);
    let nf = u64::from(m.n[0]) | (u64::from(m.n[1]) << 32);
    let qf = u64::from(m.q[0]) | (u64::from(m.q[1]) << 32);
    // The first three commas, from the first 64 bytes.
    let a0 = tz(cf);
    let t = blsr(cf);
    let a1 = tz(t);
    let t = blsr(t);
    let a2 = tz(t);
    // The 64 bytes from the third comma: bit 0 is that comma.
    let sh = a2 & 63;
    let wide = |lo: u64, hi: u32| ((((u128::from(hi)) << 64) | u128::from(lo)) >> sh) as u64;
    let cb = wide(cf, m.c[2]);
    let qb = wide(qf, m.q[2]);
    // The first line break, straight from its mask, so that the next line
    // does not wait for the comma chain: 128 when there is none in reach.
    // (A select: one line in six is shorter than 64 bytes, a coin toss for
    // a branch.)
    let nl = std::hint::select_unpredictable(nf != 0, tz(nf), 64 + tz(u64::from(m.n[2])));
    // Its offset from the third comma; huge when it comes first.
    let lb = nl.wrapping_sub(a2);
    let cl = bzhi(cb, lb);
    // Refused: fewer than three commas in the first 64 bytes, a line break
    // before the third, no line break within 64 bytes of it (or past the
    // 96 read), or not exactly six commas before the line break. Any of
    // them leaves a bit set.
    let mut bad = u64::from((cl.count_ones() ^ 4) | (a2 & 64) | (lb & !63));
    let t = blsr(cl);
    let b3 = tz(t);
    let t = blsr(t);
    let b4 = tz(t);
    let t = blsr(t);
    let b5 = tz(t);
    // Quotes. Fields 0 to 2 in the first word: starts 0, a0+1, a1+1, ends
    // before a0, a1, a2. Fields 3 to 6 in the second: starts after bits 0,
    // b3, b4, b5, ends before b3, b4, b5, lb.
    let ef = bzhi(cf, a2 + 1);
    let sf = 1 | (bzhi(cf, a1 + 1) << 1);
    let sb = cl << 1;
    let eb = blsr(cl) | (1u64 << (lb & 63));
    let open = pext(qf, sf) | (pext(qb, sb) << 3);
    let close = pext(qf << 1, ef) | (pext(qb << 1, eb) << 3);
    // Every quote of the line is the first or last byte of a field.
    let stray = (bzhi(qf, a2) & !(sf | (ef >> 1))) | (bzhi(qb, lb) & !(sb | (eb >> 1)));
    bad |= stray | (open ^ close);
    let lone = bounds::<7>([u32::MAX, a0, a1, a2], [b3, b4, b5, lb], a2, open, out);
    if (bad | u64::from(lone)) != 0 {
        return REFUSED;
    }
    nl
}

/// The click line (4 fields) at `p`, as `event_line`. Reads 64 bytes.
#[inline(always)]
pub unsafe fn click_line(p: *const u8, out: &mut Bounds, clean: &mut bool) -> u32 {
    let m = masks::<2>(p);
    *clean = m.clean;
    let cf = u64::from(m.c[0]) | (u64::from(m.c[1]) << 32);
    let nf = u64::from(m.n[0]) | (u64::from(m.n[1]) << 32);
    let qf = u64::from(m.q[0]) | (u64::from(m.q[1]) << 32);
    let l = tz(nf);
    let cl = bzhi(cf, l);
    let mut bad = u64::from((cl.count_ones() ^ 3) | (l & 64));
    let a0 = tz(cl);
    let t = blsr(cl);
    let a1 = tz(t);
    let a2 = tz(blsr(t));
    let s = 1 | (cl << 1);
    let e = cl | (1u64 << (l & 63));
    let open = pext(qf, s);
    let close = pext(qf << 1, e);
    let stray = bzhi(qf, l) & !(s | (e >> 1));
    bad |= stray | (open ^ close);
    let lone = bounds::<4>([u32::MAX, a0, a1, a2], [0; 4], l, open, out);
    if (bad | u64::from(lone)) != 0 {
        return REFUSED;
    }
    l
}

/// Bytes `wide_line` takes masks of. A line whose line break is not below
/// this is refused. Its byte reads end at most 3 bytes past the line break,
/// so it reads at most `WIDE + 2` bytes from the line start.
pub const WIDE: usize = 160;

/// Masks of the first 160 bytes: delimiters (commas and line breaks) as
/// three words and two zero words after them, quotes of the first 128
/// bytes as two words, and whether all 160 bytes are in 1..=127.
struct WideMasks {
    d: [u64; 5],
    q: [u64; 2],
    clean: bool,
}

#[inline(always)]
unsafe fn masks_wide(p: *const u8) -> WideMasks {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        // A byte is a delimiter when it equals the table entry of its low
        // nibble: ',' at 0xC, '\n' at 0xA, 0xFF (no byte below 0x80) at the
        // rest. A byte from 0x80 up looks up 0 and is not 0.
        let table = _mm256_setr_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, b'\n' as i8, -1, b',' as i8, -1, -1, -1, //
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, b'\n' as i8, -1, b',' as i8, -1, -1, -1,
        );
        let quote = _mm256_set1_epi8(b'"' as i8);
        let ld = |i: usize| _mm256_loadu_si256(p.add(32 * i) as *const __m256i);
        let x = [ld(0), ld(1), ld(2), ld(3), ld(4)];
        let d = |x: __m256i| u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_shuffle_epi8(table, x), x)) as u32);
        let q = |x: __m256i| u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, quote)) as u32);
        // Signed: a byte is in 1..=127 exactly when it is above zero.
        let low = _mm256_min_epi8(_mm256_min_epi8(_mm256_min_epi8(x[0], x[1]), _mm256_min_epi8(x[2], x[3])), x[4]);
        WideMasks {
            d: [d(x[0]) | (d(x[1]) << 32), d(x[2]) | (d(x[3]) << 32), d(x[4]), 0, 0],
            q: [q(x[0]) | (q(x[1]) << 32), q(x[2]) | (q(x[3]) << 32)],
            clean: _mm256_movemask_epi8(_mm256_cmpgt_epi8(low, _mm256_setzero_si256())) == -1,
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut out = WideMasks { d: [0; 5], q: [0; 2], clean: true };
        for i in 0..WIDE {
            let b = *p.add(i);
            let (w, bit) = (i / 64, 1u64 << (i % 64));
            if b == b',' || b == b'\n' {
                out.d[w] |= bit;
            }
            if i < 128 && b == b'"' {
                out.q[w] |= bit;
            }
            if !(1..=127).contains(&b) {
                out.clean = false;
            }
        }
        out
    }
}

/// The quotes of the 32 bytes from `p + 128`.
#[inline(always)]
unsafe fn quotes_tail(p: *const u8) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let x = _mm256_loadu_si256(p.add(128) as *const __m256i);
        u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, _mm256_set1_epi8(b'"' as i8))) as u32)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut q = 0;
        for i in 0..32 {
            q |= u64::from(*p.add(128 + i) == b'"') << i;
        }
        q
    }
}

/// What `wide_line` expects around delimiter `k` (lane `k`, four bytes
/// from the byte before it): a quote, a comma (a line break for the last
/// delimiter and the lanes after it), a quote, and any byte.
struct Pattern<const N: usize>;

impl<const N: usize> Pattern<N> {
    const BYTES: [u8; 32] = {
        let mut out = [0u8; 32];
        let mut k = 0;
        while k < 8 {
            out[4 * k] = b'"';
            out[4 * k + 1] = if k + 1 < N { b',' } else { b'\n' };
            out[4 * k + 2] = b'"';
            k += 1;
        }
        out
    };
}

/// The delimiters of a line in eight lanes: `e[k]` for `k < N`, then
/// zeros.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
type Lanes = core::arch::x86_64::__m256i;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
type Lanes = [u32; 8];

#[inline(always)]
unsafe fn lanes<const N: usize>(e: &[u32; 8]) -> Lanes {
    let at = |k: usize| if k < N { e[k] } else { 0 };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
    {
        use core::arch::x86_64::*;
        let a = |k: usize| at(k) as i32;
        _mm256_setr_epi32(a(0), a(1), a(2), a(3), a(4), a(5), a(6), a(7))
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
    {
        [at(0), at(1), at(2), at(3), at(4), at(5), at(6), at(7)]
    }
}

/// The bytes around the delimiters `v` against `Pattern<N>`: bit `4k + j`
/// (for `k < N`) is set when byte `v[k] - 1 + j` matches. Each delimiter
/// must be in 1..160, so the reads are within `p[0..162]`. Lanes from `N`
/// on are not read. Also returns lane `k` as 1 when byte `v[k] - 1` (the
/// last of field `k`) is a quote, else 0.
#[inline(always)]
unsafe fn around<const N: usize>(p: *const u8, v: Lanes) -> (u32, Lanes) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
    {
        use core::arch::x86_64::*;
        let live = _mm256_cmpgt_epi32(_mm256_set1_epi32(N as i32), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        let g = _mm256_mask_i32gather_epi32::<1>(_mm256_setzero_si256(), p.wrapping_sub(1) as *const i32, v, live);
        let pat = _mm256_loadu_si256(Pattern::<N>::BYTES.as_ptr() as *const __m256i);
        let hit = _mm256_cmpeq_epi8(g, pat);
        (_mm256_movemask_epi8(hit) as u32, _mm256_and_si256(hit, _mm256_set1_epi32(1)))
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
    {
        let mut hits = 0u32;
        for k in 0..N {
            for j in 0..4 {
                if *p.add(v[k] as usize - 1 + j) == Pattern::<N>::BYTES[4 * k + j] {
                    hits |= 1 << (4 * k + j);
                }
            }
        }
        let mut q = [0u32; 8];
        for (k, q) in q.iter_mut().enumerate() {
            *q = (hits >> (4 * k)) & 1;
        }
        (hits, q)
    }
}

/// `bounds` for the delimiters in lanes: field `k` from after `v[k - 1]`
/// (-1 for `k` = 0) to `v[k]`, quotes stripped when lane `k` of `q` is 1.
/// Returns the fields of the first `N` that end before they start.
#[inline(always)]
unsafe fn lane_bounds<const N: usize>(v: Lanes, q: Lanes, out: &mut Bounds) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
    {
        use core::arch::x86_64::*;
        let ones = _mm256_set1_epi32(-1);
        let prev = _mm256_blend_epi32::<1>(_mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6)), ones);
        let start = _mm256_sub_epi32(_mm256_add_epi32(prev, q), ones);
        let len = _mm256_sub_epi32(_mm256_sub_epi32(v, start), q);
        _mm256_store_si256(out.s.as_mut_ptr() as *mut __m256i, start);
        _mm256_store_si256(out.n.as_mut_ptr() as *mut __m256i, len);
        _mm256_movemask_ps(_mm256_castsi256_ps(len)) as u32 & ((1 << N) - 1)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
    {
        let quoted = (0..8).fold(0u64, |a, k| a | (u64::from(q[k]) << k));
        bounds::<N>([u32::MAX, v[0], v[1], v[2]], [v[3], v[4], v[5], v[6]], 0, quoted, out)
    }
}

/// The line of `N` fields (7 or 4) at `p` whose line break is below byte
/// 160: as `event_line` and `click_line`, for the lines too wide for them.
/// `clean` is set when the 160 bytes from `p` are all in 1..=127 (so both ID
/// fields are).
///
/// The delimiters come from 64-bit windows of the delimiter mask, each
/// starting right after a delimiter found so far: field 0 from the first
/// word, then fields 1 and 2, 3 and 4, 5 and 6 two at a time. So it also
/// refuses a line whose field 0 is empty or longer than 62 bytes, or whose
/// fields 1 and 2 (3 and 4, 5 and 6) are together longer than 63 bytes
/// (for 4 fields: field 3 longer than 64). Generated lines have at most
/// 35 + 27 there.
#[inline(never)]
pub unsafe fn wide_line<const N: usize>(p: *const u8, out: &mut Bounds, clean: &mut bool) -> u32 {
    let m = masks_wide(p);
    // e[k] ends field k: the commas, then the line break e[N - 1]. Each
    // window holds the 64 bits from `x`, right after delimiter k - 1, and
    // gives the next two delimiters, or x + 64 for one it does not hold.
    // Such a guess is kept only when the byte there is a delimiter (checked
    // below), and then it is the next one, as the window held none. A
    // window without any gives the same guess twice, and `lane_bounds`
    // flags the field between them, of length -1.
    let mut e = [0u32; 8];
    e[0] = tz(m.d[0]);
    let mut k = 1;
    while k < N {
        let x = e[k - 1] + 1;
        let w = if k == 1 {
            // x is in 2..=63 once e[0] is checked below.
            (((u128::from(m.d[1]) << 64) | u128::from(m.d[0])) >> (x & 63)) as u64
        } else {
            // Bound: e[0] <= 64 and each window adds at most 65, so x <=
            // 195 for the third window, x >> 6 <= 3, and the words read are
            // at most d[3] and d[4] (both zero).
            let i = (x >> 6) as usize & 3;
            (((u128::from(m.d[i + 1]) << 64) | u128::from(m.d[i])) >> (x & 63)) as u64
        };
        e[k] = x + tz(w);
        e[k + 1] = x + tz(blsr(w));
        k += 2;
    }
    let nl = e[N - 1];
    // Field 0 is 1 to 62 bytes (not empty, so the reads below start at p),
    // and the line break is in 64..160 (the narrow readers take every
    // well-formed line shorter than 64 bytes). The delimiters do not fall (each
    // window starts after the last), so all of e[0..N] are in 1..160.
    if (e[0].wrapping_sub(1) >= 62) | (nl.wrapping_sub(64) >= WIDE as u32 - 64) {
        return REFUSED;
    }
    let v = lanes::<N>(&e);
    let (hits, quoted) = around::<N>(p, v);
    let live = (1u32 << (4 * N)) - 1;
    // The first N - 1 delimiters are commas and the next a line break.
    let delims = 0x2222_2222 & live;
    let mut bad = (hits & delims) ^ delims;
    // Quotes at the first and last byte of each field, at bit 4k for field
    // k. Lane k holds the last byte of field k (bit 4k) and the first of
    // field k + 1 (bit 4k + 2). An empty field has its delimiters there,
    // no quotes.
    let close = hits & 0x1111_1111 & live;
    let open = (u32::from(*p == b'"') | ((hits << 2) & 0x1111_1110)) & live;
    bad |= open ^ close;
    // Every quote below the line break is one of those (a field that is
    // one lone quote counts twice here, and `bounds` flags it).
    let mut quotes = m.q[0].count_ones() + bzhi(m.q[1], nl - 64).count_ones();
    if nl > 128 {
        quotes += bzhi(quotes_tail(p), nl - 128).count_ones();
    }
    bad |= quotes ^ (2 * close.count_ones());
    let lone = lane_bounds::<N>(v, quoted, out);
    *clean = m.clean;
    if (bad | lone) != 0 {
        return REFUSED;
    }
    nl
}
