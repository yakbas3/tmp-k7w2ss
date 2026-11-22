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
//! The caller must make the 96 bytes from the line start readable.

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
