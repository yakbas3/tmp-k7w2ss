//! Stage 1 of the CSV reader: one pass over a window of bytes that records
//! where the commas and newlines are and which quotes are illegal.
//!
//! The dialect makes this simple: no value holds a comma, quote or newline,
//! so every comma and every `\n` is structural whatever the quoting. A quote
//! is legal only as the first or last byte of a field, so a quote with no
//! delimiter on either side makes its line malformed. Those are rare and
//! go to an exception list; the per-field open/close check in stage 2
//! catches the rest (`"abc`, `abc"`, a lone `"`).

/// 64-byte block to three bitmasks: commas, newlines, quotes.
#[inline(always)]
unsafe fn masks(p: *const u8) -> (u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let m = |x: __m256i, c: i8| -> u64 {
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, _mm256_set1_epi8(c))) as u32 as u64
        };
        return (
            m(a, b',' as i8) | (m(b, b',' as i8) << 32),
            m(a, b'\n' as i8) | (m(b, b'\n' as i8) << 32),
            m(a, b'"' as i8) | (m(b, b'"' as i8) << 32),
        );
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        use core::arch::x86_64::*;
        let mut out = (0u64, 0u64, 0u64);
        for i in 0..4 {
            let x = _mm_loadu_si128(p.add(16 * i) as *const __m128i);
            let m = |c: i8| -> u64 { _mm_movemask_epi8(_mm_cmpeq_epi8(x, _mm_set1_epi8(c))) as u32 as u64 };
            out.0 |= m(b',' as i8) << (16 * i);
            out.1 |= m(b'\n' as i8) << (16 * i);
            out.2 |= m(b'"' as i8) << (16 * i);
        }
        return out;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let x0 = vld1q_u8(p);
        let x1 = vld1q_u8(p.add(16));
        let x2 = vld1q_u8(p.add(32));
        let x3 = vld1q_u8(p.add(48));
        const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vld1q_u8(BITS.as_ptr());
        let m = |c: u8| -> u64 {
            let k = vdupq_n_u8(c);
            let t0 = vandq_u8(vceqq_u8(x0, k), bits);
            let t1 = vandq_u8(vceqq_u8(x1, k), bits);
            let t2 = vandq_u8(vceqq_u8(x2, k), bits);
            let t3 = vandq_u8(vceqq_u8(x3, k), bits);
            let s0 = vpaddq_u8(t0, t1);
            let s1 = vpaddq_u8(t2, t3);
            let s = vpaddq_u8(s0, s1);
            let s = vpaddq_u8(s, s);
            vgetq_lane_u64(vreinterpretq_u64_u8(s), 0)
        };
        return (m(b','), m(b'\n'), m(b'"'));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        masks_scalar(p)
    }
}

/// Running signed minimum of the bytes scanned. It stays above zero exactly
/// while every byte is in 1..=127, which is what an ID needs to be packed
/// without looking at its bytes again.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
type Lowest = core::arch::x86_64::__m256i;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
type Lowest = i8;

#[inline(always)]
unsafe fn lowest_start() -> Lowest {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        core::arch::x86_64::_mm256_set1_epi8(127)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        127
    }
}

/// Fold the 64 bytes at `p` into the minimum.
#[inline(always)]
unsafe fn lowest_add(acc: Lowest, p: *const u8) -> Lowest {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        _mm256_min_epi8(acc, _mm256_min_epi8(a, b))
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut m = acc;
        for i in 0..64 {
            m = m.min(*p.add(i) as i8);
        }
        m
    }
}

#[inline(always)]
unsafe fn lowest_above_zero(acc: Lowest) -> bool {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        _mm256_movemask_epi8(_mm256_cmpgt_epi8(acc, _mm256_setzero_si256())) == -1
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        acc > 0
    }
}

/// Portable reference for `masks`; also what the tests compare against.
#[allow(dead_code)]
pub unsafe fn masks_scalar(p: *const u8) -> (u64, u64, u64) {
    let mut out = (0u64, 0u64, 0u64);
    for i in 0..64 {
        let b = *p.add(i);
        out.0 |= u64::from(b == b',') << i;
        out.1 |= u64::from(b == b'\n') << i;
        out.2 |= u64::from(b == b'"') << i;
    }
    out
}

pub struct Scan {
    /// Positions of commas and newlines, ascending, relative to the window.
    /// Slot 0 stands for the line break before the window (position -1);
    /// the positions start at slot 1. See `delims()`.
    d: Vec<u32>,
    /// One byte per delimiter, laid out like `d`: `CLOSES` when the byte
    /// before the delimiter is a quote, `OPENS` when the byte after it is,
    /// `NEWLINE` when the delimiter is a line break. See `flags()`.
    fl: Vec<u8>,
    /// Positions of quotes with no delimiter on either side.
    pub badq: Vec<u32>,
    pub nd: usize,
    /// Every byte of the window (and of the up to 63 bytes scanned past
    /// it) is in 1..=127.
    pub clean: bool,
}

const PAD: usize = 80;
/// How far ahead of the scan the input is prefetched. Flat from 512 to
/// 4096 bytes on the test box.
const AHEAD: usize = 1024;

/// The field that ends at this delimiter ends with a quote.
const CLOSES: u8 = 1;
/// The field that starts after this delimiter starts with a quote.
const OPENS: u8 = 2;
/// The delimiter is a line break, not a comma.
const NEWLINE: u8 = 4;
const CLOSES8: u64 = 0x0101_0101_0101_0101;
const OPENS8: u64 = 0x0202_0202_0202_0202;
const NEWLINE8: u64 = 0x0404_0404_0404_0404;

/// Flag bytes of up to eight delimiters. `before` and `after` have bit `p`
/// set when the byte before, or after, position `p` is a quote; `nm` when
/// position `p` is a line break. `skip` delimiters of `delim` are passed
/// over first.
#[inline(always)]
fn flag_bytes(before: u64, after: u64, nm: u64, delim: u64, skip: u32) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        use core::arch::x86_64::{_pdep_u64, _pext_u64};
        let c = _pext_u64(before, delim) >> skip;
        let o = _pext_u64(after, delim) >> skip;
        let n = _pext_u64(nm, delim) >> skip;
        _pdep_u64(c, CLOSES8) | _pdep_u64(o, OPENS8) | _pdep_u64(n, NEWLINE8)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut m = delim;
        for _ in 0..skip {
            m &= m.wrapping_sub(1);
        }
        let mut out = 0u64;
        let mut i = 0;
        while m != 0 && i < 8 {
            let p = m.trailing_zeros();
            let f = (((before >> p) & 1) * CLOSES as u64)
                | (((after >> p) & 1) * OPENS as u64)
                | (((nm >> p) & 1) * NEWLINE as u64);
            out |= f << (8 * i);
            m &= m - 1;
            i += 1;
        }
        out
    }
}

/// `m` without its lowest set bit, as one `blsr`. Written plainly, the
/// compiler turns most of the eight into a `lea` and an `and`.
#[inline(always)]
pub(crate) fn clear_lowest(m: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi1"))]
    unsafe {
        let r: u64;
        core::arch::asm!("blsr {r}, {m}", m = in(reg) m, r = lateout(reg) r, options(pure, nomem, nostack));
        r
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi1")))]
    {
        m & m.wrapping_sub(1)
    }
}

/// `m.trailing_zeros()` (64 for 0) as one `tzcnt`. The compiler puts a
/// zeroing `xor` in front of each of its own to break a false dependency
/// on the output register; that costs a front-end slot per delimiter.
#[inline(always)]
pub(crate) fn lowest_bit(m: u64) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi1"))]
    unsafe {
        let r: u64;
        core::arch::asm!("tzcnt {r}, {m}", m = in(reg) m, r = lateout(reg) r, options(pure, nomem, nostack));
        r as u32
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi1")))]
    {
        m.trailing_zeros()
    }
}

/// Append the illegal quotes of the block at `at` to the list: the last
/// byte of the previous block when `carried`, then the bits of `illegal`.
#[cold]
#[inline(never)]
unsafe fn bad_quotes(bad: *mut u32, mut nbad: usize, at: u32, mut illegal: u64, carried: u64) -> usize {
    if carried != 0 {
        *bad.add(nbad) = at - 1;
        nbad += 1;
    }
    while illegal != 0 {
        *bad.add(nbad) = at + illegal.trailing_zeros();
        nbad += 1;
        illegal &= illegal - 1;
    }
    nbad
}

/// Stage 1 variants, decided once at start-up (`init_simd`). AVX2 is the
/// default.
pub const MODE_AVX2: u8 = 0;
/// AVX-512 F, BW and VL: one 64-byte load, compares straight to masks.
pub const MODE_AVX512: u8 = 1;
/// MODE_AVX512 plus VBMI2 `vpcompressb` for positions and flag bytes.
pub const MODE_VBMI2: u8 = 2;

static SIMD_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(MODE_AVX2);

#[inline(always)]
pub fn simd_mode() -> u8 {
    SIMD_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Pick the stage 1 variant for this CPU. 512-bit instructions lower the
/// clock of the whole core on Skylake and Cascade Lake (measured: parse
/// +43 ms), so AVX-512 is on by default only where VBMI2 says the core is
/// Ice Lake or newer. `ENGINE_AVX512=0|1` forces it off or on (on still
/// needs F, BW and VL); `ENGINE_AVX512_VBMI2=1` adds the `vpcompressb`
/// variant, and only on a CPU that has VBMI2. Returns the mode.
pub fn init_simd() -> u8 {
    #[allow(unused_mut)]
    let mut mode = MODE_AVX2;
    #[cfg(target_arch = "x86_64")]
    {
        let fbw = std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("popcnt");
        let vbmi2 = fbw && std::is_x86_feature_detected!("avx512vbmi2");
        let on = match std::env::var("ENGINE_AVX512").as_deref() {
            Ok("0") => false,
            Ok("1") => fbw,
            _ => vbmi2,
        };
        if on {
            mode = MODE_AVX512;
            if vbmi2 && std::env::var("ENGINE_AVX512_VBMI2").is_ok_and(|v| v == "1") {
                mode = MODE_VBMI2;
            }
        }
    }
    SIMD_MODE.store(mode, std::sync::atomic::Ordering::Relaxed);
    mode
}

/// Tests: whether this CPU can run `mode`.
#[cfg(test)]
pub fn can_run(mode: u8) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let fbw = std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl");
        match mode {
            MODE_AVX2 => true,
            MODE_AVX512 => fbw,
            _ => fbw && std::is_x86_feature_detected!("avx512vbmi2"),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        mode == MODE_AVX2
    }
}

/// `masks` with one 64-byte load and three compares into mask registers;
/// also folds the block into the running signed minimum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn masks512(p: *const u8, lowest: &mut core::arch::x86_64::__m512i) -> (u64, u64, u64) {
    use core::arch::x86_64::*;
    let v = _mm512_loadu_si512(p as *const _);
    *lowest = _mm512_min_epi8(*lowest, v);
    (
        _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b',' as i8)),
        _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\n' as i8)),
        _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'"' as i8)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn all_above_zero512(acc: core::arch::x86_64::__m512i) -> bool {
    use core::arch::x86_64::*;
    _mm512_cmpgt_epi8_mask(acc, _mm512_setzero_si512()) == u64::MAX
}

/// Positions and flag bytes of the (at most 16) delimiters of one block:
/// byte offsets and per-byte flags are packed by `delim` with
/// `vpcompressb`, widened and stored whole. Writes 16 slots of each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2")]
#[inline]
unsafe fn compress16(out: *mut u32, fout: *mut u8, base_pos: u32, delim: u64, before: u64, after: u64, nm: u64) {
    use core::arch::x86_64::*;
    const IOTA: [u8; 64] = {
        let mut t = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            t[i] = i as u8;
            i += 1;
        }
        t
    };
    let iota = _mm512_loadu_si512(IOTA.as_ptr() as *const _);
    let idx = _mm512_maskz_compress_epi8(delim, iota);
    let pos = _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(idx)), _mm512_set1_epi32(base_pos as i32));
    _mm512_storeu_si512(out as *mut _, pos);
    let flags = _mm512_or_si512(
        _mm512_or_si512(_mm512_maskz_set1_epi8(before, CLOSES as i8), _mm512_maskz_set1_epi8(after, OPENS as i8)),
        _mm512_maskz_set1_epi8(nm, NEWLINE as i8),
    );
    let packed = _mm512_maskz_compress_epi8(delim, flags);
    _mm_storeu_si128(fout as *mut __m128i, _mm512_castsi512_si128(packed));
}

impl Scan {
    pub fn new(window: usize) -> Self {
        Scan {
            d: vec![u32::MAX; window + 64 + PAD + 1],
            fl: vec![0; window + 64 + PAD + 1],
            badq: vec![u32::MAX; window + 64 + PAD],
            nd: 0,
            clean: false,
        }
    }

    /// The delimiter positions. `delims().sub(1)` is readable and holds
    /// `u32::MAX`: the line break before the window.
    #[inline(always)]
    pub fn delims(&self) -> *const u32 {
        unsafe { self.d.as_ptr().add(1) }
    }

    /// The flag byte of each delimiter. `flags().sub(1)` is the flag of the
    /// line break before the window.
    #[inline(always)]
    pub fn flags(&self) -> *const u8 {
        unsafe { self.fl.as_ptr().add(1) }
    }

    /// Scan `len` bytes at `base`. The byte before `base` must be a line
    /// break (or the start of the file), and at least 64 bytes past
    /// `base + len` must be readable.
    #[inline]
    pub unsafe fn scan(&mut self, base: *const u8, len: usize) {
        #[cfg(target_arch = "x86_64")]
        match simd_mode() {
            MODE_AVX512 => return self.scan512(base, len),
            MODE_VBMI2 => return self.scan512v(base, len),
            _ => {}
        }
        self.scan_body::<MODE_AVX2>(base, len)
    }

    /// Tests: `scan` in a given mode, which `can_run` must allow.
    #[cfg(test)]
    pub unsafe fn scan_in(&mut self, mode: u8, base: *const u8, len: usize) {
        #[cfg(target_arch = "x86_64")]
        match mode {
            MODE_AVX512 => return self.scan512(base, len),
            MODE_VBMI2 => return self.scan512v(base, len),
            _ => {}
        }
        let _ = mode;
        self.scan_body::<MODE_AVX2>(base, len)
    }

    /// Stage 1 with AVX-512 compares into mask registers (F, BW, VL).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,popcnt")]
    unsafe fn scan512(&mut self, base: *const u8, len: usize) {
        self.scan_body::<MODE_AVX512>(base, len)
    }

    /// The same, with the positions and flag bytes of up to 16 delimiters
    /// a block packed by `vpcompressb` (VBMI2: Ice Lake and newer).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2,bmi1,bmi2,popcnt")]
    unsafe fn scan512v(&mut self, base: *const u8, len: usize) {
        self.scan_body::<MODE_VBMI2>(base, len)
    }

    /// The scan loop. `MODE` picks how a block becomes masks and how the
    /// masks become positions; everything else is shared, so the variants
    /// cannot disagree.
    #[inline(always)]
    unsafe fn scan_body<const MODE: u8>(&mut self, base: *const u8, len: usize) {
        debug_assert!(len + 64 + PAD < self.d.len());
        let d = self.d.as_mut_ptr().add(1);
        let fl = self.fl.as_mut_ptr().add(1);
        let bad = self.badq.as_mut_ptr();
        let (mut nd, mut nbad) = (0usize, 0usize);
        let mut prev_delim = 1u64;
        let mut prev_quote = 0u64;
        let mut pending = 0u64;
        let mut at = 0usize;
        let mut lowest = lowest_start();
        #[cfg(target_arch = "x86_64")]
        let mut lowest512: core::arch::x86_64::__m512i = std::mem::transmute([127i8; 64]);
        *d.sub(1) = u32::MAX;
        *fl.sub(1) = if *base == b'"' { OPENS } else { 0 };
        while at < len {
            // The input comes straight from the page cache in 4 KiB pages,
            // and the hardware prefetcher stops at every page end. Without
            // this a third of the scan is spent waiting on the two loads.
            crate::sys::prefetch(base.wrapping_add(at + AHEAD));
            #[cfg(target_arch = "x86_64")]
            let (mut cm, mut nm, mut qm) = if MODE == MODE_AVX2 {
                lowest = lowest_add(lowest, base.add(at));
                masks(base.add(at))
            } else {
                masks512(base.add(at), &mut lowest512)
            };
            #[cfg(not(target_arch = "x86_64"))]
            let (mut cm, mut nm, mut qm) = {
                lowest = lowest_add(lowest, base.add(at));
                masks(base.add(at))
            };
            let left = len - at;
            if left < 64 {
                let keep = (1u64 << left) - 1;
                cm &= keep;
                nm &= keep;
                qm &= keep;
            }
            let delim = cm | nm;
            // A quote that does not follow a delimiter must precede one.
            // Four blocks in five hold a quote, so a branch on "any quote"
            // is a coin toss; the check runs on every block and only the
            // rare illegal quote branches.
            let nonopen = qm & !((delim << 1) | prev_delim);
            let illegal = nonopen & !(delim >> 1) & !(1u64 << 63);
            let carried = pending & !delim & 1;
            pending = nonopen >> 63;
            if (illegal | carried) != 0 {
                nbad = bad_quotes(bad, nbad, at as u32, illegal, carried);
            }
            prev_delim = delim >> 63;
            // Quote flags per delimiter, so stage 2 never looks at the
            // bytes around a field. The byte after position 63 is the
            // first of the next block.
            let before = (qm << 1) | prev_quote;
            let after = (qm >> 1) | (u64::from(*base.add(at + 64) == b'"') << 63);
            prev_quote = qm >> 63;

            let base_pos = at as u32;
            let count = delim.count_ones() as usize;
            #[cfg(target_arch = "x86_64")]
            if MODE == MODE_VBMI2 && count <= 16 {
                compress16(d.add(nd), fl.add(nd), base_pos, delim, before, after, nm);
                nd += count;
                at += 64;
                continue;
            }
            let mut m = delim;
            let out = d.add(nd);
            let fout = fl.add(nd);
            for i in 0..8 {
                // Volatile keeps these eight plain stores. Left alone the
                // compiler packs them into one vector with seven inserts,
                // which is slower here.
                out.add(i).write_volatile(base_pos + lowest_bit(m));
                m = clear_lowest(m);
            }
            (fout as *mut u64).write_unaligned(flag_bytes(before, after, nm, delim, 0).to_le());
            if count > 8 {
                for i in 8..16 {
                    *out.add(i) = base_pos + m.trailing_zeros();
                    m &= m.wrapping_sub(1);
                }
                (fout.add(8) as *mut u64).write_unaligned(flag_bytes(before, after, nm, delim, 8).to_le());
                if count > 16 {
                    let mut i = 16;
                    while m != 0 {
                        *out.add(i) = base_pos + m.trailing_zeros();
                        m &= m - 1;
                        i += 1;
                    }
                    let mut k = 16;
                    while k < count {
                        (fout.add(k) as *mut u64)
                            .write_unaligned(flag_bytes(before, after, nm, delim, k as u32).to_le());
                        k += 8;
                    }
                }
            }
            nd += count;
            at += 64;
        }
        // Sentinels: stage 2 reads up to 8 entries past the last delimiter.
        for i in 0..16 {
            *d.add(nd + i) = u32::MAX;
        }
        *bad.add(nbad) = u32::MAX;
        self.nd = nd;
        #[cfg(target_arch = "x86_64")]
        if MODE != MODE_AVX2 {
            self.clean = all_above_zero512(lowest512);
            return;
        }
        self.clean = lowest_above_zero(lowest);
    }
}

/// Field bounds, quotes stripped, of the `N` fields (at most 7) that the
/// next `N` delimiters end, and whether that is not a well-formed line:
/// one of the first `N - 1` delimiters is a line break, the `N`th is not,
/// or a field has a quote at one end only or is one lone quote. `d` and
/// `fl` point at the `N` delimiters and their flags; the entry before each
/// belongs to the line break before the line. Only the first `N` slots of
/// the result mean anything.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
pub unsafe fn fields<const N: usize>(d: *const u32, fl: *const u8) -> ([u32; 8], [u32; 8], u32) {
    let mut s = [0u32; 8];
    let mut e = [0u32; 8];
    let mut bad = 0u32;
    for i in 0..N {
        let open = u32::from(*fl.add(i).sub(1) & OPENS != 0);
        let close = u32::from(*fl.add(i) & CLOSES != 0);
        let newline = *fl.add(i) & NEWLINE != 0;
        let st = (*d.add(i).sub(1)).wrapping_add(1) + open;
        let en = *d.add(i) - close;
        s[i] = st;
        e[i] = en;
        // A field that is one lone quote opens and closes, and ends
        // before it starts.
        bad |= (open ^ close) | u32::from(en < st) | u32::from(newline != (i == N - 1));
    }
    (s, e, bad)
}

/// The same with all fields at once: starts are the delimiters before plus
/// one plus the open flag, ends the own delimiters minus the close flag.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn fields<const N: usize>(d: *const u32, fl: *const u8) -> ([u32; 8], [u32; 8], u32) {
    use core::arch::x86_64::*;
    debug_assert!(N <= 7);
    // Byte k: the flags of delimiter k - 1 of the line.
    let f = u64::from_le((fl.sub(1) as *const u64).read_unaligned());
    let opens = (f >> 1) & CLOSES8;
    let closes = f & CLOSES8;
    let uneven = (opens ^ (closes >> 8)) & (u64::MAX >> (64 - 8 * N));
    // Bytes 1..=N: commas, then the line break.
    let breaks = (f & (NEWLINE8 >> (56 - 8 * N)) & !0xFF) ^ ((NEWLINE as u64) << (8 * N));
    let v = _mm256_loadu_si256(d.sub(1) as *const __m256i);
    let open = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(opens as i64));
    let close = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(closes as i64));
    let start = _mm256_sub_epi32(_mm256_add_epi32(v, open), _mm256_set1_epi32(-1));
    let end = _mm256_permutevar8x32_epi32(_mm256_sub_epi32(v, close), _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7));
    // A field that is one lone quote opens and closes, and ends before it
    // starts.
    let lone = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_sub_epi32(end, start))) as u32 & ((1 << N) - 1);
    let mut s = [0u32; 8];
    let mut e = [0u32; 8];
    _mm256_storeu_si256(s.as_mut_ptr() as *mut __m256i, start);
    _mm256_storeu_si256(e.as_mut_ptr() as *mut __m256i, end);
    (s, e, lone | u32::from((uneven | breaks) != 0))
}

/// Starts and lengths of the fields of one line, in memory. The parsers
/// read each one with a plain load: pulling the lanes out of a vector
/// register costs two uops each, a load one.
#[repr(C, align(32))]
pub struct Bounds {
    pub(crate) s: [u32; 8],
    pub(crate) n: [u32; 8],
}

impl Bounds {
    pub fn new() -> Self {
        Bounds { s: [0; 8], n: [0; 8] }
    }

    #[inline(always)]
    pub fn start(&self, i: usize) -> usize {
        // Volatile, or the compiler forwards the vector store and goes back
        // to extracting lanes.
        unsafe { std::ptr::read_volatile(self.s.as_ptr().add(i)) as usize }
    }
    #[inline(always)]
    pub fn len(&self, i: usize) -> usize {
        unsafe { std::ptr::read_volatile(self.n.as_ptr().add(i)) as usize }
    }
}

/// `fields`, with the starts and lengths stored to `out`. Returns `bad`.
#[inline(always)]
unsafe fn fields_into<const N: usize>(d: *const u32, fl: *const u8, out: *mut Bounds) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let (s, e, bad) = fields::<N>(d, fl);
        let start = _mm256_loadu_si256(s.as_ptr() as *const __m256i);
        let end = _mm256_loadu_si256(e.as_ptr() as *const __m256i);
        _mm256_store_si256((*out).s.as_mut_ptr() as *mut __m256i, start);
        _mm256_store_si256((*out).n.as_mut_ptr() as *mut __m256i, _mm256_sub_epi32(end, start));
        bad
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let (s, e, bad) = fields::<N>(d, fl);
        for i in 0..8 {
            (*out).s[i] = s[i];
            (*out).n[i] = e[i].wrapping_sub(s[i]);
        }
        bad
    }
}

/// `for_each_line`, with the fields handed over as `Bounds`.
#[inline(always)]
pub unsafe fn for_each_line_b<const N: usize>(scan: &Scan, limit: u32, mut f: impl FnMut(&Bounds)) -> u32 {
    let d = scan.delims();
    let fl = scan.flags();
    let badq = scan.badq.as_ptr();
    let nd = scan.nd;
    let mut b = Bounds { s: [0; 8], n: [0; 8] };
    let mut di = 0usize;
    let mut bq = 0usize;
    let mut ls = 0u32;
    while ls < limit {
        if di + N <= nd {
            let p = *d.add(di + N - 1);
            let bad = fields_into::<N>(d.add(di), fl.add(di), &mut b);
            if (bad == 0) & (*badq.add(bq) >= p) {
                f(&b);
                di += N;
                ls = p + 1;
                continue;
            }
        }
        match skip_line(scan, di, bq) {
            Some(next) => (di, bq, ls) = next,
            None => break,
        }
    }
    ls
}

/// The line whose delimiters start at `d[di]` is not a well-formed line of
/// the expected width, or fewer delimiters are left than a line needs.
/// Skip to the end of that line: the new `(di, bq, line start)`, or `None`
/// when no line break is left in the window.
#[cold]
#[inline(never)]
unsafe fn skip_line(scan: &Scan, mut di: usize, mut bq: usize) -> Option<(usize, usize, u32)> {
    let fl = scan.flags();
    let badq = scan.badq.as_ptr();
    while di < scan.nd && *fl.add(di) & NEWLINE == 0 {
        di += 1;
    }
    if di >= scan.nd {
        return None;
    }
    let p = *scan.delims().add(di);
    while *badq.add(bq) < p {
        bq += 1;
    }
    Some((di + 1, bq, p + 1))
}

/// Walk the complete lines of a scanned window. For each line that has
/// exactly `N` fields and no misplaced quote, `f(starts, ends)` is called.
/// Lines starting at or after `limit` are left alone. Returns the offset of
/// the first unconsumed line.
#[inline(always)]
pub unsafe fn for_each_line<const N: usize>(
    scan: &Scan,
    limit: u32,
    mut f: impl FnMut(&[u32; N], &[u32; N]),
) -> u32 {
    let d = scan.delims();
    let fl = scan.flags();
    let badq = scan.badq.as_ptr();
    let nd = scan.nd;
    let mut di = 0usize;
    let mut bq = 0usize;
    let mut ls = 0u32;
    while ls < limit {
        if di + N <= nd {
            let p = *d.add(di + N - 1);
            let (s, e, bad) = fields::<N>(d.add(di), fl.add(di));
            if (bad == 0) & (*badq.add(bq) >= p) {
                f(&*(s.as_ptr() as *const [u32; N]), &*(e.as_ptr() as *const [u32; N]));
                di += N;
                ls = p + 1;
                continue;
            }
        }
        match skip_line(scan, di, bq) {
            Some(next) => (di, bq, ls) = next,
            None => break,
        }
    }
    ls
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::split_line;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn masks_match_scalar() {
        let mut state = 1;
        for _ in 0..2000 {
            let mut buf = [0u8; 64];
            for b in buf.iter_mut() {
                *b = match rng(&mut state) % 6 {
                    0 => b',',
                    1 => b'\n',
                    2 => b'"',
                    _ => (rng(&mut state) % 256) as u8,
                };
            }
            unsafe {
                assert_eq!(masks(buf.as_ptr()), masks_scalar(buf.as_ptr()));
            }
        }
    }

    /// Random soup of the structural bytes: the fast line walker must accept
    /// exactly the lines the plain splitter accepts, with the same fields.
    #[test]
    fn lines_match_split_line() {
        let mut state = 7;
        for round in 0..3000 {
            let n = 1 + (rng(&mut state) % 700) as usize;
            let mut text = vec![b'\n'];
            let alphabet: &[u8] = if round % 3 == 0 { b"a,\"\n" } else { b"ab,,\"\n\r 1" };
            for _ in 0..n {
                text.push(alphabet[(rng(&mut state) % alphabet.len() as u64) as usize]);
            }
            text.push(b'\n');
            let body_len = text.len() - 1;
            text.extend_from_slice(&[0u8; 160]);
            for fields_n in [2usize, 4, 7] {
                let mut expect: Vec<Vec<Vec<u8>>> = Vec::new();
                let mut tmp = Vec::new();
                for line in text[1..body_len].split(|&b| b == b'\n') {
                    if split_line(line, fields_n, &mut tmp) {
                        expect.push(tmp.iter().map(|f| f.to_vec()).collect());
                    }
                }
                let mut scan = Scan::new(body_len);
                let mut got: Vec<Vec<Vec<u8>>> = Vec::new();
                unsafe {
                    let base = text.as_ptr().add(1);
                    scan.scan(base, body_len);
                    let mut collect = |s: &[u32], e: &[u32]| {
                        got.push(
                            (0..s.len())
                                .map(|i| text[1 + s[i] as usize..1 + e[i] as usize].to_vec())
                                .collect(),
                        );
                    };
                    let end = match fields_n {
                        2 => for_each_line::<2>(&scan, u32::MAX, |s, e| collect(s, e)),
                        4 => for_each_line::<4>(&scan, u32::MAX, |s, e| collect(s, e)),
                        _ => for_each_line::<7>(&scan, u32::MAX, |s, e| collect(s, e)),
                    };
                    assert_eq!(end as usize, body_len);
                }
                assert_eq!(got, expect, "round {round} fields {fields_n}: {:?}", String::from_utf8_lossy(&text[1..body_len]));
                // The same lines and fields through `Bounds`.
                let mut got_b: Vec<Vec<Vec<u8>>> = Vec::new();
                unsafe {
                    let mut collect = |b: &Bounds| {
                        got_b.push(
                            (0..fields_n).map(|i| text[1 + b.start(i)..1 + b.start(i) + b.len(i)].to_vec()).collect(),
                        );
                    };
                    let end = match fields_n {
                        2 => for_each_line_b::<2>(&scan, u32::MAX, |b| collect(b)),
                        4 => for_each_line_b::<4>(&scan, u32::MAX, |b| collect(b)),
                        _ => for_each_line_b::<7>(&scan, u32::MAX, |b| collect(b)),
                    };
                    assert_eq!(end as usize, body_len);
                }
                assert_eq!(got_b, expect, "bounds, round {round} fields {fields_n}");
            }
        }
    }

    /// Every delimiter's flag byte says what the bytes around it are, at
    /// every alignment to the 64-byte blocks, with one `Scan` reused.
    #[test]
    fn flags_match_bytes() {
        let mut state = 21;
        let mut scan = Scan::new(4096);
        for round in 0..4000 {
            let n = 1 + (rng(&mut state) % 600) as usize;
            let alphabet: &[u8] = match round % 4 {
                0 => b"\",\n",
                1 => b"ab\"\",\n",
                2 => b"abcdefgh\",",
                _ => b"abcdefghijklmnopqrstuvwxyz0123456789\"\"\"\",,,,,\n",
            };
            let mut text = vec![b'\n'];
            for _ in 0..n {
                text.push(alphabet[(rng(&mut state) % alphabet.len() as u64) as usize]);
            }
            // Hostile bytes past the window must not leak into it.
            text.extend_from_slice(&[b'"'; 160]);
            unsafe {
                let base = text.as_ptr().add(1);
                scan.scan(base, n);
                let body = &text[1..];
                let want: Vec<u32> = (0..n as u32).filter(|&i| matches!(body[i as usize], b',' | b'\n')).collect();
                assert_eq!(scan.nd, want.len(), "round {round}");
                assert_eq!(*scan.delims().sub(1), u32::MAX);
                assert_eq!(*scan.flags().sub(1) & OPENS != 0, body[0] == b'"');
                for (k, &p) in want.iter().enumerate() {
                    let p = p as usize;
                    assert_eq!(*scan.delims().add(k), p as u32, "round {round}");
                    let flag = *scan.flags().add(k);
                    assert_eq!(flag & CLOSES != 0, p > 0 && body[p - 1] == b'"', "round {round} delimiter {k} at {p}");
                    assert_eq!(flag & NEWLINE != 0, body[p] == b'\n', "round {round} delimiter {k} at {p}");
                    // The byte after the window's last byte belongs to no
                    // line of this window.
                    if p + 1 < n {
                        assert_eq!(flag & OPENS != 0, body[p + 1] == b'"', "round {round} delimiter {k} at {p}");
                    }
                }
            }
        }
    }

    /// Every stage 1 variant this CPU can run gives the AVX2 result exactly:
    /// positions, flags, illegal quotes, sentinels and `clean`, on soups of
    /// structural bytes with quotes and delimiters at the 64-byte block
    /// edges, dense blocks (over 8 and over 16 delimiters), odd bytes, and
    /// windows of every length mod 64.
    #[test]
    fn simd_variants_match() {
        let modes: Vec<u8> = [MODE_AVX512, MODE_VBMI2].into_iter().filter(|&m| can_run(m)).collect();
        eprintln!("stage 1 variants checked against AVX2: {modes:?}");
        let mut state = 77;
        let mut reference = Scan::new(4096);
        let mut other = Scan::new(4096);
        for round in 0..6000 {
            let n = 1 + (rng(&mut state) % 900) as usize;
            let alphabet: &[u8] = match round % 6 {
                0 => b"\",\n",
                1 => b",,,,,,,\n",
                2 => b"ab\"\",\n",
                3 => b"abcdefghij\",",
                4 => b"\x80\x00a,\"\n\x7f\x01",
                _ => b"abcdefghijklmnopqrstuvwxyz0123456789\"\"\",,,,,\n",
            };
            let mut text = vec![b'\n'];
            for i in 0..n {
                // Push structural bytes onto the block edges now and then.
                let edge = (i % 64 == 63 || i % 64 == 0) && rng(&mut state) % 3 == 0;
                let c = if edge {
                    b"\",\n"[(rng(&mut state) % 3) as usize]
                } else {
                    alphabet[(rng(&mut state) % alphabet.len() as u64) as usize]
                };
                text.push(c);
            }
            let tail: &[u8] = if round % 2 == 0 { b"\"" } else { b"a" };
            for _ in 0..160 {
                text.push(tail[0]);
            }
            unsafe {
                let base = text.as_ptr().add(1);
                reference.scan_in(MODE_AVX2, base, n);
                for &mode in &modes {
                    other.scan_in(mode, base, n);
                    assert_eq!(other.nd, reference.nd, "round {round} mode {mode}");
                    assert_eq!(other.clean, reference.clean, "round {round} mode {mode}");
                    let nd = reference.nd;
                    for k in 0..nd + 16 {
                        assert_eq!(*other.delims().add(k), *reference.delims().add(k), "round {round} mode {mode} slot {k}");
                    }
                    assert_eq!(*other.delims().sub(1), *reference.delims().sub(1));
                    for k in 0..nd {
                        assert_eq!(*other.flags().add(k), *reference.flags().add(k), "round {round} mode {mode} flag {k}");
                    }
                    assert_eq!(*other.flags().sub(1), *reference.flags().sub(1));
                    let mut k = 0;
                    loop {
                        assert_eq!(other.badq[k], reference.badq[k], "round {round} mode {mode} badq {k}");
                        if reference.badq[k] == u32::MAX {
                            break;
                        }
                        k += 1;
                    }
                }
            }
        }
    }

    /// `clean` is never set when the window holds a byte outside 1..=127,
    /// wherever that byte sits, and is set for plain text.
    #[test]
    fn clean_means_every_byte_is_plain() {
        let mut state = 55;
        let mut scan = Scan::new(4096);
        for round in 0..6000 {
            let n = 1 + (rng(&mut state) % 700) as usize;
            let mut text = vec![b'\n'];
            for _ in 0..n + 64 {
                text.push(b"abcxyz019,\"\n\r \x01\x7f"[(rng(&mut state) % 16) as usize]);
            }
            text.extend_from_slice(&[b'a'; 96]);
            let odd = round % 3 != 0;
            if odd {
                // One odd byte anywhere in the window, first and last
                // positions included.
                let at = match rng(&mut state) % 4 {
                    0 => 0,
                    1 => n - 1,
                    _ => (rng(&mut state) % n as u64) as usize,
                };
                text[1 + at] = [0x00u8, 0x80, 0xFF, 0xC3][(rng(&mut state) % 4) as usize];
            }
            unsafe {
                scan.scan(text.as_ptr().add(1), n);
            }
            assert_eq!(scan.clean, !odd, "round {round} len {n}");
        }
        // Zero padding right after the window only costs the shortcut.
        let mut text = vec![b'\n'];
        text.extend_from_slice(b"a,b\n");
        text.extend_from_slice(&[0u8; 160]);
        unsafe {
            scan.scan(text.as_ptr().add(1), 4);
        }
        assert!(!scan.clean);
    }

    /// Lines shaped like the real files (field lengths, 20% quoted, some
    /// malformed) cut into windows of every size: the walker and the plain
    /// splitter agree, also on what is left for the next window.
    #[test]
    fn realistic_lines_match_split_line() {
        let mut state = 33;
        let mut scan = Scan::new(1 << 12);
        for round in 0..1500 {
            let fields_n = [2usize, 4, 7][round % 3];
            let mut text = vec![b'\n'];
            let lines = 1 + rng(&mut state) % 40;
            for _ in 0..lines {
                let width = match rng(&mut state) % 40 {
                    0 => fields_n + 1,
                    1 => fields_n - 1,
                    2 => 0,
                    _ => fields_n,
                };
                for i in 0..width {
                    if i > 0 {
                        text.push(b',');
                    }
                    let len = match rng(&mut state) % 10 {
                        0 => 0,
                        1 => 1,
                        2 => 25 + rng(&mut state) % 50,
                        _ => 3 + rng(&mut state) % 20,
                    };
                    let quoting = rng(&mut state) % 100;
                    if quoting < 20 || quoting == 97 {
                        text.push(b'"');
                    }
                    for _ in 0..len {
                        text.push(b"abcxyz0189 :-TZ\r"[(rng(&mut state) % 16) as usize]);
                    }
                    if quoting == 96 {
                        text.push(b'"');
                        text.push(b'x');
                    }
                    if quoting < 20 || quoting == 98 {
                        text.push(b'"');
                    }
                }
                text.push(b'\n');
            }
            let body_len = text.len() - 1;
            text.extend_from_slice(&[0u8; 160]);
            let mut expect: Vec<Vec<Vec<u8>>> = Vec::new();
            let mut tmp = Vec::new();
            for line in text[1..body_len].split(|&b| b == b'\n') {
                if split_line(line, fields_n, &mut tmp) {
                    expect.push(tmp.iter().map(|f| f.to_vec()).collect());
                }
            }
            // Walk it in windows of a random size, as run_chunk does.
            let window = 100 + (rng(&mut state) % 3000) as usize;
            let mut got: Vec<Vec<Vec<u8>>> = Vec::new();
            let mut pos = 0usize;
            unsafe {
                while pos < body_len {
                    let wlen = window.min(body_len - pos);
                    let base = text.as_ptr().add(1 + pos);
                    scan.scan(base, wlen);
                    let mut collect = |s: &[u32], e: &[u32]| {
                        got.push(
                            (0..s.len())
                                .map(|i| text[1 + pos + s[i] as usize..1 + pos + e[i] as usize].to_vec())
                                .collect(),
                        );
                    };
                    let used = match fields_n {
                        2 => for_each_line::<2>(&scan, u32::MAX, |s, e| collect(s, e)),
                        4 => for_each_line::<4>(&scan, u32::MAX, |s, e| collect(s, e)),
                        _ => for_each_line::<7>(&scan, u32::MAX, |s, e| collect(s, e)),
                    };
                    if used == 0 {
                        // A line longer than the window: take it whole.
                        let end = text[1 + pos..1 + body_len].iter().position(|&b| b == b'\n').unwrap();
                        if split_line(&text[1 + pos..1 + pos + end], fields_n, &mut tmp) {
                            got.push(tmp.iter().map(|f| f.to_vec()).collect());
                        }
                        pos += end + 1;
                    } else {
                        pos += used as usize;
                    }
                }
            }
            assert_eq!(got, expect, "round {round} fields {fields_n} window {window}");
        }
    }
}
