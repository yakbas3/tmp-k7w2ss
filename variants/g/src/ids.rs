//! String IDs as exact 64-bit keys.
//!
//! An ID of at most 9 bytes, all in 1..=127, packs into 63 bits: 7 bits per
//! byte, first byte highest, zero padded. That map is injective (no byte is
//! zero, so the padding cannot be confused with content) and order
//! preserving (integer order equals byte-wise order). Every generated ID
//! has this shape, or the long one below. Any other ID (longer, non-ASCII,
//! or holding a NUL) is interned and gets `FALLBACK | address of its copy`;
//! the key spaces cannot collide and all map back to the exact bytes.
//!
//! An ID of exactly 33 bytes whose last 32 are lowercase hex digits (one in
//! eight generated IDs) is a 128-bit value under a first byte. The value
//! lives in one slot of a fixed open-addressing table that threads fill
//! without locks, and the key is `FALLBACK | HEX | first byte | slot`. A
//! slot never changes once written and an ID always lands in the first free
//! slot of its probe run, so equal IDs get equal keys from any thread. An ID
//! whose probe run is full of others, or whose low half is 0 or 1 (the
//! slot's empty and busy marks), goes to the interner instead: a property
//! that never changes for that ID, so the choice is the same every time.

use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Mutex, OnceLock};

use crate::ts::load64;

pub const FALLBACK: u64 = 1 << 63;

// The intern table. A long ID is copied once into memory that is never
// freed or moved, and its key is `FALLBACK | address` of that copy, so a
// key maps back to its bytes with no lock and no lookup. The table that
// finds the copy of a given ID is split by hash into shards, each behind
// its own lock: the parser threads intern millions of long IDs at once and
// one lock would make them queue.

const SHARDS: usize = 1024;
const CHUNK: usize = 64 << 10;
const ADDRESS: u64 = (1 << 48) - 1;

struct Shard {
    /// Open addressing. 0 is empty, else 16 hash bits over a 48-bit address.
    slots: Vec<u64>,
    used: usize,
    free: *mut u8,
    left: usize,
}
unsafe impl Send for Shard {}

static INTERN: [Mutex<Shard>; SHARDS] =
    [const { Mutex::new(Shard { slots: Vec::new(), used: 0, free: std::ptr::null_mut(), left: 0 }) }; SHARDS];

#[inline(always)]
fn fold(a: u64, b: u64) -> u64 {
    let m = u128::from(a) * u128::from(b);
    (m as u64) ^ ((m >> 64) as u64)
}

/// Drawn once a run, so no input can be built to make IDs collide.
fn seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| std::collections::hash_map::RandomState::new().build_hasher().finish() | 1)
}

fn hash(bytes: &[u8]) -> u64 {
    let mut h = seed() ^ bytes.len() as u64;
    let mut words = bytes.chunks_exact(8);
    for w in &mut words {
        h = fold(h ^ u64::from_le_bytes(w.try_into().unwrap()), 0xD6E8_FEB8_6659_FD93);
    }
    let rest = words.remainder();
    let mut tail = [0u8; 8];
    tail[..rest.len()].copy_from_slice(rest);
    fold(h ^ u64::from_le_bytes(tail) ^ 0x2545_F491_4F6C_DD1D, 0xA076_1D64_78BD_642F)
}

/// The bytes of the copy at `p`: a length and the low hash half, then the ID.
#[inline(always)]
unsafe fn record(p: *const u8) -> &'static [u8] {
    unsafe { std::slice::from_raw_parts(p.add(8), *(p as *const u32) as usize) }
}

impl Shard {
    #[cold]
    fn grow(&mut self) {
        let size = (self.slots.len() * 2).max(256);
        let mut slots = vec![0u64; size];
        for &e in &self.slots {
            if e != 0 {
                let low = unsafe { *(((e & ADDRESS) as *const u32).add(1)) };
                let mut i = low as usize & (size - 1);
                while slots[i] != 0 {
                    i = (i + 1) & (size - 1);
                }
                slots[i] = e;
            }
        }
        self.slots = slots;
    }
}

#[inline(never)]
fn intern(bytes: &[u8]) -> u64 {
    let h = hash(bytes);
    let tag = (h >> 32) & 0xFFFF;
    let mut guard = INTERN[(h >> 54) as usize].lock().unwrap();
    let s = &mut *guard;
    if s.used * 2 >= s.slots.len() {
        s.grow();
    }
    let mask = s.slots.len() - 1;
    let mut i = h as u32 as usize & mask;
    loop {
        let e = s.slots[i];
        if e == 0 {
            break;
        }
        if e >> 48 == tag && unsafe { record((e & ADDRESS) as *const u8) } == bytes {
            return FALLBACK | (e & ADDRESS);
        }
        i = (i + 1) & mask;
    }
    let need = (8 + bytes.len() + 7) & !7;
    if s.left < need {
        let size = need.max(CHUNK);
        let block: &'static mut [u64] = Vec::leak(vec![0u64; size / 8]);
        s.free = block.as_mut_ptr() as *mut u8;
        s.left = size;
    }
    let p = s.free;
    assert!(p as u64 <= ADDRESS && bytes.len() <= u32::MAX as usize);
    unsafe {
        *(p as *mut u32) = bytes.len() as u32;
        *(p as *mut u32).add(1) = h as u32;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(8), bytes.len());
        s.free = p.add(need);
    }
    s.left -= need;
    s.slots[i] = tag << 48 | p as u64;
    s.used += 1;
    FALLBACK | p as u64
}

/// Marks a long hex key (with `FALLBACK`): the first byte sits in bits 32
/// to 39 and the slot number in the low 32 bits.
pub const HEX: u64 = 1 << 62;
/// One byte plus 32 hex digits.
pub const HEX_LEN: usize = 33;
/// Slots tried before an ID goes to the interner.
const PROBES: usize = 32;
const EMPTY: u64 = 0;
const BUSY: u64 = 1;

/// `lo` is the slot's state: `EMPTY`, `BUSY` while its writer stores `hi`,
/// then the value's low half for good.
#[repr(C, align(16))]
struct Slot {
    lo: AtomicU64,
    hi: AtomicU64,
}

struct HexTable {
    slots: *const Slot,
    /// Slot count, any value from 1 to `MAX_SLOTS`.
    n: usize,
}

impl HexTable {
    /// The slot of the value, taking the first free one of its probe run if
    /// the value is new. `None`: the run holds 32 other values, which stays
    /// true from then on.
    #[inline(always)]
    unsafe fn slot_of(&self, hi: u64, lo: u64) -> Option<u32> {
        // One wide multiply mixes both halves into all 64 bits; the range
        // reduction then takes the top bits of that. `lo` is at least 2, so
        // the product is 0 only for the one `hi` equal to the constant.
        let h = fold(hi ^ 0x2545_F491_4F6C_DD1D, lo);
        // Range reduction: i < n for any h, with no power of two needed.
        let mut i = ((u128::from(h) * self.n as u128) >> 64) as usize;
        for _ in 0..PROBES {
            let s = &*self.slots.add(i);
            let mut cur = s.lo.load(Acquire);
            loop {
                if cur == EMPTY {
                    match s.lo.compare_exchange(EMPTY, BUSY, AcqRel, Acquire) {
                        Ok(_) => {
                            s.hi.store(hi, Relaxed);
                            s.lo.store(lo, Release);
                            return Some(i as u32);
                        }
                        Err(now) => cur = now,
                    }
                } else if cur == BUSY {
                    std::hint::spin_loop();
                    cur = s.lo.load(Acquire);
                } else {
                    break;
                }
            }
            if cur == lo && s.hi.load(Relaxed) == hi {
                return Some(i as u32);
            }
            i += 1;
            if i == self.n {
                i = 0;
            }
        }
        None
    }

    #[inline(always)]
    unsafe fn value(&self, slot: u32) -> (u64, u64) {
        let s = &*self.slots.add(slot as usize);
        let lo = s.lo.load(Acquire);
        (s.hi.load(Relaxed), lo)
    }
}

/// The global table as one word, so the address and the slot count are
/// published together: the slot count over the page number of the mapping.
/// 0 until the first long hex ID.
static TABLE: AtomicU64 = AtomicU64::new(0);
/// Slots to map; fixed once the table exists.
static WANT_SLOTS: AtomicUsize = AtomicUsize::new(1 << 16);
/// Page numbers below 2^36 cover addresses below 2^48 (every user space
/// address on x86-64 and arm64); that leaves 28 bits for the slot count.
const PAGE_BITS: u32 = 36;
const MAX_SLOTS: usize = (1 << (64 - PAGE_BITS)) - 1;

/// Input bytes per table slot. 190 leaves several slots per long ID that
/// generated data holds; a larger value (2048 once only user IDs use the
/// table) gives a smaller table. Too small a table is slower, never wrong.
const BYTES_PER_SLOT: u64 = 190;

/// Slots for an input of this many bytes: any count, not only powers of two.
fn slots_for(input_bytes: u64, bytes_per_slot: u64) -> usize {
    (input_bytes / bytes_per_slot).clamp(1 << 12, MAX_SLOTS as u64) as usize
}

/// Size the table for an input of this many bytes. No effect once the
/// table exists.
pub fn size_for(input_bytes: u64) {
    WANT_SLOTS.store(slots_for(input_bytes, BYTES_PER_SLOT), Relaxed);
}

#[inline(always)]
fn table() -> HexTable {
    let t = TABLE.load(Acquire);
    decode(if t != 0 { t } else { table_init() })
}

#[inline(always)]
fn decode(t: u64) -> HexTable {
    HexTable { slots: ((t << (64 - PAGE_BITS)) >> (52 - PAGE_BITS)) as *const Slot, n: (t >> PAGE_BITS) as usize }
}

#[cold]
#[inline(never)]
fn table_init() -> u64 {
    let n = WANT_SLOTS.load(Relaxed).clamp(1, MAX_SLOTS);
    // Fresh anonymous memory is zero: every slot EMPTY. A thread that loses
    // the race leaves its mapping untouched, which costs nothing.
    let addr = crate::sys::anon(std::mem::size_of::<Slot>() * n) as u64;
    assert!(addr & 4095 == 0 && addr >> 12 < 1 << PAGE_BITS, "table address {addr:#x}");
    let fresh = (n as u64) << PAGE_BITS | addr >> 12;
    match TABLE.compare_exchange(0, fresh, AcqRel, Acquire) {
        Ok(_) => fresh,
        Err(now) => now,
    }
}

/// Eight bytes (the first in the low byte) as lowercase hex digits: their
/// 32-bit value, and a word that is 0 only if all eight are such digits.
#[inline(always)]
fn hex8(x: u64) -> (u64, u64) {
    const H: u64 = 0x8080_8080_8080_8080;
    // With every byte below 0x80 these sums stay inside their bytes, and
    // the high bit of each says "at least '0'", "above '9'", "at least
    // 'a'", "above 'f'". A byte of 0x80 or more fails through `!x`, and
    // then the carries it may cause do not matter.
    let digit = x.wrapping_add(0x5050_5050_5050_5050) & !x.wrapping_add(0x4646_4646_4646_4646);
    let alpha = x.wrapping_add(0x1F1F_1F1F_1F1F_1F1F) & !x.wrapping_add(0x1919_1919_1919_1919);
    let bad = ((digit | alpha) & !x & H) ^ H;
    let n = (x & 0x0F0F_0F0F_0F0F_0F0F) + ((alpha & H) >> 7) * 9;
    let n = n.swap_bytes();
    let n = (n | (n >> 4)) & 0x00FF_00FF_00FF_00FF;
    let n = (n | (n >> 8)) & 0x0000_FFFF_0000_FFFF;
    ((n | (n >> 16)) & 0xFFFF_FFFF, bad)
}

/// The 128-bit value of the 32 bytes at `p`, or `None` if one of them is
/// not a lowercase hex digit. Portable path: four SWAR words.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
unsafe fn hex_value(p: *const u8) -> Option<(u64, u64)> {
    let word = |at: usize| u64::from_le((p.add(at) as *const u64).read_unaligned());
    let (a, ba) = hex8(word(0));
    let (b, bb) = hex8(word(8));
    let (c, bc) = hex8(word(16));
    let (d, bd) = hex8(word(24));
    if (ba | bb | bc | bd) != 0 {
        return None;
    }
    Some(((a << 32) | b, (c << 32) | d))
}

/// `hex_value` on AVX2: one 32-byte load, a range test per byte, and the
/// nibbles joined by a multiply-add and a pack.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn hex_value(p: *const u8) -> Option<(u64, u64)> {
    use core::arch::x86_64::*;
    let v = _mm256_loadu_si256(p as *const __m256i);
    // Unsigned: a byte is a digit if c - '0' <= 9, a letter if c - 'a' <= 5.
    let d = _mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8));
    let a = _mm256_sub_epi8(v, _mm256_set1_epi8(b'a' as i8));
    let is_digit = _mm256_cmpeq_epi8(_mm256_min_epu8(d, _mm256_set1_epi8(9)), d);
    let is_alpha = _mm256_cmpeq_epi8(_mm256_min_epu8(a, _mm256_set1_epi8(5)), a);
    if _mm256_movemask_epi8(_mm256_or_si256(is_digit, is_alpha)) != -1 {
        return None;
    }
    let nib = _mm256_blendv_epi8(_mm256_add_epi8(a, _mm256_set1_epi8(10)), d, is_digit);
    // Each pair of nibbles, first one high, into one byte of a 16-bit word.
    let pairs = _mm256_maddubs_epi16(nib, _mm256_set1_epi16(0x0110));
    // Bytes 0 to 7 of each lane hold that lane's 8 values in order.
    let bytes = _mm256_packus_epi16(pairs, pairs);
    let joined = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(bytes));
    let first = _mm_cvtsi128_si64(joined) as u64;
    let second = _mm_extract_epi64::<1>(joined) as u64;
    Some((first.swap_bytes(), second.swap_bytes()))
}

/// The key of a long hex ID at `p` (33 readable bytes), or `None` if it is
/// not one or has no slot.
#[inline(always)]
unsafe fn hex_key(p: *const u8, table: HexTable) -> Option<u64> {
    let (hi, lo) = hex_value(p.add(1))?;
    if lo <= BUSY {
        return None;
    }
    let slot = table.slot_of(hi, lo)?;
    Some(FALLBACK | HEX | (u64::from(*p) << 32) | u64::from(slot))
}

/// The key of an ID the 9-byte packing does not take.
#[inline(never)]
unsafe fn long(p: *const u8, len: usize) -> u64 {
    if len == HEX_LEN {
        // Every call out of here is a tail call, so the hot path keeps
        // its values in scratch registers and saves none.
        let t = TABLE.load(Acquire);
        if t == 0 {
            return long_first(p, len);
        }
        if let Some(key) = hex_key(p, decode(t)) {
            return key;
        }
    }
    intern(std::slice::from_raw_parts(p, len))
}

/// `long` for the first long hex ID of the run: map the table, then go on.
#[cold]
#[inline(never)]
unsafe fn long_first(p: *const u8, len: usize) -> u64 {
    table_init();
    long(p, len)
}

/// The 33 bytes of a long hex key.
#[inline(always)]
pub fn unpack_hex(key: u64, out: &mut [u8; HEX_LEN]) {
    debug_assert!(key & FALLBACK != 0 && key & HEX != 0);
    let (hi, lo) = unsafe { table().value(key as u32) };
    out[0] = (key >> 32) as u8;
    for (half, v) in [hi, lo].into_iter().enumerate() {
        for d in 0..16 {
            out[1 + half * 16 + d] = b"0123456789abcdef"[((v >> (60 - 4 * d)) & 15) as usize];
        }
    }
}

const LEN_MASK: [u64; 10] = [
    0,
    0xFF,
    0xFFFF,
    0xFF_FFFF,
    0xFFFF_FFFF,
    0xFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF,
    0xFF_FFFF_FFFF_FFFF,
    u64::MAX,
    u64::MAX,
];

/// The key of the `len` bytes at `p`. May read 16 bytes from `p`.
#[inline(always)]
pub unsafe fn pack(p: *const u8, len: usize) -> u64 {
    if len <= 9 {
        let mask = *LEN_MASK.get_unchecked(len);
        let v = load64(p) & mask;
        let ninth = u64::from(*p.add(8)) & (0u64.wrapping_sub(u64::from(len == 9)));
        // Every byte of the ID must be in 1..=127. Bytes past the ID are
        // set to 1 so they pass; a zero byte borrows and sets its high bit.
        let filled = v | (!mask & 0x0101_0101_0101_0101);
        let bad = (filled.wrapping_sub(0x0101_0101_0101_0101) | filled | ninth) & 0x8080_8080_8080_8080;
        if bad == 0 && (len != 9 || ninth != 0) {
            return (squeeze(v.swap_bytes()) << 7) | ninth;
        }
    }
    long(p, len)
}

/// `pack` for an ID whose bytes the caller already knows to be in 1..=127
/// (stage 1 checks whole windows): the same key without the byte checks.
#[inline(always)]
pub unsafe fn pack_known(p: *const u8, len: usize) -> u64 {
    if len <= 9 {
        // All nine bytes packed, then the key bits past the ID cleared:
        // the byte swap folds into the load (`movbe`), and one mask does
        // the work of the length mask and the ninth byte select. Bytes 1
        // to 8 go through `squeeze`, which drops their high bits; byte 0
        // is the ID's own (in 1..=127) unless `len` is 0, and then the
        // mask is 0.
        let all = squeeze(load64(p.add(1)).swap_bytes()) | (u64::from(*p) << 56);
        return all & *KEY_MASK.get_unchecked(len);
    }
    long(p, len)
}

/// The key bits of the first `len` bytes: 7 a byte from bit 62 down.
const KEY_MASK: [u64; 10] = {
    let mut m = [0u64; 10];
    let mut len = 1;
    while len <= 9 {
        m[len] = ((1u64 << (7 * len)) - 1) << (63 - 7 * len);
        len += 1;
    }
    m
};

/// `pack`, by the short route when `known` says the bytes are in 1..=127.
#[inline(always)]
pub unsafe fn pack_in(p: *const u8, len: usize, known: bool) -> u64 {
    if known {
        pack_known(p, len)
    } else {
        pack(p, len)
    }
}

/// Drop the high bit of each of eight bytes: 64 bits to 56, order kept.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
fn squeeze(x: u64) -> u64 {
    unsafe { core::arch::x86_64::_pext_u64(x, 0x7F7F_7F7F_7F7F_7F7F) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline(always)]
fn squeeze(x: u64) -> u64 {
    let x = ((x & 0x7F00_7F00_7F00_7F00) >> 1) | (x & 0x007F_007F_007F_007F);
    let x = ((x & 0x3FFF_0000_3FFF_0000) >> 2) | (x & 0x0000_3FFF_0000_3FFF);
    ((x & 0x0FFF_FFFF_0000_0000) >> 4) | (x & 0x0000_0000_0FFF_FFFF)
}

/// Safe wrapper for tests and cold paths.
pub fn pack_slice(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 16];
    if bytes.len() <= 9 {
        buf[..bytes.len()].copy_from_slice(bytes);
        unsafe { pack(buf.as_ptr(), bytes.len()) }
    } else {
        unsafe { long(bytes.as_ptr(), bytes.len()) }
    }
}

/// Bytes of a packed (not fallback) key into `out[..9]`; returns the length.
#[inline(always)]
pub fn unpack_packed(key: u64, out: &mut [u8; 16]) -> usize {
    debug_assert!(key & FALLBACK == 0);
    let ninth = (key & 0x7F) as u8;
    let x = key >> 7;
    let x = ((x & 0x00FF_FFFF_F000_0000) << 4) | (x & 0x0000_0000_0FFF_FFFF);
    let x = ((x & 0x0FFF_C000_0FFF_C000) << 2) | (x & 0x0000_3FFF_0000_3FFF);
    let x = ((x & 0x3F80_3F80_3F80_3F80) << 1) | (x & 0x007F_007F_007F_007F);
    out[..8].copy_from_slice(&x.swap_bytes().to_le_bytes());
    out[8] = ninth;
    if key == 0 {
        0
    } else {
        9 - (key.trailing_zeros() / 7) as usize
    }
}

/// The exact bytes of any key. `buf` holds them unless the key is interned.
#[inline(always)]
fn key_bytes(key: u64, buf: &mut [u8; HEX_LEN]) -> &[u8] {
    if key & FALLBACK == 0 {
        let mut short = [0u8; 16];
        let len = unpack_packed(key, &mut short);
        buf[..9].copy_from_slice(&short[..9]);
        &buf[..len]
    } else if key & HEX != 0 {
        unpack_hex(key, buf);
        &buf[..]
    } else {
        unsafe { record((key & ADDRESS) as *const u8) }
    }
}

/// Append the exact bytes of any key.
pub fn unpack(key: u64, out: &mut Vec<u8>) {
    let mut buf = [0u8; HEX_LEN];
    out.extend_from_slice(key_bytes(key, &mut buf));
}

#[inline(never)]
fn cmp_slow(a: u64, b: u64) -> Ordering {
    const BOTH: u64 = FALLBACK | HEX;
    if a & b & BOTH == BOTH {
        // Two hex IDs: same length, and 32 hex digits in a fixed width sort
        // as their value, so (first byte, hi, lo) is the byte-wise order.
        let t = table();
        let (x, y) = unsafe { (t.value(a as u32), t.value(b as u32)) };
        return ((a >> 32) as u8, x.0, x.1).cmp(&((b >> 32) as u8, y.0, y.1));
    }
    let (mut x, mut y) = ([0u8; HEX_LEN], [0u8; HEX_LEN]);
    key_bytes(a, &mut x).cmp(key_bytes(b, &mut y))
}

/// Byte-wise order of the IDs two keys stand for.
#[inline(always)]
pub fn cmp_ids(a: u64, b: u64) -> Ordering {
    if (a | b) & FALLBACK == 0 {
        a.cmp(&b)
    } else if a == b {
        Ordering::Equal
    } else {
        cmp_slow(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn sample(state: &mut u64) -> Vec<u8> {
        let len = (rng(state) % 13) as usize;
        let style = rng(state) % 8;
        (0..len)
            .map(|_| match style {
                0 => (rng(state) % 256) as u8,
                1 => b"\x00\x01\x7f\x80\xff a"[(rng(state) % 7) as usize],
                2 => b"uU"[(rng(state) % 2) as usize],
                _ => b"0123456789abcdefghijklmnopqrstuvwxyzAZ _\t\r"[(rng(state) % 42) as usize],
            })
            .collect()
    }

    #[test]
    fn round_trip_and_order() {
        let mut state = 3;
        let mut ids: Vec<Vec<u8>> = [
            "", "u", "u1", "U1", " u1", "u1 ", "uduffvac", "e42huhoug", "e42huhou", "e42huhougx", "A\tB", "e3\r",
            "user_id_a", "user_id_b", "\u{ff12}", "zzzzzzzzz", "zzzzzzzzzz", "\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f",
            "\x01", "\x01\x01",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
        ids.push(vec![b'a', 0]);
        ids.push(vec![0]);
        ids.push(vec![b'a', 0, b'b']);
        ids.push(vec![b'a'; 9]);
        ids.push(vec![b'a', b'a', b'a', b'a', b'a', b'a', b'a', b'a', 0]);
        ids.push(vec![b'a', b'a', b'a', b'a', b'a', b'a', b'a', b'a', 0x80]);
        for _ in 0..20_000 {
            ids.push(sample(&mut state));
        }
        let keys: Vec<u64> = ids.iter().map(|id| pack_slice(id)).collect();
        for (id, &key) in ids.iter().zip(&keys) {
            let mut back = Vec::new();
            unpack(key, &mut back);
            assert_eq!(&back, id, "round trip of {id:?}");
            assert_eq!(pack_slice(id), key, "stable key for {id:?}");
            let packable = id.len() <= 9 && id.iter().all(|&b| (1..=127).contains(&b));
            assert_eq!(key & FALLBACK == 0, packable, "{id:?}");
        }
        for i in 0..ids.len() {
            for j in (0..ids.len()).step_by(37) {
                assert_eq!(keys[i] == keys[j], ids[i] == ids[j], "{:?} {:?}", ids[i], ids[j]);
                assert_eq!(cmp_ids(keys[i], keys[j]), ids[i].cmp(&ids[j]), "{:?} {:?}", ids[i], ids[j]);
            }
        }
    }

    fn hex_id(first: u8, hi: u64, lo: u64) -> Vec<u8> {
        let mut id = vec![first];
        id.extend_from_slice(format!("{hi:016x}{lo:016x}").as_bytes());
        id
    }

    #[test]
    fn hex8_agrees_with_a_table() {
        let mut state = 11;
        let pool = b"0123456789abcdefABCDEFg`/:@G \x00\x7f\x80\xff\xe6";
        for round in 0..400_000 {
            let mut bytes = [0u8; 8];
            for b in &mut bytes {
                *b = if round % 3 == 0 { (rng(&mut state) % 256) as u8 } else { pool[(rng(&mut state) % pool.len() as u64) as usize] };
            }
            if round % 5 != 0 {
                // Mostly valid, with at most one odd byte.
                for b in &mut bytes {
                    *b = b"0123456789abcdef"[(rng(&mut state) % 16) as usize];
                }
                if round % 2 == 0 {
                    bytes[(rng(&mut state) % 8) as usize] = (rng(&mut state) % 256) as u8;
                }
            }
            let want = std::str::from_utf8(&bytes)
                .ok()
                .filter(|s| s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)))
                .map(|s| u64::from_str_radix(s, 16).unwrap());
            let (value, bad) = hex8(u64::from_le_bytes(bytes));
            assert_eq!(bad == 0, want.is_some(), "{bytes:?}");
            if let Some(w) = want {
                assert_eq!(value, w, "{bytes:?}");
            }
        }
    }

    #[test]
    fn hex_value_agrees_with_a_table() {
        let mut state = 13;
        let pool = b"0123456789abcdefABCDEFg`/:@G \x00\x7f\x80\xff\xe6";
        for round in 0..200_000 {
            let mut bytes = [0u8; 32];
            for b in &mut bytes {
                *b = b"0123456789abcdef"[(rng(&mut state) % 16) as usize];
            }
            match round % 4 {
                0 => {}
                1 => bytes[(rng(&mut state) % 32) as usize] = pool[(rng(&mut state) % pool.len() as u64) as usize],
                2 => bytes[(rng(&mut state) % 32) as usize] = (rng(&mut state) % 256) as u8,
                _ => {
                    for b in &mut bytes {
                        *b = pool[(rng(&mut state) % pool.len() as u64) as usize];
                    }
                }
            }
            let want = std::str::from_utf8(&bytes)
                .ok()
                .filter(|s| s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)))
                .map(|s| (u64::from_str_radix(&s[..16], 16).unwrap(), u64::from_str_radix(&s[16..], 16).unwrap()));
            assert_eq!(unsafe { hex_value(bytes.as_ptr()) }, want, "{bytes:?}");
        }
    }

    #[test]
    fn long_hex_ids_round_trip_and_order() {
        let mut state = 5;
        let mut ids: Vec<Vec<u8>> = Vec::new();
        for i in 0..3_000u64 {
            let (hi, lo) = (rng(&mut state), rng(&mut state));
            ids.push(hex_id(b"uec"[(i % 3) as usize], hi, lo));
            if i % 7 == 0 {
                // Same value under another first byte; halves swapped or shared.
                ids.push(hex_id(b'U', hi, lo));
                ids.push(hex_id(b'u', lo, hi));
                ids.push(hex_id(b'u', hi, rng(&mut state)));
                ids.push(hex_id(b'u', rng(&mut state), lo));
            }
            if i % 11 == 0 {
                // Near misses: one byte off the shape, a byte short or long.
                let mut odd = hex_id(b'u', hi, lo);
                let at = 1 + (rng(&mut state) % 32) as usize;
                odd[at] = b"AFg/:`\x00\x80 "[(rng(&mut state) % 9) as usize];
                ids.push(odd);
                ids.push(hex_id(b'u', hi, lo)[..32].to_vec());
                let mut more = hex_id(b'u', hi, lo);
                more.push(b'0');
                ids.push(more);
            }
        }
        // The slot's own marks as low halves, a zero value, a NUL first byte.
        for (hi, lo) in [(0, 0), (0, 1), (7, 0), (7, 1), (0, 2), (u64::MAX, u64::MAX)] {
            ids.push(hex_id(b'u', hi, lo));
        }
        ids.push(hex_id(0, 9, 9));
        ids.push(hex_id(0xFF, 9, 9));
        ids.extend(["u", "u1", "uzzzzzzzz", "e42huhoug", "U1", "v"].iter().map(|s| s.as_bytes().to_vec()));
        let keys: Vec<u64> = ids.iter().map(|id| pack_slice(id)).collect();
        for (id, &key) in ids.iter().zip(&keys) {
            let mut back = Vec::new();
            unpack(key, &mut back);
            assert_eq!(&back, id, "round trip");
            assert_eq!(pack_slice(id), key, "stable key");
            let mut padded = id.clone();
            padded.extend_from_slice(b",\"\n0123456789abc");
            unsafe {
                assert_eq!(pack(padded.as_ptr(), id.len()), key, "pack");
                if id.iter().all(|&c| (1..=127).contains(&c)) {
                    assert_eq!(pack_known(padded.as_ptr(), id.len()), key, "pack_known");
                }
            }
            let shaped = id.len() == HEX_LEN && id[1..].iter().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c));
            let low_mark = shaped && id[17..32].iter().all(|&c| c == b'0') && id[32] <= b'1';
            assert_eq!(key & HEX != 0 && key & FALLBACK != 0, shaped && !low_mark, "{id:?}");
        }
        for i in 0..ids.len() {
            for j in (i % 13..ids.len()).step_by(13) {
                assert_eq!(keys[i] == keys[j], ids[i] == ids[j], "{:?} {:?}", ids[i], ids[j]);
                assert_eq!(cmp_ids(keys[i], keys[j]), ids[i].cmp(&ids[j]), "{:?} {:?}", ids[i], ids[j]);
            }
        }
    }

    /// Threads that meet the same IDs in different orders agree on every
    /// key, in a roomy table and in one small enough that probe runs fill
    /// up and most values are refused (the engine then interns them).
    #[test]
    fn hex_table_gives_one_slot_a_value_from_any_thread() {
        // Roomy, then sizes that are no power of two, small enough that
        // probe runs wrap past the end and fill up.
        // Then the engine's own sizes at both slot densities, roomy and
        // overflowing into the interner.
        let mut sizes = vec![(16_384usize, 6_000u64), (12_289, 4_000), (4_099, 3_000), (97, 2_000), (37, 40), (1, 5)];
        for (bytes, per, values) in [(30_000_000, 190, 60_000), (3_000_000, 190, 30_000), (30_000_000, 2048, 20_000), (300_000, 2048, 9_000)] {
            let n = slots_for(bytes, per);
            assert!(n >= 4096 && n <= MAX_SLOTS && n as u64 == (bytes / per).max(4096));
            sizes.push((n, values));
        }
        for (n, values) in sizes {
            let slots: Vec<Slot> = (0..n).map(|_| Slot { lo: AtomicU64::new(0), hi: AtomicU64::new(0) }).collect();
            let addr = slots.as_ptr() as usize;
            let value = |v: u64| {
                let mut s = v.wrapping_mul(0x2545_F491_4F6C_DD1D) + 1;
                (rng(&mut s), rng(&mut s) | 2)
            };
            let seen: Vec<Vec<Option<u32>>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..8u64)
                    .map(|t| {
                        scope.spawn(move || {
                            let table = HexTable { slots: addr as *const Slot, n };
                            let mut got = vec![None; values as usize];
                            let mut state = t + 1;
                            for _ in 0..values * 6 {
                                let v = rng(&mut state) % values;
                                let (hi, lo) = value(v);
                                let slot = unsafe { table.slot_of(hi, lo) };
                                if let Some(first) = got[v as usize].replace(slot) {
                                    assert_eq!(first, slot, "a value kept its answer");
                                }
                            }
                            for v in 0..values {
                                let (hi, lo) = value(v);
                                got[v as usize] = Some(unsafe { table.slot_of(hi, lo) });
                            }
                            got.into_iter().map(|g| g.unwrap()).collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let table = HexTable { slots: addr as *const Slot, n };
            let mut owner = std::collections::HashMap::new();
            let mut placed = 0;
            for v in 0..values {
                for t in 1..8 {
                    assert_eq!(seen[t][v as usize], seen[0][v as usize], "threads agree on value {v}");
                }
                if let Some(slot) = seen[0][v as usize] {
                    placed += 1;
                    assert_eq!(unsafe { table.value(slot) }, value(v));
                    assert!(owner.insert(slot, v).is_none(), "one value a slot");
                }
            }
            // A run that starts in the last 31 slots goes on at slot 0.
            let wrapped = owner.keys().filter(|&&slot| (slot as usize) < PROBES).count();
            if values * 2 <= n as u64 {
                assert_eq!(placed, values, "a roomy table places everything");
            } else {
                assert!(placed < values && placed > 0, "a small table refuses some: {placed} of {values}");
                if n <= 97 {
                    assert_eq!(placed, n as u64, "a full table: every slot taken");
                }
            }
            if n == 97 {
                assert!(wrapped > 0, "runs wrap at the end");
            }
        }
    }

    #[test]
    fn pack_ignores_bytes_past_the_field() {
        let mut state = 9;
        for _ in 0..50_000 {
            let id = sample(&mut state);
            if id.len() > 9 {
                continue;
            }
            let mut a = id.clone();
            let mut b = id.clone();
            a.extend_from_slice(&[0xFF; 16]);
            b.extend_from_slice(b",\"\n0123456789abc");
            unsafe {
                assert_eq!(pack(a.as_ptr(), id.len()), pack(b.as_ptr(), id.len()), "{id:?}");
                // The short route gives the same key for every ID it may
                // be used on.
                if id.iter().all(|&c| (1..=127).contains(&c)) {
                    assert_eq!(pack_known(a.as_ptr(), id.len()), pack(a.as_ptr(), id.len()), "{id:?}");
                    assert_eq!(pack_known(b.as_ptr(), id.len()), pack(a.as_ptr(), id.len()), "{id:?}");
                    assert_eq!(pack_in(b.as_ptr(), id.len(), true), pack_in(b.as_ptr(), id.len(), false), "{id:?}");
                }
            }
        }
    }
}
