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
    bits: u32,
}

impl HexTable {
    /// The slot of the value, taking the first free one of its probe run if
    /// the value is new. `None`: the run holds 32 other values, which stays
    /// true from then on.
    #[inline(always)]
    unsafe fn slot_of(&self, hi: u64, lo: u64) -> Option<u32> {
        let mask = (1usize << self.bits) - 1;
        let h = (hi ^ lo.rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let h = (h ^ (h >> 32)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let mut i = (h >> (64 - self.bits)) as usize;
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
            i = (i + 1) & mask;
        }
        None
    }

    #[inline(always)]
    unsafe fn value(&self, slot: u32) -> (u64, u64) {
        let s = &*self.slots.add(slot as usize);
        (s.hi.load(Relaxed), s.lo.load(Acquire))
    }
}

/// The global table: its address with log2 of its slot count in the low
/// bits (mappings are page aligned), 0 until the first long hex ID.
static TABLE: AtomicUsize = AtomicUsize::new(0);
/// log2 of the slots to map; fixed once the table exists.
static WANT_BITS: AtomicUsize = AtomicUsize::new(16);

/// Size the table for an input of this many bytes: about a slot per 128
/// input bytes, several times the long IDs generated data holds. Too small
/// a table is slower, never wrong. No effect once the table exists.
pub fn size_for(input_bytes: u64) {
    let slots = (input_bytes / 128).next_power_of_two().clamp(1 << 12, 1 << 30);
    WANT_BITS.store(slots.trailing_zeros() as usize, Relaxed);
}

#[inline(always)]
fn table() -> HexTable {
    let t = TABLE.load(Acquire);
    let t = if t != 0 { t } else { table_init() };
    HexTable { slots: (t & !63) as *const Slot, bits: (t & 63) as u32 }
}

#[cold]
#[inline(never)]
fn table_init() -> usize {
    let bits = WANT_BITS.load(Relaxed);
    // Fresh anonymous memory is zero: every slot EMPTY. A thread that loses
    // the race leaves its mapping untouched, which costs nothing.
    let fresh = crate::sys::anon(std::mem::size_of::<Slot>() << bits) as usize | bits;
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

/// The key of a long hex ID at `p` (33 readable bytes), or `None` if it is
/// not one or has no slot.
#[inline(always)]
unsafe fn hex_key(p: *const u8) -> Option<u64> {
    let word = |at: usize| u64::from_le((p.add(at) as *const u64).read_unaligned());
    let (a, ba) = hex8(word(1));
    let (b, bb) = hex8(word(9));
    let (c, bc) = hex8(word(17));
    let (d, bd) = hex8(word(25));
    let (hi, lo) = ((a << 32) | b, (c << 32) | d);
    if (ba | bb | bc | bd) != 0 || lo <= BUSY {
        return None;
    }
    let slot = table().slot_of(hi, lo)?;
    Some(FALLBACK | HEX | (u64::from(*p) << 32) | u64::from(slot))
}

/// The key of an ID the 9-byte packing does not take.
#[inline(never)]
unsafe fn long(p: *const u8, len: usize) -> u64 {
    if len == HEX_LEN {
        if let Some(key) = hex_key(p) {
            return key;
        }
    }
    intern(std::slice::from_raw_parts(p, len))
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
        for (bits, values) in [(14u32, 6_000u64), (6, 2_000), (5, 40)] {
            let slots: Vec<Slot> = (0..1usize << bits).map(|_| Slot { lo: AtomicU64::new(0), hi: AtomicU64::new(0) }).collect();
            let addr = slots.as_ptr() as usize;
            let value = |v: u64| {
                let mut s = v.wrapping_mul(0x2545_F491_4F6C_DD1D) + 1;
                (rng(&mut s), rng(&mut s) | 2)
            };
            let seen: Vec<Vec<Option<u32>>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..8u64)
                    .map(|t| {
                        scope.spawn(move || {
                            let table = HexTable { slots: addr as *const Slot, bits };
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
            let table = HexTable { slots: addr as *const Slot, bits };
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
            if bits == 14 {
                assert_eq!(placed, values, "a roomy table places everything");
            } else {
                assert!(placed < values && placed > 0, "a small table refuses some: {placed} of {values}");
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
