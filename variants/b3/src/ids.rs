//! String IDs as exact 64-bit keys.
//!
//! An ID of at most 9 bytes, all in 1..=127, packs into 63 bits: 7 bits per
//! byte, first byte highest, zero padded. That map is injective (no byte is
//! zero, so the padding cannot be confused with content) and order
//! preserving (integer order equals byte-wise order). Every generated ID
//! used to have this shape. Any other ID (longer, non-ASCII, or holding a
//! NUL) is interned and gets `FALLBACK | address of its copy`; the two key
//! spaces cannot collide and both map back to the exact bytes.
//!
//! Event IDs are the exception, see "event IDs" below: a long one is not
//! interned, so equal IDs can have different keys, and event keys are
//! compared with `same_event` and hashed through `event_hash_part`.

use std::cmp::Ordering;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering as Atomic};
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
    intern(std::slice::from_raw_parts(p, len))
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
    intern(std::slice::from_raw_parts(p, len))
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
// ---- event IDs
//
// An event ID is only ever compared for equality with another event ID, and
// nearly every one is new. So a long one takes no lock and no table: its
// bytes are copied to a block the parsing thread owns, and the key names
// the copy. Two copies of one ID get two different keys, so event keys are
// compared with `same_event`, never with `==`, and hashed by
// `event_hash_part`.
//
// A long event key: bit 63, then 31 bits of the ID's hash, then the place
// of the copy: 15 bits of block number and 17 bits of offset in 8-byte
// units. A copy is its length as a u64 followed by the bytes.

/// Bytes of one block of event ID copies.
const EVENT_BLOCK: usize = 1 << 20;
const EVENT_OFF_BITS: u32 = 17;
const EVENT_BLOCKS: usize = 1 << 15;
const _: () = assert!(EVENT_BLOCK >> 3 == 1 << EVENT_OFF_BITS);

static EVENT_DIR: [AtomicPtr<u8>; EVENT_BLOCKS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; EVENT_BLOCKS];
static EVENT_NEXT: AtomicUsize = AtomicUsize::new(0);

/// Tests cut the hash of a long event ID down with this mask, to force
/// different IDs onto equal hash bits. Keys made under two masks do not
/// mix, so a test that changes it holds `EVENT_MASK_GATE` alone and every
/// other test that makes event keys holds it shared.
#[cfg(test)]
static EVENT_HASH_MASK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
#[cfg(test)]
pub static EVENT_MASK_GATE: std::sync::RwLock<()> = std::sync::RwLock::new(());

#[cfg(test)]
pub fn set_event_hash_mask(mask: u64) {
    EVENT_HASH_MASK.store(mask, Atomic::Relaxed);
}

#[cfg(test)]
#[inline(always)]
fn event_hash(bytes: &[u8]) -> u64 {
    hash(bytes) & EVENT_HASH_MASK.load(Atomic::Relaxed)
}

#[cfg(not(test))]
#[inline(always)]
fn event_hash(bytes: &[u8]) -> u64 {
    hash(bytes)
}

/// The block this thread is filling: its number, base, bytes used and size.
#[derive(Clone, Copy)]
struct EventBlock {
    id: usize,
    base: *mut u8,
    used: usize,
    size: usize,
}

// A block is plain memory that is never freed, and one thread fills it at a
// time.
unsafe impl Send for EventBlock {}

/// The block of a thread. When the thread ends, what is left of the block
/// goes to `EVENT_SPARE` for the next thread that needs one, so a process
/// that starts threads again and again (the tests) does not use a block,
/// and a place in the directory, per thread.
struct EventHere(std::cell::Cell<EventBlock>);

impl Drop for EventHere {
    fn drop(&mut self) {
        let b = self.0.get();
        if b.used < b.size.min(EVENT_BLOCK) {
            EVENT_SPARE.lock().unwrap_or_else(|e| e.into_inner()).push(b);
        }
    }
}

static EVENT_SPARE: Mutex<Vec<EventBlock>> = Mutex::new(Vec::new());

thread_local! {
    static EVENT_HERE: EventHere =
        const { EventHere(std::cell::Cell::new(EventBlock { id: 0, base: std::ptr::null_mut(), used: 0, size: 0 })) };
}

/// The key of an event ID. See the note above: compare with `same_event`.
///
/// # Safety
/// As `pack_in`.
#[inline(always)]
pub unsafe fn event_key(p: *const u8, len: usize, known: bool) -> u64 {
    if len <= 9 {
        if known {
            return pack_known(p, len);
        }
        // The same test as in `pack`.
        let mask = *LEN_MASK.get_unchecked(len);
        let v = load64(p) & mask;
        let ninth = u64::from(*p.add(8)) & (0u64.wrapping_sub(u64::from(len == 9)));
        let filled = v | (!mask & 0x0101_0101_0101_0101);
        let bad = (filled.wrapping_sub(0x0101_0101_0101_0101) | filled | ninth) & 0x8080_8080_8080_8080;
        if bad == 0 && (len != 9 || ninth != 0) {
            return (squeeze(v.swap_bytes()) << 7) | ninth;
        }
    }
    event_long(std::slice::from_raw_parts(p, len))
}

/// `event_key` of a slice with nothing readable past its end.
pub fn event_key_slice(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 16];
    if bytes.len() <= 9 {
        buf[..bytes.len()].copy_from_slice(bytes);
        unsafe { event_key(buf.as_ptr(), bytes.len(), false) }
    } else {
        event_long(bytes)
    }
}

/// Copy a long event ID to this thread's block and name the copy.
#[inline(never)]
fn event_long(bytes: &[u8]) -> u64 {
    let h = event_hash(bytes);
    let need = (8 + bytes.len() + 7) & !7;
    EVENT_HERE.with(|here| {
        let here = &here.0;
        let mut b = here.get();
        if b.used + need > b.size {
            b = event_block(need);
        }
        // Only the first megabyte of a block can be named, and a copy
        // starts there: a larger block holds one copy, at offset 0.
        debug_assert!(b.used < EVENT_BLOCK);
        unsafe {
            let rec = b.base.add(b.used);
            (rec as *mut u64).write(bytes.len() as u64);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), rec.add(8), bytes.len());
        }
        let key = FALLBACK | ((h >> 33) << 32) | ((b.id as u64) << EVENT_OFF_BITS) | (b.used as u64 >> 3);
        b.used += need;
        here.set(b);
        key
    })
}

/// A block with room for `need` bytes: one that an ended thread left, or a
/// new one. An ID too large for a block gets one of its own size, and the
/// next ID starts another.
#[cold]
#[inline(never)]
fn event_block(need: usize) -> EventBlock {
    {
        let mut spare = EVENT_SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(at) = spare.iter().position(|b| b.used + need <= b.size) {
            return spare.swap_remove(at);
        }
    }
    let size = need.max(EVENT_BLOCK);
    let id = EVENT_NEXT.fetch_add(1, Atomic::Relaxed);
    assert!(id < EVENT_BLOCKS, "too many long event IDs");
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    EVENT_DIR[id].store(base, Atomic::Release);
    EventBlock { id, base, used: 0, size }
}

/// The bytes a long event key names.
unsafe fn event_bytes(key: u64) -> &'static [u8] {
    let place = key as u32;
    let base = EVENT_DIR[(place >> EVENT_OFF_BITS) as usize].load(Atomic::Acquire);
    let rec = base.add(((place & ((1 << EVENT_OFF_BITS) - 1)) as usize) << 3);
    let len = (rec as *const u64).read() as usize;
    std::slice::from_raw_parts(rec.add(8), len)
}

/// Two long event keys with equal hash bits: are the IDs equal?
#[cold]
#[inline(never)]
fn event_long_eq(a: u64, b: u64) -> bool {
    unsafe { event_bytes(a) == event_bytes(b) }
}

/// Whether two event keys are keys of the same event ID. Both must come
/// from `event_key` or `event_key_slice`: a long key names a copy.
#[inline(always)]
pub fn same_event(a: u64, b: u64) -> bool {
    // Equal keys are one ID: a short one packed, or one copy of a long one.
    // Otherwise only two long keys with the same hash bits can be.
    a == b || ((a ^ b) >> 32 == 0 && a & FALLBACK != 0 && event_long_eq(a, b))
}

/// What to hash of an event key, so that equal IDs hash equal: a short key
/// whole, a long key without the place of its copy.
#[inline(always)]
pub fn event_hash_part(key: u64) -> u64 {
    let long = ((key as i64) >> 63) as u64;
    key & !(long & 0xFFFF_FFFF)
}

pub fn pack_slice(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 16];
    if bytes.len() <= 9 {
        buf[..bytes.len()].copy_from_slice(bytes);
        unsafe { pack(buf.as_ptr(), bytes.len()) }
    } else {
        intern(bytes)
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

/// The exact bytes of any key. `buf` holds them when the key is packed.
#[inline(always)]
fn key_bytes(key: u64, buf: &mut [u8; 16]) -> &[u8] {
    if key & FALLBACK == 0 {
        let len = unpack_packed(key, buf);
        &buf[..len]
    } else {
        unsafe { record((key & ADDRESS) as *const u8) }
    }
}

/// Append the exact bytes of any key.
pub fn unpack(key: u64, out: &mut Vec<u8>) {
    let mut buf = [0u8; 16];
    out.extend_from_slice(key_bytes(key, &mut buf));
}

#[inline(never)]
fn cmp_slow(a: u64, b: u64) -> Ordering {
    let (mut x, mut y) = ([0u8; 16], [0u8; 16]);
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

    fn gate() -> std::sync::RwLockReadGuard<'static, ()> {
        EVENT_MASK_GATE.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Short event IDs get the packed key. Long ones get a key per copy:
    /// equal IDs are the same event and hash equal, whichever thread made
    /// the key, and no two different IDs are the same event.
    #[test]
    fn event_keys_are_ids() {
        let _gate = gate();
        let mut state = 11;
        let mut ids: Vec<Vec<u8>> = (0..3000).map(|_| sample(&mut state)).collect();
        // Long IDs: a shared prefix with one byte changed, a prefix of
        // another, lengths around the 8 byte steps of a copy, and bytes of
        // every value.
        for len in [10usize, 15, 16, 17, 23, 24, 25, 32, 33, 34, 64, 300, 4096] {
            let base: Vec<u8> = (0..len).map(|i| b'a' + (i % 23) as u8).collect();
            for at in [0, len / 2, len - 1] {
                let mut other = base.clone();
                other[at] ^= 1;
                ids.push(other);
            }
            ids.push(base[..len - 1].to_vec());
            let mut more = base.clone();
            more.push(0);
            ids.push(more);
            ids.push((0..len).map(|i| (rng(&mut state) % 256) as u8 | u8::from(i == 0)).collect());
            ids.push(base);
        }
        ids.sort();
        ids.dedup();
        let keys: Vec<u64> = ids.iter().map(|id| event_key_slice(id)).collect();
        // A second key of every ID, made on another thread.
        let again: Vec<u64> = std::thread::scope(|s| s.spawn(|| ids.iter().map(|id| event_key_slice(id)).collect()).join().unwrap());
        let mut long = 0;
        for (i, id) in ids.iter().enumerate() {
            let short = id.len() <= 9 && id.iter().all(|&c| (1..=127).contains(&c));
            if short {
                assert_eq!(keys[i], pack_slice(id), "{id:?}");
                assert_eq!(keys[i], again[i]);
                assert_eq!(keys[i] & FALLBACK, 0);
            } else {
                long += 1;
                assert_ne!(keys[i], again[i], "one copy for two keys: {id:?}");
                assert_ne!(keys[i] & FALLBACK, 0);
                assert_eq!(unsafe { event_bytes(keys[i]) }, &id[..]);
                assert_eq!(unsafe { event_bytes(again[i]) }, &id[..]);
            }
            assert!(same_event(keys[i], again[i]) && same_event(again[i], keys[i]), "{id:?}");
            assert_eq!(event_hash_part(keys[i]), event_hash_part(again[i]), "{id:?}");
            // The pointer form, with bytes after the field.
            let mut buf = id.clone();
            buf.extend_from_slice(b",tail of the line, sixteen bytes and more");
            let k = unsafe { event_key(buf.as_ptr(), id.len(), false) };
            assert!(same_event(k, keys[i]), "{id:?}");
            assert_eq!(event_hash_part(k), event_hash_part(keys[i]));
            if short && !id.contains(&b'"') && !id.contains(&b',') {
                assert_eq!(unsafe { event_key(buf.as_ptr(), id.len(), true) }, keys[i]);
            }
        }
        assert!(long > 500, "only {long} long IDs");
        for i in 0..ids.len() {
            for j in 0..ids.len() {
                assert_eq!(same_event(keys[i], again[j]), i == j, "{:?} {:?}", ids[i], ids[j]);
            }
        }
    }

    /// The copies of long event IDs through many blocks, an ID larger than
    /// a block between small ones, on several threads at once.
    #[test]
    fn event_copies_through_block_changes() {
        let _gate = gate();
        let name = |t: usize, i: usize| -> Vec<u8> {
            let mut id = format!("thread-{t}-event-{i}-").into_bytes();
            id.resize(40 + (i * 7 + t) % 400, b'x');
            id
        };
        std::thread::scope(|s| {
            for t in 0..6 {
                s.spawn(move || {
                    // About 3 MB of copies a thread, and two of 1.5 MB.
                    let keys: Vec<u64> = (0..12_000)
                        .map(|i| {
                            if i == 4000 || i == 4001 {
                                let mut big = name(t, i);
                                big.resize(EVENT_BLOCK + (EVENT_BLOCK >> 1) + i, b'y');
                                return event_key_slice(&big);
                            }
                            event_key_slice(&name(t, i))
                        })
                        .collect();
                    for (i, &k) in keys.iter().enumerate() {
                        let bytes = unsafe { event_bytes(k) };
                        if i == 4000 || i == 4001 {
                            assert_eq!(bytes.len(), EVENT_BLOCK + (EVENT_BLOCK >> 1) + i);
                            assert!(bytes.starts_with(&name(t, i)) && bytes.ends_with(b"yyyy"));
                        } else {
                            assert_eq!(bytes, &name(t, i)[..], "thread {t}, ID {i}");
                        }
                        assert!(same_event(k, event_key_slice(bytes)));
                    }
                    assert!(!same_event(keys[4000], keys[4001]));
                    let blocks: std::collections::HashSet<u32> = keys.iter().map(|&k| (k as u32) >> EVENT_OFF_BITS).collect();
                    assert!(blocks.len() >= 5, "only {} blocks", blocks.len());
                });
            }
        });
    }

    /// With the hash cut to nothing or to two bits, different long IDs
    /// share their hash bits and are still told apart.
    #[test]
    fn event_keys_with_equal_hash_bits() {
        let _alone = EVENT_MASK_GATE.write().unwrap_or_else(|e| e.into_inner());
        for mask in [0u64, 3 << 33, 1 << 63] {
            set_event_hash_mask(mask);
            let ids: Vec<Vec<u8>> = (0..400).map(|i| format!("event-{i:05}-{}", "z".repeat(i % 50)).into_bytes()).collect();
            let a: Vec<u64> = ids.iter().map(|id| event_key_slice(id)).collect();
            let b: Vec<u64> = std::thread::scope(|s| s.spawn(|| ids.iter().map(|id| event_key_slice(id)).collect()).join().unwrap());
            let mut shared = 0;
            for i in 0..ids.len() {
                for j in 0..ids.len() {
                    assert_eq!(same_event(a[i], b[j]), i == j, "mask {mask:x}");
                    shared += usize::from(i != j && event_hash_part(a[i]) == event_hash_part(b[j]));
                }
            }
            assert!(shared > 30_000, "mask {mask:x}: only {shared} pairs share their hash bits");
        }
        set_event_hash_mask(u64::MAX);
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

    /// Eight threads intern the same two million long IDs at once, each in
    /// its own order. Every shard grows its table several times and fills
    /// more than one chunk on the way.
    #[test]
    fn threads_agree_through_growth_and_chunk_changes() {
        const N: usize = 2_000_000;
        const THREADS: usize = 8;
        fn id(i: usize) -> Vec<u8> {
            let (mut a, mut b) = (i as u64, !(i as u64));
            let (x, y) = (rng(&mut a), rng(&mut b));
            format!("u{x:016x}{y:016x}").into_bytes()
        }
        let keys: Vec<Vec<u64>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..THREADS)
                .map(|t| {
                    scope.spawn(move || {
                        let mut keys = vec![0u64; N];
                        for step in 0..N {
                            let i = (t * (N / THREADS) + step) % N;
                            keys[i] = pack_slice(&id(i));
                        }
                        keys
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).collect()
        });
        for t in 1..THREADS {
            assert!(keys[t] == keys[0], "thread {t} got other keys than thread 0");
        }
        let keys = &keys[0];
        let mut distinct = keys.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), N, "two IDs share a key");
        {
            let shard = INTERN[0].lock().unwrap();
            assert!(shard.slots.len() >= 2048, "table of shard 0 grew to {}", shard.slots.len());
            assert!(shard.used * 48 > CHUNK, "shard 0 holds {} IDs, one chunk", shard.used);
        }
        let mut back = Vec::new();
        for (i, &key) in keys.iter().enumerate() {
            assert!(key & FALLBACK != 0);
            back.clear();
            unpack(key, &mut back);
            assert_eq!(back, id(i), "round trip of ID {i}");
        }
        for i in (0..N).step_by(997) {
            let j = (i * 31 + 7) % N;
            assert_eq!(cmp_ids(keys[i], keys[j]), id(i).cmp(&id(j)), "order of {i} and {j}");
        }
        // IDs around the size of a chunk, and far over it.
        for len in [CHUNK - 16, CHUNK - 9, CHUNK - 8, CHUNK - 7, CHUNK, 70_000, 200_000] {
            let big: Vec<u8> = (0..len).map(|n| b'a' + (n % 23) as u8).collect();
            let key = pack_slice(&big);
            assert_eq!(pack_slice(&big), key, "stable key at {len} bytes");
            back.clear();
            unpack(key, &mut back);
            assert_eq!(back, big, "round trip at {len} bytes");
        }
        for i in (0..N).step_by(101) {
            assert_eq!(pack_slice(&id(i)), keys[i], "key of ID {i} after the big ones");
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
