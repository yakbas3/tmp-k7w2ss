//! The fast engine. Three parallel phases, each a pool of threads pulling
//! jobs off an atomic counter:
//!
//! 1. parse: event and click chunks are scanned and parsed; every
//!    well-formed event copy becomes a 32-byte entry scattered by a hash of
//!    its event key. One job meanwhile reads all links and builds the
//!    user -> person map (union-find).
//! 2. resolve: per event partition, dedup on the exact event key keeping the
//!    smallest ingest, then map the survivors' users to persons and scatter
//!    16-byte records by person. Clicks get the same person scatter.
//! 3. person: per person partition, group by person, sort by ts, sessionize,
//!    attribute clicks, aggregate, and write user_daily row groups.
//!
//! The one-pass shape (`ENGINE_ONEPASS=1`, off by default) replaces 1 and 2:
//! the links are joined first, on all threads; the parsers then look every
//! event's person up at once and write a valid copy straight to its person
//! partition as a 16-byte record, and leave a 16-byte note (event key,
//! ingest, record name) for every copy in its event-key partition. Dedup
//! walks only the notes and marks the record of every losing copy dropped
//! in place. Clicks go straight to their person partitions too.

use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::common::*;
use crate::ids::{self, cmp_ids};
use crate::output::{write_camp_cols, CampCols};
use crate::usercols::UserOut;
use crate::fuse;
use crate::scan::{for_each_line, for_each_line_b, Bounds, Scan};
use crate::sys::{self, prefetch, prof, List, Mapped, Scatter};
use crate::ts::{canonical_number, load32, load64, parse_price, parse_ts, parse_ts_ingest, DateCache, BAD};

mod dense;

pub struct Config {
    pub threads: usize,
    pub timing: bool,
    /// Collect user rows in memory instead of writing files (tests).
    pub collect: bool,
    pub chunk_bytes: usize,
    pub p1_bits: Option<u32>,
    pub p2_bits: Option<u32>,
    /// The one-pass shape (see the top of this file).
    pub onepass: bool,
    /// The dense shape (see `dense.rs`); wins over `onepass`.
    pub dense: bool,
    /// Dense shape: slots of the ID table (a power of two), for tests.
    pub dense_slots: Option<usize>,
    /// Event and click lines by the one-pass reader (`fuse.rs`). On by
    /// default where AVX2 and BMI2 are compiled in; `ENGINE_FUSE=0|1`.
    pub fused: bool,
}

impl Config {
    pub fn from_env() -> Self {
        let num = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok());
        Config {
            threads: sys::threads(),
            timing: std::env::var_os("ENGINE_TIMING").is_some(),
            collect: false,
            chunk_bytes: num("ENGINE_CHUNK_KB").map_or(4 << 20, |kb| (kb as usize) << 10),
            p1_bits: num("ENGINE_P1_BITS"),
            p2_bits: num("ENGINE_P2_BITS"),
            onepass: std::env::var("ENGINE_ONEPASS").is_ok_and(|v| v == "1"),
            dense: std::env::var("ENGINE_DENSE").is_ok_and(|v| v == "1"),
            dense_slots: std::env::var("ENGINE_DENSE_SLOTS").ok().and_then(|v| v.parse().ok()),
            fused: match std::env::var("ENGINE_FUSE").as_deref() {
                Ok("0") => false,
                Ok("1") => true,
                _ => cfg!(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")),
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Entry {
    ekey: u64,
    ukey: u64,
    ingest: i64,
    /// `ts_rel << 32 | price << 8 | category`, or `DEAD` for a copy that is
    /// invalid or lost the dedup.
    w3: u64,
}

const DEAD: u64 = u64::MAX;

/// What dedup needs of one event copy (one-pass shape): its key, when it
/// was ingested, and the name of its person record (`NO_RECORD` for an
/// invalid copy, which has none).
///
/// `ingest` is whole seconds from `ingest_base` (2^31 seconds before the
/// window start), which keeps the order of the millisecond values; an
/// ingest that is not a timestamp (`i64::MAX` after parsing) is
/// `u32::MAX` and sorts last. A copy whose ingest is not a whole second
/// after the base, or too far from it, leaves a `Wide` note instead, with
/// the exact value; those are compared exactly, out of line.
#[repr(C)]
#[derive(Clone, Copy)]
struct Note {
    ekey: u64,
    ingest: u32,
    rec: u32,
}

const NO_RECORD: u32 = u32::MAX;

/// Where the ingest clock of the notes starts: 2^31 seconds before the
/// window start, so ingests within 68 years either side fit.
fn note_base(ws: i64) -> i64 {
    ws.saturating_sub(1000 << 31)
}

#[derive(Clone, Copy)]
struct Wide {
    ekey: u64,
    ingest: i64,
    rec: u32,
}

/// A valid click. `key` is the user key after parsing and the person key
/// after resolve. `tc` is `ts_rel << 32 | campaign`, ts relative to the
/// start of the click window.
#[repr(C)]
#[derive(Clone, Copy)]
struct Click {
    key: u64,
    ckey: u64,
    tc: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Ev {
    pkey: u64,
    v: u64,
}

const CHUNK_BYTES: usize = 16384;
pub(crate) const CLICK_BLOCK: usize = 16384;
const WINDOW: usize = 1 << 16;
const WINDOW_SLACK: usize = 4096;

#[inline(always)]
fn mix(k: u64) -> u64 {
    let x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let x = (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 32)
}

/// A cheaper hash: one multiply, then the high half folded into the low.
/// The top bits are the top bits of the product.
#[inline(always)]
fn mix1(k: u64) -> u64 {
    let x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^ (x >> 32)
}

/// The hash of the link map and of the build's node table: one multiply
/// with the high half folded in. Home bits come from 17 up, filter bits
/// from the low 12 and from 43 up.
#[inline(always)]
fn link_hash(k: u64) -> u64 {
    mix1(k)
}

/// The person partition of a person key: `mix1(pkey) >> shift2`. The fold
/// of `mix1` only touches the low 32 bits and `shift2` is at least 50
/// (`bits_for` gives at most 14 bits), so the top bits of the product are
/// the same partition with no fold.
#[inline(always)]
fn person_part(pkey: u64, shift2: u32) -> usize {
    debug_assert!(shift2 >= 32);
    (pkey.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift2) as usize
}

/// The event partition of an event key: the top bits of one multiply. Every
/// copy of an event must land in the same partition, so the line parsers
/// all use this. Its own multiplier keeps it independent of `mix1`, whose
/// bits the dedup table uses inside a partition.
#[inline(always)]
fn event_part(ekey: u64, shift1: u32) -> usize {
    (ekey.wrapping_mul(0xD6E8_FEB8_6659_FD93) >> shift1) as usize
}

/// Lookup tables derived from the catalogs for the fast paths.
struct Fast<'a> {
    lk: &'a Lookups,
    ws: i64,
    wlen: u64,
    cws: i64,
    cwlen: u64,
    /// Category of product `p<n>`, by `n`; `unknown` where the file has none.
    prod_cat: Vec<u8>,
    unknown: u8,
    /// Campaign index of `c_<n>`, by `n`; `u32::MAX` where the file has none.
    camp_idx: Vec<u32>,
    ncamp: usize,
    ndays: usize,
    ws_day: i64,
    ws_in_day: u64,
}

impl<'a> Fast<'a> {
    /// `None` when the dataset is outside what the compact records can
    /// hold; the caller then runs the reference pipeline instead.
    fn new(lk: &'a Lookups, event_bytes: u64) -> Option<Self> {
        let wlen = lk.window_end.checked_sub(lk.window_start)?;
        if wlen <= 0 || wlen + CLICK_REACH_MS >= (u32::MAX as i64) {
            return None;
        }
        if lk.cats.len() > 255 || lk.camp_ids.len() >= (1 << 24) {
            return None;
        }
        if event_bytes >= (1 << 34) {
            return None;
        }
        // The late check adds 24 hours to a `ts` inside the window.
        lk.window_end.checked_add(LATE_MS)?;
        let ws_day = lk.window_start.div_euclid(DAY_MS);
        let ndays = ((lk.window_end - 1).div_euclid(DAY_MS) - ws_day + 1) as usize;
        if (lk.camp_ids.len() + 1) * ndays > (1 << 19) {
            return None;
        }
        let unknown = lk.unknown_cat as u8;
        let prod_cat: Vec<u8> = lk.product_dense.iter().map(|&c| if c == u32::MAX { unknown } else { c as u8 }).collect();
        let mut camp_idx = Vec::new();
        for (id, &idx) in &lk.campaigns {
            if let Some(n) = canonical_suffix(id, b"c_") {
                let n = n as usize;
                if camp_idx.len() <= n {
                    camp_idx.resize(n + 1, u32::MAX);
                }
                camp_idx[n] = idx;
            }
        }
        Some(Fast {
            lk,
            ws: lk.window_start,
            wlen: wlen as u64,
            cws: lk.window_start - CLICK_REACH_MS,
            cwlen: (wlen + CLICK_REACH_MS) as u64,
            prod_cat,
            unknown: lk.unknown_cat as u8,
            camp_idx,
            ncamp: lk.camp_ids.len(),
            ndays,
            ws_day,
            ws_in_day: lk.window_start.rem_euclid(DAY_MS) as u64,
        })
    }

    #[inline(always)]
    unsafe fn category(&self, p: *const u8, len: usize) -> u8 {
        if len >= 2 && *p == b'p' {
            if let Some(n) = canonical_number(p.add(1), len - 1) {
                return match self.prod_cat.get(n as usize) {
                    Some(&c) => c,
                    None => self.unknown,
                };
            }
        }
        self.category_slow(std::slice::from_raw_parts(p, len))
    }

    #[cold]
    #[inline(never)]
    fn category_slow(&self, id: &[u8]) -> u8 {
        self.lk.product_cat(id) as u8
    }

    #[inline(always)]
    unsafe fn campaign(&self, p: *const u8, len: usize) -> u32 {
        if len >= 3 && *p == b'c' && *p.add(1) == b'_' {
            if let Some(n) = canonical_number(p.add(2), len - 2) {
                return match self.camp_idx.get(n as usize) {
                    Some(&c) => c,
                    None => u32::MAX,
                };
            }
        }
        self.campaign_slow(std::slice::from_raw_parts(p, len))
    }

    #[cold]
    #[inline(never)]
    fn campaign_slow(&self, id: &[u8]) -> u32 {
        self.lk.campaigns.get(id).copied().unwrap_or(u32::MAX)
    }
}

// ---------------------------------------------------------------- links

/// user key -> person key (the byte-wise smallest ID of the person), for
/// every linked ID whose person is not named by that ID itself. A miss
/// means the ID is its own person. On huge pages where the kernel gives
/// them.
///
/// The table is an array of buckets of one cache line: four keys, then
/// their four persons. A key lives in the bucket its hash names unless
/// that bucket was full when it came, in which case it went to the next
/// one with room. So a lookup reads one line, compares four keys at once,
/// and goes further only when the bucket is full and does not hold the
/// key. The table has room for two to four times its keys, so a bucket
/// holds one to two on average and under 1% of the keys overflow at 1.5 a
/// bucket, about 4% at 2.
///
/// Most IDs that are looked up are their own person's name and so are not
/// in the map at all. A small blocked Bloom filter (two bits in one word
/// per key, about a megabyte at 20M events) answers "not in the map"
/// from the cache, so only likely hits pay for a probe of the big table. A
/// filter miss is exact; a filter hit is always confirmed by the table.
#[derive(Clone, Copy)]
pub struct LinkMap {
    /// During the build of the node ids this memory is a plain table of
    /// (key, value) slots; the final map reads it as buckets.
    slots: *mut [u64; 2],
    /// Slots (node id table) or buckets (final map), minus one.
    mask: usize,
    /// Index of a bucket past the table that is always empty: where
    /// lookups the filter rejected are sent, so the probe needs no extra
    /// branch.
    nowhere: usize,
    filter: *mut u64,
    fmask: usize,
}
unsafe impl Send for LinkMap {}
unsafe impl Sync for LinkMap {}

/// Keys in a bucket of the link map: a 64-byte line of keys and persons.
const BUCKET: usize = 4;

impl LinkMap {
    /// A table of `cap` slots, or of `cap` buckets when `bucket` is set, and
    /// a filter sized for `keys` keys.
    fn with_slots(cap: usize, keys: usize, bucket: bool) -> LinkMap {
        // Fresh anonymous memory is zero, and 0 is the empty key. One extra
        // slot or bucket at the end stays empty for ever (`nowhere`).
        let unit = if bucket { BUCKET * 16 } else { 16 };
        let words = (keys / 8).next_power_of_two().clamp(64, 1 << 21);
        // The filter is small; put it on a 2 MiB boundary so one huge page
        // can back it.
        const HUGE: usize = 2 << 20;
        let raw = sys::anon(words * 8 + HUGE) as usize;
        let filter = ((raw + HUGE - 1) & !(HUGE - 1)) as *mut u64;
        LinkMap { slots: sys::anon((cap + 1) * unit) as *mut [u64; 2], mask: cap - 1, nowhere: cap, filter, fmask: words - 1 }
    }
    /// Bucket `b` of the final map: `BUCKET` keys, then their persons.
    #[inline(always)]
    fn bucket(&self, b: usize) -> *mut u64 {
        unsafe { self.slots.add(b * BUCKET) as *mut u64 }
    }
    #[inline(always)]
    fn home_of(&self, h: u64) -> usize {
        (h >> 17) as usize & self.mask
    }
    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        self.home_of(link_hash(key))
    }
    #[inline(always)]
    fn filter_word(&self, h: u64) -> *mut u64 {
        unsafe { self.filter.add((h >> 43) as usize & self.fmask) }
    }
    #[inline(always)]
    fn filter_bits(h: u64) -> u64 {
        (1u64 << (h & 63)) | (1u64 << ((h >> 6) & 63))
    }
    /// Where the lookup of a key with hash `h` starts: its home bucket if
    /// the filter says the key may be present, else the empty bucket.
    /// As a byte offset into the table, so neither the prefetch nor the
    /// probe shifts it: `home_of(h) * 64` straight from the hash bits.
    #[inline(always)]
    fn start(&self, h: u64) -> usize {
        let bits = Self::filter_bits(h);
        let maybe = unsafe { *self.filter_word(h) } & bits == bits;
        const _: () = assert!(BUCKET * 16 == 64);
        let home = (h >> 11) as usize & (self.mask << 6);
        // A coin toss: this must be a conditional move, never a branch.
        std::hint::select_unpredictable(maybe, home, self.nowhere << 6)
    }
    /// The bucket at byte offset `at`.
    #[inline(always)]
    fn bucket_at(&self, at: usize) -> *mut u64 {
        unsafe { (self.slots as *mut u8).add(at) as *mut u64 }
    }
    /// Prefetch for the build, which probes before the filter exists.
    #[inline(always)]
    fn prefetch_home(&self, key: u64) {
        prefetch(unsafe { self.slots.add(self.home(key)) });
    }
    /// A lookup in three stages, so that a block of them can overlap their
    /// cache misses: `stage1` hashes the key and asks for its filter word,
    /// `stage2` reads the filter and asks for the bucket (or for the empty
    /// bucket), `stage3` reads the bucket and returns the person.
    #[inline(always)]
    fn stage1(&self, key: u64) -> u64 {
        let h = link_hash(key);
        prefetch(self.filter_word(h));
        h
    }
    #[inline(always)]
    fn stage2(&self, h: u64) -> usize {
        let at = self.start(h);
        prefetch(self.bucket_at(at));
        at
    }
    /// Which of the bucket's keys equal `key`, as a bit mask.
    #[inline(always)]
    unsafe fn matches(b: *const u64, key: u64) -> u32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use core::arch::x86_64::*;
            let keys = _mm256_load_si256(b as *const __m256i);
            let eq = _mm256_cmpeq_epi64(keys, _mm256_set1_epi64x(key as i64));
            _mm256_movemask_pd(_mm256_castsi256_pd(eq)) as u32
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut m = 0u32;
            for j in 0..BUCKET {
                m |= u32::from(*b.add(j) == key) << j;
            }
            m
        }
    }
    #[inline(always)]
    fn stage3(&self, at: usize, key: u64) -> u64 {
        unsafe {
            let b = self.bucket_at(at);
            let m = Self::matches(b, key);
            // Hit or miss is another coin toss. A full bucket without the
            // key is rare and goes out of line.
            let person = *b.add(BUCKET + (m.trailing_zeros() as usize & (BUCKET - 1)));
            let found = std::hint::select_unpredictable(m != 0, person, key);
            if (m == 0) & (*b.add(BUCKET - 1) != 0) {
                return self.person_far(key, at >> 6);
            }
            found
        }
    }
    /// The rest of the probe sequence after a full bucket without the key.
    #[inline(never)]
    fn person_far(&self, key: u64, first: usize) -> u64 {
        let mut i = (first + 1) & self.mask;
        loop {
            unsafe {
                let b = self.bucket(i);
                let m = Self::matches(b, key);
                if m != 0 {
                    return *b.add(BUCKET + m.trailing_zeros() as usize);
                }
                if *b.add(BUCKET - 1) == 0 {
                    return key;
                }
            }
            i = (i + 1) & self.mask;
        }
    }

    fn build(pairs: &[(u64, u64)]) -> LinkMap {
        // Room for twice the keys, so a bucket holds one to two on average.
        Self::build_with_room(pairs, 8).0
    }

    /// `build`, and the memory of its work tables (node ids, keys,
    /// parents, renamed pairs): dead, mostly faulted in already, and never
    /// unmapped or reused, as `(address, bytes)` regions.
    #[cfg(test)]
    fn build_with_spare(pairs: &[(u64, u64)]) -> (LinkMap, Vec<(usize, usize)>) {
        Self::build_with_room(pairs, 8)
    }

    /// `room` is the table's key capacity in quarters of the key count.
    fn build_with_room(pairs: &[(u64, u64)], room: usize) -> (LinkMap, Vec<(usize, usize)>) {
        // Pass 1: a dense node id per distinct ID, and union-find over them.
        let cap = (pairs.len() * 4).next_power_of_two().max(16);
        let ids = LinkMap::with_slots(cap, 0, false);
        let mut keys: Vec<u64> = Vec::with_capacity(pairs.len() * 2 + 16);
        let mut parent: Vec<u32> = Vec::with_capacity(pairs.len() * 2 + 16);
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            loop {
                let p = parent[x as usize];
                if p == x {
                    return x;
                }
                let g = parent[p as usize];
                parent[x as usize] = g;
                x = g;
            }
        }
        for block in pairs.chunks(16) {
            for &(a, b) in block {
                ids.prefetch_home(a);
                ids.prefetch_home(b);
            }
            for &(a, b) in block {
                let mut node = [0u32; 2];
                for (side, key) in [a, b].into_iter().enumerate() {
                    let mut i = ids.home(key);
                    node[side] = loop {
                        let s = unsafe { &mut *ids.slots.add(i) };
                        if s[0] == key {
                            break s[1] as u32;
                        }
                        if s[0] == 0 {
                            let id = keys.len() as u32;
                            *s = [key, u64::from(id)];
                            keys.push(key);
                            parent.push(id);
                            break id;
                        }
                        i = (i + 1) & ids.mask;
                    };
                }
                let ra = find(&mut parent, node[0]);
                let rb = find(&mut parent, node[1]);
                if ra != rb {
                    // The root is always the smallest ID of its set, so the
                    // root's key is the person's name.
                    if cmp_ids(keys[ra as usize], keys[rb as usize]).is_lt() {
                        parent[rb as usize] = ra;
                    } else {
                        parent[ra as usize] = rb;
                    }
                }
            }
        }
        // Pass 2: the final map holds only IDs that are not their own name.
        let mut renamed: Vec<[u64; 2]> = Vec::with_capacity(keys.len());
        for id in 0..keys.len() as u32 {
            let root = find(&mut parent, id);
            if root != id {
                renamed.push([keys[id as usize], keys[root as usize]]);
            }
        }
        let buckets = (renamed.len() * room / 4).next_power_of_two().max(16) / BUCKET;
        let map = LinkMap::with_slots(buckets, renamed.len(), true);
        for block in renamed.chunks(16) {
            for pair in block {
                prefetch(map.bucket(map.home(pair[0])));
            }
            for pair in block {
                let h = link_hash(pair[0]);
                let mut i = map.home_of(h);
                unsafe {
                    // Buckets fill from the front and nothing is ever
                    // removed, so the last key tells whether one is full.
                    while *map.bucket(i).add(BUCKET - 1) != 0 {
                        i = (i + 1) & map.mask;
                    }
                    let b = map.bucket(i);
                    let mut j = 0;
                    while *b.add(j) != 0 {
                        j += 1;
                    }
                    *b.add(j) = pair[0];
                    *b.add(BUCKET + j) = pair[1];
                    *map.filter_word(h) |= Self::filter_bits(h);
                }
            }
        }
        // Vectors under `alloc::BIG` live in the system allocator and stay
        // with it; the rest is bump memory that is never handed out again.
        let mut spare = vec![(ids.slots as usize, (cap + 1) * 16)];
        fn give<T>(v: Vec<T>, spare: &mut Vec<(usize, usize)>) {
            let bytes = v.capacity() * std::mem::size_of::<T>();
            if bytes >= crate::alloc::BIG {
                let v = std::mem::ManuallyDrop::new(v);
                spare.push((v.as_ptr() as usize, bytes));
            }
        }
        give(keys, &mut spare);
        give(parent, &mut spare);
        give(renamed, &mut spare);
        (map, spare)
    }

    /// What `build_with_spare` makes, on one thread, with the cache misses
    /// of many pairs in flight at once.
    ///
    /// A node is a slot of an open-addressing table: its ID, and its
    /// parent (0 for a root, else the parent's slot plus one), in one
    /// 16-byte slot, so a step of `find` is one miss. Unions link the root
    /// with the larger ID under the one with the smaller, so a root is the
    /// smallest ID of its set, the person's name; each link is noted as
    /// (ID, parent then). The map then puts each noted ID under its root's
    /// ID. Both passes run as software pipelines: while one pair or note
    /// does its dependent work, the lines of the next few dozen are
    /// already on their way.
    fn build_serial(pairs: &[(u64, u64)]) -> (LinkMap, Vec<(usize, usize)>) {
        Self::build_serial_with_room(pairs, 8)
    }

    fn build_serial_with_room(pairs: &[(u64, u64)], room: usize) -> (LinkMap, Vec<(usize, usize)>) {
        Self::build_serial_tuned::<32, 16, 48, 32, 16>(pairs, room)
    }

    fn build_serial_tuned<const DH: usize, const DI: usize, const MA: usize, const MC: usize, const MB: usize>(
        pairs: &[(u64, u64)],
        room: usize,
    ) -> (LinkMap, Vec<(usize, usize)>) {
        const RING: usize = 64;
        const { assert!(DI <= DH && DH < RING && MB <= MC && MC <= MA && MA < RING) };
        let total = pairs.len();
        if total == 0 {
            return (LinkMap::with_slots(4, 0, true), Vec::new());
        }
        // Bound: at most two distinct IDs a pair, so at most 2 * total keys
        // go in, and the table has 2 * total + 1 slots or more: an empty
        // slot always remains, which ends every probe. Real links form
        // trees (about one new ID a pair), so it is under half full.
        let t0 = Instant::now();
        let cap = (total * 2 + 1).next_power_of_two().max(16);
        // Slot numbers are noted as u32.
        assert!(cap <= 1 << 32, "too many links");
        let mask = cap - 1;
        let nodes = sys::anon(cap * 16) as *mut [u64; 2];
        let home = |key: u64| (link_hash(key) >> 17) as usize & mask;
        // Every slot index below is `& mask` or a parent pointer, which is
        // a slot index plus one, so all accesses are inside the `cap` slots.
        // A hit and a new key are a coin toss, so both end the probe on one
        // branch, and the key is stored either way.
        let intern = |key: u64| -> usize {
            let mut i = home(key);
            loop {
                let k = unsafe { (*nodes.add(i))[0] };
                if (k == key) | (k == 0) {
                    break;
                }
                i = (i + 1) & mask;
            }
            unsafe { (*nodes.add(i))[0] = key };
            i
        };
        // Most nodes are a root or one step below one: those two cases take
        // no branch that depends on which. Deeper paths go out of line.
        let find_far = |mut x: usize| -> usize {
            loop {
                let up = unsafe { (*nodes.add(x))[1] };
                if up == 0 {
                    return x;
                }
                let p = up as usize - 1;
                let above = unsafe { (*nodes.add(p))[1] };
                if above != 0 {
                    // Path halving.
                    unsafe { (*nodes.add(x))[1] = above };
                }
                x = p;
            }
        };
        let find = |x: usize| -> usize {
            let up = unsafe { (*nodes.add(x))[1] };
            let p = std::hint::select_unpredictable(up == 0, x, (up as usize).wrapping_sub(1));
            // A root reads its own parent word again: 0.
            if unsafe { (*nodes.add(p))[1] } == 0 {
                return p;
            }
            find_far(x)
        };
        // Bound: a pair makes at most one link, so at most `total` notes;
        // reserved up front, so pushes never reallocate (a plain `push`,
        // correct either way).
        let mut made: Vec<(u64, u32)> = Vec::with_capacity(total);
        let union = |made: &mut Vec<(u64, u32)>, na: usize, nb: usize| {
            let (ra, rb) = (find(na), find(nb));
            if ra != rb {
                let (ka, kb) = unsafe { ((*nodes.add(ra))[0], (*nodes.add(rb))[0]) };
                let lt = cmp_ids(ka, kb).is_lt();
                let small = std::hint::select_unpredictable(lt, ra, rb);
                let large = std::hint::select_unpredictable(lt, rb, ra);
                let key = std::hint::select_unpredictable(lt, kb, ka);
                unsafe { (*nodes.add(large))[1] = small as u64 + 1 };
                made.push((key, small as u32));
            }
        };
        // Pass 1, a pipeline of three stages over the pairs: ask for the
        // homes of pair i + DH, intern pair i + DH - DI and ask for one step
        // up from each of its nodes, union pair i.
        let mut ring = [(0usize, 0usize); RING];
        let n = pairs.len();
        for i in 0..n + DH {
            if let Some(&(a, b)) = pairs.get(i) {
                prefetch(unsafe { nodes.add(home(a)) });
                prefetch(unsafe { nodes.add(home(b)) });
            }
            if let Some(j) = i.checked_sub(DH - DI) {
                if let Some(&(a, b)) = pairs.get(j) {
                    let (na, nb) = (intern(a), intern(b));
                    ring[j % RING] = (na, nb);
                    for node in [na, nb] {
                        let up = unsafe { (*nodes.add(node))[1] };
                        prefetch(unsafe { nodes.add(std::hint::select_unpredictable(up == 0, node, (up as usize).wrapping_sub(1))) });
                    }
                }
            }
            if let Some(k) = i.checked_sub(DH) {
                let (na, nb) = ring[k % RING];
                union(&mut made, na, nb);
            }
        }
        // Pass 2, the same pipeline over the notes: ask for the parent
        // node of note i + MA, find the root of note i + MA - MB and ask
        // for its bucket and filter word, place note i.
        let t1 = Instant::now();
        let count = made.len();
        let buckets = (count * room / 4).next_power_of_two().max(16) / BUCKET;
        let map = LinkMap::with_slots(buckets, count, true);
        let mut ready = [(0u64, 0u64, 0u64); RING];
        for i in 0..count + MA {
            if let Some(&(_, up)) = made.get(i) {
                prefetch(unsafe { nodes.add(up as usize) });
            }
            if let Some(j) = i.checked_sub(MA - MC) {
                if let Some(&(_, up)) = made.get(j) {
                    let up = up as usize;
                    let above = unsafe { (*nodes.add(up))[1] };
                    prefetch(unsafe { nodes.add(std::hint::select_unpredictable(above == 0, up, (above as usize).wrapping_sub(1))) });
                }
            }
            if let Some(j) = i.checked_sub(MA - MB) {
                if let Some(&(key, up)) = made.get(j) {
                    let h = link_hash(key);
                    let name = unsafe { (*nodes.add(find(up as usize)))[0] };
                    ready[j % RING] = (key, name, h);
                    prefetch(map.bucket(map.home_of(h)));
                    prefetch(map.filter_word(h));
                }
            }
            if let Some(k) = i.checked_sub(MA) {
                let (key, name, h) = ready[k % RING];
                let mut i = map.home_of(h);
                unsafe {
                    // Buckets fill from the front and nothing is ever
                    // removed, so the last key tells whether one is
                    // full. The map has more key room than notes
                    // (`room` >= 5 quarters of them), so a bucket with
                    // room is always found.
                    let (b, free) = loop {
                        let b = map.bucket(i);
                        let free = Self::matches(b, 0);
                        if free != 0 {
                            break (b, free);
                        }
                        i = (i + 1) & map.mask;
                    };
                    // `free` is a nonzero mask of BUCKET bits: j < BUCKET.
                    let j = free.trailing_zeros() as usize;
                    *b.add(j) = key;
                    *b.add(BUCKET + j) = name;
                    *map.filter_word(h) |= Self::filter_bits(h);
                }
            }
        }
        if prof::ON.load(Ordering::Relaxed) {
            let t2 = Instant::now();
            eprintln!(
                "[engine]   links serial: union {:.1} ms, map {:.1} ms; pairs {total}, node cap {cap}, renamed {count}",
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3
            );
        }
        let mut spare = vec![(nodes as usize, cap * 16)];
        let bytes = made.capacity() * std::mem::size_of::<(u64, u32)>();
        if bytes >= crate::alloc::BIG {
            let v = std::mem::ManuallyDrop::new(made);
            spare.push((v.as_ptr() as usize, bytes));
        }
        (map, spare)
    }
}


/// A raw pointer that threads may share; what it points at is atomic.
#[derive(Clone, Copy)]
struct Shared<T>(*const T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    /// Element `i`. The caller keeps `i` inside the allocation.
    #[inline(always)]
    fn at(&self, i: usize) -> &T {
        unsafe { &*self.0.add(i) }
    }
    #[inline(always)]
    fn ptr(&self, i: usize) -> *const T {
        unsafe { self.0.add(i) }
    }
}

/// Pairs (or linked nodes) a job of the parallel link build takes.
const LINK_STEP: usize = 4096;
/// The link build works in groups of this many, one stage at a time.
const LINK_GROUP: usize = 32;

fn link_jobs<T>(lists: &[Vec<T>]) -> Vec<(usize, usize)> {
    let mut jobs = Vec::new();
    for (l, list) in lists.iter().enumerate() {
        let mut at = 0;
        while at < list.len() {
            jobs.push((l, at));
            at += LINK_STEP;
        }
    }
    jobs
}

impl LinkMap {
    /// Join all pairs and build the map on `threads` threads: what `build`
    /// makes, for the one-pass shape, where nothing else runs meanwhile.
    ///
    /// Two parallel steps. (1) Every ID is interned in a shared
    /// open-addressing table; its slot number is its node. Each pair is a
    /// union in a lock-free union-find whose links always go from the root
    /// with the larger ID to the root with the smaller, so pointers only
    /// ever lead to smaller IDs: no cycle can form whatever the
    /// interleaving, and when all unions are done a root is the smallest
    /// ID of its set, the person's name. Every successful link makes one
    /// node a non-root, and the thread that made it notes the node. (2)
    /// Each noted node goes into the final map under its root's ID; threads
    /// race only for the free slots of a bucket, which fill from the front.
    fn build_parallel(lists: &[Vec<(u64, u64)>], threads: usize) -> LinkMap {
        Self::build_parallel_with_room(lists, threads, 8)
    }

    fn build_parallel_with_room(lists: &[Vec<(u64, u64)>], threads: usize, room: usize) -> LinkMap {
        let total: usize = lists.iter().map(|l| l.len()).sum();
        if total == 0 {
            return LinkMap::with_slots(4, 0, true);
        }
        let t0 = Instant::now();
        // At most two IDs a pair, so the table never fills. Real links form
        // trees (about one new ID a pair), so it is about half full; 3x
        // measured the same time with 35 MB more memory at 20M events.
        let cap = (total * 2).next_power_of_two().max(16);
        let mask = cap - 1;
        // A node is two words: its ID, and its parent. Fresh anonymous
        // memory is zero: key 0 is an empty slot, parent 0 is "root", any
        // other parent is that node's slot number plus one.
        let nodes = Shared(sys::anon(cap * 16) as *const [AtomicU64; 2]);
        let home = |key: u64| (link_hash(key) >> 17) as usize & mask;
        let intern = |key: u64| -> usize {
            let mut i = home(key);
            loop {
                let slot = &nodes.at(i)[0];
                let k = slot.load(Ordering::Relaxed);
                if k == key {
                    return i;
                }
                if k == 0 {
                    match slot.compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => return i,
                        Err(now) if now == key => return i,
                        Err(_) => {}
                    }
                }
                i = (i + 1) & mask;
            }
        };
        let find = |mut x: usize| -> usize {
            loop {
                let px = &nodes.at(x)[1];
                let up = px.load(Ordering::Relaxed);
                if up == 0 {
                    return x;
                }
                let above = nodes.at(up as usize - 1)[1].load(Ordering::Relaxed);
                if above != 0 {
                    // Path halving. Losing the race is harmless: the old
                    // pointer is still an ancestor.
                    let _ = px.compare_exchange(up, above, Ordering::Relaxed, Ordering::Relaxed);
                }
                x = up as usize - 1;
            }
        };
        let key_of = |node: usize| nodes.at(node)[0].load(Ordering::Relaxed);
        let jobs = link_jobs(lists);
        let next = AtomicUsize::new(0);
        // Per thread: each node it linked, as (its ID, its new parent).
        let mut linked: Vec<Vec<(u64, u32)>> = Vec::new();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads.clamp(1, jobs.len()))
                .map(|_| {
                    scope.spawn(|| {
                        let mut made: Vec<(u64, u32)> = Vec::with_capacity(total / threads.max(1) + 1024);
                        let mut ends = [(0usize, 0usize); LINK_GROUP];
                        while let Some(&(l, at)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let list = &lists[l];
                            for block in list[at..list.len().min(at + LINK_STEP)].chunks(LINK_GROUP) {
                                // The table is far bigger than the cache: each
                                // stage runs for the whole group, so its misses
                                // overlap. Homes first, then the nodes, then
                                // one step up from each node.
                                for &(a, b) in block {
                                    prefetch(nodes.ptr(home(a)));
                                    prefetch(nodes.ptr(home(b)));
                                }
                                for (end, &(a, b)) in ends.iter_mut().zip(block) {
                                    *end = (intern(a), intern(b));
                                }
                                for &(na, nb) in &ends[..block.len()] {
                                    for node in [na, nb] {
                                        let up = nodes.at(node)[1].load(Ordering::Relaxed);
                                        if up != 0 {
                                            prefetch(nodes.ptr(up as usize - 1));
                                        }
                                    }
                                }
                                for &(na, nb) in &ends[..block.len()] {
                                    loop {
                                        let (ra, rb) = (find(na), find(nb));
                                        if ra == rb {
                                            break;
                                        }
                                        let (ka, kb) = (key_of(ra), key_of(rb));
                                        let (small, large, key) = if cmp_ids(ka, kb).is_lt() { (ra, rb, kb) } else { (rb, ra, ka) };
                                        // Only a root can be linked, and only once.
                                        if nodes.at(large)[1]
                                            .compare_exchange(0, small as u64 + 1, Ordering::Relaxed, Ordering::Relaxed)
                                            .is_ok()
                                        {
                                            made.push((key, small as u32));
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        // A thread of its own: all its time is the build's.
                        prof::add(3, 0.0);
                        made
                    })
                })
                .collect();
            for h in handles {
                linked.push(h.join().unwrap());
            }
        });
        let t1 = Instant::now();
        let count: usize = linked.iter().map(|l| l.len()).sum();
        let buckets = (count * room / 4).next_power_of_two().max(16) / BUCKET;
        let map = LinkMap::with_slots(buckets, count, true);
        let jobs = link_jobs(&linked);
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads.clamp(1, jobs.len().max(1)) {
                scope.spawn(|| unsafe {
                    let mut ready = [(0u64, 0u64, 0u64); LINK_GROUP];
                    while let Some(&(l, at)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let list = &linked[l];
                        for block in list[at..list.len().min(at + LINK_STEP)].chunks(LINK_GROUP) {
                            // The node's parent when it was linked; it may have
                            // been linked on since, `find` goes the rest.
                            for &(_, up) in block {
                                prefetch(nodes.ptr(up as usize));
                            }
                            for (out, &(key, up)) in ready.iter_mut().zip(block) {
                                let h = link_hash(key);
                                *out = (key, key_of(find(up as usize)), h);
                                prefetch(map.bucket(map.home_of(h)));
                                prefetch(map.filter_word(h));
                            }
                            for &(key, name, h) in &ready[..block.len()] {
                                let mut i = map.home_of(h);
                                'place: loop {
                                    let b = map.bucket(i) as *const AtomicU64;
                                    for j in 0..BUCKET {
                                        let slot = &*b.add(j);
                                        if slot.load(Ordering::Relaxed) == 0
                                            && slot.compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed).is_ok()
                                        {
                                            (*b.add(BUCKET + j)).store(name, Ordering::Relaxed);
                                            break 'place;
                                        }
                                    }
                                    i = (i + 1) & map.mask;
                                }
                                (*(map.filter_word(h) as *const AtomicU64)).fetch_or(Self::filter_bits(h), Ordering::Relaxed);
                            }
                        }
                    }
                    prof::add(3, 0.0);
                });
            }
        });
        if prof::ON.load(Ordering::Relaxed) {
            let t2 = Instant::now();
            eprintln!(
                "[engine]   links wall: union {:.1} ms, map {:.1} ms; pairs {total}, node cap {cap}, renamed {count}",
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3
            );
        }
        map
    }
}

// ---------------------------------------------------------------- phase 1

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Events,
    Clicks,
    Links,
}

struct Job {
    kind: Kind,
    file: usize,
    start: usize,
    end: usize,
}

struct ParseState<'a> {
    fast: &'a Fast<'a>,
    cache: DateCache,
    entries: Scatter<Entry>,
    shift1: u32,
    clicks: Vec<Vec<Click>>,
    pairs: Vec<(u64, u64)>,
    /// The one-pass shape: where parsed copies and clicks go instead of
    /// `entries` and `clicks`.
    emit: Option<Box<Emit>>,
    /// The dense shape: the same, with person numbers.
    dense: Option<Box<dense::DenseEmit>>,
}

/// Where a one-pass parser sends what it parsed: person lookups in blocks,
/// then the scatters by person and by event key.
struct Emit {
    links: LinkMap,
    notes: Scatter<Note>,
    ev: Scatter<Ev>,
    cl: Scatter<Click>,
    shift1: u32,
    shift2: u32,
    /// Start of the ingest clock of the notes, in epoch milliseconds.
    ingest_base: i64,
    /// Notes whose ingest does not fit a `Note` (never under the rules).
    wide: Vec<Wide>,
    /// Event copies and clicks waiting for their lookup.
    nevents: usize,
    nclicks: usize,
    /// Lookups per block (at most `BLOCK`).
    limit: usize,
    events: [Entry; EMIT_BLOCK],
    clicks: [Click; EMIT_BLOCK],
}

impl Emit {
    fn new(links: LinkMap, notes: Scatter<Note>, ev: Scatter<Ev>, cl: Scatter<Click>, shift1: u32, shift2: u32, ws: i64) -> Box<Emit> {
        Box::new(Emit {
            links,
            notes,
            ev,
            cl,
            shift1,
            shift2,
            ingest_base: note_base(ws),
            wide: Vec::new(),
            nevents: 0,
            nclicks: 0,
            limit: std::env::var("ENGINE_LOOKUP_BLOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(EMIT_BLOCK).clamp(1, EMIT_BLOCK),
            events: [Entry { ekey: 0, ukey: 0, ingest: 0, w3: DEAD }; EMIT_BLOCK],
            clicks: [Click { key: 0, ckey: 0, tc: 0 }; EMIT_BLOCK],
        })
    }

    #[inline(always)]
    unsafe fn event(&mut self, entry: Entry) {
        *self.events.get_unchecked_mut(self.nevents) = entry;
        self.nevents += 1;
        if self.nevents == self.limit {
            self.flush_events();
        }
    }

    #[inline(always)]
    unsafe fn click(&mut self, click: Click) {
        *self.clicks.get_unchecked_mut(self.nclicks) = click;
        self.nclicks += 1;
        if self.nclicks == self.limit {
            self.flush_clicks();
        }
    }

    /// Look up the persons of the waiting event copies and write them out:
    /// a person record for each valid copy, a note for every copy.
    ///
    /// The link map is far bigger than the cache. Lookups go a block at a
    /// time, each stage for the whole block, so the misses of a block are
    /// all in flight together and every loop stays small.
    #[inline(never)]
    unsafe fn flush_events(&mut self) {
        let t = prof::start();
        let n = self.nevents;
        self.nevents = 0;
        let links = self.links;
        let (shift1, shift2) = (self.shift1, self.shift2);
        let base = self.ingest_base;
        let block = std::slice::from_raw_parts(self.events.as_ptr(), n);
        let mut hashes = [0u64; EMIT_BLOCK];
        let mut starts = [0usize; EMIT_BLOCK];
        for (h, entry) in hashes.iter_mut().zip(block) {
            *h = links.stage1(entry.ukey);
        }
        for (at, &h) in starts.iter_mut().zip(&hashes[..n]) {
            *at = links.stage2(h);
        }
        for (&at, entry) in starts.iter().zip(block) {
            let mut rec = NO_RECORD;
            if entry.w3 != DEAD {
                let pkey = links.stage3(at, entry.ukey);
                rec = self.ev.push_named(person_part(pkey, shift2), Ev { pkey, v: entry.w3 });
            }
            // Exact as u64 whenever `ingest >= base`.
            let since = (entry.ingest as u64).wrapping_sub(base as u64);
            let seconds = since / 1000;
            let whole = (entry.ingest >= base) & (since % 1000 == 0) & (seconds < u64::from(u32::MAX));
            let none = entry.ingest == i64::MAX;
            if whole | none {
                let ingest = if none { u32::MAX } else { seconds as u32 };
                self.notes.push(event_part(entry.ekey, shift1), Note { ekey: entry.ekey, ingest, rec });
            } else {
                self.wide_note(entry.ekey, entry.ingest, rec);
            }
        }
        prof::add(10, t);
    }

    #[cold]
    #[inline(never)]
    fn wide_note(&mut self, ekey: u64, ingest: i64, rec: u32) {
        self.wide.push(Wide { ekey, ingest, rec });
    }

    #[inline(never)]
    unsafe fn flush_clicks(&mut self) {
        let t = prof::start();
        let n = self.nclicks;
        self.nclicks = 0;
        let links = self.links;
        let shift2 = self.shift2;
        let block = std::slice::from_raw_parts(self.clicks.as_ptr(), n);
        let mut hashes = [0u64; EMIT_BLOCK];
        let mut starts = [0usize; EMIT_BLOCK];
        for (h, click) in hashes.iter_mut().zip(block) {
            *h = links.stage1(click.key);
        }
        for (at, &h) in starts.iter_mut().zip(&hashes[..n]) {
            *at = links.stage2(h);
        }
        for (&at, click) in starts.iter().zip(block) {
            let pkey = links.stage3(at, click.key);
            self.cl.push(person_part(pkey, shift2), Click { key: pkey, ckey: click.ckey, tc: click.tc });
        }
        prof::add(10, t);
    }
}

// With `--cfg ss_hot`: the section of `events`, 4096-byte aligned (see
// `sys::layout`).
#[cfg(all(ss_hot, target_arch = "x86_64"))]
core::arch::global_asm!(".pushsection .text.hot_events,\"ax\",@progbits", ".p2align 12", ".popsection");

const VIEW: u32 = u32::from_le_bytes(*b"view");
const CART: u32 = u32::from_le_bytes(*b"cart");
const PURCHASE: u64 = u64::from_le_bytes(*b"purchase");

impl<'a> ParseState<'a> {
    /// The event lines of a window. `MODE` 1 is the one-pass shape: every
    /// copy goes to `emit` for its person lookup, not to `entries`; 2 is the
    /// dense shape, where it goes to `dense`.
    #[inline(never)]
    #[cfg_attr(all(ss_hot, target_arch = "x86_64"), link_section = ".text.hot_events")]
    unsafe fn events<const MODE: u8, L: Lines>(&mut self, src: L) -> usize {
        crate::layout_pad!(sys::layout::EVENTS_PAD);
        let fast = self.fast;
        let (ws, wlen) = (fast.ws, fast.wlen);
        let shift1 = self.shift1;
        let cache = &mut self.cache;
        let entries = &mut self.entries;
        let emit: *mut Emit = match &mut self.emit {
            Some(e) => &mut **e,
            None => std::ptr::null_mut(),
        };
        let dense: *mut dense::DenseEmit = match &mut self.dense {
            Some(e) => &mut **e,
            None => std::ptr::null_mut(),
        };
        src.run::<7>(|base, b, known| {
            let id_len = b.len(0);
            if id_len == 0 {
                return;
            }
            let ekey = ids::pack_in(base.add(b.start(0)), id_len, known);
            let user_len = b.len(1);
            let ukey = ids::pack_in(base.add(b.start(1)), user_len, known);
            let (ts, ingest) = parse_ts_ingest(base.add(b.start(2)), b.len(2), base.add(b.start(3)), b.len(3), cache);
            let type_len = b.len(4);
            let tp = base.add(b.start(4));
            let word = load32(tp);
            let ts_rel = ts.wrapping_sub(ws) as u64;
            // Late is `ingest - ts > 24 h`, which is `ingest > ts + 24 h`
            // whenever that sum does not overflow; with `ts` in the window
            // (the only case where the answer is used) it cannot, see
            // `Fast::new`. An ingest_ts that is no timestamp is `i64::MAX`
            // here, later than any `ts + 24 h`: the copy sorts last and is
            // never valid, as in the reference.
            let fresh = ingest <= ts.wrapping_add(LATE_MS);
            // Nearly every row passes these four rules and is a view or a
            // cart: one well-predicted branch for the rules, one for the
            // length of the type.
            let prod_len = b.len(5);
            let sound = (user_len != 0) & (prod_len != 0) & (ts_rel < wlen) & fresh;
            let mut w3 = DEAD;
            if !sound {
                std::hint::cold_path();
            } else if type_len == 4 {
                // View or cart is a coin toss, so it selects, it does not
                // branch.
                let plain = (word == VIEW) | (word == CART);
                w3 = std::hint::select_unpredictable(plain, ts_rel << 32, DEAD);
            } else if type_len == 8 && load64(tp) == PURCHASE {
                let price = parse_price(base.add(b.start(6)), b.len(6));
                if price != 0 {
                    let cat = fast.category(base.add(b.start(5)), prod_len);
                    w3 = (ts_rel << 32) | (price << 8) | u64::from(cat);
                }
            }
            if MODE == 2 {
                (*dense).event(Entry { ekey, ukey, ingest, w3 });
            } else if MODE == 1 {
                (*emit).event(Entry { ekey, ukey, ingest, w3 });
            } else {
                let part = event_part(ekey, shift1);
                entries.push(part, Entry { ekey, ukey, ingest, w3 });
            }
        })
    }

    #[inline(never)]
    unsafe fn clicks<const MODE: u8, L: Lines>(&mut self, src: L) -> usize {
        let fast = self.fast;
        let (cws, cwlen) = (fast.cws, fast.cwlen);
        let cache = &mut self.cache;
        let clicks = &mut self.clicks;
        // The free room of the last click block, as raw pointers: no length
        // reload and no capacity check per click. Its length is set when it
        // is full and when this window ends.
        let (mut at, mut stop) = match clicks.last_mut() {
            Some(v) => (v.as_mut_ptr().add(v.len()), v.as_mut_ptr().add(CLICK_BLOCK)),
            None => (std::ptr::null_mut(), std::ptr::null_mut()),
        };
        let emit: *mut Emit = match &mut self.emit {
            Some(e) => &mut **e,
            None => std::ptr::null_mut(),
        };
        let dense: *mut dense::DenseEmit = match &mut self.dense {
            Some(e) => &mut **e,
            None => std::ptr::null_mut(),
        };
        let left = src.run::<4>(|base, b, known| {
            let id_len = b.len(0);
            let user_len = b.len(1);
            if id_len == 0 || user_len == 0 {
                return;
            }
            let ts = parse_ts(base.add(b.start(3)), b.len(3), cache);
            let rel = ts.wrapping_sub(cws) as u64;
            if rel >= cwlen {
                return;
            }
            let camp = fast.campaign(base.add(b.start(2)), b.len(2));
            if camp == u32::MAX {
                return;
            }
            let ckey = ids::pack_in(base.add(b.start(0)), id_len, known);
            let ukey = ids::pack_in(base.add(b.start(1)), user_len, known);
            let click = Click { key: ukey, ckey, tc: (rel << 32) | u64::from(camp) };
            if MODE == 2 {
                (*dense).click(click);
                return;
            }
            if MODE == 1 {
                (*emit).click(click);
                return;
            }
            if at == stop {
                if let Some(v) = clicks.last_mut() {
                    v.set_len(CLICK_BLOCK);
                }
                let mut v = Vec::with_capacity(CLICK_BLOCK);
                at = v.as_mut_ptr();
                stop = at.add(CLICK_BLOCK);
                clicks.push(v);
            }
            at.write(click);
            at = at.add(1);
        });
        if let Some(v) = clicks.last_mut() {
            v.set_len(at.offset_from(v.as_ptr()) as usize);
        }
        left
    }

    #[inline(never)]
    unsafe fn links(&mut self, base: *const u8, scan: &Scan, limit: u32) -> u32 {
        let pairs = &mut self.pairs;
        let known = scan.clean;
        for_each_line::<2>(scan, limit, |s, e| {
            let (la, lb) = ((e[0] - s[0]) as usize, (e[1] - s[1]) as usize);
            if la == 0 || lb == 0 {
                return;
            }
            pairs.push((ids::pack_in(base.add(s[0] as usize), la, known), ids::pack_in(base.add(s[1] as usize), lb, known)));
        })
    }

    /// One line that no scan window holds whole: longer than a window, or
    /// the last line of a file with no line break after it. Split and
    /// parsed with the reference readers, on slices, so neither the line
    /// nor a field has a length limit. It must decide exactly what the
    /// three loops above decide.
    #[cold]
    #[inline(never)]
    unsafe fn long_line(&mut self, line: &[u8], kind: Kind) {
        let fast = self.fast;
        match kind {
            Kind::Events => {
                let Some(f) = split_long::<7>(line) else {
                    return;
                };
                if f[0].is_empty() {
                    return;
                }
                let ekey = ids::pack_slice(f[0]);
                let ukey = ids::pack_slice(f[1]);
                let ingest = parse_ts_naive(f[3]);
                let ts = parse_ts_naive(f[2]).unwrap_or(BAD);
                let purchase = f[4] == b"purchase";
                let typed = purchase || f[4] == b"view" || f[4] == b"cart";
                let ts_rel = ts.wrapping_sub(fast.ws) as u64;
                let mut valid = typed
                    && !f[1].is_empty()
                    && !f[5].is_empty()
                    && ts_rel < fast.wlen
                    && ingest.is_some_and(|at| at.saturating_sub(ts) <= LATE_MS);
                let mut pc = 0u64;
                if purchase && valid {
                    match parse_price_naive(f[6]) {
                        Some(price) => pc = (price << 8) | u64::from(fast.lk.product_cat(f[5]) as u8),
                        None => valid = false,
                    }
                }
                let w3 = if valid { (ts_rel << 32) | pc } else { DEAD };
                let entry = Entry { ekey, ukey, ingest: ingest.unwrap_or(i64::MAX), w3 };
                if let Some(dense) = &mut self.dense {
                    dense.event(entry);
                    return;
                }
                match &mut self.emit {
                    Some(emit) => emit.event(entry),
                    None => self.entries.push(event_part(ekey, self.shift1), entry),
                }
            }
            Kind::Clicks => {
                let Some(f) = split_long::<4>(line) else {
                    return;
                };
                if f[0].is_empty() || f[1].is_empty() {
                    return;
                }
                let ts = parse_ts_naive(f[3]).unwrap_or(BAD);
                let rel = ts.wrapping_sub(fast.cws) as u64;
                if rel >= fast.cwlen {
                    return;
                }
                let camp = fast.campaign_slow(f[2]);
                if camp == u32::MAX {
                    return;
                }
                let click = Click { key: ids::pack_slice(f[1]), ckey: ids::pack_slice(f[0]), tc: (rel << 32) | u64::from(camp) };
                if let Some(dense) = &mut self.dense {
                    dense.click(click);
                    return;
                }
                if let Some(emit) = &mut self.emit {
                    emit.click(click);
                    return;
                }
                if self.clicks.last().is_none_or(|b| b.len() == CLICK_BLOCK) {
                    self.clicks.push(Vec::with_capacity(CLICK_BLOCK));
                }
                self.clicks.last_mut().unwrap_unchecked().push(click);
            }
            Kind::Links => {
                let Some(f) = split_long::<2>(line) else {
                    return;
                };
                if f[0].is_empty() || f[1].is_empty() {
                    return;
                }
                self.pairs.push((ids::pack_slice(f[0]), ids::pack_slice(f[1])));
            }
        }
    }

    #[inline(always)]
    unsafe fn consume(&mut self, kind: Kind, base: *const u8, scan: &Scan, limit: u32) -> u32 {
        let src = Scanned { base, scan, limit };
        let one = self.emit.is_some();
        let dense = self.dense.is_some();
        (match kind {
            Kind::Events if dense => self.events::<2, _>(src),
            Kind::Events if one => self.events::<1, _>(src),
            Kind::Events => self.events::<0, _>(src),
            Kind::Clicks if dense => self.clicks::<2, _>(src),
            Kind::Clicks if one => self.clicks::<1, _>(src),
            Kind::Clicks => self.clicks::<0, _>(src),
            Kind::Links => return self.links(base, scan, limit),
        }) as u32
    }

    /// The event or click lines of `m` that start in `[pos, end)`, `pos` a
    /// line start, by the one-pass reader. Returns where it stopped: `end`
    /// or later, or the start of a line that only `long_line` can take.
    #[inline(always)]
    unsafe fn consume_fused(&mut self, kind: Kind, m: Mapped, pos: usize, end: usize, scan: &mut Scan) -> usize {
        let src = Fused { m, pos, end, scan };
        let one = self.emit.is_some();
        let dense = self.dense.is_some();
        match kind {
            Kind::Events if dense => self.events::<2, _>(src),
            Kind::Events if one => self.events::<1, _>(src),
            Kind::Events => self.events::<0, _>(src),
            Kind::Clicks if dense => self.clicks::<2, _>(src),
            Kind::Clicks if one => self.clicks::<1, _>(src),
            Kind::Clicks => self.clicks::<0, _>(src),
            Kind::Links => unreachable!(),
        }
    }
}

/// Where `events` and `clicks` get their lines: `f(base, bounds, known)`
/// for each well-formed line of `N` fields, the bounds relative to `base`,
/// `known` when both ID fields are bytes in 1..=127. Returns a position
/// whose meaning depends on the source.
trait Lines {
    unsafe fn run<const N: usize>(self, f: impl FnMut(*const u8, &Bounds, bool)) -> usize;
}

/// The lines of a window that stage 1 has scanned, starting before
/// `limit`. Returns the offset of the first line left.
struct Scanned<'s> {
    base: *const u8,
    scan: &'s Scan,
    limit: u32,
}

impl Lines for Scanned<'_> {
    #[inline(always)]
    unsafe fn run<const N: usize>(self, mut f: impl FnMut(*const u8, &Bounds, bool)) -> usize {
        let (base, known) = (self.base, self.scan.clean);
        for_each_line_b::<N>(self.scan, self.limit, |b| f(base, b, known)) as usize
    }
}

/// The lines of a mapped file from `pos` (a line start) that start before
/// `end`, one at a time by `fuse`. A line that `fuse` refuses goes through
/// stage 1 and 2 on a small window of its own, so it is decided exactly as
/// before. Returns where it stopped: at or past `end`, or at a line that
/// no small window holds whole (too long, or the last line of the file
/// without a line break).
struct Fused<'s> {
    m: Mapped,
    pos: usize,
    end: usize,
    scan: &'s mut Scan,
}

/// The window a refused line is read with.
const REFUSED_WINDOW: usize = 256;
/// How far ahead of the reader the input is prefetched.
const FUSED_AHEAD: usize = 1024;

impl Lines for Fused<'_> {
    #[inline(always)]
    unsafe fn run<const N: usize>(self, mut f: impl FnMut(*const u8, &Bounds, bool)) -> usize {
        let Fused { m, mut pos, end, scan } = self;
        let mut b = Bounds::new();
        while pos < end {
            let p = m.ptr.add(pos);
            sys::prefetch(p.wrapping_add(FUSED_AHEAD));
            sys::prefetch(p.wrapping_add(FUSED_AHEAD + 64));
            let mut known = false;
            let nl = if N == 7 { fuse::event_line(p, &mut b, &mut known) } else { fuse::click_line(p, &mut b, &mut known) };
            let mut used = nl as usize + 1;
            if nl == fuse::REFUSED {
                // One call site for `f`, so it is inlined.
                (used, known) = refused_line::<N>(scan, m, pos, &mut b);
                if used == 0 {
                    return pos;
                }
                if !known {
                    pos += used;
                    continue;
                }
                known = scan.clean;
            }
            f(p, &b, known);
            pos += used;
        }
        pos
    }
}

/// A line the one-pass reader refused, through stage 1 and 2 on a window
/// of its own. Returns its length with the line break (0 when the window
/// does not hold it whole) and whether it is a well-formed line, whose
/// bounds are then in `out`.
#[cold]
#[inline(never)]
unsafe fn refused_line<const N: usize>(scan: &mut Scan, m: Mapped, pos: usize, out: &mut Bounds) -> (usize, bool) {
    let p = m.ptr.add(pos);
    scan.scan(p, REFUSED_WINDOW.min(m.len - pos));
    let mut good = false;
    let used = for_each_line_b::<N>(scan, 1, |b| {
        out.s = b.s;
        out.n = b.n;
        good = true;
    }) as usize;
    (used, good)
}

struct Parser<'a> {
    st: ParseState<'a>,
    scan: Scan,
    /// Event and click lines by the one-pass reader (`fuse`).
    fused: bool,
}

unsafe fn find_newline(p: *const u8, len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let hit = libc::memchr(p as *const libc::c_void, i32::from(b'\n'), len);
    if hit.is_null() {
        None
    } else {
        Some(hit as usize - p as usize)
    }
}

/// Stage 1 as a function of its own. With one caller the compiler would
/// inline the scan loop into `run_chunk`, where it measures 3% slower.
#[inline(never)]
unsafe fn scan_window(scan: &mut Scan, base: *const u8, len: usize) {
    scan.scan(base, len);
}

fn find_byte(hay: &[u8], byte: u8) -> Option<usize> {
    if hay.is_empty() {
        return None;
    }
    let hit = unsafe { libc::memchr(hay.as_ptr() as *const libc::c_void, i32::from(byte), hay.len()) };
    if hit.is_null() {
        None
    } else {
        Some(hit as usize - hay.as_ptr() as usize)
    }
}

/// The `N` fields of a line, quotes stripped, or `None` when the line is
/// malformed: what `split_line` answers, for a line of any length. It gives
/// up at the first comma too many, so a long run of commas costs nothing.
fn split_long<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    let mut out: [&[u8]; N] = [&[]; N];
    let mut rest = line;
    for (i, slot) in out.iter_mut().enumerate() {
        let field = match find_byte(rest, b',') {
            Some(at) if i + 1 < N => {
                let field = &rest[..at];
                rest = &rest[at + 1..];
                field
            }
            None if i + 1 == N => rest,
            _ => return None,
        };
        *slot = match find_byte(field, b'"') {
            None => field,
            // The only legal quotes are one pair around the whole field.
            Some(0) if field.len() >= 2 && field[field.len() - 1] == b'"' => {
                let inner = &field[1..field.len() - 1];
                if find_byte(inner, b'"').is_some() {
                    return None;
                }
                inner
            }
            Some(_) => return None,
        };
    }
    Some(out)
}

impl<'a> Parser<'a> {
    /// Parse the lines of `m` that start in `[a, b)`. The header line
    /// (the one starting at 0) is skipped.
    unsafe fn run_chunk(&mut self, m: Mapped, a: usize, b: usize, kind: Kind) {
        let len = m.len;
        let from = a.saturating_sub(1);
        // A line starts in `[a, b)` exactly when a line break sits in
        // `[a - 1, b - 1)`, so the search stops at `b`: a chunk in the
        // middle of a very long line must not read on to that line's end.
        let Some(first) = find_newline(m.ptr.add(from), b.min(len) - from) else {
            return;
        };
        let mut pos = from + first + 1;
        if self.fused && kind != Kind::Links {
            while pos < b {
                let t = prof::start();
                pos = self.st.consume_fused(kind, m, pos, b, &mut self.scan);
                prof::add(1, t);
                if pos < b {
                    let end = find_newline(m.ptr.add(pos), len - pos).map_or(len, |i| pos + i);
                    self.st.long_line(std::slice::from_raw_parts(m.ptr.add(pos), end - pos), kind);
                    pos = end + 1;
                }
            }
            return;
        }
        while pos < b {
            let wlen = WINDOW.min(len - pos).min(b - pos + WINDOW_SLACK);
            let base = m.ptr.add(pos);
            let t = prof::start();
            scan_window(&mut self.scan, base, wlen);
            let t = prof::add(0, t);
            let limit = (b - pos).min(u32::MAX as usize) as u32;
            let used = self.st.consume(kind, base, &self.scan, limit);
            prof::add(1, t);
            if used > 0 {
                pos += used as usize;
                continue;
            }
            // No complete line in the window: a very long line, or the last
            // line of a file that does not end in a newline.
            let after = pos + wlen;
            let end = find_newline(m.ptr.add(after), len - after).map_or(len, |i| after + i);
            self.st.long_line(std::slice::from_raw_parts(base, end - pos), kind);
            pos = end + 1;
        }
    }
}

// ---------------------------------------------------------------- phase 2

struct Resolver<'a> {
    links: &'a LinkMap,
    table: Vec<u32>,
    /// The entry chunks of the partition at hand, by number from 1.
    bases: Vec<*mut Entry>,
    ev: Scatter<Ev>,
    cl: Scatter<Click>,
    shift2: u32,
    entry_chunk_bytes: usize,
}

/// Entries per chunk fit in this many bits of a dedup slot.
const OFF_BITS: u32 = 9;
const OFF_MASK: u32 = (1 << OFF_BITS) - 1;
const _: () = assert!((CHUNK_BYTES - 64) / std::mem::size_of::<Entry>() <= 1 << OFF_BITS);
/// Dedup is the first reader of entries the parser wrote with streaming
/// stores, so they come from memory, and the hardware prefetcher starts
/// over at every 4 KiB page and every chunk. Measured at 20M events:
/// 256, 512, 1024 and 2048 bytes ahead give 374, 367, 355 and 366 CPU ms
/// of dedup against 404 to 415 without.
const DEDUP_AHEAD: usize = 1024;
/// The same read-ahead for the second walk of the entries (person lookup).
const EMIT_AHEAD: usize = 1024;
/// And for the click blocks, which the parser wrote long before.
const CLICK_AHEAD: usize = 1024;
/// Link lookups run in blocks of this many records, one stage at a time.
const BLOCK: usize = 128;
/// Lookups a one-pass parser runs as one block. Inside the parser the
/// lookups compete with the input reads and three scatters; measured at
/// 20M events, blocks of 64, 128, 256 and 1024 cost 3940, 3470, 3300 and
/// about 3250 CPU ms of parse (2048 and 4096 no better).
const EMIT_BLOCK: usize = 1024;

impl<'a> Resolver<'a> {
    /// Dedup one event partition and scatter its surviving valid events by
    /// person.
    unsafe fn events(&mut self, lists: &[List<Entry>]) {
        let t = prof::start();
        let n: usize = lists.iter().map(|l| l.count()).sum();
        if n == 0 {
            return;
        }
        // Slots of 32 bits: a tag from the hash above the position of the
        // entry, `chunk << OFF_BITS | offset`, chunks counted from 1 so a
        // used slot is never 0. Half the bytes of a pointer table at the
        // same load. A partition has far fewer than 2^23 chunks (the
        // events are under 16 GB), so the position always fits; the tag
        // gets what is left and only ever filters: equality is decided by
        // the full key.
        let chunks: usize = lists.iter().map(|l| l.chunks as usize).sum();
        let pos_bits = usize::BITS - chunks.leading_zeros() + OFF_BITS;
        let pos_mask = if pos_bits >= 32 { u32::MAX } else { (1u32 << pos_bits) - 1 };
        let cap = (n * 2).next_power_of_two().max(64);
        self.table.clear();
        self.table.resize(cap, 0);
        self.bases.clear();
        self.bases.push(std::ptr::null_mut());
        let mask = cap - 1;
        let table = self.table.as_mut_ptr();
        let bases = &mut self.bases;
        for list in lists {
            list.for_each_chunk_ahead(|chunk, next| {
                let first = (bases.len() as u32) << OFF_BITS;
                bases.push(chunk.as_mut_ptr());
                let bases = bases.as_ptr();
                // Ask for the entries a little ahead, into the next chunk at
                // the end of this one: `DEDUP_AHEAD` bytes past entry `k` is
                // `here + k` below `split` and `there + k` from it on, so
                // the loop runs in two parts with no per-entry select.
                let len = chunk.len();
                let split = len.saturating_sub(DEDUP_AHEAD / std::mem::size_of::<Entry>());
                let here = (chunk.as_ptr() as *const u8).wrapping_add(DEDUP_AHEAD) as *const Entry;
                let there = if next.is_null() { here } else { (next as *const Entry).wrapping_sub(split) };
                let data = chunk.as_mut_ptr();
                for (from, to, ahead) in [(0, split, here), (split, len, there)] {
                for k in from..to {
                    let entry = &mut *data.add(k);
                    prefetch(ahead.wrapping_add(k));
                    let h = mix1(entry.ekey);
                    // Low hash bits, apart from the home bits (20 and up).
                    let tag = ((h as u32) << (pos_bits & 31)) & !pos_mask;
                    let me = first | k as u32;
                    let mut i = (h >> 20) as usize & mask;
                    loop {
                        let slot = *table.add(i);
                        if slot == 0 {
                            *table.add(i) = tag | me;
                            break;
                        }
                        if slot & !pos_mask == tag {
                            let at = slot & pos_mask;
                            let other = (*bases.add((at >> OFF_BITS) as usize)).add((at & OFF_MASK) as usize);
                            if (*other).ekey == entry.ekey {
                                // Same event: the smaller ingest stays. On
                                // a tie the copies are identical.
                                if entry.ingest < (*other).ingest {
                                    (*other).w3 = DEAD;
                                    *table.add(i) = tag | me;
                                } else {
                                    entry.w3 = DEAD;
                                }
                                break;
                            }
                        }
                        i = (i + 1) & mask;
                    }
                }
                }
            });
        }
        let t = prof::add(4, t);
        // The link map is far bigger than the cache. Lookups go a block at
        // a time, each stage for the whole block, so the misses of a block
        // are all in flight together and every loop stays small.
        let links = *self.links;
        let shift2 = self.shift2;
        let ev = &mut self.ev;
        let hot = ev.hot();
        let mut hashes = [0u64; BLOCK];
        let mut starts = [0usize; BLOCK];
        for list in lists {
            list.for_each_chunk_ahead(|chunk, next| {
                // The entries left the cache during the dedup walk: read
                // ahead again, across chunk ends.
                // As in the dedup walk: `EMIT_AHEAD` bytes past entry `k`.
                let len = chunk.len();
                let split = len.saturating_sub(EMIT_AHEAD / std::mem::size_of::<Entry>());
                let here = (chunk.as_ptr() as *const u8).wrapping_add(EMIT_AHEAD) as *const Entry;
                let there = if next.is_null() { here } else { (next as *const Entry).wrapping_sub(split) };
                for (b, block) in chunk.chunks(BLOCK).enumerate() {
                    let (k0, k1) = (b * BLOCK, b * BLOCK + block.len());
                    let mid = split.clamp(k0, k1);
                    for (from, to, ahead) in [(k0, mid, here), (mid, k1, there)] {
                        for k in from..to {
                            prefetch(ahead.wrapping_add(k));
                            *hashes.get_unchecked_mut(k - k0) = links.stage1((*block.as_ptr().add(k - k0)).ukey);
                        }
                    }
                    for (at, &h) in starts.iter_mut().zip(&hashes[..block.len()]) {
                        *at = links.stage2(h);
                    }
                    for (&at, entry) in starts.iter().zip(block) {
                        if entry.w3 != DEAD {
                            let pkey = links.stage3(at, entry.ukey);
                            ev.push_hot(&hot, person_part(pkey, shift2), Ev { pkey, v: entry.w3 });
                        }
                    }
                }
            });
        }
        // The entries of this partition are dead: their chunks become the
        // next output chunks, so resolve needs almost no fresh memory.
        let bytes = self.entry_chunk_bytes;
        for list in lists {
            list.release(|chunk| {
                if self.cl.spare() < 16 {
                    self.cl.recycle(chunk, bytes);
                } else {
                    self.ev.recycle(chunk, bytes);
                }
            });
        }
        prof::add(5, t);
    }

    unsafe fn clicks(&mut self, raw: &ClickBlock) {
        let t = prof::start();
        // Event jobs all come before click jobs, so the event scatter has
        // no more use for its spare chunks. The first block alone needs a
        // chunk for nearly every partition.
        self.ev.give_spare(&mut self.cl);
        let block = std::slice::from_raw_parts(raw.ptr, raw.len);
        let links = *self.links;
        let shift2 = self.shift2;
        let cl = &mut self.cl;
        let hot = cl.hot();
        let mut hashes = [0u64; BLOCK];
        let mut starts = [0usize; BLOCK];
        for part in block.chunks(BLOCK) {
            for (h, click) in hashes.iter_mut().zip(part) {
                // A prefetch past the end of the block is harmless.
                prefetch((click as *const Click as *const u8).wrapping_add(CLICK_AHEAD));
                *h = links.stage1(click.key);
            }
            for (at, &h) in starts.iter_mut().zip(&hashes[..part.len()]) {
                *at = links.stage2(h);
            }
            for (&at, click) in starts.iter().zip(part) {
                let pkey = links.stage3(at, click.key);
                cl.push_hot(&hot, person_part(pkey, shift2), Click { key: pkey, ckey: click.ckey, tc: click.tc });
            }
        }
        // The block is dead now. Its pages are faulted in already, so its
        // memory becomes click chunks instead of fresh pages. Big blocks come
        // from the bump allocator, which never reuses or frees them.
        let bytes = self.cl.chunk_bytes();
        let total = raw.cap * std::mem::size_of::<Click>();
        let base = raw.ptr as *mut u8;
        if total >= crate::alloc::BIG && base as usize % 64 == 0 {
            for k in 0..total / bytes {
                self.cl.recycle(base.add(k * bytes), bytes);
            }
        }
        prof::add(6, t);
    }
}

/// A click block of the parse phase, owned by the resolve phase: the
/// resolver that takes it reuses its memory once the clicks are scattered.
struct ClickBlock {
    ptr: *mut Click,
    len: usize,
    cap: usize,
}
unsafe impl Send for ClickBlock {}
unsafe impl Sync for ClickBlock {}

impl ClickBlock {
    fn from_vec(v: Vec<Click>) -> Self {
        let mut v = std::mem::ManuallyDrop::new(v);
        ClickBlock { ptr: v.as_mut_ptr(), len: v.len(), cap: v.capacity() }
    }
}

// ---------------------------------------------------------------- one-pass dedup

/// Notes per chunk fit in this many bits of a dedup slot.
const NOTE_OFF_BITS: u32 = 10;
const NOTE_OFF_MASK: u32 = (1 << NOTE_OFF_BITS) - 1;
const _: () = assert!((CHUNK_BYTES - 64) / std::mem::size_of::<Note>() <= 1 << NOTE_OFF_BITS);

struct Deduper<'a> {
    dir: &'a sys::Directory,
    table: Vec<u32>,
    /// The note chunks of the partition at hand, by number from 1.
    bases: Vec<*mut Note>,
    ingest_base: i64,
    /// The records are the dense shape's 8-byte words.
    dense: bool,
}

impl<'a> Deduper<'a> {
    /// Mark the person record a losing copy wrote, if it wrote one.
    #[inline(always)]
    unsafe fn drop_record(&self, rec: u32) {
        if rec != NO_RECORD {
            if self.dense {
                *self.dir.record::<u64>(rec) = dense::DROPPED;
            } else {
                (*self.dir.record::<Ev>(rec)).v = DEAD;
            }
        }
    }

    /// Dedup one event-key partition: of the copies of one event, the one
    /// with the smallest ingest stays, and the person record of every other
    /// copy is marked dropped. If the copy that stays is invalid it has no
    /// record, so the event is gone, as the rules ask: dedup comes before
    /// validation. Copies with equal ingest are identical; the first seen
    /// stays. The table is the one `Resolver::events` uses.
    unsafe fn partition(&mut self, lists: &[List<Note>], wide: &[Wide]) {
        let t = prof::start();
        let n: usize = lists.iter().map(|l| l.count()).sum();
        if n + wide.len() == 0 {
            return;
        }
        let chunks: usize = lists.iter().map(|l| l.chunks as usize).sum();
        let pos_bits = usize::BITS - chunks.leading_zeros() + NOTE_OFF_BITS;
        let pos_mask = if pos_bits >= 32 { u32::MAX } else { (1u32 << pos_bits) - 1 };
        let cap = (n * 2).next_power_of_two().max(64);
        self.table.clear();
        self.table.resize(cap, 0);
        self.bases.clear();
        self.bases.push(std::ptr::null_mut());
        let mask = cap - 1;
        let table = self.table.as_mut_ptr();
        for list in lists {
            list.for_each_chunk_ahead(|chunk, next| {
                let end = chunk.as_ptr().add(chunk.len()) as usize;
                let first = (self.bases.len() as u32) << NOTE_OFF_BITS;
                self.bases.push(chunk.as_mut_ptr());
                let bases = self.bases.as_ptr();
                for (k, note) in chunk.iter().enumerate() {
                    let ahead = note as *const Note as usize + DEDUP_AHEAD;
                    let ahead = if ahead < end || next.is_null() { ahead } else { next as usize + (ahead - end) };
                    prefetch(ahead as *const u8);
                    let h = mix1(note.ekey);
                    let tag = ((h as u32) << (pos_bits & 31)) & !pos_mask;
                    let me = first | k as u32;
                    let mut i = (h >> 20) as usize & mask;
                    loop {
                        let slot = *table.add(i);
                        if slot == 0 {
                            *table.add(i) = tag | me;
                            break;
                        }
                        if slot & !pos_mask == tag {
                            let at = slot & pos_mask;
                            let other = (*bases.add((at >> NOTE_OFF_BITS) as usize)).add((at & NOTE_OFF_MASK) as usize);
                            if (*other).ekey == note.ekey {
                                // Same event: the smaller ingest stays.
                                if note.ingest < (*other).ingest {
                                    self.drop_record((*other).rec);
                                    *table.add(i) = tag | me;
                                } else {
                                    self.drop_record(note.rec);
                                }
                                break;
                            }
                        }
                        i = (i + 1) & mask;
                    }
                }
            });
        }
        if !wide.is_empty() {
            self.wide(wide, pos_bits, pos_mask, mask);
        }
        prof::add(4, t);
    }

    /// The copies whose ingest did not fit a note, against the winners of
    /// the table and each other, with exact ingest values.
    #[cold]
    #[inline(never)]
    unsafe fn wide(&mut self, wide: &[Wide], pos_bits: u32, pos_mask: u32, mask: usize) {
        use std::collections::HashMap;
        // Per key: the exact ingest and record of the copy that stays so
        // far, from the table or from this list.
        let mut best: HashMap<u64, (i64, u32)> = HashMap::new();
        let table = self.table.as_ptr();
        let exact = |ingest: u32| -> i64 {
            if ingest == u32::MAX {
                i64::MAX
            } else {
                self.ingest_base + i64::from(ingest) * 1000
            }
        };
        for w in wide {
            let current = match best.get(&w.ekey) {
                Some(&b) => Some(b),
                None => {
                    // The winner among the notes, if the key has any.
                    let h = mix1(w.ekey);
                    let tag = ((h as u32) << (pos_bits & 31)) & !pos_mask;
                    let mut i = (h >> 20) as usize & mask;
                    let mut found = None;
                    loop {
                        let slot = *table.add(i);
                        if slot == 0 {
                            break;
                        }
                        if slot & !pos_mask == tag {
                            let at = slot & pos_mask;
                            let other = (*self.bases.as_ptr().add((at >> NOTE_OFF_BITS) as usize)).add((at & NOTE_OFF_MASK) as usize);
                            if (*other).ekey == w.ekey {
                                found = Some((exact((*other).ingest), (*other).rec));
                                break;
                            }
                        }
                        i = (i + 1) & mask;
                    }
                    found
                }
            };
            let stays = match current {
                None => (w.ingest, w.rec),
                Some((ingest, rec)) if w.ingest < ingest => {
                    self.drop_record(rec);
                    (w.ingest, w.rec)
                }
                Some(kept) => {
                    self.drop_record(w.rec);
                    kept
                }
            };
            best.insert(w.ekey, stays);
        }
    }
}

// ---------------------------------------------------------------- phase 3

struct PersonPhase<'a> {
    fast: &'a Fast<'a>,
    /// Person key -> local id: an open-addressing table of `id + 1`, 0 for
    /// an empty slot, and the keys by `id + 1`. `keys[0]` is 0, which is no
    /// person, so one compare tells a hit from an empty or foreign slot.
    tab: Vec<u32>,
    keys: Vec<u64>,
    counts: Vec<u32>,
    /// One word per event, `ts_rel << 32 | local person id`, in input
    /// order.
    words: Vec<u64>,
    /// `ts_rel` of every event, person after person, in input order.
    ev: Vec<u32>,
    /// The times of the person at hand, sorted, and scratch for sorting.
    sorted: Vec<u32>,
    sort_tmp: Vec<u32>,
    /// The purchases: `(value, local person id)` as they come, then the
    /// values person after person, with the end offset of each person.
    buys: Vec<(u64, u32)>,
    bcounts: Vec<u32>,
    bought: Vec<u64>,
    ccounts: Vec<u32>,
    cids: Vec<u32>,
    cl: Vec<(u64, u64)>,
    /// Per campaign, for the person at hand: how many of its clicks are in
    /// reach of the current session, and the session and day slot at which
    /// it came into reach.
    reach: Vec<[u32; 4]>,
    /// The person's days so far, in order: the day, and the index of the
    /// first session of that day. One slot past the last holds `u32::MAX`.
    slot_day: Vec<u32>,
    slot_first: Vec<u32>,
    /// The sessions of the person at hand, when the person has clicks: the
    /// start on the click clock, the day and the revenue.
    s_at: Vec<u32>,
    s_day: Vec<u32>,
    s_revenue: Vec<u64>,
    /// Per campaign: the sessions of the person it assists, one bit each
    /// (people with few sessions only).
    cover: Vec<u64>,
    catsum: Vec<u64>,
    touched: Vec<u8>,
    /// (campaign, day) -> [sessions | purchasing << 32, revenue]. Half the
    /// bytes of four words, and one add per session for both counts: a
    /// thread's cell counts stay far below 2^32 (fewer than 2^30 events).
    agg: Vec<[u64; 2]>,
    /// (campaign, day) -> assisted sessions. Apart from `agg` and narrow:
    /// these cells are hit at random, far more often than the others, and
    /// this way they all stay in the cache.
    assists: Vec<u32>,
    out: UserOut<'a>,
    /// Summed sessions and revenue of the rows emitted, for the self-check.
    check: (u64, u64),
    /// Person records may be marked `DEAD` (one-pass dedup); the
    /// three-phase resolve never writes one.
    dropped: bool,
}

/// The events of a person are only times here: each person's times are
/// sorted on their own (`sort_times`), and a session ends where the next
/// time is more than the gap later. The purchases (3% of the events)
/// travel on the side, sorted by time per person, and a session takes
/// those up to its last event when it closes.
///
/// Times past the events of a person are read but never used: `ev` has
/// this much padding.
const EV_PAD: usize = 64;

/// The hash of a person key inside one partition: the top bits of one
/// multiply. The partition was chosen by `mix`, which shares no constant
/// with this.
#[inline(always)]
fn person_hash(key: u64) -> u64 {
    key.wrapping_mul(0xD6E8_FEB8_6659_FD93)
}

/// One session in the first word of a campaign cell (purchasing sessions
/// count in its top half).
const SESSION: u64 = 1;

/// The campaign cells of a thread, or their sum over threads.
#[derive(Default)]
struct Cells {
    agg: Vec<[u64; 2]>,
    assists: Vec<u32>,
}

impl Cells {
    /// Add `other` in. The sums stay exact: all sessions of a run number
    /// fewer than 2^30, so the low half of the first word never carries.
    fn add(&mut self, other: Cells) {
        if self.agg.is_empty() {
            *self = other;
            return;
        }
        for (a, b) in self.agg.iter_mut().zip(&other.agg) {
            a[0] += b[0];
            a[1] += b[1];
        }
        for (a, b) in self.assists.iter_mut().zip(&other.assists) {
            *a += b;
        }
    }

    /// `[sessions, purchasing, revenue, assisted]` of cell `i`.
    fn get(&self, i: usize) -> [u64; 4] {
        let [w, revenue] = self.agg[i];
        [w & 0xFFFF_FFFF, w >> 32, revenue, u64::from(self.assists[i])]
    }
}

/// Sessions are compared with a click eight at a time.
const LANES: usize = 8;
/// The most sessions a person can have for their clicks to be attributed
/// without sorting them (`PersonPhase::few_sessions`).
const FEW: usize = 32;

/// One click against a block of `LANES` sessions. `at` holds their starts,
/// `best_t` and `best_c` the time plus one and the campaign of the latest
/// click in reach of each so far. The click is at `t`, of campaign `camp`,
/// and reaches the starts in `t..=upto`. All times are under 2^31.
///
/// Returns the sessions of the block the click reaches, one bit each, and
/// adds to `tie` whether the click ties with the latest click of a session
/// while being of another campaign (`tied` reads it out at the end).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn click_block(at: *const u32, best_t: *mut u32, best_c: *mut u32, t: u32, upto: u32, camp: u32, tie_acc: &mut Tie) -> u64 {
    use core::arch::x86_64::*;
    let starts = _mm256_loadu_si256(at as *const __m256i);
    let before = _mm256_cmpgt_epi32(_mm256_set1_epi32(t as i32), starts);
    let after = _mm256_cmpgt_epi32(starts, _mm256_set1_epi32(upto as i32));
    let out = _mm256_or_si256(before, after);
    let key = _mm256_set1_epi32(t as i32 + 1);
    let camps = _mm256_set1_epi32(camp as i32);
    let old_t = _mm256_loadu_si256(best_t as *const __m256i);
    let old_c = _mm256_loadu_si256(best_c as *const __m256i);
    let later = _mm256_andnot_si256(out, _mm256_cmpgt_epi32(key, old_t));
    let same = _mm256_andnot_si256(out, _mm256_cmpeq_epi32(key, old_t));
    let tie = _mm256_andnot_si256(_mm256_cmpeq_epi32(old_c, camps), same);
    _mm256_storeu_si256(best_t as *mut __m256i, _mm256_blendv_epi8(old_t, key, later));
    _mm256_storeu_si256(best_c as *mut __m256i, _mm256_blendv_epi8(old_c, camps, later));
    let reached = !_mm256_movemask_ps(_mm256_castsi256_ps(out)) as u32 & 0xFF;
    *tie_acc = _mm256_or_si256(*tie_acc, tie);
    u64::from(reached)
}

/// The ties `click_block` saw, kept as lanes so the test is done once.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
type Tie = core::arch::x86_64::__m256i;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
type Tie = bool;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn no_tie() -> Tie {
    unsafe { core::arch::x86_64::_mm256_setzero_si256() }
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
fn no_tie() -> Tie {
    false
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn tied(t: Tie) -> bool {
    unsafe { core::arch::x86_64::_mm256_testz_si256(t, t) == 0 }
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
fn tied(t: Tie) -> bool {
    t
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
unsafe fn click_block(at: *const u32, best_t: *mut u32, best_c: *mut u32, t: u32, upto: u32, camp: u32, tie_acc: &mut Tie) -> u64 {
    let key = t + 1;
    let (mut reached, mut tie) = (0u64, false);
    for k in 0..LANES {
        let start = *at.add(k);
        let covers = (start >= t) & (start <= upto);
        reached |= u64::from(covers) << k;
        let (old_t, old_c) = (*best_t.add(k), *best_c.add(k));
        let later = covers & (key > old_t);
        tie |= covers & (key == old_t) & (old_c != camp);
        *best_t.add(k) = if later { key } else { old_t };
        *best_c.add(k) = if later { camp } else { old_c };
    }
    *tie_acc |= tie;
    reached
}

/// Bitonic sorting network over 8 times in 1 register, ascending
/// (generated: element `i` is lane `i % 8` of register `i / 8`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn bitonic1(v: &mut [core::arch::x86_64::__m256i; 1]) {
    use core::arch::x86_64::*;
    let mut v0 = v[0];
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    *v = [v0];
}

/// Bitonic sorting network over 16 times in 2 registers, ascending
/// (generated: element `i` is lane `i % 8` of register `i / 8`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn bitonic2(v: &mut [core::arch::x86_64::__m256i; 2]) {
    use core::arch::x86_64::*;
    let (mut v0, mut v1) = (v[0], v[1]);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x55);
    let lo = _mm256_min_epu32(v0, v1);
    let hi = _mm256_max_epu32(v0, v1);
    v0 = lo;
    v1 = hi;
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xAA);
    *v = [v0, v1];
}

/// Bitonic sorting network over 32 times in 4 registers, ascending
/// (generated: element `i` is lane `i % 8` of register `i / 8`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn bitonic4(v: &mut [core::arch::x86_64::__m256i; 4]) {
    use core::arch::x86_64::*;
    let (mut v0, mut v1, mut v2, mut v3) = (v[0], v[1], v[2], v[3]);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x55);
    let lo = _mm256_min_epu32(v0, v1);
    let hi = _mm256_max_epu32(v0, v1);
    v0 = lo;
    v1 = hi;
    let lo = _mm256_min_epu32(v2, v3);
    let hi = _mm256_max_epu32(v2, v3);
    v2 = hi;
    v3 = lo;
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x55);
    let lo = _mm256_min_epu32(v0, v2);
    let hi = _mm256_max_epu32(v0, v2);
    v0 = lo;
    v2 = hi;
    let lo = _mm256_min_epu32(v1, v3);
    let hi = _mm256_max_epu32(v1, v3);
    v1 = lo;
    v3 = hi;
    let lo = _mm256_min_epu32(v0, v1);
    let hi = _mm256_max_epu32(v0, v1);
    v0 = lo;
    v1 = hi;
    let lo = _mm256_min_epu32(v2, v3);
    let hi = _mm256_max_epu32(v2, v3);
    v2 = lo;
    v3 = hi;
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0xAA);
    *v = [v0, v1, v2, v3];
}

/// Bitonic sorting network over 64 times in 8 registers, ascending
/// (generated: element `i` is lane `i % 8` of register `i / 8`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn bitonic8(v: &mut [core::arch::x86_64::__m256i; 8]) {
    use core::arch::x86_64::*;
    let (mut v0, mut v1, mut v2, mut v3, mut v4, mut v5, mut v6, mut v7) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v4, 0xB1);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v5, 0xB1);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v6, 0xB1);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v7, 0xB1);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x66);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v4, 0x4E);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v5, 0x4E);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v6, 0x4E);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v7, 0x4E);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x3C);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v4, 0xB1);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v5, 0xB1);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v6, 0xB1);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_shuffle_epi32(v7, 0xB1);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x5A);
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v4, 0x4E);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v5, 0x4E);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v6, 0x4E);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v7, 0x4E);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v4, 0x4E);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v5, 0x4E);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v6, 0x4E);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v7, 0x4E);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v4, 0xB1);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v5, 0xB1);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v6, 0xB1);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v7, 0xB1);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x55);
    let lo = _mm256_min_epu32(v0, v1);
    let hi = _mm256_max_epu32(v0, v1);
    v0 = lo;
    v1 = hi;
    let lo = _mm256_min_epu32(v2, v3);
    let hi = _mm256_max_epu32(v2, v3);
    v2 = hi;
    v3 = lo;
    let lo = _mm256_min_epu32(v4, v5);
    let hi = _mm256_max_epu32(v4, v5);
    v4 = lo;
    v5 = hi;
    let lo = _mm256_min_epu32(v6, v7);
    let hi = _mm256_max_epu32(v6, v7);
    v6 = hi;
    v7 = lo;
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v4, 0x4E);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v5, 0x4E);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v6, 0x4E);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v7, 0x4E);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v4, 0x4E);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v5, 0x4E);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v6, 0x4E);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v7, 0x4E);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v4, 0xB1);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v5, 0xB1);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v6, 0xB1);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v7, 0xB1);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x55);
    let lo = _mm256_min_epu32(v0, v2);
    let hi = _mm256_max_epu32(v0, v2);
    v0 = lo;
    v2 = hi;
    let lo = _mm256_min_epu32(v1, v3);
    let hi = _mm256_max_epu32(v1, v3);
    v1 = lo;
    v3 = hi;
    let lo = _mm256_min_epu32(v4, v6);
    let hi = _mm256_max_epu32(v4, v6);
    v4 = hi;
    v6 = lo;
    let lo = _mm256_min_epu32(v5, v7);
    let hi = _mm256_max_epu32(v5, v7);
    v5 = hi;
    v7 = lo;
    let lo = _mm256_min_epu32(v0, v1);
    let hi = _mm256_max_epu32(v0, v1);
    v0 = lo;
    v1 = hi;
    let lo = _mm256_min_epu32(v2, v3);
    let hi = _mm256_max_epu32(v2, v3);
    v2 = lo;
    v3 = hi;
    let lo = _mm256_min_epu32(v4, v5);
    let hi = _mm256_max_epu32(v4, v5);
    v4 = hi;
    v5 = lo;
    let lo = _mm256_min_epu32(v6, v7);
    let hi = _mm256_max_epu32(v6, v7);
    v6 = hi;
    v7 = lo;
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v4, 0x4E);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v5, 0x4E);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v6, 0x4E);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_permute4x64_epi64(v7, 0x4E);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x0F);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v4, 0x4E);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v5, 0x4E);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v6, 0x4E);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v7, 0x4E);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x33);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v4, 0xB1);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v5, 0xB1);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v6, 0xB1);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0x55);
    let other = _mm256_shuffle_epi32(v7, 0xB1);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0x55);
    let lo = _mm256_min_epu32(v0, v4);
    let hi = _mm256_max_epu32(v0, v4);
    v0 = lo;
    v4 = hi;
    let lo = _mm256_min_epu32(v1, v5);
    let hi = _mm256_max_epu32(v1, v5);
    v1 = lo;
    v5 = hi;
    let lo = _mm256_min_epu32(v2, v6);
    let hi = _mm256_max_epu32(v2, v6);
    v2 = lo;
    v6 = hi;
    let lo = _mm256_min_epu32(v3, v7);
    let hi = _mm256_max_epu32(v3, v7);
    v3 = lo;
    v7 = hi;
    let lo = _mm256_min_epu32(v0, v2);
    let hi = _mm256_max_epu32(v0, v2);
    v0 = lo;
    v2 = hi;
    let lo = _mm256_min_epu32(v1, v3);
    let hi = _mm256_max_epu32(v1, v3);
    v1 = lo;
    v3 = hi;
    let lo = _mm256_min_epu32(v4, v6);
    let hi = _mm256_max_epu32(v4, v6);
    v4 = lo;
    v6 = hi;
    let lo = _mm256_min_epu32(v5, v7);
    let hi = _mm256_max_epu32(v5, v7);
    v5 = lo;
    v7 = hi;
    let lo = _mm256_min_epu32(v0, v1);
    let hi = _mm256_max_epu32(v0, v1);
    v0 = lo;
    v1 = hi;
    let lo = _mm256_min_epu32(v2, v3);
    let hi = _mm256_max_epu32(v2, v3);
    v2 = lo;
    v3 = hi;
    let lo = _mm256_min_epu32(v4, v5);
    let hi = _mm256_max_epu32(v4, v5);
    v4 = lo;
    v5 = hi;
    let lo = _mm256_min_epu32(v6, v7);
    let hi = _mm256_max_epu32(v6, v7);
    v6 = lo;
    v7 = hi;
    let other = _mm256_permute4x64_epi64(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v4, 0x4E);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v5, 0x4E);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v6, 0x4E);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_permute4x64_epi64(v7, 0x4E);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0xF0);
    let other = _mm256_shuffle_epi32(v0, 0x4E);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v1, 0x4E);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v2, 0x4E);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v3, 0x4E);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v4, 0x4E);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v5, 0x4E);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v6, 0x4E);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v7, 0x4E);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0xCC);
    let other = _mm256_shuffle_epi32(v0, 0xB1);
    let lo = _mm256_min_epu32(v0, other);
    let hi = _mm256_max_epu32(v0, other);
    v0 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v1, 0xB1);
    let lo = _mm256_min_epu32(v1, other);
    let hi = _mm256_max_epu32(v1, other);
    v1 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v2, 0xB1);
    let lo = _mm256_min_epu32(v2, other);
    let hi = _mm256_max_epu32(v2, other);
    v2 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v3, 0xB1);
    let lo = _mm256_min_epu32(v3, other);
    let hi = _mm256_max_epu32(v3, other);
    v3 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v4, 0xB1);
    let lo = _mm256_min_epu32(v4, other);
    let hi = _mm256_max_epu32(v4, other);
    v4 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v5, 0xB1);
    let lo = _mm256_min_epu32(v5, other);
    let hi = _mm256_max_epu32(v5, other);
    v5 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v6, 0xB1);
    let lo = _mm256_min_epu32(v6, other);
    let hi = _mm256_max_epu32(v6, other);
    v6 = _mm256_blend_epi32(lo, hi, 0xAA);
    let other = _mm256_shuffle_epi32(v7, 0xB1);
    let lo = _mm256_min_epu32(v7, other);
    let hi = _mm256_max_epu32(v7, other);
    v7 = _mm256_blend_epi32(lo, hi, 0xAA);
    *v = [v0, v1, v2, v3, v4, v5, v6, v7];
}

/// Sort the `n <= 8 * R` times at `src` into `8 * R` slots at `dst`, the
/// slots past `n` holding `u32::MAX`. Reads `8 * R` values at `src`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn sort_net<const R: usize>(src: *const u32, n: usize, dst: *mut u32) {
    use core::arch::x86_64::*;
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut v = [_mm256_setzero_si256(); R];
    for (r, reg) in v.iter_mut().enumerate() {
        let x = _mm256_loadu_si256(src.add(8 * r) as *const __m256i);
        let real = _mm256_cmpgt_epi32(_mm256_set1_epi32(n as i32 - 8 * r as i32), lane);
        *reg = _mm256_or_si256(x, _mm256_andnot_si256(real, _mm256_set1_epi32(-1)));
    }
    let p = v.as_mut_ptr();
    match R {
        1 => bitonic1(&mut *(p as *mut [__m256i; 1])),
        2 => bitonic2(&mut *(p as *mut [__m256i; 2])),
        4 => bitonic4(&mut *(p as *mut [__m256i; 4])),
        _ => bitonic8(&mut *(p as *mut [__m256i; 8])),
    }
    for (r, reg) in v.iter().enumerate() {
        _mm256_storeu_si256(dst.add(8 * r) as *mut __m256i, *reg);
    }
}

/// Merge two sorted runs into `out`, choosing with a select, not a branch.
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2")), allow(dead_code))]
#[inline(always)]
unsafe fn merge32(a: *const u32, na: usize, b: *const u32, nb: usize, out: *mut u32) {
    let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
    while i < na && j < nb {
        let (x, y) = (*a.add(i), *b.add(j));
        let take_b = y < x;
        *out.add(k) = if take_b { y } else { x };
        i += usize::from(!take_b);
        j += usize::from(take_b);
        k += 1;
    }
    std::ptr::copy_nonoverlapping(a.add(i), out.add(k), na - i);
    std::ptr::copy_nonoverlapping(b.add(j), out.add(k + na - i), nb - j);
}

/// Merge two sorted runs of `w` values each into `out` from both ends at
/// once: two independent chains of selects, so twice the work per load
/// latency. The front takes the `w` smallest, the back the `w` largest;
/// neither can run past its runs, since `w` steps take at most `w` values
/// of one run.
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2")), allow(dead_code))]
#[inline(always)]
unsafe fn merge32_even(a: *const u32, b: *const u32, w: usize, out: *mut u32) {
    let (mut i, mut j) = (0usize, 0usize);
    let (mut ia, mut jb) = (w - 1, w - 1);
    let mut back = 2 * w - 1;
    for k in 0..w {
        let (x, y) = (*a.add(i), *b.add(j));
        let take_b = y < x;
        *out.add(k) = if take_b { y } else { x };
        i += usize::from(!take_b);
        j += usize::from(take_b);
        let (x, y) = (*a.add(ia), *b.add(jb));
        let take_a = x > y;
        *out.add(back) = if take_a { x } else { y };
        ia = ia.wrapping_sub(usize::from(take_a));
        jb = jb.wrapping_sub(usize::from(!take_a));
        back = back.wrapping_sub(1);
    }
}

/// People with at least this many events are put in order of time bin
/// only (`bin_sort`).
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2")), allow(dead_code))]
const BIN_EVENTS: usize = 512;

/// Bins of 2^20 ms. Two events of one bin are under 2^20 ms apart, which is
/// less than the session gap, so a bin never holds a session break. A break
/// can only fall between two consecutive non-empty bins of a person, and it
/// does exactly when the smallest time of the later bin is more than the
/// gap after the largest time of the earlier one.
const BIN_SHIFT: u32 = 20;
const _: () = assert!((1i64 << BIN_SHIFT) - 1 <= SESSION_GAP_MS);

/// The run of events in the bin of `ev[i]` that starts at `i`, up to `n`:
/// its smallest and largest time and its end. `ev` is readable for
/// `LANES` values past `n`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn bin_run(ev: *const u32, i: usize, n: usize) -> (u32, u32, usize) {
    use core::arch::x86_64::*;
    let bin = _mm256_set1_epi32((*ev.add(i) >> BIN_SHIFT) as i32);
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let (mut lo, mut hi) = (_mm256_set1_epi32(-1), _mm256_setzero_si256());
    let mut j = i;
    loop {
        // A run of `LANES` goes on in the next block: rare, so a branch.
        let v = _mm256_loadu_si256(ev.add(j) as *const __m256i);
        let same = _mm256_cmpeq_epi32(_mm256_srli_epi32(v, BIN_SHIFT as i32), bin);
        let m = _bzhi_u32(_mm256_movemask_ps(_mm256_castsi256_ps(same)) as u32, (n - j).min(LANES) as u32);
        let len = (!m).trailing_zeros();
        let take = _mm256_cmpgt_epi32(_mm256_set1_epi32(len as i32), lane);
        lo = _mm256_min_epu32(lo, _mm256_or_si256(v, _mm256_xor_si256(take, _mm256_set1_epi32(-1))));
        hi = _mm256_max_epu32(hi, _mm256_and_si256(v, take));
        j += len as usize;
        if len < LANES as u32 {
            break;
        }
    }
    let lo = _mm256_min_epu32(lo, _mm256_permute2x128_si256(lo, lo, 1));
    let lo = _mm256_min_epu32(lo, _mm256_shuffle_epi32(lo, 0b01_00_11_10));
    let lo = _mm256_min_epu32(lo, _mm256_shuffle_epi32(lo, 0b10_11_00_01));
    let hi = _mm256_max_epu32(hi, _mm256_permute2x128_si256(hi, hi, 1));
    let hi = _mm256_max_epu32(hi, _mm256_shuffle_epi32(hi, 0b01_00_11_10));
    let hi = _mm256_max_epu32(hi, _mm256_shuffle_epi32(hi, 0b10_11_00_01));
    (_mm256_cvtsi256_si32(lo) as u32, _mm256_cvtsi256_si32(hi) as u32, j)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
#[inline(always)]
unsafe fn bin_run(ev: *const u32, i: usize, n: usize) -> (u32, u32, usize) {
    let bin = *ev.add(i) >> BIN_SHIFT;
    let (mut first, mut latest) = (u32::MAX, 0u32);
    let mut j = i;
    while j < n {
        let t = *ev.add(j);
        if t >> BIN_SHIFT != bin {
            break;
        }
        first = first.min(t);
        latest = latest.max(t);
        j += 1;
    }
    (first, latest, j)
}

/// The session of a person in bin order (`bin_sort`) that starts at `i`:
/// its end, first time and last time.
#[inline(always)]
unsafe fn session_end_binned(s: *const u32, i: usize, n: usize) -> (usize, u32, u32) {
    let (start, mut last, mut j) = bin_run(s, i, n);
    while j < n {
        let (lo, hi, next) = bin_run(s, j, n);
        if lo - last > SESSION_GAP_MS as u32 {
            break;
        }
        last = hi;
        j = next;
    }
    (j, start, last)
}

/// The `n` times at `src` into `dst` in order of their time bin, by one
/// counting pass over the bins the person spans. `hist` is scratch.
#[cfg_attr(not(all(target_arch = "x86_64", target_feature = "avx2")), allow(dead_code))]
#[inline(never)]
unsafe fn bin_sort(src: *const u32, n: usize, dst: *mut u32, hist: &mut Vec<u32>) {
    let times = std::slice::from_raw_parts(src, n);
    let lo = times.iter().fold(u32::MAX, |a, &t| a.min(t)) >> BIN_SHIFT;
    let hi = times.iter().fold(0, |a, &t| a.max(t)) >> BIN_SHIFT;
    hist.clear();
    hist.resize((hi - lo) as usize + 1, 0);
    for &t in times {
        *hist.get_unchecked_mut(((t >> BIN_SHIFT) - lo) as usize) += 1;
    }
    let mut sum = 0u32;
    for c in hist.iter_mut() {
        let k = *c;
        *c = sum;
        sum += k;
    }
    for &t in times {
        let w = hist.get_unchecked_mut(((t >> BIN_SHIFT) - lo) as usize);
        *dst.add(*w as usize) = t;
        *w += 1;
    }
}

/// The `n` times at `src` (readable for `EV_PAD` past `n`), sorted, or for
/// `BIN_EVENTS` and more in bin order only (then true). The result is
/// readable for `LANES` past `n`. Up to 64 times: one sorting network.
/// Below `BIN_EVENTS`: networks of 64, then merges without branches.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn sort_times(src: *const u32, n: usize, dst: &mut Vec<u32>, tmp: &mut Vec<u32>) -> (*const u32, bool) {
    let room = n.div_ceil(64) * 64 + LANES;
    if dst.capacity() < room {
        dst.reserve(room);
    }
    let d = dst.as_mut_ptr();
    if n <= 8 {
        sort_net::<1>(src, n, d);
        return (d, false);
    }
    if n <= 16 {
        sort_net::<2>(src, n, d);
        return (d, false);
    }
    sort_big(src, n, dst, tmp)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(never)]
unsafe fn sort_big(src: *const u32, n: usize, dst: &mut Vec<u32>, tmp: &mut Vec<u32>) -> (*const u32, bool) {
    let d = dst.as_mut_ptr();
    if n <= 32 {
        sort_net::<4>(src, n, d);
        return (d, false);
    }
    if n >= BIN_EVENTS {
        bin_sort(src, n, d, tmp);
        return (d, true);
    }
    let room = n.div_ceil(64) * 64 + LANES;
    if tmp.capacity() < room {
        tmp.reserve(room);
    }
    let t = tmp.as_mut_ptr();
    {
        let m = n.div_ceil(64) * 64;
        let mut at = 0;
        while at < n {
            sort_net::<8>(src.add(at), n - at, d.add(at));
            at += 64;
        }
        let (mut from, mut to) = (d, t);
        let mut width = 64;
        while width < m {
            let mut lo = 0;
            while lo < m {
                let mid = (lo + width).min(m);
                let hi = (lo + 2 * width).min(m);
                if hi - mid == width {
                    merge32_even(from.add(lo), from.add(mid), width, to.add(lo));
                } else {
                    merge32(from.add(lo), mid - lo, from.add(mid), hi - mid, to.add(lo));
                }
                lo = hi;
            }
            std::mem::swap(&mut from, &mut to);
            width *= 2;
        }
        (from, false)
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
unsafe fn sort_times(src: *const u32, n: usize, dst: &mut Vec<u32>, _tmp: &mut Vec<u32>) -> (*const u32, bool) {
    dst.clear();
    dst.extend_from_slice(std::slice::from_raw_parts(src, n));
    dst.sort_unstable();
    dst.resize(n + LANES, u32::MAX);
    (dst.as_ptr(), false)
}

/// The end of the session that starts at `i` among the `n` sorted times
/// at `s`: the first `k > i` more than the gap after `k - 1`, or `n`.
/// `s` is readable for `LANES` past `n`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn session_end(s: *const u32, i: usize, n: usize) -> usize {
    use core::arch::x86_64::*;
    // Unsigned compare by flipping the top bit.
    let flip = _mm256_set1_epi32(i32::MIN);
    let gap = _mm256_set1_epi32((SESSION_GAP_MS as u32 ^ 0x8000_0000) as i32);
    let mut k = i + 1;
    while k < n {
        let a = _mm256_loadu_si256(s.add(k) as *const __m256i);
        let b = _mm256_loadu_si256(s.add(k - 1) as *const __m256i);
        let over = _mm256_cmpgt_epi32(_mm256_xor_si256(_mm256_sub_epi32(a, b), flip), gap);
        let m = _bzhi_u32(_mm256_movemask_ps(_mm256_castsi256_ps(over)) as u32, (n - k).min(LANES) as u32);
        if m != 0 {
            return k + m.trailing_zeros() as usize;
        }
        k += LANES;
    }
    n
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
#[inline(always)]
unsafe fn session_end(s: *const u32, i: usize, n: usize) -> usize {
    let mut k = i + 1;
    while k < n && *s.add(k) - *s.add(k - 1) <= SESSION_GAP_MS as u32 {
        k += 1;
    }
    k
}

#[inline(always)]
fn sort_u64(v: &mut [u64]) {
    if v.len() <= 20 {
        for i in 1..v.len() {
            let x = v[i];
            let mut j = i;
            while j > 0 && v[j - 1] > x {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = x;
        }
    } else {
        v.sort_unstable();
    }
}

impl<'a> PersonPhase<'a> {
    fn new(fast: &'a Fast<'a>, out: UserOut<'a>) -> Self {
        PersonPhase {
            fast,
            tab: Vec::new(),
            keys: Vec::new(),
            counts: Vec::new(),
            words: Vec::new(),
            ev: Vec::new(),
            sorted: Vec::new(),
            sort_tmp: Vec::new(),
            buys: Vec::new(),
            bcounts: Vec::new(),
            bought: Vec::new(),
            ccounts: Vec::new(),
            cids: Vec::new(),
            cl: Vec::new(),
            reach: vec![[0; 4]; fast.ncamp],
            slot_day: vec![0; fast.ndays + 2],
            slot_first: vec![0; fast.ndays + 2],
            s_at: Vec::new(),
            s_day: Vec::new(),
            s_revenue: Vec::new(),
            cover: vec![0; fast.ncamp],
            catsum: vec![0; 256],
            touched: Vec::new(),
            agg: vec![[0; 2]; (fast.ncamp + 1) * fast.ndays],
            assists: vec![0; (fast.ncamp + 1) * fast.ndays],
            out,
            check: (0, 0),
            dropped: true,
        }
    }

    /// Double the table and put the `people` keys back in. Returns the new
    /// shift of the hash.
    #[cold]
    #[inline(never)]
    unsafe fn grow_table(tab: &mut Vec<u32>, keys: *const u64, people: usize) -> u32 {
        let cap = tab.len() * 2;
        tab.clear();
        tab.resize(cap, 0);
        let shift = 64 - cap.trailing_zeros();
        for id1 in 1..=people {
            let mut i = (person_hash(*keys.add(id1)) >> shift) as usize;
            while tab[i] != 0 {
                i = (i + 1) & (cap - 1);
            }
            tab[i] = id1 as u32;
        }
        shift
    }

    /// Pass 1 of a partition: a local id per person (`keys[id + 1]`), how
    /// many events each person has (`counts`), and one word per event,
    /// `time << 32 | id` (`words`); purchases also go to `buys`. Out of
    /// line, so the loop gets the registers to itself. Returns the people,
    /// the words, and the shift and mask of the table.
    #[inline(never)]
    unsafe fn group<const DROPPED: bool>(
        evs: &[List<Ev>],
        table: &mut Vec<u32>,
        keys: *mut u64,
        counts: *mut u32,
        words: *mut u64,
        buys: &mut Vec<(u64, u32)>,
    ) -> (usize, usize, u32, usize) {
        let cap = table.len();
        *keys = 0;
        let mut tab = table.as_mut_ptr();
        let mut mask = cap - 1;
        let mut shift = 64 - cap.trailing_zeros();
        let mut people = 0usize;
        let mut wp = words;
        // Pass 1: a local id per person, how many events each person has,
        // and one word per event.
        for list in evs {
            for chunk in list.chunks() {
                for rec in chunk.iter() {
                    if DROPPED && rec.v == DEAD {
                        // Dropped by the one-pass dedup.
                        continue;
                    }
                    let mut i = (person_hash(rec.pkey) >> shift) as usize;
                    let id = loop {
                        let id1 = *tab.add(i) as usize;
                        if *keys.add(id1) == rec.pkey {
                            break id1 as u32 - 1;
                        }
                        if id1 == 0 {
                            if (people + 1) * 2 > mask + 1 {
                                shift = Self::grow_table(table, keys, people);
                                tab = table.as_mut_ptr();
                                mask = table.len() - 1;
                                i = (person_hash(rec.pkey) >> shift) as usize;
                                continue;
                            }
                            *counts.add(people) = 0;
                            people += 1;
                            *keys.add(people) = rec.pkey;
                            *tab.add(i) = people as u32;
                            break people as u32 - 1;
                        }
                        i = (i + 1) & mask;
                    };
                    *counts.add(id as usize) += 1;
                    *wp = (rec.v & !0xFFFF_FFFF) | u64::from(id);
                    wp = wp.add(1);
                    if rec.v as u32 != 0 {
                        buys.push((rec.v, id));
                    }
                }
            }
        }
        (people, wp.offset_from(words) as usize, shift, mask)
    }

    unsafe fn partition(&mut self, evs: &[List<Ev>], cls: &[List<Click>]) {
        let t = prof::start();
        let n: usize = evs.iter().map(|l| l.count()).sum();
        if n == 0 {
            return;
        }
        let cap = (n / 8).next_power_of_two().max(256);
        self.tab.clear();
        self.tab.resize(cap, 0);
        // At most one person per event. The vectors are filled through
        // pointers, so the loop below reloads nothing.
        self.keys.clear();
        self.keys.reserve(n + 1);
        self.counts.clear();
        self.counts.reserve(n);
        let keys = self.keys.as_mut_ptr();
        let counts = self.counts.as_mut_ptr();
        self.words.clear();
        self.words.reserve(n);
        self.buys.clear();
        let words = self.words.as_mut_ptr();
        let (people, n, shift, mask) = if self.dropped {
            Self::group::<true>(evs, &mut self.tab, keys, counts, words, &mut self.buys)
        } else {
            Self::group::<false>(evs, &mut self.tab, keys, counts, words, &mut self.buys)
        };
        let tab = self.tab.as_ptr();
        // Records the dedup dropped are not counted from here on.
        self.words.set_len(n);
        self.keys.set_len(people + 1);
        self.counts.set_len(people);
        // Pass 2: the times person after person.
        // counts -> start offsets (counts[i] becomes the write cursor).
        let mut sum = 0u32;
        for c in self.counts.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
        self.ev.clear();
        self.ev.reserve(n + EV_PAD);
        self.ev.set_len(n + EV_PAD);
        self.ev[n..].fill(0);
        {
            // Through pointers, so nothing is reloaded per word.
            let (cursor, ev) = (self.counts.as_mut_ptr(), self.ev.as_mut_ptr());
            for &word in &self.words {
                let w = cursor.add(word as u32 as usize);
                *ev.add(*w as usize) = (word >> 32) as u32;
                *w += 1;
            }
        }
        // counts[i] is now the end offset of person i.

        // The purchases, person after person.
        self.bcounts.clear();
        self.bcounts.resize(people, 0);
        let bcounts = self.bcounts.as_mut_ptr();
        for &(_, id) in &self.buys {
            *bcounts.add(id as usize) += 1;
        }
        let mut sum = 0u32;
        for c in self.bcounts.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
        self.bought.clear();
        self.bought.reserve(self.buys.len());
        self.bought.set_len(self.buys.len());
        let bought = self.bought.as_mut_ptr();
        for &(v, id) in &self.buys {
            let w = bcounts.add(id as usize);
            *bought.add(*w as usize) = v;
            *w += 1;
        }

        // Clicks of people that have events here; the rest cannot matter.
        let nc: usize = cls.iter().map(|l| l.count()).sum();
        self.ccounts.clear();
        self.ccounts.resize(people + 1, 0);
        self.cl.clear();
        if nc > 0 {
            self.cids.clear();
            self.cids.reserve(nc);
            let (cids, ccounts) = (self.cids.as_mut_ptr(), self.ccounts.as_mut_ptr());
            let mut at = 0usize;
            for list in cls {
                for chunk in list.chunks() {
                    for rec in chunk.iter() {
                        let mut i = (person_hash(rec.key) >> shift) as usize;
                        let id = loop {
                            let id1 = *tab.add(i) as usize;
                            if *keys.add(id1) == rec.key {
                                break id1 as u32 - 1;
                            }
                            if id1 == 0 {
                                break u32::MAX;
                            }
                            i = (i + 1) & mask;
                        };
                        *cids.add(at) = id;
                        at += 1;
                        if id != u32::MAX {
                            *ccounts.add(id as usize) += 1;
                        }
                    }
                }
            }
            self.cids.set_len(at);
            let mut sum = 0u32;
            for c in self.ccounts.iter_mut() {
                let k = *c;
                *c = sum;
                sum += k;
            }
            self.cl.reserve(sum as usize);
            self.cl.set_len(sum as usize);
            let mut at = 0usize;
            let (cids, cursor, out) = (self.cids.as_ptr(), self.ccounts.as_mut_ptr(), self.cl.as_mut_ptr());
            for list in cls {
                for chunk in list.chunks() {
                    for rec in chunk.iter() {
                        let id = *cids.add(at);
                        at += 1;
                        if id != u32::MAX {
                            let w = cursor.add(id as usize);
                            *out.add(*w as usize) = (rec.tc, rec.ckey);
                            *w += 1;
                        }
                    }
                }
            }
        }

        let t = prof::add(7, t);
        let mut ev_start = 0usize;
        let mut cl_start = 0usize;
        let mut buy_start = 0usize;
        self.reserve_rows(people);
        for p in 0..people {
            let ev_end = *self.counts.get_unchecked(p) as usize;
            let cl_end = if nc > 0 { *self.ccounts.get_unchecked(p) as usize } else { 0 };
            let buy_end = *self.bcounts.get_unchecked(p) as usize;
            self.person(p, ev_start, ev_end, cl_start, cl_end, buy_start, buy_end);
            ev_start = ev_end;
            cl_start = cl_end;
            buy_start = buy_end;
        }
        let t = prof::add(8, t);
        if self.out.flush_full() {
            prof::add(9, t);
        }
    }

    /// Room in the output for the rows of `people` persons: at most one
    /// per event (`counts` holds the end offsets by now).
    fn reserve_rows(&mut self, people: usize) {
        let events = if people == 0 { 0 } else { self.counts[people - 1] as usize };
        self.out.reserve(events);
    }

    #[inline(always)]
    fn emit(&mut self, pkey: u64, day_rel: u32, sessions: u32, events: u32, revenue: u64) {
        let mut top = 0xFFu8;
        if !self.touched.is_empty() {
            let mut best = 0u64;
            for &c in &self.touched {
                let sum = self.catsum[c as usize];
                if sum > best || (sum == best && c < top) {
                    best = sum;
                    top = c;
                }
                self.catsum[c as usize] = 0;
            }
            self.touched.clear();
        }
        self.check.0 += u64::from(sessions);
        self.check.1 += revenue;
        self.out.push(pkey, day_rel, sessions, events, revenue, top);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn person(
        &mut self,
        p: usize,
        ev_start: usize,
        ev_end: usize,
        cl_start: usize,
        cl_end: usize,
        buy_start: usize,
        buy_end: usize,
    ) {
        let pkey = *self.keys.get_unchecked(p + 1);
        let ev = std::slice::from_raw_parts(self.ev.as_ptr().add(ev_start), ev_end - ev_start);
        let cl = std::slice::from_raw_parts_mut(self.cl.as_mut_ptr().add(cl_start), cl_end - cl_start);
        if buy_end - buy_start > 1 {
            // The time is the top half of a value.
            sort_u64(std::slice::from_raw_parts_mut(self.bought.as_mut_ptr().add(buy_start), buy_end - buy_start));
        }
        let mut buy = buy_start;
        let ws_in_day = self.fast.ws_in_day;
        let ndays = self.fast.ndays;
        let organic = self.fast.ncamp;
        let n = ev.len();
        let clicked = !cl.is_empty();
        if clicked {
            // Room for every session and a block of padding.
            for v in [&mut self.s_at, &mut self.s_day] {
                if v.capacity() < n + LANES {
                    v.reserve(n + LANES - v.len());
                }
            }
            if self.s_revenue.capacity() < n {
                self.s_revenue.reserve(n - self.s_revenue.len());
            }
        }
        // Sessions opened and day slots filled so far.
        let (mut opened, mut slots) = (0u32, 0usize);
        let mut cur_day = u32::MAX;
        let (mut d_sessions, mut d_events, mut d_revenue) = (0u32, 0u32, 0u64);
        let (s, binned) = sort_times(ev.as_ptr(), n, &mut self.sorted, &mut self.sort_tmp);
        let mut i = 0usize;
        while i < n {
            let (j, start, last) = if binned {
                session_end_binned(s, i, n)
            } else {
                let j = session_end(s, i, n);
                (j, *s.add(i), *s.add(j - 1))
            };
            let day_rel = ((u64::from(start) + ws_in_day) / DAY_MS as u64) as u32;
            if day_rel != cur_day {
                if cur_day != u32::MAX {
                    self.emit(pkey, cur_day, d_sessions, d_events, d_revenue);
                }
                cur_day = day_rel;
                d_sessions = 0;
                d_events = 0;
                d_revenue = 0;
                *self.slot_day.get_unchecked_mut(slots) = day_rel;
                *self.slot_first.get_unchecked_mut(slots) = opened;
                slots += 1;
                *self.slot_first.get_unchecked_mut(slots) = u32::MAX;
            }
            d_sessions += 1;
            opened += 1;
            d_events += (j - i) as u32;
            // The purchases of the session: those not taken yet that are
            // no later than its last event.
            let mut revenue = 0u64;
            while buy < buy_end && (*self.bought.get_unchecked(buy) >> 32) as u32 <= last {
                let pc = *self.bought.get_unchecked(buy) as u32;
                let price = u64::from(pc >> 8);
                let cat = (pc & 0xFF) as u8;
                revenue += price;
                let sum = self.catsum.get_unchecked_mut(cat as usize);
                if *sum == 0 {
                    self.touched.push(cat);
                }
                *sum += price;
                buy += 1;
            }
            d_revenue += revenue;
            if clicked {
                // Click times are relative to window_start - 7 days, so a
                // session start S is S + reach on that clock.
                let at = opened as usize - 1;
                *self.s_at.as_mut_ptr().add(at) = start + CLICK_REACH_MS as u32;
                *self.s_day.as_mut_ptr().add(at) = cur_day;
                *self.s_revenue.as_mut_ptr().add(at) = revenue;
            } else {
                let cell = self.agg.get_unchecked_mut(organic * ndays + cur_day as usize);
                cell[0] += SESSION | (u64::from(revenue > 0) << 32);
                cell[1] += revenue;
            }
            i = j;
        }
        if cur_day != u32::MAX {
            self.emit(pkey, cur_day, d_sessions, d_events, d_revenue);
        }
        if clicked {
            let sessions = opened as usize;
            // The few-sessions path compares times as signed 32-bit.
            let narrow = self.fast.cwlen < (1 << 31) - 1;
            if sessions > FEW || !narrow || !self.few_sessions(cl, sessions, slots) {
                self.many_sessions(cl, sessions);
            }
        }
    }

    /// Attribute the clicks of a person with at most `FEW` sessions, the
    /// clicks in any order. False, with nothing done, if two clicks of
    /// different campaigns tie for a session: then the click ids decide and
    /// the caller takes the sorted path.
    ///
    /// A click at `t` is in reach of the sessions that start in `t..=t +
    /// reach`. Each session keeps the latest click in reach, and each
    /// campaign the set of sessions its clicks reach.
    #[inline(always)]
    unsafe fn few_sessions(&mut self, cl: &[(u64, u64)], sessions: usize, slots: usize) -> bool {
        let ndays = self.fast.ndays;
        let reach = CLICK_REACH_MS as u64;
        // Whole blocks: past the last session come starts that no click
        // reaches. A real start is under `LAST - 1` (the caller checked).
        const LAST: u32 = i32::MAX as u32;
        let lanes = sessions.div_ceil(LANES) * LANES;
        let at = self.s_at.as_mut_ptr();
        for pad in sessions..lanes {
            *at.add(pad) = LAST;
        }
        // Per session: the time of the latest click in reach, plus one, and
        // its campaign.
        let mut best_t = [0u32; FEW];
        let mut best_c = [self.fast.ncamp as u32; FEW];
        let mut tie = no_tie();
        for click in cl.iter() {
            let t = (click.0 >> 32) as u32;
            let camp = click.0 as u32;
            let upto = (u64::from(t) + reach).min(u64::from(LAST) - 1) as u32;
            let mut reached = 0u64;
            let mut lane = 0;
            while lane < lanes {
                let bits = click_block(at.add(lane), best_t.as_mut_ptr().add(lane), best_c.as_mut_ptr().add(lane), t, upto, camp, &mut tie);
                reached |= bits << lane;
                lane += LANES;
            }
            *self.cover.get_unchecked_mut(camp as usize) |= reached;
        }
        if tied(tie) {
            for click in cl.iter() {
                *self.cover.get_unchecked_mut(click.0 as u32 as usize) = 0;
            }
            return false;
        }
        // Assists: per campaign and day, the sessions of that day it covers.
        let mut day_bits = [0u64; FEW + 1];
        for k in 0..slots {
            let from = *self.slot_first.get_unchecked(k);
            let to = (*self.slot_first.get_unchecked(k + 1)).min(sessions as u32);
            *day_bits.get_unchecked_mut(k) = ((1u64 << to) - 1) & !((1u64 << from) - 1);
        }
        for click in cl.iter() {
            let c = click.0 as u32 as usize;
            let bits = *self.cover.get_unchecked(c);
            if bits != 0 {
                *self.cover.get_unchecked_mut(c) = 0;
                let row = c * ndays;
                for k in 0..slots {
                    let count = (bits & *day_bits.get_unchecked(k)).count_ones();
                    *self.assists.get_unchecked_mut(row + *self.slot_day.get_unchecked(k) as usize) += count;
                }
            }
        }
        for k in 0..sessions {
            let revenue = *self.s_revenue.as_ptr().add(k);
            let cell = self.agg.get_unchecked_mut(*best_c.get_unchecked(k) as usize * ndays + *self.s_day.as_ptr().add(k) as usize);
            cell[0] += SESSION | (u64::from(revenue > 0) << 32);
            cell[1] += revenue;
        }
        true
    }

    /// Attribute the clicks of a person with any number of sessions: sort
    /// the clicks and slide over them, session by session.
    #[inline(never)]
    unsafe fn many_sessions(&mut self, cl: &mut [(u64, u64)], sessions: usize) {
        if cl.len() > 1 {
            // By ts, then by click id descending: the last click at or
            // before a session start is the latest with the smallest id.
            // One integer compare per step: by the first word (ts, then
            // campaign), then each run of equal ts put in id order.
            cl.sort_unstable_by_key(|c| c.0);
            let mut i = 0;
            while i + 1 < cl.len() {
                let ts = cl.get_unchecked(i).0 >> 32;
                if cl.get_unchecked(i + 1).0 >> 32 != ts {
                    i += 1;
                    continue;
                }
                let mut j = i + 2;
                while j < cl.len() && cl.get_unchecked(j).0 >> 32 == ts {
                    j += 1;
                }
                cl.get_unchecked_mut(i..j).sort_unstable_by(|a, b| cmp_ids(b.1, a.1));
                i = j;
            }
        }
        let mut win = Window { lo: 0, hi: 0, session: 0, slot: 0 };
        for k in 0..sessions {
            win.session = k as u32;
            win.slot += u32::from(*self.slot_first.get_unchecked(win.slot as usize + 1) == k as u32);
            let (at, day, revenue) = (*self.s_at.as_ptr().add(k), *self.s_day.as_ptr().add(k), *self.s_revenue.as_ptr().add(k));
            self.credit(cl, &mut win, u64::from(at), day as usize, revenue);
        }
        // The campaigns still in reach after the last session assisted up
        // to it.
        for click in cl.get_unchecked(win.lo..win.hi) {
            let c = click.0 as u32 as usize;
            let state = *self.reach.get_unchecked(c);
            if state[0] != 0 {
                self.reach.get_unchecked_mut(c)[0] = 0;
                self.assisted(c, state[1], state[2], sessions as u32);
            }
        }
    }

    /// Credit one session to a click. `cl` is the person's clicks in
    /// order, `win` the range of them in reach of the previous session, `at`
    /// the start of the session on the click clock; sessions come in order
    /// of their start.
    ///
    /// Assists are not counted session by session. A campaign assists every
    /// session from the one at which its count of clicks in reach leaves
    /// zero to the one at which it returns to zero, so that stretch of
    /// sessions is counted when it ends, once per day it touches.
    #[inline(always)]
    unsafe fn credit(&mut self, cl: &[(u64, u64)], win: &mut Window, at: u64, day: usize, revenue: u64) {
        let ndays = self.fast.ndays;
        let mut credit = self.fast.ncamp;
        {
            let (mut lo, mut hi) = (win.lo, win.hi);
            let start = at - CLICK_REACH_MS as u64;
            while hi < cl.len() && (cl.get_unchecked(hi).0 >> 32) <= at {
                let c = cl.get_unchecked(hi).0 as u32 as usize;
                let state = self.reach.get_unchecked_mut(c);
                let fresh = state[0] == 0;
                state[1] = if fresh { win.session } else { state[1] };
                state[2] = if fresh { win.slot } else { state[2] };
                state[0] += 1;
                hi += 1;
            }
            while lo < hi && (cl.get_unchecked(lo).0 >> 32) < start {
                let c = cl.get_unchecked(lo).0 as u32 as usize;
                let state = self.reach.get_unchecked_mut(c);
                state[0] -= 1;
                if state[0] == 0 {
                    let (from, slot) = (state[1], state[2]);
                    self.assisted(c, from, slot, win.session);
                }
                lo += 1;
            }
            if hi > lo {
                credit = cl.get_unchecked(hi - 1).0 as u32 as usize;
            }
            win.lo = lo;
            win.hi = hi;
        }
        let cell = self.agg.get_unchecked_mut(credit * ndays + day);
        cell[0] += SESSION | (u64::from(revenue > 0) << 32);
        cell[1] += revenue;
    }

    /// Campaign `c` assisted the person's sessions `from..to`; `slot` is the
    /// day slot of session `from`. Sessions of a day are consecutive, so
    /// each day gets the length of its overlap with the stretch.
    #[inline(always)]
    unsafe fn assisted(&mut self, c: usize, from: u32, slot: u32, to: u32) {
        let row = c * self.fast.ndays;
        let mut k = slot as usize;
        loop {
            let a = from.max(*self.slot_first.get_unchecked(k));
            let next = *self.slot_first.get_unchecked(k + 1);
            let b = to.min(next);
            *self.assists.get_unchecked_mut(row + *self.slot_day.get_unchecked(k) as usize) += b - a;
            k += 1;
            if next >= to {
                break;
            }
        }
    }

    /// The campaign cells of this thread and its assist counts.
    fn cells(&mut self) -> Cells {
        Cells { agg: std::mem::take(&mut self.agg), assists: std::mem::take(&mut self.assists) }
    }
}

/// The clicks in reach of the session being credited, and where that
/// session stands among the person's sessions and days.
struct Window {
    lo: usize,
    hi: usize,
    session: u32,
    slot: u32,
}

// ---------------------------------------------------------------- driver

/// Partition count, as a power of two, for about `per_part` estimated rows a
/// partition.
fn bits_for(estimate: u64, per_part: u64, forced: Option<u32>) -> u32 {
    match forced {
        Some(b) => b.clamp(1, 14),
        None => {
            let parts = (estimate / per_part).next_power_of_two().clamp(16, 4096);
            parts.trailing_zeros()
        }
    }
}

struct Timer {
    on: bool,
    t0: Instant,
    last: Instant,
}

impl Timer {
    fn lap(&mut self, name: &str, cpu: f64) {
        if self.on {
            let now = Instant::now();
            eprintln!(
                "[engine] {name:<10} wall {:7.1} ms  cpu {:7.1} ms  (at {:7.1} ms)",
                (now - self.last).as_secs_f64() * 1e3,
                cpu * 1e3,
                (now - self.t0).as_secs_f64() * 1e3
            );
            self.last = now;
        }
    }
}

struct Inputs<'a> {
    events: &'a [Mapped],
    clicks: &'a [Mapped],
    links: &'a [Mapped],
}

/// The one-pass shape up to the person phase: links joined on all threads,
/// then parse with person lookups and notes, then dedup of the notes.
/// Returns the person partitions of events and clicks per thread.
#[allow(clippy::too_many_arguments)]
fn front_onepass(
    fast: &Fast,
    cfg: &Config,
    inputs: &Inputs,
    jobs: &[Job],
    (bits1, bits2): (u32, u32),
    drop_pages: bool,
    cpu_total: &Mutex<f64>,
    timer: &mut Timer,
) -> (Vec<Vec<List<Ev>>>, Vec<Vec<List<Click>>>) {
    let threads = cfg.threads;
    let (parts1, parts2) = (1usize << bits1, 1usize << bits2);
    let knob = |name: &str, default: usize| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    // Staging buffer sizes, in units of whole records filling cache lines
    // (64 bytes for 16-byte records, 192 for clicks).
    let (note_lines, ev_lines, cl_lines) = (knob("ENGINE_NOTE_LINES", 3), knob("ENGINE_EV_LINES", 3), knob("ENGINE_CL_LINES", 1));
    let new_parser = |emit: Option<Box<Emit>>| Parser {
        st: ParseState {
            fast,
            cache: DateCache::new(),
            entries: Scatter::new(1, CHUNK_BYTES),
            shift1: 64 - bits1,
            clicks: Vec::new(),
            pairs: Vec::new(),
            emit,
            dense: None,
        },
        scan: Scan::new(WINDOW),
        fused: cfg.fused,
    };

    // ---- links: read and joined before anything else, so that the
    // parsers can send every event and click straight to its person.
    let link_chunk = cfg.chunk_bytes.min(1 << 18);
    let link_bytes: usize = inputs.links.iter().map(|m| m.len).sum();
    let mut link_jobs: Vec<Job> = Vec::new();
    for (file, m) in inputs.links.iter().enumerate() {
        let mut start = 0;
        while start < m.len {
            let end = if m.len - start < link_chunk * 3 / 2 { m.len } else { start + link_chunk };
            link_jobs.push(Job { kind: Kind::Links, file, start, end });
            start = end;
        }
    }
    let next = AtomicUsize::new(0);
    let mut pair_lists: Vec<Vec<(u64, u64)>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads.min(link_jobs.len()))
            .map(|_| {
                scope.spawn(|| unsafe {
                    let t = prof::start();
                    let mut parser = new_parser(None);
                    // Twice a fair share of pairs at 20 bytes a line, so the
                    // list rarely moves (fresh memory is only address space
                    // until it is written).
                    parser.st.pairs.reserve(link_bytes / 10 / threads.max(1));
                    while let Some(job) = link_jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        parser.run_chunk(inputs.links[job.file], job.start, job.end, Kind::Links);
                    }
                    prof::add(2, t);
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    parser.st.pairs
                })
            })
            .collect();
        for h in handles {
            pair_lists.push(h.join().unwrap());
        }
    });
    let build_cpu = prof::get(3);
    let links = LinkMap::build_parallel(&pair_lists, threads);
    drop(pair_lists);
    timer.lap("links", std::mem::take(&mut *cpu_total.lock().unwrap()) + prof::get(3) - build_cpu);

    // ---- parse
    let directory = sys::Directory::new();
    let directory = &directory;
    let next = AtomicUsize::new(0);
    let mut note_lists: Vec<Vec<List<Note>>> = Vec::new();
    let mut ev_lists: Vec<Vec<List<Ev>>> = Vec::new();
    let mut cl_lists: Vec<Vec<List<Click>>> = Vec::new();
    let mut wide: Vec<Wide> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| unsafe {
                    let emit = Emit::new(
                        links,
                        Scatter::with_lines(parts1, CHUNK_BYTES, note_lines),
                        Scatter::named(parts2, CHUNK_BYTES, ev_lines, directory),
                        Scatter::with_lines(parts2, CHUNK_BYTES, cl_lines),
                        64 - bits1,
                        64 - bits2,
                        fast.ws,
                    );
                    let mut parser = new_parser(Some(emit));
                    while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let files = if job.kind == Kind::Events { inputs.events } else { inputs.clicks };
                        parser.run_chunk(files[job.file], job.start, job.end, job.kind);
                        sys::drop_pages(files[job.file], job.start, job.end, drop_pages);
                    }
                    let mut emit = parser.st.emit.take().unwrap_unchecked();
                    if emit.nevents > 0 {
                        emit.flush_events();
                    }
                    if emit.nclicks > 0 {
                        emit.flush_clicks();
                    }
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    let emit = *emit;
                    (emit.notes.finish(), emit.ev.finish(), emit.cl.finish(), emit.wide)
                })
            })
            .collect();
        for h in handles {
            let (notes, ev, cl, odd) = h.join().unwrap();
            note_lists.push(notes);
            ev_lists.push(ev);
            cl_lists.push(cl);
            wide.extend(odd);
        }
    });
    timer.lap("parse", std::mem::take(&mut *cpu_total.lock().unwrap()));

    // ---- dedup of the notes
    let mut wide_parts: Vec<Vec<Wide>> = Vec::new();
    if !wide.is_empty() {
        wide_parts.resize(parts1, Vec::new());
        for w in wide {
            wide_parts[event_part(w.ekey, 64 - bits1)].push(w);
        }
    }
    let (note_lists, wide_parts) = (&note_lists, &wide_parts);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| unsafe {
                let mut dedup = Deduper { dir: directory, table: Vec::new(), bases: Vec::new(), ingest_base: note_base(fast.ws), dense: false };
                let mut lists: Vec<List<Note>> = Vec::with_capacity(threads);
                loop {
                    let at = next.fetch_add(1, Ordering::Relaxed);
                    if at >= parts1 {
                        break;
                    }
                    lists.clear();
                    lists.extend(note_lists.iter().map(|t| t[at]));
                    dedup.partition(&lists, wide_parts.get(at).map_or(&[][..], |v| &v[..]));
                }
                *cpu_total.lock().unwrap() += sys::thread_cpu();
            });
        }
    });
    timer.lap("resolve", std::mem::take(&mut *cpu_total.lock().unwrap()));
    (ev_lists, cl_lists)
}

/// Run the fast engine. Returns `None` without doing any work when the
/// dataset breaks a precondition (see `Fast::new`). With `cfg.collect` the
/// rows come back in memory; otherwise they are written under `out`.
pub fn run(data: &Path, out: Option<&Path>, lk: &Lookups, cfg: &Config, started: Instant) -> Option<Tables> {
    let mut timer = Timer { on: cfg.timing, t0: started, last: started };
    timer.lap("lookups", 0.0);
    prof::ON.store(cfg.timing, Ordering::Relaxed);
    let simd = crate::scan::init_simd();
    if cfg.timing {
        eprintln!("[engine] stage 1 simd mode {simd}");
    }
    let event_files: Vec<Mapped> = list_csv(&data.join("events")).iter().map(|p| sys::map_file(p)).collect();
    let click_files: Vec<Mapped> = list_csv(&data.join("clicks")).iter().map(|p| sys::map_file(p)).collect();
    let link_files: Vec<Mapped> = list_csv(&data.join("links")).iter().map(|p| sys::map_file(p)).collect();
    let event_bytes: u64 = event_files.iter().map(|m| m.len as u64).sum();
    ids::size_for(event_bytes + click_files.iter().chain(&link_files).map(|m| m.len as u64).sum::<u64>());
    let fast = Fast::new(lk, event_bytes)?;
    let fast = &fast;
    let threads = cfg.threads;

    let estimate = event_bytes / 64;
    // A scatter costs more per record the more partitions it has, and the
    // consumer of a partition is flat up to about 40k records. Measured at
    // 20M events, person side: 1024, 2048 and 4096 partitions cost 150, 166
    // and 196 ms of resolve. At 60M events, event side: 2048 against 4096
    // saves 57 ms of parse and costs 28 ms of resolve. So both sides aim
    // at 20k to 40k records a partition.
    let bits1 = bits_for(estimate, 48_000, cfg.p1_bits);
    let bits2 = bits_for(estimate, 48_000, cfg.p2_bits);
    let (parts1, parts2) = (1usize << bits1, 1usize << bits2);

    // The one-pass shape names person records in 32 bits through a chunk
    // directory. Bound the chunks it can need: every record is a valid
    // event line of at least 16 bytes, a chunk holds over 1000 records, and
    // each thread leaves at most one part-filled chunk a partition.
    let max_chunks = (event_bytes / 16 / 1000) as usize + threads * (parts2 + 1) + 16;
    let onepass = cfg.onepass && max_chunks < sys::DIR_CHUNKS;

    // Jobs: the links job first (unless the links are joined before), then
    // event chunks (largest files first), then click chunks.
    let mut jobs: Vec<Job> = if onepass { Vec::new() } else { vec![Job { kind: Kind::Links, file: 0, start: 0, end: 0 }] };
    for (kind, files) in [(Kind::Events, &event_files), (Kind::Clicks, &click_files)] {
        let mut order: Vec<usize> = (0..files.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(files[i].len));
        for file in order {
            let len = files[file].len;
            let mut start = 0;
            while start < len {
                let end = if len - start < cfg.chunk_bytes * 3 / 2 { len } else { start + cfg.chunk_bytes };
                jobs.push(Job { kind, file, start, end });
                start = end;
            }
        }
    }
    timer.lap("setup", 0.0);

    // Parsed chunks are unmapped by the worker that parsed them. Measured
    // neutral on time at 20M, a little faster at 60M, and it halves peak
    // RSS (the kernel no longer tears the mappings down inside exit).
    let drop_pages = std::env::var("ENGINE_DROP").map_or(true, |v| v != "0");
    let cpu_total = Mutex::new(0.0f64);
    let inputs = Inputs { events: &event_files, clicks: &click_files, links: &link_files };
    if cfg.dense {
        if let Some(tables) = dense::run(fast, lk, cfg, out, &inputs, &jobs, (bits1, bits2), drop_pages, &cpu_total, &mut timer) {
            return Some(tables);
        }
    }
    let (ev_lists, cl_lists, unmapper) = if onepass {
        let (ev, cl) = front_onepass(fast, cfg, &inputs, &jobs, (bits1, bits2), drop_pages, &cpu_total, &mut timer);
        (ev, cl, None)
    } else {
    // ---- phase 1
    let next = AtomicUsize::new(0);
    let link_map: Mutex<Option<LinkMap>> = Mutex::new(None);
    let link_spare: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut entry_lists: Vec<Vec<List<Entry>>> = Vec::new();
    let mut click_blocks: Vec<Vec<Click>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| unsafe {
                    let mut parser = Parser {
                        st: ParseState {
                            fast,
                            cache: DateCache::new(),
                            entries: Scatter::new(parts1, CHUNK_BYTES),
                            shift1: 64 - bits1,
                            clicks: Vec::new(),
                            pairs: Vec::new(),
                            emit: None,
                            dense: None,
                        },
                        scan: Scan::new(WINDOW),
                        fused: cfg.fused,
                    };
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(job) = jobs.get(at) else {
                            break;
                        };
                        match job.kind {
                            Kind::Links => {
                                let t = prof::start();
                                for m in &link_files {
                                    parser.run_chunk(*m, 0, m.len, Kind::Links);
                                }
                                let t = prof::add(2, t);
                                let (map, spare) = LinkMap::build_serial(&parser.st.pairs);
                                prof::add(3, t);
                                *link_map.lock().unwrap() = Some(map);
                                *link_spare.lock().unwrap() = spare;
                            }
                            Kind::Events => {
                                parser.run_chunk(event_files[job.file], job.start, job.end, Kind::Events);
                                sys::drop_pages(event_files[job.file], job.start, job.end, drop_pages);
                            }
                            Kind::Clicks => {
                                parser.run_chunk(click_files[job.file], job.start, job.end, Kind::Clicks);
                                sys::drop_pages(click_files[job.file], job.start, job.end, drop_pages);
                            }
                        }
                    }
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    (parser.st.entries.finish(), parser.st.clicks)
                })
            })
            .collect();
        for h in handles {
            let (entries, clicks) = h.join().unwrap();
            entry_lists.push(entries);
            click_blocks.extend(clicks);
        }
    });
    // The inputs are dead from here on. Unmapping them on a side thread
    // moves about 10 ms of page-table work out of exit at 20M events, but
    // costs the resolve phase as much (measured), so it is off by default.
    let unmapper = std::env::var_os("ENGINE_UNMAP").is_some().then(|| {
        let maps: Vec<Mapped> = event_files.iter().chain(&click_files).chain(&link_files).copied().collect();
        std::thread::spawn(move || {
            for m in maps {
                sys::unmap_file(m);
            }
        })
    });
    let links = link_map.into_inner().unwrap().unwrap();
    // The dead node id table of the link build, cut into chunks: each
    // resolver starts with an equal share, so the first partitions it
    // scatters need fewer fresh pages.
    let link_spare = link_spare.into_inner().unwrap();
    let link_spare = &link_spare;
    let share = AtomicUsize::new(0);
    let links = &links;
    timer.lap("parse", std::mem::take(&mut *cpu_total.lock().unwrap()));

    // ---- phase 2
    let next = AtomicUsize::new(0);
    let entry_lists = &entry_lists;
    let click_blocks: Vec<ClickBlock> = click_blocks.into_iter().map(ClickBlock::from_vec).collect();
    let click_blocks = &click_blocks;
    let mut ev_lists: Vec<Vec<List<Ev>>> = Vec::new();
    let mut cl_lists: Vec<Vec<List<Click>>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| unsafe {
                    let mut r = Resolver {
                        links,
                        table: Vec::new(),
                        bases: Vec::new(),
                        ev: Scatter::new(parts2, CHUNK_BYTES),
                        cl: Scatter::new(parts2, CHUNK_BYTES),
                        shift2: 64 - bits2,
                        entry_chunk_bytes: Scatter::<Entry>::new(1, CHUNK_BYTES).chunk_bytes(),
                    };
                    {
                        let bytes = r.ev.chunk_bytes();
                        let k = share.fetch_add(1, Ordering::Relaxed);
                        let mut j = 0;
                        for &(at, len) in link_spare {
                            // Chunks start on a cache line.
                            let first = (at + 63) & !63;
                            let mut c = first;
                            while c + bytes <= at + len {
                                if j % threads == k {
                                    r.ev.recycle(c as *mut u8, bytes);
                                }
                                j += 1;
                                c += bytes;
                            }
                        }
                    }
                    let mut lists: Vec<List<Entry>> = Vec::with_capacity(threads);
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        if at < parts1 {
                            lists.clear();
                            lists.extend(entry_lists.iter().map(|t| t[at]));
                            r.events(&lists);
                        } else if let Some(block) = click_blocks.get(at - parts1) {
                            r.clicks(block);
                        } else {
                            break;
                        }
                    }
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    (r.ev.finish(), r.cl.finish())
                })
            })
            .collect();
        for h in handles {
            let (ev, cl) = h.join().unwrap();
            ev_lists.push(ev);
            cl_lists.push(cl);
        }
    });
    timer.lap("resolve", std::mem::take(&mut *cpu_total.lock().unwrap()));
    (ev_lists, cl_lists, unmapper)
    };

    // ---- phase 3: biggest partitions first, so the tail is short jobs.
    let mut order: Vec<usize> = (0..parts2).collect();
    let weight = |p: usize| -> u32 { ev_lists.iter().map(|t| t[p].chunks).sum() };
    order.sort_by_key(|&p| std::cmp::Reverse(weight(p)));
    let next = AtomicUsize::new(0);
    let ev_lists = &ev_lists;
    let cl_lists = &cl_lists;
    let order = &order;
    let collected: Mutex<Vec<UserRow>> = Mutex::new(Vec::new());
    let collected_ref = &collected;
    // Each thread adds its cells in as it finishes, off the critical path
    // but for the last one; the first thread's cells become the sum, so no
    // fresh table is touched.
    let shared_cells: Mutex<Cells> = Mutex::new(Cells::default());
    let shared_cells = &shared_cells;
    let mut total_rows = 0u64;
    let out_dir = out.map(|o| o.join("user_daily"));
    let out_dir = &out_dir;
    let next = &next;
    let cpu_total = &cpu_total;
    let mut check = (0u64, 0u64);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || unsafe {
                    let user_out =
                        UserOut::new(out_dir.as_deref(), t, lk, fast.ws_day, fast.ndays, cfg.collect.then_some(collected_ref));
                    let mut phase = PersonPhase::new(fast, user_out);
                    phase.dropped = onepass;
                    let mut evs: Vec<List<Ev>> = Vec::with_capacity(threads);
                    let mut cls: Vec<List<Click>> = Vec::with_capacity(threads);
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&p) = order.get(at) else {
                            break;
                        };
                        evs.clear();
                        evs.extend(ev_lists.iter().map(|t| t[p]));
                        cls.clear();
                        cls.extend(cl_lists.iter().map(|t| t[p]));
                        phase.partition(&evs, &cls);
                    }
                    let t0 = prof::start();
                    phase.out.finish();
                    prof::add(9, t0);
                    let cells = phase.cells();
                    shared_cells.lock().unwrap().add(cells);
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    (phase.out.rows_written, phase.check)
                })
            })
            .collect();
        for h in handles {
            let (rows, sums) = h.join().unwrap();
            check.0 += sums.0;
            check.1 += sums.1;
            total_rows += rows;
        }
    });
    let cells = std::mem::take(&mut *shared_cells.lock().unwrap());
    timer.lap("person", std::mem::take(&mut *cpu_total.lock().unwrap()));
    if let Some(handle) = unmapper {
        let _ = handle.join();
    }
    Some(finish(fast, lk, cfg, out, cells, total_rows, check, collected.into_inner().unwrap(), &mut timer))
}

/// After the person phase: campaign_daily from the summed cells, the empty
/// user_daily file if no row was written, the self-check.
#[allow(clippy::too_many_arguments)]
fn finish(
    fast: &Fast,
    lk: &Lookups,
    cfg: &Config,
    out: Option<&Path>,
    cells: Cells,
    total_rows: u64,
    check: (u64, u64),
    collected: Vec<UserRow>,
    timer: &mut Timer,
) -> Tables {
    // campaign_daily straight from the cells into columns, names borrowed;
    // owned rows only when the caller collects the tables (tests).
    let live = |i: usize| cells.agg[i][0] != 0 || cells.assists[i] != 0;
    let used = (0..cells.agg.len()).filter(|&i| live(i)).count();
    let mut camp = CampCols::with_capacity(used);
    for c in 0..=fast.ncamp {
        let row = c * fast.ndays..(c + 1) * fast.ndays;
        if !row.clone().any(live) {
            continue;
        }
        let (campaign, channel): (&[u8], &[u8]) =
            if c == fast.ncamp { (b"organic", b"organic") } else { (&lk.camp_ids[c], &lk.camp_channels[c]) };
        camp.campaign(campaign, channel);
        for (d, i) in row.enumerate() {
            if !live(i) {
                continue;
            }
            let cell = cells.get(i);
            camp.push((fast.ws_day + d as i64) as i32, cell[0] as i64, cell[1] as i64, cell[2] as i64, cell[3] as i64);
        }
    }
    if let Some(out) = out {
        write_camp_cols(&out.join("campaign_daily/part-0.parquet"), &camp);
        if total_rows == 0 {
            crate::output::write_user_rows(&out.join("user_daily/part-0.parquet"), &[]);
        }
    }
    timer.lap("finish", 0.0);
    if cfg.timing {
        prof::report();
        let sums = (camp.sessions.iter().sum::<i64>(), camp.revenue.iter().sum::<i64>());
        eprintln!(
            "[engine] self-check user_daily sessions {} revenue {} rows {}; campaign_daily sessions {} revenue {} rows {}; {}",
            check.0,
            check.1,
            total_rows,
            sums.0,
            sums.1,
            camp.len(),
            if (check.0 as i64, check.1 as i64) == sums { "OK" } else { "MISMATCH" }
        );
    }
    let mut tables = Tables::default();
    if cfg.collect {
        tables.camp = camp.rows();
    }
    tables.user = collected;
    tables
}

#[cfg(test)]
mod link_tests {
    use super::*;
    use std::collections::HashMap;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(*state)
    }

    fn root(parent: &HashMap<u64, u64>, mut x: u64) -> u64 {
        while let Some(&p) = parent.get(&x) {
            if p == x {
                break;
            }
            x = p;
        }
        x
    }

    /// The link map (filter, staged lookup, far probes) against a plain
    /// union-find over a `HashMap`, for keys in the map and keys not in it.
    #[test]
    fn link_map_matches_union_find() {
        let shapes = [(0usize, 8u64), (1, 4), (7, 16), (50, 60), (3000, 4000), (3000, 500), (200_000, 300_000)];
        for (round, &(pairs_n, universe)) in shapes.iter().enumerate() {
            let mut state = round as u64 + 1;
            // Packed IDs are below 2^63 and never 0.
            let key = |state: &mut u64| 1 + (rng(state) % universe).wrapping_mul(0x0001_0000_0100_0001) % (1 << 62);
            let pairs: Vec<(u64, u64)> = (0..pairs_n).map(|_| (key(&mut state), key(&mut state))).collect();
            let mut parent: HashMap<u64, u64> = HashMap::new();
            for &(a, b) in &pairs {
                parent.entry(a).or_insert(a);
                parent.entry(b).or_insert(b);
                let (ra, rb) = (root(&parent, a), root(&parent, b));
                if ra != rb {
                    let (lo, hi) = if cmp_ids(ra, rb).is_lt() { (ra, rb) } else { (rb, ra) };
                    parent.insert(hi, lo);
                }
            }
            // The table as the engine builds it (at most half full), and a
            // denser one (up to 80% full) where far more buckets overflow.
            for room in [8, 5] {
                let map = LinkMap::build_with_room(&pairs, room).0;
                let mut probes: Vec<u64> = parent.keys().copied().collect();
                for _ in 0..(2 * pairs_n + 1000) {
                    probes.push(key(&mut state));
                    probes.push(1 + rng(&mut state) % (1 << 62));
                }
                probes.push(0);
                let mut renamed = 0;
                for block in probes.chunks(BLOCK) {
                    let hashes: Vec<u64> = block.iter().map(|&k| map.stage1(k)).collect();
                    let starts: Vec<usize> = hashes.iter().map(|&h| map.stage2(h)).collect();
                    for (&k, &at) in block.iter().zip(&starts) {
                        let want = if parent.contains_key(&k) { root(&parent, k) } else { k };
                        assert_eq!(map.stage3(at, k), want, "round {round} room {room} key {k:#x}");
                        renamed += usize::from(want != k);
                    }
                }
                assert!(pairs_n < 50 || renamed > pairs_n / 4, "round {round}: only {renamed} renamed keys probed");
                // Keys that are not in their own bucket were found above by
                // the out-of-line probe; make sure the big round has some.
                let far = parent
                    .keys()
                    .filter(|&&k| root(&parent, k) != k && unsafe { LinkMap::matches(map.bucket(map.home(k)), k) } == 0)
                    .count();
                assert!(pairs_n < 100_000 || far > 20, "round {round} room {room}: only {far} keys overflowed");
            }
        }
    }

    /// The plain union-find answer for `pairs`: key -> person name.
    fn reference(pairs: &[(u64, u64)]) -> HashMap<u64, u64> {
        let mut parent: HashMap<u64, u64> = HashMap::new();
        for &(a, b) in pairs {
            parent.entry(a).or_insert(a);
            parent.entry(b).or_insert(b);
            let (ra, rb) = (root(&parent, a), root(&parent, b));
            if ra != rb {
                let (lo, hi) = if cmp_ids(ra, rb).is_lt() { (ra, rb) } else { (rb, ra) };
                parent.insert(hi, lo);
            }
        }
        parent.keys().map(|&k| (k, root(&parent, k))).collect()
    }

    /// Every key of `want`, and keys not in it, looked up through the
    /// staged lookup of `map`.
    fn check_map(map: &LinkMap, want: &HashMap<u64, u64>, extra: &[u64], what: &str) {
        let probes: Vec<u64> = want.keys().copied().chain(extra.iter().copied()).collect();
        for block in probes.chunks(BLOCK) {
            let hashes: Vec<u64> = block.iter().map(|&k| map.stage1(k)).collect();
            let starts: Vec<usize> = hashes.iter().map(|&h| map.stage2(h)).collect();
            for (&k, &at) in block.iter().zip(&starts) {
                let w = want.get(&k).copied().unwrap_or(k);
                assert_eq!(map.stage3(at, k), w, "{what}: key {k:#x}");
            }
        }
    }

    /// The serial link build (and the parallel one at 1, 2, 3 and 8
    /// threads) against a plain union-find, on random and adversarial
    /// link sets: chains in random and sorted order, stars, cycles,
    /// duplicates both ways round, self links, IDs that take the FALLBACK
    /// key, IDs that all share one home slot, a single pair, and a
    /// component that holds every ID.
    #[test]
    fn link_builds_match_union_find() {
        let mut state = 99u64;
        let mut sets: Vec<(String, Vec<(u64, u64)>)> = Vec::new();
        let packed = |n: u64| 1 + (n.wrapping_mul(0x0001_0000_0100_0001) % (1 << 62));
        // Random sets of several densities.
        for (n, universe) in [(1usize, 2u64), (2, 3), (50, 40), (5000, 6000), (5000, 1500), (100_000, 150_000)] {
            let p = (0..n).map(|_| (packed(rng(&mut state) % universe), packed(rng(&mut state) % universe))).collect();
            sets.push((format!("random {n}/{universe}"), p));
        }
        // A long chain in random order, and in both sorted orders.
        let mut chain: Vec<(u64, u64)> = (0..3000u64).map(|i| (packed(i), packed(i + 1))).collect();
        sets.push(("chain up".into(), chain.clone()));
        sets.push(("chain down".into(), chain.iter().rev().map(|&(a, b)| (b, a)).collect()));
        for i in (1..chain.len()).rev() {
            chain.swap(i, (rng(&mut state) % (i as u64 + 1)) as usize);
        }
        sets.push(("chain shuffled".into(), chain));
        // Stars around the smallest and around the largest ID.
        sets.push(("star low".into(), (1..4000u64).map(|i| (packed(i), packed(0))).collect()));
        sets.push(("star high".into(), (0..4000u64).map(|i| (packed(5000), packed(i))).collect()));
        // Cycles, duplicates either way round, self links.
        let mut c: Vec<(u64, u64)> = Vec::new();
        for ring in 0..300u64 {
            let len = 2 + ring % 9;
            for i in 0..len {
                let (a, b) = (packed(ring * 16 + i), packed(ring * 16 + (i + 1) % len));
                c.push((a, b));
                c.push((b, a));
                c.push((a, a));
                c.push((a, b));
            }
        }
        sets.push(("cycles with repeats and self links".into(), c));
        sets.push(("self links only".into(), (0..500u64).map(|i| (packed(i), packed(i))).collect()));
        // Long IDs (FALLBACK keys) mixed with short ones; a long ID can be
        // the smallest of its person.
        let long: Vec<u64> = (0..400).map(|i| ids::pack_slice(format!("{}{:012}", ["a", "u", "z"][i % 3], i * 7919).as_bytes())).collect();
        let short: Vec<u64> = (0..400).map(|i| ids::pack_slice(format!("u{i:07}").as_bytes())).collect();
        let mut m: Vec<(u64, u64)> = Vec::new();
        for _ in 0..3000 {
            let pick = |s: &mut u64| if rng(s) % 2 == 0 { long[(rng(s) % 400) as usize] } else { short[(rng(s) % 400) as usize] };
            m.push((pick(&mut state), pick(&mut state)));
        }
        sets.push(("fallback keys".into(), m));
        // Keys that all share one home in any table of up to 2^16 slots or
        // buckets: the probes run through long full stretches.
        let mut same: Vec<u64> = Vec::new();
        let mut k = 1u64;
        while same.len() < 600 {
            if (link_hash(k) >> 17) & 0xFFFF == 0x1234 {
                same.push(k);
            }
            k += 1;
        }
        let p: Vec<(u64, u64)> = (0..1500).map(|_| (same[(rng(&mut state) % 600) as usize], same[(rng(&mut state) % 600) as usize])).collect();
        sets.push(("one home".into(), p));
        sets.push(("one home, all distinct".into(), same.chunks(2).map(|w| (w[0], w[1])).collect()));
        // All distinct pairs of distinct IDs: 2 * pairs keys, the most the
        // node table can be asked to hold; 2 * 8192 is a power of two.
        sets.push(("all distinct".into(), (0..8192u64).map(|i| (packed(2 * i), packed(2 * i + 1))).collect()));
        sets.push(("no pairs".into(), Vec::new()));

        for (what, pairs) in &sets {
            let want = reference(pairs);
            let mut extra: Vec<u64> = (0..2000).map(|_| 1 + rng(&mut state) % (1 << 62)).collect();
            extra.extend((0..2000u64).map(packed));
            for room in [8, 5] {
                let (map, spare) = LinkMap::build_serial_with_room(pairs, room);
                check_map(&map, &want, &extra, &format!("serial room {room}, {what}"));
                for &(at, len) in &spare {
                    assert!(at != 0 && len > 0, "{what}: spare region");
                }
            }
            for threads in [1, 2, 3, 8] {
                let lists: Vec<Vec<(u64, u64)>> = (0..threads).map(|t| pairs.iter().skip(t).step_by(threads).copied().collect()).collect();
                check_map(&LinkMap::build_parallel(&lists, threads), &want, &extra, &format!("parallel {threads}, {what}"));
            }
        }
    }

    /// Timing of the link builds on the links of a dataset:
    /// `LINKBENCH=<data dir> cargo test --release link_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn link_bench() {
        let Some(dir) = std::env::var_os("LINKBENCH") else {
            return;
        };
        let mut pairs: Vec<(u64, u64)> = Vec::new();
        for path in list_csv(&std::path::Path::new(&dir).join("links")) {
            let text = std::fs::read(&path).unwrap();
            for line in text.split(|&c| c == b'\n').skip(1) {
                let f: Vec<&[u8]> = line.split(|&c| c == b',').collect();
                if f.len() != 2 {
                    continue;
                }
                let strip = |s: &[u8]| -> Vec<u8> { if s.len() >= 2 && s[0] == b'"' && s[s.len() - 1] == b'"' { s[1..s.len() - 1].to_vec() } else { s.to_vec() } };
                let (a, b) = (strip(f[0]), strip(f[1]));
                if a.is_empty() || b.is_empty() {
                    continue;
                }
                pairs.push((ids::pack_slice(&a), ids::pack_slice(&b)));
            }
        }
        prof::ON.store(true, Ordering::Relaxed);
        let which = std::env::var("LINKBENCH_WHICH").unwrap_or_else(|_| "old,serial".into());
        for name in which.split(',') {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                let map = match name {
                    "old" => LinkMap::build_with_spare(&pairs).0,
                    "par1" => LinkMap::build_parallel(std::slice::from_ref(&pairs), 1),
                    "serial" => LinkMap::build_serial(&pairs).0,
                    _ => panic!("unknown build {name}"),
                };
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(map);
            }
            eprintln!("link_bench {name}: {best:.1} ms, {} pairs", pairs.len());
        }
    }
}

#[cfg(test)]
mod long_line_tests {
    use super::*;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(*state)
    }

    fn check<const N: usize>(line: &[u8]) {
        let mut want = Vec::new();
        let ok = split_line(line, N, &mut want);
        match split_long::<N>(line) {
            Some(got) => assert!(ok && got[..] == want[..], "{N} fields of {:?}", String::from_utf8_lossy(line)),
            None => assert!(!ok, "{N} fields of {:?}", String::from_utf8_lossy(line)),
        }
    }

    /// The slice splitter accepts exactly the lines the plain splitter
    /// accepts, with the same fields.
    #[test]
    fn split_long_matches_split_line() {
        let mut state = 17;
        let mut accepted = 0;
        for round in 0..200_000 {
            let n = (rng(&mut state) % 24) as usize;
            let alphabet: &[u8] = match round % 4 {
                0 => b"a,\"",
                1 => b"ab,,\"\r 1",
                2 => b"abcdefgh,",
                _ => b"abc\"\",,,",
            };
            let line: Vec<u8> = (0..n).map(|_| alphabet[(rng(&mut state) % alphabet.len() as u64) as usize]).collect();
            check::<2>(&line);
            check::<4>(&line);
            check::<7>(&line);
            accepted += usize::from(split_long::<4>(&line).is_some());
        }
        assert!(accepted > 2_000, "only {accepted} lines accepted");
        for line in [
            "", ",", "\"", "\"\"", "\"\",\"\"", "a,b", "\"a\",\"b\"", "\"a,b\"", "a\"b,c", "\"a\"b,c", ",,,,,,", ",,,,,,,",
            "a,b,c,d", "a,b,c,d,", "\"\"\"\",b", "a,\"", "\"a\",b\"",
        ] {
            check::<2>(line.as_bytes());
            check::<4>(line.as_bytes());
            check::<7>(line.as_bytes());
        }
        // A megabyte of commas is rejected at the first comma too many.
        assert!(split_long::<7>(&vec![b','; 1 << 20]).is_none());
    }
}

#[cfg(test)]
mod sort_tests {
    use super::*;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(*state)
    }

    /// Every size class of `sort_times` against the standard sort, with
    /// times that repeat, sit near the top of the range, and span the gap.
    #[test]
    fn sort_times_and_session_end_match_scalar() {
        let mut state = 7u64;
        let (mut dst, mut tmp) = (Vec::new(), Vec::new());
        for n in (1..=200).chain([255, 256, 257, 511, 640, 1000, 1023, 1024, 1025, 3000, 70_000]) {
            for round in 0..4 {
                let spread = [100u64, 4_000_000, 1 << 32, 50][round];
                let base = if round == 2 { 0 } else { u32::MAX as u64 - spread - 1 };
                let mut src: Vec<u32> = (0..n).map(|_| (base + rng(&mut state) % spread) as u32).collect();
                src.extend(std::iter::repeat_n(0xDEAD_BEEF, EV_PAD));
                let mut want = src[..n].to_vec();
                want.sort_unstable();
                let (s, binned) = unsafe { sort_times(src.as_ptr(), n, &mut dst, &mut tmp) };
                let got = unsafe { std::slice::from_raw_parts(s, n) }.to_vec();
                // The AVX2 build puts big people in bin order only; the
                // portable build always sorts fully.
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                assert_eq!(binned, n >= BIN_EVENTS);
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                assert!(!binned);
                if binned {
                    assert!(got.windows(2).all(|w| w[0] >> BIN_SHIFT <= w[1] >> BIN_SHIFT), "n {n} round {round}");
                    let mut sorted = got.clone();
                    sorted.sort_unstable();
                    assert_eq!(sorted, want, "n {n} round {round}");
                } else {
                    assert_eq!(got, want, "n {n} round {round}");
                }
                let mut i = 0;
                while i < n {
                    let (j, start, last) = if binned {
                        unsafe { session_end_binned(s, i, n) }
                    } else {
                        let j = unsafe { session_end(s, i, n) };
                        (j, got[i], got[j - 1])
                    };
                    let mut k = i + 1;
                    while k < n && want[k] - want[k - 1] <= SESSION_GAP_MS as u32 {
                        k += 1;
                    }
                    assert_eq!((j, start, last), (k, want[i], want[k - 1]), "n {n} round {round} at {i}");
                    i = j;
                }
            }
        }
    }
}

#[cfg(test)]
mod fused_tests {
    use super::*;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(*state)
    }

    type Got = Vec<(usize, Vec<Vec<u8>>, bool)>;

    /// `text` followed by the zero page a mapping has after its end.
    fn padded(text: &[u8]) -> Vec<u8> {
        let mut buf = text.to_vec();
        buf.extend_from_slice(&[0u8; 4096]);
        buf
    }

    unsafe fn collect<const N: usize>(base: *const u8, start: *const u8, b: &Bounds, known: bool, out: &mut Got) {
        let fields = (0..N).map(|i| std::slice::from_raw_parts(base.add(b.start(i)), b.len(i)).to_vec()).collect();
        out.push((base as usize - start as usize, fields, known));
    }

    fn long<const N: usize>(at: usize, line: &[u8], out: &mut Got) {
        if let Some(f) = split_long::<N>(line) {
            out.push((at, f.iter().map(|x| x.to_vec()).collect(), false));
        }
    }

    /// The lines of `text` (from offset 0) as the two-stage reader in
    /// `run_chunk` sees them, in windows of `window` bytes.
    fn staged<const N: usize>(text: &[u8], window: usize) -> Got {
        let buf = padded(text);
        let len = text.len();
        let mut scan = Scan::new(window.max(REFUSED_WINDOW));
        let mut out = Got::new();
        let mut pos = 0;
        unsafe {
            while pos < len {
                let wlen = window.min(len - pos);
                let base = buf.as_ptr().add(pos);
                scan.scan(base, wlen);
                let src = Scanned { base, scan: &scan, limit: (len - pos) as u32 };
                let start = buf.as_ptr();
                let used = src.run::<N>(|p, b, k| collect::<N>(p, start, b, k, &mut out));
                if used > 0 {
                    pos += used;
                    continue;
                }
                let after = pos + wlen;
                let end = text[after..].iter().position(|&c| c == b'\n').map_or(len, |i| after + i);
                long::<N>(pos, &text[pos..end], &mut out);
                pos = end + 1;
            }
        }
        out
    }

    /// The same by the one-pass reader, as `run_chunk` drives it.
    fn fused<const N: usize>(text: &[u8]) -> Got {
        let buf = padded(text);
        let len = text.len();
        let m = Mapped { ptr: buf.as_ptr(), len };
        let mut scan = Scan::new(WINDOW);
        let mut out = Got::new();
        let mut pos = 0;
        unsafe {
            while pos < len {
                let start = buf.as_ptr();
                let src = Fused { m, pos, end: len, scan: &mut scan };
                pos = src.run::<N>(|p, b, k| collect::<N>(p, start, b, k, &mut out));
                if pos < len {
                    let end = text[pos..].iter().position(|&c| c == b'\n').map_or(len, |i| pos + i);
                    long::<N>(pos, &text[pos..end], &mut out);
                    pos = end + 1;
                }
            }
        }
        out
    }

    /// The plain splitter on every line.
    fn reference<const N: usize>(text: &[u8]) -> Vec<(usize, Vec<Vec<u8>>)> {
        let mut out = Vec::new();
        let mut tmp = Vec::new();
        let mut at = 0;
        while at < text.len() {
            let end = text[at..].iter().position(|&c| c == b'\n').map_or(text.len(), |i| at + i);
            if split_line(&text[at..end], N, &mut tmp) {
                out.push((at, tmp.iter().map(|f| f.to_vec()).collect()));
            }
            at = end + 1;
        }
        out
    }

    fn check<const N: usize>(text: &[u8], what: &str) {
        // Line offsets are not compared: a window source reports its base.
        let want: Vec<_> = reference::<N>(text).into_iter().map(|(_, f)| f).collect();
        let strip = |g: &Got| g.iter().map(|(_, f, _)| f.clone()).collect::<Vec<_>>();
        let got = fused::<N>(text);
        assert_eq!(strip(&got), want, "fused, {N} fields, {what}: {:?}", String::from_utf8_lossy(text));
        for (_, fields, known) in &got {
            if *known {
                for id in &fields[..2] {
                    assert!(id.iter().all(|&c| (1..=127).contains(&c)), "known but not plain: {what}");
                }
            }
        }
        for window in [97, 1000, WINDOW] {
            assert_eq!(strip(&staged::<N>(text, window)), want, "staged {window}, {N} fields, {what}");
        }
    }

    /// One line, followed by `next`: the one-pass verdict and bounds
    /// against stage 1 and 2 on a window holding just that line.
    fn line<const N: usize>(line: &[u8], next: &[u8]) -> bool {
        let mut text = line.to_vec();
        text.push(b'\n');
        text.extend_from_slice(next);
        let buf = padded(&text);
        let mut b = Bounds::new();
        let mut known = false;
        let nl = unsafe {
            if N == 7 {
                fuse::event_line(buf.as_ptr(), &mut b, &mut known)
            } else {
                fuse::click_line(buf.as_ptr(), &mut b, &mut known)
            }
        };
        let mut scan = Scan::new(4096);
        let mut staged = Got::new();
        let used = unsafe {
            scan.scan(buf.as_ptr(), text.len());
            let src = Scanned { base: buf.as_ptr(), scan: &scan, limit: 1 };
            src.run::<N>(|p, bb, k| collect::<N>(p, buf.as_ptr(), bb, k, &mut staged))
        };
        assert_eq!(used, line.len() + 1);
        if nl == fuse::REFUSED {
            return false;
        }
        assert_eq!(nl as usize, line.len(), "line end of {:?}", String::from_utf8_lossy(line));
        assert_eq!(staged.len(), 1, "fused accepted a malformed line {:?}", String::from_utf8_lossy(line));
        for i in 0..N {
            let f = &buf[b.start(i)..b.start(i) + b.len(i)];
            assert_eq!(f, &staged[0].1[i][..], "field {i} of {:?}", String::from_utf8_lossy(line));
        }
        true
    }

    const EVENT: [&str; 7] = ["e1234567a", "u12345678", "2026-10-03T23:58:10Z", "2026-10-03T23:59:10Z", "view", "p123", "100"];
    const CLICK: [&str; 4] = ["k1234567a", "u12345678", "c_12", "2026-10-03T23:58:10Z"];

    fn join(fields: &[Vec<u8>]) -> Vec<u8> {
        fields.join(&b","[..])
    }

    fn quote_variants(x: &[u8]) -> Vec<Vec<u8>> {
        let mut v = vec![x.to_vec(), Vec::new()];
        let mut w = |pre: &[u8], mid: &[u8], post: &[u8]| v.push([pre, mid, post].concat());
        w(b"\"", x, b"\"");
        w(b"\"", b"", b"\"");
        w(b"\"", b"", b"");
        w(b"\"", x, b"");
        w(b"", x, b"\"");
        w(b"\"\"", x, b"");
        w(b"", x, b"\"\"");
        w(b"\"\"", x, b"\"\"");
        w(b"\"", b"\"", b"\"");
        w(b"\"\"\"", b"", b"");
        w(b"\"\"\"\"", b"", b"");
        if x.len() >= 2 {
            w(&x[..1], b"\"", &x[1..]);
            w(b"\"", &x[..1], &[b"\"", &x[1..], b"\""].concat());
        }
        v
    }

    fn quotes<const N: usize>(base: &[&str]) {
        let fields: Vec<Vec<u8>> = base.iter().map(|f| f.as_bytes().to_vec()).collect();
        let mut text = Vec::new();
        for k in 0..N {
            for (n, v) in quote_variants(&fields[k]).into_iter().enumerate() {
                let mut f = fields.clone();
                f[k] = v.clone();
                let l = join(&f);
                // The legal ones: plain, empty, quoted, quoted empty.
                let legal = n < 4;
                assert_eq!(line::<N>(&l, b"x,y\n"), legal, "{:?}", String::from_utf8_lossy(&l));
                assert_eq!(line::<N>(&l, &join(&fields)), legal);
                text.extend_from_slice(&l);
                text.push(b'\n');
                // Two fields at once.
                let j = (k + 3) % N;
                for u in quote_variants(&fields[j]) {
                    let mut g = f.clone();
                    g[j] = u;
                    line::<N>(&join(&g), b"");
                }
            }
        }
        check::<N>(&text, "quote placements");
    }

    #[test]
    fn every_quote_placement() {
        quotes::<7>(&EVENT);
        quotes::<4>(&CLICK);
    }

    /// Lines of every length from 20 to 150 bytes, padded or cut in each
    /// field in turn, plain and quoted: commas and line breaks on every side
    /// of the 32, 64, 96 and 128 byte edges.
    fn lengths<const N: usize>(base: &[&str]) {
        let fields: Vec<Vec<u8>> = base.iter().map(|f| f.as_bytes().to_vec()).collect();
        let mut text = Vec::new();
        let mut accepted = 0;
        for k in 0..N {
            for quoted in [false, true] {
                for total in 20..=150usize {
                    let mut f = fields.clone();
                    if quoted {
                        f[k] = [&b"\""[..], &f[k], b"\""].concat();
                    }
                    let now = join(&f).len();
                    if total < now {
                        let cut = (now - total).min(fields[k].len());
                        f[k] = fields[k][..fields[k].len() - cut].to_vec();
                        if quoted {
                            f[k] = [&b"\""[..], &f[k], b"\""].concat();
                        }
                    } else {
                        let at = if quoted { f[k].len() - 1 } else { f[k].len() };
                        f[k].splice(at..at, std::iter::repeat_n(b'7', total - now));
                    }
                    let l = join(&f);
                    let fast = line::<N>(&l, &join(&fields));
                    assert_eq!(line::<N>(&l, b"\"\",,\n"), fast);
                    accepted += usize::from(fast);
                    text.extend_from_slice(&l);
                    text.push(b'\n');
                }
            }
        }
        eprintln!("{N} fields: {accepted} of {} lengths by the fast path", N * 2 * 131);
        assert!(accepted > N * 2 * 131 / 4);
        check::<N>(&text, "lengths");
    }

    #[test]
    fn line_lengths_around_block_edges() {
        lengths::<7>(&EVENT);
        lengths::<4>(&CLICK);
    }

    /// Random lines shaped like the real files, with every kind of dirt.
    fn random<const N: usize>(seed: u64) {
        let mut state = seed;
        for round in 0..3000 {
            let mut text = Vec::new();
            let lines = 1 + rng(&mut state) % 30;
            for _ in 0..lines {
                let width = match rng(&mut state) % 30 {
                    0 => N + 1,
                    1 => N - 1,
                    2 => 0,
                    3 => 1,
                    _ => N,
                };
                for i in 0..width {
                    if i > 0 {
                        text.push(b',');
                    }
                    let len = match rng(&mut state) % 12 {
                        0 => 0,
                        1 => 1,
                        2 => 20 + rng(&mut state) % 60,
                        _ => 3 + rng(&mut state) % 20,
                    };
                    let quoting = rng(&mut state) % 100;
                    if quoting < 20 || quoting == 97 {
                        text.push(b'"');
                    }
                    for _ in 0..len {
                        let c = match rng(&mut state) % 200 {
                            0 => b'\r',
                            1 => 0x80,
                            2 => 0,
                            3 => 0xFF,
                            4 => b'"',
                            _ => b"abcxyz0189:-TZ"[(rng(&mut state) % 14) as usize],
                        };
                        text.push(c);
                    }
                    if quoting == 96 {
                        text.extend_from_slice(b"\"x");
                    }
                    if quoting < 20 || quoting == 98 {
                        text.push(b'"');
                    }
                }
                text.push(b'\n');
            }
            if round % 3 == 0 {
                // The last line of the file without its line break.
                text.pop();
            }
            for at in 0..text.len().min(40) {
                if text[at] == b'\n' {
                    let end = text[at + 1..].iter().position(|&c| c == b'\n').map_or(text.len(), |i| at + 1 + i);
                    if end < text.len() {
                        line::<N>(&text[at + 1..end], &text[end + 1..]);
                    }
                }
            }
            check::<N>(&text, &format!("round {round}"));
        }
    }

    #[test]
    fn random_lines() {
        random::<7>(5);
        random::<4>(6);
    }

    #[test]
    fn edge_files() {
        for text in [
            &b""[..],
            b"\n",
            b"\n\n\n",
            b"a,b,c,d",
            b"a,b,c,d\n",
            b"a,b,c,d,e,f,g",
            b"a,b,c,d,e,f,g\n",
            b",,,,,,\n,,,\n",
            b"\"\",\"\",\"\",\"\",\"\",\"\",\"\"\n\"\",\"\",\"\",\"\"",
            b"\r\n\r",
            b"a,b,c,d\r\n",
            b"\x00,\x00,\x00,\x00\n",
        ] {
            check::<7>(text, "edge");
            check::<4>(text, "edge");
        }
        // A line far longer than the refused-line window, and one longer
        // than a stage 1 window.
        for n in [300usize, 70_000] {
            let mut text = format!("e1,u1,{},x,view,p1,5\n", "1".repeat(n)).into_bytes();
            text.extend_from_slice(b"k1,u1,c_1,5\n");
            check::<7>(&text, "long");
            check::<4>(&text, "long");
        }
    }
}
