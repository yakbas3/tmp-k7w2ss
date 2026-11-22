//! The dense shape (`ENGINE_DENSE=1`, off by default): the one-pass shape
//! with every user ID numbered.
//!
//! One shared table holds every user ID: those of the links first, then
//! every ID a parser meets, inserted lock-free (a compare-and-swap on the
//! key word). An ID's number is its slot. A person is the number of its
//! name's slot (the root of the links, the byte-wise smallest ID; an
//! unlinked ID is its own name). The slot's value word says which: 0 for
//! a name, else the name's slot plus one.
//!
//! The person partition is the top bits of the person number, so each
//! partition owns a range of slots, and the low bits are the person's
//! index inside it. A partition's per-person state is then plain arrays
//! indexed by that number: no hash table, no key compares. Event records
//! are 8 bytes, `index << 32 | ts`; purchases and clicks have side lists.
//! Dedup works on notes, as in the one-pass shape, and marks the record of
//! a losing copy `DROPPED` in place.
//!
//! Exactness: the table is sized from an estimate. If it fills past three
//! quarters (or a probe goes round it), the shape gives up before writing
//! anything and the three-phase shape runs instead. The same happens when
//! a click record cannot hold the person index and the campaign.

use super::*;
use std::sync::atomic::AtomicBool;

/// An event record the dedup dropped: index `u32::MAX`, which no person
/// has (a partition has at most 2^31 slots).
pub(super) const DROPPED: u64 = u64::MAX;

/// Keys in a bucket of the ID table: a cache line of four keys, then their
/// four values.
const IDB: usize = 4;

/// Lookups a dense parser runs as one block. Each block starts its
/// prefetch ring cold; measured at 20M events, 1024, 2048 and 4096 cost the
/// same, 256 about 90 CPU ms more, 64 about 450 more.
const DENSE_BLOCK: usize = 1024;

/// Record chunks of the dense scatter: 8-byte records, so that a chunk
/// holds fewer than `2^SLOT_BITS` and records can be named.
const DENSE_CHUNK: usize = 8192;

/// The shared ID table. Fresh anonymous memory is zero, and 0 is no key.
#[derive(Clone, Copy)]
pub(super) struct IdTable {
    words: *mut u64,
    /// Buckets minus one.
    bmask: usize,
}
unsafe impl Send for IdTable {}
unsafe impl Sync for IdTable {}

impl IdTable {
    fn new(slots: usize) -> IdTable {
        assert!(slots.is_power_of_two() && slots >= IDB);
        IdTable { words: sys::anon(slots * 16) as *mut u64, bmask: slots / IDB - 1 }
    }
    #[inline(always)]
    fn bucket_of(&self, h: u64) -> usize {
        (h >> 20) as usize & self.bmask
    }
    #[inline(always)]
    fn bucket(&self, b: usize) -> *mut u64 {
        unsafe { self.words.add(b * 2 * IDB) }
    }
    #[inline(always)]
    fn key(&self, slot: usize) -> u64 {
        unsafe { *self.bucket(slot / IDB).add(slot % IDB) }
    }
    #[inline(always)]
    fn value(&self, slot: usize) -> &AtomicU64 {
        unsafe { &*(self.bucket(slot / IDB).add(IDB + slot % IDB) as *const AtomicU64) }
    }
    #[inline(always)]
    fn prefetch(&self, h: u64) {
        prefetch(self.bucket(self.bucket_of(h)));
    }

    /// The slot of `key` (never 0), inserted if it is not there; `None`
    /// when the probe went round the whole table. `h` is `mix(key)`.
    ///
    /// Buckets fill from the front and keys never move, so a key is in the
    /// first bucket from its home that holds it or has a free slot. Two
    /// threads may insert the same key at once: both try the same free
    /// slot, one wins, and the other sees the key there.
    #[inline(always)]
    unsafe fn slot(&self, key: u64, h: u64, inserted: &mut usize) -> Option<usize> {
        let mut b = self.bucket_of(h);
        let mut left = self.bmask + 1;
        loop {
            let p = self.bucket(b);
            let m = LinkMap::matches(p, key);
            if m != 0 {
                return Some(b * IDB + m.trailing_zeros() as usize);
            }
            let free = LinkMap::matches(p, 0);
            if free != 0 {
                let j = free.trailing_zeros() as usize;
                match (*(p.add(j) as *const AtomicU64)).compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => {
                        *inserted += 1;
                        return Some(b * IDB + j);
                    }
                    Err(now) if now == key => return Some(b * IDB + j),
                    // Another key took the slot: look at the bucket again.
                    Err(_) => continue,
                }
            }
            left -= 1;
            if left == 0 {
                return None;
            }
            b = (b + 1) & self.bmask;
        }
    }

    /// The person of the ID in `slot`.
    #[inline(always)]
    fn person_of(&self, slot: usize) -> u32 {
        let v = self.value(slot).load(Ordering::Relaxed);
        std::hint::select_unpredictable(v == 0, slot as u32, v.wrapping_sub(1) as u32)
    }

    /// Put every ID of the links in the table and join them, on `threads`
    /// threads; afterwards the value of every ID that is not its person's
    /// name is the name's slot plus one. Returns the IDs inserted, or
    /// `None` if the table filled.
    ///
    /// The union-find is the one of `LinkMap::build_parallel`: links go
    /// from the root with the larger ID to the root with the smaller, so
    /// a root is the smallest ID of its set once all unions are done. Then
    /// each linked node points straight at its root.
    fn join_links(&self, lists: &[Vec<(u64, u64)>], threads: usize) -> Option<usize> {
        let total: usize = lists.iter().map(|l| l.len()).sum();
        if total == 0 {
            return Some(0);
        }
        let table = *self;
        let find = |mut x: usize| -> usize {
            loop {
                let px = table.value(x);
                let up = px.load(Ordering::Relaxed);
                if up == 0 {
                    return x;
                }
                let above = table.value(up as usize - 1).load(Ordering::Relaxed);
                if above != 0 {
                    let _ = px.compare_exchange(up, above, Ordering::Relaxed, Ordering::Relaxed);
                }
                x = up as usize - 1;
            }
        };
        let jobs = link_jobs(lists);
        let next = AtomicUsize::new(0);
        let full = AtomicBool::new(false);
        let inserted = AtomicUsize::new(0);
        let mut linked: Vec<Vec<u32>> = Vec::new();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads.clamp(1, jobs.len()))
                .map(|_| {
                    scope.spawn(|| unsafe {
                        let mut made: Vec<u32> = Vec::with_capacity(total / threads.max(1) + 1024);
                        let mut ins = 0usize;
                        let mut ends = [(0usize, 0usize); LINK_GROUP];
                        let mut hs = [(0u64, 0u64); LINK_GROUP];
                        'jobs: while let Some(&(l, at)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let list = &lists[l];
                            for block in list[at..list.len().min(at + LINK_STEP)].chunks(LINK_GROUP) {
                                for (h, &(a, b)) in hs.iter_mut().zip(block) {
                                    *h = (mix(a), mix(b));
                                    table.prefetch(h.0);
                                    table.prefetch(h.1);
                                }
                                for ((end, &(a, b)), &(ha, hb)) in ends.iter_mut().zip(block).zip(&hs) {
                                    match (table.slot(a, ha, &mut ins), table.slot(b, hb, &mut ins)) {
                                        (Some(x), Some(y)) => *end = (x, y),
                                        _ => {
                                            full.store(true, Ordering::Relaxed);
                                            break 'jobs;
                                        }
                                    }
                                }
                                for &(na, nb) in &ends[..block.len()] {
                                    for node in [na, nb] {
                                        let up = table.value(node).load(Ordering::Relaxed);
                                        if up != 0 {
                                            prefetch(table.value(up as usize - 1));
                                        }
                                    }
                                }
                                for &(na, nb) in &ends[..block.len()] {
                                    loop {
                                        let (ra, rb) = (find(na), find(nb));
                                        if ra == rb {
                                            break;
                                        }
                                        let (ka, kb) = (table.key(ra), table.key(rb));
                                        let (small, large) = if cmp_ids(ka, kb).is_lt() { (ra, rb) } else { (rb, ra) };
                                        if table.value(large)
                                            .compare_exchange(0, small as u64 + 1, Ordering::Relaxed, Ordering::Relaxed)
                                            .is_ok()
                                        {
                                            made.push(large as u32);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        inserted.fetch_add(ins, Ordering::Relaxed);
                        prof::add(3, 0.0);
                        made
                    })
                })
                .collect();
            for h in handles {
                linked.push(h.join().unwrap());
            }
        });
        if full.load(Ordering::Relaxed) {
            return None;
        }
        // Every node that was linked now points at its root.
        let jobs = link_jobs(&linked);
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads.clamp(1, jobs.len().max(1)) {
                scope.spawn(|| {
                    while let Some(&(l, at)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let list = &linked[l];
                        for &node in &list[at..list.len().min(at + LINK_STEP)] {
                            let mut r = node as usize;
                            loop {
                                let v = table.value(r).load(Ordering::Relaxed);
                                if v == 0 {
                                    break;
                                }
                                r = v as usize - 1;
                            }
                            table.value(node as usize).store(r as u64 + 1, Ordering::Relaxed);
                        }
                    }
                    prof::add(3, 0.0);
                });
            }
        });
        Some(inserted.into_inner())
    }
}

/// A valid purchase: `ts << 32 | price << 8 | category`, the person's
/// index, and the name of its event record (to see whether dedup dropped
/// it).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Buy {
    tp: u64,
    idx: u32,
    rec: u32,
}

/// A valid click: its key, and `ts << 32 | index << cbits | campaign`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct DClick {
    ckey: u64,
    w: u64,
}

/// Where a dense parser sends what it parsed: ID lookups in blocks, then
/// the scatters by person and by event key.
pub(super) struct DenseEmit {
    table: IdTable,
    notes: Scatter<Note>,
    ev: Scatter<u64>,
    buys: Scatter<Buy>,
    cl: Scatter<DClick>,
    shift1: u32,
    /// Bits of the index inside a partition, and of a campaign.
    sbits: u32,
    cbits: u32,
    ingest_base: i64,
    wide: Vec<Wide>,
    nevents: usize,
    nclicks: usize,
    limit: usize,
    /// Lookups prefetched ahead of the one at hand (a whole block when it
    /// is the block size).
    ahead: usize,
    /// IDs this parser inserted since it last added them to `count`.
    inserted: usize,
    count: *const AtomicUsize,
    full: *const AtomicBool,
    max_ids: usize,
    events: [Entry; DENSE_BLOCK],
    clicks: [Click; DENSE_BLOCK],
}

impl DenseEmit {
    #[inline(always)]
    pub(super) unsafe fn event(&mut self, entry: Entry) {
        *self.events.get_unchecked_mut(self.nevents) = entry;
        self.nevents += 1;
        if self.nevents == self.limit {
            self.flush_events();
        }
    }

    #[inline(always)]
    pub(super) unsafe fn click(&mut self, click: Click) {
        *self.clicks.get_unchecked_mut(self.nclicks) = click;
        self.nclicks += 1;
        if self.nclicks == self.limit {
            self.flush_clicks();
        }
    }

    #[inline(always)]
    unsafe fn person(&mut self, key: u64, h: u64) -> u32 {
        match self.table.slot(key, h, &mut self.inserted) {
            Some(slot) => self.table.person_of(slot),
            None => self.overflow(),
        }
    }

    /// The table went round: give up (the caller sees `full`).
    #[cold]
    #[inline(never)]
    fn overflow(&mut self) -> u32 {
        unsafe { (*self.full).store(true, Ordering::Relaxed) };
        0
    }

    /// Add this parser's inserts to the shared count; past the limit the
    /// shape gives up.
    #[inline(always)]
    unsafe fn settle(&mut self) {
        if self.inserted > 0 {
            let total = (*self.count).fetch_add(self.inserted, Ordering::Relaxed) + self.inserted;
            self.inserted = 0;
            if total > self.max_ids {
                (*self.full).store(true, Ordering::Relaxed);
            }
        }
    }

    #[inline(never)]
    unsafe fn flush_events(&mut self) {
        let t = prof::start();
        let n = self.nevents;
        self.nevents = 0;
        let table = self.table;
        let (shift1, sbits) = (self.shift1, self.sbits);
        let smask = (1u32 << sbits) - 1;
        let base = self.ingest_base;
        let block = std::slice::from_raw_parts(self.events.as_ptr(), n);
        let mut hashes = [0u64; DENSE_BLOCK];
        for (h, entry) in hashes.iter_mut().zip(block) {
            *h = mix(entry.ukey);
        }
        let ahead = self.ahead;
        for &h in &hashes[..n.min(ahead)] {
            table.prefetch(h);
        }
        // The probes first, all of them, then the scatters: the probes'
        // misses do not wait behind the scatters' streaming stores.
        let mut persons = [0u32; DENSE_BLOCK];
        for (k, entry) in block.iter().enumerate() {
            if k + ahead < n {
                table.prefetch(*hashes.get_unchecked(k + ahead));
            }
            if entry.w3 != DEAD {
                *persons.get_unchecked_mut(k) = self.person(entry.ukey, *hashes.get_unchecked(k));
            }
        }
        for (k, entry) in block.iter().enumerate() {
            let mut rec = NO_RECORD;
            if entry.w3 != DEAD {
                let person = *persons.get_unchecked(k);
                let part = (person >> sbits) as usize;
                let idx = person & smask;
                let ts = entry.w3 >> 32;
                rec = self.ev.push_named(part, (u64::from(idx) << 32) | ts);
                if entry.w3 as u32 != 0 {
                    self.buys.push(part, Buy { tp: (ts << 32) | (entry.w3 & 0xFFFF_FFFF), idx, rec });
                }
            }
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
        self.settle();
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
        let table = self.table;
        let (sbits, cbits) = (self.sbits, self.cbits);
        let smask = (1u32 << sbits) - 1;
        let block = std::slice::from_raw_parts(self.clicks.as_ptr(), n);
        let mut hashes = [0u64; DENSE_BLOCK];
        for (h, click) in hashes.iter_mut().zip(block) {
            *h = mix(click.key);
        }
        let ahead = self.ahead;
        for &h in &hashes[..n.min(ahead)] {
            table.prefetch(h);
        }
        for (k, click) in block.iter().enumerate() {
            if k + ahead < n {
                table.prefetch(*hashes.get_unchecked(k + ahead));
            }
            let h = *hashes.get_unchecked(k);
            let person = self.person(click.key, h);
            let part = (person >> sbits) as usize;
            let low = ((person & smask) << cbits) | (click.tc as u32);
            self.cl.push(part, DClick { ckey: click.ckey, w: (click.tc & !0xFFFF_FFFF) | u64::from(low) });
        }
        self.settle();
        prof::add(10, t);
    }
}

/// The person phase of the dense shape: `PersonPhase` with its grouping
/// replaced by counting on the person index.
struct DensePhase<'a> {
    ph: PersonPhase<'a>,
    table: IdTable,
    dir: &'a sys::Directory,
    sbits: u32,
    cbits: u32,
    /// Per index of the partition: events, then the write cursor. Zero
    /// between partitions.
    cnt: Vec<u32>,
    /// Per index: the person's number in this partition, `u32::MAX` when
    /// the person has no events here (and between partitions).
    kof: Vec<u32>,
    /// The indexes of the persons with events, in index order.
    active: Vec<u32>,
}

impl<'a> DensePhase<'a> {
    unsafe fn partition(&mut self, p: usize, evs: &[List<u64>], buys: &[List<Buy>], cls: &[List<DClick>]) {
        let t = prof::start();
        let s = 1usize << self.sbits;
        let base = p << self.sbits;
        let cnt = self.cnt.as_mut_ptr();
        // Pass 1: events per index.
        let mut n = 0usize;
        for list in evs {
            list.for_each_chunk(|chunk| {
                for &r in chunk.iter() {
                    if r == DROPPED {
                        continue;
                    }
                    *cnt.add((r >> 32) as usize) += 1;
                    n += 1;
                }
            });
        }
        if n == 0 {
            return;
        }
        // The persons with events, their names, and where their times go.
        let kof = self.kof.as_mut_ptr();
        let ph = &mut self.ph;
        self.active.clear();
        ph.keys.clear();
        ph.keys.push(0);
        let mut sum = 0u32;
        for idx in 0..s {
            let c = *cnt.add(idx);
            if c != 0 {
                *kof.add(idx) = self.active.len() as u32;
                self.active.push(idx as u32);
                ph.keys.push(self.table.key(base + idx));
                *cnt.add(idx) = sum;
                sum += c;
            }
        }
        let people = self.active.len();
        // Pass 2: the times person after person.
        ph.ev.clear();
        ph.ev.reserve(n + EV_PAD);
        ph.ev.set_len(n + EV_PAD);
        ph.ev[n..].fill(0);
        let ev = ph.ev.as_mut_ptr();
        for list in evs {
            list.for_each_chunk(|chunk| {
                for &r in chunk.iter() {
                    if r == DROPPED {
                        continue;
                    }
                    let w = cnt.add((r >> 32) as usize);
                    *ev.add(*w as usize) = r as u32;
                    *w += 1;
                }
            });
        }
        // End offsets per person; the counts go back to zero.
        ph.counts.clear();
        for &idx in &self.active {
            ph.counts.push(*cnt.add(idx as usize));
            *cnt.add(idx as usize) = 0;
        }

        // The purchases whose event survived, person after person.
        ph.buys.clear();
        let dir = self.dir;
        for list in buys {
            list.for_each_chunk(|chunk| {
                for b in chunk.iter() {
                    if *dir.record::<u64>(b.rec) == DROPPED {
                        continue;
                    }
                    ph.buys.push((b.tp, *kof.add(b.idx as usize)));
                }
            });
        }
        ph.bcounts.clear();
        ph.bcounts.resize(people, 0);
        for &(_, id) in &ph.buys {
            *ph.bcounts.get_unchecked_mut(id as usize) += 1;
        }
        let mut sum = 0u32;
        for c in ph.bcounts.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
        ph.bought.clear();
        ph.bought.reserve(ph.buys.len());
        ph.bought.set_len(ph.buys.len());
        for &(v, id) in &ph.buys {
            let w = ph.bcounts.get_unchecked_mut(id as usize);
            *ph.bought.get_unchecked_mut(*w as usize) = v;
            *w += 1;
        }

        // Clicks of persons with events here; the rest cannot matter.
        let cbits = self.cbits;
        let cmask = (1u64 << cbits) - 1;
        let nc: usize = cls.iter().map(|l| l.count()).sum();
        ph.ccounts.clear();
        ph.ccounts.resize(people + 1, 0);
        ph.cl.clear();
        if nc > 0 {
            ph.cids.clear();
            ph.cids.reserve(nc);
            for list in cls {
                list.for_each_chunk(|chunk| {
                    for c in chunk.iter() {
                        let k = *kof.add(((c.w as u32) >> cbits) as usize);
                        ph.cids.push(k);
                        if k != u32::MAX {
                            *ph.ccounts.get_unchecked_mut(k as usize) += 1;
                        }
                    }
                });
            }
            let mut sum = 0u32;
            for c in ph.ccounts.iter_mut() {
                let k = *c;
                *c = sum;
                sum += k;
            }
            ph.cl.reserve(sum as usize);
            ph.cl.set_len(sum as usize);
            let mut at = 0usize;
            for list in cls {
                list.for_each_chunk(|chunk| {
                    for c in chunk.iter() {
                        let k = *ph.cids.get_unchecked(at);
                        at += 1;
                        if k != u32::MAX {
                            let w = ph.ccounts.get_unchecked_mut(k as usize);
                            *ph.cl.get_unchecked_mut(*w as usize) = ((c.w & !0xFFFF_FFFF) | (c.w & cmask), c.ckey);
                            *w += 1;
                        }
                    }
                });
            }
        }
        for &idx in &self.active {
            *kof.add(idx as usize) = u32::MAX;
        }

        let t = prof::add(7, t);
        let (mut ev_start, mut cl_start, mut buy_start) = (0usize, 0usize, 0usize);
        ph.reserve_rows(people);
        for k in 0..people {
            let ev_end = ph.counts[k] as usize;
            let cl_end = if nc > 0 { ph.ccounts[k] as usize } else { 0 };
            let buy_end = ph.bcounts[k] as usize;
            ph.person(k, ev_start, ev_end, cl_start, cl_end, buy_start, buy_end);
            ev_start = ev_end;
            cl_start = cl_end;
            buy_start = buy_end;
        }
        let t = prof::add(8, t);
        if ph.out.flush_full() {
            prof::add(9, t);
        }
    }
}

/// Run the dense shape after the setup of `engine::run`. `None` when it
/// gives up (see the top of this file); nothing has been written then.
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    fast: &Fast,
    lk: &Lookups,
    cfg: &Config,
    out: Option<&Path>,
    inputs: &Inputs,
    jobs: &[Job],
    (bits1, bits2): (u32, u32),
    drop_pages: bool,
    cpu_total: &Mutex<f64>,
    timer: &mut Timer,
) -> Option<Tables> {
    let threads = cfg.threads;
    let event_bytes: u64 = inputs.events.iter().map(|m| m.len as u64).sum();
    let click_bytes: u64 = inputs.clicks.iter().map(|m| m.len as u64).sum();
    // Named records: bound the chunks as `run` does for the one-pass shape.
    let parts2_max = 1usize << bits2;
    let max_chunks = (event_bytes / 16 / 1000) as usize + threads * (parts2_max + 1) + 16;
    if max_chunks >= sys::DIR_CHUNKS {
        return None;
    }
    let knob = |name: &str, default: usize| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let (note_lines, ev_lines, cl_lines) = (knob("ENGINE_NOTE_LINES", 3), knob("ENGINE_EV_LINES", 3), knob("ENGINE_CL_LINES", 1));
    let new_parser = |dense: Option<Box<DenseEmit>>| Parser {
        st: ParseState {
            fast,
            cache: DateCache::new(),
            entries: Scatter::new(1, CHUNK_BYTES),
            shift1: 64 - bits1,
            clicks: Vec::new(),
            pairs: Vec::new(),
            emit: None,
            dense,
        },
        scan: Scan::new(WINDOW),
        fused: cfg.fused,
    };

    // ---- links: read first, as in the one-pass shape.
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

    // ---- the ID table. Real data has one distinct ID per 1 KB of input
    // or so (see NOTES); the table has room for about four times that and
    // gives up past three quarters full.
    let pairs: usize = pair_lists.iter().map(|l| l.len()).sum();
    let want = 2 * pairs + (event_bytes / 1024) as usize + (click_bytes / 1024) as usize;
    let slots = cfg.dense_slots.unwrap_or_else(|| want.next_power_of_two()).clamp(64, 1 << 31);
    let tbits = slots.trailing_zeros();
    // Half the person partitions of the other shapes: the scatter in parse
    // costs more per record the more partitions it has, and a dense
    // partition costs less per record than a hashed one. Measured at 20M
    // events: 512 against 1024 saves about 100 CPU ms of parse for 10 of
    // person; 256 loses 50 in person.
    let bits2 = if cfg.p2_bits.is_some() { bits2 } else { bits2.saturating_sub(1).max(1) };
    let bits2 = bits2.min(tbits - 2);
    let sbits = tbits - bits2;
    let parts2 = 1usize << bits2;
    let cbits = u32::BITS - (fast.ncamp as u32).leading_zeros();
    if sbits + cbits > 32 {
        return None;
    }
    let max_ids = slots / 4 * 3;
    let build_cpu = prof::get(3);
    let table = IdTable::new(slots);
    let link_ids = table.join_links(&pair_lists, threads)?;
    drop(pair_lists);
    if link_ids > max_ids {
        return None;
    }
    timer.lap("links", std::mem::take(&mut *cpu_total.lock().unwrap()) + prof::get(3) - build_cpu);

    // ---- parse
    let directory = sys::Directory::new();
    let directory = &directory;
    let count = AtomicUsize::new(link_ids);
    let full = AtomicBool::new(false);
    let (count_ref, full_ref) = (Shared(&count as *const AtomicUsize), Shared(&full as *const AtomicBool));
    let jobs: Vec<&Job> = jobs.iter().filter(|j| j.kind != Kind::Links).collect();
    let next = AtomicUsize::new(0);
    let mut note_lists: Vec<Vec<List<Note>>> = Vec::new();
    let mut ev_lists: Vec<Vec<List<u64>>> = Vec::new();
    let mut buy_lists: Vec<Vec<List<Buy>>> = Vec::new();
    let mut cl_lists: Vec<Vec<List<DClick>>> = Vec::new();
    let mut wide: Vec<Wide> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| unsafe {
                    let emit = Box::new(DenseEmit {
                        table,
                        notes: Scatter::with_lines(1 << bits1, CHUNK_BYTES, note_lines),
                        ev: Scatter::named(parts2, DENSE_CHUNK, ev_lines, directory),
                        buys: Scatter::with_lines(parts2, CHUNK_BYTES, 1),
                        cl: Scatter::with_lines(parts2, CHUNK_BYTES, cl_lines),
                        shift1: 64 - bits1,
                        sbits,
                        cbits,
                        ingest_base: note_base(fast.ws),
                        wide: Vec::new(),
                        nevents: 0,
                        nclicks: 0,
                        limit: knob("ENGINE_LOOKUP_BLOCK", DENSE_BLOCK).clamp(1, DENSE_BLOCK),
                        ahead: knob("ENGINE_DENSE_AHEAD", 32),
                        inserted: 0,
                        count: count_ref.ptr(0),
                        full: full_ref.ptr(0),
                        max_ids,
                        events: [Entry { ekey: 0, ukey: 0, ingest: 0, w3: DEAD }; DENSE_BLOCK],
                        clicks: [Click { key: 0, ckey: 0, tc: 0 }; DENSE_BLOCK],
                    });
                    let mut parser = new_parser(Some(emit));
                    while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if full_ref.at(0).load(Ordering::Relaxed) {
                            break;
                        }
                        let files = if job.kind == Kind::Events { inputs.events } else { inputs.clicks };
                        parser.run_chunk(files[job.file], job.start, job.end, job.kind);
                        sys::drop_pages(files[job.file], job.start, job.end, drop_pages);
                    }
                    let mut emit = parser.st.dense.take().unwrap_unchecked();
                    if emit.nevents > 0 {
                        emit.flush_events();
                    }
                    if emit.nclicks > 0 {
                        emit.flush_clicks();
                    }
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    let emit = *emit;
                    (emit.notes.finish(), emit.ev.finish(), emit.buys.finish(), emit.cl.finish(), emit.wide)
                })
            })
            .collect();
        for h in handles {
            let (notes, ev, buys, cl, odd) = h.join().unwrap();
            note_lists.push(notes);
            ev_lists.push(ev);
            buy_lists.push(buys);
            cl_lists.push(cl);
            wide.extend(odd);
        }
    });
    timer.lap("parse", std::mem::take(&mut *cpu_total.lock().unwrap()));
    if full.load(Ordering::Relaxed) {
        if cfg.timing {
            eprintln!("[engine] dense: ID table of {slots} slots too full, falling back");
        }
        return None;
    }
    if cfg.timing {
        let sum = |lists: &[Vec<List<u64>>]| -> usize { lists.iter().flatten().map(|l| unsafe { l.count() }).sum() };
        let notes: usize = note_lists.iter().flatten().map(|l| unsafe { l.count() }).sum();
        let buys: usize = buy_lists.iter().flatten().map(|l| unsafe { l.count() }).sum();
        let clicks: usize = cl_lists.iter().flatten().map(|l| unsafe { l.count() }).sum();
        eprintln!(
            "[engine]   dense: slots {slots}, ids {} (links {link_ids}), parts {parts2}, index bits {sbits}, records: events {} x8, notes {notes} x16, buys {buys} x16, clicks {clicks} x16",
            count.load(Ordering::Relaxed),
            sum(&ev_lists)
        );
    }

    // ---- dedup of the notes
    let parts1 = 1usize << bits1;
    let mut wide_parts: Vec<Vec<Wide>> = Vec::new();
    if !wide.is_empty() {
        wide_parts.resize(parts1, Vec::new());
        for w in wide {
            wide_parts[event_part(w.ekey, 64 - bits1)].push(w);
        }
    }
    let (note_lists_ref, wide_parts) = (&note_lists, &wide_parts);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| unsafe {
                let mut dedup =
                    Deduper { dir: directory, table: Vec::new(), bases: Vec::new(), ingest_base: note_base(fast.ws), dense: true };
                let mut lists: Vec<List<Note>> = Vec::with_capacity(threads);
                loop {
                    let at = next.fetch_add(1, Ordering::Relaxed);
                    if at >= parts1 {
                        break;
                    }
                    lists.clear();
                    lists.extend(note_lists_ref.iter().map(|t| t[at]));
                    dedup.partition(&lists, wide_parts.get(at).map_or(&[][..], |v| &v[..]));
                }
                *cpu_total.lock().unwrap() += sys::thread_cpu();
            });
        }
    });
    timer.lap("resolve", std::mem::take(&mut *cpu_total.lock().unwrap()));

    // ---- person phase: biggest partitions first.
    let mut order: Vec<usize> = (0..parts2).collect();
    let weight = |p: usize| -> u32 { ev_lists.iter().map(|t| t[p].chunks).sum() };
    order.sort_by_key(|&p| std::cmp::Reverse(weight(p)));
    let next = AtomicUsize::new(0);
    let (ev_lists, buy_lists, cl_lists, order) = (&ev_lists, &buy_lists, &cl_lists, &order);
    let collected: Mutex<Vec<UserRow>> = Mutex::new(Vec::new());
    let collected_ref = &collected;
    let shared_cells: Mutex<Cells> = Mutex::new(Cells::default());
    let shared_cells = &shared_cells;
    let mut total_rows = 0u64;
    let out_dir = out.map(|o| o.join("user_daily"));
    let out_dir = &out_dir;
    let next = &next;
    let mut check = (0u64, 0u64);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || unsafe {
                    let user_out =
                        UserOut::new(out_dir.as_deref(), t, lk, fast.ws_day, fast.ndays, cfg.collect.then_some(collected_ref));
                    let mut phase = DensePhase {
                        ph: PersonPhase::new(fast, user_out),
                        table,
                        dir: directory,
                        sbits,
                        cbits,
                        cnt: vec![0; 1 << sbits],
                        kof: vec![u32::MAX; 1 << sbits],
                        active: Vec::new(),
                    };
                    let mut evs: Vec<List<u64>> = Vec::with_capacity(threads);
                    let mut bys: Vec<List<Buy>> = Vec::with_capacity(threads);
                    let mut cls: Vec<List<DClick>> = Vec::with_capacity(threads);
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&p) = order.get(at) else {
                            break;
                        };
                        evs.clear();
                        evs.extend(ev_lists.iter().map(|t| t[p]));
                        bys.clear();
                        bys.extend(buy_lists.iter().map(|t| t[p]));
                        cls.clear();
                        cls.extend(cl_lists.iter().map(|t| t[p]));
                        phase.partition(p, &evs, &bys, &cls);
                    }
                    let t0 = prof::start();
                    phase.ph.out.finish();
                    prof::add(9, t0);
                    let cells = phase.ph.cells();
                    shared_cells.lock().unwrap().add(cells);
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    (phase.ph.out.rows_written, phase.ph.check)
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
    let mut tables = finish(fast, lk, cfg, out, cells, total_rows, check, collected.into_inner().unwrap(), timer);
    tables.dense = true;
    Some(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Threads insert the same keys at the same moment, each in its own
    /// order: every key gets one slot, all threads agree on it, no two keys
    /// share one, and the inserts are counted once each. A table with no
    /// room left says so instead of probing for ever.
    #[test]
    fn concurrent_inserts_agree() {
        for round in 0..40u64 {
            let keys: Vec<u64> = (1..=1500u64).map(|k| (mix(k * 7 + round) >> 1) | 1).collect();
            let slots = if round % 2 == 0 { 1 << 12 } else { 2048 };
            let table = IdTable::new(slots);
            let threads = 16;
            let barrier = std::sync::Barrier::new(threads);
            let found: Vec<(Vec<usize>, usize)> = std::thread::scope(|scope| {
                let hs: Vec<_> = (0..threads)
                    .map(|t| {
                        let (keys, barrier) = (&keys, &barrier);
                        scope.spawn(move || {
                            let mut order: Vec<usize> = (0..keys.len()).collect();
                            let mut s = t as u64 + round * 100;
                            for i in (1..order.len()).rev() {
                                s = mix(s + 1);
                                order.swap(i, (s % (i as u64 + 1)) as usize);
                            }
                            let mut got = vec![0usize; keys.len()];
                            let mut ins = 0;
                            barrier.wait();
                            for &i in &order {
                                got[i] = unsafe { table.slot(keys[i], mix(keys[i]), &mut ins) }.unwrap();
                            }
                            (got, ins)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let first = &found[0].0;
            assert!(found.iter().all(|(g, _)| g == first), "round {round}: threads disagree");
            assert_eq!(found.iter().map(|f| f.1).sum::<usize>(), keys.len(), "round {round}");
            let mut seen = first.clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), keys.len());
            for (i, &slot) in first.iter().enumerate() {
                assert_eq!(table.key(slot), keys[i]);
                assert_eq!(table.person_of(slot), slot as u32);
            }
        }
        // Full: 64 slots, then a 65th key.
        let table = IdTable::new(64);
        let mut ins = 0;
        for k in 1..=64u64 {
            assert!(unsafe { table.slot(k, mix(k), &mut ins) }.is_some());
        }
        assert!(unsafe { table.slot(65, mix(65), &mut ins) }.is_none());
        assert!(unsafe { table.slot(7, mix(7), &mut ins) }.is_some());
    }

    /// Links joined in the table name every ID's person by the smallest ID
    /// of its set, as a plain union-find does.
    #[test]
    fn joined_links_name_the_smallest() {
        use std::collections::HashMap;
        fn root(p: &HashMap<u64, u64>, mut x: u64) -> u64 {
            while p[&x] != x {
                x = p[&x];
            }
            x
        }
        for round in 0..6u64 {
            let mut s = round;
            let mut rng = || {
                s = mix(s + 0x9E37);
                s
            };
            let universe = [50u64, 2000, 20000][(round % 3) as usize];
            let pairs: Vec<(u64, u64)> = (0..universe).map(|_| (1 + rng() % universe, 1 + rng() % universe)).collect();
            let lists: Vec<Vec<(u64, u64)>> = pairs.chunks(97).map(|c| c.to_vec()).collect();
            let table = IdTable::new((universe as usize * 4).next_power_of_two());
            table.join_links(&lists, 8).unwrap();
            let mut parent: HashMap<u64, u64> = HashMap::new();
            for &(a, b) in &pairs {
                parent.entry(a).or_insert(a);
                parent.entry(b).or_insert(b);
                let (ra, rb) = (root(&parent, a), root(&parent, b));
                parent.insert(ra.max(rb), ra.min(rb));
            }
            let mut ins = 0;
            for &k in parent.keys() {
                let slot = unsafe { table.slot(k, mix(k), &mut ins) }.unwrap();
                assert_eq!(table.key(table.person_of(slot) as usize), root(&parent, k), "round {round} key {k}");
            }
            assert_eq!(ins, 0, "every linked ID is in the table");
        }
    }
}
