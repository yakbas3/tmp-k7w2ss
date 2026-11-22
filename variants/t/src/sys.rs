//! OS glue: file mappings with readable padding, bump arenas over lazily
//! committed anonymous memory, chunked append-only lists, thread CPU clocks.

use std::ffi::CString;
use std::marker::PhantomData;
use std::path::Path;

use crate::common::die;

pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// A read-only file mapping followed by at least one zero page, so SIMD and
/// 8-byte loads may read past the end of the file.
#[derive(Clone, Copy)]
pub struct Mapped {
    pub ptr: *const u8,
    pub len: usize,
}
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

static ZEROS: [u8; 256] = [0; 256];

pub fn map_file(path: &Path) -> Mapped {
    let c = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    unsafe {
        let fd = libc::open(c.as_ptr(), libc::O_RDONLY);
        if fd < 0 {
            die(&format!("{}: {}", path.display(), std::io::Error::last_os_error()));
        }
        let mut st: libc::stat = std::mem::zeroed();
        if libc::fstat(fd, &mut st) != 0 {
            die(&format!("{}: {}", path.display(), std::io::Error::last_os_error()));
        }
        let len = st.st_size as usize;
        if len == 0 {
            libc::close(fd);
            return Mapped { ptr: ZEROS.as_ptr().add(64), len: 0 };
        }
        let page = page_size();
        let rounded = (len + page - 1) & !(page - 1);
        // Reserve the file span plus one anonymous page, then map the file
        // over the front. The tail page stays anonymous zeros.
        let base = libc::mmap(
            std::ptr::null_mut(),
            rounded + page,
            libc::PROT_READ,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        if base == libc::MAP_FAILED {
            die(&format!("mmap reserve: {}", std::io::Error::last_os_error()));
        }
        let got = libc::mmap(base, len, libc::PROT_READ, libc::MAP_PRIVATE | libc::MAP_FIXED, fd, 0);
        if got == libc::MAP_FAILED {
            die(&format!("{}: mmap: {}", path.display(), std::io::Error::last_os_error()));
        }
        libc::close(fd);
        Mapped { ptr: base as *const u8, len }
    }
}

/// Ask for transparent huge pages on the big anonymous arenas. On unless
/// `ENGINE_THP=0`.
#[cfg(target_os = "linux")]
fn huge_pages() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_THP").map_or(true, |v| v != "0"))
}

pub fn unmap_file(m: Mapped) {
    if m.len > 0 {
        let page = page_size();
        let rounded = (m.len + page - 1) & !(page - 1);
        unsafe { libc::munmap(m.ptr as *mut libc::c_void, rounded + page) };
    }
}

/// Unmap the whole pages of `[start, end)` of a parsed file chunk. A
/// neighbouring chunk may still be reading the first and last partial
/// lines, so the range is shrunk by a page at both ends; if a later read
/// does touch a dropped page it simply faults back in from the page cache.
pub fn drop_pages(m: Mapped, start: usize, end: usize, on: bool) {
    if !on {
        return;
    }
    let page = page_size();
    let from = (start + page - 1) & !(page - 1);
    let to = end & !(page - 1);
    if to > from + 2 * page {
        unsafe { libc::madvise(m.ptr.add(from + page) as *mut libc::c_void, to - from - 2 * page, libc::MADV_DONTNEED) };
    }
}

use std::sync::atomic::AtomicUsize;

/// Every anonymous region handed out, so they can be returned in parallel
/// before exit. A fixed table: it must not live in the memory it lists.
static REGIONS: [(AtomicUsize, AtomicUsize); 1024] = [const { (AtomicUsize::new(0), AtomicUsize::new(0)) }; 1024];
static REGIONS_USED: AtomicUsize = AtomicUsize::new(0);

/// Unmap all anonymous regions with `threads` threads. Call only when
/// nothing in them is needed any more, right before exit.
///
/// Why: when the process exits, the kernel frees its memory on one CPU,
/// inside the exit, and under a memory cgroup (the scorer's container)
/// that costs about 0.13 ms per MB of 4 KiB pages. The container is not
/// done until that finishes. Unmapping from all threads does the same work
/// in parallel while the clock is still ours.
pub fn release_all(threads: usize) {
    use std::sync::atomic::Ordering;
    const PIECE: usize = 32 << 20;
    let used = REGIONS_USED.load(Ordering::Acquire).min(REGIONS.len());
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    for region in &REGIONS[..used] {
        let (base, bytes) = (region.0.load(Ordering::Acquire), region.1.load(Ordering::Acquire));
        let mut at = 0;
        while at < bytes {
            let len = PIECE.min(bytes - at);
            pieces.push((base + at, len));
            at += len;
        }
    }
    // Small enough to live in the system allocator, not in a region.
    debug_assert!(pieces.capacity() * 16 < (64 << 10));
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(base, len)) = pieces.get(i) else {
                    break;
                };
                unsafe { libc::munmap(base as *mut libc::c_void, len) };
            });
        }
    });
}

pub fn anon(bytes: usize) -> *mut u8 {
    unsafe {
        #[cfg(target_os = "linux")]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE;
        #[cfg(not(target_os = "linux"))]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
        let p = libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0);
        if p == libc::MAP_FAILED {
            die(&format!("mmap {bytes} bytes: {}", std::io::Error::last_os_error()));
        }
        #[cfg(target_os = "linux")]
        if huge_pages() {
            libc::madvise(p, bytes, libc::MADV_HUGEPAGE);
        }
        let slot = REGIONS_USED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if slot < REGIONS.len() {
            REGIONS[slot].0.store(p as usize, std::sync::atomic::Ordering::Relaxed);
            REGIONS[slot].1.store(bytes, std::sync::atomic::Ordering::Release);
        }
        p as *mut u8
    }
}

/// Bump allocator over big anonymous slabs. Nothing is ever freed: the
/// process exits without teardown.
pub struct Arena {
    cur: *mut u8,
    end: *mut u8,
}

const SLAB: usize = 1 << 28;

impl Arena {
    pub fn new() -> Self {
        Arena { cur: std::ptr::null_mut(), end: std::ptr::null_mut() }
    }
    #[inline]
    pub fn alloc(&mut self, bytes: usize) -> *mut u8 {
        let bytes = (bytes + 63) & !63;
        if (self.end as usize) - (self.cur as usize) < bytes {
            let slab = SLAB.max(bytes);
            self.cur = anon(slab);
            self.end = unsafe { self.cur.add(slab) };
        }
        let p = self.cur;
        self.cur = unsafe { self.cur.add(bytes) };
        p
    }
}

#[repr(C, align(64))]
pub struct ChunkHdr {
    next: *mut ChunkHdr,
    len: u32,
}

const HDR: usize = 64;
const STAGE_BUDGET: usize = 256 << 10;

/// One finished list of chunks: the output of a scatter for one partition.
pub struct List<T> {
    head: *mut ChunkHdr,
    pub chunks: u32,
    /// Records in all chunks, so counting them reads no chunk header.
    len: usize,
    _t: PhantomData<T>,
}
impl<T> Clone for List<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for List<T> {}
unsafe impl<T> Send for List<T> {}
unsafe impl<T> Sync for List<T> {}

impl<T> List<T> {
    /// Visit every chunk as a mutable slice.
    #[inline]
    pub unsafe fn for_each_chunk(&self, mut f: impl FnMut(&mut [T])) {
        let mut at = self.head;
        while !at.is_null() {
            let len = (*at).len as usize;
            let data = (at as *mut u8).add(HDR) as *mut T;
            f(std::slice::from_raw_parts_mut(data, len));
            at = (*at).next;
        }
    }
    /// Like `for_each_chunk`, and also passes where the records of the
    /// chunk after this one start (null after the last), for prefetching.
    #[inline]
    pub unsafe fn for_each_chunk_ahead(&self, mut f: impl FnMut(&mut [T], *const u8)) {
        let mut at = self.head;
        while !at.is_null() {
            let len = (*at).len as usize;
            let data = (at as *mut u8).add(HDR) as *mut T;
            let next = (*at).next;
            let ahead = if next.is_null() { std::ptr::null() } else { (next as *const u8).add(HDR) };
            f(std::slice::from_raw_parts_mut(data, len), ahead);
            at = next;
        }
    }
    /// The chunks as an iterator of slices, for loops that keep their state
    /// in locals rather than in a closure's captures.
    #[inline]
    pub unsafe fn chunks(&self) -> Chunks<T> {
        Chunks { at: self.head, _t: PhantomData }
    }
    /// Hand every chunk's memory to `f`. The list must not be used again.
    pub unsafe fn release(&self, mut f: impl FnMut(*mut u8)) {
        let mut at = self.head;
        while !at.is_null() {
            let next = (*at).next;
            f(at as *mut u8);
            at = next;
        }
    }
    pub unsafe fn count(&self) -> usize {
        debug_assert_eq!(self.len, {
            let mut n = 0;
            let mut at = self.head;
            while !at.is_null() {
                n += (*at).len as usize;
                at = (*at).next;
            }
            n
        });
        self.len
    }
}

pub struct Chunks<T> {
    at: *mut ChunkHdr,
    _t: PhantomData<T>,
}

impl<T: 'static> Iterator for Chunks<T> {
    type Item = &'static [T];
    #[inline]
    fn next(&mut self) -> Option<&'static [T]> {
        if self.at.is_null() {
            return None;
        }
        unsafe {
            let len = (*self.at).len as usize;
            let data = (self.at as *const u8).add(HDR) as *const T;
            self.at = (*self.at).next;
            Some(std::slice::from_raw_parts(data, len))
        }
    }
}

/// See `Scatter::hot`.
#[derive(Clone, Copy)]
pub struct Hot {
    fill: *mut u8,
    stage: *mut u8,
    stage_bytes: usize,
    per: usize,
}

#[derive(Clone, Copy)]
struct Part {
    /// Next line-aligned write position in the current chunk, and its end.
    dst: *mut u8,
    end: *mut u8,
    head: *mut ChunkHdr,
    chunks: u32,
    /// Number of the current chunk in the directory, for a named scatter.
    id: u32,
}

/// Bits of a record name that hold the slot in its chunk.
pub const SLOT_BITS: u32 = 10;
/// Chunk numbers a directory can hand out. `u32::MAX` is never a name.
pub const DIR_CHUNKS: usize = (1 << (32 - SLOT_BITS)) - 1;

/// A numbered list of chunks shared by the scatters of all threads, so a
/// record can be named in 32 bits: chunk number, then slot in the chunk.
/// The table is lazily committed anonymous memory.
pub struct Directory {
    chunks: *const AtomicUsize,
    next: AtomicUsize,
}
unsafe impl Send for Directory {}
unsafe impl Sync for Directory {}

impl Directory {
    pub fn new() -> Self {
        Directory { chunks: anon(DIR_CHUNKS * 8) as *const AtomicUsize, next: AtomicUsize::new(0) }
    }
    fn add(&self, chunk: *mut ChunkHdr) -> u32 {
        use std::sync::atomic::Ordering;
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        if id >= DIR_CHUNKS {
            // The caller bounds the chunk count before it starts.
            die("chunk directory is full");
        }
        unsafe { (*self.chunks.add(id)).store(chunk as usize, Ordering::Relaxed) };
        id as u32
    }
    /// The record a name stands for. Call only after the scatters that
    /// wrote the records have finished.
    #[inline(always)]
    pub unsafe fn record<T>(&self, name: u32) -> *mut T {
        use std::sync::atomic::Ordering;
        let chunk = (*self.chunks.add((name >> SLOT_BITS) as usize)).load(Ordering::Relaxed) as *mut u8;
        (chunk.add(HDR) as *mut T).add((name & ((1 << SLOT_BITS) - 1)) as usize)
    }
}

/// Four cache lines from `s` to `d` with non-temporal stores.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn stream4(s: *const u8, d: *mut u8) {
    use core::arch::x86_64::*;
    let v0 = _mm256_load_si256(s as *const __m256i);
    let v1 = _mm256_load_si256(s.add(32) as *const __m256i);
    let v2 = _mm256_load_si256(s.add(64) as *const __m256i);
    let v3 = _mm256_load_si256(s.add(96) as *const __m256i);
    let v4 = _mm256_load_si256(s.add(128) as *const __m256i);
    let v5 = _mm256_load_si256(s.add(160) as *const __m256i);
    let v6 = _mm256_load_si256(s.add(192) as *const __m256i);
    let v7 = _mm256_load_si256(s.add(224) as *const __m256i);
    // One asm block with constant offsets: the intrinsic takes its
    // address in a register, so each store would cost an `lea`.
    core::arch::asm!(
        "vmovntdq [{d}], {v0}",
        "vmovntdq [{d} + 32], {v1}",
        "vmovntdq [{d} + 64], {v2}",
        "vmovntdq [{d} + 96], {v3}",
        "vmovntdq [{d} + 128], {v4}",
        "vmovntdq [{d} + 160], {v5}",
        "vmovntdq [{d} + 192], {v6}",
        "vmovntdq [{d} + 224], {v7}",
        d = in(reg) d,
        v0 = in(ymm_reg) v0, v1 = in(ymm_reg) v1, v2 = in(ymm_reg) v2, v3 = in(ymm_reg) v3,
        v4 = in(ymm_reg) v4, v5 = in(ymm_reg) v5, v6 = in(ymm_reg) v6, v7 = in(ymm_reg) v7,
        options(nostack, preserves_flags),
    );
}

/// Copy `lines` cache lines from an aligned staging buffer to an aligned
/// destination with non-temporal stores: no read-for-ownership of the
/// destination and no cache pollution.
#[inline(always)]
unsafe fn stream_lines<const FOUR: bool>(src: *const u8, dst: *mut u8, lines: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        // Four lines a step (a staging buffer is usually 4 or 12 lines),
        // then one line a step: fewer loop instructions per line.
        let (mut s, mut d) = (src, dst);
        let end = src.add(lines * 64);
        if FOUR && lines == 4 {
            // The usual staging buffer: no loop. (Only where it measured
            // fewer instructions: the parse scatter.)
            stream4(s, d);
            return;
        }
        while s.add(256) <= end {
            stream4(s, d);
            s = s.add(256);
            d = d.add(256);
        }
        while s < end {
            let v0 = _mm256_load_si256(s as *const __m256i);
            let v1 = _mm256_load_si256(s.add(32) as *const __m256i);
            core::arch::asm!(
                "vmovntdq [{d}], {v0}",
                "vmovntdq [{d} + 32], {v1}",
                d = in(reg) d,
                v0 = in(ymm_reg) v0, v1 = in(ymm_reg) v1,
                options(nostack, preserves_flags),
            );
            s = s.add(64);
            d = d.add(64);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    std::ptr::copy_nonoverlapping(src, dst, lines * 64);
}

/// Append-only scatter into `parts` partitions. Records first land in a
/// small per-partition staging buffer (a whole number of cache lines) and
/// move to the partition's current chunk one full buffer at a time, so the
/// scatter touches one staging line per record, not a chunk tail and a TLB
/// entry per partition. Chunks are cut from a thread-local arena.
pub struct Scatter<T> {
    parts: Vec<Part>,
    /// Records waiting in each staging buffer.
    fill: Vec<u8>,
    stage: *mut u8,
    /// Bytes and records per staging buffer.
    stage_bytes: usize,
    per: usize,
    arena: Arena,
    /// Chunks handed back by a consumer, reused before the arena grows.
    free: Vec<*mut u8>,
    chunk_bytes: usize,
    /// Records per chunk; a multiple of `per`.
    cap: usize,
    /// Where chunks get their numbers; null for a scatter without names.
    dir: *const Directory,
    /// Named scatter: the name of the next record of each partition.
    names: Vec<u32>,
    _t: PhantomData<T>,
}
unsafe impl<T> Send for Scatter<T> {}

impl<T: Copy> Scatter<T> {
    pub fn new(parts: usize, chunk_bytes: usize) -> Self {
        Self::with_lines(parts, chunk_bytes, 4)
    }

    /// Like `new`, with at most `max_lines` cache lines of records in each
    /// staging buffer (rounded up to a whole record count).
    pub fn with_lines(parts: usize, chunk_bytes: usize, max_lines: usize) -> Self {
        let size = std::mem::size_of::<T>();
        let mut per = 1;
        while (per * size) % 64 != 0 {
            per += 1;
        }
        // A longer staging buffer flushes less often, which mostly saves
        // mispredicted flush branches; keep all buffers within ~256 KB so
        // they stay in L2 next to the parser's own data.
        let lines = (STAGE_BUDGET / (parts * per * size)).clamp(1, max_lines.max(1));
        per *= lines;
        assert!(per <= 255);
        let stage_bytes = per * size;
        let chunk_bytes = chunk_bytes.max(HDR + stage_bytes);
        let cap = (chunk_bytes - HDR) / stage_bytes * per;
        let mut arena = Arena::new();
        let stage = arena.alloc(parts * stage_bytes);
        let empty = Part { dst: std::ptr::null_mut(), end: std::ptr::null_mut(), head: std::ptr::null_mut(), chunks: 0, id: 0 };
        Scatter {
            parts: vec![empty; parts],
            fill: vec![0; parts],
            stage,
            stage_bytes,
            per,
            arena,
            free: Vec::new(),
            chunk_bytes,
            cap,
            dir: std::ptr::null(),
            names: Vec::new(),
            _t: PhantomData,
        }
    }

    /// A scatter whose records have names (see `push_named`). `dir` must
    /// outlive it.
    pub fn named(parts: usize, chunk_bytes: usize, max_lines: usize, dir: &Directory) -> Self {
        let mut scatter = Self::with_lines(parts, chunk_bytes, max_lines);
        assert!(scatter.cap < 1 << SLOT_BITS);
        scatter.dir = dir;
        scatter.names = vec![scatter.cap as u32; parts];
        scatter
    }

    /// Give a dead chunk of exactly `bytes` bytes to this scatter to reuse.
    #[inline]
    pub fn recycle(&mut self, chunk: *mut u8, bytes: usize) {
        if bytes == self.chunk_bytes {
            self.free.push(chunk);
        }
    }

    pub fn spare(&self) -> usize {
        self.free.len()
    }

    /// Hand every spare chunk to `to`, if its chunks are the same size.
    pub fn give_spare<U: Copy>(&mut self, to: &mut Scatter<U>) {
        if self.chunk_bytes == to.chunk_bytes {
            to.free.append(&mut self.free);
        }
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    /// The fields `push_hot` reads on every record. They never change for
    /// the life of the scatter, so a caller keeps a copy in locals, which
    /// its own stores cannot alias.
    pub fn hot(&mut self) -> Hot {
        Hot { fill: self.fill.as_mut_ptr(), stage: self.stage, stage_bytes: self.stage_bytes, per: self.per }
    }

    /// `push` with the per-record fields from `hot()`.
    #[inline(always)]
    pub unsafe fn push_hot(&mut self, h: &Hot, p: usize, value: T) {
        let fill = h.fill.add(p);
        let n = *fill as usize;
        let buf = h.stage.add(p * h.stage_bytes);
        (buf as *mut T).add(n).write(value);
        if n + 1 == h.per {
            *fill = 0;
            self.write_out(p, buf);
        } else {
            *fill = (n + 1) as u8;
        }
    }

    /// Move the full staging buffer `buf` of partition `p` to its chunk.
    #[inline(always)]
    unsafe fn write_out(&mut self, p: usize, buf: *mut u8) {
        let mut part = self.parts.as_mut_ptr().add(p);
        if (*part).dst == (*part).end {
            self.grow(p);
            part = self.parts.as_mut_ptr().add(p);
        }
        stream_lines::<false>(buf, (*part).dst, self.stage_bytes / 64);
        (*part).dst = (*part).dst.add(self.stage_bytes);
    }

    #[inline(always)]
    pub unsafe fn push(&mut self, p: usize, value: T) {
        let n = *self.fill.get_unchecked(p) as usize;
        let buf = self.stage.add(p * self.stage_bytes);
        (buf as *mut T).add(n).write(value);
        if n + 1 == self.per {
            *self.fill.get_unchecked_mut(p) = 0;
            let mut part = self.parts.as_mut_ptr().add(p);
            if (*part).dst == (*part).end {
                self.grow(p);
                part = self.parts.as_mut_ptr().add(p);
            }
            stream_lines::<true>(buf, (*part).dst, self.stage_bytes / 64);
            (*part).dst = (*part).dst.add(self.stage_bytes);
        } else {
            *self.fill.get_unchecked_mut(p) = (n + 1) as u8;
        }
    }

    /// Like `push`, and returns the name of the record: where it will be
    /// once its staging buffer has been written out (by a later push or by
    /// `finish`), as chunk number and slot. The chunk is chosen when the
    /// first record of a staging buffer arrives, so the name is final from
    /// the start. A scatter must use either `push` or `push_named`
    /// throughout.
    ///
    /// The name of the next record of each partition is kept in `names`
    /// (4 bytes a partition, so it stays in L1 with `fill`); the partition
    /// itself is read only when a staging buffer is written out. A slot
    /// equal to `cap` means the chunk is full, or there is none yet.
    #[inline(always)]
    pub unsafe fn push_named(&mut self, p: usize, value: T) -> u32 {
        let n = *self.fill.get_unchecked(p) as usize;
        let mut at = *self.names.get_unchecked(p);
        if (at & ((1 << SLOT_BITS) - 1)) as usize == self.cap {
            // Only at the start of a staging buffer: `cap` is a multiple
            // of `per`.
            self.grow(p);
            at = *self.names.get_unchecked(p);
        }
        *self.names.get_unchecked_mut(p) = at + 1;
        let buf = self.stage.add(p * self.stage_bytes);
        (buf as *mut T).add(n).write(value);
        if n + 1 == self.per {
            *self.fill.get_unchecked_mut(p) = 0;
            let part = self.parts.as_mut_ptr().add(p);
            stream_lines::<false>(buf, (*part).dst, self.stage_bytes / 64);
            (*part).dst = (*part).dst.add(self.stage_bytes);
        } else {
            *self.fill.get_unchecked_mut(p) = (n + 1) as u8;
        }
        at
    }

    #[cold]
    #[inline(never)]
    unsafe fn grow(&mut self, p: usize) {
        let size = std::mem::size_of::<T>();
        let part = &mut self.parts[p];
        if !part.head.is_null() {
            (*part.head).len = self.cap as u32;
        }
        let chunk = match self.free.pop() {
            Some(chunk) => chunk as *mut ChunkHdr,
            None => self.arena.alloc(self.chunk_bytes) as *mut ChunkHdr,
        };
        (*chunk).next = part.head;
        (*chunk).len = 0;
        part.head = chunk;
        part.chunks += 1;
        if !self.dir.is_null() {
            part.id = (*self.dir).add(chunk);
        }
        part.dst = (chunk as *mut u8).add(HDR);
        part.end = part.dst.add(self.cap * size);
        if !self.dir.is_null() {
            self.names[p] = self.parts[p].id << SLOT_BITS;
        }
    }

    pub fn finish(mut self) -> Vec<List<T>> {
        let size = std::mem::size_of::<T>();
        unsafe {
            for p in 0..self.parts.len() {
                let n = self.fill[p] as usize;
                if n > 0 {
                    if self.parts[p].dst == self.parts[p].end {
                        self.grow(p);
                    }
                    let part = &mut self.parts[p];
                    std::ptr::copy_nonoverlapping(self.stage.add(p * self.stage_bytes), part.dst, n * size);
                    part.dst = part.dst.add(n * size);
                }
            }
            #[cfg(target_arch = "x86_64")]
            core::arch::x86_64::_mm_sfence();
        }
        self.parts
            .iter()
            .map(|part| {
                let mut len = 0;
                if !part.head.is_null() {
                    let start = unsafe { (part.head as *mut u8).add(HDR) };
                    let used = (part.dst as usize - start as usize) / size;
                    unsafe { (*part.head).len = used as u32 };
                    // Every chunk but the head is full (`grow`).
                    len = (part.chunks as usize - 1) * self.cap + used;
                }
                List { head: part.head, chunks: part.chunks, len, _t: PhantomData }
            })
            .collect()
    }
}

#[inline(always)]
pub fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

/// Sub-phase CPU accounting, summed over threads. Off (one predictable
/// branch per call) unless `ENGINE_TIMING` is set.
pub mod prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    pub const NAMES: [&str; 11] = [
        "parse.scan",
        "parse.fields",
        "links.parse",
        "links.build",
        "resolve.dedup",
        "resolve.emit",
        "resolve.clicks",
        "person.group",
        "person.sessions",
        "person.flush",
        "parse.lookup",
    ];
    pub static ON: AtomicBool = AtomicBool::new(false);
    static NANOS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

    #[inline(always)]
    pub fn start() -> f64 {
        if ON.load(Ordering::Relaxed) {
            super::thread_cpu()
        } else {
            0.0
        }
    }

    /// Charge the time since `since` to `slot`; returns the new start.
    #[inline(always)]
    pub fn add(slot: usize, since: f64) -> f64 {
        if ON.load(Ordering::Relaxed) {
            let now = super::thread_cpu();
            NANOS[slot].fetch_add(((now - since) * 1e9) as u64, Ordering::Relaxed);
            now
        } else {
            0.0
        }
    }

    /// Seconds charged to `slot` so far.
    pub fn get(slot: usize) -> f64 {
        NANOS[slot].load(Ordering::Relaxed) as f64 / 1e9
    }

    pub fn report() {
        for (name, nanos) in NAMES.iter().zip(&NANOS) {
            eprintln!("[engine]   {name:<16} cpu {:7.1} ms", nanos.load(Ordering::Relaxed) as f64 / 1e6);
        }
    }
}

/// CPU seconds used by the calling thread.
pub fn thread_cpu() -> f64 {
    unsafe {
        let mut ts: libc::timespec = std::mem::zeroed();
        libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts);
        ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
    }
}

pub fn threads() -> usize {
    let from_env = std::env::var("NCPU").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0);
    let n = from_env.unwrap_or_else(|| std::thread::available_parallelism().map_or(8, usize::from));
    n.clamp(1, 256)
}

pub fn exit_now(code: i32) -> ! {
    crate::layout_pad!(layout::EXIT_PAD);
    unsafe { libc::_exit(code) }
}

/// Code layout knobs, read at build time; all 0 (no effect) by default.
/// They exist to time builds that differ only in where code sits. None
/// changes an executed instruction except one `jmp` over the padding.
///
/// - `SS_EVENTS_PAD=N`: N bytes (a 2-byte `jmp` over N-2 `int3`) at the head
///   of `ParseState::events`, which moves its loop by N bytes.
/// - `SS_EXIT_PAD=N`: the same in `exit_now`, run once at exit; it moves the
///   code laid out after `exit_now`.
/// - `SS_DIE_PAD=N`: the same in `common::die`, never run on a good input;
///   it moves the code laid out after `die`.
/// - RUSTFLAGS `--cfg ss_hot`: `events` in a section of its own, aligned to
///   4096 bytes, so its address mod 4096 does not depend on other code.
pub mod layout {
    const fn num(v: Option<&str>) -> usize {
        let Some(s) = v else {
            return 0;
        };
        let b = s.as_bytes();
        let mut i = 0;
        let mut n = 0;
        while i < b.len() {
            assert!(b[i].is_ascii_digit(), "layout knob: digits only");
            n = n * 10 + (b[i] - b'0') as usize;
            i += 1;
        }
        assert!(n == 0 || (n >= 2 && n <= 65536), "layout knob: 0 or 2..=65536");
        n
    }
    pub const EVENTS_PAD: usize = num(option_env!("SS_EVENTS_PAD"));
    pub const EXIT_PAD: usize = num(option_env!("SS_EXIT_PAD"));
    pub const DIE_PAD: usize = num(option_env!("SS_DIE_PAD"));
}

/// `$n` bytes of padding jumped over (nothing for 0); see `layout`.
#[macro_export]
macro_rules! layout_pad {
    ($n:expr) => {
        #[cfg(target_arch = "x86_64")]
        if $n >= 2 {
            unsafe {
                core::arch::asm!("jmp 2f", ".skip {n}, 0xcc", "2:", n = const $n.saturating_sub(2), options(nomem, nostack, preserves_flags));
            }
        }
    };
}
