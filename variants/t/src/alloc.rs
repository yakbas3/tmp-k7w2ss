//! Global allocator: big blocks come from a never-freed bump region on
//! (transparent) huge pages; small ones go to the system allocator.
//!
//! Why: the process lives for a second and exits without teardown. A
//! general allocator gives every big vector its own mmap and returns it
//! with munmap, and each munmap in a threaded process is a TLB shootdown
//! (an IPI to every core) plus fresh page faults on the next mmap. Fresh
//! bump memory is also known to be zero, so `alloc_zeroed` is free.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub struct Bump;

pub const BIG: usize = 64 << 10;
const SLAB: usize = 1 << 30;

static NEXT: AtomicUsize = AtomicUsize::new(0);
static END: AtomicUsize = AtomicUsize::new(0);
static LOCK: AtomicBool = AtomicBool::new(false);

#[cold]
unsafe fn refill(need: usize) {
    while LOCK.swap(true, Ordering::Acquire) {
        std::hint::spin_loop();
    }
    let next = NEXT.load(Ordering::Relaxed);
    let end = END.load(Ordering::Relaxed);
    if end.wrapping_sub(next) < need || end == 0 {
        let size = SLAB.max(need + (2 << 20));
        let base = crate::sys::anon(size) as usize;
        // Order matters for the lock-free readers: shrink the window first.
        NEXT.store(usize::MAX / 2, Ordering::Release);
        END.store(base + size, Ordering::Release);
        NEXT.store(base, Ordering::Release);
    }
    LOCK.store(false, Ordering::Release);
}

unsafe fn bump(size: usize, align: usize) -> *mut u8 {
    let align = align.max(64);
    loop {
        let next = NEXT.load(Ordering::Acquire);
        let end = END.load(Ordering::Acquire);
        let start = (next.wrapping_add(align - 1)) & !(align - 1);
        let stop = start.wrapping_add(size);
        if next == 0 || next > end || stop > end || stop < start {
            refill(size + align);
            continue;
        }
        if NEXT.compare_exchange_weak(next, stop, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
            // `end` may be stale only if a refill replaced the slab between
            // the two loads; then `next` changed as well and the exchange
            // failed. So the block lies inside the current slab.
            return start as *mut u8;
        }
    }
}

unsafe impl GlobalAlloc for Bump {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() < BIG {
            System.alloc(layout)
        } else {
            bump(layout.size(), layout.align())
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() < BIG {
            System.alloc_zeroed(layout)
        } else {
            // Bump memory is fresh anonymous pages and is never reused.
            bump(layout.size(), layout.align())
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() < BIG {
            System.dealloc(ptr, layout);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old = layout.size();
        if old < BIG && new_size < BIG {
            return System.realloc(ptr, layout, new_size);
        }
        if old >= BIG && new_size >= BIG {
            if new_size <= old {
                return ptr;
            }
            // Grow in place when this block is the last one handed out.
            let top = ptr as usize + old;
            let want = ptr as usize + new_size;
            if want <= END.load(Ordering::Acquire)
                && NEXT.compare_exchange(top, want, Ordering::AcqRel, Ordering::Relaxed).is_ok()
            {
                return ptr;
            }
        }
        let fresh = self.alloc(Layout::from_size_align_unchecked(new_size, layout.align()));
        if !fresh.is_null() {
            std::ptr::copy_nonoverlapping(ptr, fresh, old.min(new_size));
            self.dealloc(ptr, layout);
        }
        fresh
    }
}
