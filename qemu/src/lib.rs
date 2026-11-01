//! Shared by the QEMU programs: heap sized like the Pico 2's, with a
//! high-water mark; output and exit status through semihosting.

#![no_std]

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};
use cortex_m as _; // its single-core critical-section implementation
use cortex_m_semihosting::{debug, hprintln};
use embedded_alloc::LlffHeap;

pub use cortex_m_rt::entry;
pub use cortex_m_semihosting::hprintln as println;

/// The Pico 2 programs' allocator, plus a high-water mark: the most heap
/// bytes in use at any moment, updated on every allocation.
pub struct Heap {
    inner: LlffHeap,
    peak: AtomicUsize,
}

impl Heap {
    const fn empty() -> Self {
        Self { inner: LlffHeap::empty(), peak: AtomicUsize::new(0) }
    }

    /// # Safety
    /// As `LlffHeap::init`: call once, with memory used for nothing else.
    pub unsafe fn init(&self, start_addr: usize, size: usize) {
        unsafe { self.inner.init(start_addr, size) }
    }

    /// Bytes in use now.
    pub fn used(&self) -> usize {
        self.inner.used()
    }

    /// The most bytes that were ever in use at once.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
}

unsafe impl GlobalAlloc for Heap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            self.peak.fetch_max(self.inner.used(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.inner.dealloc(ptr, layout) }
    }
}

#[global_allocator]
pub static HEAP: Heap = Heap::empty();

pub fn init_heap() {
    unsafe { embedded_alloc::init!(HEAP, 256 * 1024); }
}

pub fn exit(success: bool) -> ! {
    debug::exit(if success { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
    unreachable!("QEMU exits")
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    hprintln!("panic: {}", info);
    exit(false)
}
