//! What a no_std WebAssembly module needs besides its own code.

#![no_std]

use core::cell::RefCell;

// The allocator std itself uses on wasm32.
#[global_allocator]
static ALLOCATOR: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

/// State shared by a module's exports, created on first use.
pub struct Global<T>(RefCell<Option<T>>);

// wasm32-unknown-unknown has no threads.
unsafe impl<T> Sync for Global<T> {}

impl<T> Global<T> {
    pub const fn new() -> Self {
        Self(RefCell::new(None))
    }

    pub fn with<R>(&self, init: impl FnOnce() -> T, f: impl FnOnce(&mut T) -> R) -> R {
        f(self.0.borrow_mut().get_or_insert_with(init))
    }

    pub fn reset(&self) {
        self.0.replace(None);
    }
}

impl<T> Default for Global<T> {
    fn default() -> Self {
        Self::new()
    }
}
