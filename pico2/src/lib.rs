//! Shared by all four programs: heap, logging, panics.

#![no_std]

use embedded_alloc::LlffHeap;
use {defmt_rtt as _, panic_probe as _};

#[global_allocator]
pub static HEAP: LlffHeap = LlffHeap::empty();

/// Half the SRAM. The largest peak measured in QEMU is 212 KiB (step 3).
pub const HEAP_SIZE: usize = 256 * 1024;

/// Call first thing in `main`.
pub fn init_heap() {
    unsafe { embedded_alloc::init!(HEAP, HEAP_SIZE); }
}

/// Program name for `picotool info`.
#[macro_export]
macro_rules! program_info {
    ($name:literal, $description:literal) => {
        #[unsafe(link_section = ".bi_entries")]
        #[used]
        static PICOTOOL_ENTRIES: [embassy_rp::binary_info::EntryAddr; 2] = [
            embassy_rp::binary_info::rp_program_name!($name),
            embassy_rp::binary_info::rp_program_description!($description),
        ];
    };
}
