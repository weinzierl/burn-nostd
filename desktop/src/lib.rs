//! What std would provide, from libc: allocator, stdout, clock, files, argv
//! and a panic handler. This is everything a no_std program needs on a
//! desktop OS.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};
use core::ffi::{CStr, c_char, c_int};
use core::fmt::{self, Write};

struct Malloc;

unsafe impl GlobalAlloc for Malloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut ptr = core::ptr::null_mut();
        let align = layout.align().max(size_of::<usize>());
        match unsafe { libc::posix_memalign(&mut ptr, align, layout.size()) } {
            0 => ptr.cast(),
            _ => core::ptr::null_mut(),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _: Layout) {
        unsafe { libc::free(ptr.cast()) }
    }
}

#[global_allocator]
static ALLOCATOR: Malloc = Malloc;

pub struct Stdout;

impl Write for Stdout {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let mut rest = s.as_bytes();
        while !rest.is_empty() {
            match unsafe { libc::write(1, rest.as_ptr().cast(), rest.len()) } {
                n if n > 0 => rest = &rest[n as usize..],
                _ => return Err(fmt::Error),
            }
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {{ let _ = core::fmt::Write::write_fmt(&mut $crate::Stdout, format_args!($($arg)*)); }};
}

#[macro_export]
macro_rules! println {
    ($($arg:tt)*) => {{ $crate::print!($($arg)*); $crate::print!("\n"); }};
}

/// Monotonic time in microseconds.
pub fn micros() -> u64 {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as u64 * 1_000_000 + t.tv_nsec as u64 / 1_000
}

/// Command-line arguments without the program name.
pub fn args(argc: c_int, argv: *const *const c_char) -> Vec<&'static str> {
    (1..argc as usize)
        .filter_map(|i| unsafe { CStr::from_ptr(*argv.add(i)) }.to_str().ok())
        .collect()
}

/// A whole file, 16-byte aligned and leaked, so model weights can be
/// borrowed in place for the rest of the program.
pub fn read(path: &str) -> Option<&'static [u8]> {
    let path = alloc::ffi::CString::new(path).ok()?;
    unsafe {
        let file = libc::fopen(path.as_ptr(), c"rb".as_ptr());
        if file.is_null() {
            return None;
        }
        libc::fseek(file, 0, libc::SEEK_END);
        let len = libc::ftell(file) as usize;
        libc::fseek(file, 0, libc::SEEK_SET);
        let buffer: &mut [u128] = Vec::leak(alloc::vec![0; len.div_ceil(16)]);
        let read = libc::fread(buffer.as_mut_ptr().cast(), 1, len, file);
        libc::fclose(file);
        Some(core::slice::from_raw_parts(buffer.as_ptr().cast(), read))
    }
}

pub fn write(path: &str, data: &[u8]) -> bool {
    let Ok(path) = alloc::ffi::CString::new(path) else { return false };
    unsafe {
        let file = libc::fopen(path.as_ptr(), c"wb".as_ptr());
        if file.is_null() {
            return false;
        }
        let written = libc::fwrite(data.as_ptr().cast(), 1, data.len(), file);
        libc::fclose(file) == 0 && written == data.len()
    }
}

/// 16 kHz mono samples from a 16-bit PCM WAV file (any sample rate, mono or
/// stereo) or from raw 16 kHz mono 16-bit PCM.
pub fn audio(file: &[u8]) -> Result<Vec<f32>, &'static str> {
    if !file.starts_with(b"RIFF") {
        return Ok(samples(file, 1));
    }
    let (mut channels, mut rate, mut data) = (0, 0, None);
    let mut at = 12;
    while at + 8 <= file.len() {
        let len = u32::from_le_bytes(file[at + 4..at + 8].try_into().unwrap()) as usize;
        let body = &file[at + 8..(at + 8 + len).min(file.len())];
        match &file[at..at + 4] {
            b"fmt " if body.len() >= 16 => {
                let bits = u16::from_le_bytes([body[14], body[15]]);
                if bits != 16 {
                    return Err("only 16-bit PCM WAV files are supported");
                }
                channels = u16::from_le_bytes([body[2], body[3]]) as usize;
                rate = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
            }
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + len + (len & 1);
    }
    let (Some(data), 1..) = (data, channels) else { return Err("not a PCM WAV file") };
    let mono = samples(data, channels);
    if rate == 16_000 {
        return Ok(mono);
    }
    // Linear interpolation is enough for speech features up to 7.6 kHz.
    let step = rate as f32 / 16_000.0;
    let n = (mono.len() as f32 / step) as usize;
    Ok((0..n)
        .map(|i| {
            let x = i as f32 * step;
            let (j, f) = (x as usize, x - libm::floorf(x));
            mono[j] * (1.0 - f) + mono.get(j + 1).copied().unwrap_or(0.0) * f
        })
        .collect())
}

/// Interleaved 16-bit frames to mono f32.
fn samples(data: &[u8], channels: usize) -> Vec<f32> {
    data.chunks_exact(2 * channels)
        .map(|frame| {
            let sum: f32 = frame.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32).sum();
            sum / channels as f32 / 32768.0
        })
        .collect()
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let _ = writeln!(Stdout, "panic: {info}");
    unsafe { libc::abort() }
}

// The prebuilt core and alloc for *-linux-gnu come compiled with unwinding
// and reference these, even under panic = "abort". Never called.
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}

#[unsafe(no_mangle)]
extern "C" fn _Unwind_Resume() -> ! {
    unsafe { libc::abort() }
}
