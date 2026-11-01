#![no_std]
#![no_main]

//! The first four stories of `pico2`'s tinystories-usb, same prompts, temperatures
//! and seeds: the text a board should print, word for word. Also the most stack
//! and heap the four stories need. Passes if the first, greedy story begins with
//! llama2.c's text.

extern crate alloc;

use alloc::vec::Vec;
use burn::tensor::Device;
use qemu::{HEAP, entry, exit, init_heap, println};
use tinystories::{Llama, Sampler, Tokenizer, generate, stories260k};

const PROMPTS: [&str; 4] = ["Once upon a time", "Lily and Ben", "The little dog", "One day, a robot"];
const CONTEXT: usize = 128;
/// llama2.c: `run stories260K.bin -z tok512.bin -t 0 -n 80 -i "Once upon a time"`
const REFERENCE: &[u8] = b"Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to";

unsafe extern "C" {
    static _stack_start: u32; // top of the stack
    static _stack_end: u32; // lowest address it may reach
}
const PAINT: u32 = 0x5a5a_5a5a;

#[entry]
fn main() -> ! {
    // Paint the free stack, leaving 1 KiB below this frame.
    let (top, bottom) = (&raw const _stack_start as usize, &raw const _stack_end as usize);
    let sp = cortex_m::register::msp::read() as usize;
    for a in (bottom..sp - 1024).step_by(4) {
        unsafe { (a as *mut u32).write_volatile(PAINT) }
    }

    init_heap();
    let device = Device::default();
    let mut model = Llama::load(stories260k::WEIGHTS, &device, CONTEXT);
    let tokenizer = Tokenizer::new(stories260k::TOKENIZER, model.config.vocab_size);
    let mut first = Vec::new();
    for round in 0..4 {
        let temperature = if round == 0 { 0.0 } else { 0.8 };
        let mut sampler = Sampler::new(temperature, round as u32 + 1);
        let mut story = Vec::new();
        let tokens = generate(&mut model, &tokenizer, &mut sampler, PROMPTS[round], CONTEXT, |piece| {
            story.extend_from_slice(piece);
        });
        println!("{}\n{} tokens\n", core::str::from_utf8(&story).unwrap_or("?"), tokens);
        if round == 0 {
            first = story;
        }
    }

    let lowest = (bottom..top).step_by(4).find(|&a| unsafe { (a as *const u32).read_volatile() } != PAINT).unwrap_or(top);
    println!("stack peak {} bytes (of {}), heap peak {} bytes", top - lowest, top - bottom, HEAP.peak());
    exit(first.starts_with(REFERENCE))
}
