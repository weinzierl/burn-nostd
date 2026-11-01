#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use burn::tensor::Device;
use qemu::{HEAP, entry, exit, init_heap, println};
use tinystories::{Llama, Sampler, Tokenizer, generate, stories260k};

/// llama2.c: `run stories260K.bin -z tok512.bin -t 0 -n 80 -i "Once upon a time"`
const REFERENCE: &[u8] = b"Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to";

#[entry]
fn main() -> ! {
    init_heap();
    let device = Device::default();
    let mut model = Llama::load(stories260k::WEIGHTS, &device, 128);
    let tokenizer = Tokenizer::new(stories260k::TOKENIZER, model.config.vocab_size);
    let mut story = Vec::new();
    generate(&mut model, &tokenizer, &mut Sampler::new(0.0, 1), "Once upon a time", 80, |piece| {
        story.extend_from_slice(piece);
    });
    println!("{}", core::str::from_utf8(&story).unwrap_or("?"));
    println!("tinystories: {} bytes of weights from flash, heap peak {} bytes", stories260k::WEIGHTS.len(), HEAP.peak());
    exit(story == REFERENCE)
}
