//! Step 4 on the Pico 2: TinyStories 260K from flash. The LED flickers with
//! every token; stories go to the debug probe.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use burn::tensor::Device;
use defmt::info;
use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_time::{Instant, Timer};
use pico2::{HEAP, init_heap, program_info};
use tinystories::{Llama, Sampler, Tokenizer, generate, stories260k};

program_info!(c"burn-tinystories", c"Step 4: llama2.c TinyStories 260K on Burn");

const PROMPTS: [&str; 4] = ["Once upon a time", "Lily and Ben", "The little dog", "One day, a robot"];
/// 128 positions keep the KV cache at 160 KiB.
const CONTEXT: usize = 128;

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    init_heap();
    let p = embassy_rp::init(Default::default());
    let mut led = Output::new(p.PIN_25, Level::Low);

    let device = Device::default();
    let mut model = Llama::load(stories260k::WEIGHTS, &device, CONTEXT);
    let tokenizer = Tokenizer::new(stories260k::TOKENIZER, model.config.vocab_size);
    info!("{} bytes of weights in flash, heap {} bytes", stories260k::WEIGHTS.len(), HEAP.used());

    for (round, prompt) in PROMPTS.iter().cycle().enumerate() {
        // The first story is greedy, so it can be compared with llama2.c.
        let temperature = if round == 0 { 0.0 } else { 0.8 };
        let mut sampler = Sampler::new(temperature, round as u32 + 1);
        let mut story = Vec::new();
        let t = Instant::now();
        let tokens = generate(&mut model, &tokenizer, &mut sampler, prompt, CONTEXT, |piece| {
            story.extend_from_slice(piece);
            led.toggle();
        });
        let ms = t.elapsed().as_millis().max(1);
        led.set_low();
        info!("{=str}", core::str::from_utf8(&story).unwrap_or("(invalid UTF-8)"));
        info!("{} tokens in {} ms: {} tokens/s, heap {} bytes", tokens, ms, tokens as u64 * 1000 / ms, HEAP.used());
        Timer::after_secs(2).await;
    }
}
