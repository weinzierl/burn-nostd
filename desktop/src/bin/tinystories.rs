//! Step 4 on the desktop.
//!
//! `cargo tinystories -- "Once upon a time" [temperature] [steps]`
//! `cargo tinystories -- --model stories15M.bin --tokenizer tokenizer.bin "Lily"`
//! loads bigger llama2.c checkpoints the same way, in place.

#![no_std]
#![no_main]

use burn::tensor::Device;
use core::ffi::{c_char, c_int};
use desktop::{micros, print, println};
use tinystories::{Llama, Sampler, Tokenizer, generate, stories260k};

#[unsafe(no_mangle)]
extern "C" fn main(argc: c_int, argv: *const *const c_char) -> c_int {
    let mut args = desktop::args(argc, argv);
    let mut option = |name: &str| {
        let i = args.iter().position(|a| *a == name)?;
        let value = args.get(i + 1).copied();
        args.drain(i..(i + 2).min(args.len()));
        value
    };
    let (weights, tokens) = match (option("--model"), option("--tokenizer")) {
        (Some(m), Some(t)) => match (desktop::read(m), desktop::read(t)) {
            (Some(m), Some(t)) => (m, t),
            _ => {
                println!("cannot read model or tokenizer");
                return 2;
            }
        },
        _ => (stories260k::WEIGHTS, stories260k::TOKENIZER),
    };
    let prompt = args.first().copied().unwrap_or("Once upon a time");
    let temperature = args.get(1).and_then(|t| t.parse().ok()).unwrap_or(0.0);
    let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);

    let device = Device::default();
    let mut model = Llama::load(weights, &device, steps);
    let tokenizer = Tokenizer::new(tokens, model.config.vocab_size);
    println!("{} bytes of weights, used in place\n", weights.len());

    let t = micros();
    let n = generate(&mut model, &tokenizer, &mut Sampler::new(temperature, micros() as u32), prompt, steps, |piece| {
        print!("{}", core::str::from_utf8(piece).unwrap_or("?"))
    });
    let ms = ((micros() - t) / 1000).max(1);
    println!("\n\n{n} tokens in {ms} ms, {} tokens/s", n as u64 * 1000 / ms);
    0
}
