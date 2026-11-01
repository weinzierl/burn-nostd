#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use keywords::{Learner, Segmenter};
use qemu::{HEAP, entry, exit, init_heap, println};

const ENROLL: &[u8] = include_bytes!("../../../3-keywords/testdata/enroll.i16");
const STREAM: &[u8] = include_bytes!("../../../3-keywords/testdata/stream.i16");
const WORDS: [&str; 4] = ["yes", "no", "left", "right"];

fn pcm(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
}

#[entry]
fn main() -> ! {
    init_heap();
    let mut learner = Learner::new(42);
    for (i, clip) in ENROLL.chunks_exact(48_000).enumerate() {
        learner.add_example(&Segmenter::longest_word(pcm(clip)).expect("a word"), i / 5);
    }
    learner.train(15);

    // Stream the unseen recordings in 20 ms chunks, as a microphone would.
    let mut segmenter = Segmenter::new();
    let mut heard = Vec::new();
    let mut chunk = [0.0; 320];
    let mut on_word = |frames: &[_]| {
        let word = learner.classify(frames).word;
        println!("heard {}", WORDS[word]);
        heard.push(word);
    };
    for bytes in STREAM.chunks(640) {
        let n = chunk.iter_mut().zip(pcm(bytes)).map(|(c, s)| *c = s).count();
        segmenter.push(&chunk[..n], &mut on_word);
    }
    segmenter.flush(&mut on_word);
    let correct = heard.iter().enumerate().filter(|(i, w)| **w == i % 4).count();
    println!("keywords: {}/12 correct, heap peak {} bytes", correct, HEAP.peak());
    exit(heard.len() == 12 && correct >= 11)
}
