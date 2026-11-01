//! Step 2 on the desktop: learning gestures from synthetic IMU data.

#![no_std]
#![no_main]

use core::ffi::{c_char, c_int};
use desktop::{micros, println};
use gesture::{Learner, synth};

#[unsafe(no_mangle)]
extern "C" fn main(_: c_int, _: *const *const c_char) -> c_int {
    let mut rng = synth::Rng::new(7);
    let mut learner = Learner::new(42);
    for class in 0..4 {
        for _ in 0..5 {
            learner.add_example(&synth::window(class, &mut rng), class);
        }
    }
    let t = micros();
    let loss = learner.train(15);
    println!("{} examples, 15 epochs: {} ms, loss {loss:.3}\n", learner.examples(), (micros() - t) / 1000);

    let mut correct = 0;
    for (class, name) in synth::NAMES.iter().enumerate() {
        let hits = (0..25).filter(|_| learner.classify(&synth::window(class, &mut rng)).class == class).count();
        println!("{name:>7}: {hits}/25 unseen windows");
        correct += hits;
    }
    let t = micros();
    learner.classify(&synth::window(1, &mut rng));
    println!("\naccuracy {correct}/100, {} µs per classification", micros() - t);
    if correct >= 90 { 0 } else { 1 }
}
