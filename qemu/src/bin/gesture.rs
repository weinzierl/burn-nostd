#![no_std]
#![no_main]

use gesture::{Learner, synth};
use qemu::{HEAP, entry, exit, init_heap, println};

#[entry]
fn main() -> ! {
    init_heap();
    let mut rng = synth::Rng::new(7);
    let mut learner = Learner::new(42);
    for class in 0..4 {
        for _ in 0..5 {
            learner.add_example(&synth::window(class, &mut rng), class);
        }
    }
    let loss = learner.train(15);
    let after = HEAP.used();
    let correct = (0..100)
        .filter(|i| learner.classify(&synth::window(i % 4, &mut rng)).class == i % 4)
        .count();
    println!("gesture: loss {}, {}/100 correct, heap after training {} bytes, heap peak {} bytes", loss, correct, after, HEAP.peak());
    exit(correct >= 90)
}
