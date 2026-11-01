//! Step 2 in the browser: learns gestures from a phone's motion sensors.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use gesture::{Learner, Sample, WINDOW};
use rt::Global;

struct State {
    learner: Learner,
    recent: [Sample; WINDOW],
    next: usize,
    filled: usize,
    probabilities: Vec<f32>,
}

static STATE: Global<State> = Global::new();

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(
        || State { learner: Learner::new(42), recent: [[0.0; 6]; WINDOW], next: 0, filled: 0, probabilities: Vec::new() },
        f,
    )
}

fn window(s: &State) -> [Sample; WINDOW] {
    core::array::from_fn(|i| s.recent[(s.next + i) % WINDOW])
}

#[unsafe(no_mangle)]
pub extern "C" fn reset() {
    STATE.reset();
}

/// One 50 Hz sample: acceleration in g, rotation in deg/s.
#[unsafe(no_mangle)]
pub extern "C" fn push(ax: f32, ay: f32, az: f32, gx: f32, gy: f32, gz: f32) {
    with(|s| {
        s.recent[s.next] = [ax, ay, az, gx, gy, gz];
        s.next = (s.next + 1) % WINDOW;
        s.filled = (s.filled + 1).min(WINDOW);
    })
}

/// Keeps the last second as an example of `class`.
#[unsafe(no_mangle)]
pub extern "C" fn enroll(class: u32) -> u32 {
    with(|s| {
        let w = window(s);
        s.learner.add_example(&w, class as usize);
        s.learner.examples() as u32
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn train(epochs: u32) -> f32 {
    with(|s| s.learner.train(epochs as usize))
}

/// Classifies the last second; returns how many classes have a
/// probability (0 before a full second is in).
#[unsafe(no_mangle)]
pub extern "C" fn classify() -> u32 {
    with(|s| {
        if s.filled < WINDOW || s.learner.examples() == 0 {
            return 0;
        }
        s.probabilities = s.learner.probabilities(&window(s));
        s.probabilities.len() as u32
    })
}

/// Probability of `class` from the last `classify`.
#[unsafe(no_mangle)]
pub extern "C" fn probability(class: u32) -> f32 {
    with(|s| s.probabilities.get(class as usize).copied().unwrap_or(0.0))
}
