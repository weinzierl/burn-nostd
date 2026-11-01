//! Synthetic IMU gestures: lets the learning loop run without a sensor
//! (tests, desktop, QEMU).

use crate::{Sample, WINDOW};
use core::f32::consts::TAU;

pub const NAMES: [&str; 4] = ["rest", "shake", "circle", "punch"];

/// xorshift32: deterministic jitter without a rand dependency.
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1 << 24) as f32
    }

    /// Uniform in [-1, 1).
    pub fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

/// One window of gesture `class` with random amplitude, speed, phase,
/// timing and sensor noise. Gravity points along z.
pub fn window(class: usize, rng: &mut Rng) -> [Sample; WINDOW] {
    let amp = 1.0 + 0.3 * rng.sym();
    let speed = 1.0 + 0.2 * rng.sym();
    let phase = rng.unit() * TAU;
    let onset = (rng.unit() * 20.0) as usize + 10;
    core::array::from_fn(|i| {
        let t = i as f32 / WINDOW as f32;
        let motion = match class {
            0 => [0.0; 6],
            1 => {
                let a = libm::sinf(TAU * 5.0 * speed * t + phase);
                [1.2 * amp * a, 0.0, 0.0, 180.0 * amp * a, 0.0, 0.0]
            }
            2 => {
                let p = TAU * 1.5 * speed * t + phase;
                [0.6 * amp * libm::cosf(p), 0.6 * amp * libm::sinf(p), 0.0, 0.0, 0.0, 120.0 * amp]
            }
            _ => {
                let y = match i.wrapping_sub(onset) {
                    0..4 => 2.5 * amp,
                    4..8 => -1.5 * amp,
                    _ => 0.0,
                };
                [0.0, y, 0.0, 0.0, 0.0, 0.0]
            }
        };
        core::array::from_fn(|axis| {
            let noise = if axis < 3 { 0.05 } else { 5.0 } * rng.sym();
            let gravity = if axis == 2 { 1.0 } else { 0.0 };
            motion[axis] + noise + gravity
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Learner;

    #[test]
    fn learns_from_five_examples_each() {
        let mut rng = Rng::new(7);
        let mut learner = Learner::new(42);
        for class in 0..4 {
            for _ in 0..5 {
                learner.add_example(&window(class, &mut rng), class);
            }
        }
        learner.train(15);
        let correct = (0..100)
            .filter(|i| learner.classify(&window(i % 4, &mut rng)).class == i % 4)
            .count();
        assert!(correct >= 90, "{correct}/100");
    }
}
