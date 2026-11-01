//! Step 2: training on the device. The network starts untrained; you show it
//! each gesture a few times and it learns with autodiff, on the device.

#![no_std]

extern crate alloc;

pub mod synth;

use adam::Adam;
use alloc::vec::Vec;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::activation::softmax;
use burn::tensor::{Device, Int, Tensor};

/// Accelerometer x, y, z in g, then gyroscope x, y, z in deg/s.
pub type Sample = [f32; 6];

/// One second at 50 Hz.
pub const WINDOW: usize = 50;
pub const MAX_CLASSES: usize = 4;
const STATS: usize = 6;
const INPUT: usize = 6 * STATS;
const HIDDEN: usize = 32;

/// Per axis: mean, standard deviation, min, max, mean absolute change and
/// mean-crossing rate. Raw samples overfit with five examples: the same
/// gesture never starts at the same sample twice.
pub fn features(window: &[Sample; WINDOW]) -> [f32; INPUT] {
    let mut out = [0.0; INPUT];
    for axis in 0..6 {
        let scale = if axis < 3 { 1.0 } else { 1.0 / 250.0 };
        let x: [f32; WINDOW] = core::array::from_fn(|t| window[t][axis] * scale);
        let n = WINDOW as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let lo = x.iter().copied().fold(f32::MAX, f32::min);
        let hi = x.iter().copied().fold(f32::MIN, f32::max);
        let change = x.windows(2).map(|p| libm::fabsf(p[1] - p[0])).sum::<f32>() / n;
        let crossings = x.windows(2).filter(|p| (p[0] - mean) * (p[1] - mean) < 0.0).count() as f32 / n;
        out[axis * STATS..][..STATS].copy_from_slice(&[mean, libm::sqrtf(var), lo, hi, change * 10.0, crossings * 4.0]);
    }
    out
}

#[derive(Module, Debug)]
struct Net {
    l1: Linear,
    l2: Linear,
    act: Relu,
}

impl Net {
    fn new(device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(INPUT, HIDDEN).init(device),
            l2: LinearConfig::new(HIDDEN, MAX_CLASSES).init(device),
            act: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.l2.forward(self.act.forward(self.l1.forward(x)))
    }
}

pub struct Prediction {
    pub class: usize,
    pub confidence: f32,
}

pub struct Learner {
    device: Device,
    train_device: Device,
    net: Net,
    adam: Adam,
    examples: Vec<([f32; INPUT], i64)>,
    classes: usize,
    rng: u32,
    pub learning_rate: f32,
}

impl Learner {
    pub fn new(seed: u64) -> Self {
        let device = Device::default();
        let train_device = device.clone().autodiff();
        train_device.seed(seed);
        Self {
            net: Net::new(&train_device),
            device,
            train_device,
            adam: Adam::default(),
            examples: Vec::new(),
            classes: 0,
            rng: seed as u32 | 1,
            learning_rate: 3e-3,
        }
    }

    pub fn add_example(&mut self, window: &[Sample; WINDOW], class: usize) {
        assert!(class < MAX_CLASSES);
        self.examples.push((features(window), class as i64));
        self.classes = self.classes.max(class + 1);
    }

    pub fn examples(&self) -> usize {
        self.examples.len()
    }

    /// Shuffled SGD, one example per step. Returns the last epoch's mean loss.
    ///
    /// Batch size 1 is deliberate: without std, gemm allocates a 512 KiB
    /// packing buffer for every matmul with m, n > 1 and k > 2. One example
    /// per step keeps every matmul a vector product, which needs none.
    pub fn train(&mut self, epochs: usize) -> f32 {
        assert!(!self.examples.is_empty(), "add examples first");
        let d = self.train_device.clone();
        let loss_fn = CrossEntropyLoss::new(None, &d);
        let mut order: Vec<usize> = (0..self.examples.len()).collect();
        let mut mean = f32::NAN;
        for _ in 0..epochs {
            shuffle(&mut order, &mut self.rng);
            let mut sum = 0.0;
            for &i in &order {
                let (x, y) = &self.examples[i];
                let x = Tensor::<1>::from_floats(x.as_slice(), &d).reshape([1, INPUT]);
                let y = Tensor::<1, Int>::from_ints([*y], &d);
                let loss = loss_fn.forward(self.net.forward(x), y);
                sum += loss.clone().into_scalar::<f32>();
                self.net = self.adam.step(self.learning_rate, self.net.clone(), loss.backward());
            }
            mean = sum / order.len() as f32;
        }
        mean
    }

    /// Probabilities of the classes taught so far.
    pub fn probabilities(&self, window: &[Sample; WINDOW]) -> Vec<f32> {
        let x = Tensor::<1>::from_floats(features(window).as_slice(), &self.device).reshape([1, INPUT]);
        let mut p: Vec<f32> = softmax(self.net.valid().forward(x), 1).into_data().try_into_vec().expect("f32");
        // Only taught classes compete.
        p.truncate(self.classes.max(1));
        p
    }

    pub fn classify(&self, window: &[Sample; WINDOW]) -> Prediction {
        let p = self.probabilities(window);
        let (class, &confidence) = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).expect("a class");
        Prediction { class, confidence }
    }
}

/// Fisher-Yates with xorshift32: no rand crate needed.
fn shuffle(v: &mut [usize], state: &mut u32) {
    for i in (1..v.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        v.swap(i, *state as usize % (i + 1));
    }
}
