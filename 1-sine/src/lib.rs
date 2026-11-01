//! Step 1: inference. A 321-parameter MLP approximates sin(x) on [0, 2π].
//! Trained once (`cargo sine-train`), embedded as Burnpack, loaded zero-copy.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::store::{BurnpackStore, ModuleSnapshot};
use burn::tensor::{Device, Tensor};

pub const X_MAX: f32 = core::f32::consts::TAU;
const HIDDEN: usize = 16;

// Burnpack aligns tensor data to 256 bytes relative to the file start, while
// include_bytes! only guarantees 1.
#[repr(C, align(256))]
struct Aligned<T: ?Sized>(T);
static ALIGNED: &Aligned<[u8]> = &Aligned(*include_bytes!("../weights/sine.bpk"));

/// The trained weights. On the microcontroller they stay in flash.
pub static WEIGHTS: &[u8] = &ALIGNED.0;

#[derive(Module, Debug)]
pub struct Sine {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    act: Relu,
}

impl Sine {
    pub fn new(device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(1, HIDDEN).init(device),
            l2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            l3: LinearConfig::new(HIDDEN, 1).init(device),
            act: Relu::new(),
        }
    }

    pub fn pretrained(device: &Device) -> Self {
        let mut model = Self::new(device);
        model
            .load_from(&mut BurnpackStore::from_static(WEIGHTS))
            .expect("weights match the architecture");
        model
    }

    /// `[batch, 1]` radians to `[batch, 1]`.
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        // Centered inputs train faster; doing it here keeps every caller consistent.
        let x = x.div_scalar(core::f32::consts::PI).sub_scalar(1.0);
        let x = self.act.forward(self.l1.forward(x));
        let x = self.act.forward(self.l2.forward(x));
        self.l3.forward(x)
    }

    pub fn predict(&self, xs: &[f32], device: &Device) -> Vec<f32> {
        let x = Tensor::<1>::from_floats(xs, device).reshape([xs.len(), 1]);
        self.forward(x).into_data().try_into_vec().expect("f32 output")
    }

    pub fn predict_one(&self, x: f32, device: &Device) -> f32 {
        self.predict(&[x], device)[0]
    }
}

pub fn reference(x: f32) -> f32 {
    libm::sinf(x)
}

/// `n` evenly spaced points on [0, 2π].
pub fn grid(n: usize) -> impl Iterator<Item = f32> {
    let step = X_MAX / (n.max(2) - 1) as f32;
    (0..n).map(move |i| i as f32 * step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretrained_approximates_sine() {
        let device = Device::default();
        let model = Sine::pretrained(&device);
        let xs: Vec<f32> = grid(101).collect();
        let error = model
            .predict(&xs, &device)
            .iter()
            .zip(&xs)
            .map(|(y, &x)| libm::fabsf(y - reference(x)))
            .fold(0.0, f32::max);
        assert!(error < 0.05, "max error {error}");
    }
}
