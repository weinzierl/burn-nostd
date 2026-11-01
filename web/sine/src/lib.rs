//! Step 1 in the browser: inference with the embedded weights, and the
//! training that produced them, replayed live.

#![no_std]

use adam::Adam;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::{Device, Tensor};
use rt::Global;
use sine::{Sine, reference};

const POINTS: usize = 256;
const EPOCHS: u32 = 2000;

struct Training {
    model: Sine,
    adam: Adam,
    x: Tensor<2>,
    y: Tensor<2>,
    epoch: u32,
}

struct State {
    device: Device,
    /// What `predict` uses: the embedded weights, or the training snapshot.
    current: Sine,
    training: Option<Training>,
}

static STATE: Global<State> = Global::new();

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(
        || {
            let device = Device::default();
            State { current: Sine::pretrained(&device), device, training: None }
        },
        f,
    )
}

/// Loads the model; returns the size of the embedded weights.
#[unsafe(no_mangle)]
pub extern "C" fn init() -> u32 {
    with(|_| sine::WEIGHTS.len() as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn predict(x: f32) -> f32 {
    with(|s| s.current.predict_one(x, &s.device))
}

/// Back to the embedded weights.
#[unsafe(no_mangle)]
pub extern "C" fn pretrained() {
    with(|s| {
        s.training = None;
        s.current = Sine::pretrained(&s.device);
    })
}

/// A fresh, random model, trained by `train`.
#[unsafe(no_mangle)]
pub extern "C" fn train_start(seed: u32) {
    with(|s| {
        let device = s.device.clone().autodiff();
        device.seed(seed as u64);
        let xs: [f32; POINTS] = core::array::from_fn(|i| i as f32 * sine::X_MAX / (POINTS - 1) as f32);
        let ys = xs.map(reference);
        let model = Sine::new(&device);
        s.current = model.valid();
        s.training = Some(Training {
            model,
            adam: Adam::default(),
            x: Tensor::<1>::from_floats(xs.as_slice(), &device).reshape([POINTS, 1]),
            y: Tensor::<1>::from_floats(ys.as_slice(), &device).reshape([POINTS, 1]),
            epoch: 0,
        });
    })
}

/// Runs up to `epochs` epochs; returns the mean squared error.
#[unsafe(no_mangle)]
pub extern "C" fn train(epochs: u32) -> f32 {
    with(|s| {
        let Some(t) = s.training.as_mut() else { return f32::NAN };
        let mut loss = f32::NAN;
        for _ in 0..epochs.min(EPOCHS - t.epoch) {
            let lr = if t.epoch < EPOCHS / 2 { 1e-2 } else { 1e-3 };
            let mse = MseLoss::new().forward(t.model.forward(t.x.clone()), t.y.clone(), Reduction::Mean);
            loss = mse.clone().into_scalar::<f32>();
            t.model = t.adam.step(lr, t.model.clone(), mse.backward());
            t.epoch += 1;
        }
        s.current = t.model.valid();
        loss
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn epoch() -> u32 {
    with(|s| s.training.as_ref().map_or(0, |t| t.epoch))
}
