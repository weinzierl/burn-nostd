//! Trains step 1's model and rewrites 1-sine/weights/sine.bpk. No std
//! here either: autodiff, the Adam crate and Burnpack all work without it.

#![no_std]
#![no_main]

use adam::Adam;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::store::{BurnpackStore, ModuleSnapshot};
use burn::tensor::{Device, Tensor};
use core::ffi::{c_char, c_int};
use desktop::println;
use sine::{Sine, grid, reference};

extern crate alloc;
use alloc::vec::Vec;

const POINTS: usize = 512;
const EPOCHS: usize = 4000;
const OUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../1-sine/weights/sine.bpk");

#[unsafe(no_mangle)]
extern "C" fn main(_: c_int, _: *const *const c_char) -> c_int {
    let device = Device::default().autodiff();
    device.seed(42);
    let xs: Vec<f32> = grid(POINTS).collect();
    let ys: Vec<f32> = xs.iter().map(|&x| reference(x)).collect();
    let x = Tensor::<1>::from_floats(xs.as_slice(), &device).reshape([POINTS, 1]);
    let y = Tensor::<1>::from_floats(ys.as_slice(), &device).reshape([POINTS, 1]);

    let mut model = Sine::new(&device);
    let mut adam = Adam::default();
    // Full batch: the desktop can afford gemm's packing buffers.
    for epoch in 1..=EPOCHS {
        let lr = if epoch <= EPOCHS / 2 { 1e-2 } else { 1e-3 };
        let loss = MseLoss::new().forward(model.forward(x.clone()), y.clone(), Reduction::Mean);
        if epoch % 1000 == 0 {
            println!("epoch {epoch}: mse {:.6}", loss.clone().into_scalar::<f32>());
        }
        model = adam.step(lr, model, loss.backward());
    }

    let mut store = BurnpackStore::from_bytes(None);
    model.valid().save_into(&mut store).expect("serializes");
    let bytes = store.get_bytes().expect("written");
    if !desktop::write(OUT, &bytes) {
        println!("cannot write {OUT}");
        return 1;
    }
    println!("wrote {} bytes to {OUT}; `cargo sine` embeds them", bytes.len());
    0
}
