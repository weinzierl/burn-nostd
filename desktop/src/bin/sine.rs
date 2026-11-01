//! Step 1 on the desktop: inference with the embedded weights.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use burn::tensor::Device;
use core::ffi::{c_char, c_int};
use desktop::{micros, println};
use sine::{Sine, WEIGHTS, grid, reference};

#[unsafe(no_mangle)]
extern "C" fn main(_: c_int, _: *const *const c_char) -> c_int {
    let device = Device::default();
    let model = Sine::pretrained(&device);
    println!("{} bytes of weights, embedded\n", WEIGHTS.len());
    println!("     x    sin(x)  model    -1 {:>17} +1", "0");

    let xs: Vec<f32> = grid(25).collect();
    let t = micros();
    let ys: Vec<f32> = xs.iter().map(|&x| model.predict_one(x, &device)).collect();
    let per_inference = (micros() - t) / 25;

    let mut error = 0.0f32;
    for (&x, &y) in xs.iter().zip(&ys) {
        let r = reference(x);
        error = error.max(libm::fabsf(y - r));
        let column = |v: f32| ((v + 1.0) * 20.0).clamp(0.0, 40.0) as usize;
        let mut plot = [b' '; 41];
        plot[20] = b'|';
        plot[column(r)] = b'.';
        plot[column(y)] = b'o';
        println!("{x:6.2} {r:7.3} {y:7.3}   {}", core::str::from_utf8(&plot).unwrap());
    }
    println!("\no model   . sin(x)   max error {error:.4}   {per_inference} µs per inference");
    if error < 0.05 { 0 } else { 1 }
}
