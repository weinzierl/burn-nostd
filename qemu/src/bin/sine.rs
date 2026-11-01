#![no_std]
#![no_main]

use burn::tensor::Device;
use qemu::{HEAP, entry, exit, init_heap, println};
use sine::{Sine, grid, reference};

#[entry]
fn main() -> ! {
    init_heap();
    let device = Device::default();
    let model = Sine::pretrained(&device);
    let error = grid(101)
        .map(|x| libm::fabsf(model.predict_one(x, &device) - reference(x)))
        .fold(0.0, f32::max);
    println!("sine: max error {}, heap {} bytes, heap peak {} bytes", error, HEAP.used(), HEAP.peak());
    exit(error < 0.05)
}
