//! Step 1 on the Pico 2: the on-board LED (GPIO 25) breathes with the
//! model's output, one forward pass per brightness value.

#![no_std]
#![no_main]

use burn::tensor::Device;
use defmt::info;
use embassy_executor::Spawner;
use embassy_rp::pwm::{Config, Pwm, SetDutyCycle};
use embassy_time::{Instant, Timer};
use pico2::{HEAP, init_heap, program_info};
use sine::{Sine, X_MAX, reference};

program_info!(c"burn-sine", c"Step 1: inference, Burn without std");

const STEPS: u32 = 100;

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    init_heap();
    let p = embassy_rp::init(Default::default());
    let mut config = Config::default();
    config.top = u16::MAX;
    let mut led = Pwm::new_output_b(p.PWM_SLICE4, p.PIN_25, config);

    let device = Device::default();
    let model = Sine::pretrained(&device);
    info!("{} bytes of weights in flash, heap {} bytes", sine::WEIGHTS.len(), HEAP.used());

    loop {
        let (mut error, mut micros) = (0.0f32, 0);
        for i in 0..STEPS {
            let x = X_MAX * i as f32 / STEPS as f32;
            let t = Instant::now();
            let y = model.predict_one(x, &device);
            micros += t.elapsed().as_micros();
            error = error.max(libm::fabsf(y - reference(x)));
            // Squared, because perceived brightness is not linear.
            let b = ((y + 1.0) / 2.0).clamp(0.0, 1.0);
            let _ = led.set_duty_cycle((b * b * u16::MAX as f32) as u16);
            Timer::after_millis(20).await;
        }
        info!("{} µs per inference, max error {}", micros / STEPS as u64, error);
    }
}
