//! Step 2 on the Pico 2, with an LSM6DSOX IMU.
//!
//! Wiring: GP4 SDA, GP5 SCL (I2C address 0x6A), 3V3, GND;
//! a button from GP15 to GND. The LED is GPIO 25 (Pico 2, not 2 W).
//!
//! Press the button and perform the gesture for a second; five times for
//! each of four gestures. Training starts by itself, then it classifies
//! continuously.

#![no_std]
#![no_main]

use defmt::{info, warn};
use embassy_executor::Spawner;
use embassy_rp::bind_interrupts;
use embassy_rp::gpio::{Input, Level, Output, Pull};
use embassy_rp::i2c::{self, I2c, InterruptHandler};
use embassy_rp::peripherals::I2C0;
use embassy_time::{Duration, Instant, Ticker, Timer};
use embedded_hal_async::i2c::I2c as _;
use gesture::{Learner, Sample, WINDOW};
use pico2::{HEAP, init_heap, program_info};

program_info!(c"burn-gesture", c"Step 2: training on the device, Burn without std");

bind_interrupts!(struct Irqs {
    I2C0_IRQ => InterruptHandler<I2C0>;
});

const GESTURES: [&str; 4] = ["rest", "shake", "circle", "punch"];
const EXAMPLES: usize = 5;
const EPOCHS: usize = 15;
const PERIOD: Duration = Duration::from_millis(20); // 50 Hz

/// LSM6DSOX registers, from the datasheet.
mod lsm6dsox {
    pub const ADDRESS: u8 = 0x6A;
    pub const WHO_AM_I: u8 = 0x0F;
    pub const CTRL1_XL: u8 = 0x10;
    pub const CTRL2_G: u8 = 0x11;
    pub const CTRL3_C: u8 = 0x12;
    pub const OUTX_L_G: u8 = 0x22;
    pub const G_PER_LSB: f32 = 0.122e-3; // ±4 g
    pub const DPS_PER_LSB: f32 = 8.75e-3; // ±250 °/s
}

struct Imu<'d>(I2c<'d, I2C0, i2c::Async>);

impl Imu<'_> {
    async fn init(&mut self) -> Result<u8, i2c::Error> {
        use lsm6dsox::*;
        let mut id = [0];
        self.0.write_read(ADDRESS, &[WHO_AM_I], &mut id).await?;
        self.0.write(ADDRESS, &[CTRL3_C, 0x44]).await?; // consistent reads, auto-increment
        self.0.write(ADDRESS, &[CTRL1_XL, 0x38]).await?; // 52 Hz, ±4 g
        self.0.write(ADDRESS, &[CTRL2_G, 0x30]).await?; // 52 Hz, ±250 °/s
        Ok(id[0])
    }

    async fn read(&mut self) -> Result<Sample, i2c::Error> {
        use lsm6dsox::*;
        let mut b = [0; 12];
        self.0.write_read(ADDRESS, &[OUTX_L_G], &mut b).await?;
        let raw = |i: usize| i16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as f32;
        // The chip sends gyroscope first; Sample wants acceleration first.
        Ok([3, 4, 5, 0, 1, 2].map(|i| raw(i) * if i >= 3 { G_PER_LSB } else { DPS_PER_LSB }))
    }
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    init_heap();
    let p = embassy_rp::init(Default::default());
    let mut config = i2c::Config::default();
    config.frequency = 400_000;
    let mut imu = Imu(I2c::new_async(p.I2C0, p.PIN_5, p.PIN_4, Irqs, config));
    let mut led = Output::new(p.PIN_25, Level::Low);
    let mut button = Input::new(p.PIN_15, Pull::Up);

    match imu.init().await {
        Ok(id) => info!("IMU WHO_AM_I {=u8:#x} (LSM6DSOX: 0x6c)", id),
        Err(e) => warn!("no IMU: {}", e),
    }

    let mut learner = Learner::new(42);
    for (class, name) in GESTURES.iter().enumerate() {
        for n in 1..=EXAMPLES {
            info!("press the button, then: {} ({}/{})", name, n, EXAMPLES);
            button.wait_for_falling_edge().await;
            Timer::after_millis(200).await;
            led.set_high();
            let mut window = [[0.0; 6]; WINDOW];
            let mut ticker = Ticker::every(PERIOD);
            for sample in &mut window {
                ticker.next().await;
                *sample = imu.read().await.unwrap_or_default();
            }
            led.set_low();
            learner.add_example(&window, class);
        }
    }

    led.set_high();
    let t = Instant::now();
    let loss = learner.train(EPOCHS);
    led.set_low();
    info!("trained in {} ms, loss {}, heap {} bytes", t.elapsed().as_millis(), loss, HEAP.used());

    let mut window = [[0.0; 6]; WINDOW];
    let mut ticker = Ticker::every(PERIOD);
    for n in 1.. {
        ticker.next().await;
        window.rotate_left(1);
        window[WINDOW - 1] = imu.read().await.unwrap_or_default();
        if n % 12 == 0 && n >= WINDOW {
            let t = Instant::now();
            let p = learner.classify(&window);
            info!("{} ({}), {} µs", GESTURES[p.class], p.confidence, t.elapsed().as_micros());
            if p.confidence > 0.8 && p.class > 0 {
                for _ in 0..=p.class {
                    led.set_high();
                    Timer::after_millis(15).await;
                    led.set_low();
                    Timer::after_millis(15).await;
                }
            }
        }
    }
}
