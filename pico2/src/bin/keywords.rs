//! Step 3 on the Pico 2, with an SPH0645 I2S microphone.
//!
//! Wiring: BCLK GP18, LRCL GP19, DOUT GP20, SEL to GND (left channel), 3V, GND;
//! a button from GP15 to GND. The LED is GPIO 25 (Pico 2, not 2 W).
//!
//! Press the button and say the word within 1.5 s; five times for each of
//! four words. Training starts by itself, then it listens continuously.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use defmt::info;
use embassy_executor::Spawner;
use embassy_rp::gpio::{Input, Level, Output, Pull};
use embassy_rp::peripherals::{DMA_CH0, PIO0};
use embassy_rp::pio::program::pio_asm;
use embassy_rp::pio::{Config, Direction, FifoJoin, InterruptHandler, Pio, ShiftConfig, ShiftDirection};
use embassy_rp::pio_programs::clock_divider::calculate_pio_clock_divider;
use embassy_rp::{bind_interrupts, dma};
use embassy_time::{Instant, Timer};
use keywords::frontend::{MelFrame, SAMPLE_RATE};
use keywords::{Learner, Segmenter};
use pico2::{HEAP, init_heap, program_info};
use static_cell::StaticCell;

program_info!(c"burn-keywords", c"Step 3: keyword spotting learnt on the device");

bind_interrupts!(struct Irqs {
    PIO0_IRQ_0 => InterruptHandler<PIO0>;
    DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>;
});

const WORDS: [&str; 4] = ["yes", "no", "left", "right"];
const EXAMPLES: usize = 5;
const EPOCHS: usize = 15;
/// 100 ms per DMA transfer: the gap between transfers loses a few samples,
/// and fewer gaps mean fewer clicks.
const CHUNK: usize = 1600;
const RECORDING: usize = 15; // chunks, 1.5 s

/// Takes the left 32-bit slot, keeps the SPH0645's 18 significant bits and
/// removes its DC offset.
struct Converter {
    x: f32,
    y: f32,
}

impl Converter {
    fn convert(&mut self, raw: &[u32], out: &mut [f32]) {
        for (o, frame) in out.iter_mut().zip(raw.chunks_exact(2)) {
            let x = (frame[0] as i32 >> 14) as f32 / (1 << 17) as f32;
            self.y = x - self.x + 0.995 * self.y;
            self.x = x;
            *o = self.y;
        }
    }
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    init_heap();
    let p = embassy_rp::init(Default::default());
    let mut led = Output::new(p.PIN_25, Level::Low);
    let mut button = Input::new(p.PIN_15, Pull::Up);

    // embassy's PioI2sIn reads 16-bit slots; the SPH0645 needs 64 clocks per
    // sample, so this is the same program with 32-bit slots.
    let Pio { mut common, mut sm0, .. } = Pio::new(p.PIO0, Irqs);
    let program = pio_asm!(
        ".side_set 2",                   // bit 0: BCLK, bit 1: LRCL
        "   set x, 30          side 0b01",
        "left:",
        "   in pins, 1         side 0b00",
        "   jmp x-- left       side 0b01",
        "   in pins, 1         side 0b10", // LRCL flips one clock before the MSB
        "   set x, 30          side 0b11",
        "right:",
        "   in pins, 1         side 0b10",
        "   jmp x-- right      side 0b11",
        "   in pins, 1         side 0b00",
    );
    let program = common.load_program(&program.program);
    let data = common.make_pio_pin(p.PIN_20);
    let bclk = common.make_pio_pin(p.PIN_18);
    let lrcl = common.make_pio_pin(p.PIN_19);
    let mut config = Config::default();
    config.use_program(&program, &[&bclk, &lrcl]);
    config.set_in_pins(&[&data]);
    config.clock_divider = calculate_pio_clock_divider(SAMPLE_RATE as u32 * 64 * 2); // 2 cycles per bit
    config.shift_in = ShiftConfig { threshold: 32, direction: ShiftDirection::Left, auto_fill: true };
    config.fifo_join = FifoJoin::RxOnly;
    sm0.set_config(&config);
    sm0.set_pin_dirs(Direction::In, &[&data]);
    sm0.set_pin_dirs(Direction::Out, &[&lrcl, &bclk]);
    let mut dma = dma::Channel::new(p.DMA_CH0, Irqs);
    sm0.set_enable(true);

    static RAW: StaticCell<[u32; CHUNK * 2]> = StaticCell::new();
    let raw = RAW.init([0; CHUNK * 2]);
    let mut audio = [0.0; CHUNK];
    let mut converter = Converter { x: 0.0, y: 0.0 };
    // The microphone needs about 50 ms to settle after power-up.
    for _ in 0..5 {
        sm0.rx().dma_pull(&mut dma, raw, false).await;
        converter.convert(raw, &mut audio);
    }

    let mut learner = Learner::new(42);
    for (word, name) in WORDS.iter().enumerate() {
        let mut taken = 0;
        while taken < EXAMPLES {
            info!("press the button, then say: {} ({}/{})", name, taken + 1, EXAMPLES);
            button.wait_for_falling_edge().await;
            led.set_high();
            let mut segmenter = Segmenter::new();
            let mut longest: Option<Vec<MelFrame>> = None;
            let mut keep = |w: &[MelFrame]| {
                if longest.as_ref().is_none_or(|l| w.len() > l.len()) {
                    longest = Some(w.to_vec());
                }
            };
            for _ in 0..RECORDING {
                sm0.rx().dma_pull(&mut dma, raw, false).await;
                converter.convert(raw, &mut audio);
                segmenter.push(&audio, &mut keep);
            }
            segmenter.flush(&mut keep);
            led.set_low();
            match longest {
                Some(frames) => {
                    info!("got {} ms", frames.len() * 20);
                    learner.add_example(&frames, word);
                    taken += 1;
                }
                None => info!("no word heard: closer or louder, please"),
            }
            Timer::after_millis(300).await;
        }
    }

    led.set_high();
    let t = Instant::now();
    let loss = learner.train(EPOCHS);
    led.set_low();
    info!("trained in {} ms, loss {}, heap {} bytes", t.elapsed().as_millis(), loss, HEAP.used());

    let mut segmenter = Segmenter::new();
    loop {
        sm0.rx().dma_pull(&mut dma, raw, false).await;
        converter.convert(raw, &mut audio);
        let mut heard = None;
        segmenter.push(&audio, |frames| {
            let t = Instant::now();
            let p = learner.classify(frames);
            info!("heard {} ({}), {} µs", WORDS[p.word], p.confidence, t.elapsed().as_micros());
            heard = (p.confidence > 0.8).then_some(p.word);
        });
        for _ in 0..heard.map_or(0, |w| w + 1) {
            led.set_high();
            Timer::after_millis(15).await;
            led.set_low();
            Timer::after_millis(15).await;
        }
    }
}
