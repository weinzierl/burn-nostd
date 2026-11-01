//! Step 4 on the Waveshare RP2350-Touch-LCD-3.5: TinyStories 260K from flash,
//! the stories on the board's screen from power-up, no computer needed. The
//! prompt is gold, the model's words cream; after each story the footer shows
//! the model's speed (time spent drawing and on USB not counted).
//!
//! The same text also goes to the USB port as a serial device, as in
//! `tinystories-usb`: while a terminal has the port open it gets every story,
//! and a newly opened terminal makes the next story the first one again.
//! USB runs in its own executor in an interrupt (SWI_IRQ_0), above the model
//! in thread mode. Built for the RP2350A; the board has an RP2350B, which
//! differs only in pin count, and the pins used here exist on both.

#![no_std]
#![no_main]

extern crate alloc;

mod screen;
mod st7796;

use alloc::format;
use burn::tensor::Device;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use embassy_executor::{InterruptExecutor, Spawner};
use embassy_futures::select::{Either, select};
use embassy_rp::gpio::{Level, Output};
use embassy_rp::interrupt;
use embassy_rp::interrupt::{InterruptExt, Priority};
use embassy_rp::peripherals::USB;
use embassy_rp::spi::{self, Spi};
use embassy_rp::usb::{Driver, InterruptHandler};
use embassy_rp::{Peri, bind_interrupts};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::pipe::Pipe;
use embassy_time::{Instant, Timer};
use embassy_usb::class::cdc_acm::{CdcAcmClass, ControlChanged, Receiver, Sender, State};
use embassy_usb::{Builder, Config, UsbDevice};
use pico2::{HEAP, init_heap, program_info};
use screen::Screen;
use st7796::St7796;
use static_cell::StaticCell;
use tinystories::{Llama, Sampler, Tokenizer, generate, stories260k};

program_info!(c"burn-tinystories-lcd", c"Step 4: llama2.c TinyStories 260K on Burn, stories on the screen and over USB serial");

bind_interrupts!(struct Irqs {
    USBCTRL_IRQ => InterruptHandler<USB>;
});

const PROMPTS: [&str; 4] = ["Once upon a time", "Lily and Ben", "The little dog", "One day, a robot"];
/// 128 positions keep the KV cache at 160 KiB.
const CONTEXT: usize = 128;
const TITLE: &str = "TinyStories 260K on Burn";

type UsbDriver = Driver<'static, USB>;

/// Text on its way from the model to the terminal.
static TEXT: Pipe<CriticalSectionRawMutex, 4096> = Pipe::new();
/// A terminal has the port open (it set DTR).
static OPEN: AtomicBool = AtomicBool::new(false);
/// Counts terminal sessions, so that each new one starts from the first story.
static SESSIONS: AtomicU32 = AtomicU32::new(0);

static USB_EXECUTOR: InterruptExecutor = InterruptExecutor::new();

#[interrupt]
unsafe fn SWI_IRQ_0() {
    unsafe { USB_EXECUTOR.on_interrupt() }
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    init_heap();
    let p = embassy_rp::init(Default::default());
    interrupt::SWI_IRQ_0.set_priority(Priority::P3);
    USB_EXECUTOR.start(interrupt::SWI_IRQ_0).spawn(start_usb(p.USB).unwrap());

    // Waveshare's pins: SPI0 clock 18, data 19; DC 20, CS 21, backlight 22, reset 23.
    let mut config = spi::Config::default();
    config.frequency = 40_000_000;
    let spi = Spi::new_blocking_txonly(p.SPI0, p.PIN_18, p.PIN_19, config);
    let (dc, cs) = (Output::new(p.PIN_20, Level::Low), Output::new(p.PIN_21, Level::High));
    let (reset, backlight) = (Output::new(p.PIN_23, Level::High), Output::new(p.PIN_22, Level::Low));
    let mut screen = Screen::new(St7796::new(spi, dc, cs, reset, backlight).await);
    screen.target.backlight(true);
    screen.header(TITLE, "loading");

    let device = Device::default();
    let mut model = Llama::load(stories260k::WEIGHTS, &device, CONTEXT);
    let tokenizer = Tokenizer::new(stories260k::TOKENIZER, model.config.vocab_size);
    let loaded = HEAP.used();

    let (mut session, mut round) = (0, 0);
    loop {
        let current = SESSIONS.load(Ordering::Acquire);
        if current != session {
            (session, round) = (current, 0);
            let weights = stories260k::WEIGHTS.len();
            say(format!("burn-tinystories-lcd\n{weights} bytes of weights in flash, heap {loaded} bytes\n\n").as_bytes());
        }
        // The first story is greedy, so it can be compared with llama2.c.
        let temperature = if round == 0 { 0.0 } else { 0.8 };
        let mut sampler = Sampler::new(temperature, round as u32 + 1);
        let prompt = PROMPTS[round % PROMPTS.len()];
        let story = format!("story {}, {}", round + 1, if round == 0 { "greedy" } else { "temp 0.8" });
        screen.header(TITLE, &story);
        screen.footer("writing", "");
        screen.new_story();

        // The prompt comes back as the first pieces: one per prompt token after BOS.
        let prompt_pieces = tokenizer.encode(prompt).len() - 1;
        let (mut pieces, mut output_us) = (0, 0);
        let t = Instant::now();
        let tokens = generate(&mut model, &tokenizer, &mut sampler, prompt, CONTEXT, |piece| {
            let start = Instant::now();
            screen.push(piece, pieces < prompt_pieces);
            say(piece);
            pieces += 1;
            output_us += start.elapsed().as_micros();
        });
        let ms = t.elapsed().as_millis().max(1);
        let model_ms = ms.saturating_sub(output_us / 1000).max(1);
        let tenths = tokens as u64 * 10_000 / model_ms; // tokens per second, times 10
        screen.footer(&format!("{tokens} tokens, {}.{} tokens/s", tenths / 10, tenths % 10), "RP2350, 150 MHz");
        let heap = HEAP.used();
        say(format!("\n{tokens} tokens in {ms} ms, {model_ms} ms of it in the model: {}.{} tokens/s, heap {heap} bytes\n\n", tenths / 10, tenths % 10).as_bytes());
        round += 1;
        Timer::after_secs(5).await;
    }
}

/// Queues text for the terminal, with \r\n line ends. Without a terminal it is
/// dropped, and so is what does not fit: then the terminal has stopped reading.
fn say(text: &[u8]) {
    if !OPEN.load(Ordering::Acquire) {
        return;
    }
    for (i, line) in text.split(|&b| b == b'\n').enumerate() {
        if i > 0 {
            send(b"\r\n");
        }
        send(line);
    }
}

fn send(mut bytes: &[u8]) {
    // A write can stop at the end of the ring buffer, so loop.
    while !bytes.is_empty() {
        match TEXT.try_write(bytes) {
            Ok(n) => bytes = &bytes[n..],
            Err(_) => return,
        }
    }
}

/// Builds the USB device inside its executor: the CDC-ACM class is not `Send`,
/// so it cannot be built in thread mode and handed over.
#[embassy_executor::task]
async fn start_usb(usb: Peri<'static, USB>) {
    // Safe here: this runs as a task of the interrupt executor.
    let spawner = unsafe { Spawner::for_current_executor() }.await;

    let mut config = Config::new(0xc0de, 0xcafe); // test IDs, as in the embassy examples
    config.manufacturer = Some("burn-nostd");
    config.product = Some("TinyStories 260K on Burn");
    config.serial_number = Some("rustlab");

    static BUFFERS: StaticCell<([u8; 256], [u8; 256], [u8; 64])> = StaticCell::new();
    static STATE: StaticCell<State<'static>> = StaticCell::new();
    let (config_descriptor, bos_descriptor, control) = BUFFERS.init(([0; 256], [0; 256], [0; 64]));
    let mut builder = Builder::new(Driver::new(usb, Irqs), config, config_descriptor, bos_descriptor, &mut [], control);
    let class = CdcAcmClass::new(&mut builder, STATE.init(State::new()), 64);
    let (tx, rx, control) = class.split_with_control();

    spawner.spawn(run_usb(builder.build()).unwrap());
    spawner.spawn(text_out(tx, control).unwrap());
    spawner.spawn(input_dropped(rx).unwrap());
}

#[embassy_executor::task]
async fn run_usb(mut device: UsbDevice<'static, UsbDriver>) -> ! {
    device.run().await
}

/// Moves text from the pipe to the terminal, while one has the port open.
#[embassy_executor::task]
async fn text_out(mut tx: Sender<'static, UsbDriver>, control: ControlChanged<'static>) -> ! {
    // 63 bytes: a full 64-byte packet waits at the host until a shorter one follows.
    let mut packet = [0; 63];
    loop {
        tx.wait_connection().await;
        while !control.dtr() {
            control.control_changed().await;
        }
        TEXT.clear();
        SESSIONS.fetch_add(1, Ordering::Release);
        OPEN.store(true, Ordering::Release);
        loop {
            let n = match select(TEXT.read(&mut packet), control.control_changed()).await {
                Either::First(n) => n,
                Either::Second(()) if control.dtr() => continue,
                Either::Second(()) => break,
            };
            // A terminal that closes now never reads this packet.
            match select(tx.write_packet(&packet[..n]), closed(&control)).await {
                Either::First(Ok(())) => {}
                _ => break,
            }
        }
        OPEN.store(false, Ordering::Release);
    }
}

async fn closed(control: &ControlChanged<'_>) {
    while control.dtr() {
        control.control_changed().await;
    }
}

/// Reads and drops what the terminal sends, so that its writes never hang.
#[embassy_executor::task]
async fn input_dropped(mut rx: Receiver<'static, UsbDriver>) -> ! {
    let mut packet = [0; 64];
    loop {
        rx.wait_connection().await;
        while rx.read_packet(&mut packet).await.is_ok() {}
    }
}
