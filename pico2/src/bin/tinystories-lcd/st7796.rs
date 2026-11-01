//! The Waveshare RP2350-Touch-LCD-3.5's panel: an ST7796 on SPI0, used in
//! landscape, 480 x 320, RGB565. Pins, reset timing and register values are
//! Waveshare's (examples/C/01_GUI: `lib/Config/DEV_Config.h` and
//! `lib/LCD/LCD_3in5.c` in waveshareteam/RP2350-Touch-LCD-3.5, Apache-2.0).

use core::convert::Infallible;
use embassy_rp::gpio::Output;
use embassy_rp::peripherals::SPI0;
use embassy_rp::spi::{Blocking, Spi};
use embassy_time::Timer;
use embedded_graphics::pixelcolor::{Rgb565, raw::RawU16};
use embedded_graphics::prelude::*;
use embedded_graphics::primitives::Rectangle;

use crate::screen::{HEIGHT, WIDTH};

/// After reset, sleep out and MADCTL: Waveshare's `LCD_3IN5_InitReg`, in order.
const INIT: [(u8, &[u8]); 14] = [
    (0x3A, &[0x05]), // 16 bits per pixel
    (0xF0, &[0xC3]), // unlock the extended commands
    (0xF0, &[0x96]),
    (0xB4, &[0x01]),
    (0xB7, &[0xC6]),
    (0xC0, &[0x80, 0x45]),
    (0xC1, &[0x13]),
    (0xC2, &[0xA7]),
    (0xC5, &[0x0A]),
    (0xE8, &[0x40, 0x8A, 0x00, 0x00, 0x29, 0x19, 0xA5, 0x33]),
    (0xE0, &[0xD0, 0x08, 0x0F, 0x06, 0x06, 0x33, 0x30, 0x33, 0x47, 0x17, 0x13, 0x13, 0x2B, 0x31]),
    (0xE1, &[0xD0, 0x0A, 0x11, 0x0B, 0x09, 0x07, 0x2F, 0x33, 0x47, 0x38, 0x15, 0x16, 0x2C, 0x32]),
    (0xF0, &[0x3C]), // lock again
    (0xF0, &[0x69]),
];

pub struct St7796<'d> {
    spi: Spi<'d, SPI0, Blocking>,
    dc: Output<'d>,
    cs: Output<'d>,
    // Kept so the pins keep their level.
    _reset: Output<'d>,
    backlight: Output<'d>,
}

impl<'d> St7796<'d> {
    /// Resets and initializes the panel; the backlight stays off.
    pub async fn new(spi: Spi<'d, SPI0, Blocking>, dc: Output<'d>, cs: Output<'d>, mut reset: Output<'d>, backlight: Output<'d>) -> Self {
        for (high, ms) in [(true, 100), (false, 100), (true, 100)] {
            reset.set_level(high.into());
            Timer::after_millis(ms).await;
        }
        let mut panel = Self { spi, dc, cs, _reset: reset, backlight };
        panel.command(0x11, &[]); // sleep out
        Timer::after_millis(120).await;
        panel.command(0x36, &[0x28]); // MADCTL: Waveshare's landscape ("HORIZONTAL") setting
        for (command, data) in INIT {
            panel.command(command, data);
        }
        Timer::after_millis(120).await;
        panel.command(0x21, &[]); // inversion on
        panel.command(0x29, &[]); // display on
        panel
    }

    pub fn backlight(&mut self, on: bool) {
        self.backlight.set_level(on.into());
    }

    fn command(&mut self, command: u8, data: &[u8]) {
        self.cs.set_low();
        self.dc.set_low();
        let _ = self.spi.blocking_write(&[command]);
        self.dc.set_high();
        let _ = self.spi.blocking_write(data);
        self.cs.set_high();
    }

    /// Sends the pixels of a rectangle, row by row (it must lie on the screen).
    fn fill(&mut self, area: &Rectangle, colors: impl IntoIterator<Item = Rgb565>) {
        let (x0, y0) = (area.top_left.x as u16, area.top_left.y as u16);
        let (x1, y1) = (x0 + area.size.width as u16 - 1, y0 + area.size.height as u16 - 1);
        self.command(0x2A, &[(x0 >> 8) as u8, x0 as u8, (x1 >> 8) as u8, x1 as u8]);
        self.command(0x2B, &[(y0 >> 8) as u8, y0 as u8, (y1 >> 8) as u8, y1 as u8]);
        self.cs.set_low();
        self.dc.set_low();
        let _ = self.spi.blocking_write(&[0x2C]); // memory write
        self.dc.set_high();
        let mut buffer = [0; 128];
        let mut n = 0;
        for color in colors {
            buffer[n..n + 2].copy_from_slice(&RawU16::from(color).into_inner().to_be_bytes());
            n += 2;
            if n == buffer.len() {
                let _ = self.spi.blocking_write(&buffer);
                n = 0;
            }
        }
        let _ = self.spi.blocking_write(&buffer[..n]);
        self.cs.set_high();
    }
}

impl OriginDimensions for St7796<'_> {
    fn size(&self) -> Size {
        Size::new(WIDTH, HEIGHT)
    }
}

impl DrawTarget for St7796<'_> {
    type Color = Rgb565;
    type Error = Infallible;

    fn draw_iter<I: IntoIterator<Item = Pixel<Rgb565>>>(&mut self, pixels: I) -> Result<(), Infallible> {
        for Pixel(point, color) in pixels {
            if self.bounding_box().contains(point) {
                self.fill(&Rectangle::new(point, Size::new(1, 1)), [color]);
            }
        }
        Ok(())
    }

    fn fill_contiguous<I: IntoIterator<Item = Rgb565>>(&mut self, area: &Rectangle, colors: I) -> Result<(), Infallible> {
        if area.is_zero_sized() {
            return Ok(());
        }
        if self.bounding_box().intersection(area) == *area {
            self.fill(area, colors.into_iter().take(area.size.width as usize * area.size.height as usize));
            Ok(())
        } else {
            // Partly off the screen: pixel by pixel, clipped.
            self.draw_iter(area.points().zip(colors).map(|(point, color)| Pixel(point, color)))
        }
    }

    fn fill_solid(&mut self, area: &Rectangle, color: Rgb565) -> Result<(), Infallible> {
        let area = self.bounding_box().intersection(area);
        if !area.is_zero_sized() {
            self.fill(&area, core::iter::repeat_n(color, area.size.width as usize * area.size.height as usize));
        }
        Ok(())
    }
}
