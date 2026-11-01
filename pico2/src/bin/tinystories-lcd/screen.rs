//! What the screen shows, independent of the panel: a header, the story with
//! word wrap (the prompt in a second colour) and a footer. Only character cells
//! that changed are drawn, so a token costs a few glyphs and a scroll a redraw.

use alloc::vec::Vec;
use embedded_graphics::mono_font::{MonoTextStyleBuilder, ascii::FONT_10X20};
use embedded_graphics::pixelcolor::Rgb565;
use embedded_graphics::prelude::*;
use embedded_graphics::primitives::{PrimitiveStyle, Rectangle};
use embedded_graphics::text::{Baseline, Text};

/// The talk's colours (`slides/theme.typ`), in RGB565.
const fn rgb(hex: u32) -> Rgb565 {
    Rgb565::new((hex >> 19) as u8 & 0x1f, (hex >> 10) as u8 & 0x3f, (hex >> 3) as u8 & 0x1f)
}
const SLATE: Rgb565 = rgb(0x2C393E);
const CREAM: Rgb565 = rgb(0xEFE8D8);
const GOLD: Rgb565 = rgb(0xE0AE4E);
const SAND: Rgb565 = rgb(0xD9CDB3);

pub const WIDTH: u32 = 480;
pub const HEIGHT: u32 = 320;
const MARGIN: i32 = 20;
const CHAR_WIDTH: i32 = 10;
const CHAR_HEIGHT: i32 = 20;
const LINE_HEIGHT: i32 = 22;
const COLS: usize = (WIDTH as usize - 2 * MARGIN as usize) / CHAR_WIDTH as usize; // 44
const ROWS: usize = 10;
const HEADER_Y: i32 = 14;
const RULE_Y: i32 = 44;
const TEXT_Y: i32 = 60;
const FOOTER_Y: i32 = 290;

/// A character and whether it belongs to the prompt.
type Cell = (u8, bool);
const EMPTY: Cell = (b' ', false);

pub struct Screen<D> {
    pub target: D,
    story: Vec<Cell>,
    shown: [[Cell; COLS]; ROWS],
}

impl<D: DrawTarget<Color = Rgb565>> Screen<D> {
    /// Fills the screen and draws the rule under the header.
    pub fn new(mut target: D) -> Self {
        let _ = target.clear(SLATE);
        let rule = Rectangle::new(Point::new(MARGIN, RULE_Y), Size::new(WIDTH - 2 * MARGIN as u32, 2));
        let _ = rule.into_styled(PrimitiveStyle::with_fill(GOLD)).draw(&mut target);
        // Reserved once, before the model allocates: a story never moves it.
        Self { target, story: Vec::with_capacity(2048), shown: [[EMPTY; COLS]; ROWS] }
    }

    pub fn header(&mut self, left: &str, right: &str) {
        self.line(HEADER_Y, left, GOLD, right);
    }

    pub fn footer(&mut self, left: &str, right: &str) {
        self.line(FOOTER_Y, left, SAND, right);
    }

    /// Empties the story area (with the next redraw).
    pub fn new_story(&mut self) {
        self.story.clear();
        self.redraw();
    }

    /// Appends a piece of the story and draws what changed.
    pub fn push(&mut self, piece: &[u8], prompt: bool) {
        let printable = |b: u8| if b == b'\n' || (b' '..=b'~').contains(&b) { b } else { b'?' };
        self.story.extend(piece.iter().map(|&b| (printable(b), prompt)));
        self.redraw();
    }

    fn redraw(&mut self) {
        let lines = layout(&self.story, COLS);
        let first = lines.len().saturating_sub(ROWS); // scrolled: the last ROWS lines
        for row in 0..ROWS {
            for col in 0..COLS {
                let cell = lines.get(first + row).and_then(|line| line.get(col)).copied().unwrap_or(EMPTY);
                if self.shown[row][col] != cell {
                    self.shown[row][col] = cell;
                    let (x, y) = (MARGIN + CHAR_WIDTH * col as i32, TEXT_Y + LINE_HEIGHT * row as i32);
                    let mut utf8 = [0; 4];
                    self.text(x, y, (cell.0 as char).encode_utf8(&mut utf8), if cell.1 { GOLD } else { CREAM });
                }
            }
        }
    }

    fn line(&mut self, y: i32, left: &str, left_color: Rgb565, right: &str) {
        let area = Rectangle::new(Point::new(MARGIN, y), Size::new(WIDTH - 2 * MARGIN as u32, CHAR_HEIGHT as u32));
        let _ = self.target.fill_solid(&area, SLATE);
        self.text(MARGIN, y, left, left_color);
        self.text(WIDTH as i32 - MARGIN - CHAR_WIDTH * right.len() as i32, y, right, SAND);
    }

    fn text(&mut self, x: i32, y: i32, text: &str, color: Rgb565) {
        // With a background colour every glyph is one rectangle of pixels.
        let style = MonoTextStyleBuilder::new().font(&FONT_10X20).text_color(color).background_color(SLATE).build();
        let _ = Text::with_baseline(text, Point::new(x, y), style, Baseline::Top).draw(&mut self.target);
    }
}

/// Greedy word wrap into lines of at most `cols` cells. A newline starts a new
/// line, spaces at the start of a line are dropped, a word longer than a line
/// is broken.
fn layout(text: &[Cell], cols: usize) -> Vec<Vec<Cell>> {
    let mut lines = alloc::vec![Vec::new()];
    let mut i = 0;
    while i < text.len() {
        let line = lines.last_mut().unwrap();
        match text[i].0 {
            b'\n' => {
                lines.push(Vec::new());
                i += 1;
            }
            b' ' => {
                if !line.is_empty() && line.len() < cols {
                    line.push(text[i]);
                }
                i += 1;
            }
            _ => {
                let end = text[i..].iter().position(|c| c.0 == b' ' || c.0 == b'\n').map_or(text.len(), |n| i + n);
                if !line.is_empty() && line.len() + (end - i) > cols {
                    lines.push(Vec::new());
                }
                for &cell in &text[i..end] {
                    if lines.last().unwrap().len() == cols {
                        lines.push(Vec::new());
                    }
                    lines.last_mut().unwrap().push(cell);
                }
                i = end;
            }
        }
    }
    lines
}
