//! Cuts words out of a continuous stream: loudness against an adaptive
//! noise floor. Streaming, so a recording never has to sit in RAM.

use crate::frontend::{FRAME, Frontend, HOP, MelFrame};
use alloc::collections::VecDeque;
use alloc::vec::Vec;

const CALIBRATION: usize = 4; // the first 80 ms measure the room
const RATIO: f32 = 4.0; // speech: RMS above 4x the noise floor
const ABS_MIN: f32 = 0.004; // never trigger on near-silence
const HANGOVER: usize = 12; // 240 ms of quiet end a word
const MIN_FRAMES: usize = 8;
const MAX_FRAMES: usize = 75;
const PRE_ROLL: usize = 3; // keep the soft onset that triggered late

pub struct Segmenter {
    frontend: Frontend,
    samples: Vec<f32>,
    recent: VecDeque<MelFrame>,
    word: Vec<MelFrame>,
    in_word: bool,
    quiet: usize,
    noise: f32,
    frames: usize,
}

impl Segmenter {
    pub fn new() -> Self {
        Self {
            frontend: Frontend::new(),
            samples: Vec::with_capacity(FRAME + HOP),
            recent: VecDeque::with_capacity(PRE_ROLL + 1),
            word: Vec::with_capacity(MAX_FRAMES),
            in_word: false,
            quiet: 0,
            noise: 0.0,
            frames: 0,
        }
    }

    /// Feeds samples in [-1, 1]; calls `on_word` for each completed word.
    pub fn push(&mut self, input: &[f32], mut on_word: impl FnMut(&[MelFrame])) {
        for &s in input {
            self.samples.push(s);
            if self.samples.len() == FRAME {
                let (frame, rms) = self.frontend.frame(&self.samples);
                self.samples.drain(..HOP);
                self.step(frame, rms, &mut on_word);
            }
        }
    }

    /// Frames analysed so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The newest frame and whether it belongs to a word, for displays.
    pub fn latest(&self) -> Option<(&MelFrame, bool)> {
        self.recent.back().map(|frame| (frame, self.in_word))
    }

    /// Ends a word still in progress, e.g. at the end of a recording.
    pub fn flush(&mut self, on_word: impl FnMut(&[MelFrame])) {
        self.push(&[0.0; HANGOVER * HOP + FRAME], on_word);
    }

    fn step(&mut self, frame: MelFrame, rms: f32, on_word: &mut impl FnMut(&[MelFrame])) {
        self.frames += 1;
        if self.frames <= CALIBRATION {
            self.noise += rms / CALIBRATION as f32;
            return;
        }
        let speech = rms > (self.noise * RATIO).max(ABS_MIN);
        if !speech {
            // Follow a quieter room quickly, a louder one slowly.
            let rate = if rms < self.noise { 0.1 } else { 0.005 };
            self.noise += rate * (rms - self.noise);
        }

        if self.in_word {
            self.word.push(frame);
            self.quiet = if speech { 0 } else { self.quiet + 1 };
            if self.quiet == HANGOVER || self.word.len() == MAX_FRAMES {
                let end = (self.word.len() + 2).saturating_sub(self.quiet).min(self.word.len());
                if end >= MIN_FRAMES + PRE_ROLL {
                    on_word(&self.word[..end]);
                }
                self.in_word = false;
            }
        } else if speech {
            self.in_word = true;
            self.quiet = 0;
            self.word.clear();
            self.word.extend(self.recent.iter());
            self.word.push(frame);
        }

        if self.recent.len() == PRE_ROLL {
            self.recent.pop_front();
        }
        self.recent.push_back(frame);
    }

    /// The longest word in a finished recording, for enrollment.
    pub fn longest_word(samples: impl IntoIterator<Item = f32>) -> Option<Vec<MelFrame>> {
        let mut segmenter = Self::new();
        let mut longest: Option<Vec<MelFrame>> = None;
        let mut keep = |w: &[MelFrame]| {
            if longest.as_ref().is_none_or(|l| w.len() > l.len()) {
                longest = Some(w.to_vec());
            }
        };
        let mut chunk = [0.0; HOP];
        let mut samples = samples.into_iter().peekable();
        while samples.peek().is_some() {
            let n = chunk.iter_mut().zip(&mut samples).map(|(c, s)| *c = s).count();
            segmenter.push(&chunk[..n], &mut keep);
        }
        segmenter.flush(&mut keep);
        longest
    }
}

impl Default for Segmenter {
    fn default() -> Self {
        Self::new()
    }
}
