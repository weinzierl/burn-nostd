//! Step 3: real signals. Spot words you teach yourself, from five recordings
//! each: audio, log-mel frames, word segmentation, a small MLP trained on
//! the device.

#![no_std]

extern crate alloc;

pub mod frontend;
pub mod segmenter;

use adam::Adam;
use alloc::vec::Vec;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::activation::softmax;
use burn::tensor::{Device, Int, Tensor};
use frontend::{MELS, MelFrame};

pub use segmenter::Segmenter;

pub const MAX_WORDS: usize = 8;
/// Time slices per word, whatever its duration.
const SLICES: usize = 10;
const INPUT: usize = SLICES * MELS + 1;
const HIDDEN: usize = 32;

/// Averages a word's frames into `SLICES` equal slices (independent of
/// speaking rate), removes the mean (independent of loudness) and appends
/// the duration.
pub fn features(word: &[MelFrame]) -> [f32; INPUT] {
    let n = word.len().max(1);
    let mut out = [0.0; INPUT];
    for s in 0..SLICES {
        let (a, b) = (s * n / SLICES, ((s + 1) * n / SLICES).clamp(s * n / SLICES + 1, n));
        for frame in &word[a..b] {
            for (o, v) in out[s * MELS..][..MELS].iter_mut().zip(frame) {
                *o += v / (b - a) as f32;
            }
        }
    }
    let mean = out[..INPUT - 1].iter().sum::<f32>() / (INPUT - 1) as f32;
    for v in &mut out[..INPUT - 1] {
        *v = (*v - mean) * 0.25;
    }
    out[INPUT - 1] = n as f32 / 50.0;
    out
}

#[derive(Module, Debug)]
struct Net {
    l1: Linear,
    l2: Linear,
    act: Relu,
}

impl Net {
    fn new(device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(INPUT, HIDDEN).init(device),
            l2: LinearConfig::new(HIDDEN, MAX_WORDS).init(device),
            act: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.l2.forward(self.act.forward(self.l1.forward(x)))
    }
}

pub struct Prediction {
    pub word: usize,
    pub confidence: f32,
}

pub struct Learner {
    device: Device,
    train_device: Device,
    net: Net,
    adam: Adam,
    examples: Vec<([f32; INPUT], i64)>,
    words: usize,
    recordings: usize,
    rng: u32,
    pub learning_rate: f32,
}

impl Learner {
    pub fn new(seed: u64) -> Self {
        let device = Device::default();
        let train_device = device.clone().autodiff();
        train_device.seed(seed);
        Self {
            net: Net::new(&train_device),
            device,
            train_device,
            adam: Adam::default(),
            examples: Vec::new(),
            words: 0,
            recordings: 0,
            rng: seed as u32 | 1,
            learning_rate: 2e-3,
        }
    }

    /// Stores the recording plus three copies with the word boundaries moved
    /// by one frame: the segmenter never cuts a word the same way twice, and
    /// with five recordings per word that jitter matters.
    pub fn add_example(&mut self, frames: &[MelFrame], word: usize) {
        assert!(word < MAX_WORDS);
        let n = frames.len();
        let variants: [&[MelFrame]; 4] = if n > 6 {
            [frames, &frames[1..], &frames[..n - 1], &frames[1..n - 1]]
        } else {
            [frames; 4]
        };
        for v in variants {
            self.examples.push((features(v), word as i64));
        }
        self.recordings += 1;
        self.words = self.words.max(word + 1);
    }

    pub fn recordings(&self) -> usize {
        self.recordings
    }

    /// Shuffled SGD with batch size 1, for the same reason as in step 2.
    pub fn train(&mut self, epochs: usize) -> f32 {
        assert!(!self.examples.is_empty(), "add examples first");
        let d = self.train_device.clone();
        let loss_fn = CrossEntropyLoss::new(None, &d);
        let mut order: Vec<usize> = (0..self.examples.len()).collect();
        let mut mean = f32::NAN;
        for _ in 0..epochs {
            shuffle(&mut order, &mut self.rng);
            let mut sum = 0.0;
            for &i in &order {
                let (x, y) = &self.examples[i];
                let x = Tensor::<1>::from_floats(x.as_slice(), &d).reshape([1, INPUT]);
                let y = Tensor::<1, Int>::from_ints([*y], &d);
                let loss = loss_fn.forward(self.net.forward(x), y);
                sum += loss.clone().into_scalar::<f32>();
                self.net = self.adam.step(self.learning_rate, self.net.clone(), loss.backward());
            }
            mean = sum / order.len() as f32;
        }
        mean
    }

    /// Probabilities of the words taught so far.
    pub fn probabilities(&self, frames: &[MelFrame]) -> Vec<f32> {
        let x = Tensor::<1>::from_floats(features(frames).as_slice(), &self.device).reshape([1, INPUT]);
        let mut p: Vec<f32> = softmax(self.net.valid().forward(x), 1).into_data().try_into_vec().expect("f32");
        p.truncate(self.words.max(1));
        p
    }

    pub fn classify(&self, frames: &[MelFrame]) -> Prediction {
        let p = self.probabilities(frames);
        let (word, &confidence) = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).expect("a word");
        Prediction { word, confidence }
    }
}

/// Fisher-Yates with xorshift32: no rand crate needed.
fn shuffle(v: &mut [usize], state: &mut u32) {
    for i in (1..v.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        v.swap(i, *state as usize % (i + 1));
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    const ENROLL: &[u8] = include_bytes!("../testdata/enroll.i16");
    const STREAM: &[u8] = include_bytes!("../testdata/stream.i16");

    fn pcm(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect()
    }

    /// Enroll 5 clips per word, then spot 12 unseen words in one stream.
    #[test]
    fn spots_words_in_a_stream() {
        let mut learner = Learner::new(42);
        for (i, clip) in pcm(ENROLL).chunks_exact(24_000).enumerate() {
            learner.add_example(&Segmenter::longest_word(clip.iter().copied()).unwrap(), i / 5);
        }
        learner.train(15);
        let mut segmenter = Segmenter::new();
        let mut heard = Vec::new();
        for chunk in pcm(STREAM).chunks(320) {
            segmenter.push(chunk, |w| heard.push(learner.classify(w).word));
        }
        segmenter.flush(|w| heard.push(learner.classify(w).word));
        let correct = heard.iter().enumerate().filter(|(i, w)| **w == i % 4).count();
        assert_eq!(heard.len(), 12);
        assert!(correct >= 11, "{correct}/12");
    }
}
