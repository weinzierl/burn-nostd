//! Step 3 in the browser. The page writes 16 kHz audio into `audio(n)`,
//! then calls `record` while teaching or `listen` otherwise. Both feed a
//! spectrogram for display.

#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use keywords::frontend::{MELS, MelFrame};
use keywords::{Learner, Segmenter};
use rt::Global;

/// Spectrogram history: 3 s of frames, each MELS bands plus a "word" flag.
const HISTORY: usize = 150;
const ROW: usize = MELS + 1;

struct State {
    learner: Learner,
    audio: Vec<f32>,
    recording: Segmenter,
    longest: Option<Vec<MelFrame>>,
    listening: Segmenter,
    probabilities: Vec<f32>,
    spectrogram: Vec<f32>,
    rows: u32,
}

static STATE: Global<State> = Global::new();

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(
        || State {
            learner: Learner::new(42),
            audio: Vec::new(),
            recording: Segmenter::new(),
            longest: None,
            listening: Segmenter::new(),
            probabilities: Vec::new(),
            spectrogram: vec![0.0; HISTORY * ROW],
            rows: 0,
        },
        f,
    )
}

fn keep_longest(longest: &mut Option<Vec<MelFrame>>) -> impl FnMut(&[MelFrame]) + '_ {
    |word| {
        if longest.as_ref().is_none_or(|l| word.len() > l.len()) {
            *longest = Some(word.to_vec());
        }
    }
}

/// Appends the segmenter's newest frame to the history, if it has a new one.
fn show(spectrogram: &mut [f32], rows: &mut u32, segmenter: &Segmenter, frames_before: usize) {
    if segmenter.frames() == frames_before {
        return;
    }
    if let Some((frame, in_word)) = segmenter.latest() {
        let row = &mut spectrogram[(*rows as usize % HISTORY) * ROW..][..ROW];
        row[..MELS].copy_from_slice(frame);
        row[MELS] = in_word as u8 as f32;
        *rows += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn reset() {
    STATE.reset();
}

/// A buffer for `n` samples in [-1, 1] at 16 kHz.
#[unsafe(no_mangle)]
pub extern "C" fn audio(n: u32) -> *mut f32 {
    with(|s| {
        s.audio.resize(n as usize, 0.0);
        s.audio.as_mut_ptr()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn record_start() {
    with(|s| {
        s.recording = Segmenter::new();
        s.longest = None;
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn record(n: u32) {
    with(|s| {
        let before = s.recording.frames();
        s.recording.push(&s.audio[..n as usize], keep_longest(&mut s.longest));
        show(&mut s.spectrogram, &mut s.rows, &s.recording, before);
    })
}

/// Stores the recording for `word`; returns its length in 20 ms frames,
/// 0 if no word was heard.
#[unsafe(no_mangle)]
pub extern "C" fn record_end(word: u32) -> u32 {
    with(|s| {
        s.recording.flush(keep_longest(&mut s.longest));
        let Some(frames) = s.longest.take() else { return 0 };
        s.learner.add_example(&frames, word as usize);
        frames.len() as u32
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn train(epochs: u32) -> f32 {
    with(|s| s.learner.train(epochs as usize))
}

/// Feeds live audio. Returns how many words have a probability if a word
/// ended in this chunk (see `probability`), otherwise 0.
#[unsafe(no_mangle)]
pub extern "C" fn listen(n: u32) -> u32 {
    with(|s| {
        let before = s.listening.frames();
        let mut heard = None;
        let learner = &s.learner;
        s.listening.push(&s.audio[..n as usize], |frames| {
            if learner.recordings() > 0 {
                heard = Some(learner.probabilities(frames));
            }
        });
        show(&mut s.spectrogram, &mut s.rows, &s.listening, before);
        match heard {
            Some(p) => {
                s.probabilities = p;
                s.probabilities.len() as u32
            }
            None => 0,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn probability(word: u32) -> f32 {
    with(|s| s.probabilities.get(word as usize).copied().unwrap_or(0.0))
}

/// The spectrogram history: `HISTORY` rows of `MELS + 1` floats, oldest
/// at `rows() % HISTORY`.
#[unsafe(no_mangle)]
pub extern "C" fn spectrogram() -> *const f32 {
    with(|s| s.spectrogram.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn rows() -> u32 {
    with(|s| s.rows)
}
