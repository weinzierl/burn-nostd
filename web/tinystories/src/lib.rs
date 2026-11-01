//! Step 4 in the browser. `step` produces one token per call, so the page
//! can show the story as it is written, along with the candidates the model
//! weighed for each token.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use burn::tensor::Device;
use rt::Global;
use tinystories::tokenizer::BOS;
use tinystories::{Llama, Sampler, Tokenizer, stories260k};

const CANDIDATES: usize = 5;

struct Story {
    prompt: Vec<usize>,
    pos: usize,
    token: usize,
    sampler: Sampler,
}

struct State {
    model: Llama,
    tokenizer: Tokenizer,
    input: Vec<u8>,
    output: Vec<u8>,
    story: Option<Story>,
    /// (token, probability), most likely first.
    candidates: Vec<(usize, f32)>,
}

static STATE: Global<State> = Global::new();

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(
        || {
            let model = Llama::load(stories260k::WEIGHTS, &Device::default(), 512);
            let tokenizer = Tokenizer::new(stories260k::TOKENIZER, model.config.vocab_size);
            State { model, tokenizer, input: Vec::new(), output: Vec::new(), story: None, candidates: Vec::new() }
        },
        f,
    )
}

/// The `CANDIDATES` most likely tokens, at the temperature that samples them
/// (at temperature 0, the plain distribution).
fn top(logits: &[f32], temperature: f32) -> Vec<(usize, f32)> {
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| libm::expf((l - max) / t)).collect();
    let sum: f32 = exp.iter().sum();
    let mut ranked: Vec<(usize, f32)> = exp.iter().map(|e| e / sum).enumerate().collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(CANDIDATES);
    ranked
}

/// Loads the model; returns the checkpoint size.
#[unsafe(no_mangle)]
pub extern "C" fn init() -> u32 {
    with(|_| stories260k::WEIGHTS.len() as u32)
}

/// A buffer for the UTF-8 prompt.
#[unsafe(no_mangle)]
pub extern "C" fn prompt(len: u32) -> *mut u8 {
    with(|s| {
        s.input.resize(len as usize, 0);
        s.input.as_mut_ptr()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn start(temperature: f32, seed: u32) {
    with(|s| {
        let prompt = s.tokenizer.encode(core::str::from_utf8(&s.input).unwrap_or(""));
        s.story = Some(Story { token: prompt[0], prompt, pos: 0, sampler: Sampler::new(temperature, seed) });
    })
}

/// One forward pass. Returns the length of the new text at `output()`,
/// or -1 when the story is finished.
#[unsafe(no_mangle)]
pub extern "C" fn step() -> i32 {
    with(|s| {
        let Some(story) = s.story.as_mut().filter(|st| st.pos < s.model.context()) else {
            s.story = None;
            return -1;
        };
        let mut logits = s.model.forward(story.token, story.pos);
        s.candidates = top(&logits, story.sampler.temperature);
        story.pos += 1;
        let next = match story.prompt.get(story.pos) {
            Some(&t) => t,
            None => story.sampler.sample(&mut logits),
        };
        if next == BOS {
            s.story = None;
            return -1;
        }
        s.output.clear();
        s.output.extend_from_slice(s.tokenizer.decode(story.token, next));
        story.token = next;
        s.output.len() as i32
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn output() -> *const u8 {
    with(|s| s.output.as_ptr())
}

/// 1 while the prompt is still being read, 0 while new text is written.
#[unsafe(no_mangle)]
pub extern "C" fn reading_prompt() -> u32 {
    with(|s| s.story.as_ref().is_some_and(|st| st.pos < st.prompt.len()) as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn candidate_probability(i: u32) -> f32 {
    with(|s| s.candidates.get(i as usize).map_or(0.0, |c| c.1))
}

/// Writes candidate `i`'s text to `output()`; returns its length.
#[unsafe(no_mangle)]
pub extern "C" fn candidate_text(i: u32) -> u32 {
    with(|s| {
        let Some(&(token, _)) = s.candidates.get(i as usize) else { return 0 };
        s.output.clear();
        // Any non-BOS predecessor keeps the leading space visible.
        s.output.extend_from_slice(s.tokenizer.decode(0, token));
        s.output.len() as u32
    })
}
