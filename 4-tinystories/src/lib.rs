//! Step 4: someone else's pretrained model. Karpathy's llama2.c TinyStories
//! checkpoints, unchanged, on Burn.
//!
//! Every weight tensor borrows the checkpoint bytes instead of copying them,
//! so the 1 MB model runs from flash on a chip with 520 KiB of RAM.

#![no_std]

extern crate alloc;

pub mod tokenizer;

use alloc::vec;
use alloc::vec::Vec;
use burn::tensor::activation::silu;
use burn::tensor::{AllocationProperty, Bytes, DType, Device, Tensor, TensorData};

pub use tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    shared_classifier: bool,
}

impl Config {
    fn parse(header: &[u8]) -> Self {
        let field = |i: usize| i32::from_le_bytes(header[4 * i..][..4].try_into().unwrap());
        Self {
            dim: field(0) as usize,
            hidden_dim: field(1) as usize,
            layers: field(2) as usize,
            heads: field(3) as usize,
            kv_heads: field(4) as usize,
            vocab_size: field(5).unsigned_abs() as usize,
            seq_len: field(6) as usize,
            // llama2.c marks an unshared classifier with a negative vocab size.
            shared_classifier: field(5) > 0,
        }
    }

    fn head_size(&self) -> usize {
        self.dim / self.heads
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_size()
    }
}

/// Weights are stored `[out, in]`, as in llama2.c.
struct Layer {
    attention_norm: Tensor<2>,
    wq: Tensor<2>,
    wk: Tensor<2>,
    wv: Tensor<2>,
    wo: Tensor<2>,
    ffn_norm: Tensor<2>,
    w1: Tensor<2>,
    w2: Tensor<2>,
    w3: Tensor<2>,
}

pub struct Llama {
    pub config: Config,
    device: Device,
    embedding: Tensor<2>,
    layers: Vec<Layer>,
    final_norm: Tensor<2>,
    classifier: Tensor<2>,
    context: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    scores: Vec<f32>,
}

/// `x Wᵀ`. The transpose is a strided view, and with one row in `x` gemm
/// stays on its matrix-vector path, which allocates no packing buffer.
fn linear(x: Tensor<2>, w: &Tensor<2>) -> Tensor<2> {
    x.matmul(w.clone().transpose())
}

fn rms_norm(x: Tensor<2>, weight: &Tensor<2>) -> Tensor<2> {
    let rms = x.clone().powi_scalar(2).mean_dim(1).add_scalar(1e-5).sqrt();
    x.div(rms).mul(weight.clone())
}

fn to_vec(t: Tensor<2>) -> Vec<f32> {
    t.into_data().try_into_vec().expect("f32")
}

impl Llama {
    /// `checkpoint`: a llama2.c `.bin` file, 4-byte aligned and `'static` so
    /// the tensors can borrow it. `context` bounds the KV cache, the only
    /// sizable allocation.
    pub fn load(checkpoint: &'static [u8], device: &Device, context: usize) -> Self {
        assert_eq!(checkpoint.as_ptr() as usize % 4, 0, "checkpoint must be 4-byte aligned");
        let config = Config::parse(checkpoint);
        let context = context.min(config.seq_len);
        let mut offset = 28;
        let mut next = |rows: usize, cols: usize| {
            let len = rows * cols * 4;
            let bytes = bytes::Bytes::from_static(&checkpoint[offset..offset + len]);
            offset += len;
            let bytes = Bytes::from_shared(bytes, AllocationProperty::Other);
            Tensor::<2>::from_data(TensorData::from_bytes(bytes, [rows, cols], DType::F32), device)
        };
        let Config { dim, hidden_dim, layers: n, vocab_size, .. } = config;
        let kv = config.kv_dim();

        let embedding = next(vocab_size, dim);
        // llama2.c stores each weight kind for all layers back to back.
        let mut kind = |rows, cols| (0..n).map(|_| next(rows, cols)).collect::<Vec<_>>().into_iter();
        let mut attention_norm = kind(1, dim);
        let mut wq = kind(dim, dim);
        let mut wk = kind(kv, dim);
        let mut wv = kind(kv, dim);
        let mut wo = kind(dim, dim);
        let mut ffn_norm = kind(1, dim);
        let mut w1 = kind(hidden_dim, dim);
        let mut w2 = kind(dim, hidden_dim);
        let mut w3 = kind(hidden_dim, dim);
        let layers = (0..n)
            .map(|_| Layer {
                attention_norm: attention_norm.next().unwrap(),
                wq: wq.next().unwrap(),
                wk: wk.next().unwrap(),
                wv: wv.next().unwrap(),
                wo: wo.next().unwrap(),
                ffn_norm: ffn_norm.next().unwrap(),
                w1: w1.next().unwrap(),
                w2: w2.next().unwrap(),
                w3: w3.next().unwrap(),
            })
            .collect();
        let final_norm = next(1, dim);
        // Skip llama2.c's precomputed RoPE tables; computing RoPE is cheaper than reading it.
        next(config.seq_len, config.head_size());
        let classifier = if config.shared_classifier { embedding.clone() } else { next(vocab_size, dim) };

        Self {
            config,
            device: device.clone(),
            embedding,
            layers,
            final_norm,
            classifier,
            context,
            keys: vec![0.0; n * context * kv],
            values: vec![0.0; n * context * kv],
            scores: vec![0.0; context],
        }
    }

    pub fn context(&self) -> usize {
        self.context
    }

    /// One decoding step: logits for the token after `token` at `pos`.
    pub fn forward(&mut self, token: usize, pos: usize) -> Vec<f32> {
        assert!(pos < self.context, "context full");
        let c = self.config;
        let (hs, kv_dim) = (c.head_size(), c.kv_dim());
        let mut x = self.embedding.clone().slice([token..token + 1, 0..c.dim]);

        for (l, layer) in self.layers.iter().enumerate() {
            let h = rms_norm(x.clone(), &layer.attention_norm);
            let mut q = to_vec(linear(h.clone(), &layer.wq));
            let mut k = to_vec(linear(h.clone(), &layer.wk));
            let v = to_vec(linear(h, &layer.wv));
            rope(&mut q, pos, hs);
            rope(&mut k, pos, hs);

            let cache = l * self.context * kv_dim;
            self.keys[cache + pos * kv_dim..][..kv_dim].copy_from_slice(&k);
            self.values[cache + pos * kv_dim..][..kv_dim].copy_from_slice(&v);

            // Attention runs on short vectors: per-head tensor ops would cost
            // more in dispatch than in arithmetic.
            let mut out = vec![0.0; c.dim];
            for head in 0..c.heads {
                let qh = &q[head * hs..][..hs];
                let kv_offset = head / (c.heads / c.kv_heads) * hs;
                let scores = &mut self.scores[..=pos];
                for (t, s) in scores.iter_mut().enumerate() {
                    let kt = &self.keys[cache + t * kv_dim + kv_offset..][..hs];
                    *s = qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() / libm::sqrtf(hs as f32);
                }
                softmax(scores);
                let oh = &mut out[head * hs..][..hs];
                for (t, s) in scores.iter().enumerate() {
                    let vt = &self.values[cache + t * kv_dim + kv_offset..][..hs];
                    oh.iter_mut().zip(vt).for_each(|(o, v)| *o += s * v);
                }
            }
            let out = Tensor::<1>::from_floats(out.as_slice(), &self.device).reshape([1, c.dim]);
            x = x + linear(out, &layer.wo);

            let h = rms_norm(x.clone(), &layer.ffn_norm);
            let gated = silu(linear(h.clone(), &layer.w1)) * linear(h, &layer.w3);
            x = x + linear(gated, &layer.w2);
        }
        to_vec(linear(rms_norm(x, &self.final_norm), &self.classifier))
    }
}

/// Rotary position embedding, as in llama2.c's run.c.
fn rope(v: &mut [f32], pos: usize, head_size: usize) {
    for i in (0..v.len()).step_by(2) {
        let freq = 1.0 / libm::powf(10_000.0, (i % head_size) as f32 / head_size as f32);
        let (sin, cos) = libm::sincosf(pos as f32 * freq);
        let (a, b) = (v[i], v[i + 1]);
        v[i] = a * cos - b * sin;
        v[i + 1] = a * sin + b * cos;
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::MIN, f32::max);
    x.iter_mut().for_each(|v| *v = libm::expf(*v - max));
    let sum: f32 = x.iter().sum();
    x.iter_mut().for_each(|v| *v /= sum);
}

/// Greedy at temperature 0, otherwise sampling with a xorshift RNG.
pub struct Sampler {
    pub temperature: f32,
    state: u32,
}

impl Sampler {
    pub fn new(temperature: f32, seed: u32) -> Self {
        Self { temperature, state: seed.max(1) }
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature <= 0.0 {
            // First maximum, like run.c.
            return logits.iter().enumerate().fold(0, |best, (i, l)| if *l > logits[best] { i } else { best });
        }
        logits.iter_mut().for_each(|l| *l /= self.temperature);
        softmax(logits);
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        let mut r = (self.state >> 8) as f32 / (1 << 24) as f32;
        for (i, p) in logits.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        logits.len() - 1
    }
}

/// Text generation, as llama2.c's `generate`: feeds the prompt, then samples
/// until `steps` positions or an end-of-text token. Returns tokens sampled.
pub fn generate(
    model: &mut Llama,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: &str,
    steps: usize,
    mut emit: impl FnMut(&[u8]),
) -> usize {
    let prompt = tokenizer.encode(prompt);
    let mut token = prompt[0];
    let mut sampled = 0;
    for pos in 0..steps.min(model.context()) {
        let mut logits = model.forward(token, pos);
        let next = match prompt.get(pos + 1) {
            Some(&t) => t,
            None => {
                sampled += 1;
                sampler.sample(&mut logits)
            }
        };
        if next == tokenizer::BOS {
            break;
        }
        emit(tokenizer.decode(token, next));
        token = next;
    }
    sampled
}

/// TinyStories 260K, embedded (about 1 MB).
pub mod stories260k {
    #[repr(C, align(16))]
    struct Aligned<T: ?Sized>(T);
    static CHECKPOINT: &Aligned<[u8]> = &Aligned(*include_bytes!("../model/stories260K.bin"));

    pub static WEIGHTS: &[u8] = &CHECKPOINT.0;
    pub static TOKENIZER: &[u8] = include_bytes!("../model/tok512.bin");
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    /// llama2.c: `run stories260K.bin -z tok512.bin -t 0 -n 80 -i "Once upon a time"`
    const REFERENCE: &str = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily's mom said, \"Lily, let's go to";

    #[test]
    fn matches_llama2c() {
        let device = Device::default();
        let mut model = Llama::load(stories260k::WEIGHTS, &device, 128);
        let tokenizer = Tokenizer::new(stories260k::TOKENIZER, model.config.vocab_size);
        let mut text = Vec::new();
        generate(&mut model, &tokenizer, &mut Sampler::new(0.0, 1), "Once upon a time", 80, |piece| {
            text.extend_from_slice(piece)
        });
        assert_eq!(String::from_utf8(text).unwrap(), REFERENCE);
    }
}
