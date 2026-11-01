//! 16 kHz audio to 16 log-mel bands per 20 ms frame.

use alloc::vec::Vec;
use core::f32::consts::PI;

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME: usize = 400;
pub const HOP: usize = 320;
pub const MELS: usize = 16;
const NFFT: usize = 512;

pub type MelFrame = [f32; MELS];

pub struct Frontend {
    window: Vec<f32>,
    /// First FFT bin and triangle weights per mel band.
    filters: Vec<(usize, Vec<f32>)>,
    twiddles: Vec<(f32, f32)>,
    re: Vec<f32>,
    im: Vec<f32>,
}

fn mel(hz: f32) -> f32 {
    2595.0 * libm::log10f(1.0 + hz / 700.0)
}

fn hz(mel: f32) -> f32 {
    700.0 * (libm::powf(10.0, mel / 2595.0) - 1.0)
}

impl Frontend {
    pub fn new() -> Self {
        let window = (0..FRAME)
            .map(|i| 0.5 - 0.5 * libm::cosf(2.0 * PI * i as f32 / (FRAME - 1) as f32))
            .collect();
        let (lo, hi) = (mel(60.0), mel(7600.0));
        let bin = |i: usize| hz(lo + (hi - lo) * i as f32 / (MELS + 1) as f32) * NFFT as f32 / SAMPLE_RATE as f32;
        let filters = (0..MELS)
            .map(|m| {
                let (l, c, r) = (bin(m), bin(m + 1), bin(m + 2));
                let first = libm::ceilf(l) as usize;
                let weights = (first..=(r as usize).min(NFFT / 2))
                    .map(|b| {
                        let b = b as f32;
                        if b <= c { (b - l) / (c - l) } else { (r - b) / (r - c) }.max(0.0)
                    })
                    .collect();
                (first, weights)
            })
            .collect();
        let twiddles = (0..NFFT / 2)
            .map(|k| libm::sincosf(-2.0 * PI * k as f32 / NFFT as f32))
            .map(|(s, c)| (c, s))
            .collect();
        Self { window, filters, twiddles, re: alloc::vec![0.0; NFFT], im: alloc::vec![0.0; NFFT] }
    }

    /// Log-mel energies and RMS of one `FRAME`-sample frame.
    pub fn frame(&mut self, x: &[f32]) -> (MelFrame, f32) {
        let rms = libm::sqrtf(x.iter().map(|v| v * v).sum::<f32>() / FRAME as f32);
        for i in 0..NFFT {
            self.re[i] = if i < FRAME { x[i] * self.window[i] } else { 0.0 };
            self.im[i] = 0.0;
        }
        self.fft();
        let power = |b: usize| self.re[b] * self.re[b] + self.im[b] * self.im[b];
        let bands = core::array::from_fn(|m| {
            let (first, weights) = &self.filters[m];
            let e: f32 = weights.iter().enumerate().map(|(j, w)| w * power(first + j)).sum();
            libm::logf(e + 1e-6)
        });
        (bands, rms)
    }

    /// In-place radix-2 FFT: 40 lines instead of another dependency.
    fn fft(&mut self) {
        let n = NFFT;
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                self.re.swap(i, j);
                self.im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = self.twiddles[k * (n / len)];
                    let (a, b) = (start + k, start + k + len / 2);
                    let tr = self.re[b] * wr - self.im[b] * wi;
                    let ti = self.re[b] * wi + self.im[b] * wr;
                    self.re[b] = self.re[a] - tr;
                    self.im[b] = self.im[a] - ti;
                    self.re[a] += tr;
                    self.im[a] += ti;
                }
            }
            len <<= 1;
        }
    }
}

impl Default for Frontend {
    fn default() -> Self {
        Self::new()
    }
}
