"""Regenerate testdata/*.i16 with espeak-ng (needs espeak-ng, numpy, scipy)."""
import random, subprocess
import numpy as np
from scipy.signal import resample_poly

WORDS, CLIP, RATE = ["yes", "no", "left", "right"], 24_000, 16_000
random.seed(1)
np.random.seed(1)


def say(word):
    wav = subprocess.run(
        ["espeak-ng", "-v", "en-us", "-s", str(random.randint(130, 200)),
         "-p", str(random.randint(25, 75)), "--stdout", word],
        capture_output=True, check=True).stdout
    x = np.frombuffer(wav[44:], dtype=np.int16).astype(np.float32) / 32768
    return resample_poly(x, 320, 441)  # espeak speaks at 22.05 kHz


def clip(word):
    x = say(word)[:20_000] * random.uniform(0.3, 1.0)
    out = np.zeros(CLIP, np.float32)
    start = random.randint(1600, CLIP - len(x) - 1600)
    out[start:start + len(x)] = x
    snr = 10 ** (random.uniform(10, 30) / 10)
    out += np.random.randn(CLIP).astype(np.float32) * np.sqrt(np.mean(x ** 2) / snr) * 0.5
    return (np.clip(out, -1, 1) * 32767).astype("<i2")


np.concatenate([clip(w) for w in WORDS for _ in range(5)]).tofile("testdata/enroll.i16")
np.concatenate([clip(w) for _ in range(3) for w in WORDS]).tofile("testdata/stream.i16")
