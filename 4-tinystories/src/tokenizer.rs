//! llama2.c's byte-pair tokenizer: `tok512.bin` or `tokenizer.bin`.

use alloc::vec::Vec;

pub const BOS: usize = 1;

pub struct Tokenizer {
    pieces: Vec<&'static [u8]>,
    scores: Vec<f32>,
}

impl Tokenizer {
    pub fn new(file: &'static [u8], vocab_size: usize) -> Self {
        let mut at = 4; // skips max_token_length
        let mut pieces = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            scores.push(f32::from_le_bytes(file[at..at + 4].try_into().unwrap()));
            let len = u32::from_le_bytes(file[at + 4..at + 8].try_into().unwrap()) as usize;
            pieces.push(&file[at + 8..at + 8 + len]);
            at += 8 + len;
        }
        Self { pieces, scores }
    }

    fn id(&self, piece: &[u8]) -> Option<usize> {
        self.pieces.iter().position(|p| *p == piece)
    }

    /// Same algorithm as run.c, so outputs match token for token: BOS, a
    /// leading space, code points (bytes as fallback), then the
    /// highest-scoring merge until none applies.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = alloc::vec![BOS];
        if !text.is_empty() {
            tokens.push(self.id(b" ").expect("space token"));
        }
        for ch in text.chars() {
            let mut buf = [0; 4];
            let bytes = ch.encode_utf8(&mut buf).as_bytes();
            match self.id(bytes) {
                Some(id) => tokens.push(id),
                // Byte tokens <0x00>..<0xFF> follow the 3 special tokens.
                None => tokens.extend(bytes.iter().map(|&b| b as usize + 3)),
            }
        }
        let mut pair = Vec::new();
        loop {
            let best = tokens
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    pair.clear();
                    pair.extend_from_slice(self.pieces[w[0]]);
                    pair.extend_from_slice(self.pieces[w[1]]);
                    self.id(&pair).map(|id| (i, id))
                })
                .max_by(|a, b| self.scores[a.1].total_cmp(&self.scores[b.1]).then(b.0.cmp(&a.0)));
            let Some((i, id)) = best else { break };
            tokens[i] = id;
            tokens.remove(i + 1);
        }
        tokens
    }

    /// The text of `token` following `previous`.
    pub fn decode(&self, previous: usize, token: usize) -> &'static [u8] {
        let piece = self.pieces[token];
        let piece = match piece.strip_prefix(b" ") {
            Some(rest) if previous == BOS => rest,
            _ => piece,
        };
        // Raw bytes are spelled "<0x0A>".
        if let [b'<', b'0', b'x', hi, lo, b'>'] = *piece {
            let hex = |c: u8| (c as char).to_digit(16).unwrap_or(0) as usize;
            let byte = hex(hi) * 16 + hex(lo);
            return &BYTES[byte..=byte];
        }
        piece
    }
}

static BYTES: [u8; 256] = {
    let mut b = [0; 256];
    let mut i = 0;
    while i < 256 {
        b[i] = i as u8;
        i += 1;
    }
    b
};
