//! Step 3 on the desktop.
//!
//! `cargo keywords` uses the bundled test recordings.
//! `cargo keywords -- --enroll yes a.wav b.wav --enroll no c.wav --listen d.wav e.wav`
//! uses your own recordings: 16-bit WAV, any sample rate, one word per
//! enrollment file. Listen files are played back to back with 0.5 s pauses.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use core::ffi::{c_char, c_int};
use desktop::{micros, println};
use keywords::{Learner, Segmenter};

const ENROLL: &[u8] = include_bytes!("../../../3-keywords/testdata/enroll.i16");
const STREAM: &[u8] = include_bytes!("../../../3-keywords/testdata/stream.i16");

#[unsafe(no_mangle)]
extern "C" fn main(argc: c_int, argv: *const *const c_char) -> c_int {
    let args = desktop::args(argc, argv);
    let mut words: Vec<&str> = Vec::new();
    let mut recordings: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut stream = Vec::new();

    if args.is_empty() {
        words.extend(["yes", "no", "left", "right"]);
        let enroll = desktop::audio(ENROLL).unwrap();
        recordings.extend(enroll.chunks_exact(24_000).enumerate().map(|(i, c)| (i / 5, c.to_vec())));
        stream = desktop::audio(STREAM).unwrap();
    } else {
        let mut mode = "";
        for &arg in &args {
            match arg {
                "--enroll" | "--listen" => mode = arg,
                word if mode == "--enroll" && !word.ends_with(".wav") => words.push(word),
                path => {
                    let audio = match desktop::read(path).ok_or("cannot read it").and_then(desktop::audio) {
                        Ok(audio) => audio,
                        Err(e) => {
                            println!("{path}: {e}");
                            return 2;
                        }
                    };
                    match mode {
                        "--listen" => {
                            stream.extend(audio);
                            stream.extend([0.0; 8_000]);
                        }
                        _ if !words.is_empty() => recordings.push((words.len() - 1, audio)),
                        _ => {
                            println!("usage: --enroll WORD FILE.wav... [--enroll ...] --listen FILE.wav");
                            return 2;
                        }
                    }
                }
            }
        }
    }

    let mut learner = Learner::new(42);
    for (word, audio) in &recordings {
        match Segmenter::longest_word(audio.iter().copied()) {
            Some(frames) => learner.add_example(&frames, *word),
            None => println!("no word in a recording of '{}'", words[*word]),
        }
    }
    let t = micros();
    learner.train(15);
    println!("{} recordings, trained in {} ms\n", learner.recordings(), (micros() - t) / 1000);

    let mut heard = Vec::new();
    let mut segmenter = Segmenter::new();
    let mut on_word = |frames: &[_]| {
        let t = micros();
        let p = learner.classify(frames);
        println!("heard {:<6} {:3.0} %  ({} µs)", words[p.word], p.confidence * 100.0, micros() - t);
        heard.push(p.word);
    };
    for chunk in stream.chunks(320) {
        segmenter.push(chunk, &mut on_word);
    }
    segmenter.flush(&mut on_word);

    if !args.is_empty() {
        return 0;
    }
    let correct = heard.iter().enumerate().filter(|(i, w)| **w == i % 4).count();
    println!("\n{correct}/12 correct (expected: yes no left right, three times)");
    if heard.len() == 12 && correct >= 11 { 0 } else { 1 }
}
