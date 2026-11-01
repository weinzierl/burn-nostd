//! no_std replacement for `wasm-bindgen-futures`, patched in via
//! `[patch.crates-io]` for the wasm32 build only.
//!
//! Why: `cubecl-environment` (pulled by Burn 0.22) depends on
//! wasm-bindgen-futures unconditionally on wasm targets, for
//! `spawn_detached`. That drags wasm-bindgen, js-sys and std into the graph.
//! The flex backend computes synchronously and never awaits JS promises, so
//! polling the future to completion on the spot is equivalent here.

#![no_std]

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};

pub fn spawn_local<F>(future: F)
where
    F: Future<Output = ()> + 'static,
{
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    while let Poll::Pending = future.as_mut().poll(&mut cx) {
        core::hint::spin_loop();
    }
}
