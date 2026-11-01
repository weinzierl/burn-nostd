# no_std shims

Burn 0.22 depends on `cubecl-environment` 0.11, which on every target except
`target_os = "none"` uses

* `web-time`, which links std on hosted targets and wasm-bindgen + js-sys on
  wasm32, and
* `wasm-bindgen-futures` (wasm32 only), which links std.

Libraries still build, but a no_std *binary* fails with
`duplicate lang item panic_impl`. These stand-ins are patched in through
`[patch.crates-io]`. Burn only reaches them from profiling and async code
paths, never from inference or training on the `flex` backend.
Bare metal (`target_os = "none"`) does not need them.
