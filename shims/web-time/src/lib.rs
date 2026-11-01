//! no_std replacement for `web-time`, patched in via `[patch.crates-io]`.
//!
//! Why: Burn 0.22 pulls `cubecl-environment`, which re-exports
//! `web_time::Instant` on every target whose `target_os` is not "none".
//! `web-time` links std on hosted targets and wasm-bindgen/js-sys on
//! wasm32-unknown-unknown. Burn only uses it for profiling/benchmarks, never
//! on the flex inference path, so a hookable clock is enough.
//!
//! Call `set_clock` to get real timings; without it, `Instant::now()` is 0.

#![no_std]

use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::sync::atomic::{AtomicUsize, Ordering};
pub use core::time::Duration;

static CLOCK: AtomicUsize = AtomicUsize::new(0);

/// Install a monotonic nanosecond clock.
pub fn set_clock(now_ns: fn() -> u64) {
    CLOCK.store(now_ns as usize, Ordering::Release);
}

fn now_ns() -> u64 {
    match CLOCK.load(Ordering::Acquire) {
        0 => 0,
        f => unsafe { core::mem::transmute::<usize, fn() -> u64>(f)() },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant(u64);

impl Instant {
    pub fn now() -> Self {
        Self(now_ns())
    }
    pub fn duration_since(&self, earlier: Instant) -> Duration {
        self.saturating_duration_since(earlier)
    }
    pub fn checked_duration_since(&self, earlier: Instant) -> Option<Duration> {
        self.0.checked_sub(earlier.0).map(Duration::from_nanos)
    }
    pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }
    pub fn elapsed(&self) -> Duration {
        Self::now().duration_since(*self)
    }
    pub fn checked_add(&self, d: Duration) -> Option<Instant> {
        u64::try_from(d.as_nanos()).ok().and_then(|n| self.0.checked_add(n)).map(Self)
    }
    pub fn checked_sub(&self, d: Duration) -> Option<Instant> {
        u64::try_from(d.as_nanos()).ok().and_then(|n| self.0.checked_sub(n)).map(Self)
    }
}

impl Add<Duration> for Instant {
    type Output = Instant;
    fn add(self, d: Duration) -> Instant {
        self.checked_add(d).expect("overflow adding duration to instant")
    }
}
impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, d: Duration) {
        *self = *self + d;
    }
}
impl Sub<Duration> for Instant {
    type Output = Instant;
    fn sub(self, d: Duration) -> Instant {
        self.checked_sub(d).expect("overflow subtracting duration from instant")
    }
}
impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, d: Duration) {
        *self = *self - d;
    }
}
impl Sub<Instant> for Instant {
    type Output = Duration;
    fn sub(self, other: Instant) -> Duration {
        self.duration_since(other)
    }
}
