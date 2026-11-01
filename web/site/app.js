// Shared by the demo pages: loading a module, sharp canvases, charts, bars.
"use strict";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const group = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, "\u2009");
const frame = () => new Promise(requestAnimationFrame);

// arrayBuffer + instantiate works whatever MIME type the server sends.
async function loadModule(name) {
  const response = await fetch(`${name}.wasm`);
  if (!response.ok) throw new Error(`${name}.wasm: HTTP ${response.status}. Serve the site with “cargo web”.`);
  const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
  return instance.exports;
}

function fail(error) {
  $("error").hidden = false;
  $("error").textContent = `WebAssembly did not start: ${error.message}`;
}

// Every reset bumps the generation; work that awaits (training loops, recordings,
// playback) remembers the one it started in and stops when it has changed.
let generation = 0;
const current = (g) => g === generation;

// A Rust panic traps the module and leaves it unusable: say so once, stop calling it.
let broken = false;
function crashed(error) {
  if (broken) return;
  broken = true;
  $("error").hidden = false;
  $("error").textContent = `The WebAssembly module stopped (${error.message}). Reload the page.`;
}

// A cleared 2D context in CSS pixels, at the screen's pixel density.
function surface(canvas, height) {
  const dpr = devicePixelRatio || 1, width = canvas.clientWidth;
  canvas.style.height = `${height}px`;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, width, height);
  return { g, width, height };
}

// A line over the whole width; log scale for losses.
function chart(canvas, height, series, log = false) {
  const { g, width } = surface(canvas, height);
  g.strokeStyle = css("--rule"); g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, height - 0.5); g.lineTo(width, height - 0.5); g.stroke();
  if (series.length < 2) return;
  const ys = series.map((v) => (log ? Math.log10(Math.max(v, 1e-9)) : v));
  const lo = Math.min(...ys), span = Math.max(...ys) - lo || 1;
  g.strokeStyle = css("--accent"); g.lineWidth = 2; g.beginPath();
  ys.forEach((y, i) => {
    const px = (i / (ys.length - 1)) * width, py = 3 + (1 - (y - lo) / span) * (height - 6);
    i ? g.lineTo(px, py) : g.moveTo(px, py);
  });
  g.stroke();
}

// One bar per label; returns a function that shows probabilities.
function bars(container, labels) {
  container.innerHTML = labels.map(() =>
    `<div class="prob"><span></span><div class="track"><div class="fill"></div></div><span class="pct"></span></div>`).join("");
  const rows = [...container.children];
  rows.forEach((row, i) => (row.firstElementChild.textContent = labels[i]));
  return (values, names = labels) => {
    const best = values.indexOf(Math.max(...values));
    rows.forEach((row, i) => {
      const v = values[i] ?? 0;
      row.firstElementChild.textContent = names[i] ?? "";
      row.querySelector(".fill").style.width = `${v * 100}%`;
      row.querySelector(".pct").textContent = values.length > i ? `${Math.round(v * 100)} %` : "";
      row.classList.toggle("best", i === best);
    });
  };
}
