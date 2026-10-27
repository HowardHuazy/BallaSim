"""A self-contained static HTML viewer for scrubbing through rendered windows.

The page needs no server: frames are embedded as base64 images (one file to
share) or written to a sibling ``<name>_frames/`` folder that the page
references by relative path (lighter for long ranges). It shows the panels side
by side with a time slider, play/pause, frame stepping, playback speed, a blink
mode that alternates the first two panels in place, 1:1 pixel zoom, and an
events-per-frame strip that seeks on click.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .compare import Renderer, ordered_map
from .events import NS_PER_MS, NS_PER_S

IMAGE_FORMATS = {
    "webp": (".webp", [cv2.IMWRITE_WEBP_QUALITY, 101], "image/webp"),  # quality > 100 is lossless
    "png": (".png", [cv2.IMWRITE_PNG_COMPRESSION, 6], "image/png"),
}
_KIND_NAMES = {
    "events": "signed event frames",
    "polarity": "polarity-count images",
    "timesurface": "time surfaces",
    "counts": "count maps",
}
# Strip colours: the reference stream in neutral ink, the others in the categorical slots.
_STRIP_COLORS = {"dark": ("#c3c2b7", "#eb6834", "#1baf7a"), "white": ("#3d3c39", "#eb6834", "#1baf7a")}


def encode_image(image: np.ndarray, image_format: str = "webp") -> bytes:
    suffix, params, _ = IMAGE_FORMATS[image_format]
    ok, buffer = cv2.imencode(suffix, np.ascontiguousarray(image[..., ::-1]), params)
    if not ok:
        raise RuntimeError(f"OpenCV cannot encode {image_format}")
    return buffer.tobytes()


def write_html(
    renderer: Renderer,
    out_path: str | Path,
    *,
    title: str = "BallaSim event viewer",
    fps: float = 25.0,
    image_format: str = "webp",
    embed: bool = True,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Render every window and write the viewer to ``out_path`` (``.html``).

    With ``embed=False`` the images go to ``<stem>_frames/`` next to the page,
    which must then be moved together with it.
    """
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"image_format must be one of {tuple(IMAGE_FORMATS)}, got {image_format!r}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path.with_name(out_path.stem + "_frames")
    if not embed:
        frames_dir.mkdir(exist_ok=True)
    suffix, _, mime = IMAGE_FORMATS[image_format]
    n = len(renderer)

    def job(k: int):
        frame = renderer.render(k)
        images = []
        for p, panel in enumerate(frame.panels):
            data = encode_image(panel.image, image_format)
            if embed:
                images.append(base64.b64encode(data).decode("ascii"))
            else:
                name = f"{p}_{k:06d}{suffix}"
                (frames_dir / name).write_bytes(data)
                images.append(f"{frames_dir.name}/{name}")
        return frame, images

    panels, images, infos, counts = [], [], [], [[] for _ in renderer.sources]
    for done, (frame, frame_images) in enumerate(ordered_map(job, range(n), workers), start=1):
        if not panels:
            panels = [{"title": p.title, "width": p.image.shape[1], "height": p.image.shape[0]} for p in frame.panels]
            images = [[] for _ in panels]
            infos = [[] for _ in panels]
        for p, panel in enumerate(frame.panels):
            images[p].append(frame_images[p])
            infos[p].append(panel.info)
        for s, c in enumerate(frame.counts):
            counts[s].append(c)
        if progress is not None:
            progress(done, n)

    timeline, options = renderer.timeline, renderer.options
    starts_s = timeline.starts_ns / NS_PER_S
    dt_ms = float(np.diff(timeline.starts_ns[:2])[0]) / NS_PER_MS if n > 1 else timeline.window_ns / NS_PER_MS
    colors = _STRIP_COLORS[options.theme]
    data = {
        "title": title,
        "subtitle": (
            f"{_KIND_NAMES[options.kind]}, {timeline.window_ns / NS_PER_MS:g} ms windows every {dt_ms:g} ms, "
            f"t = {starts_s[0]:.3f} to {starts_s[-1] + timeline.window_ns / NS_PER_S:.3f} s, {n} frames, "
            f"colour clip {renderer.clip:g} events"
        ),
        "theme": options.theme,
        "fps": fps,
        "dt_ms": dt_ms,
        "times_s": [round(float(t), 9) for t in starts_s],
        "panels": panels,
        "infos": infos,
        "streams": [
            {"label": role, "color": colors[min(i, len(colors) - 1)], "counts": c}
            for i, (role, c) in enumerate(zip(renderer.roles, counts, strict=True))
        ],
        "images": {"mime": mime, "data": images} if embed else {"paths": images},
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out_path.write_text(_TEMPLATE.replace("__TITLE__", _escape(title)).replace("__DATA__", payload), encoding="utf-8")
    return out_path


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg: #0d0d0e; --surface: #161618; --ink: #ecece8; --muted: #969590; --line: #2c2c2a; --accent: #eb6834; }
:root[data-theme="white"] {
  --bg: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --muted: #52514e; --line: #e1e0d9; --accent: #d95926;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
header { padding: 14px 16px 4px; }
h1 { font-size: 15px; font-weight: 600; margin: 0; }
.sub { color: var(--muted); }
.panels { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 8px; padding: 8px 16px; }
.panels.actual { grid-template-columns: repeat(var(--n), max-content); overflow-x: auto; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
figure[hidden] { display: none; }
figcaption { padding: 6px 8px; min-width: 0; }
.ptitle { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pinfo {
  color: var(--muted); font-variant-numeric: tabular-nums; white-space: pre; overflow: hidden; text-overflow: ellipsis;
}
figure img { display: block; width: 100%; height: auto; }
/* Nearest-neighbour only at or above native size: shrinking with it drops whole rows and columns. */
figure img.crisp { image-rendering: pixelated; }
.panels.actual figure img { width: auto; max-width: none; }
.controls { display: flex; align-items: center; gap: 8px; padding: 6px 16px; flex-wrap: wrap; }
button, select {
  background: var(--surface); color: var(--ink); border: 1px solid var(--line); border-radius: 6px;
  padding: 4px 10px; font: inherit; cursor: pointer;
}
button:hover, select:hover { border-color: var(--muted); }
button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
input[type=range] { flex: 1 1 240px; accent-color: var(--accent); }
.time { font-variant-numeric: tabular-nums; min-width: 17ch; }
canvas { display: block; width: calc(100% - 32px); height: 76px; margin: 4px 16px; cursor: pointer; }
.help { color: var(--muted); padding: 4px 16px 18px; }
kbd { border: 1px solid var(--line); border-radius: 3px; padding: 0 4px; font: 11px ui-monospace, monospace; }
@media (max-width: 700px) { .panels { grid-template-columns: minmax(0, 1fr); } }
</style>
</head>
<body>
<header><h1 id="title"></h1><div class="sub" id="subtitle"></div></header>
<div class="panels" id="panels"></div>
<div class="controls">
  <button id="play" aria-pressed="false">Play</button>
  <button id="prev" title="Previous frame (Left)" aria-label="Previous frame">&#8592;</button>
  <button id="next" title="Next frame (Right)" aria-label="Next frame">&#8594;</button>
  <input type="range" id="slider" min="0" value="0" step="1" aria-label="Frame">
  <span class="time" id="time"></span>
  <label>Speed <select id="speed"></select></label>
  <button id="blink" aria-pressed="false" title="Alternate the first two panels in place (B)">Blink</button>
  <button id="zoom" aria-pressed="false" title="Show images at 1:1 pixels (Z)">1:1 pixels</button>
</div>
<canvas id="strip" aria-label="Events per frame; click to seek"></canvas>
<div class="help">
  <kbd>Space</kbd> play/pause &nbsp; <kbd>&#8592;</kbd> <kbd>&#8594;</kbd> step (<kbd>Shift</kbd> &times;10) &nbsp;
  <kbd>Home</kbd> <kbd>End</kbd> &nbsp; <kbd>B</kbd> blink &nbsp; <kbd>Z</kbd> 1:1 pixels &nbsp; click the strip to seek
</div>
<script type="application/json" id="viewer-data">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("viewer-data").textContent);
const N = D.times_s.length;
const $ = (id) => document.getElementById(id);
document.documentElement.dataset.theme = D.theme;
$("title").textContent = D.title;
$("subtitle").textContent = D.subtitle;

const src = (p, k) => D.images.data ? "data:" + D.images.mime + ";base64," + D.images.data[p][k] : D.images.paths[p][k];
const panelsEl = $("panels");
panelsEl.style.setProperty("--n", D.panels.length);
const figs = D.panels.map((panel) => {
  const fig = document.createElement("figure");
  const cap = document.createElement("figcaption");
  const title = document.createElement("div");
  title.className = "ptitle";
  title.textContent = panel.title;
  const info = document.createElement("div");
  info.className = "pinfo";
  const img = document.createElement("img");
  img.width = panel.width;
  img.height = panel.height;
  img.alt = panel.title;
  cap.append(title, info);
  fig.append(cap, img);
  panelsEl.append(fig);
  return { fig, title, info, img };
});

// Enlarged images stay crisp (every event pixel a sharp square); shrunk ones are smoothed by the
// browser, which averages the dropped pixels instead of skipping them.
function updateCrisp(img) {
  const native = img.naturalWidth || +img.getAttribute("width");
  img.classList.toggle("crisp", img.clientWidth * (window.devicePixelRatio || 1) >= native);
}
const sizeWatch = new ResizeObserver((entries) => entries.forEach((e) => updateCrisp(e.target)));
figs.forEach((f) => {
  sizeWatch.observe(f.img);
  f.img.addEventListener("load", () => updateCrisp(f.img));
});

let k = 0, playing = false, timer = null, blink = false, blinkPhase = 0, blinkTimer = null;
const slider = $("slider");
slider.max = N - 1;

function show(next) {
  k = Math.max(0, Math.min(N - 1, next));
  slider.value = k;
  const t = D.times_s[k];
  $("time").textContent = "t = " + t.toFixed(4) + " s  (" + (k + 1) + "/" + N + ")";
  figs.forEach((f, p) => {
    let shown = p;
    if (blink && p === 0) shown = blinkPhase;
    f.fig.hidden = blink && p === 1;
    f.img.src = src(shown, k);
    f.title.textContent = D.panels[shown].title;
    f.info.textContent = D.infos[shown][k];
  });
  if (D.images.paths) {  // prefetch the next frames from the sibling folder
    for (let j = 1; j <= 4 && k + j < N; j++) for (let p = 0; p < D.panels.length; p++) new Image().src = src(p, k + j);
  }
  drawStrip();
}

const speeds = [1, 2, 5, 10, 25, 50, 100];
if (!speeds.includes(D.fps)) speeds.push(D.fps);
speeds.sort((a, b) => a - b);
for (const fps of speeds) {
  const o = document.createElement("option");
  o.value = fps;
  o.textContent = fps + " fps (" + +(fps * D.dt_ms / 1000).toPrecision(3) + "x real time)";
  if (fps === D.fps) o.selected = true;
  $("speed").append(o);
}

function setPlaying(on) {
  playing = on && N > 1;
  $("play").textContent = playing ? "Pause" : "Play";
  $("play").setAttribute("aria-pressed", playing);
  clearInterval(timer);
  if (playing) timer = setInterval(() => show(k + 1 < N ? k + 1 : 0), 1000 / +$("speed").value);
}

function setBlink(on) {
  blink = on && D.panels.length >= 2;
  $("blink").setAttribute("aria-pressed", blink);
  clearInterval(blinkTimer);
  blinkPhase = 0;
  panelsEl.style.setProperty("--n", D.panels.length - (blink ? 1 : 0));
  if (blink) blinkTimer = setInterval(() => { blinkPhase ^= 1; show(k); }, 450);
  show(k);
}

function setZoom(on) {
  panelsEl.classList.toggle("actual", on);
  $("zoom").setAttribute("aria-pressed", on);
}

const strip = $("strip");
function drawStrip() {
  const dpr = window.devicePixelRatio || 1;
  const w = strip.clientWidth, h = strip.clientHeight;
  if (strip.width !== Math.round(w * dpr) || strip.height !== Math.round(h * dpr)) {
    strip.width = Math.round(w * dpr);
    strip.height = Math.round(h * dpr);
  }
  const g = strip.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const top = 16, bottom = h - 2;
  const max = D.streams.reduce((m, s) => s.counts.reduce((a, c) => Math.max(a, c), m), 1);
  const x = (i) => N > 1 ? (i / (N - 1)) * (w - 1) : w / 2;
  const y = (c) => bottom - (c / max) * (bottom - top);
  g.strokeStyle = css.getPropertyValue("--line");
  g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, bottom + 0.5); g.lineTo(w, bottom + 0.5); g.stroke();
  g.font = "11px system-ui, sans-serif";
  let lx = 0;
  for (const s of D.streams) {
    g.strokeStyle = s.color;
    g.lineWidth = 1.5;
    g.beginPath();
    s.counts.forEach((c, i) => (i ? g.lineTo(x(i), y(c)) : g.moveTo(x(i), y(c))));
    g.stroke();
    g.fillStyle = s.color;
    g.fillRect(lx, 4, 10, 3);
    g.fillStyle = css.getPropertyValue("--muted");
    g.fillText(s.label, lx + 14, 10);
    lx += 24 + g.measureText(s.label).width;
  }
  g.fillText("events per frame (max " + max.toLocaleString() + ")", lx + 8, 10);
  g.strokeStyle = css.getPropertyValue("--accent");
  g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(x(k), top - 4); g.lineTo(x(k), bottom); g.stroke();
}

function seekFromStrip(ev) {
  const r = strip.getBoundingClientRect();
  show(Math.round(((ev.clientX - r.left) / r.width) * (N - 1)));
}
strip.addEventListener("pointerdown", (ev) => { strip.setPointerCapture(ev.pointerId); seekFromStrip(ev); });
strip.addEventListener("pointermove", (ev) => { if (strip.hasPointerCapture(ev.pointerId)) seekFromStrip(ev); });
slider.addEventListener("input", () => show(+slider.value));
$("play").addEventListener("click", () => setPlaying(!playing));
$("prev").addEventListener("click", () => show(k - 1));
$("next").addEventListener("click", () => show(k + 1));
$("speed").addEventListener("change", () => setPlaying(playing));
$("blink").addEventListener("click", () => setBlink(!blink));
$("zoom").addEventListener("click", () => setZoom(!panelsEl.classList.contains("actual")));
window.addEventListener("resize", drawStrip);
document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "SELECT" || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const step = ev.shiftKey ? 10 : 1;
  const actions = {
    " ": () => setPlaying(!playing),
    ArrowLeft: () => show(k - step),
    ArrowRight: () => show(k + step),
    Home: () => show(0),
    End: () => show(N - 1),
    b: () => setBlink(!blink),
    z: () => setZoom(!panelsEl.classList.contains("actual")),
  };
  const action = actions[ev.key.length === 1 ? ev.key.toLowerCase() : ev.key];
  if (action) { ev.preventDefault(); action(); }
});
if (D.panels.length < 2) $("blink").hidden = true;
show(0);
</script>
</body>
</html>
"""
