// The timeline strip (docs/design/phase3/ui.md §4.3 step 1, §5.1 "Open a run"): the stream's rate overview (BSRT,
// totals at once, the ON share where blocks are built), the view window and the playhead; scrub (drag), zoom (wheel:
// the window length ×2^{±1/4}), play (real time) and step (arrow keys). Times are BigInt ns on the view clock; the
// window is [playhead − length, playhead), half-open like ballasim.viz's windows.
// Tracks (§3.3 /tracks, BSTK; UI-7: a Blender job's preview frames) are drawn in a band along the strip's top: a tick
// at each mark's time and its window shaded. "[" and "]", or a click in the band, put the view window on a mark's
// window on the sensor clock (a preview frame's events; its PNG shows the same ones when the run's clock offset and
// drift are 0, BL's default).

import { api, apiBytes } from "../app/api.js";
import { decodeBSRT, decodeBSTK } from "../app/wire.js";

const MAX_BINS = 2048;
const MIN_LENGTH = 1000n;                              // 1 µs
const NO_COUNT = 0xffffffff;
const TRACK_BAND = 9;                                  // CSS px

export class Timeline {
  // t0, t1: the stream's span [t0, t1) on the view clock; onChange({w0, w1, playhead}) after every move. Aborting
  // `signal` removes the strip's listeners and stops playback for good.
  constructor(canvas, { t0, t1, length, onChange = () => {}, signal = null }) {
    Object.assign(this, { canvas, t0, t1, onChange, signal });
    this.span = t1 > t0 ? t1 - t0 : 1n;
    this.length = length ?? (this.span < 33_000_000n ? this.span : 33_000_000n);
    this.playhead = t0 + this.length;
    canvas.dataset.window = `${t0} ${this.playhead}`;
    this.playing = false;
    this.rate = null;
    this.tracks = [];
    this.marks = [];                                   // [{t, w0, w1}] of every track, by t
    this.colors = { on: [255, 75, 62], off: [79, 143, 255] };
    canvas.tabIndex = 0;
    const on = (type, fn, options = {}) => canvas.addEventListener(type, fn, signal ? { ...options, signal } : options);
    on("pointerdown", (e) => {
      const r = canvas.getBoundingClientRect();
      if (this.marks.length && e.clientY - r.top < TRACK_BAND && this.snap(e.clientX)) return;
      canvas.setPointerCapture(e.pointerId);
      this.scrub(e);
    });
    on("pointermove", (e) => { if (canvas.hasPointerCapture(e.pointerId)) this.scrub(e); });
    on("wheel", (e) => {
      e.preventDefault();
      this.zoom(e.deltaY > 0 ? 1 : -1);
    }, { passive: false });
    on("keydown", (e) => {
      if (e.key === " ") this.toggle();
      else if (e.key === "ArrowRight") this.set(this.playhead + this.length);
      else if (e.key === "ArrowLeft") this.set(this.playhead - this.length);
      else if (e.key === "]" || e.key === "[") this.jump(e.key === "]" ? 1 : -1);
      else return;
      e.preventDefault();
    });
    signal?.addEventListener("abort", () => { this.playing = false; });
  }

  get window() { return { w0: this.playhead - this.length, w1: this.playhead, playhead: this.playhead }; }

  // The rate overview of stream `sid` (offset δ onto the view clock), at most 2048 bins on its 1 ms grid.
  async load(sid, delta = 0n) {
    const binMs = Math.max(1, Math.ceil(Number(this.span) / 1e6 / MAX_BINS));
    const r = decodeBSRT(await apiBytes(`/streams/${sid}/rate?bin_ms=${binMs}`, { cache: "no-store" }));
    this.rate = { ...r, t0: r.t0 + BigInt(delta) };
    this.draw();
  }

  // The tracks of run `rid` for `camera` (a track of another camera is on that camera's clock: left out), placed on
  // the view clock with the stream's offset δ.
  async loadTracks(rid, camera, delta = 0n) {
    const tracks = [];
    for (const t of await api(`/runs/${rid}/tracks`)) {
      if (t.camera && camera && t.camera !== camera) continue;
      const m = decodeBSTK(await apiBytes(`/runs/${rid}/tracks/${t.id}`, { cache: "no-store" }));
      tracks.push({ ...t, tNs: m.tNs, windows: m.columns.window?.values ?? null });
    }
    this.setTracks(tracks, delta);
    return tracks;
  }

  setTracks(tracks, delta = 0n) {
    const d = BigInt(delta);
    this.tracks = tracks;
    this.marks = tracks.flatMap((tr) => Array.from(tr.tNs, (t, i) => ({ t: t + d,
      w0: tr.windows ? tr.windows[2 * i] + d : t + d, w1: tr.windows ? tr.windows[2 * i + 1] + d : t + d })))
      .sort((a, b) => (a.t < b.t ? -1 : a.t > b.t ? 1 : 0));
    this.canvas.dataset.tracks = tracks.map((tr) => `${tr.id}:${tr.tNs.length}`).join(" ");
    this.draw();
  }

  // The view window on mark `m`'s window (or, for an empty one, the playhead on its time).
  toMark(m) {
    if (m.w1 > m.w0) this.set(m.w1, m.w1 - m.w0);
    else this.set(m.t);
  }

  // The next (direction 1) or previous (−1) mark: the first whose window ends after the playhead, or the last ending
  // before it (a mark without a window: its time).
  jump(direction) {
    const end = (k) => (k.w1 > k.w0 ? k.w1 : k.t);
    const m = direction > 0 ? this.marks.find((k) => end(k) > this.playhead)
      : this.marks.findLast((k) => end(k) < this.playhead);
    if (m) this.toMark(m);
    return m ?? null;
  }

  // A click in the track band: the mark nearest to it within 6 px, if any.
  snap(clientX) {
    const r = this.canvas.getBoundingClientRect(), x = clientX - r.left, span = Number(this.span);
    let best = null, dist = 6;
    for (const m of this.marks) {
      const d = Math.abs((Number(m.t - this.t0) / span) * r.width - x);
      if (d <= dist) [best, dist] = [m, d];
    }
    if (best) this.toMark(best);
    return best;
  }

  set(playhead, length = this.length) {
    length = length < MIN_LENGTH ? MIN_LENGTH : length > this.span ? this.span : length;
    const lo = this.t0 + 1n;
    this.playhead = playhead < lo ? lo : playhead > this.t1 ? this.t1 : playhead;
    this.length = length;
    this.canvas.dataset.window = `${this.playhead - this.length} ${this.playhead}`;
    this.draw();
    this.onChange(this.window);
  }

  timeAt(clientX) {
    const r = this.canvas.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (clientX - r.left) / Math.max(1, r.width)));
    return this.t0 + BigInt(Math.round(f * Number(this.span)));
  }

  scrub(e) { this.set(this.timeAt(e.clientX)); }

  zoom(direction) {
    const f = 2 ** (direction / 4);
    this.set(this.playhead, BigInt(Math.max(1, Math.round(Number(this.length) * f))));
  }

  toggle() {
    if (this.signal?.aborted) return;
    this.playing = !this.playing;
    if (!this.playing) return;
    if (this.playhead >= this.t1) this.set(this.t0 + this.length);
    let last = performance.now();
    const step = (now) => {
      if (!this.playing || this.signal?.aborted) return;
      const dt = BigInt(Math.round((now - last) * 1e6));
      last = now;
      this.set(this.playhead + dt);
      if (this.playhead >= this.t1) this.playing = false;
      else requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  draw() {
    const c = this.canvas, dpr = devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) Object.assign(c, { width: w, height: h });
    const g = c.getContext("2d");
    g.clearRect(0, 0, w, h);
    const x = (t) => (Number(t - this.t0) / Number(this.span)) * w;
    if (this.rate) {
      const { total, on, binNs } = this.rate, cols = new Float64Array(w), ons = new Float64Array(w),
        known = new Float64Array(w), base = Number(this.rate.t0 - this.t0), perNs = w / Number(this.span);
      for (let i = 0; i < total.length; i++) {
        const col = Math.floor((base + i * binNs) * perNs);
        if (col < 0 || col >= w) continue;
        cols[col] += total[i];
        if (on[i] !== NO_COUNT) {
          ons[col] += on[i];
          known[col] += total[i];
        }
      }
      const max = Math.log1p(cols.reduce((a, b) => Math.max(a, b), 0)) || 1;
      const [r1, g1, b1] = this.colors.on, [r0, g0, b0] = this.colors.off;
      for (let i = 0; i < w; i++) {
        if (!cols[i]) continue;
        const f = known[i] ? ons[i] / known[i] : null;
        g.fillStyle = f === null ? "#888" : `rgb(${r0 + f * (r1 - r0)},${g0 + f * (g1 - g0)},${b0 + f * (b1 - b0)})`;
        const bh = (Math.log1p(cols[i]) / max) * (h - 2);
        g.fillRect(i, h - bh, 1, bh);
      }
    }
    if (this.marks.length) {
      const band = TRACK_BAND * dpr;
      g.fillStyle = "rgba(128,128,128,0.18)";
      g.fillRect(0, 0, w, band);
      this.marks.forEach((m, i) => {                   // windows shaded in turn, so adjacent ones stay apart
        if (m.w1 <= m.w0) return;
        g.fillStyle = i % 2 ? "rgba(224,160,0,0.40)" : "rgba(224,160,0,0.20)";
        g.fillRect(x(m.w0), 0, Math.max(1, x(m.w1) - x(m.w0)), band);
      });
      g.fillStyle = "#a05a00";
      for (const m of this.marks) g.fillRect(Math.round(x(m.t)) - dpr, 0, 2 * dpr, band);
    }
    const { w0, w1 } = this.window;
    g.fillStyle = "rgba(128,128,128,0.25)";
    g.fillRect(x(w0), 0, Math.max(1, x(w1) - x(w0)), h);
    g.fillStyle = "#e0a000";
    g.fillRect(Math.min(w - 2, x(this.playhead)), 0, 2 * dpr, h);
  }
}
