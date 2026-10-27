// A small canvas plot (docs/design/phase3/ui.md §5.2; Q3: no third-party code): line series over one x axis, linear
// axes with 1-2-5 ticks, a legend and a hover readout. Text and grid colours come from the page's CSS variables, so
// the plot follows the light and dark themes. NaN values leave gaps.

const PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2", "#9d755d"];
const PAD = { left: 58, right: 12, top: 22, bottom: 30 };

// Ticks of [lo, hi]: about `n` values at 1, 2 or 5 times a power of ten, at least `minStep` apart.
export function ticks(lo, hi, n = 5, minStep = 0) {
  if (!(hi > lo)) return [lo];
  const raw = Math.max((hi - lo) / n, minStep), p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

export function fmtNumber(v) {
  if (!Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(2);
  return String(Number(v.toPrecision(4)));
}

export class Plot {
  // series: [{name, color?}]; the canvas's CSS size sets the plot's (its owner redraws on resize).
  constructor(canvas, { title = "", xLabel = "", yLabel = "", series = [], signal = null } = {}) {
    Object.assign(this, { canvas, title, xLabel, yLabel, signal });
    this.series = series.map((s, i) => ({ color: PALETTE[i % PALETTE.length], ...s }));
    this.x = [];
    this.ys = this.series.map(() => []);
    this.hover = null;
    const on = (type, fn) => canvas.addEventListener(type, fn, signal ? { signal } : {});
    on("pointermove", (e) => {
      const r = canvas.getBoundingClientRect();
      this.hover = e.clientX - r.left;
      this.draw();
    });
    on("pointerleave", () => {
      this.hover = null;
      this.draw();
    });
  }

  // x: numbers; ys: one array per series, aligned with x.
  setData(x, ys) {
    this.x = x;
    this.ys = ys;
    this.canvas.dataset.points = String(x.length);
    this.draw();
  }

  range() {
    let xl = Infinity, xh = -Infinity, yl = Infinity, yh = -Infinity;
    for (const v of this.x) if (Number.isFinite(v)) [xl, xh] = [Math.min(xl, v), Math.max(xh, v)];
    for (const ys of this.ys) for (const v of ys) if (Number.isFinite(v)) [yl, yh] = [Math.min(yl, v), Math.max(yh, v)];
    if (!Number.isFinite(xl)) [xl, xh] = [0, 1];
    if (!Number.isFinite(yl)) [yl, yh] = [0, 1];
    yl = Math.min(yl, 0);
    if (xh === xl) xh = xl + 1;
    if (yh === yl) yh = yl + 1;
    return { xl, xh, yl, yh: yh + (yh - yl) * 0.05 };
  }

  draw() {
    const c = this.canvas, dpr = devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) Object.assign(c, { width: w, height: h });
    const g = c.getContext("2d"), css = getComputedStyle(c);
    const fg = css.getPropertyValue("--fg").trim() || "#222", muted = css.getPropertyValue("--muted").trim() || "#777";
    const line = css.getPropertyValue("--line").trim() || "#ccc";
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = w / dpr, H = h / dpr, pw = W - PAD.left - PAD.right, ph = H - PAD.top - PAD.bottom;
    g.clearRect(0, 0, W, H);
    g.font = "11px system-ui, sans-serif";
    if (pw < 20 || ph < 20) return;
    const { xl, xh, yl, yh } = this.range();
    const X = (v) => PAD.left + ((v - xl) / (xh - xl)) * pw, Y = (v) => PAD.top + ph - ((v - yl) / (yh - yl)) * ph;
    g.strokeStyle = line;
    g.fillStyle = muted;
    g.lineWidth = 1;
    g.textAlign = "right";
    g.textBaseline = "middle";
    for (const v of ticks(yl, yh, Math.max(3, Math.round(ph / 30)))) {
      g.beginPath();
      g.moveTo(PAD.left, Math.round(Y(v)) + 0.5);
      g.lineTo(PAD.left + pw, Math.round(Y(v)) + 0.5);
      g.stroke();
      g.fillText(fmtNumber(v), PAD.left - 6, Y(v));
    }
    g.textAlign = "center";
    g.textBaseline = "top";
    const xStep = this.x.every(Number.isInteger) ? 1 : 0;      // window indices: whole ticks
    for (const v of ticks(xl, xh, Math.max(2, Math.round(pw / 90)), xStep)) {
      g.fillText(fmtNumber(v), X(v), PAD.top + ph + 4);
    }
    g.strokeRect(PAD.left + 0.5, PAD.top + 0.5, pw, ph);
    if (this.xLabel) g.fillText(this.xLabel, PAD.left + pw / 2, PAD.top + ph + 16);
    g.save();
    g.translate(12, PAD.top + ph / 2);
    g.rotate(-Math.PI / 2);
    g.textBaseline = "middle";
    if (this.yLabel) g.fillText(this.yLabel, 0, 0);
    g.restore();
    g.lineWidth = 1.5;
    this.series.forEach((s, i) => {
      g.strokeStyle = s.color;
      g.beginPath();
      let pen = false;
      const ys = this.ys[i] ?? [];
      for (let k = 0; k < this.x.length; k++) {
        const v = ys[k];
        if (!Number.isFinite(v) || !Number.isFinite(this.x[k])) {
          pen = false;
          continue;
        }
        if (pen) g.lineTo(X(this.x[k]), Y(v));
        else g.moveTo(X(this.x[k]), Y(v));
        pen = true;
      }
      if (this.x.length === 1 && Number.isFinite(ys[0])) g.arc(X(this.x[0]), Y(ys[0]), 2, 0, 2 * Math.PI);
      g.stroke();
    });
    // title and legend on the top line; the hover readout replaces the legend's values
    g.textAlign = "left";
    g.textBaseline = "middle";
    g.fillStyle = fg;
    let at = PAD.left;
    if (this.title) {
      g.fillText(this.title, at, 10);
      at += g.measureText(this.title).width + 14;
    }
    let k = null;
    if (this.hover !== null && this.x.length) {
      const xv = xl + ((this.hover - PAD.left) / pw) * (xh - xl);
      k = 0;
      for (let j = 1; j < this.x.length; j++) if (Math.abs(this.x[j] - xv) < Math.abs(this.x[k] - xv)) k = j;
      g.strokeStyle = muted;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(Math.round(X(this.x[k])) + 0.5, PAD.top);
      g.lineTo(Math.round(X(this.x[k])) + 0.5, PAD.top + ph);
      g.stroke();
      const text = `${this.xLabel || "x"} ${fmtNumber(this.x[k])}`;
      g.fillStyle = muted;
      g.fillText(text, at, 10);
      at += g.measureText(text).width + 12;
    }
    this.series.forEach((s, i) => {
      const label = k === null ? s.name : `${s.name} ${fmtNumber(this.ys[i]?.[k])}`;
      g.fillStyle = s.color;
      g.fillRect(at, 6, 10, 8);
      g.fillStyle = fg;
      g.fillText(label, at + 13, 10);
      at += g.measureText(label).width + 26;
    });
  }
}
