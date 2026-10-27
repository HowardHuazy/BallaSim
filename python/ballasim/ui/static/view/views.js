// Event views (docs/design/phase3/ui.md §4.4, §4.5) and the viewer screen (#/view, §5.1 "Open a run"). The 2D kinds
// draw the window [W0, W1) (rolling shutter included) from the planner's pages into per-pixel planes, then the colour
// pass, from colours the server serves (GET /colors: viz's themes, LUTs and exact tables), so Python and the browser
// share one source:
// - "events", "polarity", "counts": ON and OFF counts; exact tables give viz.render's event_frame, polarity_image and
//   colorize_magnitude bit for bit, f32 formulas cover what the tables do not (§4.5);
// - "timesurface" and "decay": history views over [t_ref − H, t_ref), H = ⌊5τ⌋, t_ref = W1, drawn only from blocks
//   complete at j = 0 (a hatch while others load, §4.3 step 7), with the split key and the exact age base of §4.4;
// - "cloud": the (x, y, t) cloud (cloud.js).
// Without a WebGPU adapter (or a pool), the screen falls back to PNG panels rendered by the server (§4.11).

import { api, apiResponse } from "../app/api.js";
import { CloudView } from "./cloud.js";
import { GpuError, computePipeline, openGpu, readBuffer, readTexture, renderPipeline, scoped } from "./gpu.js";
import { Inspector } from "./inspector.js";
import { ACCUMULATE, DECAY, HISTOGRAM, HISTOGRAM_BINS, LUT_ENTRIES, MODE, PRESENT, PV, PV_WORDS, RESOLVE,
  SEQUENTIAL_AT, TIME_SURFACE, dispatchPages, linearRgb, packPages } from "./kernels.js";
import { Planner } from "./planner.js";
import { Pool, poolBytes } from "./pool.js";
import { Timeline } from "./timeline.js";

export const KINDS = ["events", "polarity", "counts", "timesurface", "decay", "cloud"];
export const HISTORY_KINDS = new Set(["timesurface", "decay"]);
const TABLE_KINDS = new Set(["events", "polarity", "counts"]);   // kinds with an exact table in /colors
const MAX_ROW_TOTAL_NS = 1 << 29;                 // §4.4: H·row_ns < 2^29 ns keeps the clamped window test exact
const AUTO_CLIP_WINDOWS = 16;
const fromBase64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const words = (s) => new Uint32Array(fromBase64(s).buffer);
const usage = (...names) => names.reduce((u, n) => u | GPUBufferUsage[n], 0);

// The history window of the time surface and decay (§4.4): H = ⌊5τ⌋ ns as viz computes it (int(5·τ_ns)), and the
// key shift s = max(0, ⌈log2(H + 1)⌉ − 30).
export function historyWindow(tauMs) {
  const tauNs = tauMs * 1e6, h = Math.trunc(5 * tauNs);
  if (!(tauNs > 0) || !Number.isSafeInteger(h) || h < 1) {
    throw new GpuError("BS-UI-REQUEST", `τ = ${tauMs} ms: need 5τ between 1 ns and 2^53 ns`);
  }
  const H = BigInt(h);
  return { tauNs, H, s: Math.max(0, H.toString(2).length - 30) };
}

// The colours of `kind` for (theme, clip, gamma) from GET /colors: the theme, the LUT buffer's contents (diverging
// LUT, then sequential LUT) and the kind's exact table (null when the server builds none).
export async function fetchColors(theme, clip, gamma, kind = "events") {
  const c = await api(`/colors?theme=${theme}&clip=${clip}&gamma=${gamma}&kind=${kind}`);
  const lut = new Uint32Array(LUT_ENTRIES);
  lut.set(words(c.lut_rgba));
  lut.set(words(c.sequential_rgba), SEQUENTIAL_AT);
  return { ...c, lut, table: c.table_rgba ? words(c.table_rgba) : null };
}

// numpy.percentile(values, 99) ("linear") of the values a 1024-bin histogram holds (bin v: the count of v, the last
// bin: of every value from 1023 on), and whether it is exact (both neighbours below 1023).
export function percentile99(hist) {
  let n = 0;
  for (const c of hist) n += c;
  if (!n) return { value: null, n, exact: true };
  const at = (index) => {
    let seen = 0;
    for (let v = 0; v < hist.length; v++) if (index < (seen += hist[v])) return v;
    return hist.length - 1;
  };
  const vi = (n - 1) * 0.99;
  let lo = Math.floor(vi), hi = lo + 1;
  if (vi >= n - 1) lo = hi = n - 1;
  const a = at(lo), b = at(hi), g = vi - lo, d = b - a;
  return { value: g >= 0.5 ? b - d * (1 - g) : a + d * g, n, exact: b < hist.length - 1 };
}

export class EventView {
  // One panel of one stream: the kind's planes, the colour pass and (with a canvas) the presentation.
  constructor(gpu, planner, stream, canvas = null) {
    Object.assign(this, { gpu, planner, stream, canvas });
    this.width = stream.width;
    this.height = stream.height;
    this.kind = "events";
    this.tauMs = 20;
    this.rowNs = 0;
    this.zoom = 1;
    this.center = [this.width / 2, this.height / 2];
    this.lastDraw = null;
    this.firstFrame = null;
    this.colorSeq = 0;
    this.destroyed = false;
  }

  // Refuses (BS-UI-TOO-WIDE) an image the device cannot hold: a side beyond maxTextureDimension2D, or counts (two u32
  // planes, 8·W·H bytes) beyond one storage binding. Nothing is allocated then.
  static checkSize(device, W, H) {
    const L = device.limits, side = L.maxTextureDimension2D;
    const binding = Math.min(L.maxStorageBufferBindingSize, L.maxBufferSize);
    if (W > side || H > side || 8 * W * H > binding) {
      throw new GpuError("BS-UI-TOO-WIDE", `${W}×${H}: this device draws images of at most ${side} px a side and ` +
        `${Math.floor(binding / 8)} pixels (8 bytes of counts each)`);
    }
  }

  async init(messages = []) {
    const { gpu, width: W, height: H } = this, d = gpu.device, pool = this.planner.pool;
    EventView.checkSize(d, W, H);
    this.kernels = {
      events: await computePipeline(gpu, ACCUMULATE, "accumulate", messages),
      timesurface: await computePipeline(gpu, TIME_SURFACE, "time_surface", messages),
      decay: await computePipeline(gpu, DECAY, "decay", messages),
    };
    this.resolve = await computePipeline(gpu, RESOLVE, "resolve", messages);
    this.histogram = await computePipeline(gpu, HISTOGRAM, "histogram", messages);
    this.counts = d.createBuffer({ size: 2 * W * H * 4, usage: usage("STORAGE", "COPY_SRC", "COPY_DST"),
      label: "counts" });
    const align = d.limits.minStorageBufferOffsetAlignment;
    this.region = Math.ceil((pool.perBuffer * PV_WORDS * 4) / align) * align;
    this.pv = d.createBuffer({ size: this.region * pool.buffers.length, usage: usage("STORAGE", "COPY_DST"),
      label: "page views" });
    this.frameUniform = d.createBuffer({ size: 16, usage: usage("UNIFORM", "COPY_DST") });
    this.resolveUniform = d.createBuffer({ size: 96, usage: usage("UNIFORM", "COPY_DST") });
    this.lut = d.createBuffer({ size: LUT_ENTRIES * 4, usage: usage("STORAGE", "COPY_DST"), label: "lut" });
    this.hist = d.createBuffer({ size: HISTOGRAM_BINS * 4, usage: usage("STORAGE", "COPY_SRC", "COPY_DST"),
      label: "clip histogram" });
    this.histUniform = d.createBuffer({ size: 16, usage: usage("UNIFORM", "COPY_DST") });
    this.image = d.createTexture({ size: [W, H], format: "rgba8unorm", label: "image",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.groups = {};
    for (const [kind, pipeline] of Object.entries(this.kernels)) {
      this.groups[kind] = pool.buffers.map((buffer, b) => d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: this.pv, offset: b * this.region, size: this.region } },
          { binding: 2, resource: { buffer: this.frameUniform } },
          { binding: 3, resource: { buffer: this.counts } }] }));
    }
    this.histGroup = d.createBindGroup({ layout: this.histogram.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.counts } }, { binding: 1, resource: { buffer: this.hist } },
      { binding: 2, resource: { buffer: this.histUniform } }] });
    this.pixel = d.createBuffer({ size: 16, usage: usage("MAP_READ", "COPY_DST"), label: "inspector" });
    if (this.canvas) {
      this.present = await renderPipeline(gpu, PRESENT, "present", gpu.format, messages);
      this.presentUniform = d.createBuffer({ size: 48, usage: usage("UNIFORM", "COPY_DST") });
      this.presentGroup = d.createBindGroup({ layout: this.present.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.image.createView() },
        { binding: 1, resource: { buffer: this.presentUniform } }] });
      this.context = this.canvas.getContext("webgpu");
      this.context.configure({ device: d, format: gpu.format, alphaMode: "opaque" });
    }
  }

  setKind(kind) {
    if (!KINDS.includes(kind) || kind === "cloud") throw new GpuError("BS-UI-REQUEST", `no 2D kind ${kind}`);
    this.kind = kind;
  }

  setTau(tauMs) {
    historyWindow(tauMs);
    this.tauMs = tauMs;
  }

  // Colours for (theme, clip, gamma) and the current kind; the table is rebuilt on the server for every clip. Answers
  // may arrive out of order: only the latest call's answer is written. An earlier call's answer, or one that arrives
  // after the view was destroyed, writes nothing and resolves to null.
  async setColors({ theme = "dark", clip = 3, gamma = 0.5 } = {}) {
    const seq = ++this.colorSeq;
    const kind = TABLE_KINDS.has(this.kind) ? this.kind : "events";
    const c = await fetchColors(theme, clip, gamma, kind), d = this.gpu.device;
    if (this.destroyed || seq !== this.colorSeq) return null;
    const table = c.table ?? new Uint32Array(1);
    if (!this.table || this.table.size < table.byteLength) {
      this.table?.destroy();
      this.table = d.createBuffer({ size: table.byteLength, usage: usage("STORAGE", "COPY_DST"), label: "table" });
    }
    d.queue.writeBuffer(this.table, 0, table);
    d.queue.writeBuffer(this.lut, 0, c.lut);
    this.colors = c;
    this.resolveGroup = d.createBindGroup({ layout: this.resolve.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.counts } }, { binding: 1, resource: { buffer: this.table } },
      { binding: 2, resource: { buffer: this.lut } }, { binding: 3, resource: { buffer: this.resolveUniform } },
      { binding: 4, resource: this.image.createView() }] });
    return c;
  }

  // The last row's window shift (H − 1)·row_ns: the planner's view reaches that far past w1.
  get readoutNs() { return BigInt(this.height - 1) * BigInt(this.rowNs); }

  // Rolling-shutter row time in ns (0: global exposure); a total readout of 2^29 ns or more is refused.
  setRowNs(rowNs) {
    if (!Number.isInteger(rowNs) || rowNs < 0 || this.height * rowNs >= MAX_ROW_TOTAL_NS) {
      throw new GpuError("BS-UI-TOO-WIDE", `row time ${rowNs} ns: H·row_ns must stay below 2^29 ns`);
    }
    this.rowNs = rowNs;
  }

  // The view-clock span the planner must hold to draw the window [w0, w1): the history [w1 − H, w1) for history
  // kinds.
  span(w0, w1) {
    return HISTORY_KINDS.has(this.kind) ? [w1 - historyWindow(this.tauMs).H, w1] : [w0, w1];
  }

  // The colour pass's mode for `lookup`: "auto" (the exact table where there is one: events only when every block is
  // drawn at j = 0, as in UI-2), "table" or "lut" (the f32 formula of the kind).
  mode(lookup, rec) {
    const c = this.colors, own = c.kind === this.kind && c.table !== null;
    switch (this.kind) {
      case "events":
        return own && (lookup === "table" || (lookup === "auto" && rec.coarsest === 0)) ? MODE.events_table
          : MODE.events_lut;
      case "polarity": return own && lookup !== "lut" ? MODE.polarity_table : MODE.polarity_formula;
      case "counts": return own && lookup !== "lut" ? MODE.counts_table : MODE.counts_formula;
      case "timesurface": return MODE.timesurface;
      default: return MODE.decay;
    }
  }

  // PageView rows of draw record `rec` for `kind`: the window and, per kind, the time-surface offsets D = t_b − O split
  // as ts_hi·2^s + ts_lo, or the decay's age base f32(t_ref − t_b) (t_b: the block start on the view clock, §4.4).
  pack(rec, origin, tRef, s, kind = this.kind) {
    const rowNs = this.rowNs, shift = BigInt(s);
    const out = packPages(this.planner.pool, this.region, this.pv.size, rec, (p, i32, u32, f32, o) => {
      u32[o + PV.row_ns] = rowNs;
      if (kind === "timesurface") {
        const D = p.tb - origin, hi = D >> shift;                  // BigInt >> floors
        i32[o + PV.ts_hi] = Number(hi);
        u32[o + PV.ts_lo] = Number(D - (hi << shift));
      } else if (kind === "decay") f32[o + PV.age_base] = Number(tRef - p.tb);
    });
    this.gpu.device.queue.writeBuffer(this.pv, 0, out.words);
    return out.groups;
  }

  // Draws the window [w0, w1) (BigInt ns, view clock); history kinds draw [w1 − H, w1) with t_ref = w1. Returns the
  // planner's draw record (plus the colour mode).
  draw(w0, w1, { lookup = "auto" } = {}) {
    const { gpu, planner, width: W, height: H, kind } = this, d = gpu.device, history = HISTORY_KINDS.has(kind);
    const hw = history ? historyWindow(this.tauMs) : { tauNs: 0, H: 0n, s: 0 };
    const origin = history ? w1 - hw.H : w0;
    const rec = planner.draw(this.stream, origin, w1, this.readoutNs, { exact: history });
    const groups = this.pack(rec, origin, w1, hw.s);
    const frame = new ArrayBuffer(16);
    new Uint32Array(frame).set([W, H, hw.s]);
    new Float32Array(frame)[3] = hw.tauNs;
    d.queue.writeBuffer(this.frameUniform, 0, frame);
    const c = this.colors, mode = this.mode(lookup, rec), r = new ArrayBuffer(96);
    const ru = new Uint32Array(r), ri = new Int32Array(r), rf = new Float32Array(r);
    ru.set([W, H, 0, mode]);
    ri[4] = c.k;
    rf.set([c.clip, c.gamma, kind === "decay" ? 2 ** -10 : 1], 5);
    ru.set([Number(hw.H >> BigInt(hw.s)), Number(hw.H & ((1n << BigInt(hw.s)) - 1n)), hw.s], 8);
    rf[11] = hw.tauNs;
    rf.set([...linearRgb(c.background), ...linearRgb(c.on), ...linearRgb(c.off)], 12);
    d.queue.writeBuffer(this.resolveUniform, 0, r);
    const submit = () => {
      const enc = d.createCommandEncoder();
      enc.clearBuffer(this.counts);
      const pass = enc.beginComputePass();
      pass.setPipeline(this.kernels[history ? kind : "events"]);
      dispatchPages(pass, this.groups[history ? kind : "events"], groups);
      pass.setPipeline(this.resolve);
      pass.setBindGroup(0, this.resolveGroup);
      pass.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 16));
      pass.end();
      if (this.canvas) this.encodePresent(enc, history && rec.missing > 0);
      d.queue.submit([enc.finish()]);
    };
    if (this.firstFrame === null) {                                                 // §4.4 shader hygiene
      this.firstFrame = scoped(gpu, "first frame", submit);
      this.firstFrame.catch((e) => gpu.report(`${e.code ?? "BS-UI-SHADER"}: ${e.message}`));   // to the status line
    } else submit();
    Object.assign(rec, { mode, kind, H: hw.H, s: hw.s });
    this.lastDraw = rec;
    return rec;
  }

  // The automatic clip (§4.5), as viz.compare.auto_clip on Timeline.from_range(t0, t1, length): up to 16 evenly
  // spaced windows of the timeline, the activity |ON − OFF| ("events", "decay") or ON + OFF of their active pixels in
  // a 1024-bin GPU histogram, the 99th percentile on the CPU, at least 1. The planner's view moves to each window and
  // waits for its blocks; the result equals viz's when every window is drawn at j = 0 and the percentile's neighbours
  // lie below 1023 (`exact`). The windows are accumulated as the "events" kind without changing the view's own kind,
  // which the screen may change meanwhile. The histogram's workgroups form a 2D grid (one dimension holds at most
  // maxComputeWorkgroupsPerDimension of them).
  async autoClip(t0, t1, length, { kind = this.kind } = {}) {
    const { gpu, planner, width: W, height: H } = this, d = gpu.device;
    if (length > t1 - t0) length = t1 - t0;
    const n = length > 0n ? Number((t1 - length - t0) / length) + 1 : 0;
    const m = Math.min(AUTO_CLIP_WINDOWS, n);
    const picks = [...new Set(Array.from({ length: m }, (_, i) => (i === m - 1 ? n - 1
      : Math.trunc(i * ((n - 1) / (m - 1))))))];
    const groups = Math.ceil((W * H) / 256), gx = Math.min(groups, d.limits.maxComputeWorkgroupsPerDimension);
    d.queue.writeBuffer(this.histUniform, 0, new Uint32Array([W, H, 0, kind === "events" || kind === "decay" ? 0 : 1]));
    let coarsest = 0, missing = 0, first = true;
    for (const k of picks) {
      const a = t0 + BigInt(k) * length, b = a + length;
      planner.setView(a, b, b, this.readoutNs);
      await planner.idle();
      if (this.destroyed) return null;
      const rec = planner.draw(this.stream, a, b, this.readoutNs);
      coarsest = Math.max(coarsest, rec.coarsest);
      missing += rec.missing;
      const pages = this.pack(rec, a, b, 0, "events");
      d.queue.writeBuffer(this.frameUniform, 0, new Uint32Array([W, H, 0, 0]));
      const enc = d.createCommandEncoder();
      if (first) enc.clearBuffer(this.hist);
      first = false;
      enc.clearBuffer(this.counts);
      const pass = enc.beginComputePass();
      pass.setPipeline(this.kernels.events);
      dispatchPages(pass, this.groups.events, pages);
      pass.setPipeline(this.histogram);
      pass.setBindGroup(0, this.histGroup);
      pass.dispatchWorkgroups(gx, Math.ceil(groups / gx));
      pass.end();
      d.queue.submit([enc.finish()]);
    }
    const p = picks.length ? percentile99(new Uint32Array(await readBuffer(gpu, this.hist, 0, HISTOGRAM_BINS * 4)))
      : { value: null, n: 0, exact: true };
    return { clip: Math.max(1, p.value ?? 1), windows: picks, values: p.n, exact: p.exact && !coarsest && !missing,
      coarsest, missing };
  }

  // Image pixel coordinates of canvas pixel (px, py): floor(p · scale + offset).
  transform() {
    const cw = this.canvas.width, ch = this.canvas.height;
    const s = Math.min(cw / this.width, ch / this.height) * this.zoom;
    return { scale: 1 / s, offset: [this.center[0] - cw / (2 * s), this.center[1] - ch / (2 * s)] };
  }

  imagePixel(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect(), dpr = this.canvas.width / Math.max(1, r.width);
    const { scale, offset } = this.transform();
    const x = Math.floor((clientX - r.left) * dpr * scale + offset[0]);
    const y = Math.floor((clientY - r.top) * dpr * scale + offset[1]);
    return x >= 0 && y >= 0 && x < this.width && y < this.height ? [x, y] : [null, null];
  }

  encodePresent(enc, hatch = false) {
    const { scale, offset } = this.transform(), bg = this.colors.background.map((v) => v / 255);
    this.gpu.device.queue.writeBuffer(this.presentUniform, 0, new Float32Array([scale, scale, ...offset, this.width,
      this.height, hatch ? 1 : 0, 0, ...bg, 1]));
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(),
      loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(this.present);
    pass.setBindGroup(0, this.presentGroup);
    pass.draw(3);
    pass.end();
  }

  // The last drawn image as RGBA bytes (H·W·4), exactly as the colour pass wrote them.
  readImage() { return readTexture(this.gpu, this.image, this.width, this.height); }

  // The last drawn planes: OFF and ON counts (weighted by 2^j), time-surface keys in plane 0, or decayed sums.
  async readCounts() {
    return new Uint32Array(await readBuffer(this.gpu, this.counts, 0, this.counts.size));
  }

  // Frees the view's buffers; safe to call again (for buffers made after an earlier call, during init()).
  destroy() {
    this.destroyed = true;
    for (const b of [this.counts, this.pv, this.frameUniform, this.resolveUniform, this.lut, this.table, this.pixel,
      this.presentUniform, this.image, this.hist, this.histUniform]) b?.destroy();
    this.context?.unconfigure();
  }

  // Planes 0 and 1 at (x, y): a 16-byte readback (the inspector, §7).
  async readPixel(x, y) {
    const d = this.gpu.device, i = (y * this.width + x) * 4, plane = this.width * this.height * 4;
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.counts, i, this.pixel, 0, 4);
    enc.copyBufferToBuffer(this.counts, plane + i, this.pixel, 4, 4);
    d.queue.submit([enc.finish()]);
    await this.pixel.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(this.pixel.getMappedRange().slice(0));
    this.pixel.unmap();
    return [v[0], v[1]];
  }

  // The inspector's text for planes [a, b] at (x, y), as the last draw's kind reads them.
  describePixel(x, y, [a, b]) {
    const rec = this.lastDraw, at = `x ${x}, y ${y}: `;
    if (rec?.kind === "timesurface") {
      if (!a) return `${at}no event in the last 5τ`;
      const age = Number(((rec.H >> BigInt(rec.s)) - BigInt((a >>> 1) - 1)) << BigInt(rec.s)) +
        Number(rec.H & ((1n << BigInt(rec.s)) - 1n));
      return `${at}latest event ${a & 1 ? "ON" : "OFF"}, age ${(age / 1e6).toFixed(3)} ms`;
    }
    if (rec?.kind === "decay") return `${at}decayed ON ${(b / 1024).toFixed(3)}, OFF ${(a / 1024).toFixed(3)}`;
    const sampled = rec?.coarsest > 0 ? " (estimate: sampled blocks are weighted 2^j)" : "";
    const net = b - a;
    return `${at}ON ${b}, OFF ${a}, net ${net > 0 ? "+" : ""}${net}${sampled}`;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The viewer screen: #/view?path=<absolute path or rid>
// ---------------------------------------------------------------------------------------------------------------

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};
const theme = () => (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "white");
const WINDOWS_MS = [1, 10, 33, 100, 1000];
const fmtTime = (ns) => `${(Number(ns) / 1e9).toFixed(6)} s`;
const KIND_LABELS = { events: "events", polarity: "polarity", counts: "counts", timesurface: "time surface",
  decay: "decay", cloud: "(x, y, t) cloud" };
const IMAGE_STYLE = (info) => `display: block; max-width: 100%; width: auto; aspect-ratio: ${info.width} / ` +
  `${info.height}; height: min(calc(100vh - 290px), calc((100vw - 48px) * ${info.height / info.width}));`;

// The badge beside the controls: what the image holds (draw record `rec`, the planner's `overfull`; `hatched`: a
// history view, whose blocks still loading are hatched), and whether that is an error. A history view is drawn at
// j = 0 only, so it overfills the pool when its [t_ref − 5τ, t_ref) holds more pages of events than the pool has.
export function badgeText(rec, overfull = null, hatched = false) {
  if (overfull && hatched) {
    return { error: true, text: `BS-UI-GPU-OOM: the history [t_ref − 5τ, t_ref) needs ${overfull.pages} pages of ` +
      `events and the pool holds ${overfull.capacity}; ${rec.missing} block(s) not drawn: shorten τ` };
  }
  if (overfull) {
    return { error: true, text: `BS-UI-GPU-OOM: the window meets ${overfull.blocks} blocks with events and the pool ` +
      `holds ${overfull.capacity} pages, one block per page; ${rec.missing} block(s) not drawn: shorten the window` };
  }
  if (rec.failed) return { error: true, text: `${rec.failed} block(s) failed to load` };
  return { error: false, text: rec.missing ? `loading ${rec.missing} block(s)${hatched ? " (hatched)" : ""}`
    : rec.coarsest ? `sampled 1/${2 ** rec.coarsest}` : "all events" };
}

// Tab state (§4.4 "Device and pool"): one device and pool per tab, made at the first open and reused; one open
// viewer. Opens run one at a time, and a newer open (or leaving #/view) supersedes the older ones.
let channel = null;                               // a second viewer tab is warned (BroadcastChannel)
let warnTab = () => {};
let reportGpu = () => {};                         // GPU errors and device loss: the mounted viewer's status line
let tabReady = null;                              // Promise<{gpu, pool}>
let viewer = null;                                // the open viewer: {close()}
let opening = Promise.resolve();                  // the queue of opens
let openSeq = 0;                                  // the latest open; older ones give up

// The tab says "hello" on the "ballasim-viewer" channel once, and answers others; `warn` (the mounted screen's status
// line: the viewer's or the compare screen's) shows that another tab also holds a viewer pool (§4.4).
export function announce(warn) {
  warnTab = warn;
  if (channel || !("BroadcastChannel" in window)) return;
  channel = new BroadcastChannel("ballasim-viewer");
  channel.onmessage = (e) => {
    if (e.data === "hello") channel.postMessage("here");
    warnTab("Another tab also holds a viewer pool on the GPU; close one of them to free its memory.");
  };
  channel.postMessage("hello");
}

const showError = (status, text) => status.replaceChildren(el("span", { className: "error", textContent: text }));

function closeViewer() {
  viewer?.close();
  viewer = null;
}

// The open viewer's view (tests).
export const openedView = () => viewer?.view ?? null;

// The tab's device and pool. The pool is sized from /meta read when it is made, not at page load: the GPU may have
// become busy, or its free memory changed, since (unreadable: 512 MiB). A lost device is replaced, with a new pool
// sized likewise, and the viewer on it is closed.
async function tabGpu() {
  const tab = await tabReady?.catch(() => null);
  if (tab && !tab.gpu.lost) return tab;
  if (tab) {
    closeViewer();
    tab.pool.destroy();
  }
  tabReady = (async () => {
    const gpu = await openGpu({ onError: (text) => reportGpu(text) });
    try {
      const meta = await api("/meta").catch(() => null);
      return { gpu, pool: await Pool.create(gpu, poolBytes(meta)) };
    } catch (e) {
      gpu.device.destroy();
      throw e;
    }
  })();
  return tabReady;
}

// The compare screen (compare.js) shares the tab's device, pool and viewer slot: `self` ({close()}) becomes the open
// viewer once the device and pool are ready, the viewer open before it closed first (so the pool is free), and GPU
// errors go to `report`. Returns {gpu, pool}.
export async function claimTab(self, report) {
  reportGpu = report;
  const tab = await tabGpu();
  closeViewer();
  viewer = self;
  return tab;
}

// `self` leaves the viewer slot (it closed itself). Leaving the screen is unmountViewer's, as for the viewer.
export function releaseTab(self) {
  if (viewer === self) viewer = null;
}

// Resolves when every open asked for so far has finished (tests).
export const viewerSettled = () => opening;

// Leaving #/view (main.js): queued and running opens give up, and the open viewer stops and frees its buffers.
export function unmountViewer() {
  openSeq++;
  closeViewer();
  reportGpu = () => {};
}

// Opens POST /streams `body` in `stage` after the opens queued before it; `then()` runs once the stream is open.
// `tracks`: {run, camera} whose timeline tracks the strip draws (UI-7).
function queueOpen(stage, status, body, { then = null, tracks = null } = {}) {
  const seq = ++openSeq;
  status.textContent = "Opening…";
  opening = opening.then(async () => {
    if (seq !== openSeq) return;                                  // a newer open (or leaving the viewer) came first
    try {
      const info = await api("/streams", { method: "POST", body });
      if (seq !== openSeq) return;
      then?.();
      await openViewer(stage, status, info, seq, tracks);
    } catch (err) {
      if (seq === openSeq) showError(status, `${err.code ?? "error"}: ${err.message}`);
    }
  });
}

// A run's camera (#/view?run=<rid>&camera=<name>, or the results page's events tab, `embedded`), with the run's
// tracks on the timeline.
function mountRunViewer(container, run, camera, embedded) {
  const status = el("p", { className: "muted" });
  const stage = el("div");
  container.replaceChildren(...(embedded ? [] : [el("h2", { textContent: `Viewer: ${camera}` })]), status, stage);
  announce((text) => status.replaceChildren(el("span", { className: "warn", textContent: text })));
  reportGpu = (text) => showError(status, text);
  queueOpen(stage, status, { source: { run, camera }, polarity: "auto" }, { tracks: { run, camera } });
}

export function mountViewer(container, { query, embedded = false }) {
  container.classList.add("wide");
  if (query?.get("run") && query?.get("camera")) return mountRunViewer(container, query.get("run"), query.get("camera"),
    embedded);
  const path = el("input", { name: "path", size: 60, value: query?.get("path") ?? "",
    placeholder: "event file or EVIMO2 sequence: absolute path in a root" });
  const polarity = el("select", { name: "polarity" }, ...["auto", "normal", "inverted"].map((v) =>
    el("option", { value: v, textContent: v })));
  const form = el("form", {}, "Source ", path, " polarity ", polarity, " ", el("button", { textContent: "Open" }));
  const status = el("p", { className: "muted" });
  const stage = el("div");
  container.replaceChildren(el("h2", { textContent: "Viewer" }), form, status, stage);
  announce((text) => status.replaceChildren(el("span", { className: "warn", textContent: text })));
  reportGpu = (text) => showError(status, text);
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const source = path.value.trim();
    queueOpen(stage, status, { source: { path: source }, polarity: polarity.value },
      { then: () => history.replaceState(null, "", `#/view?path=${encodeURIComponent(source)}`) });
  });
  if (path.value) form.requestSubmit();
}

// The controls both viewers share: kind, window length, clip, γ, τ.
function controls(kinds) {
  const c = {
    kind: el("select", { name: "kind" }, ...kinds.map((k) => el("option", { value: k, textContent: KIND_LABELS[k] }))),
    length: el("select", {}, ...WINDOWS_MS.map((ms) => el("option", { value: ms, textContent: `${ms} ms` })),
      el("option", { value: "all", textContent: "whole stream" })),
    clip: el("input", { name: "clip", type: "number", min: "0.1", step: "any", value: "3", size: 5 }),
    gamma: el("input", { name: "gamma", type: "number", min: "0.1", max: "4", step: "0.05", value: "0.5", size: 5 }),
    tau: el("input", { name: "tau", type: "number", min: "0.001", step: "any", value: "20", size: 6 }),
  };
  c.tauLabel = el("span", { hidden: true }, " τ ", c.tau, " ms");
  c.length.value = "33";
  return c;
}

// Opens stream `info` in `stage`, replacing the open viewer. Runs alone (the queue of opens); gives up, freeing what it
// made, when a newer open or leaving #/view supersedes it. Without WebGPU (or a pool), PNG panels (§4.11).
async function openViewer(stage, status, info, seq, tracks = null) {
  let tab;
  try {
    tab = await tabGpu();
  } catch (e) {
    if (seq !== openSeq) return;
    if (e.code === "BS-UI-WEBGPU" || e.code === "BS-UI-GPU-OOM") openFallback(stage, status, info, e, tracks);
    else showError(status, `${e.code ?? "error"}: ${e.message}`);
    return;
  }
  if (seq !== openSeq) return;
  closeViewer();                                  // its planner stops before the pool is handed over
  const { gpu, pool } = tab;
  pool.reset();
  const life = new AbortController(), signal = life.signal;
  let closed = false, frameWanted = false, timeline = null, resize = null, inspector = null, cloud = null;
  let kind = "events", busy = false;
  const requestFrame = () => {
    if (closed || frameWanted) return;
    frameWanted = true;
    requestAnimationFrame(render);
  };
  const planner = new Planner(pool, { onChange: requestFrame });
  const stream = planner.add(info);
  const canvas = el("canvas", { className: "image" });
  canvas.style.cssText = IMAGE_STYLE(info);
  const cloudCanvas = el("canvas", { className: "image cloud" });
  cloudCanvas.style.cssText = "display: block; width: 100%; height: min(calc(100vh - 290px), 70vw); cursor: grab;";
  const strip = el("canvas", { className: "timeline" });
  strip.style.cssText = "width: 100%; height: 64px; display: block; margin-top: 6px; cursor: ew-resize;";
  const view = new EventView(gpu, planner, stream, canvas);
  const self = {
    view,
    // Playback, listeners, fetches and frames stop before the views' buffers are freed.
    close() {
      closed = true;
      life.abort();
      resize?.disconnect();
      inspector?.close();
      planner.close();
      view.destroy();
      cloud?.destroy();
      stage.replaceChildren();
    },
  };
  viewer = self;
  const drop = () => {
    if (viewer === self) closeViewer();
    view.destroy();                               // again: buffers made after an earlier close
    cloud?.destroy();
  };
  const superseded = () => {
    if (!closed && seq === openSeq) return false;
    drop();
    return true;
  };
  const fail = (e) => {                           // a half-built viewer is not left open
    drop();
    throw e;
  };
  await view.init().catch(fail);
  if (superseded()) return;
  const c = controls(KINDS);
  const auto = el("button", { type: "button", name: "autoclip", textContent: "Auto" });
  const play = el("button", { type: "button", name: "play", textContent: "Play" });
  const badge = el("span", { className: "muted" });
  const readout = el("span", { className: "muted" });
  const inspectorOut = el("code");
  stage.replaceChildren(el("div", {}, c.kind, " window ", c.length, " clip ", c.clip, " ", auto, " γ ", c.gamma,
    c.tauLabel, " ", play, " ", badge, " ", readout), canvas, strip, el("p", {}, inspectorOut));
  const t0 = BigInt(info.t_start_ns), t1 = BigInt(info.t_end_ns);
  let heads = 0;                                  // the overview's ON shares grow as blocks are built
  const refreshRate = () => {
    if (closed || stream.blocks.size === heads) return;
    heads = stream.blocks.size;
    timeline.load(info.sid).catch(report);
  };
  // pointer moves and playback only request a frame; the frame hands the latest window to the planner (§5.3)
  timeline = new Timeline(strip, { t0, t1, length: 33_000_000n, onChange: requestFrame, signal });
  if (tracks) timeline.loadTracks(tracks.run, tracks.camera).catch(report);
  let planned = "";
  const colors = async () => {
    const col = await view.setColors({ theme: theme(), clip: Number(c.clip.value) || 3,
      gamma: Number(c.gamma.value) || 0.5 });
    if (closed || !col) return;                   // closed, or a later change's colours won
    cloud?.setColors(col);
    timeline.colors = { on: col.on, off: col.off };
    timeline.draw();
    requestFrame();
  };
  await colors().catch(fail);
  if (superseded()) return;
  // The kind: history kinds load every block of [t_ref − H, t_ref) at j = 0 (§4.3 step 7); the cloud draws on its
  // own canvas, made when chosen and freed when left. Changes apply one at a time.
  let kindChange = Promise.resolve();
  const setKind = (k) => (kindChange = kindChange.catch(() => {}).then(() => applyKind(k)));
  const applyKind = async (k) => {
    if (closed) return;
    try {                                         // before anything changes: a refused τ leaves the last valid one
      view.setTau(Number(c.tau.value) || 20);
    } catch {
      c.tau.value = String(view.tauMs);
    }
    if (k === "cloud" && !cloud) {
      const made = new CloudView(gpu, planner, stream, cloudCanvas);
      try {
        await made.init();
      } catch (e) {
        made.destroy();
        throw e;
      }
      if (closed) return made.destroy();
      cloud = made;
      canvas.replaceWith(cloudCanvas);
    } else if (k !== "cloud") {
      if (cloud) {
        cloud.destroy();
        cloud = null;
        cloudCanvas.replaceWith(canvas);
      }
      view.setKind(k);
    }
    kind = k;
    planner.fixedLevel = HISTORY_KINDS.has(k) ? 0 : null;
    planned = "";
    c.tauLabel.hidden = !HISTORY_KINDS.has(k);
    await colors();
  };
  c.kind.addEventListener("change", () => setKind(c.kind.value).catch(report), { signal });
  c.tau.addEventListener("change", () => {
    try {
      view.setTau(Number(c.tau.value));
      planned = "";
      requestFrame();
    } catch (e) {
      c.tau.value = String(view.tauMs);           // the field keeps showing the τ drawn
      report(e);
    }
  }, { signal });
  for (const input of [c.clip, c.gamma]) input.addEventListener("change", () => colors().catch(report), { signal });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => colors().catch(report), { signal });
  c.length.addEventListener("change", () => {
    const span = t1 - t0;
    timeline.set(c.length.value === "all" ? t1 : timeline.playhead,
      c.length.value === "all" ? span : BigInt(Number(c.length.value) * 1e6));
  }, { signal });
  play.addEventListener("click", () => {
    timeline.toggle();
    play.textContent = timeline.playing ? "Pause" : "Play";
  }, { signal });
  auto.addEventListener("click", async () => {      // viz's automatic clip over the timeline's windows (§4.5)
    if (busy) return;
    busy = true;
    badge.textContent = "auto clip…";
    try {
      const r = await view.autoClip(t0, t1, timeline.length, { kind: kind === "cloud" ? "polarity" : kind });
      if (!r || closed) return;
      c.clip.value = String(r.clip);
      auto.title = `99th percentile over ${r.windows.length} window(s)${r.exact ? "" : " (estimated)"}`;
    } catch (e) {
      report(e);
    } finally {
      busy = false;
      planned = "";
    }
    await colors().catch(report);
  }, { signal });
  inspector = new Inspector(view, inspectorOut);
  canvas.addEventListener("pointermove", (e) => inspector.point(...view.imagePixel(e.clientX, e.clientY)),
    { signal });
  canvas.addEventListener("pointerleave", () => inspector.point(null), { signal });
  canvas.addEventListener("wheel", (e) => {                        // spatial zoom about the pointer
    e.preventDefault();
    const [x, y] = view.imagePixel(e.clientX, e.clientY);
    const z = Math.min(64, Math.max(1, view.zoom * 2 ** (e.deltaY > 0 ? -0.25 : 0.25)));
    if (x !== null) view.center = view.center.map((v, i) => [x, y][i] + (v - [x, y][i]) * (view.zoom / z));
    view.zoom = z;
    if (z === 1) view.center = [info.width / 2, info.height / 2];
    requestFrame();
  }, { passive: false, signal });
  let drag = null;                                                  // the cloud: drag orbits, wheel dollies
  cloudCanvas.addEventListener("pointerdown", (e) => {
    cloudCanvas.setPointerCapture(e.pointerId);
    drag = [e.clientX, e.clientY];
  }, { signal });
  cloudCanvas.addEventListener("pointermove", (e) => {
    if (!drag || !cloud) return;
    cloud.orbit(e.clientX - drag[0], e.clientY - drag[1]);
    drag = [e.clientX, e.clientY];
    requestFrame();
  }, { signal });
  cloudCanvas.addEventListener("pointerup", () => { drag = null; }, { signal });
  cloudCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    cloud?.dolly(e.deltaY > 0 ? 1 : -1);
    requestFrame();
  }, { passive: false, signal });
  resize = new ResizeObserver(() => {
    const dpr = devicePixelRatio || 1;
    for (const cv of [canvas, cloudCanvas]) {
      if (!cv.isConnected) continue;
      cv.width = Math.max(1, Math.round(cv.clientWidth * dpr));
      cv.height = Math.max(1, Math.round(cv.clientHeight * dpr));
    }
    timeline.draw();
    requestFrame();
  });
  resize.observe(canvas);
  resize.observe(cloudCanvas);
  function report(e) {
    if (!closed) showError(status, `${e.code ?? "error"}: ${e.message}`);
  }
  let frames = [];
  function render(now) {
    frameWanted = false;
    if (closed || gpu.lost || !view.colors || busy) return;
    const { w0, w1, playhead } = timeline.window, t = performance.now(), shown = kind === "cloud" && cloud;
    if (shown && !cloud.colors) return;
    const [from, to] = shown ? [w0, w1] : view.span(w0, w1), extra = shown ? 0n : view.readoutNs;
    if (planned !== `${from}:${to}:${extra}`) {
      planned = `${from}:${to}:${extra}`;
      planner.setView(from, to, playhead, extra);
      planner.ready.then(refreshRate);
    }
    const rec = shown ? cloud.draw(w0, w1) : view.draw(w0, w1);
    frames = frames.filter((f) => now - f < 1000).concat(now);
    const label = badgeText(rec, planner.overfull, HISTORY_KINDS.has(kind));
    if (label.error) badge.replaceChildren(el("span", { className: "error", textContent: label.text }));
    else badge.textContent = label.text;
    play.textContent = timeline.playing ? "Pause" : "Play";
    readout.textContent = `${fmtTime(w0)} – ${fmtTime(w1)} · pool ${pool.used}/${pool.capacity} pages · ` +
      `${(performance.now() - t).toFixed(1)} ms CPU${timeline.playing ? ` · ${frames.length} fps` : ""}`;
    if (planner.errors.length) report(planner.errors.at(-1));
  }
  status.textContent = `${info.label}: ${info.width}×${info.height}, ${info.n_events} events, ` +
    `${fmtTime(t0)} – ${fmtTime(t1)}${gpu.info?.vendor ? ` · WebGPU ${gpu.info.vendor}` : ""}`;
  timeline.set(timeline.playhead);
}

// The PNG fallback (§4.11): without a WebGPU adapter (or a pool), the 2D kinds come as PNG panels that the server
// renders with ballasim.viz (GET /streams/{sid}/image), one request at a time, the latest window winning. No cloud.
// An empty clip ("auto") is the server's automatic clip of the timeline: one per window length and kind, shared by
// every panel (§4.5).
function openFallback(stage, status, info, why, tracks = null) {
  closeViewer();
  const life = new AbortController(), signal = life.signal;
  let closed = false, url = null, inflight = false, again = false;
  const c = controls(["events", "polarity", "counts", "timesurface"]);
  c.clip.value = "";
  c.clip.placeholder = "auto";
  const readout = el("span", { className: "muted" });
  const img = el("img", { className: "image fallback", alt: "event image rendered by the server" });
  img.style.cssText = IMAGE_STYLE(info) + " image-rendering: pixelated;";
  const strip = el("canvas", { className: "timeline" });
  strip.style.cssText = "width: 100%; height: 64px; display: block; margin-top: 6px; cursor: ew-resize;";
  const resize = new ResizeObserver(() => timeline.draw());
  viewer = {
    close() {
      closed = true;
      life.abort();
      resize.disconnect();
      if (url) URL.revokeObjectURL(url);
      stage.replaceChildren();
    },
  };
  status.replaceChildren(el("span", { className: "warn", textContent: `${why.code}: ${why.message}. ` +
    "The viewer shows PNG panels rendered by the server (no cloud); " }), el("a", { href: "#/check",
    textContent: "Browser check" }), ` · ${info.label}: ${info.width}×${info.height}, ${info.n_events} events`);
  stage.replaceChildren(el("div", {}, c.kind, " window ", c.length, " clip ", c.clip, " γ ", c.gamma, c.tauLabel, " ",
    readout), img, strip);
  const t0 = BigInt(info.t_start_ns), t1 = BigInt(info.t_end_ns);
  const report = (e) => {
    if (!closed) readout.replaceChildren(el("span", { className: "error", textContent: `${e.code ?? "error"}: ` +
      e.message }));
  };
  // One panel request at a time; changes meanwhile ask for one more, of the latest state, after it (also when it
  // fails: the error is shown, then the latest state is asked for). A panel that fails to decode leaves the last one.
  async function load() {
    if (inflight) {
      again = true;
      return;
    }
    inflight = true;
    try {
      do {
        again = false;
        let next = null;
        try {
          const { w0, w1 } = timeline.window, kind = c.kind.value, tau = Number(c.tau.value) || 20;
          const q = new URLSearchParams({ t0_ns: w0, t1_ns: w1, kind, theme: theme(),
            gamma: Number(c.gamma.value) || 0.5, tau_ms: tau });
          if (Number(c.clip.value) > 0) q.set("clip", c.clip.value);
          const r = await apiResponse(`/streams/${info.sid}/image?${q}`);
          const blob = await r.blob();
          if (closed) return;
          next = URL.createObjectURL(blob);
          img.src = next;
          await img.decode();
          if (url) URL.revokeObjectURL(url);
          [url, next] = [next, null];
          const clip = r.headers.get("X-BallaSim-Clip") ?? "";        // none for the time surface
          Object.assign(img.dataset, { w0: String(w0), w1: String(w1), clip, kind, theme: q.get("theme") });
          readout.textContent = `${fmtTime(w0)} – ${fmtTime(w1)} · ${clip ? `clip ${clip}` : `τ ${tau} ms`}`;
        } catch (e) {
          if (next) {
            if (url) img.src = url;
            URL.revokeObjectURL(next);
          }
          report(e);
        }
      } while (again && !closed);
    } finally {
      inflight = false;
    }
  }
  const timeline = new Timeline(strip, { t0, t1, length: 33_000_000n, onChange: () => load(), signal });
  c.kind.addEventListener("change", () => {
    c.tauLabel.hidden = c.kind.value !== "timesurface";
    load();
  }, { signal });
  for (const input of [c.clip, c.gamma, c.tau]) input.addEventListener("change", () => load(), { signal });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => load(), { signal });
  c.length.addEventListener("change", () => {
    timeline.set(c.length.value === "all" ? t1 : timeline.playhead,
      c.length.value === "all" ? t1 - t0 : BigInt(Number(c.length.value) * 1e6));
  }, { signal });
  resize.observe(strip);
  timeline.load(info.sid).catch(report);
  if (tracks) timeline.loadTracks(tracks.run, tracks.camera).catch(report);
  timeline.set(timeline.playhead);
}
