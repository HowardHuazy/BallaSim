// Real vs sim at the viewer level (docs/design/phase3/ui.md §1 G4, §4.5, §4.6, §5.1 "Real vs sim"; UI-8a): the
// comparison panels and the compare screen (#/compare?ref=…&test=…).
// - Panels: the reference ("real") and the test stream ("sim"), each drawn by an EventView into its own counts, then,
//   as viz.compare.Renderer's extra panels (compare.py:329-345), the difference blur_σ((on_t − off_t) −
//   (on_r − off_r))/clip through the diverging LUT with γ, and the ratio round(log2((a_t + ε)/(a_r + ε))/0.125)·0.125
//   over ±2 octaves, a = blur_σ(on + off), ε = 0.25 (mean a_r + mean a_t): a separable Gaussian with cv2.GaussianBlur's
//   kernel size and reflect-101 border (σ 0 for the difference and 2 px for the ratio, as viz), the means from a GPU
//   reduction; and a blink panel that alternates the two streams.
// - Region (/compare/region: none, crop, viewer: the reference's GT, EVIMO2 GT at t + δ) with the clock rule's
//   δ_stream proposal and its BS-UI-CLOCK warnings (§4.6). The mask of the interval that holds the window's centre
//   (BSMK) is outlined on the event-free pixels of every panel, as viz draws mask contours (compare.py:362-381).
// - Viewer statistics (/compare/stats) of the window through the region.

import { api, apiBytes } from "../app/api.js";
import { decodeBSMK } from "../app/wire.js";
import { GpuError, computePipeline, readTexture, renderPipeline, scoped } from "./gpu.js";
import { Planner } from "./planner.js";
import { Timeline } from "./timeline.js";
import { EventView, announce, badgeText, claimTab, releaseTab, unmountViewer } from "./views.js";

export const COMPARE_KINDS = ["events", "polarity", "counts"];
export const DEFAULT_SIGMA = { diff: 0, ratio: 2 };       // viz RenderOptions.sigma_for: diff 0 px, ratio 2 px
export const MAX_SIGMA = 16;
const MAX_TAPS = 2 * Math.round(MAX_SIGMA * 4) + 1;        // cv2's kernel size at MAX_SIGMA
const RATIO_STEP = 0.125, RATIO_RANGE = 2.0;               // octaves (compare.py _RATIO_STEP/_RATIO_RANGE_OCTAVES)
const BLINK_MS = 500;
export const MASK_CHUNK = 16;                              // masks per BSMK request
const usage = (...names) => names.reduce((u, n) => u | GPUBufferUsage[n], 0);

// The Gaussian of cv2.GaussianBlur(image, (0, 0), σ) on a float image: cvRound(σ·4·2 + 1) | 1 taps,
// exp(−x²/(2σ²)) normalised to sum 1 (getGaussianKernel), as float32; σ = 0: no blur.
export function gaussianKernel(sigma) {
  if (!(sigma > 0)) return new Float32Array([1]);
  const n = Math.round(sigma * 8 + 1) | 1, scale = -0.5 / (sigma * sigma), w = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = i - (n - 1) / 2;
    sum += (w[i] = Math.exp(scale * x * x));
  }
  return Float32Array.from(w, (v) => v / sum);
}

const SIGNED_LUT = /* wgsl */`
fn signed_lut(value: f32, gamma: f32) -> u32 {          // viz.render.colorize_signed (as RESOLVE's)
  let v = clamp(value, -1.0, 1.0);
  var index = 255;
  if (v != 0.0) {
    var m = abs(v);
    if (gamma != 1.0) { m = pow(m, gamma); }
    index = i32(round(sign(v) * m * 255.0)) + 255;
  }
  return lut[u32(index)];
}`;

// The fields: plane 0 the net difference (on_t − off_t) − (on_r − off_r), planes 1 and 2 the activities on + off of
// the reference and the test, from the two views' counts (plane 0 OFF, plane 1 ON).
export const FIELDS = /* wgsl */`
struct Size { width: u32, height: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> ref_counts: array<u32>;
@group(0) @binding(1) var<storage, read> test_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> fields: array<f32>;
@group(0) @binding(3) var<uniform> size: Size;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= size.width || g.y >= size.height) { return; }
  let plane = size.width * size.height;
  let i = g.y * size.width + g.x;
  let off_r = ref_counts[i];
  let on_r = ref_counts[i + plane];
  let off_t = test_counts[i];
  let on_t = test_counts[i + plane];
  fields[i] = f32((i32(on_t) - i32(off_t)) - (i32(on_r) - i32(off_r)));
  fields[plane + i] = f32(on_r + off_r);
  fields[2u * plane + i] = f32(on_t + off_t);
}`;

// One direction of the separable Gaussian over planes [plane0, plane0 + z) with taps weights[k0 .. k0 + 2r], OpenCV's
// BORDER_REFLECT_101 (cv2.GaussianBlur's default border).
export const BLUR = /* wgsl */`
struct Blur { width: u32, height: u32, plane0: u32, horizontal: u32, radius: u32, k0: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> b: Blur;
fn reflect101(i: i32, n: i32) -> i32 {                   // -1 -> 1, n -> n - 2 (borderInterpolate)
  if (n == 1) { return 0; }
  var j = i;
  for (var guard = 0; guard < 64; guard++) {
    if (j < 0) { j = -j; } else if (j >= n) { j = 2 * n - 2 - j; } else { break; }
  }
  return clamp(j, 0, n - 1);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= b.width || g.y >= b.height) { return; }
  let base = (b.plane0 + g.z) * b.width * b.height;
  let r = i32(b.radius);
  var acc = 0.0;
  for (var k = -r; k <= r; k++) {
    var x = i32(g.x);
    var y = i32(g.y);
    if (b.horizontal != 0u) { x = reflect101(x + k, i32(b.width)); } else { y = reflect101(y + k, i32(b.height)); }
    acc += weights[b.k0 + u32(k + r)] * src[base + u32(y) * b.width + u32(x)];
  }
  dst[base + g.y * b.width + g.x] = acc;
}`;

// The activity sums (planes 1 and 2 of the blurred fields, workgroup z): partial sums of 256 pixels per workgroup on a
// 2D grid of workgroups, then one workgroup per plane adds the partials.
export const PARTIALS = /* wgsl */`
struct Reduce { plane: u32, groups: u32, gx: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> blurred: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<uniform> r: Reduce;
var<workgroup> scratch: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let gid = wg.y * r.gx + wg.x;
  let i = gid * 256u + li;
  var v = 0.0;
  if (gid < r.groups && i < r.plane) { v = blurred[(1u + wg.z) * r.plane + i]; }
  scratch[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { scratch[li] += scratch[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u && gid < r.groups) { partials[wg.z * r.groups + gid] = scratch[0]; }
}`;

export const TOTALS = /* wgsl */`
struct Reduce { plane: u32, groups: u32, gx: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> totals: array<f32>;
@group(0) @binding(2) var<uniform> r: Reduce;
var<workgroup> scratch: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  var acc = 0.0;
  for (var i = li; i < r.groups; i += 256u) { acc += partials[wg.x * r.groups + i]; }
  scratch[li] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { scratch[li] += scratch[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { totals[wg.x] = scratch[0]; }
}`;

// The difference and ratio images.
export const COLOUR = /* wgsl */`
struct Colour { width: u32, height: u32, pad0: u32, pad1: u32, clip: f32, gamma: f32, pad2: f32, pad3: f32 };
@group(0) @binding(0) var<storage, read> blurred: array<f32>;
@group(0) @binding(1) var<storage, read> totals: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<uniform> c: Colour;
@group(0) @binding(4) var diff_image: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var ratio_image: texture_storage_2d<rgba8unorm, write>;
` + SIGNED_LUT + /* wgsl */`
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= c.width || g.y >= c.height) { return; }
  let plane = c.width * c.height;
  let i = g.y * c.width + g.x;
  textureStore(diff_image, vec2<i32>(g.xy), unpack4x8unorm(signed_lut(blurred[i] / c.clip, c.gamma)));
  let eps = 0.25 * (totals[0] / f32(plane) + totals[1] / f32(plane));
  var q = 0.0;
  if (eps > 0.0) {
    q = round(log2((blurred[2u * plane + i] + eps) / (blurred[plane + i] + eps)) / ${RATIO_STEP}) * ${RATIO_STEP};
  }
  textureStore(ratio_image, vec2<i32>(g.xy), unpack4x8unorm(signed_lut(q / ${RATIO_RANGE}, 1.0)));
}`;

// A panel on its canvas (as kernels.js PRESENT), with the region's mask outlined (render.mask_boundary "valid": inside
// pixels with a 4-neighbour outside, within the image) on pixels of the background colour only.
export const PRESENT_MASK = /* wgsl */`
struct Present { scale: vec2<f32>, offset: vec2<f32>, size: vec2<f32>, has_mask: f32, pad: f32,
  background: vec4<f32>, contour: vec4<f32> };
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<uniform> p: Present;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
fn inside(x: i32, y: i32) -> bool {
  let i = u32(y * i32(p.size.x) + x);
  return ((mask[i >> 5u] >> (i & 31u)) & 1u) != 0u;
}
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let q = floor(pos.xy * p.scale + p.offset);
  if (any(q < vec2<f32>(0.0)) || any(q >= p.size)) { return p.background; }
  let c = textureLoad(image, vec2<i32>(q), 0);
  if (p.has_mask > 0.0) {
    let x = i32(q.x);
    let y = i32(q.y);
    let w = i32(p.size.x);
    let h = i32(p.size.y);
    if (inside(x, y) && ((x > 0 && !inside(x - 1, y)) || (x + 1 < w && !inside(x + 1, y)) ||
        (y > 0 && !inside(x, y - 1)) || (y + 1 < h && !inside(x, y + 1)))) {
      if (all(round(c.rgb * 255.0) == round(p.background.rgb * 255.0))) { return p.contour; }
    }
  }
  return c;
}`;

// Every module, for the shader-hygiene check (B5).
export const MODULES = { compare_fields: FIELDS, compare_blur: BLUR, compare_partials: PARTIALS,
  compare_totals: TOTALS, compare_colour: COLOUR, compare_present: PRESENT_MASK };

export class CompareView {
  // The comparison of two EventViews of one size (no canvases of their own): call draw() after both views drew the
  // frame's window (the queue orders the passes).
  constructor(gpu, ref, test) {
    if (ref.width !== test.width || ref.height !== test.height) {
      throw new GpuError("BS-UI-SOURCE", `the streams' sizes differ (${ref.width}×${ref.height} and ` +
        `${test.width}×${test.height})`);
    }
    Object.assign(this, { gpu, ref, test, width: ref.width, height: ref.height });
    this.sigma = { ...DEFAULT_SIGMA };
    this.hasMask = false;
    this.canvases = new Map();
    this.firstFrame = null;
    this.destroyed = false;
  }

  // Refuses (BS-UI-TOO-WIDE) an image whose three f32 fields exceed one storage binding.
  static checkSize(device, W, H) {
    EventView.checkSize(device, W, H);
    const binding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    if (12 * W * H > binding) {
      throw new GpuError("BS-UI-TOO-WIDE", `${W}×${H}: the comparison panels need 12 bytes per pixel in one ` +
        "storage binding");
    }
  }

  async init(messages = []) {
    const { gpu, width: W, height: H } = this, d = gpu.device;
    CompareView.checkSize(d, W, H);
    this.pipes = {
      fields: await computePipeline(gpu, FIELDS, "compare_fields", messages),
      blur: await computePipeline(gpu, BLUR, "compare_blur", messages),
      partials: await computePipeline(gpu, PARTIALS, "compare_partials", messages),
      totals: await computePipeline(gpu, TOTALS, "compare_totals", messages),
      colour: await computePipeline(gpu, COLOUR, "compare_colour", messages),
    };
    this.present = await renderPipeline(gpu, PRESENT_MASK, "compare_present", gpu.format, messages);
    const buffer = (size, label, ...u) => d.createBuffer({ size, label, usage: usage(...u) });
    const planes = 3 * W * H * 4;
    this.fields = buffer(planes, "compare fields", "STORAGE");
    this.tmp = buffer(planes, "compare blur", "STORAGE");
    this.blurred = buffer(planes, "compare blurred", "STORAGE", "COPY_SRC");
    this.groups = Math.ceil((W * H) / 256);
    this.gx = Math.min(this.groups, d.limits.maxComputeWorkgroupsPerDimension);
    this.partials = buffer(2 * this.groups * 4, "compare partials", "STORAGE");
    this.totals = buffer(16, "compare totals", "STORAGE", "COPY_SRC");
    this.weights = buffer(2 * MAX_TAPS * 4, "compare weights", "STORAGE", "COPY_DST");
    this.sizeUniform = buffer(16, "compare size", "UNIFORM", "COPY_DST");
    this.blurUniforms = [0, 1, 2, 3].map(() => buffer(32, "compare blur", "UNIFORM", "COPY_DST"));
    this.reduceUniform = buffer(16, "compare reduce", "UNIFORM", "COPY_DST");
    this.colourUniform = buffer(32, "compare colour", "UNIFORM", "COPY_DST");
    this.mask = buffer(Math.max(1, Math.ceil((W * H) / 32)) * 4, "compare mask", "STORAGE", "COPY_DST");
    const texture = (label) => d.createTexture({ size: [W, H], format: "rgba8unorm", label,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.images = { ref: this.ref.image, test: this.test.image, diff: texture("difference"), ratio: texture("ratio") };
    d.queue.writeBuffer(this.sizeUniform, 0, new Uint32Array([W, H, 0, 0]));
    d.queue.writeBuffer(this.reduceUniform, 0, new Uint32Array([W * H, this.groups, this.gx, 0]));
    const group = (pipe, ...resources) => d.createBindGroup({ layout: pipe.getBindGroupLayout(0),
      entries: resources.map((r, binding) => ({ binding, resource: r.createView ? r.createView() : { buffer: r } })) });
    const p = this.pipes;
    this.bind = {
      fields: group(p.fields, this.ref.counts, this.test.counts, this.fields, this.sizeUniform),
      blur: [group(p.blur, this.fields, this.tmp, this.weights, this.blurUniforms[0]),      // horizontal: difference
        group(p.blur, this.fields, this.tmp, this.weights, this.blurUniforms[1]),           // horizontal: activities
        group(p.blur, this.tmp, this.blurred, this.weights, this.blurUniforms[2]),          // vertical: difference
        group(p.blur, this.tmp, this.blurred, this.weights, this.blurUniforms[3])],         // vertical: activities
      partials: group(p.partials, this.blurred, this.partials, this.reduceUniform),
      totals: group(p.totals, this.partials, this.totals, this.reduceUniform),
      colour: group(p.colour, this.blurred, this.totals, this.ref.lut, this.colourUniform, this.images.diff,
        this.images.ratio),
    };
    this.setBlur(this.sigma);
  }

  // σ in px of the difference and the ratio (0: none), at most MAX_SIGMA.
  setBlur({ diff = this.sigma.diff, ratio = this.sigma.ratio } = {}) {
    for (const s of [diff, ratio]) {
      if (!(s >= 0 && s <= MAX_SIGMA)) throw new GpuError("BS-UI-REQUEST", `σ = ${s} px: need 0 to ${MAX_SIGMA}`);
    }
    this.sigma = { diff, ratio };
    const kd = gaussianKernel(diff), kr = gaussianKernel(ratio), d = this.gpu.device;
    const w = new Float32Array(2 * MAX_TAPS);
    w.set(kd, 0);
    w.set(kr, MAX_TAPS);
    d.queue.writeBuffer(this.weights, 0, w);
    const { width: W, height: H } = this;
    // [plane0, horizontal, radius, k0] per pass: H difference, H activities, V difference, V activities
    [[0, 1, kd, 0], [1, 1, kr, MAX_TAPS], [0, 0, kd, 0], [1, 0, kr, MAX_TAPS]].forEach(([p0, hz, k, k0], i) => {
      d.queue.writeBuffer(this.blurUniforms[i], 0, new Uint32Array([W, H, p0, hz, (k.length - 1) / 2, k0, 0, 0]));
    });
  }

  // The difference and ratio images of the views' last draws, with their colours (clip, γ, LUT).
  draw() {
    const { gpu, width: W, height: H } = this, d = gpu.device, c = this.ref.colors;
    const u = new ArrayBuffer(32);
    new Uint32Array(u).set([W, H]);
    new Float32Array(u).set([c.clip, c.gamma], 4);
    d.queue.writeBuffer(this.colourUniform, 0, u);
    const tiles = [Math.ceil(W / 16), Math.ceil(H / 16)];
    const submit = () => {
      const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
      pass.setPipeline(this.pipes.fields);
      pass.setBindGroup(0, this.bind.fields);
      pass.dispatchWorkgroups(...tiles);
      pass.setPipeline(this.pipes.blur);
      this.bind.blur.forEach((g, i) => {
        pass.setBindGroup(0, g);
        pass.dispatchWorkgroups(...tiles, i % 2 ? 2 : 1);
      });
      pass.setPipeline(this.pipes.partials);
      pass.setBindGroup(0, this.bind.partials);
      pass.dispatchWorkgroups(this.gx, Math.ceil(this.groups / this.gx), 2);
      pass.setPipeline(this.pipes.totals);
      pass.setBindGroup(0, this.bind.totals);
      pass.dispatchWorkgroups(2);
      pass.setPipeline(this.pipes.colour);
      pass.setBindGroup(0, this.bind.colour);
      pass.dispatchWorkgroups(...tiles);
      pass.end();
      d.queue.submit([enc.finish()]);
    };
    if (this.firstFrame === null) {
      this.firstFrame = scoped(gpu, "first comparison", submit);
      this.firstFrame.catch((e) => gpu.report(`${e.code ?? "BS-UI-SHADER"}: ${e.message}`));
    } else submit();
  }

  // The region mask to outline (Uint32Array of ⌈W·H/32⌉ words, BSMK layout), or null for none.
  setMask(words) {
    this.hasMask = Boolean(words);
    if (words) this.gpu.device.queue.writeBuffer(this.mask, 0, words);
  }

  // Shows panels: [[canvas, "ref" | "test" | "diff" | "ratio"], …], each at zoom 1, centred, outlined by the mask.
  showPanels(list) {
    const d = this.gpu.device, c = this.ref.colors, enc = d.createCommandEncoder();
    const bg = c.background.map((v) => v / 255), contour = c.contour.map((v) => v / 255);
    for (const [canvas, panel] of list) {
      let s = this.canvases.get(canvas);
      if (!s) {
        const context = canvas.getContext("webgpu");
        context.configure({ device: d, format: this.gpu.format, alphaMode: "opaque" });
        const uniform = d.createBuffer({ size: 64, usage: usage("UNIFORM", "COPY_DST") });
        const groups = {};
        for (const [name, image] of Object.entries(this.images)) {
          groups[name] = d.createBindGroup({ layout: this.present.getBindGroupLayout(0), entries: [
            { binding: 0, resource: image.createView() }, { binding: 1, resource: { buffer: uniform } },
            { binding: 2, resource: { buffer: this.mask } }] });
        }
        s = { context, uniform, groups };
        this.canvases.set(canvas, s);
      }
      const cw = canvas.width, ch = canvas.height, k = Math.min(cw / this.width, ch / this.height);
      d.queue.writeBuffer(s.uniform, 0, new Float32Array([1 / k, 1 / k, this.width / 2 - cw / (2 * k),
        this.height / 2 - ch / (2 * k), this.width, this.height, this.hasMask ? 1 : 0, 0, ...bg, 1, ...contour, 1]));
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: s.context.getCurrentTexture().createView(),
        loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
      pass.setPipeline(this.present);
      pass.setBindGroup(0, s.groups[panel]);
      pass.draw(3);
      pass.end();
    }
    d.queue.submit([enc.finish()]);
  }

  // RGBA bytes (H·W·4) of a panel's image as last drawn.
  readImage(panel) { return readTexture(this.gpu, this.images[panel], this.width, this.height); }

  destroy() {
    this.destroyed = true;
    for (const b of [this.fields, this.tmp, this.blurred, this.partials, this.totals, this.weights,
      this.sizeUniform, ...(this.blurUniforms ?? []), this.reduceUniform, this.colourUniform, this.mask,
      this.images?.diff, this.images?.ratio]) b?.destroy();
    for (const s of this.canvases.values()) {
      s.uniform.destroy();
      s.context.unconfigure();
    }
    this.canvases.clear();
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The compare screen: #/compare?ref=<path>&test=<path>
// ---------------------------------------------------------------------------------------------------------------

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};
const theme = () => (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "white");
const WINDOWS_MS = [1, 10, 33, 100, 1000];
const fmtTime = (ns) => `${(Number(ns) / 1e9).toFixed(6)} s`;
const showError = (status, text) => status.replaceChildren(el("span", { className: "error", textContent: text }));
const PANELS = [["ref", "real"], ["test", "sim"], ["diff", "sim − real"], ["ratio", "sim / real (log2, ±2)"],
  ["blink", "blink"]];

let openSeq = 0;                                  // the latest open; older ones give up
let opening = Promise.resolve();                  // the queue of opens
let current = null;                               // the open comparison (tests)

// Resolves when every open asked for so far has finished; the open comparison (tests).
export const compareSettled = () => opening;
export const openedCompare = () => current;

// Leaving #/compare (main.js): queued and running opens give up; the open comparison stops and frees its buffers.
export function unmountCompare() {
  openSeq++;
  unmountViewer();
}

export function mountCompare(container, { query }) {
  container.classList.add("wide");
  const path = (name, hint) => el("input", { name, size: 56, value: query?.get(name) ?? "", placeholder: hint });
  const polarity = (name) => el("select", { name }, ...["auto", "normal", "inverted"].map((v) =>
    el("option", { value: v, textContent: v })));
  const ref = path("ref", "real: an EVIMO2 sequence or an event file (absolute path in a root)");
  const test = path("test", "sim: an event file, e.g. <run>/events/<camera>.h5");
  const refPolarity = polarity("ref_polarity"), testPolarity = polarity("test_polarity");
  const form = el("form", {}, el("div", {}, "Real ", ref, " polarity ", refPolarity),
    el("div", {}, "Sim ", test, " polarity ", testPolarity, " ", el("button", { textContent: "Compare" })));
  const status = el("p", { className: "muted" });
  const stage = el("div");
  container.replaceChildren(el("h2", { textContent: "Real vs sim" }), form, status, stage);
  announce((text) => status.replaceChildren(el("span", { className: "warn", textContent: text })));
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const seq = ++openSeq, sources = [ref.value.trim(), test.value.trim()];
    const modes = [refPolarity.value, testPolarity.value];
    status.textContent = "Opening…";
    opening = opening.then(async () => {
      if (seq !== openSeq) return;
      try {
        const infos = [];
        for (let i = 0; i < 2; i++) {
          infos.push(await api("/streams", { method: "POST", body: { source: { path: sources[i] },
            polarity: modes[i] } }));
        }
        if (seq !== openSeq) return;
        history.replaceState(null, "", `#/compare?ref=${encodeURIComponent(sources[0])}&test=` +
          encodeURIComponent(sources[1]));
        await openCompare(stage, status, infos, seq);
      } catch (err) {
        if (seq === openSeq) showError(status, `${err.code ?? "error"}: ${err.message}`);
      }
    });
  });
  if (ref.value && test.value) form.requestSubmit();
}

// The region's intervals as BigInt arrays, and the index of the one that holds t (−1: none).
function intervalsOf(region) {
  return { lo: region.intervals.map((i) => BigInt(i.t0_ns)), hi: region.intervals.map((i) => BigInt(i.t1_ns)) };
}
export function maskAt(iv, t) {
  let a = 0, b = iv.lo.length;                    // the last interval with lo <= t
  while (a < b) {
    const m = (a + b) >> 1;
    if (iv.lo[m] <= t) a = m + 1;
    else b = m;
  }
  return a > 0 && t < iv.hi[a - 1] ? a - 1 : -1;
}

// The masks of a region on the client, fetched in chunks of MASK_CHUNK (BSMK, /compare/region/{id}/masks) by
// fetchChunk(m0, m1) → ArrayBuffer. Each chunk is asked for once: onLoad() when it arrives; a failure is reported once
// (onError) and not retried, so Apply asks for the region again; a region the server no longer holds (404: evicted
// from its registry, at most 32 regions and 512 MiB) stops every request and calls onGone(error) once.
export class RegionMasks {
  constructor(region, { fetchChunk, onLoad = () => {}, onError = () => {}, onGone = () => {} }) {
    Object.assign(this, { region, fetchChunk, onLoad, onError, onGone });
    this.masks = new Map();
    this.chunks = new Map();                      // m0 → "loading" | "loaded" | "failed"
    this.closed = false;
  }

  // The words of mask m (Uint32Array, ⌈W·H/32⌉), or null while its chunk loads or after it failed.
  get(m) {
    const words = this.masks.get(m);
    if (words || this.closed) return words ?? null;
    const m0 = Math.floor(m / MASK_CHUNK) * MASK_CHUNK;
    if (this.chunks.has(m0)) return null;
    this.chunks.set(m0, "loading");
    this.fetchChunk(m0, Math.min(this.region.n_masks, m0 + MASK_CHUNK)).then((buf) => {
      if (this.closed) return;
      const msg = decodeBSMK(buf), n = msg.wordsPerMask;
      for (let i = 0; i < msg.n; i++) this.masks.set(m0 + i, msg.words.slice(i * n, (i + 1) * n));
      this.chunks.set(m0, "loaded");
      this.onLoad();
    }).catch((e) => {
      if (this.closed) return;
      this.chunks.set(m0, "failed");
      if (e?.status === 404) {
        this.closed = true;
        this.onGone(e);
      } else this.onError(e);
    });
    return null;
  }

  close() {
    this.closed = true;
  }
}

async function openCompare(stage, status, [refInfo, testInfo], seq) {
  if (refInfo.width !== testInfo.width || refInfo.height !== testInfo.height) {
    showError(status, `BS-UI-SOURCE: the streams' sizes differ (${refInfo.width}×${refInfo.height} and ` +
      `${testInfo.width}×${testInfo.height}); comparison panels need one size`);
    return;
  }
  const life = new AbortController(), signal = life.signal;
  let closed = false, frameWanted = false, planner = null, refView = null, testView = null, compare = null;
  let resize = null, blink = null, timeline = null;
  const self = {
    close() {
      closed = true;
      life.abort();
      clearInterval(blink);
      resize?.disconnect();
      planner?.close();
      refView?.destroy();
      testView?.destroy();
      compare?.destroy();
      stage.replaceChildren();
      releaseTab(self);
      if (current === self) current = null;
    },
  };
  const report = (e) => {
    if (!closed) showError(status, typeof e === "string" ? e : `${e.code ?? "error"}: ${e.message}`);
  };
  let tab;
  try {
    tab = await claimTab(self, report);
  } catch (e) {
    if (seq === openSeq) showError(status, `${e.code ?? "error"}: ${e.message}. The comparison panels need WebGPU.`);
    return;
  }
  const superseded = () => {
    if (!closed && seq === openSeq) return false;
    self.close();
    return true;
  };
  if (superseded()) return;
  current = self;
  const { gpu, pool } = tab;
  pool.reset();
  const requestFrame = () => {
    if (closed || frameWanted) return;
    frameWanted = true;
    requestAnimationFrame(render);
  };
  // the region and the clock rule
  const ask = (mode, crop) => api("/compare/region", { method: "POST", body: { ref: refInfo.sid,
    test: testInfo.sid, mode, ...(crop ? { crop } : {}) } });
  let region, note = "";
  try {
    region = await ask("viewer");
  } catch (e) {
    if (e.code !== "BS-UI-GT-MISSING") return report(e);
    note = `${e.code}: ${e.message}`;
    region = await ask("none").catch((err) => report(err));
    if (!region) return undefined;
  }
  if (superseded()) return undefined;
  let delta = BigInt(region.offsets.test);
  planner = new Planner(pool, { onChange: requestFrame });
  const refPlan = planner.add(refInfo, 0n), testPlan = planner.add(testInfo, delta);
  refView = new EventView(gpu, planner, refPlan);
  testView = new EventView(gpu, planner, testPlan);
  try {
    await refView.init();
    await testView.init();
    compare = new CompareView(gpu, refView, testView);
    await compare.init();
  } catch (e) {
    self.close();
    throw e;
  }
  if (superseded()) return undefined;
  self.view = compare;
  const canvases = {}, figures = [];
  for (const [name, title] of PANELS) {
    const canvas = el("canvas", { className: `image ${name}` });
    canvas.style.cssText = `display: block; width: 100%; aspect-ratio: ${refInfo.width} / ${refInfo.height};`;
    canvases[name] = canvas;
    const label = name === "ref" ? `real: ${refInfo.label}` : name === "test" ? `sim: ${testInfo.label}` : title;
    figures.push(el("figure", { style: "margin: 0" }, canvas, el("figcaption", { className: "muted",
      textContent: label })));
  }
  const kind = el("select", { name: "kind" }, ...COMPARE_KINDS.map((k) => el("option", { value: k, textContent: k })));
  const length = el("select", { name: "length" }, ...WINDOWS_MS.map((ms) => el("option", { value: ms,
    textContent: `${ms} ms` })));
  length.value = "33";
  const clip = el("input", { name: "clip", type: "number", min: "0.1", step: "any", value: "3", size: 5 });
  const gamma = el("input", { name: "gamma", type: "number", min: "0.1", max: "4", step: "0.05", value: "0.5",
    size: 5 });
  const sigmaDiff = el("input", { name: "sigma_diff", type: "number", min: "0", max: String(MAX_SIGMA), step: "0.5",
    value: String(DEFAULT_SIGMA.diff), size: 4 });
  const sigmaRatio = el("input", { name: "sigma_ratio", type: "number", min: "0", max: String(MAX_SIGMA), step: "0.5",
    value: String(DEFAULT_SIGMA.ratio), size: 4 });
  const auto = el("button", { type: "button", name: "autoclip", textContent: "Auto" });
  const play = el("button", { type: "button", name: "play", textContent: "Play" });
  const mode = el("select", { name: "mode" }, ...["viewer", "crop", "none"].map((m) => el("option", { value: m,
    textContent: m })));
  mode.value = region.mode;
  const cropText = el("input", { name: "crop", size: 14, placeholder: "x0,y0,x1,y1" });
  const applyRegion = el("button", { type: "button", name: "region", textContent: "Apply" });
  const deltaInput = el("input", { name: "delta", size: 14, value: String(delta) });
  const applyDelta = el("button", { type: "button", name: "apply_delta", textContent: "Apply" });
  const statsButton = el("button", { type: "button", name: "stats", textContent: "Statistics of the window" });
  const badge = el("span", { className: "muted" });
  const regionLine = el("p", { className: "muted" });
  badge.dataset.role = "badge";
  regionLine.dataset.role = "region";
  const strip = el("canvas", { className: "timeline" });
  strip.style.cssText = "width: 100%; height: 64px; display: block; margin-top: 6px; cursor: ew-resize;";
  const statsOut = el("div");
  const grid = el("div", { style: "display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); " +
    "gap: 8px;" }, ...figures);
  stage.replaceChildren(el("div", {}, kind, " window ", length, " clip ", clip, " ", auto, " γ ", gamma,
    " σ difference ", sigmaDiff, " σ ratio ", sigmaRatio, " px ", play, " ", badge),
  el("div", {}, "region ", mode, " ", cropText, " ", applyRegion, " · δ_stream (sim) ", deltaInput, " ns ", applyDelta,
    " ", statsButton), regionLine, grid, strip, statsOut);
  // region state: intervals, masks fetched in chunks; shownMask is the mask the panels outline (NO_MASK: none)
  const NO_MASK = -2;                             // maskAt answers −1 outside every interval, never −2
  let renewed = false;                            // asked for again after a 404, until it serves masks or statistics
  const regionMasks = (r) => new RegionMasks(r, {
    fetchChunk: (m0, m1) => apiBytes(`/compare/region/${r.region_id}/masks?m0=${m0}&m1=${m1}`),
    onLoad: () => {
      renewed = false;
      requestFrame();
    },
    onError: report,
    onGone: (e) => renew(e),
  });
  let iv = intervalsOf(region), masks = regionMasks(region), shownMask = NO_MASK;
  const describeRegion = () => {
    const d = region.delta;
    const parts = [`region ${region.mode} (${region.label}): ${region.n_masks} mask(s)`,
      d ? `δ ${d.ns === null ? "unavailable" : `${d.ns} ns`} (${d.source})` : "no EVIMO2 stream",
      `δ_stream proposed ${region.offsets.test} ns`];
    regionLine.replaceChildren(parts.join(" · "), ...[...region.warnings.map((w) => `${w.code}: ${w.message}`),
      ...(note ? [note] : [])].map((t) => el("div", { className: "warn", textContent: t })));
  };
  describeRegion();
  const setRegion = (r) => {
    masks.close();
    region = r;
    iv = intervalsOf(r);
    masks = regionMasks(r);
    compare.setMask(null);
    shownMask = NO_MASK;
    describeRegion();
    requestFrame();
  };
  // A region the server dropped (404) is asked for once more: its id names its content, so the same region comes
  // back. A second 404 before it served anything stops there (Apply asks again). Resolves true when renewed.
  const renew = async (e) => {
    if (closed) return false;
    if (renewed) {
      report(`${e.code ?? "error"}: ${e.message}: the server dropped the region again; Apply asks for it again`);
      return false;
    }
    renewed = true;
    const before = region;
    try {
      const r = await ask(region.mode, region.crop);
      if (!closed && region === before) setRegion(r);
      return true;
    } catch (err) {
      report(err);
      return false;
    }
  };
  // The outline of the mask of the interval that holds the window's centre; none while that mask loads (or failed),
  // rather than another interval's.
  const updateMask = (w0, w1) => {
    const m = maskAt(iv, (w0 + w1) / 2n);
    if (m === shownMask) return;
    const words = m < 0 ? null : masks.get(m);
    if (m >= 0 && !words) {
      if (shownMask !== NO_MASK) compare.setMask(null);
      shownMask = NO_MASK;
      return;
    }
    compare.setMask(words);
    shownMask = m;
  };
  const t0 = BigInt(refInfo.t_start_ns), t1 = BigInt(refInfo.t_end_ns);
  timeline = new Timeline(strip, { t0, t1, length: 33_000_000n, onChange: requestFrame, signal });
  timeline.load(refInfo.sid).catch(report);
  Object.assign(self, { timeline, planner, delta: () => delta, region: () => region });
  let planned = "", busy = false;
  const colors = async () => {
    const options = { theme: theme(), clip: Number(clip.value) || 3, gamma: Number(gamma.value) || 0.5 };
    const [a, b] = await Promise.all([refView.setColors(options), testView.setColors(options)]);
    if (closed || !a || !b) return;
    timeline.colors = { on: a.on, off: a.off };
    timeline.draw();
    requestFrame();
  };
  await colors().catch(report);
  if (superseded()) return undefined;
  kind.addEventListener("change", () => {
    refView.setKind(kind.value);
    testView.setKind(kind.value);
    colors().catch(report);
  }, { signal });
  for (const input of [clip, gamma]) input.addEventListener("change", () => colors().catch(report), { signal });
  for (const input of [sigmaDiff, sigmaRatio]) {
    input.addEventListener("change", () => {
      try {
        compare.setBlur({ diff: Number(sigmaDiff.value), ratio: Number(sigmaRatio.value) });
        requestFrame();
      } catch (e) {
        sigmaDiff.value = String(compare.sigma.diff);
        sigmaRatio.value = String(compare.sigma.ratio);
        report(e);
      }
    }, { signal });
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => colors().catch(report), { signal });
  length.addEventListener("change", () => timeline.set(timeline.playhead, BigInt(Number(length.value) * 1e6)),
    { signal });
  play.addEventListener("click", () => {
    timeline.toggle();
    play.textContent = timeline.playing ? "Pause" : "Play";
  }, { signal });
  auto.addEventListener("click", async () => {      // viz's automatic clip of the reference (§4.5)
    if (busy) return;
    busy = true;
    badge.textContent = "auto clip…";
    try {
      const r = await refView.autoClip(t0, t1, timeline.length, { kind: kind.value });
      if (!r || closed) return;
      clip.value = String(r.clip);
      auto.title = `99th percentile of the real stream over ${r.windows.length} window(s)` +
        `${r.exact ? "" : " (estimated)"}`;
    } catch (e) {
      report(e);
    } finally {
      busy = false;
      planned = "";
    }
    await colors().catch(report);
  }, { signal });
  applyRegion.addEventListener("click", async () => {
    let crop = null;
    if (cropText.value.trim()) {
      crop = cropText.value.split(",").map((v) => Number(v.trim()));
      if (crop.length !== 4 || !crop.every(Number.isInteger)) return report("BS-UI-REQUEST: crop: x0,y0,x1,y1");
    }
    try {
      note = "";
      renewed = false;
      setRegion(await ask(mode.value, crop));
    } catch (e) {
      report(e);
    }
    return undefined;
  }, { signal });
  applyDelta.addEventListener("click", () => {
    if (!/^-?[0-9]{1,18}$/.test(deltaInput.value.trim())) return report("BS-UI-REQUEST: δ_stream: integer ns");
    delta = BigInt(deltaInput.value.trim());
    testPlan.delta = delta;
    planned = "";
    requestFrame();
    return undefined;
  }, { signal });
  statsButton.addEventListener("click", async () => {
    const { w0, w1 } = timeline.window;
    statsOut.textContent = "computing…";
    // both offsets given: the reference at 0, the test at the δ_stream field's value (a stream compared with itself
    // is listed twice)
    const stats = () => api("/compare/stats", { method: "POST", body: { streams: [refInfo.sid, testInfo.sid],
      t0_ns: String(w0), t1_ns: String(w1), region_id: region.region_id, offsets: ["0", String(delta)],
      options: { bin_ms: 1, segment_s: Math.min(1, Number(w1 - w0) / 1e9) } } });
    try {
      let s;
      try {
        s = await stats();
      } catch (e) {
        if (e.status !== 404) throw e;
        if (!(await renew(e))) {                  // renew reported why
          statsOut.textContent = "";
          return;
        }
        s = await stats();                        // on the region asked for again
      }
      renewed = false;
      if (!closed) statsOut.replaceChildren(statsTable(s));
    } catch (e) {
      statsOut.textContent = "";
      report(e);
    }
  }, { signal });
  resize = new ResizeObserver(() => {
    const dpr = devicePixelRatio || 1;
    for (const cv of Object.values(canvases)) {
      cv.width = Math.max(1, Math.round(cv.clientWidth * dpr));
      cv.height = Math.max(1, Math.round(cv.clientHeight * dpr));
    }
    timeline.draw();
    requestFrame();
  });
  for (const cv of Object.values(canvases)) resize.observe(cv);
  blink = setInterval(requestFrame, BLINK_MS);
  function render(now) {
    frameWanted = false;
    if (closed || gpu.lost || !refView.colors || !testView.colors || busy) return;
    const { w0, w1, playhead } = timeline.window;
    if (planned !== `${w0}:${w1}:${delta}`) {
      planned = `${w0}:${w1}:${delta}`;
      planner.setView(w0, w1, playhead);
    }
    const a = refView.draw(w0, w1), b = testView.draw(w0, w1);
    compare.draw();
    updateMask(w0, w1);
    const blinked = Math.floor(now / BLINK_MS) % 2 ? "test" : "ref";
    compare.showPanels(PANELS.map(([name]) => [canvases[name], name === "blink" ? blinked : name]));
    const la = badgeText(a, planner.overfull), lb = badgeText(b, planner.overfull);
    const text = `real: ${la.text} · sim: ${lb.text} · ${fmtTime(w0)} – ${fmtTime(w1)}`;
    if (la.error || lb.error) badge.replaceChildren(el("span", { className: "error", textContent: text }));
    else badge.textContent = text;
    play.textContent = timeline.playing ? "Pause" : "Play";
    if (planner.errors.length) report(planner.errors.at(-1));
  }
  status.textContent = `${refInfo.label} vs ${testInfo.label}: ${refInfo.width}×${refInfo.height}` +
    `${gpu.info?.vendor ? ` · WebGPU ${gpu.info.vendor}` : ""}`;
  timeline.set(timeline.playhead);
  return undefined;
}

// The viewer statistics of both streams as a table (rows: statistics; columns: streams).
function statsTable(s) {
  const fmt = (v, digits = 4) => (v === null || v === undefined ? "–" : typeof v === "number" &&
    !Number.isInteger(v) ? Number(v.toPrecision(digits)).toString() : String(v));
  const rows = [["events", (x) => x.summary.events], ["mean rate (Mev/s)", (x) => x.summary.mean_rate_mev_s],
    ["peak 1 ms rate (Mev/s)", (x) => x.summary.peak_rate_1ms_mev_s], ["ON fraction", (x) => x.summary.on_fraction],
    ...[...new Set(s.stats.flatMap((x) => Object.keys(x.summary.vector_strength)))].map((f) => [
      `vector strength ${f} Hz`, (x) => x.summary.vector_strength[f]]),
    ["silent pixels", (x) => x.summary.silent_pixel_fraction],
    ["per-pixel interval median (ms)", (x) => x.summary.per_pixel_interval_ms_median],
    ["region time share", (x) => x.summary.region?.time_fraction],
    ["region event share", (x) => x.summary.region?.event_share],
    ["δ_stream (ns)", (x) => x.offset_ns], ["count filter", (x) => x.count_filter]];
  const table = el("table", { className: "stats" });
  const head = table.insertRow();
  head.append(el("th", { textContent: "" }), ...s.stats.map((x) => el("th", { textContent: x.label })));
  for (const [name, get] of rows) {
    const tr = table.insertRow();
    tr.append(el("th", { textContent: name }));
    for (const x of s.stats) tr.insertCell().textContent = fmt(get(x));
  }
  return table;
}
