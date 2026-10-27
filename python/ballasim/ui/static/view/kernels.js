// WGSL of the viewer (docs/design/phase3/ui.md §4.4, §4.5; from the measured kernels, VZ/webgpu/kernels.wgsl.js).
// The pool is array<vec2<u32>>: word 0 = x | y<<14 | tag<<28 | p<<31, word 1 = t − t_base in ns (< 2^28). Each
// dispatch binds one pool buffer and its PageView rows; workgroup row wg.y draws one page, 256 threads × 4 events.
// No WGSL reserved words, no buffer-to-itself copies (V5).

import { PAGE_SHIFT } from "./pool.js";

export const EVENTS_PER_WORKGROUP = 1024;
// PageView (48 B, i32/u32/f32 words), per page and frame, computed on the CPU in 64-bit integers (§4.4). `page` (the
// page within the bound pool buffer) uses 4 B of the padding.
export const PV = { lo: 0, hi: 1, count: 2, weight: 3, row_ns: 4, plane_set: 5, ts_hi: 6, ts_lo: 7, age_base: 8,
  base_ms: 9, page: 10 };
export const PV_WORDS = 12;
// The colour pass's modes (RESOLVE). Decay is the signed LUT of the decayed net count (§4.4 "Decay").
export const MODE = { events_table: 0, events_lut: 1, polarity_table: 2, polarity_formula: 3, counts_table: 4,
  counts_formula: 5, timesurface: 6, decay: 7 };
// The LUT buffer: viz's 511-entry diverging LUT, then its 256-entry sequential LUT.
export const SEQUENTIAL_AT = 511;
export const LUT_ENTRIES = 767;
export const HISTOGRAM_BINS = 1024;

// viz.render._srgb_to_linear of an sRGB8 colour, as [r, g, b, 1]: the theme colours of the polarity formula (mode 3).
export const linearRgb = (rgb) => [...rgb.map((c) => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}), 1];

// PageView rows of planner draw record `rec` (pages {slot, count, weight, lo, hi, tb}), grouped by pool buffer:
// {words (one region of PV rows per buffer), groups: [[page, …] per buffer]}. fill(page, i32, u32, f32, o) sets the
// kind's own fields of the row at word offset o.
export function packPages(pool, region, size, rec, fill = () => {}) {
  const groups = pool.buffers.map(() => []);
  for (const p of rec.pages) {
    const [b, page] = pool.locate(p.slot);
    groups[b].push({ ...p, page });
  }
  const words = new ArrayBuffer(size), i32 = new Int32Array(words), u32 = new Uint32Array(words);
  const f32 = new Float32Array(words);
  groups.forEach((g, b) => g.forEach((p, i) => {
    const o = (b * region) / 4 + i * PV_WORDS;
    i32[o + PV.lo] = p.lo;
    i32[o + PV.hi] = p.hi;
    u32[o + PV.count] = p.count;
    u32[o + PV.weight] = p.weight;
    u32[o + PV.page] = p.page;
    fill(p, i32, u32, f32, o);
  }));
  return { words, groups };
}

// One dispatch per pool buffer that has pages: workgroup rows = pages, columns = the largest page's events / 1024.
export function dispatchPages(pass, bindGroups, groups) {
  groups.forEach((g, b) => {
    if (!g.length) return;
    pass.setBindGroup(0, bindGroups[b]);
    const most = g.reduce((m, p) => Math.max(m, p.count), 0);
    pass.dispatchWorkgroups(Math.ceil(most / EVENTS_PER_WORKGROUP), g.length);
  });
}

export const COMMON = /* wgsl */`
struct PageView {
  lo: i32, hi: i32,          // the window mapped to the page's block, clamped to ±2^30
  count: u32, weight: u32,   // events to draw (the finest complete level's prefix), 2^j
  row_ns: u32,               // rolling-shutter row time (0: global exposure)
  plane_set: u32,            // plane pair (OFF, ON) of the counts image
  ts_hi: i32, ts_lo: u32,    // time surface: D = t_b + δ − O = ts_hi·2^s + ts_lo, 0 <= ts_lo < 2^s
  age_base: f32,             // decay: f32(t_ref − t_b − δ)
  base_ms: f32,              // cloud: f32((t_b + δ − w0)·1e−6)
  page: u32,                 // the page within the bound pool buffer
  pad: u32,
};
struct Frame { width: u32, height: u32, s: u32, tau: f32 };   // s: time-surface key shift; tau: τ in ns (decay)
@group(0) @binding(0) var<storage, read> ev: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> pv: array<PageView>;
@group(0) @binding(2) var<uniform> frame: Frame;
fn ev_x(w: u32) -> u32 { return w & 0x3FFFu; }
fn ev_y(w: u32) -> u32 { return (w >> 14u) & 0x3FFFu; }
fn ev_p(w: u32) -> u32 { return w >> 31u; }
// §4.4: in the window iff lo + y row_ns <= dt < hi + y row_ns, exact in i32 (|lo|, |hi| <= 2^30, y row_ns < 2^29)
fn in_window(v: PageView, y: u32, dt: u32) -> bool {
  let shift = i32(y * v.row_ns);
  let t = i32(dt);
  return t >= v.lo + shift && t < v.hi + shift;
}
`;

// The loop every event kernel shares: event i of the page drawn by workgroup row wg.y; BODY sees v, e, x, y, p.
const EVENT_LOOP = (body) => /* wgsl */`
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = pv[wg.y];
  let base = v.page << ${PAGE_SHIFT}u;
  for (var k = 0u; k < 4u; k++) {
    let i = (wg.x * 4u + k) * 256u + li;          // consecutive threads read consecutive events
    if (i >= v.count) { break; }
    let e = ev[base + i];
    let x = ev_x(e.x);
    let y = ev_y(e.x);
    let p = ev_p(e.x);
    if (x < frame.width && y < frame.height && in_window(v, y, e.y)) {
      ${body}
    }
  }
}`;

// Accumulation: atomicAdd of the page weight into counts[((plane_set·2 + p)·H + y)·W + x], plane 0 OFF, 1 ON.
export const ACCUMULATE = COMMON + /* wgsl */`
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;
` + EVENT_LOOP(/* wgsl */`
      atomicAdd(&counts[((v.plane_set * 2u + p) * frame.height + y) * frame.width + x], v.weight);`);

// Time surface (§4.4): over the window [O, t_ref) (row-shifted), k = ⌊(t − O − y row_ns)/2^s⌋ from the split offset
// D = ts_hi·2^s + ts_lo: dt − y row_ns + ts_lo lies in (−2^29, 2^28 + 2^30), so the i32 sum and its arithmetic shift
// (a floor) are exact. key = ((k + 1) << 1) | p, combined with atomicMax into plane 0 of the plane set; 0: no event.
export const TIME_SURFACE = COMMON + /* wgsl */`
@group(0) @binding(3) var<storage, read_write> keys: array<atomic<u32>>;
` + EVENT_LOOP(/* wgsl */`
      let k = ((i32(e.y) - i32(y * v.row_ns) + i32(v.ts_lo)) >> frame.s) + v.ts_hi;
      atomicMax(&keys[(v.plane_set * 2u * frame.height + y) * frame.width + x], (u32(k + 1) << 1u) | p);`);

// Decay (§4.4): fixed point, w = round(2^10 · 2^j · exp(−age/τ)), age = age_base + y row_ns − dt, summed per plane.
export const DECAY = COMMON + /* wgsl */`
@group(0) @binding(3) var<storage, read_write> sums: array<atomic<u32>>;
` + EVENT_LOOP(/* wgsl */`
      let age = v.age_base + f32(i32(y * v.row_ns) - i32(e.y));
      let w = u32(round(f32(v.weight) * 1024.0 * exp(-age / frame.tau)));
      atomicAdd(&sums[((v.plane_set * 2u + p) * frame.height + y) * frame.width + x], w);`);

// (x, y, t) cloud (§4.4): (x + 0.5, y + 0.5, t_ms, 1) through the 4×4 matrix to canvas pixels (after the perspective
// divide), atomicAdd of the page weight into [2][H_c][W_c]; t_ms = base_ms + f32(dt)·1e−6. Compute splatting.
export const CLOUD = COMMON + /* wgsl */`
struct Camera { m: mat4x4<f32>, width: u32, height: u32, pad0: u32, pad1: u32 };
@group(0) @binding(3) var<storage, read_write> splat: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> cam: Camera;
` + EVENT_LOOP(/* wgsl */`
      let q = cam.m * vec4<f32>(f32(x) + 0.5, f32(y) + 0.5, v.base_ms + f32(e.y) * 1e-6, 1.0);
      if (q.w > 0.0) {
        let c = floor(q.xy / q.w);
        if (all(c >= vec2<f32>(0.0)) && c.x < f32(cam.width) && c.y < f32(cam.height)) {
          atomicAdd(&splat[(p * cam.height + u32(c.y)) * cam.width + u32(c.x)], v.weight);
        }
      }`);

// The colour pass of every 2D kind (§4.5), from plane set `plane_set` of the counts image:
//   0 events, exact table T[clamp(net, −K, K) + K] = viz colorize_signed(n / clip) (built on the server);
//   1 events, f32 sign(v)|v|^γ then the diverging LUT, as colorize_signed;
//   2 polarity, exact table T[min(on, K)·(K + 1) + min(off, K)] = viz polarity_image (built on the server);
//   3 polarity, f32 linear-light mixing as polarity_image: a = clip(c/clip, 0, 1)^γ per plane;
//   4 counts, exact table T[min(on + off, K)] = viz colorize_magnitude(n, clip);
//   5 counts, f32 log(1 + v)/log(1 + clip), anything present at least step 1, the sequential LUT;
//   6 time surface: the latest event's signed exp(−age/τ), age = (hq − k)·2^s + hr, through the diverging LUT, γ = 1;
//   7 decay: the decayed net count (ON − OFF)·scale through the diverging LUT, as mode 1.
export const RESOLVE = /* wgsl */`
struct Resolve {
  width: u32, height: u32, plane_set: u32, mode: u32,
  k: i32, clip: f32, gamma: f32, scale: f32,
  hq: u32, hr: u32, s: u32, tau: f32,
  bg: vec4<f32>, on: vec4<f32>, off: vec4<f32>,   // linear-light theme colours (mode 3)
};
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<uniform> r: Resolve;
@group(0) @binding(4) var image: texture_storage_2d<rgba8unorm, write>;
fn signed_lut(value: f32, gamma: f32) -> u32 {
  let v = clamp(value, -1.0, 1.0);
  var index = 255;
  if (v != 0.0) {
    var m = abs(v);
    if (gamma != 1.0) { m = pow(m, gamma); }
    index = i32(round(sign(v) * m * 255.0)) + 255;
  }
  return lut[u32(index)];
}
fn alpha(n: u32) -> f32 {
  let a = clamp(f32(n) / r.clip, 0.0, 1.0);
  return select(pow(a, r.gamma), 0.0, a == 0.0);
}
fn srgb8(c: f32) -> f32 {
  let x = clamp(c, 0.0, 1.0);
  return round(select(1.055 * pow(x, 1.0 / 2.4) - 0.055, 12.92 * x, x <= 0.0031308) * 255.0);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= r.width || g.y >= r.height) { return; }
  let plane = r.width * r.height;
  let i = r.plane_set * 2u * plane + g.y * r.width + g.x;
  let off = counts[i];
  let on = counts[i + plane];
  let k = u32(r.k);
  var c: u32;
  switch r.mode {
    case 0u: { c = table[u32(clamp(i32(on) - i32(off), -r.k, r.k) + r.k)]; }
    case 1u: { c = signed_lut(f32(i32(on) - i32(off)) / r.clip, r.gamma); }
    case 2u: { c = table[min(on, k) * (k + 1u) + min(off, k)]; }
    case 3u: {
      let lin = r.bg.rgb + alpha(on) * (r.on.rgb - r.bg.rgb) + alpha(off) * (r.off.rgb - r.bg.rgb);
      textureStore(image, vec2<i32>(g.xy), vec4<f32>(srgb8(lin.r), srgb8(lin.g), srgb8(lin.b), 255.0) / 255.0);
      return;
    }
    case 4u: { c = table[min(on + off, k)]; }
    case 5u: {
      let v = f32(on + off);
      var index = u32(round(clamp(log(1.0 + v) / log(1.0 + r.clip), 0.0, 1.0) * 255.0));
      if (v > 0.0) { index = max(index, 1u); }
      c = lut[${SEQUENTIAL_AT}u + index];
    }
    case 6u: {
      let key = off;                                  // plane 0 of the set holds the keys
      var v = 0.0;
      if (key != 0u) {
        let age = f32(r.hq - ((key >> 1u) - 1u)) * exp2(f32(r.s)) + f32(r.hr);
        v = exp(-age / r.tau);
        if ((key & 1u) == 0u) { v = -v; }
      }
      c = signed_lut(v, 1.0);
    }
    default: { c = signed_lut((f32(on) - f32(off)) * r.scale / r.clip, r.gamma); }
  }
  textureStore(image, vec2<i32>(g.xy), unpack4x8unorm(c));
}`;

// Auto clip (§4.5): a 1024-bin histogram of |ON − OFF| (mode 0) or ON + OFF (mode 1) over the active pixels; values
// of 1023 or more share the last bin. Successive windows add to the same histogram.
export const HISTOGRAM = /* wgsl */`
struct Hist { width: u32, height: u32, plane_set: u32, mode: u32 };
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> h: Hist;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let plane = h.width * h.height;
  let i = (wg.y * nwg.x + wg.x) * 256u + li;          // a 2D grid of workgroups: images beyond 65535·256 pixels
  if (i >= plane) { return; }
  let off = counts[h.plane_set * 2u * plane + i];
  let on = counts[h.plane_set * 2u * plane + plane + i];
  var a = on + off;
  if (h.mode == 0u) { a = select(on - off, off - on, off > on); }
  if (a > 0u) { atomicAdd(&hist[min(a, ${HISTOGRAM_BINS - 1}u)], 1u); }
}`;

// The image on the canvas: nearest image pixel floor(canvas px · scale + offset); background outside. With `hatch`
// (a history view whose blocks are not all loaded, §4.3 step 7) diagonal stripes cover the image.
export const PRESENT = /* wgsl */`
struct Present { scale: vec2<f32>, offset: vec2<f32>, size: vec2<f32>, hatch: f32, pad: f32,
  background: vec4<f32> };
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<uniform> p: Present;
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
  if (p.hatch > 0.0 && (u32(pos.x + pos.y) & 15u) < 3u) { return vec4<f32>(mix(c.rgb, vec3<f32>(0.5), 0.6), 1.0); }
  return c;
}`;

// Every module, for the shader-hygiene check (B5).
export const MODULES = { accumulate: ACCUMULATE, time_surface: TIME_SURFACE, decay: DECAY, cloud: CLOUD,
  resolve: RESOLVE, histogram: HISTOGRAM, present: PRESENT };
