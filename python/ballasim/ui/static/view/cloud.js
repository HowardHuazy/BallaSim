// The (x, y, t) cloud (docs/design/phase3/ui.md §4.4 "(x, y, t) cloud", G3): each event of the window [w0, w1) is
// the point (x + 0.5, y + 0.5, t_ms), t_ms = (t − w0)·1e−6, projected through one 4×4 matrix (model: the window as a
// box, the image width deep in time; an orbit camera; perspective; the canvas viewport) and splatted with atomicAdd
// into ON and OFF planes at the canvas size (compute splatting: points are 1 px, and V3 measured them 5.6–6.4× slower).
// The planes are coloured as the polarity kind's f32 formula with the viewer's clip and γ. Sampled blocks splat their
// weight 2^j, as everywhere. Drag orbits, the wheel dollies.

import { computePipeline, readBuffer, readTexture, renderPipeline } from "./gpu.js";
import { CLOUD, LUT_ENTRIES, MODE, PRESENT, PV, PV_WORDS, RESOLVE, dispatchPages, linearRgb, packPages }
  from "./kernels.js";

const usage = (...names) => names.reduce((u, n) => u | GPUBufferUsage[n], 0);
const FOV = Math.PI / 4;

// 4×4 matrices, column-major (WGSL mat4x4<f32>): mul(a, b) = a·b.
function mul(a, b) {
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}

// The matrix from (x, y, t_ms, 1) to (px·w, py·w, 0, w) for a canvas of w_c × h_c pixels: the sensor image centred
// at the origin (y up), time along −z over the image width, orbit (yaw, pitch) at `distance`, perspective, viewport.
export function cameraMatrix({ width, height, lengthMs, yaw, pitch, distance, canvas: [wc, hc] }) {
  const s = Math.max(width, height), depth = width / s;
  const model = [1 / s, 0, 0, 0, 0, -1 / s, 0, 0, 0, 0, -depth / Math.max(lengthMs, 1e-9), 0,
    -width / (2 * s), height / (2 * s), depth / 2, 1];
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const rotY = [cy, 0, -sy, 0, 0, 1, 0, 0, sy, 0, cy, 0, 0, 0, 0, 1];
  const rotX = [1, 0, 0, 0, 0, cp, sp, 0, 0, -sp, cp, 0, 0, 0, 0, 1];
  const back = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -distance, 1];
  const f = 1 / Math.tan(FOV / 2), aspect = wc / hc;
  const project = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0];   // w = −z
  const viewport = [wc / 2, 0, 0, 0, 0, -hc / 2, 0, 0, 0, 0, 0, 0, wc / 2, hc / 2, 0, 1];
  return new Float32Array([viewport, project, back, rotX, rotY, model].reduce((a, b) => mul(a, b)));
}

export class CloudView {
  // canvas: the canvas to present on (null: draw at `size` only, for tests). `matrix`: a fixed matrix instead of the
  // orbit camera (tests).
  constructor(gpu, planner, stream, canvas = null) {
    Object.assign(this, { gpu, planner, stream, canvas });
    this.width = stream.width;
    this.height = stream.height;
    this.yaw = -0.6;
    this.pitch = 0.35;
    this.distance = 2.2;
    this.matrix = null;
    this.size = [stream.width, stream.height];
    this.shape = null;
    this.destroyed = false;
  }

  async init(messages = []) {
    const { gpu } = this, d = gpu.device, pool = this.planner.pool;
    this.splat = await computePipeline(gpu, CLOUD, "cloud", messages);
    this.resolve = await computePipeline(gpu, RESOLVE, "resolve", messages);
    const align = d.limits.minStorageBufferOffsetAlignment;
    this.region = Math.ceil((pool.perBuffer * PV_WORDS * 4) / align) * align;
    this.pv = d.createBuffer({ size: this.region * pool.buffers.length, usage: usage("STORAGE", "COPY_DST"),
      label: "cloud page views" });
    this.frameUniform = d.createBuffer({ size: 16, usage: usage("UNIFORM", "COPY_DST") });
    d.queue.writeBuffer(this.frameUniform, 0, new Uint32Array([this.width, this.height, 0, 0]));
    this.camUniform = d.createBuffer({ size: 80, usage: usage("UNIFORM", "COPY_DST") });
    this.resolveUniform = d.createBuffer({ size: 96, usage: usage("UNIFORM", "COPY_DST") });
    this.table = d.createBuffer({ size: 4, usage: usage("STORAGE") });                    // unused by mode 3
    this.lut = d.createBuffer({ size: LUT_ENTRIES * 4, usage: usage("STORAGE") });
    if (this.canvas) {
      this.present = await renderPipeline(gpu, PRESENT, "present", gpu.format, messages);
      this.presentUniform = d.createBuffer({ size: 48, usage: usage("UNIFORM", "COPY_DST") });
      this.context = this.canvas.getContext("webgpu");
      this.context.configure({ device: d, format: gpu.format, alphaMode: "opaque" });
    }
  }

  setColors(colors) { this.colors = colors; }

  orbit(dx, dy) {
    this.yaw += dx * 0.01;
    this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.01));
  }

  dolly(direction) { this.distance = Math.max(0.3, Math.min(20, this.distance * 2 ** (direction / 4))); }

  // The canvas-sized planes [2][h][w] and image, (re)made when the size changes.
  shapeTo(w, h) {
    if (this.shape?.[0] === w && this.shape?.[1] === h) return;
    const d = this.gpu.device, pool = this.planner.pool;
    this.counts?.destroy();
    this.image?.destroy();
    this.shape = [w, h];
    this.counts = d.createBuffer({ size: 2 * w * h * 4, usage: usage("STORAGE", "COPY_SRC", "COPY_DST"),
      label: "cloud planes" });
    this.image = d.createTexture({ size: [w, h], format: "rgba8unorm", label: "cloud image",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.groups = pool.buffers.map((buffer, b) => d.createBindGroup({ layout: this.splat.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.pv, offset: b * this.region, size: this.region } },
        { binding: 2, resource: { buffer: this.frameUniform } }, { binding: 3, resource: { buffer: this.counts } },
        { binding: 4, resource: { buffer: this.camUniform } }] }));
    this.resolveGroup = d.createBindGroup({ layout: this.resolve.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.counts } }, { binding: 1, resource: { buffer: this.table } },
      { binding: 2, resource: { buffer: this.lut } }, { binding: 3, resource: { buffer: this.resolveUniform } },
      { binding: 4, resource: this.image.createView() }] });
    if (this.canvas) {
      this.presentGroup = d.createBindGroup({ layout: this.present.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.image.createView() },
        { binding: 1, resource: { buffer: this.presentUniform } }] });
    }
  }

  // Draws the window [w0, w1) (BigInt ns, view clock); returns the planner's draw record.
  draw(w0, w1) {
    const { gpu, planner } = this, d = gpu.device, c = this.colors;
    const [w, h] = this.canvas ? [this.canvas.width, this.canvas.height] : this.size;
    this.shapeTo(w, h);
    const rec = planner.draw(this.stream, w0, w1, 0n);
    const { words, groups } = packPages(planner.pool, this.region, this.pv.size, rec, (p, i32, u32, f32, o) => {
      f32[o + PV.base_ms] = Number(p.tb - w0) * 1e-6;
    });
    d.queue.writeBuffer(this.pv, 0, words);
    const cam = new ArrayBuffer(80);
    new Float32Array(cam, 0, 16).set(this.matrix ?? cameraMatrix({ width: this.width, height: this.height,
      lengthMs: Number(w1 - w0) * 1e-6, yaw: this.yaw, pitch: this.pitch, distance: this.distance, canvas: [w, h] }));
    new Uint32Array(cam, 64, 2).set([w, h]);
    d.queue.writeBuffer(this.camUniform, 0, cam);
    const r = new ArrayBuffer(96), rf = new Float32Array(r);
    new Uint32Array(r).set([w, h, 0, MODE.polarity_formula]);
    rf.set([c.clip, c.gamma, 1], 5);
    rf.set([...linearRgb(c.background), ...linearRgb(c.on), ...linearRgb(c.off)], 12);
    d.queue.writeBuffer(this.resolveUniform, 0, r);
    const enc = d.createCommandEncoder();
    enc.clearBuffer(this.counts);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.splat);
    dispatchPages(pass, this.groups, groups);
    pass.setPipeline(this.resolve);
    pass.setBindGroup(0, this.resolveGroup);
    pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
    pass.end();
    if (this.canvas) {
      d.queue.writeBuffer(this.presentUniform, 0, new Float32Array([1, 1, 0, 0, w, h, 0, 0,
        ...c.background.map((v) => v / 255), 1]));
      const draw = enc.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(),
        loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
      draw.setPipeline(this.present);
      draw.setBindGroup(0, this.presentGroup);
      draw.draw(3);
      draw.end();
    }
    d.queue.submit([enc.finish()]);
    this.lastDraw = rec;
    return rec;
  }

  // The last splat: plane 0 OFF, plane 1 ON, each h·w.
  async readCounts() {
    return new Uint32Array(await readBuffer(this.gpu, this.counts, 0, this.counts.size));
  }

  readImage() { return readTexture(this.gpu, this.image, ...this.shape); }

  destroy() {
    this.destroyed = true;
    for (const b of [this.pv, this.frameUniform, this.camUniform, this.resolveUniform, this.table, this.lut,
      this.presentUniform, this.counts, this.image]) b?.destroy();
    this.context?.unconfigure();
  }
}
