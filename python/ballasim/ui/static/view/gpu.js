// The WebGPU device and shader hygiene (docs/design/phase3/ui.md §4.4 "Device and pool", "Shader hygiene"; V5):
// limits raised to at most 512 MiB per buffer and 16384 px per texture side, getCompilationInfo() on every module,
// error scopes around pipeline creation and first dispatches, uncaptured errors and device loss recorded
// (BS-UI-SHADER).

export const MiB = 1 << 20;
export const MAX_BINDING = 512 * MiB;
export const MAX_SIDE = 16384;                    // the wire carries coordinates up to 16383 (§4.11)

export class GpuError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

// {adapter, device, info, format, errors, lost, report}. report(text) records an error in `errors` and passes it to
// onError(text): uncaptured errors and errors caught by an error scope go through it; device loss calls onError only.
export async function openGpu({ onError = () => {} } = {}) {
  if (!navigator.gpu) throw new GpuError("BS-UI-WEBGPU", "this browser has no WebGPU (Chrome 147 or newer)");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new GpuError("BS-UI-WEBGPU", "no WebGPU adapter (see Browser check)");
  const cap = (name, most) => Math.min(adapter.limits[name], most);
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: cap("maxStorageBufferBindingSize", MAX_BINDING),
    maxBufferSize: cap("maxBufferSize", MAX_BINDING),
    maxTextureDimension2D: cap("maxTextureDimension2D", MAX_SIDE) } });
  const gpu = { adapter, device, info: adapter.info, format: navigator.gpu.getPreferredCanvasFormat(), errors: [],
    lost: null };
  gpu.report = (text) => {
    gpu.errors.push(text);
    onError(text);
  };
  device.addEventListener("uncapturederror", (e) => gpu.report(`BS-UI-SHADER: ${e.error?.message ?? e.error}`));
  device.lost.then((info) => {
    gpu.lost = info;
    if (info.reason !== "destroyed") onError(`BS-UI-SHADER: the GPU device was lost (${info.message})`);
  });
  return gpu;
}

// A shader module whose compilation info holds no error; `messages` collects every message (B5).
export async function compile(gpu, code, label, messages = []) {
  const module = gpu.device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  for (const m of info.messages) messages.push({ label, type: m.type, line: m.lineNum, text: m.message });
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) {
    throw new GpuError("BS-UI-SHADER", `${label}: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`)
      .join("; ")}`);
  }
  return module;
}

export async function computePipeline(gpu, code, label, messages) {
  const module = await compile(gpu, code, label, messages);
  try {
    return await gpu.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" },
      label });
  } catch (e) {
    throw new GpuError("BS-UI-SHADER", `${label}: ${e.message}`);
  }
}

export async function renderPipeline(gpu, code, label, format, messages) {
  const module = await compile(gpu, code, label, messages);
  try {
    return await gpu.device.createRenderPipelineAsync({ layout: "auto", label, vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] }, primitive: { topology: "triangle-list" } });
  } catch (e) {
    throw new GpuError("BS-UI-SHADER", `${label}: ${e.message}`);
  }
}

// Runs fn() (synchronous GPU calls) inside a validation and an out-of-memory error scope; rejects with the first error
// caught. Both scopes are popped in the same task as they were pushed, so no other code's GPU calls fall inside them.
export async function scoped(gpu, label, fn) {
  const d = gpu.device;
  d.pushErrorScope("out-of-memory");
  d.pushErrorScope("validation");
  let value, thrown = null;
  try {
    value = fn();
  } catch (e) {
    thrown = e;
  }
  const [validation, oom] = await Promise.all([d.popErrorScope(), d.popErrorScope()]);
  if (oom) throw new GpuError("BS-UI-GPU-OOM", `${label}: ${oom.message}`);
  if (validation) throw new GpuError("BS-UI-SHADER", `${label}: ${validation.message}`);
  if (thrown) throw thrown;
  return value;
}

// Copies `bytes` of a buffer (offset and size multiples of 4) to the CPU.
export async function readBuffer(gpu, src, offset, bytes) {
  const d = gpu.device;
  const staging = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(src, offset, staging, 0, bytes);
  d.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.destroy();
  return out;
}

// The RGBA bytes (H·W·4) of an rgba8unorm texture, rows unpadded.
export async function readTexture(gpu, texture, width, height) {
  const d = gpu.device, row = Math.ceil((width * 4) / 256) * 256;
  const staging = d.createBuffer({ size: row * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = d.createCommandEncoder();
  enc.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: row }, [width, height]);
  d.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(staging.getMappedRange()), out = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) out.set(src.subarray(y * row, y * row + width * 4), y * width * 4);
  staging.destroy();
  return out;
}
