// The viewer's page pool (docs/design/phase3/ui.md §4.4 "Device and pool"): pages of 2^18 events × 8 B = 2 MiB, each
// holding events of one block of one stream in the block's order; storage buffers of at most 512 MiB, allocated once
// per tab at the first open (destroyed memory is released lazily, V2). A failed allocation is retried once at half
// size, then the viewer falls back (BS-UI-GPU-OOM).

import { GpuError, MAX_BINDING, MiB, scoped } from "./gpu.js";

export const PAGE_SHIFT = 18;
export const PAGE_EVENTS = 1 << PAGE_SHIFT;
export const PAGE_BYTES = PAGE_EVENTS * 8;
export const POOL_BUSY_BYTES = 512 * MiB;
export const POOL_MAX_BYTES = 2048 * MiB;

// §4.4: 512 MiB while the GPU is busy or its memory is unknown (/meta gpu.nvml null); else min(2 GiB, 25 % of the
// free memory). WebGPU itself reports no free memory.
export function poolBytes(meta) {
  const g = meta?.gpu;
  if (!g || g.busy || !g.nvml) return POOL_BUSY_BYTES;
  const free = Math.max(0, g.nvml.total_mib - g.nvml.used_mib) * MiB;
  return Math.min(POOL_MAX_BYTES, Math.floor(free / 4));
}

export class Pool {
  // A pool of about `bytes` (whole pages), retried once at half size.
  static async create(gpu, bytes) {
    for (const size of [bytes, bytes / 2]) {
      try {
        return await Pool.allocate(gpu, size);
      } catch (e) {
        if (!(e instanceof GpuError)) throw e;
      }
    }
    throw new GpuError("BS-UI-GPU-OOM", `the viewer pool (${Math.round(bytes / MiB)} MiB) could not be allocated`);
  }

  static async allocate(gpu, bytes) {
    const d = gpu.device;
    const perBuffer = Math.floor(Math.min(d.limits.maxBufferSize, d.limits.maxStorageBufferBindingSize, MAX_BINDING) /
      PAGE_BYTES);
    let pages = Math.floor(bytes / PAGE_BYTES);
    if (pages < 1 || perBuffer < 1) throw new GpuError("BS-UI-GPU-OOM", "no room for one page");
    const buffers = [];
    try {
      await scoped(gpu, "viewer pool", () => {
        for (; pages > 0; pages -= perBuffer) {
          buffers.push(d.createBuffer({ size: Math.min(perBuffer, pages) * PAGE_BYTES, label: "viewer pool",
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
        }
      });
    } catch (e) {
      for (const b of buffers) b.destroy();
      throw e;
    }
    return new Pool(gpu, buffers, perBuffer);
  }

  constructor(gpu, buffers, perBuffer) {
    this.gpu = gpu;
    this.buffers = buffers;
    this.perBuffer = perBuffer;
    this.capacity = buffers.reduce((n, b) => n + b.size / PAGE_BYTES, 0);
    this.reset();                                   // pop() gives slot 0 first
  }

  get used() { return this.capacity - this.free.length; }
  reset() { this.free = Array.from({ length: this.capacity }, (_, i) => this.capacity - 1 - i); }
  alloc() { return this.free.length ? this.free.pop() : -1; }
  release(slot) { this.free.push(slot); }
  // [buffer index, page within that buffer]
  locate(slot) { return [Math.floor(slot / this.perBuffer), slot % this.perBuffer]; }

  // ev words (Uint32Array of pairs) of events [i, i + n) of the page in slot `slot`.
  write(slot, i, words) {
    const [b, p] = this.locate(slot);
    this.gpu.device.queue.writeBuffer(this.buffers[b], (p * PAGE_EVENTS + i) * 8, words);
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
    this.buffers = [];
  }
}
