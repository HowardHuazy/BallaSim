// The pixel inspector (docs/design/phase3/ui.md §4.4, §7): the planes under the pointer (ON and OFF counts, the
// time surface's latest event, decayed sums), read back 16 bytes at a time, at most 10 times a second, and described
// by the view. Counts drawn from sampled blocks are estimates (weights 2^j, §4.3).

const PERIOD_MS = 100;

export class Inspector {
  constructor(view, out) {
    this.view = view;
    this.out = out;
    this.pending = null;
    this.busy = false;
    this.last = 0;
    this.closed = false;
  }

  // No readback after this (the view's buffers are about to be destroyed).
  close() {
    this.closed = true;
    this.pending = null;
  }

  // Image pixel (x, y), or null when the pointer leaves the image.
  point(x, y) {
    if (this.closed) return;
    if (x === null) {
      this.pending = null;
      this.out.textContent = "";
      return;
    }
    this.pending = [x, y];
    if (!this.busy) setTimeout(() => this.run(), Math.max(0, this.last + PERIOD_MS - performance.now()));
    this.busy = true;
  }

  async run() {
    const p = this.pending;
    this.pending = null;
    this.last = performance.now();
    if (p && !this.closed) {
      try {
        const planes = await this.view.readPixel(...p);
        if (this.closed) return;
        this.out.textContent = this.view.describePixel(...p, planes);
      } catch (e) {
        if (!this.closed) this.out.textContent = `${e.code ?? "error"}: ${e.message}`;
      }
    }
    this.busy = false;
    if (this.pending) this.point(...this.pending);
  }
}
