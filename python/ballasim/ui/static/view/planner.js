// The LOD planner (docs/design/phase3/ui.md §4.3): which slices of which blocks to fetch into the page pool, and how
// each block is drawn. Blocks are [k 2^28, (k + 1) 2^28) ns of a stream's own clock, in LOD order: the prefix
// [0, n_ge[j]) holds exactly the events of level >= j, a binomial thinning by 2^−j (§4.1).
//
// - First paint: the coarsest level j0 whose prefixes of the visible blocks hold at most 2^23 events (64 MiB) and fit
//   the pool, fetched in slices of at most one page (2^18 events), 4 requests in flight. A page holds events of one
//   block, so a window that meets more blocks with events than the pool has pages fits at no level (`overfull`).
// - Refinement: block by block, nearest to the playhead first, the prefix grows level by level to j = 0 or until the
//   pool is full; a full pool is tried again whenever a block's requests finish (it may have become evictable).
// - Drawing: a block with resident prefix [0, r_b) is drawn at its finest complete level j_b = min{j : n_ge[j] <= r_b},
//   count n_ge[j_b] and weight 2^{j_b}; events beyond it are not drawn, so an interrupted refinement never draws a
//   partial level. Each block's estimate is unbiased, so the image is.
// - Eviction: blocks drawn least recently, then farthest from the playhead, lose pages from their highest one down.
//   Visible blocks keep their first-paint prefix; pages above it go to another visible block's first-paint prefix
//   (every visible block's first paint comes before any block's refinement) or to the refinement of a block nearer to
//   the playhead.

import { api, apiBytes } from "../app/api.js";
import { decodeBSEV } from "../app/wire.js";
import { PAGE_EVENTS } from "./pool.js";

export const BLOCK_SHIFT = 28n;
export const LEVELS = 8;
export const FIRST_PAINT_EVENTS = 1 << 23;
const LANES = 4;
const HEADS_PER_REQUEST = 256;
const CLAMP = 1n << 30n;

export const floorDiv = (a, b) => (a >= 0n ? a / b : -((-a + b - 1n) / b));
const clampWindow = (v) => Number(v < -CLAMP ? -CLAMP : v > CLAMP ? CLAMP : v);
const pagesFor = (n) => Math.ceil(n / PAGE_EVENTS);

// One block of one stream: its head (§3.3 /blocks) and its resident prefix in pool slots, page by page.
export class BlockState {
  constructor(head) {
    this.k = head.k;
    this.head = head;
    this.slots = [];
    this.resident = 0;          // r_b: every event of [0, r_b) is in the pool
    this.requested = 0;         // [0, requested) is resident or in flight
    this.inflight = 0;
    this.arrived = [];          // completed slices beyond r_b: [i0, i1)
    this.lastDrawn = -1;
    this.failed = null;
  }

  get n() { return this.head.n; }

  // j_b = min{j : n_ge[j] <= r_b}, or −1 while not even the coarsest level is resident (§4.3 step 5). A level with no
  // events (n_ge[j] = 0) is complete before anything arrives.
  level() {
    if (!this.head.lod) return this.resident === this.n ? 0 : -1;    // a growing (time-ordered) block: whole or none
    const g = this.head.n_ge;
    for (let j = 0; j <= LEVELS; j++) if (g[j] <= this.resident) return j;
    return -1;
  }

  arrive(i0, i1) {
    this.arrived.push([i0, i1]);
    this.arrived.sort((a, b) => a[0] - b[0]);
    while (this.arrived.length && this.arrived[0][0] === this.resident) this.resident = this.arrived.shift()[1];
  }

  // Drops the highest page; the prefixes shrink to its start.
  dropPage() {
    const slot = this.slots.pop();
    const end = this.slots.length * PAGE_EVENTS;
    this.resident = Math.min(this.resident, end);
    this.requested = Math.min(this.requested, end);
    this.arrived = this.arrived.filter(([i0]) => i0 < end);
    return slot;
  }
}

// One stream as the planner sees it: its blocks and its offset δ onto the view clock (t_view = t + δ, §4.6).
export class StreamPlan {
  constructor(planner, info, delta = 0n) {
    this.planner = planner;
    this.info = info;
    this.sid = info.sid;
    this.delta = BigInt(delta);
    this.width = info.width;
    this.height = info.height;
    const t0 = BigInt(info.t_start_ns), t1 = BigInt(info.t_end_ns);
    this.kFirst = Number(t0 >> BLOCK_SHIFT);
    this.kEnd = info.n_events ? Number((t1 - 1n) >> BLOCK_SHIFT) + 1 : this.kFirst;
    this.blocks = new Map();
    this.pendingHeads = new Map();      // k -> the request that brings its head
    this.failedHeads = new Map();       // k -> the error of its head, not asked for again by this plan
    this.visible = [0, 0];
  }

  // Blocks [k0, k1) that meet the view window [w0, w1).
  range(w0, w1) {
    if (w1 <= w0) return [0, 0];
    const k0 = Math.max(this.kFirst, Number(floorDiv(w0 - this.delta, 1n << BLOCK_SHIFT)));
    const k1 = Math.min(this.kEnd, Number(floorDiv(w1 - 1n - this.delta, 1n << BLOCK_SHIFT)) + 1);
    return k0 < k1 ? [k0, k1] : [0, 0];
  }

  isVisible(k) { return k >= this.visible[0] && k < this.visible[1]; }

  // Heads of blocks [k0, k1), from GET /blocks (the server builds them as needed): runs of at most 256 blocks that are
  // neither known nor asked for already. Never rejects; a block whose head failed is in `failedHeads`.
  async loadHeads(k0, k1) {
    const waits = new Set();
    let a = null;
    const run = (b) => {
      if (a !== null) waits.add(this.loadRun(a, b));
      a = null;
    };
    for (let k = k0; k < k1; k++) {
      const pending = this.pendingHeads.get(k);
      if (pending || this.blocks.has(k) || this.failedHeads.has(k)) {
        run(k);
        if (pending) waits.add(pending);
      } else if (a === null) a = k;
      else if (k - a === HEADS_PER_REQUEST) {
        run(k);
        a = k;
      }
    }
    run(k1);
    await Promise.all(waits);
  }

  // Heads [a, b) in one request. The server answers with one error when any block of the request fails, so a failed
  // request is split in halves down to single blocks: the other blocks still draw, and a failing block is recorded
  // (`failedHeads`, one planner error) and not built again on every frame.
  loadRun(a, b) {
    const request = api(`/streams/${this.sid}/blocks?k0=${a}&k1=${b}`).then((heads) => {
      for (const h of heads) if (!this.blocks.has(h.k)) this.blocks.set(h.k, new BlockState(h));
    }, (e) => {
      if (b - a > 1) {
        const m = a + ((b - a) >> 1);
        return Promise.all([this.loadRun(a, m), this.loadRun(m, b)]);
      }
      this.failedHeads.set(a, e);
      this.planner.fail(e);
      return undefined;
    }).finally(() => {
      for (let k = a; k < b; k++) if (this.pendingHeads.get(k) === request) this.pendingHeads.delete(k);
    });
    for (let k = a; k < b; k++) this.pendingHeads.set(k, request);
    return request;
  }
}

export class Planner {
  // options: lanes (requests in flight), firstPaintEvents, refine (false: stop at the first paint), level (a fixed
  // first-paint level: tests, and 0 for history views, §4.3 step 7), onChange (called whenever drawing may change).
  constructor(pool, { lanes = LANES, firstPaintEvents = FIRST_PAINT_EVENTS, refine = true, level = null,
    onChange = () => {} } = {}) {
    Object.assign(this, { pool, lanes, firstPaintEvents, refine, fixedLevel: level, onChange });
    this.streams = [];
    this.view = null;
    this.frame = 0;
    this.inflight = 0;
    this.j0 = LEVELS;
    this.full = false;
    this.overfull = null;               // {blocks, pages, capacity} while the first paint cannot fit the pool
    this.errors = [];
    this.waiters = [];
    this.generation = 0;
    this.closed = false;
    this.ready = Promise.resolve();
  }

  add(info, delta = 0n) {
    const s = new StreamPlan(this, info, delta);
    this.streams.push(s);
    if (this.view) this.setView(this.view.w0, this.view.w1, this.view.playhead, this.view.extra);
    return s;
  }

  // The view window [w0, w1) and the playhead, on the view clock (BigInt ns). `extra`: the rolling-shutter readout
  // (H − 1)·row_ns, by which the last row's window reaches past w1 (§4.4).
  setView(w0, w1, playhead = w1, extra = 0n) {
    this.view = { w0, w1, playhead, extra };
    const generation = ++this.generation;
    const loads = this.streams.map((s) => {
      s.visible = s.range(w0, w1 + extra);
      return s.loadHeads(...s.visible);
    });
    this.ready = Promise.all(loads).then(() => {
      if (generation === this.generation) this.plan();
    }, (e) => this.fail(e));
    this.plan();
  }

  visibleBlocks() {
    const out = [];
    for (const s of this.streams) {
      for (let k = s.visible[0]; k < s.visible[1]; k++) {
        const b = s.blocks.get(k);
        if (b) out.push([s, b]);
      }
    }
    return out;
  }

  distance(s, b) {
    const kp = Number(floorDiv(this.view.playhead - s.delta, 1n << BLOCK_SHIFT));
    return Math.abs(b.k - kp);
  }

  // §4.3 step 2: j0 = the coarsest level needed, i.e. the smallest j whose visible prefixes hold at most 2^23 events
  // and fit the pool (the exact form of clamp(⌈log2(N_view / 2^23)⌉, 0, 8) on the blocks' own counts), else 8. Pages
  // hold events of one block, so at j = 8 the visible blocks with events still need a page each: when even that does
  // not fit, `overfull` records it and the first paint fills the pool from the playhead outwards.
  firstLevel(visible) {
    const need = (j) => {
      let events = 0, pages = 0;
      for (const [, b] of visible) {
        const n = b.head.lod ? b.head.n_ge[j] : b.n;
        events += n;
        pages += pagesFor(n);
      }
      return { events, pages };
    };
    let j0 = this.fixedLevel;
    for (let j = 0; j0 === null && j < LEVELS; j++) {
      const { events, pages } = need(j);
      if (events <= this.firstPaintEvents && pages <= this.pool.capacity) j0 = j;
    }
    j0 ??= LEVELS;
    const pages = need(j0).pages, capacity = this.pool.capacity;
    this.overfull = pages > capacity ? { blocks: visible.filter(([, b]) => b.n > 0).length, pages, capacity } : null;
    return j0;
  }

  plan() {
    if (!this.view || this.closed) return;
    this.j0 = this.firstLevel(this.visibleBlocks());
    this.full = false;
    this.pump();
    this.onChange();
  }

  // The next slice to fetch: first-paint prefixes (nearest block first), then refinement block by block.
  next() {
    const visible = this.visibleBlocks().filter(([, b]) => !b.failed);
    visible.sort((x, y) => this.distance(...x) - this.distance(...y) || x[1].k - y[1].k);
    for (const [s, b] of visible) {
      const goal = b.head.lod ? b.head.n_ge[this.j0] : b.n;
      if (b.requested < goal) return this.slice(s, b, goal, true);
    }
    if (!this.refine || this.full) return null;
    for (const [s, b] of visible) {
      if (b.requested < b.n) {
        const g = b.head.n_ge;                                   // the next level boundary above the prefix
        let goal = b.n;
        for (let j = LEVELS; j >= 0; j--) if (g[j] > b.requested) { goal = g[j]; break; }
        const req = this.slice(s, b, goal, false);
        if (!req) this.full = true;
        return req;
      }
    }
    return null;
  }

  // [requested, min(goal, next page boundary)), with a pool slot for a new page (evicting if needed); null when full.
  slice(s, b, goal, firstPaint) {
    const i0 = b.requested;
    const page = Math.floor(i0 / PAGE_EVENTS);
    if (page === b.slots.length) {
      let slot = this.pool.alloc();
      if (slot < 0 && this.evict(s, b, firstPaint)) slot = this.pool.alloc();
      if (slot < 0) return null;
      b.slots.push(slot);
    }
    const i1 = Math.min(goal, (page + 1) * PAGE_EVENTS);
    b.requested = i1;
    return { s, b, i0, i1, slot: b.slots[page] };
  }

  // Frees one page for block `b`: from blocks out of view (drawn least recently, then farthest), else from visible
  // blocks above their first-paint prefix, farthest first: any of them for a first-paint request, only those farther
  // from the playhead than `b` for a refinement (§4.3 step 6). Blocks with requests in flight are kept.
  evict(s, b, firstPaint) {
    const d = this.distance(s, b);
    let best = null, bestKey = null;
    for (const t of this.streams) {
      for (const c of t.blocks.values()) {
        if (c === b || c.inflight || !c.slots.length) continue;
        const visible = t.isVisible(c.k), dc = this.distance(t, c);
        if (visible) {
          const keep = pagesFor(c.head.lod ? c.head.n_ge[this.j0] : c.n);
          if (c.slots.length <= keep || (!firstPaint && dc <= d)) continue;
        }
        const key = visible ? [1, -dc, 0] : [0, c.lastDrawn, -dc];
        if (!bestKey || (key[0] - bestKey[0] || key[1] - bestKey[1] || key[2] - bestKey[2]) < 0) {
          best = c;
          bestKey = key;
        }
      }
    }
    if (!best) return false;
    this.pool.release(best.dropPage());
    this.onChange();
    return true;
  }

  // Stops fetching and drawing: answers still in flight are dropped (the pool may already serve another planner) and
  // onChange is never called again (its view may be destroyed).
  close() {
    this.closed = true;
    this.onChange = () => {};
    for (const w of this.waiters.splice(0)) w();
  }

  pump() {
    if (this.closed) return;
    while (this.inflight < this.lanes) {
      const req = this.next();
      if (!req) break;
      this.fetch(req);
    }
    if (this.inflight === 0) for (const w of this.waiters.splice(0)) w();
  }

  async fetch({ s, b, i0, i1, slot }) {
    this.inflight++;
    b.inflight++;
    try {
      const msg = await decodeBSEV(await apiBytes(`/streams/${s.sid}/blocks/${b.k}?i0=${i0}&i1=${i1}&cols=ev`));
      if (msg.i0 !== i0 || msg.n !== i1 - i0) throw new Error(`block ${b.k}: slice [${i0}, ${i1}) answered wrongly`);
      if (this.closed) return;
      this.pool.write(slot, i0 % PAGE_EVENTS, msg.columns.ev.values);
      b.arrive(i0, i1);
    } catch (e) {
      b.failed = e;
      this.fail(e);
    } finally {
      this.inflight--;
      b.inflight--;
      this.full = false;                // a block with requests in flight was kept; it may be evictable now
      this.onChange();
      this.pump();
    }
  }

  fail(e) {
    this.errors.push(e);
    this.onChange();
  }

  // Resolves when nothing is in flight and nothing more can be fetched (tests, first-paint timing).
  async idle() {
    let ready;
    do {
      ready = this.ready;
      await ready;
    } while (ready !== this.ready);
    await new Promise((resolve) => {
      this.waiters.push(resolve);
      this.pump();
    });
  }

  // The pages to draw for stream `s` over the view window [w0, w1), rows shifted by up to `extra` (§4.3 step 5, §4.4
  // PageView lo/hi): {pages: [{slot, count, weight, lo, hi, tb}], coarsest, missing, failed, blocks: [{k, j, count,
  // resident, n_ge}]}; tb is the block start on the view clock. Blocks not drawn are `failed` (their head or a slice
  // failed: they load no further) or `missing`. With `exact` (history views, §4.3 step 7) only blocks complete at
  // j = 0 are drawn.
  draw(s, w0 = this.view.w0, w1 = this.view.w1, extra = 0n, { exact = false } = {}) {
    this.frame++;
    const out = { pages: [], coarsest: 0, missing: 0, failed: 0, blocks: [] };
    const [k0, k1] = s.range(w0, w1 + extra);
    for (let k = k0; k < k1; k++) {
      const b = s.blocks.get(k);
      const j = b ? b.level() : -1;
      if (j < 0 || (exact && j > 0)) {
        if (b ? b.failed : s.failedHeads.has(k)) out.failed++;
        else out.missing++;
        continue;
      }
      const count = b.head.lod ? b.head.n_ge[j] : b.n;
      const tb = (BigInt(k) << BLOCK_SHIFT) + s.delta;
      const lo = clampWindow(w0 - tb), hi = clampWindow(w1 - tb);
      b.lastDrawn = this.frame;
      out.coarsest = Math.max(out.coarsest, j);
      out.blocks.push({ k, j, count, resident: b.resident, n_ge: b.head.n_ge });
      for (let p = 0; p * PAGE_EVENTS < count; p++) {
        out.pages.push({ slot: b.slots[p], count: Math.min(PAGE_EVENTS, count - p * PAGE_EVENTS), weight: 1 << j, lo,
          hi, tb });
      }
    }
    return out;
  }
}
