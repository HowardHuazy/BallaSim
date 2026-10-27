// Helpers the runs, job and results screens share (docs/design/phase3/ui.md §5.1): elements, resource ids, formats,
// state badges, tabs and the perf plots of a run (logs/perf.jsonl, one record per window). No third-party code.

import { apiResponse } from "../app/api.js";
import { Plot } from "../plot/plot.js";

const PERF_READS = 256;                           // reads per refresh at most (1 GB of perf.jsonl)

export const el = (tag, { dataset, ...props } = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  if (dataset) Object.assign(e.dataset, dataset);
  e.append(...children.filter((c) => c !== null && c !== undefined));
  return e;
};

export const errorText = (e) => `${e.code ?? "error"}: ${e.message}`;
export const errorLine = (e) => el("p", { className: "error", textContent: errorText(e) });

// rid (§3.3): base64url without padding of UTF-8 "<root_id>/<relpath>".
export function encodeRid(root, rel) {
  const bytes = new TextEncoder().encode(`${root}/${rel}`);
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function decodeRid(rid) {
  const text = atob(rid.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (rid.length % 4)) % 4));
  const s = new TextDecoder().decode(Uint8Array.from(text, (c) => c.charCodeAt(0)));
  const i = s.indexOf("/");
  return [s.slice(0, i), s.slice(i + 1)];
}

export function fmtDuration(s) {
  if (s === null || s === undefined || !Number.isFinite(Number(s))) return "–";
  s = Number(s);
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}

export const fmtNs = (ns) => (ns === null || ns === undefined ? "–" : `${(Number(ns) / 1e9).toFixed(6)} s`);

export function fmtBytes(n) {
  if (!Number.isFinite(n)) return "";
  const units = ["B", "kB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1000 && i < units.length - 1) [n, i] = [n / 1000, i + 1];
  return `${i ? n.toFixed(1) : n} ${units[i]}`;
}

export function fmtWhen(text) {
  if (!text) return "–";
  const d = new Date(text);
  return Number.isNaN(d.getTime()) ? String(text) : d.toLocaleString();
}

export const FINAL = new Set(["complete", "failed", "cancelled", "interrupted"]);

// A job or run state as a badge (class state-<state>, coloured by the page's styles).
export const stateBadge = (state) => el("span", { className: `state state-${state ?? "unknown"}`,
  textContent: state ?? "unknown" });

// Tabs: buttons that call select(name); returns {bar, set(name)}.
export function tabs(names, select) {
  const buttons = new Map(names.map(([name, label]) => [name, el("button", { type: "button", textContent: label,
    name: `tab-${name}` })]));
  const bar = el("div", { className: "tabs", role: "tablist" }, ...buttons.values());
  const set = (name) => {
    for (const [n, b] of buttons) b.setAttribute("aria-selected", String(n === name));
  };
  for (const [name, b] of buttons) b.addEventListener("click", () => select(name));
  return { bar, set };
}

// The perf plots of run `rid` (§4.10: logs/perf.jsonl): wall time per window and its parts (sampling, which holds the
// plan and the render, sensing, ground truth, writing), events and pixel-samples per window; x is the window's place
// in the file (cameras one after the other). refresh() reads the complete lines after the last offset (GET
// /runs/{rid}/perf?since=, at most 4 MiB a request) until a read returns none, one request at a time; a WebSocket
// `perf` message only asks for a refresh.
export class PerfPlots {
  constructor(rid, { signal = null } = {}) {
    this.rid = rid;
    this.signal = signal;
    this.next = 0;
    this.hubOffset = null;                        // the offset of the hub's last perf message
    this.gen = 0;
    this.records = [];
    this.busy = null;
    this.again = false;
    const canvas = () => {
      const c = el("canvas", { className: "plot" });
      c.style.cssText = "display: block; width: 100%; height: 190px; margin: 6px 0 14px;";
      return c;
    };
    this.summary = el("p", { className: "muted" });
    this.time = new Plot(canvas(), { title: "Wall time per window", xLabel: "window", yLabel: "s", signal,
      series: [{ name: "total" }, { name: "sampling" }, { name: "sensing" }, { name: "GT" }, { name: "write" }] });
    this.events = new Plot(canvas(), { title: "Events per window", xLabel: "window", yLabel: "events", signal,
      series: [{ name: "events" }] });
    this.samples = new Plot(canvas(), { title: "Pixel-samples per window", xLabel: "window", yLabel: "samples",
      signal, series: [{ name: "pixel-samples" }] });
    this.element = el("div", { className: "perf" }, this.summary, this.time.canvas, this.events.canvas,
      this.samples.canvas);
    this.resize = new ResizeObserver(() => requestAnimationFrame(() => this.draw()));   // after layout
    this.resize.observe(this.element);
    signal?.addEventListener("abort", () => this.resize.disconnect());
  }

  refresh() {
    if (this.busy) {
      this.again = true;
      return this.busy;
    }
    this.busy = (async () => {
      try {
        do {
          this.again = false;
          const gen = this.gen;
          for (let i = 0; i < PERF_READS; i++) {
            const since = this.next;
            const r = await apiResponse(`/runs/${this.rid}/perf?since=${since}`);
            const text = await r.text();
            if (this.signal?.aborted) return;
            if (gen !== this.gen) {               // reset meanwhile: read again from the start
              this.again = true;
              break;
            }
            const next = Number(r.headers.get("X-Next-Offset"));
            for (const line of text.split("\n")) {
              if (!line) continue;
              try {
                this.records.push(JSON.parse(line));
              } catch { /* a malformed line is skipped */ }
            }
            if (Number.isFinite(next)) this.next = next;
            this.draw();
            if (!text || !(next > since)) break;  // the file's end; a longer file comes in further reads
          }
        } while (this.again && !this.signal?.aborted);
      } finally {
        this.busy = null;
      }
    })();
    return this.busy;
  }

  // A WebSocket `perf` message asks for a refresh from the page's own offset. Its offset is only compared with the
  // hub's earlier ones (the page's HTTP reads run ahead of the hub's tail): the hub's offsets grow until perf.jsonl
  // shrinks (an overwritten run), when its tail reads the new file from 0, or a new tail starts at a smaller file's
  // end. So an offset below the previous message's, or 0 once the page holds records (also the first message of a
  // tail that found the run directory after the page read it: one read again), means a new file: read from 0.
  onMessage(msg) {
    const offset = Number(msg.offset);
    if (Number.isFinite(offset)) {
      if ((this.hubOffset !== null && offset < this.hubOffset) || (offset === 0 && this.next > 0)) {
        this.next = 0;
        this.gen++;
        this.records = [];
      }
      this.hubOffset = offset;
    }
    return this.refresh();
  }

  draw() {
    const recs = this.records, x = recs.map((_, i) => i);
    const num = (r, k) => (Number.isFinite(Number(r[k])) ? Number(r[k]) : NaN);
    const parts = ["sample_s", "sense_s", "gt_s", "write_s"];
    const total = recs.map((r) => parts.reduce((a, k) => a + (num(r, k) || 0), 0));
    this.time.setData(x, [total, ...parts.map((k) => recs.map((r) => num(r, k)))]);
    this.events.setData(x, [recs.map((r) => num(r, "events"))]);
    this.samples.setData(x, [recs.map((r) => num(r, "pixel_samples"))]);
    const wall = total.reduce((a, b) => a + b, 0), events = recs.reduce((a, r) => a + (num(r, "events") || 0), 0);
    const simulated = recs.length * 2 ** 23 / 1e9;
    this.summary.textContent = recs.length ? `${recs.length} windows (${simulated.toFixed(3)} s simulated) in ` +
      `${fmtDuration(wall)} of window time (${(wall / simulated).toFixed(1)} s per simulated second); ${events} ` +
      "events" : "No windows recorded yet.";
  }
}
