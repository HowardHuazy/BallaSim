// The results page of a run, #/runs/<rid>[?tab=events|perf|provenance|files&camera=<name>] (docs/design/phase3/ui.md
// §3.3 /runs, §4.10, §4.11; §5.1 "Open a run", "Configure and run" step 8): the run's summary and live progress
// (run:<rid>, for a run that is still being written), and tabs:
// - events: each camera in the viewer (UI-2, UI-3), with the run's timeline tracks (a Blender job's preview frames);
// - perf: the perf plots of logs/perf.jsonl;
// - provenance: run.json's inputs (configuration file, CLI layer, profiles), the resolved record (value, layer and
//   status of every key, per scope), stage digests, versions, git, renderer and light groups;
// - files: the run's files, downloaded through the API (the Bearer header, a blob: URL).

import { api, apiResponse } from "../app/api.js";
import { subscribe } from "../app/ws.js";
import { PerfPlots, decodeRid, el, errorLine, errorText, fmtDuration, fmtNs, fmtWhen, stateBadge,
  tabs } from "./common.js";

const TABS = [["events", "Events"], ["perf", "Performance"], ["provenance", "Provenance"], ["files", "Files"]];
const REFUSALS = {
  "BS-UI-PARTIAL": "The event files are still being written; they can be viewed when the run completes.",
  "BS-UI-NOT-FOUND": "The run has no event file.",
};

const kv = (rows) => el("table", {}, ...rows.map(([k, v]) => el("tr", {}, el("th", { textContent: k }),
  v instanceof Node ? el("td", {}, v) : el("td", { textContent: v === null || v === undefined ? "–" : String(v) }))));
const json = (v) => el("pre", { textContent: JSON.stringify(v, null, 2) });
const details = (summary, ...body) => el("details", {}, el("summary", { textContent: summary }), ...body);

export function mountResults(view, { rid, query }) {
  const life = new AbortController(), signal = life.signal;
  view.classList.add("wide");
  const [, rel] = (() => { try { return decodeRid(rid); } catch { return ["", rid]; } })();
  const name = rel.split("/").filter(Boolean).at(-1) ?? rel;
  const title = el("h2", { textContent: `Run: ${name}` });
  const head = el("div");
  const live = el("p");
  const body = el("div");
  let tab = TABS.some(([t]) => t === query.get("tab")) ? query.get("tab") : "events";
  let camera = query.get("camera");
  const bar = tabs(TABS, (t) => select(t));
  view.replaceChildren(title, head, live, bar.bar, body);
  let page = null, plots = null, record = null, viewer = null, unsubRun = null, shown = null;

  const setQuery = () => {
    const q = new URLSearchParams({ tab, ...(camera ? { camera } : {}) });
    history.replaceState(null, "", `#/runs/${rid}?${q}`);
  };

  // The viewer module once a camera was opened: closing it is synchronous, so leaving this page never closes a
  // viewer the next page opened.
  function closeViewer() {
    viewer?.unmountViewer();
    viewer = null;
  }

  function summary() {
    const cams = page.cameras ?? [];
    const events = cams.reduce((a, c) => a + (Number(c.events) || 0), 0);
    head.replaceChildren(el("p", {}, stateBadge(page.status), " ", stateBadge(page.live),
      ` · format ${page.format}${page.inputs?.bundle?.supplied ? " · supplied bundle" : ""}`,
      ...(page.capabilities?.resolve === true ? [" · ", el("a", { href: `#/setup?run=${encodeURIComponent(rid)}`,
        textContent: "Edit parameters" })] : [])),
    kv([["Cameras", cams.map((c) => `${c.name} (${c.events ?? "?"} events)`).join(", ") || "–"],
      ["Span", page.span_ns ? `${fmtNs(page.span_ns[0])} – ${fmtNs(page.span_ns[1])}` : "–"],
      ["Simulated", page.simulated_s !== undefined ? fmtDuration(page.simulated_s) : "–"],
      ["Wall time", page.wall_s !== undefined ? fmtDuration(page.wall_s) : "–"],
      ["Events", cams.length ? events : "–"], ["Started", fmtWhen(page.started)],
      ["Finished", fmtWhen(page.finished)]]),
    ...(page.error ? [el("p", { className: "error", textContent: `Failed: ${page.error}` })] : []),
    ...(page.warnings ?? []).map((w) => el("p", { className: "warn", textContent: w })));
    view.dataset.status = page.status ?? "";
  }

  // While the run is live the page is read again: every 3 s (a run that fails, is cancelled or dies before its last
  // window never reports fraction 1, and its run.json turns final, or `stale`, with no message), each second after
  // its last window (run.json turns final once the writers close), until it is no longer live.
  const POLL_MS = 3000, FINISHING_MS = 1000;
  let pollTimer = null, polling = false, fast = false;
  function schedule(ms) {
    if (signal.aborted || polling) return;          // a running poll schedules the next one itself
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      pollTimer = null;
      polling = true;
      try {
        await load();
      } finally {
        polling = false;
      }
      if (!signal.aborted && page?.live === "live") schedule(fast ? FINISHING_MS : POLL_MS);
    }, ms);
  }
  signal.addEventListener("abort", () => clearTimeout(pollTimer));
  function finishing() {
    if (fast) return;
    fast = true;
    schedule(500);
  }

  // A run still being written (a bare CLI run, or a job's): its progress (§4.10 fraction and ETA) from run:<rid>.
  function follow() {
    if (unsubRun || page.live !== "live") return;
    const progress = el("progress", { max: 1 }), text = el("span", { className: "muted" });
    live.replaceChildren(progress, " ", text);
    progress.removeAttribute("value");
    unsubRun = subscribe([`run:${rid}`], (msg) => {
      if (msg.op === "progress") {
        if (Number.isFinite(msg.fraction)) progress.value = msg.fraction;
        const p = msg.progress ?? {};
        text.textContent = [Number.isFinite(msg.fraction) ? `${(100 * msg.fraction).toFixed(1)} %` : null,
          p.camera ? `camera ${p.camera} (${msg.camera_index + 1}/${msg.cameras})` : null,
          p.window ? `window ${p.window}/${p.windows}` : null,
          Number.isFinite(msg.eta_s) ? `ETA ${fmtDuration(msg.eta_s)}` : null].filter(Boolean).join(" · ");
        if (msg.fraction === 1) finishing();
      } else if (msg.op === "perf") plots?.onMessage(msg).catch(() => {});
    }, { onSync: () => load() });
    schedule(POLL_MS);
  }

  async function load() {
    try {
      page = await api(`/runs/${rid}`);
    } catch (e) {
      if (!signal.aborted) head.replaceChildren(errorLine(e));
      return;
    }
    if (signal.aborted) return;
    summary();
    follow();
    if (page.live !== "live" && unsubRun) {
      unsubRun();
      unsubRun = null;
      live.replaceChildren();
    }
    if (shown !== page.status) {                    // first load, or the run finished: the tab again
      shown = page.status;
      select(tab);
    }
  }

  async function select(t) {
    tab = t;
    bar.set(t);
    setQuery();
    closeViewer();
    if (signal.aborted || !page) return;
    body.dataset.tab = t;
    try {
      if (t === "events") await showEvents();
      else if (t === "perf") showPerf();
      else if (t === "provenance") await showProvenance();
      else showFiles();
    } catch (e) {
      if (!signal.aborted) body.replaceChildren(errorLine(e));
    }
  }

  async function showEvents() {
    const cap = page.capabilities?.view;
    if (cap !== true) {
      body.replaceChildren(el("p", { className: "muted", textContent: REFUSALS[cap] ?? String(cap) }));
      return;
    }
    const names = (page.cameras ?? []).map((c) => c.name).filter((n) => page.files.includes(`events/${n}.h5`));
    for (const f of page.files) {                   // a camera without a summary (a failed run) still has its file
      const m = /^events\/([^/]+)\.h5$/.exec(f);
      if (m && !names.includes(m[1])) names.push(m[1]);
    }
    const tracks = await api(`/runs/${rid}/tracks`).catch(() => []);
    if (signal.aborted) return;
    const stage = el("div", { className: "viewer" });
    const pick = el("p", {}, "Camera: ", ...names.map((n) => el("button", { type: "button", name: `camera-${n}`,
      textContent: n, onclick: () => open(n) })));
    const note = el("p", { className: "muted", textContent: tracks.length ? `Timeline tracks: ${tracks.map((t) =>
      `${t.label}, ${t.n} marks`).join("; ")}. Click a mark in the strip's top band, or press [ and ], to view the ` +
      "events of its window." : "" });
    body.replaceChildren(pick, note, stage);
    async function open(n) {
      camera = n;
      setQuery();
      for (const b of pick.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.textContent === n));
      const views = await import("../view/views.js");
      if (signal.aborted || tab !== "events") return;
      viewer = views;
      views.mountViewer(stage, { query: new URLSearchParams({ run: rid, camera: n }), embedded: true });
    }
    if (names.length) await open(camera && names.includes(camera) ? camera : names[0]);
  }

  function showPerf() {
    if (!plots) {
      plots = new PerfPlots(rid, { signal });
      plots.refresh().catch((e) => { plots.summary.textContent = errorText(e); });
    }
    body.replaceChildren(plots.element);
    plots.draw();
  }

  async function showProvenance() {
    record ??= await api(`/runs/${rid}/record`);
    if (signal.aborted || tab !== "provenance") return;
    const inputs = record.inputs ?? {}, cli = inputs.cli ?? {};
    const filter = el("input", { type: "search", name: "filter", placeholder: "filter keys", size: 30 });
    const hideDefaults = el("input", { type: "checkbox", name: "set-only", checked: true });
    const tables = el("div");
    const draw = () => {
      const f = filter.value.trim().toLowerCase(), scopes = [["run", record.resolved?.run ?? {}],
        ...Object.entries(record.resolved?.cameras ?? {}).map(([c, v]) => [`camera ${c}`, v])];
      tables.replaceChildren(...scopes.map(([label, entries]) => {
        const rows = Object.entries(entries).filter(([k, e]) => (!f || k.toLowerCase().includes(f)) &&
          !(hideDefaults.checked && String(e?.layer ?? "").startsWith("default")));
        return el("div", {}, el("h4", { textContent: `${label}: ${rows.length} of ${Object.keys(entries).length} ` +
          "keys" }), el("table", { className: "record" }, el("tr", {}, ...["key", "value", "layer", "status"].map((h) =>
          el("th", { textContent: h }))), ...rows.map(([k, e]) => el("tr", {},
          el("td", {}, el("code", { textContent: k })),
          el("td", {}, el("code", { textContent: JSON.stringify(e?.value) })),
          el("td", { textContent: e?.layer ?? "" }), el("td", { textContent: e?.status ?? "" })))));
      }));
    };
    filter.addEventListener("input", draw, { signal });
    hideDefaults.addEventListener("change", draw, { signal });
    const digests = record.stage_digests ?? {};
    body.replaceChildren(
      el("h3", { textContent: "Inputs" }),
      kv([["Configuration file", inputs.file ?? "(given as a document)"],
        ["Profiles", (inputs.profiles ?? []).map((p) => `${p.id} (${String(p.sha256 ?? "").slice(0, 12)})`)
          .join(", ") ||
          "–"],
        ["Command-line profiles", (cli.profiles ?? []).join(", ") || "–"],
        ["Bundle", inputs.bundle?.supplied ? "supplied" : "built from the scene"]]),
      el("h4", { textContent: "Command-line layer (--set)" }),
      kv(Object.entries(cli.set ?? {}).map(([k, v]) => [k, JSON.stringify(v)])),
      details("Configuration as given", json(inputs.config ?? {})),
      el("h3", { textContent: "Resolved record" }),
      el("p", {}, filter, " ", el("label", {}, hideDefaults, " hide schema defaults")), tables,
      el("h3", { textContent: "Stage digests" }),
      ...Object.entries(digests).map(([c, d]) => el("div", {}, el("h4", { textContent: c }),
        kv(Object.entries(d ?? {}).map(([s, v]) => [s, el("code", { textContent: String(v) })])))),
      el("h3", { textContent: "Software" }),
      kv([["BallaSim", page.versions?.ballasim], ["Git", page.git?.commit ? `${page.git.commit}${page.git.dirty ?
        " (modified)" : ""}` : "–"], ["Renderer", page.renderer ? JSON.stringify(page.renderer) : "–"]]),
      details("Versions", json(page.versions ?? {})),
      details("Light groups", json(page.light_groups ?? [])),
      details("Derived values", json(page.derived ?? {})));
    draw();
  }

  function showFiles() {
    const files = page.files ?? [];
    const status = el("span", { className: "muted" });
    body.replaceChildren(el("p", { className: "muted", textContent: `${files.length} files` }), status,
      el("ul", { className: "files" }, ...files.map((f) => el("li", {}, el("a", { href: "#", textContent: f,
        onclick: (e) => {
          e.preventDefault();
          download(f, status);
        } })))));
  }

  async function download(f, status) {
    try {
      const r = await apiResponse(`/runs/${rid}/files/${f.split("/").map(encodeURIComponent).join("/")}`);
      const url = URL.createObjectURL(await r.blob());
      el("a", { href: url, download: f.split("/").at(-1) }).click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (e) {
      status.textContent = errorText(e);
    }
  }

  load();
  return () => {
    life.abort();
    unsubRun?.();
    closeViewer();
  };
}

