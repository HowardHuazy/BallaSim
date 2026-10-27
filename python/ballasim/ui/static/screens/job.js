// The job page, #/jobs/<id> (docs/design/phase3/ui.md §3.7, §4.10; §5.1 "Configure and run" step 7, "Blender" step
// 5): the job's state as its record gives it (status.json's state, step, stage, fraction and ETA, shown as they are),
// what `queued`, `held`, `waiting`, `stalled`, `interrupted` and `failed` mean, cancel, kill (stalled jobs, after a
// confirmation) and resume, the run's perf plots and preview track, and the runner's logs. Live through the
// WebSocket topics job:<id> and run:<rid>; a reconnect re-reads everything over HTTP.

import { api, apiResponse } from "../app/api.js";
import { subscribe } from "../app/ws.js";
import { FINAL, PerfPlots, el, errorLine, errorText, fmtDuration, fmtWhen, stateBadge, tabs } from "./common.js";

const LOG_KEEP = 256 * 1024;                      // characters of each log the page keeps (the end)
const LOG_READS = 256;                            // 64 kB reads per refresh at most (16 MB)
const STREAMS = ["stderr", "stdout"];

const ordinal = (n) => `${n}${["th", "st", "nd", "rd"][(n % 100 > 10 && n % 100 < 14) || n % 10 > 3 ? 0 : n % 10]}`;

// What a state means for the user (§3.7 States, §4.10).
function explain(rec) {
  const st = rec.status ?? {}, err = rec.error;
  switch (rec.state) {
    case "queued": return `Queued: ${ordinal(rec.position ?? 1)} in the ${rec.gpu ? "GPU" : "CPU"} lane.`;
    case "held": return "Held: an earlier server queued it; it starts when the queue is resumed.";
    case "starting": return "Starting: the runner has not reported yet.";
    case "waiting": return "Waiting for the GPU lock: another producer (a Blender job, another server's job, a " +
      "batch run) holds it; the job starts when it is free.";
    case "stalled": return `Stalled: the runner is alive but has not written a heartbeat for more than 30 s ` +
      `(last ${fmtWhen(st.heartbeat)}). It may be stuck in a long call; kill it if it does not recover.`;
    case "interrupted": return "Interrupted: the runner ended without a final state" +
      `${rec.exit_code !== null && rec.exit_code !== undefined ? ` (exit code ${rec.exit_code})` : ""}.`;
    case "failed": return `Failed: ${err?.type ?? "error"}: ${err?.message ?? "see the log"}` +
      `${err?.key ? ` (key ${err.key}${err.layer ? `, layer ${err.layer}` : ""})` : ""}` +
      `${rec.exit_code !== null && rec.exit_code !== undefined ? `; exit code ${rec.exit_code}` : ""}.`;
    case "cancelled": return "Cancelled.";
    case "complete": return "Complete.";
    default: return "";
  }
}

export function mountJob(view, { id }) {
  const life = new AbortController(), signal = life.signal;
  view.classList.add("wide");
  const title = el("h2", { textContent: "Job" });
  const stateLine = el("p", { className: "job-state" });
  const notice = el("p", { dataset: { role: "notice" } });
  const bar = el("progress", { max: 1 });
  const barText = el("span", { className: "muted", dataset: { role: "progress" } });
  const cancel = el("button", { type: "button", name: "cancel", textContent: "Cancel" });
  const kill = el("button", { type: "button", name: "kill", textContent: "Kill", hidden: true });
  const resume = el("button", { type: "button", name: "resume", textContent: "Resume queue", hidden: true });
  const results = el("a", { hidden: true });
  const actionError = el("span", { className: "error" });
  const facts = el("table", { dataset: { role: "facts" } });
  const tracksLine = el("p", { className: "muted", dataset: { role: "tracks" } });
  const perfBox = el("div");
  const logPre = el("pre", { className: "log" });
  const logTabs = tabs(STREAMS.map((s) => [s, s]), (s) => showLog(s));
  view.replaceChildren(title, stateLine, notice, el("p", {}, bar, " ", barText),
    el("p", {}, cancel, " ", kill, " ", resume, " ", results, " ", actionError), facts, tracksLine, perfBox,
    el("h3", { textContent: "Runner log" }), logTabs.bar, logPre);
  let rec = null, plots = null, runRid = null, unsubRun = null, shownLog = "stderr", tracksFor = null;
  let wsRecords = 0;                                // job records the WebSocket gave (load() keeps a newer one)
  const logs = Object.fromEntries(STREAMS.map((s) => [s, { text: "", next: 0, busy: null, again: false,
    decoder: new TextDecoder() }]));

  function show() {
    if (!rec || signal.aborted) return;
    const st = rec.status ?? {};
    title.textContent = `Job: ${rec.label}`;
    view.dataset.state = rec.state;
    stateLine.replaceChildren(stateBadge(rec.state), ` ${rec.kind ?? "job"} · ${rec.producer ?? "unknown producer"}` +
      ` · runner ${rec.liveness}${rec.attached ? " · attached (read-only)" : ""}`);
    notice.textContent = explain(rec);
    notice.className = rec.state === "failed" || rec.state === "interrupted" ? "error"
      : rec.state === "stalled" || rec.state === "waiting" || rec.state === "held" ? "warn" : "muted";
    const fraction = rec.state === "complete" ? 1 : st.fraction;
    if (Number.isFinite(fraction)) bar.value = fraction;
    else if (FINAL.has(rec.state)) bar.value = 0;
    else bar.removeAttribute("value");              // indeterminate
    const parts = [];
    if (Number.isFinite(fraction)) parts.push(`${(100 * fraction).toFixed(1)} %`);
    if (st.step) parts.push(`step ${st.step}`);
    if (st.stage) parts.push(st.stage);
    if (Number.isFinite(st.eta_s) && !FINAL.has(rec.state)) parts.push(`ETA ${fmtDuration(st.eta_s)}`);
    if (Number.isFinite(st.elapsed_s)) parts.push(`elapsed ${fmtDuration(st.elapsed_s)}`);
    barText.textContent = parts.join(" · ");
    const final = FINAL.has(rec.state);
    cancel.hidden = final;
    kill.hidden = rec.state !== "stalled";
    resume.hidden = rec.state !== "held";
    runRid = rec.run_dir;
    results.hidden = !runRid;
    if (runRid) {
      results.href = `#/runs/${runRid}`;
      results.textContent = rec.state === "complete" ? "Results" : "Run directory";
    }
    const rows = [["Kind", rec.kind ?? "–"], ["Producer", rec.producer ?? "–"], ["Created", fmtWhen(rec.created)],
      ["Started", fmtWhen(rec.started)], ["Finished", fmtWhen(rec.finished)], ["Lane", rec.gpu ? "GPU" : "CPU"],
      ["Runner pid", rec.pid ?? "–"], ["Exit code", rec.exit_code ?? "–"],
      ["Events", st.events ?? "–"], ["Camera", st.camera ?? "–"],
      ["Window", st.window !== null && st.window !== undefined ? `${st.window} / ${st.windows ?? "?"}` : "–"]];
    facts.replaceChildren(...rows.map(([k, v]) => el("tr", {}, el("th", { textContent: k }),
      el("td", { textContent: String(v) }))));
    for (const w of rec.warnings ?? []) facts.append(el("tr", {}, el("th", { textContent: "Warning" }),
      el("td", { className: "warn", textContent: w })));
    if (runRid && !plots) watchRun();
    if (runRid && final && tracksFor !== rec.state) {
      tracksFor = rec.state;
      api(`/runs/${runRid}/tracks`).then((list) => {
        if (signal.aborted) return;
        tracksLine.textContent = list.map((t) => `${t.label}: ${t.n} marks on the timeline (open the results' ` +
          "events and step through them with [ and ])").join("; ");
      }).catch(() => {});
    }
  }

  // The run's perf plots, fed by run:<rid> (perf messages ask for a refresh).
  function watchRun() {
    plots = new PerfPlots(runRid, { signal });
    perfBox.replaceChildren(el("h3", { textContent: "Performance" }), plots.element);
    plots.refresh().catch(() => {});                // a run directory not made yet: empty until it is
    unsubRun = subscribe([`run:${runRid}`], (msg) => {
      if (msg.op === "perf") plots.onMessage(msg).catch(() => {});
    }, { onSync: () => plots.refresh().catch(() => {}) });
  }

  // The record over HTTP, unless a WebSocket record came while it was read: after the subscription's
  // acknowledgement every change is published, so that one is as new (an HTTP answer read before the hub's tick can
  // resolve after the tick's message).
  async function load() {
    const seen = wsRecords;
    try {
      const r = await api(`/jobs/${id}`);
      if (signal.aborted || seen !== wsRecords) return;
      rec = r;
      show();
    } catch (e) {
      if (!signal.aborted) view.replaceChildren(title, errorLine(e));
    }
  }

  // The log from its byte offset over HTTP (64 kB per request), keeping the last LOG_KEEP characters.
  function readLog(stream) {
    const log = logs[stream];
    if (log.busy) {
      log.again = true;
      return log.busy;
    }
    log.busy = (async () => {
      try {
        do {
          log.again = false;
          for (let i = 0; i < LOG_READS; i++) {
            const r = await apiResponse(`/jobs/${id}/log?since=${log.next}&stream=${stream}`);
            const buf = await r.arrayBuffer();
            if (signal.aborted) return;
            const next = Number(r.headers.get("X-Next-Offset"));
            append(stream, log.decoder.decode(buf, { stream: true }));
            if (!(next > log.next)) break;
            log.next = next;
          }
        } while (log.again && !signal.aborted);
      } catch (e) {
        if (!signal.aborted && stream === shownLog) actionError.textContent = errorText(e);
      } finally {
        log.busy = null;
      }
    })();
    return log.busy;
  }

  function append(stream, text) {
    if (!text) return;
    const log = logs[stream];
    log.text = (log.text + text).slice(-LOG_KEEP);
    if (stream === shownLog) {
      const atEnd = logPre.scrollTop + logPre.clientHeight >= logPre.scrollHeight - 4;
      logPre.textContent = log.text;
      if (atEnd) logPre.scrollTop = logPre.scrollHeight;
    }
  }

  function showLog(stream) {
    shownLog = stream;
    logTabs.set(stream);
    logPre.dataset.stream = stream;
    logPre.textContent = logs[stream].text;
    logPre.scrollTop = logPre.scrollHeight;
  }

  // A `log` message continues the text read so far when it starts where that ends; otherwise the gap is read. Text
  // with U+FFFD may hold bytes the hub replaced (invalid UTF-8, or a character split at the message's end), so its
  // UTF-8 length is not the file's: that range is read over HTTP (byte offsets, a streaming decoder) instead.
  function onLog(msg) {
    const log = logs[msg.stream];
    if (!log || typeof msg.text !== "string") return;
    const offset = Number(msg.offset);
    if (msg.text.includes("\uFFFD")) {
      readLog(msg.stream);
      return;
    }
    const bytes = new TextEncoder().encode(msg.text).length;
    if (!log.busy && offset === log.next) {
      log.next += bytes;
      append(msg.stream, msg.text);
    } else if (offset + bytes > log.next) readLog(msg.stream);
  }

  async function act(path, what) {
    actionError.textContent = "";
    try {
      await api(path, { method: "POST" });
      if (what === "cancel") notice.textContent = "Cancelling…";
    } catch (e) {
      actionError.textContent = errorText(e);
    }
  }

  cancel.addEventListener("click", () => act(`/jobs/${id}/cancel`, "cancel"), { signal });
  kill.addEventListener("click", () => {
    if (confirm(`Kill job "${rec?.label}"? SIGKILL ends its runner and every child at once; the run is left ` +
      "failed or incomplete.")) act(`/jobs/${id}/kill`, "kill");
  }, { signal });
  resume.addEventListener("click", () => act("/jobs/queue/resume", "resume"), { signal });
  showLog("stderr");
  const unsub = subscribe([`job:${id}`], (msg) => {
    if (msg.op === "job" && msg.job?.job_id === id) {
      wsRecords++;
      rec = msg.job;
      show();
    } else if (msg.op === "log") onLog(msg);
  }, { onSync: () => {
    load();
    STREAMS.forEach(readLog);
  } });
  load();
  STREAMS.forEach(readLog);
  return () => {
    life.abort();
    unsub();
    unsubRun?.();
  };
}
