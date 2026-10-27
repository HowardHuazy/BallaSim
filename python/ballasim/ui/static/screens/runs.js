// The runs screen, #/runs[?root=<id>&path=<rel>] and #/jobs (docs/design/phase3/ui.md §3.3 /fs, /jobs; §5.1 "Open a
// run"): the jobs (this jobs root's and the attached ones; live through the WebSocket topic "jobs") and a browser of
// the roots, one directory level at a time. Runs open their results page, jobs their job page (a job directory that
// /jobs does not list, elsewhere in a root, its run's results page), event files and EVIMO2 sequences the viewer.

import { api } from "../app/api.js";
import { subscribe } from "../app/ws.js";
import { el, encodeRid, errorLine, errorText, fmtBytes, fmtDuration, fmtWhen, stateBadge } from "./common.js";

const VIEWABLE = new Set(["events_h5", "evimo2_seq"]);

function jobRow(rec) {
  const where = rec.state === "queued" && rec.position ? ` (${rec.position} in the ${rec.gpu ? "GPU" : "CPU"} lane)`
    : "";
  return el("tr", { dataset: { job: rec.job_id, state: rec.state } },
    el("td", {}, el("a", { href: `#/jobs/${rec.job_id}`, textContent: rec.label })),
    el("td", { textContent: rec.kind ?? "–" }), el("td", { textContent: rec.producer ?? "–" }),
    el("td", {}, stateBadge(rec.state), where),
    el("td", { textContent: Number.isFinite(rec.status?.fraction) && !["complete", "failed", "cancelled"]
      .includes(rec.state) ? `${(100 * rec.status.fraction).toFixed(0)} %` : "" }),
    el("td", { textContent: fmtWhen(rec.created) }),
    el("td", {}, rec.run_dir ? el("a", { href: `#/runs/${rec.run_dir}`, textContent: "run" }) : ""));
}

// A listed job's page; a job /jobs/{id} does not know (in a root, not attached) opens its run, if it has one.
function jobHref(job) {
  if (job?.listed && job.job_id) return `#/jobs/${job.job_id}`;
  return job?.run_dir ? `#/runs/${job.run_dir}` : null;
}

function details(e) {
  if (e.run) {
    const r = e.run;
    return [stateBadge(r.status), " ", stateBadge(r.live), ` format ${r.format}`, r.supplied ? ", supplied bundle" : "",
      r.cameras?.length ? ` · ${r.cameras.join(", ")}` : "", r.events !== null && r.events !== undefined ?
        ` · ${r.events} events` : "", r.simulated_s ? ` · ${fmtDuration(r.simulated_s)} simulated` : "",
      r.wall_s ? ` in ${fmtDuration(r.wall_s)}` : "", r.warnings?.length ? ` · ${r.warnings.length} warning(s)` : ""];
  }
  if (e.job) return [stateBadge(e.job.state), ` ${e.job.kind ?? ""} ${e.job.producer ?? ""}`,
    e.job.listed ? "" : ` · not attached${e.job.run_dir ? ": opens its run" : ""}`];
  return [];
}

export function mountRuns(view, { query, meta }) {
  const life = new AbortController(), signal = life.signal;
  view.classList.add("wide");
  const jobsBody = el("tbody");
  const jobsNote = el("p", { className: "muted" });
  const resume = el("button", { type: "button", name: "resume", textContent: "Resume queue", hidden: true });
  const listing = el("div");
  const roots = meta?.roots ?? [];
  const rootPick = el("select", { name: "root" }, ...roots.map((r) => el("option", { value: r.id,
    textContent: `${r.id}${r.readonly ? " (read-only)" : ""}` })));
  const root = roots.some((r) => r.id === query.get("root")) ? query.get("root") : roots[0]?.id;
  const path = (query.get("path") ?? "").replace(/^\/+|\/+$/g, "");
  if (root) rootPick.value = root;
  view.replaceChildren(el("h2", { textContent: "Jobs" }), el("p", {}, resume, " ", jobsNote),
    el("table", { className: "jobs" }, el("thead", {}, el("tr", {}, ...["label", "kind", "producer", "state", "",
      "created", ""].map((h) => el("th", { textContent: h })))), jobsBody),
    el("h2", { textContent: "Browse" }), el("p", {}, "Root ", rootPick), listing);
  const jobs = new Map();
  let wsRecords = 0;
  const wsSeen = new Map();                         // job id → wsRecords when its last WebSocket record came

  function drawJobs() {
    const recs = [...jobs.values()].sort((a, b) => String(b.created ?? "").localeCompare(String(a.created ?? "")) ||
      a.job_id.localeCompare(b.job_id));
    jobsBody.replaceChildren(...recs.map(jobRow));
    jobsNote.textContent = recs.length ? "" : "No jobs yet.";
    resume.hidden = !recs.some((r) => r.state === "held");
  }

  // The records over HTTP, except a job's whose WebSocket record came while they were read (as new: after the
  // acknowledgement every change is published); a job neither lists nor sent meanwhile is gone.
  async function loadJobs() {
    const seen = wsRecords;
    try {
      const recs = await api("/jobs");
      if (signal.aborted) return;
      const fresh = (jobId) => (wsSeen.get(jobId) ?? 0) > seen;
      const listed = new Set(recs.map((r) => r.job_id));
      for (const jobId of [...jobs.keys()]) if (!listed.has(jobId) && !fresh(jobId)) jobs.delete(jobId);
      for (const r of recs) if (!fresh(r.job_id)) jobs.set(r.job_id, r);
      drawJobs();
    } catch (e) {
      if (!signal.aborted) jobsNote.replaceChildren(errorLine(e));
    }
  }

  async function loadListing() {
    if (!root) {
      listing.replaceChildren(el("p", { className: "muted", textContent: "No roots." }));
      return;
    }
    const parts = path ? path.split("/") : [];
    const crumbs = el("p", { className: "crumbs" }, el("a", { href: `#/runs?root=${encodeURIComponent(root)}`,
      textContent: root }), ...parts.flatMap((p, i) => [" / ", el("a", {
      href: `#/runs?root=${encodeURIComponent(root)}` +
      `&path=${encodeURIComponent(parts.slice(0, i + 1).join("/"))}`, textContent: p })]));
    try {
      const entries = await api(`/fs?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`);
      if (signal.aborted) return;
      const rows = entries.map((e) => {
        const rel = path ? `${path}/${e.name}` : e.name, rid = encodeRid(root, rel);
        const href = e.kind === "dir" ? `#/runs?root=${encodeURIComponent(root)}&path=${encodeURIComponent(rel)}`
          : e.kind === "run" ? `#/runs/${rid}` : e.kind === "job" ? jobHref(e.job)
            : e.kind === "config" ? `#/setup?config=${encodeURIComponent(rid)}`
            : VIEWABLE.has(e.kind) ? `#/view?path=${encodeURIComponent(rid)}` : null;
        return el("tr", { dataset: { name: e.name, kind: e.kind } },
          el("td", {}, href ? el("a", { href, textContent: e.name }) : e.name),
          el("td", { textContent: e.kind }), el("td", {}, ...details(e)),
          el("td", { textContent: fmtWhen(Number(e.mtime_ns) / 1e6) }),
          el("td", { textContent: e.kind === "dir" || e.kind === "run" || e.kind === "job" ? "" : fmtBytes(e.size) }));
      });
      const head = el("thead", {}, el("tr", {}, ...["name", "kind", "", "modified", "size"].map((h) =>
        el("th", { textContent: h }))));
      const table = el("table", { className: "listing" }, head, el("tbody", {}, ...rows));
      listing.replaceChildren(crumbs, rows.length ? table : el("p", { className: "muted",
        textContent: "Empty directory." }));
    } catch (e) {
      if (!signal.aborted) listing.replaceChildren(crumbs, errorLine(e));
    }
  }

  rootPick.addEventListener("change", () => { location.hash = `#/runs?root=${encodeURIComponent(rootPick.value)}`; },
    { signal });
  resume.addEventListener("click", async () => {
    try {
      await api("/jobs/queue/resume", { method: "POST" });
    } catch (e) {
      jobsNote.textContent = errorText(e);
    }
  }, { signal });
  const unsub = subscribe(["jobs"], (msg) => {
    if (msg.op !== "job" || !msg.job?.job_id) return;
    wsSeen.set(msg.job.job_id, ++wsRecords);
    jobs.set(msg.job.job_id, msg.job);
    drawJobs();
  }, { onSync: loadJobs });
  loadJobs();
  loadListing();
  return () => {
    life.abort();
    unsub();
  };
}
