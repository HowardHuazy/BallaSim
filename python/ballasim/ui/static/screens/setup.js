// The setup screen, #/setup[?config=<rid>|run=<rid>] (docs/design/phase3/ui.md §3.6, §4.9, §4.12, §5.1 "Configure
// and run" steps 1 to 6): a specification (a new configuration, a configuration file in a root, or the rebuilt
// specification of a run), the profile stack, camera tabs, and a form of every schema row (params/widgets.js). Each
// edit resolves again through POST /resolve 100 ms later (scene-stage edits wait for "Apply"); a 202 offers a `bundle`
// job and resolves again when it completes; the missing required keys are listed together; the specification can be
// saved (POST /configs) and run (POST /jobs, the edits as the job's `sets`). The server's C++ resolver does all the
// checking. No third-party code.

import { api } from "../app/api.js";
import { subscribe } from "../app/ws.js";
import { Help } from "../params/help.js";
import { ALL, FormModel, FormSession } from "../params/model.js";
import { makeParam, pickPath } from "../params/widgets.js";
import { decodeRid, el, encodeRid, errorLine, errorText, tabs } from "./common.js";

const SOURCES = [["new", "New configuration"], ["config", "Configuration file"], ["run", "Run"]];

function ensureStyles() {
  const href = "/static/params/params.css";
  if (!document.querySelector(`link[href="${href}"]`)) document.head.append(el("link", { rel: "stylesheet", href }));
}

const button = (name, text, props = {}) => el("button", { type: "button", name, textContent: text, ...props });

export function mountSetup(view, { query, meta }) {
  const life = new AbortController(), { signal } = life;
  ensureStyles();
  view.classList.add("wide");
  const roots = meta?.roots ?? [];
  const writable = roots.find((r) => !r.readonly) ?? roots[0];

  // ---- the skeleton
  const sourceKind = el("select", { name: "source" }, ...SOURCES.map(([v, t]) => el("option", { value: v,
    textContent: t })));
  const sourceRef = el("input", { type: "text", name: "ref", size: 48, spellcheck: false,
    placeholder: "resource id or absolute path inside a root" });
  const sourceBrowse = button("source-browse", "Browse…");
  const sourceOpen = button("source-open", "Open");
  const sourceNote = el("span", { className: "muted", dataset: { role: "source" } });
  const profileList = el("ol", { className: "profile-stack" });
  const profileSel = el("select", { name: "profile" });
  const profileAdd = button("profile-add", "Add");
  const status = el("div", { className: "setup-status", dataset: { role: "status" } });
  const missing = el("div", { className: "setup-missing", dataset: { role: "missing" } });
  const warnings = el("div", { dataset: { role: "warnings" } });
  const apply = button("apply", "Apply scene edits", { hidden: true,
    title: "scene-stage edits move the bundle key: they resolve once applied" });
  const build = button("build", "Build the scene bundle", { hidden: true });
  const filter = el("input", { type: "search", name: "filter", placeholder: "filter rows by key or help text" });
  const tabBox = el("div", { dataset: { role: "tabs" } });
  const runBox = el("div", { dataset: { role: "run-rows" } });
  const camBox = el("div", { dataset: { role: "camera-rows" } });
  const savePath = el("input", { type: "text", name: "save-path", size: 56, spellcheck: false });
  const saveDir = button("save-dir", "Directory…");
  const saveBtn = button("save", "Save configuration");
  const saveNote = el("span", { dataset: { role: "save" } });
  const saveWarn = el("span", { className: "warn", dataset: { role: "save-warn" } });
  const jobLabel = el("input", { type: "text", name: "label", value: "run", size: 20 });
  const runBtn = button("run", "Run", { disabled: true });
  const runNote = el("span", { dataset: { role: "run" } });
  const loading = el("p", { className: "muted", textContent: "Loading the form…" });
  view.replaceChildren(el("h2", { textContent: "Setup" }),
    el("div", { className: "setup-bar" }, sourceKind, sourceRef, sourceBrowse, sourceOpen, sourceNote),
    el("div", { className: "setup-bar" }, el("strong", { textContent: "Profiles" }), profileSel, profileAdd),
    profileList,
    status, missing, warnings, el("div", { className: "setup-bar" }, apply, build, filter), loading,
    el("div", { hidden: true, dataset: { role: "form" } }, el("h3", { textContent: "Run" }), runBox,
      el("h3", { textContent: "Cameras" }), tabBox, camBox,
      el("h3", { textContent: "Save and run" }),
      el("p", {}, savePath, " ", saveDir, " ", saveBtn, " ", saveNote, " ", saveWarn),
      el("p", {}, "Job name ", jobLabel, " ", runBtn, " ", runNote)));

  let model = null, session = null, help = null, tab = ALL, params = [], tabNames = null, jobWatch = null;
  // The bundle job the page waits for ({job, key}: the bundle `/resolve` named), the reason one ended without a bundle,
  // a run being submitted, and the absolute path of the opened configuration file.
  let building = null, buildError = "", running = false, source = null;

  const setNote = (node, text, kind = "muted") => {
    node.className = kind;
    node.textContent = text;
  };

  // ---- the form
  function buildForm() {
    const ctx = { session, tab: () => tab, help, signal, dev: Boolean(meta?.dev), roots };
    const advanced = [];
    for (const g of model.groups) {
      const sections = { run: [], camera: [] };
      for (const key of g.keys) {
        const row = model.rows.get(key);
        if ((row.editable === "accuracy" || key.startsWith("accuracy.")) && key !== "accuracy.profile") {
          advanced.push(row);
          continue;
        }
        sections[row.scope === "camera" ? "camera" : "run"].push(row);
      }
      for (const [where, rows] of Object.entries(sections)) {
        if (!rows.length) continue;
        const rowsBox = el("div", { className: "rows" });
        for (const r of rows) {
          const p = makeParam(r, ctx);
          params.push(p);
          rowsBox.append(p);
        }
        const group = el("details", { className: "group", open: g.open, dataset: { group: g.id } },
          el("summary", { textContent: `${g.title} (${rows.length})` }), rowsBox);
        (where === "camera" ? camBox : runBox).append(group);
      }
    }
    const box = el("div", { className: "rows" });
    for (const r of advanced) {
      const p = makeParam(r, ctx);
      params.push(p);
      box.append(p);
    }
    runBox.querySelector('details[data-group="accuracy"] .rows')?.after(el("details", { className: "advanced",
      dataset: { group: "advanced-accuracy" } },
    el("summary", { textContent: `Advanced accuracy (${advanced.length})` }), box));
  }

  const refreshAll = () => params.forEach((p) => p.refresh());

  function syncTabs() {
    const names = session.cameras();
    if (JSON.stringify(names) === tabNames) return;
    tabNames = JSON.stringify(names);
    if (tab !== ALL && !names.includes(tab)) tab = ALL;
    const t = tabs([[ALL, "All cameras"], ...names.map((n) => [n, n])], (name) => {
      tab = name;
      t.set(name);
      tabBox.dataset.tab = name;
      refreshAll();
    });
    t.set(tab);
    tabBox.replaceChildren(t.bar);
    tabBox.hidden = !names.length;
    tabBox.dataset.tab = tab;
  }

  // Shows the row of a full key (cameras.<name>.<key> or a run key), opening the groups around it.
  function goTo(full) {
    const m = /^cameras\.([^.]+)\.(.+)$/.exec(full);
    if (m) {                                        // a camera's tab once the resolver knows the camera, else All
      tab = session.cameras().includes(m[1]) ? m[1] : ALL;
      tabNames = null;
      syncTabs();
      refreshAll();
    }
    const p = params.find((q) => q.row.key === (m ? m[2] : full));
    if (!p) return;
    for (let d = p.closest("details"); d; d = d.parentElement?.closest("details")) d.open = true;
    p.dataset.filtered = "";
    p.scrollIntoView({ block: "center" });
    p.querySelector("input, select")?.focus();
  }

  // ---- status, missing keys, warnings
  function renderStatus() {
    const st = session.state;
    view.dataset.state = st.phase;
    status.dataset.phase = st.phase;
    const pending = session.pending();
    apply.hidden = !pending;
    apply.textContent = `Apply scene edits (${pending})`;
    const busy = st.phase === "needs_bundle" && building !== null && building.key === st.needs.key;
    build.hidden = st.phase !== "needs_bundle" || busy;
    const lines = [];
    if (st.phase === "idle") lines.push(el("p", { textContent: "Open a specification to edit it." }));
    else if (st.phase === "resolving") lines.push(el("p", { className: "muted", textContent: "Resolving…" }));
    else if (st.phase === "ok") {
      const d = st.doc, span = d.span_ns.map((t) => (Number(t) / 1e9).toFixed(3));
      lines.push(el("p", { textContent: `Resolved: ${d.cameras.length} camera(s) ${d.cameras.join(", ")}; span ` +
        `${span[0]}–${span[1]} s; ${d.timing_ms.resolve.toFixed(1)} ms.` }));
      if (d.tier) lines.push(el("p", { textContent: `Tier: ${JSON.stringify(d.tier)}` }));
    } else if (st.phase === "needs_bundle") {
      if (busy) {
        lines.push(el("p", {}, "Building the bundle in ", el("a", { href: `#/jobs/${building.job}`,
          textContent: `job ${building.job}` }), "…"));
      } else {
        lines.push(el("p", { textContent: `The ${st.needs.kind} bundle of ${st.needs.source} is not built; build ` +
          "it, then the form resolves again." }));
        if (buildError) lines.push(el("p", { className: "error", dataset: { role: "build-error" },
          textContent: buildError }));
      }
    } else {
      lines.push(el("p", { className: "error", textContent: st.message }));
      let errors = st.diagnostics.filter((d) => d.severity === "error");
      if (errors.length && st.message.includes(errors[0].message)) errors = errors.slice(1);   // shown above
      if (errors.length) lines.push(el("ul", {}, ...errors.map((d) => el("li", { dataset: { kind: d.kind },
        textContent: d.message }))));
    }
    status.replaceChildren(...lines);
    const miss = st.phase === "error" ? st.missing : [];
    missing.replaceChildren(...(miss.length ? [el("strong", { textContent: `Missing required keys (${miss.length})` }),
      el("ul", {}, ...miss.map((m) => {
        const go = button("goto", m.key, { title: "show the row" });
        go.addEventListener("click", () => goTo(m.key), { signal });
        return el("li", { dataset: { key: m.key } }, go, ` — ${m.hint}`, el("br"), el("span", { className: "muted",
          textContent: m.help }));
      }))] : []));
    const warn = session.good?.diagnostics?.filter((d) => d.severity !== "error") ?? [];
    warnings.replaceChildren(...(warn.length ? [el("details", {}, el("summary", { textContent:
      `Warnings (${warn.length})` }), el("ul", {}, ...warn.map((d) => el("li", { dataset: { kind: d.kind },
      textContent: d.message }))))] : []));
    runBtn.disabled = running || !(st.phase === "ok" && !pending);
  }

  // ---- the profile stack
  function renderProfiles() {
    const info = new Map(model.profiles.map((p) => [p.id, p]));
    profileList.replaceChildren(...session.profiles.map((id, i) => {
      const move = (d) => () => {
        const next = [...session.profiles];
        [next[i], next[i + d]] = [next[i + d], next[i]];
        session.setProfiles(next);
      };
      const up = button("profile-up", "↑", { disabled: i === 0 }), down = button("profile-down", "↓",
        { disabled: i === session.profiles.length - 1 }), rm = button("profile-remove", "remove");
      up.addEventListener("click", move(-1), { signal });
      down.addEventListener("click", move(1), { signal });
      rm.addEventListener("click", () => session.setProfiles(session.profiles.filter((_, j) => j !== i)), { signal });
      const p = info.get(id);
      return el("li", { dataset: { profile: id } }, el("code", { textContent: id }),
        el("span", { className: "muted", textContent: p ? ` ${p.frozen ? "frozen" : ""} ${p.sha256.slice(0, 12)}`
          : " unknown" }), " ", up, down, rm);
    }));
    profileList.hidden = !session.profiles.length;
  }

  // ---- the specification
  const newBase = () => ({ config: { format: "ballasim.run/2" }, base_dir: writable?.path ?? "/" });

  // The bundle job is no longer waited for (a new specification, or a resolve that found no bundle missing).
  function forgetBuild() {
    jobWatch?.();
    jobWatch = null;
    building = null;
    buildError = "";
  }

  function open(kind, ref) {
    forgetBuild();
    const base = kind === "new" ? newBase() : kind === "config" ? { config_path: ref } : { run: ref };
    sourceKind.value = kind;
    sourceRef.value = kind === "new" ? "" : ref;
    sourceRef.disabled = sourceBrowse.disabled = kind === "new";
    setNote(sourceNote, kind === "new" ? `a new configuration (paths start from ${base.base_dir})` : "");
    setNote(saveNote, "");
    setNote(runNote, "");
    history.replaceState(null, "", `#/setup${kind === "new" ? "" : `?${kind}=${encodeURIComponent(ref)}`}`);
    source = kind === "config" ? absolutePath(ref) : null;
    savePath.value = defaultSavePath(kind, ref);
    warnSave();
    session.setBase(base);
  }

  // The absolute path a configuration reference names (an absolute path, or a resource id of a root), or null.
  function absolutePath(ref) {
    if (ref.startsWith("/")) return ref;
    try {
      const [root, rel] = decodeRid(ref), r = roots.find((x) => x.id === root);
      return r ? `${r.path.replace(/\/+$/, "")}/${rel}` : null;
    } catch {
      return null;
    }
  }

  // Saving rewrites the file whole (YAML without comments, every path absolute): it never defaults to the opened file.
  const siblingEdit = (path) => path.replace(/(?<=[^/])(\.[^./]*)?$/, "-edit.yaml");

  function warnSave() {
    saveWarn.textContent = source && savePath.value.trim() === source
      ? "This replaces the opened file; its comments and layout are not kept." : "";
  }

  function defaultSavePath(kind, ref) {
    const dir = writable?.path.replace(/\/+$/, "") ?? "";
    if (kind === "config") {
      if (ref.startsWith("/")) return siblingEdit(ref);
      try {
        const [root, rel] = decodeRid(ref);
        const r = roots.find((x) => x.id === root);
        if (r && !r.readonly) return siblingEdit(`${r.path.replace(/\/+$/, "")}/${rel}`);
        return siblingEdit(`${dir}/${rel.split("/").at(-1)}`);
      } catch {
        return `${dir}/setup.yaml`;
      }
    }
    if (kind === "run") {
      try {
        return `${dir}/${decodeRid(ref)[1].split("/").filter(Boolean).at(-1)}-edit.yaml`;
      } catch { /* fall through */ }
    }
    return `${dir}/setup.yaml`;
  }

  // ---- wiring
  sourceKind.addEventListener("change", () => {
    sourceRef.disabled = sourceBrowse.disabled = sourceKind.value === "new";
  }, { signal });
  sourceOpen.addEventListener("click", () => open(sourceKind.value, sourceRef.value.trim()), { signal });
  sourceBrowse.addEventListener("click", async () => {
    const kind = sourceKind.value;
    const got = await pickPath({ roots, title: kind === "run" ? "Open a run" : "Open a configuration", dirs: false,
      allow: (e) => e.kind === (kind === "run" ? "run" : "config") });
    if (got) open(kind, encodeRid(got.root, got.rel));
  }, { signal });
  profileAdd.addEventListener("click", () => {
    if (profileSel.value && !session.profiles.includes(profileSel.value)) {
      session.setProfiles([...session.profiles, profileSel.value]);
    }
  }, { signal });
  apply.addEventListener("click", () => session.apply(), { signal });
  filter.addEventListener("input", () => {
    const q = filter.value.trim().toLowerCase();
    for (const p of params) {
      p.dataset.filtered = !q || p.row.key.toLowerCase().includes(q) || p.row.help.toLowerCase().includes(q) ? "" : "1";
    }
  }, { signal });
  savePath.addEventListener("input", warnSave, { signal });
  saveDir.addEventListener("click", async () => {
    const got = await pickPath({ roots: roots.filter((r) => !r.readonly), title: "Save in a directory",
      allow: () => false });
    if (got) {
      savePath.value = `${got.abs.replace(/\/+$/, "")}/${savePath.value.split("/").at(-1) || "setup.yaml"}`;
      warnSave();
    }
  }, { signal });
  saveBtn.addEventListener("click", async () => {
    setNote(saveNote, "Saving…");
    try {
      const r = await session.save(savePath.value.trim());
      setNote(saveNote, `Saved ${r.path} (sha256 ${r.sha256.slice(0, 12)}).`);
    } catch (e) {
      setNote(saveNote, errorText(e) + (e.body?.diagnostics?.length ? `: ${e.body.diagnostics[0].message}` : ""),
        "error");
    }
  }, { signal });
  // Submits the run once (the button stays disabled while the request is out, and for good once it succeeds: the page
  // goes to the job). A 409 for an existing output offers to replace it.
  async function startRun(overwrite = false) {
    if (running) return;
    running = true;
    setNote(runNote, "");
    renderStatus();
    try {
      const job = await session.submit("run", jobLabel.value.trim() || "run", { overwrite });
      location.hash = `#/jobs/${job.job_id}`;
    } catch (e) {
      running = false;
      setNote(runNote, errorText(e), "error");
      if (e.code === "BS-UI-RUN-EXISTS") {
        const again = button("run-overwrite", "Overwrite and run");
        again.addEventListener("click", () => startRun(true), { signal });
        runNote.append(" ", again);
      }
      renderStatus();
    }
  }
  runBtn.addEventListener("click", () => startRun(), { signal });

  // A bundle job ended: complete resolves again (that answer says whether the bundle is there); anything else is
  // reported with the Build button back.
  function buildDone(id, rec) {
    if (building?.job !== id) return;                 // an older job, or the specification was replaced
    if (rec.state === "complete") {
      jobWatch?.();
      jobWatch = null;
      building = null;
      session.resolveSoon(0);
    } else if (["failed", "cancelled", "interrupted"].includes(rec.state)) {
      jobWatch?.();
      jobWatch = null;
      building = null;
      buildError = `The bundle job ${rec.state}.`;
      renderStatus();
    }
  }

  build.addEventListener("click", async () => {
    const key = session.state.needs?.key;
    build.disabled = true;
    buildError = "";
    try {
      const job = await session.submit("bundle", "bundle");
      jobWatch?.();
      building = { job: job.job_id, key };
      renderStatus();                                 // "Building the bundle in job …", the button hidden
      jobWatch = subscribe([`job:${job.job_id}`], (msg) => msg.op === "job" && buildDone(job.job_id, msg.job),
        { onSync: async () => buildDone(job.job_id, await api(`/jobs/${job.job_id}`)) });
    } catch (e) {
      buildError = errorText(e);
      renderStatus();
    } finally {
      build.disabled = false;
    }
  }, { signal });

  (async () => {
    help = new Help();
    model = await FormModel.load();
    if (signal.aborted) return;
    session = new FormSession(model, { meta, placeholder: writable ? `${writable.path.replace(/\/+$/, "")}/.ui-resolve`
      : null });
    profileSel.replaceChildren(...model.profiles.map((p) => el("option", { value: p.id, textContent: p.id })));
    buildForm();
    session.on((what) => {
      if (what === "edit") renderStatus();
      else {
        if (["ok", "error"].includes(session.state.phase)) forgetBuild();
        if (what === "profiles" || what === "base") renderProfiles();
        syncTabs();
        refreshAll();
        renderStatus();
      }
    });
    loading.remove();
    view.querySelector('[data-role="form"]').hidden = false;
    renderProfiles();
    renderStatus();
    view.dataset.ready = "1";
    if (query.get("run")) open("run", query.get("run"));
    else if (query.get("config")) open("config", query.get("config"));
    else open("new", "");
  })().catch((e) => {
    loading.replaceWith(errorLine(e));
  });

  return () => {
    life.abort();
    delete view.dataset.ready;
    jobWatch?.();
    clearTimeout(session?.timer);
    session?.listeners.clear();
    help?.destroy();
  };
}
