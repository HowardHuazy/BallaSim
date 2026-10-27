// The state of a parameter form (docs/design/phase3/ui.md §3.6, §4.9, §4.12, §5.1 "Configure and run"): the form
// model of GET /form, the specification being edited (a base and the user's edits), the last resolve and what it
// says about every row. The widgets (widgets.js) read and write it; the server's C++ resolver does all the checking
// (the client's own range colours are only a preview of it). No third-party code.

import { api, apiResponse } from "../app/api.js";

export const ALL = "default";                     // the "All cameras" tab writes cameras.default.<key>
const DEBOUNCE_MS = 100;                          // a non-scene edit resolves after this quiet time (§5.1 step 2)
const SLOW = ["scene.evimo2", "scene.trajectory", "scene.blender", "dataset.evimo2", "execution"];

export class FormModel {
  constructor(form, profiles) {
    this.form = form;
    this.rows = new Map(form.rows.map((r) => [r.key, r]));
    this.profiles = profiles;
    this.groups = form.groups.map((g) => ({ ...g, open: !SLOW.includes(g.id) }));
  }

  static async load() {
    const [form, profiles] = await Promise.all([api("/form"), api("/profiles")]);
    return new FormModel(form, profiles);
  }

  // The derived accuracy keys of an accuracy profile's table, with the formula the profile gives them (null: a value).
  profileEntry(profile, key) {
    return this.form.accuracy_profiles.find((p) => p.name === profile)?.entries.find((e) => e.key === key) ?? null;
  }
}

// The keys a row's `when`/`also_when` conditions read.
export const dependsOn = (row) => [...new Set([...Object.keys(row.when ?? {}), ...Object.keys(row.also_when ?? {})])];

// The key a tab writes for a row (§3.6): camera rows under cameras.default / cameras.<name>, the rest as they are.
export const fullKey = (row, tab) => (row.scope === "camera" ? `cameras.${tab}.${row.key}` : row.key);

export class FormSession {
  // placeholder: an absolute directory that /resolve takes as output.dir while the user sets none (every UI job writes
  // its own output.dir, §3.6, and the resolver requires one).
  constructor(model, { meta = null, placeholder = null } = {}) {
    this.model = model;
    this.meta = meta;
    this.placeholder = placeholder;
    this.base = null;                             // {config_path} | {config, base_dir} | {run} (§3.6)
    this.profiles = [];
    // full key -> {typed, unit, wire, value, error, soft, scene, applied, appliedWire}: a scene-stage edit is applied
    // once "Apply" is pressed; while its new text is pending it keeps the wire it was applied with (`appliedWire`), so
    // that every resolve describes the applied specification
    this.edits = new Map();
    this.unlocked = new Set();                    // full keys of D-measured rows unlocked for an override
    this.good = null;                             // the last 200 answer of /resolve
    this.state = { phase: "idle" };               // idle | resolving | ok | needs_bundle | error
    this.listeners = new Set();
    this.seq = 0;
    this.timer = null;
  }

  on(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  emit(what) {
    for (const fn of this.listeners) fn(what);
  }

  // ---- the specification ----------------------------------------------------------------------------------------

  setBase(base) {
    this.base = base;
    this.profiles = [];
    this.edits.clear();
    this.unlocked.clear();
    this.good = null;
    this.state = { phase: "idle" };
    this.seq++;                                   // an answer for the old base is dropped
    this.emit("base");
    this.resolveSoon(0);
  }

  setProfiles(ids) {
    this.profiles = ids;
    this.emit("profiles");
    this.resolveSoon();
  }

  // A row's edit under `key`: `value` is {typed, unit, wire, value, error?, soft?}; wire null: not sent.
  setEdit(key, row, value) {
    const scene = row.stage === "scene", prev = this.edits.get(key);
    const appliedWire = scene && prev ? (prev.applied ? prev.wire : prev.appliedWire) : undefined;
    this.edits.set(key, { ...value, scene, applied: !scene, appliedWire });
    this.emit("edit");
    this.resolveSoon();
  }

  clearEdit(key) {
    if (this.edits.delete(key)) {
      this.emit("edit");
      this.resolveSoon();
    }
  }

  unlock(key, on) {
    if (on) this.unlocked.add(key);
    else this.unlocked.delete(key);
    this.emit("edit");
  }

  // The sets of a request: the wire text of every edit that is sent. A scene-stage edit counts as applied (it moves
  // the bundle key, so a new text waits for "Apply", §4.12; until then its applied wire, if any, is sent); `all`: the
  // pending texts too (save, run).
  sets(all = false) {
    const out = {};
    for (const [key, e] of this.edits) {
      const wire = all || e.applied ? e.wire : e.appliedWire;
      if (wire !== null && wire !== undefined) out[key] = wire;
    }
    return out;
  }

  // Whether the text of an edit differs from what the last "Apply" (or a non-scene edit) sends.
  isPending(key) {
    const e = this.edits.get(key);
    return Boolean(e) && !e.applied && (e.wire ?? null) !== (e.appliedWire ?? null);
  }

  pending() {
    return [...this.edits.keys()].filter((k) => this.isPending(k)).length;
  }

  apply() {
    for (const e of this.edits.values()) {
      e.applied = true;
      e.appliedWire = undefined;
    }
    this.emit("edit");
    this.resolveSoon(0);
  }

  // ---- resolve --------------------------------------------------------------------------------------------------

  resolveSoon(delay = DEBOUNCE_MS) {
    clearTimeout(this.timer);
    this.timer = this.base ? setTimeout(() => this.resolve(), delay) : null;
  }

  request(extra = {}) {
    const sets = this.sets();
    if (this.placeholder && !this.edits.has("output.dir")) sets["output.dir"] = JSON.stringify(this.placeholder);
    return { base: this.base, sets, profiles: this.profiles, ...extra };
  }

  async resolve() {
    if (!this.base) return;
    const seq = ++this.seq;
    this.state = { ...this.state, phase: "resolving" };
    this.emit("state");
    let state;
    try {
      const r = await apiResponse("/resolve", { method: "POST", body: this.request() });
      const doc = await r.json();
      state = r.status === 202 ? { phase: "needs_bundle", needs: doc.needs_bundle } : { phase: "ok", doc };
    } catch (e) {
      const body = e.body ?? {};
      state = { phase: "error", message: `${e.code ?? "error"}: ${e.message}`, code: e.code,
        diagnostics: body.diagnostics ?? [], missing: body.missing ?? [] };
    }
    if (seq !== this.seq) return;                 // a newer request is on its way (an older answer is dropped whole)
    if (state.phase === "ok") this.good = state.doc;
    this.state = state;
    this.emit("state");
  }

  // The layers that offer a value for `key` (explain, lowest first), for the row of `tab`.
  async explain(row, tab) {
    const doc = await api("/resolve", { method: "POST", body: this.request({ explain: [row.key] }) });
    const e = doc.explain ?? {};
    if (row.scope !== "camera") return e.run?.[row.key] ?? [];
    const cams = tab === ALL ? Object.keys(e.cameras ?? {}) : [tab];
    return e.cameras?.[cams[0]]?.[row.key] ?? [];
  }

  async save(path) {
    return api("/configs", { method: "POST", body: { path, base: this.base, sets: this.sets(true),
      profiles: this.profiles } });
  }

  // A job's arguments: the base and the edits; the output edits are the job's own `out` and `overwrite` (UI-6 refuses
  // them in `sets`). A `bundle` job builds what the last resolve asked for (the applied edits only) and writes no run.
  jobArgs(kind = "run") {
    const bundle = kind === "bundle", sets = this.sets(!bundle), out = sets["output.dir"];
    delete sets["output.dir"];
    delete sets["output.overwrite"];
    return { ...this.base, sets, profiles: this.profiles,
      ...(out === undefined || bundle ? {} : { out: JSON.parse(out) }) };
  }

  // overwrite: replace an existing output directory (the row output.overwrite, or confirmed by the user).
  async submit(kind, label, { overwrite = false } = {}) {
    const over = kind !== "bundle" && (overwrite || this.edits.get("output.overwrite")?.value === true);
    return api("/jobs", { method: "POST",
      body: { kind, label, args: this.jobArgs(kind), ...(over ? { overwrite: true } : {}) } });
  }

  // ---- what the last resolve says -------------------------------------------------------------------------------

  cameras() {
    return this.good?.cameras ?? [];
  }

  accuracyProfile() {
    return this.good?.record?.run?.["accuracy.profile"]?.value ??
      this.model.rows.get("accuracy.profile")?.default?.value;
  }

  // The resolved entry of a row for a tab: {entry: {value, layer, status}, varies?} or null. The All-cameras tab shows
  // the cameras' common value (varies: they differ; the first is given).
  entry(row, tab) {
    const rec = this.good?.record;
    if (!rec) return null;
    if (row.scope !== "camera") return rec.run?.[row.key] ? { entry: rec.run[row.key] } : null;
    if (tab !== ALL) return rec.cameras?.[tab]?.[row.key] ? { entry: rec.cameras[tab][row.key] } : null;
    const all = this.cameras().map((c) => rec.cameras?.[c]?.[row.key]).filter(Boolean);
    if (!all.length) return null;
    const first = JSON.stringify(all[0].value);
    return { entry: all[0], varies: all.some((e) => JSON.stringify(e.value) !== first) };
  }

  // Whether the last resolve found the row relevant on `tab`: true, false, or null (no answer: no resolve yet, or
  // the answer does not name the row).
  relevant(row, tab) {
    const rel = this.good?.relevant;
    if (!rel) return null;
    const cams = row.scope === "camera" && tab !== ALL ? [tab] : Object.keys(rel);
    const answers = cams.map((c) => rel[c]?.[row.key]).filter((v) => typeof v === "boolean");
    return answers.length ? answers.some(Boolean) : null;
  }

  edited(row, tab) {
    return this.edits.has(fullKey(row, tab));
  }

  // The resolver's diagnostics about a row on `tab`: those naming its key (key or keys) for the tab's camera (or none).
  diagnostics(row, tab) {
    const all = [...(this.state.phase === "error" ? this.state.diagnostics : this.good?.diagnostics ?? [])];
    return all.filter((d) => (d.key === row.key || (d.keys ?? []).includes(row.key)) &&
      (!d.camera || tab === ALL || d.camera === tab || row.scope !== "camera"));
  }

  // Rows are shown unless the last resolve found them irrelevant (§4.9); a row the user set stays, greyed out.
  visibility(row, tab) {
    if (!row.when && !row.also_when) return { hidden: false };
    const rel = this.relevant(row, tab);
    if (rel === true) return { hidden: false };
    if (rel === null) return { hidden: false, depends: dependsOn(row) };
    const set = this.edited(row, tab) || this.diagnostics(row, tab).some((d) => d.kind === "inert");
    return set ? { hidden: false, inert: true } : { hidden: true };
  }
}
