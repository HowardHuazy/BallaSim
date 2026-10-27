// The widgets of the parameter forms (docs/design/phase3/ui.md §3.6 widgets, §4.9): one <bs-param> element per schema
// row, with a slider and number field (exact decimal units, log steppers, a "0" stop), a toggle, a select, a vector
// field, a text field or a server-side path picker, its provenance and badge chips, the resolver's messages and the
// help popover. A row reads the FormSession (model.js) and writes its edits; every check is the resolver's, the
// range colours are a preview. No third-party code.

import { api } from "../app/api.js";
import { el } from "../screens/common.js";
import { fullKey } from "./model.js";
import { fromNumber, fromValue, listFromText, numberFromText, parseDecimal, rangeText, SLIDER_STEPS, showDecimal,
  showList, showNumber, sliderPosition, sliderValue, splitTyped, stepDecimal, unitsOf, violation } from "./units.js";

// `title` is set only when given: el() assigns the property, and an undefined title would read "undefined".
const button = (name, text, title) => el("button", { type: "button", name, textContent: text,
  ...(title ? { title } : {}) });
const OPENABLE = new Set(["dir", "run", "job", "evimo2_seq", "signal_cache"]);   // /fs kinds that are directories

// ---- the server-side path picker ----------------------------------------------------------------------------------

// A dialog over the roots (GET /fs, one directory level at a time). `allow(entry)` says which entries can be used (a
// directory can always be opened); `dirs`: the current directory itself can be used. Resolves to
// {abs, rid-parts, entry} or null.
export function pickPath({ roots, title, allow = () => true, dirs = true, start = null }) {
  return new Promise((resolve) => {
    const dialog = el("dialog", { className: "bs-picker" });
    let root = start?.root ?? roots[0]?.id, rel = start?.rel ?? "";
    const rootSel = el("select", { name: "root" }, ...roots.map((r) => el("option", { value: r.id,
      textContent: r.id })));
    const crumbs = el("p", { className: "muted" });
    const list = el("div", { dataset: { role: "entries" } });
    const useHere = button("use-here", "Use this directory");
    const done = (value) => {
      dialog.close();
      dialog.remove();
      resolve(value);
    };
    const absolute = (r, rel_) => `${roots.find((x) => x.id === r).path.replace(/\/+$/, "")}${rel_ ? `/${rel_}` : ""}`;
    const result = (entry, rel_) => ({ abs: absolute(root, rel_), root, rel: rel_, entry });

    async function load() {
      rootSel.value = root;
      crumbs.textContent = `${root}:/${rel}`;
      useHere.hidden = !dirs;
      list.replaceChildren(el("p", { className: "muted", textContent: "Loading…" }));
      try {
        const entries = await api(`/fs?root=${encodeURIComponent(root)}&path=${encodeURIComponent(rel)}`);
        const up = rel ? [el("tr", {}, el("td", {}, button("up", "..", "parent directory")))] : [];
        const rows = entries.map((e) => {
          const next = rel ? `${rel}/${e.name}` : e.name, open = OPENABLE.has(e.kind);
          const tr = el("tr", { dataset: { name: e.name, kind: e.kind } },
            el("td", {}, open ? button("open", e.name, "open") : e.name), el("td", { textContent: e.kind }),
            el("td", {}, allow(e) ? button("use", "Use") : ""));
          tr.querySelector('button[name="open"]')?.addEventListener("click", () => {
            rel = next;
            load();
          });
          tr.querySelector('button[name="use"]')?.addEventListener("click", () => done(result(e, next)));
          return tr;
        });
        list.replaceChildren(el("table", { className: "listing" }, el("tbody", {}, ...up, ...rows)));
        list.querySelector('button[name="up"]')?.addEventListener("click", () => {
          rel = rel.split("/").slice(0, -1).join("/");
          load();
        });
      } catch (e) {
        list.replaceChildren(el("p", { className: "error", textContent: `${e.code ?? "error"}: ${e.message}` }));
      }
    }

    rootSel.addEventListener("change", () => {
      root = rootSel.value;
      rel = "";
      load();
    });
    useHere.addEventListener("click", () => done(result({ name: rel.split("/").at(-1) || root, kind: "dir" }, rel)));
    const cancel = button("cancel", "Cancel");
    cancel.addEventListener("click", () => done(null));
    dialog.addEventListener("cancel", (e) => {
      e.preventDefault();
      done(null);
    });
    dialog.append(el("h3", { textContent: title }), el("p", {}, "Root ", rootSel), crumbs, list,
      el("p", {}, useHere, " ", cancel));
    document.body.append(dialog);
    dialog.showModal();
    load();
  });
}

// ---- the row element ---------------------------------------------------------------------------------------------

// ctx: {session, tab(): the shown tab, help, signal, dev: whether E-heuristic rows are editable, roots}.
export class BsParam extends HTMLElement {
  init(row, ctx) {
    this.row = row;
    this.ctx = ctx;
    this.unit = unitsOf(row.unit)[0]?.label ?? null;
    this.placeholder = "";
    Object.assign(this.dataset, { key: row.key, scope: row.scope, editable: row.editable, widget: row.widget,
      stage: row.stage });
    const { signal } = ctx;
    this.name = el("label", { className: "name", tabIndex: 0, textContent: row.key });
    ctx.help.attach(this.name, row, signal);
    this.chips = el("span", { className: "chips" });
    this.messages = el("span", { className: "messages" });
    this.why = el("details", { className: "why" }, el("summary", { textContent: "layers" }));
    this.why.hidden = true;
    this.why.addEventListener("toggle", () => this.why.open && this.explain(), { signal });
    this.control = el("span", { className: "control" });
    this.build();
    this.reset = button("reset", "×", "drop the edit: back to the resolved value");
    this.reset.hidden = true;
    this.reset.addEventListener("click", () => {
      const s = ctx.session, key = this.fk;
      s.unlock(key, false);
      s.clearEdit(key);
      this.refresh();
    }, { signal });
    this.append(this.name, this.control, this.reset, el("div", { className: "meta" }, this.chips, this.why),
      this.messages);
    return this;
  }

  get fk() {
    return fullKey(this.row, this.ctx.tab());
  }

  // ---- building the controls

  build() {
    const { row, ctx } = this, { signal } = ctx;
    const on = (target, type, fn) => target.addEventListener(type, fn, { signal });
    const w = row.widget;
    if (w === "slider" || w === "number") {
      this.field = el("input", { type: "text", className: "num", inputMode: "decimal", name: "value",
        autocomplete: "off", spellcheck: false });
      on(this.field, "input", () => this.submit(this.field.value.trim()));
      on(this.field, "change", () => this.normalize());
      on(this.field, "keydown", (e) => {
        if (e.key === "ArrowUp" || e.key === "ArrowDown") {
          e.preventDefault();
          this.step(e.key === "ArrowUp" ? 1 : -1);
        }
      });
      const parts = [];
      if (w === "slider") {
        const t = row.track;
        this.range = el("input", { type: "range", name: "slider", min: t.zero_stop ? -1 : 0, max: SLIDER_STEPS, step: 1,
          value: 0 });
        on(this.range, "input", () => this.slide());
        parts.push(this.range);
      }
      parts.push(this.field);
      const units = unitsOf(row.unit);
      if (units.length > 1) {
        this.unitSel = el("select", { name: "unit", className: "unit" }, ...units.map((u) => el("option",
          { value: u.label, textContent: u.label })));
        on(this.unitSel, "change", () => this.changeUnit());
        parts.push(this.unitSel);
      } else if (units.length) parts.push(el("span", { className: "unit muted", textContent: units[0].label }));
      this.down = button("step-down", "−", row.scale === "log" ? "× 2^(−1/4)" : "− one unit of the third digit");
      this.up = button("step-up", "+", row.scale === "log" ? "× 2^(1/4)" : "+ one unit of the third digit");
      on(this.down, "click", () => this.step(-1));
      on(this.up, "click", () => this.step(1));
      parts.push(this.down, this.up);
      if (row.nullable) parts.push(this.nullBox());
      this.control.append(...parts);
    } else if (w === "toggle") {
      this.box = el("input", { type: "checkbox", name: "value" });
      on(this.box, "change", () => this.setEdit({ typed: String(this.box.checked), wire: String(this.box.checked),
        value: this.box.checked }));
      this.control.append(this.box);
    } else if (w === "select") {
      this.select = el("select", { name: "value" }, el("option", { value: "", textContent: "" }),
        ...row.choices.map((c) => el("option", { value: c, textContent: c })));
      on(this.select, "change", () => this.submit(this.select.value));
      this.control.append(this.select);
    } else {
      this.field = el("input", { type: "text", name: "value", autocomplete: "off", spellcheck: false });
      if (w === "vector") {
        this.placeholder = row.length === "any" || row.length === null ? "values, separated by commas"
          : `${row.length} values, separated by commas`;
        this.field.placeholder = this.placeholder;
      }
      on(this.field, "input", () => this.submit(this.field.value.trim()));
      this.control.append(this.field);
      if (w === "path") {
        this.browse = button("browse", "Browse…");
        on(this.browse, "click", () => this.pick());
        this.control.append(this.browse);
      }
      if (row.nullable) this.control.append(this.nullBox());
    }
    if (row.editable === "data") {
      this.lock = button("unlock", "Unlock", "override the measured value (the CLI layer)");
      on(this.lock, "click", () => {                 // Unlock, or Lock again: the override is dropped
        const s = ctx.session, key = this.fk;
        if (s.unlocked.has(key) || s.edits.has(key)) {
          s.unlock(key, false);
          s.clearEdit(key);
        } else s.unlock(key, true);
        this.refresh();
      });
      this.control.append(this.lock);
    }
  }

  nullBox() {
    this.nullInput = el("input", { type: "checkbox", name: "null" });
    this.nullInput.addEventListener("change", () => {
      if (this.nullInput.checked) this.setEdit({ typed: "null", wire: "", value: null, isNull: true });
      else {
        this.ctx.session.clearEdit(this.fk);
        this.refresh();
      }
    }, { signal: this.ctx.signal });
    return el("label", { className: "null" }, this.nullInput, " null");
  }

  // ---- edits

  setEdit(edit) {
    this.ctx.session.setEdit(this.fk, this.row, { unit: this.unit, ...edit });
    this.refresh();
  }

  // The typed text of a number, vector, text, path or select row as an edit (empty: drop it).
  submit(typed) {
    if (typed === "") {
      this.ctx.session.clearEdit(this.fk);
      this.refresh();
      return;
    }
    const r = this.parse(typed);
    if (r.error) this.setEdit({ typed, unit: r.unit ?? this.unit, wire: null, error: r.error });
    else this.setEdit({ typed, unit: r.unit ?? this.unit, wire: r.text, value: r.value, soft: r.soft ?? false });
  }

  parse(typed) {
    const { row } = this;
    switch (row.widget) {
      case "slider":
      case "number": {
        const r = numberFromText(row, typed, this.unit);
        if (r.error) return r;
        const v = violation(row, r.value);
        if (v === "hard") {
          return { error: `outside the hard range ${rangeText(row.hard, row.unit)}: not sent`, unit: r.unit };
        }
        return { ...r, soft: v === "soft" };
      }
      case "vector": return listFromText(row, typed);
      case "path":
        if (!/^[/$~]/.test(typed)) return { error: "an absolute path (or one starting with $ or ~)" };
        return { text: JSON.stringify(typed), value: typed };
      case "select":
        return row.choices.includes(typed) ? { text: JSON.stringify(typed), value: typed }
          : { error: `one of ${row.choices.join(", ")}` };
      default: return { text: JSON.stringify(typed), value: typed };
    }
  }

  // A typed unit ("1.5 µs") moves into the unit selector once the field is left.
  normalize() {
    const edit = this.ctx.session.edits.get(this.fk);
    const parts = this.unitSel && edit && !edit.error ? splitTyped(this.field.value) : null;
    if (parts?.unit) {
      this.unit = edit.unit;
      this.field.value = parts.number;
      this.submit(parts.number);
    }
  }

  // Another display unit. A valid edit keeps its value: the wire text (the schema unit) is shifted as an exact
  // decimal, so an integer beyond 2^53 keeps its digits. An edit in error is read again in the new unit, which is how
  // a forgotten unit is fixed; a null edit just takes the unit (the selector must not snap back).
  changeUnit() {
    const { row } = this, edit = this.ctx.session.edits.get(this.fk);
    this.unit = this.unitSel.value;
    if (edit?.isNull) edit.unit = this.unit;
    else if (edit) {
      const d = edit.error ? null : parseDecimal(edit.wire);
      if (d) this.field.value = showDecimal(row, d, this.unit);
      this.submit(this.field.value.trim());
    }
    this.refresh();
  }

  slide() {
    const { row } = this, i = Number(this.range.value);
    const v = i < 0 ? 0 : sliderValue(i / SLIDER_STEPS, row.track, row.scale, row.type === "int");
    this.field.value = showNumber(row, v, this.unit);
    this.submit(this.field.value);
  }

  step(dir) {
    const { row } = this, s = this.ctx.session, edit = s.edits.get(this.fk);
    let cur = edit && !edit.error && Number.isFinite(edit.value) ? fromNumber(edit.value) : null;
    cur ??= fromValue(s.entry(row, this.ctx.tab())?.entry.value) ?? fromNumber(row.track?.lo ?? row.hard?.lo ?? 0);
    if (!cur) return;
    this.field.value = showDecimal(row, stepDecimal(cur, dir, row), this.unit);
    this.submit(this.field.value);
  }

  async pick() {
    const { row } = this;
    const dirOnly = row.key === "output.dir";
    const got = await pickPath({ roots: this.ctx.roots, title: row.key, dirs: dirOnly || row.key === "scene.path",
      allow: dirOnly ? (e) => e.kind === "dir" : () => true });
    if (got) {
      this.field.value = got.abs;
      this.submit(got.abs);
    }
  }

  async explain() {
    try {
      const layers = await this.ctx.session.explain(this.row, this.ctx.tab());
      this.why.replaceChildren(el("summary", { textContent: "layers" }), el("ol", {}, ...layers.map((l) => el("li", {
        textContent: `${l.layer} · ${l.status}: ${JSON.stringify(l.value)}${l.source ? ` (${l.source})` : ""}` }))));
    } catch (e) {
      this.why.append(el("span", { className: "error", textContent: `${e.code ?? "error"}: ${e.message}` }));
    }
  }

  // ---- showing the state

  locked() {
    const { row, ctx } = this, key = this.fk;
    if (row.editable === "data") return !ctx.session.unlocked.has(key) && !ctx.session.edits.has(key);
    return row.editable === "heuristic" && !ctx.dev;
  }

  refresh() {
    const { row, ctx } = this, s = ctx.session, tab = ctx.tab(), key = this.fk;
    const edit = s.edits.get(key), res = s.entry(row, tab), vis = s.visibility(row, tab);
    this.hidden = vis.hidden;
    this.dataset.inert = vis.inert ? "1" : "";
    this.dataset.state = !edit ? "default" : edit.error ? "error" : edit.soft ? "soft" : "edited";
    const locked = this.locked();
    this.dataset.locked = locked ? "1" : "";
    for (const c of this.control.querySelectorAll("input, select, button")) {
      if (c !== this.lock) c.disabled = locked;
    }
    if (this.lock) this.lock.textContent = s.unlocked.has(key) || edit ? "Lock" : "Unlock";
    this.reset.hidden = !edit;
    this.showValue(edit, res);
    this.showChips(edit, res, vis);
    this.showMessages(edit, tab);
  }

  showValue(edit, res) {
    const { row } = this, focus = document.activeElement;
    if (edit?.unit && this.unitSel) this.unit = edit.unit;
    if (this.unitSel) this.unitSel.value = this.unit;
    const value = res?.entry.value;
    if (this.field && focus !== this.field) {
      this.field.value = edit ? (edit.isNull ? "" : edit.typed) : res?.varies ? "" : this.format(value);
      this.field.placeholder = edit?.isNull || (!edit && value === null) ? "null" : res?.varies ? "varies by camera"
        : res ? this.placeholder : this.hint();
    }
    if (this.select) this.select.options[0].textContent = edit || res ? "" : this.hint();
    if (this.nullInput) {
      this.nullInput.checked = Boolean(edit?.isNull);
      if (edit?.isNull) this.field.disabled = true;
    }
    if (this.box) {
      this.box.checked = edit ? edit.value : Boolean(value ?? (row.default.kind === "literal" && row.default.value));
    }
    if (this.select) this.select.value = edit ? edit.typed : typeof value === "string" ? value : "";
    if (this.range) {
      const v = edit && !edit.error ? edit.value : typeof value === "number" ? value : null;
      this.range.value = v === null || !Number.isFinite(v) ? 0 : v === 0 && row.track.zero_stop ? -1
        : Math.round(sliderPosition(v, row.track, row.scale) * SLIDER_STEPS);
    }
  }

  // What a row shows before the resolver has answered: its default (§3.6 `default`).
  hint() {
    const { row } = this, d = row.default;
    if (d.kind === "literal") return d.value === null ? "null" : this.format(d.value);
    return d.kind === "family" ? "family prior" : d.kind;
  }

  format(value) {
    const { row } = this;
    if (value === null || value === undefined) return "";
    if (row.widget === "slider" || row.widget === "number") return showNumber(row, value, this.unit);
    if (row.widget === "vector") return showList(row, value);
    return String(value);
  }

  showChips(edit, res, vis) {
    const { row, ctx } = this, s = ctx.session, chips = [];
    const chip = (text, kind, title) => el("span", { className: `chip ${kind}`, textContent: text,
      ...(title ? { title } : {}) });
    if (edit) chips.push(chip(edit.wire === null ? "not sent" : "edited", "edit"));
    else if (res) {
      chips.push(chip(res.varies ? "varies by camera" : `${res.entry.layer} · ${res.entry.status}`, "layer",
        "where the value comes from; open 'layers' for every layer that offers one"));
      const profile = s.accuracyProfile();
      const formula = row.editable === "accuracy" ? s.model.profileEntry(profile, row.key)?.formula : null;
      if (formula && res.entry.status === "derived") chips.push(chip(`derived: ${formula}`, "formula"));
    } else if (row.default.kind === "required") chips.push(chip("required", "required"));
    if (row.key === "output.dir" && !edit) {
      chips.push(chip("placeholder while resolving: a job writes its own output (out)", "placeholder"));
    }
    if (row.editable === "data") {
      chips.push(chip("measured", "data", "a D-measured value: read-only until unlocked"));
      if (edit?.wire !== null && edit) chips.push(chip("measured value overridden", "override"));
    }
    if (row.editable === "heuristic") {
      chips.push(chip(ctx.dev ? "heuristic (--dev)" : "heuristic: read-only", "heuristic"));
    }
    if (s.isPending(this.fk)) chips.push(chip("pending: Apply", "pending"));
    if (vis.depends) chips.push(chip(`depends on ${vis.depends.join(", ")}`, "depends"));
    if (vis.inert) chips.push(chip("inert here", "inert"));
    const feeds = row.feeds?.[s.accuracyProfile()] ?? [];
    if (feeds.length) {
      const label = feeds.map((k) => `${k} (${s.model.rows.get(k)?.stage ?? "?"})`).join(", ");
      chips.push(chip(`may move ${label} under ${s.accuracyProfile()}`, "feeds"));
    }
    chips.push(chip(`tier ${row.tier_hint}`, "tier", "a hint from the row's stage"));
    this.chips.replaceChildren(...chips);
    this.why.hidden = !res;
  }

  showMessages(edit, tab) {
    const { row } = this, msgs = [];
    const add = (text, kind) => msgs.push(el("span", { className: `msg ${kind}`, textContent: text }));
    if (edit?.error) add(edit.error, "error");
    else if (edit?.soft) add(`outside the soft range ${rangeText(row.soft, row.unit)} (${row.source})`, "warn");
    for (const d of this.ctx.session.diagnostics(row, tab)) {
      if (edit?.soft && d.kind === "soft_range") continue;
      add(d.message, d.severity === "error" ? "error" : "warn");
    }
    this.messages.replaceChildren(...msgs);
    this.dataset.diag = msgs.some((m) => m.classList.contains("error")) ? "error" : msgs.length ? "warn" : "";
  }
}

customElements.define("bs-param", BsParam);

export const makeParam = (row, ctx) => document.createElement("bs-param").init(row, ctx);
