// Hover help of the parameter forms (docs/design/phase3/ui.md §3.6, §5.2): one `popover="manual"` element in the top
// layer, positioned with CSS anchor positioning on whichever label the pointer or the keyboard focus is on (Chrome
// 125 and newer; a browser without it gets the popover below the label from its bounding box). No third-party code.

import { el } from "../screens/common.js";
import { rangeText } from "./units.js";

const ANCHOR = "--bs-help";
// The text of a default for the help: a literal, the family priors with their sources, the derivation's formula.
function defaultLines(row) {
  const d = row.default;
  if (d.kind === "literal") return [`default ${JSON.stringify(d.value)} (${d.status})`];
  if (d.kind === "derived") return [`derived: ${d.formula}`];
  if (d.kind === "required") return ["required: no default"];
  return Object.entries(d.by_family).map(([fam, e]) => `${fam}: ${JSON.stringify(e.value)} (${e.status}` +
    `${e.evimo2_derived ? ", derived from EVIMO2" : ""}${e.no_literature ? ", no literature value" : ""}; ` +
    `${e.source})`);
}

export class Help {
  constructor() {
    this.pop = el("div", { className: "bs-help", popover: "manual", role: "tooltip" });
    document.body.append(this.pop);
    this.target = null;
    this.supported = typeof CSS !== "undefined" && CSS.supports("anchor-name", ANCHOR);
  }

  // The popover content of a row: its schema help first, then what the form knows about it.
  content(row) {
    const lines = [`${row.key}${row.unit && row.unit !== "1" ? ` (${row.unit})` : ""}: ${row.type}` +
      `${row.scale === "log" ? ", log scale" : ""}`];
    if (row.hard) lines.push(`hard range ${rangeText(row.hard)}`);
    if (row.soft) lines.push(`soft range ${rangeText(row.soft)}`);
    if (row.choices) lines.push(`one of ${row.choices.join(", ")}`);
    lines.push(...defaultLines(row));
    if (row.source) lines.push(`source: ${row.source}`);
    lines.push(`${row.class}, stage ${row.stage}, tier hint ${row.tier_hint}`);
    if (row.when || row.also_when) lines.push(`relevant when ${JSON.stringify({ ...row.when, ...row.also_when })}`);
    return [el("p", { className: "help-text", textContent: row.help }),
      el("ul", {}, ...lines.map((l) => el("li", { textContent: l })))];
  }

  show(target, row) {
    this.hide();
    this.target = target;
    this.pop.replaceChildren(...this.content(row));
    if (this.supported) target.style.setProperty("anchor-name", ANCHOR);
    this.pop.showPopover();
    if (!this.supported) {
      const r = target.getBoundingClientRect();
      Object.assign(this.pop.style, { top: `${r.bottom + 4}px`, left: `${Math.max(4, r.left)}px` });
    }
  }

  hide() {
    if (this.pop.matches(":popover-open")) this.pop.hidePopover();
    this.target?.style.removeProperty("anchor-name");
    this.target = null;
  }

  // Hover or focus on `target` shows the row's help; leaving or blurring hides it.
  attach(target, row, signal) {
    const show = () => this.show(target, row), hide = () => this.hide();
    target.addEventListener("pointerenter", show, { signal });
    target.addEventListener("focus", show, { signal });
    target.addEventListener("pointerleave", hide, { signal });
    target.addEventListener("blur", hide, { signal });
  }

  destroy() {
    this.hide();
    this.pop.remove();
  }
}
