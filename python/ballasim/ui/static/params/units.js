// Numbers of the parameter forms (docs/design/phase3/ui.md §3.6 value text, §4.9 units and sliders). Values travel as
// decimal text into the CLI layer, so every conversion here is exact decimal arithmetic on (digits, exponent) pairs:
// a display unit shifts the exponent, an integer row refuses a result with a fractional part (the resolver refuses a
// float or an exponent, RV2 int_exponent), and a binary product never touches the text. Numbers (doubles) are used
// only to compare with ranges and to place sliders. Pure functions; no DOM. No third-party code.

const DECIMAL = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/;
const WITH_UNIT = /^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$/;
const MAX_EXP = 400;                              // a decimal exponent beyond doubles and 64-bit integers alike

// A decimal value: {neg, digits: BigInt >= 0, exp}, the value being (-1)^neg · digits · 10^exp, without trailing
// zeros in `digits` (zero is {digits: 0n, exp: 0}).
function make(neg, digits, exp) {
  if (digits === 0n) return { neg: false, digits: 0n, exp: 0 };
  while (digits % 10n === 0n) {
    digits /= 10n;
    exp++;
  }
  return { neg, digits, exp };
}

export function parseDecimal(text) {
  const m = DECIMAL.exec(String(text).trim());
  if (!m) return null;
  const [, sign, whole = "", frac = "", ex] = m;
  if (!whole && !frac) return null;
  const exp = (ex ? Number(ex) : 0) - frac.length;
  if (!Number.isFinite(exp) || Math.abs(exp) > 10 * MAX_EXP) return null;
  return make(sign === "-", BigInt(whole + frac), exp);
}

export const shift = (d, n) => (d.digits === 0n ? d : { ...d, exp: d.exp + n });

export function fromNumber(x) {
  return Number.isFinite(x) ? parseDecimal(String(x)) : null;
}

// A record or default value (JSON number, or a decimal string for an integer beyond 2^53).
export function fromValue(v) {
  if (typeof v === "number") return fromNumber(v);
  if (typeof v === "string" || typeof v === "bigint") return parseDecimal(String(v));
  return null;
}

export function toNumber(d) {
  return Number(`${d.neg ? "-" : ""}${d.digits}e${d.exp}`);
}

// The scientific exponent of a non-zero decimal (1.5e-6 → -6).
const magnitude = (d) => d.exp + String(d.digits).length - 1;

// The text of a decimal. kind "float": always a float in YAML (a point or an exponent), plain between 1e-4 and 1e16;
// "int": plain digits, null when the value has a fractional part; "show": for display, plain, no ".0".
export function decimalText(d, kind = "float") {
  const sign = d.neg ? "-" : "";
  const s = String(d.digits);
  if (kind === "int") {
    if (d.exp < 0 || d.exp > MAX_EXP) return null;
    return d.digits === 0n ? "0" : sign + s + "0".repeat(d.exp);
  }
  const dotZero = kind === "float" ? ".0" : "";
  if (d.digits === 0n) return `0${dotZero}`;
  const sci = magnitude(d);
  if (kind === "show" ? sci < -5 || sci >= 12 : sci < -4 || sci >= 16) {
    return `${sign}${s[0]}${s.length > 1 ? `.${s.slice(1)}` : dotZero}e${sci}`;
  }
  if (d.exp >= 0) return `${sign}${s}${"0".repeat(d.exp)}${dotZero}`;
  const point = s.length + d.exp;
  if (point > 0) return `${sign}${s.slice(0, point)}.${s.slice(point)}`;
  return `${sign}0.${"0".repeat(-point)}${s}`;
}

// Whether a decimal is a finite double / a safe value of its type (the resolver's own checks follow).
export const inDoubleRange = (d) => d.digits === 0n || (magnitude(d) <= 308 && magnitude(d) >= -323);

// ---- display units --------------------------------------------------------------------------------------------

// [label, power]: one label is 10^power of the row's schema unit. The first is the schema unit itself.
const FAMILIES = {
  s: [["s", 0], ["ms", -3], ["µs", -6], ["ns", -9]],
  ns: [["ns", 0], ["µs", 3], ["ms", 6], ["s", 9]],
  Hz: [["Hz", 0], ["kHz", 3], ["MHz", 6]],
  m: [["m", 0], ["mm", -3], ["µm", -6], ["nm", -9]],
  um: [["µm", 0], ["nm", -3], ["mm", 3]],
  nm: [["nm", 0], ["µm", 3], ["mm", 6]],
  fA: [["fA", 0], ["pA", 3], ["nA", 6], ["µA", 9]],
  MB: [["MB", 0], ["GB", 3]],
};
const ALIASES = { us: "µs", μs: "µs", um: "µm", μm: "µm", uA: "µA", μA: "µA" };

export function unitsOf(unit) {
  const family = FAMILIES[unit];
  if (family) return family.map(([label, power]) => ({ label, power }));
  return unit && unit !== "1" ? [{ label: unit, power: 0 }] : [];
}

// The number and unit of a typed text → {decimal, label}: the unit is the row's unit (or the display unit) when the
// text names none.
export function splitTyped(text) {
  const m = WITH_UNIT.exec(text);
  if (!m) return null;
  const d = parseDecimal(m[1]);
  return d ? { decimal: d, number: m[1], unit: m[2] || null } : null;
}

// ---- value text -----------------------------------------------------------------------------------------------

// The value of a number row typed in display unit `unit` (a label of unitsOf(row.unit), or null: the schema unit):
// {text: the wire text, value: the Number in the schema unit, display: the typed value without a unit} or {error}.
export function numberFromText(row, typed, unit = null) {
  const units = unitsOf(row.unit);
  const parts = splitTyped(typed);
  if (!parts) return { error: "not a number" };
  let label = unit ?? units[0]?.label ?? null;
  if (parts.unit) {
    const name = ALIASES[parts.unit] ?? parts.unit;
    const known = units.find((u) => u.label === name || u.label === ALIASES[name]);
    if (!known) {
      return { error: units.length ? `unknown unit ${parts.unit} (${units.map((u) => u.label).join(", ")})`
        : "this value has no unit" };
    }
    label = known.label;
  }
  const power = units.find((u) => u.label === label)?.power ?? 0;
  const d = shift(parts.decimal, power);
  if (row.type === "int") {
    if (d.digits !== 0n && magnitude(d) >= 30) return { error: "out of range" };      // beyond 64-bit integers
    const text = decimalText(d, "int");
    if (text === null) return { error: `whole ${units[0]?.label ?? "numbers"} only` };
    return { text, value: Number(text), unit: label };
  }
  if (!inDoubleRange(d)) return { error: "out of range" };
  return { text: decimalText(d, "float"), value: toNumber(d), unit: label };
}

// A decimal (schema unit) as the text shown in display unit `unit` (plain digits for a whole integer).
export function showDecimal(row, d, unit = null) {
  const power = unitsOf(row.unit).find((u) => u.label === unit)?.power ?? 0;
  const shown = shift(d, -power);
  return decimalText(shown, row.type === "int" && shown.exp >= 0 ? "int" : "show");
}

// A value (a JSON number, or a decimal string for an integer beyond 2^53) as the text shown in display unit `unit`.
export function showNumber(row, value, unit = null) {
  const d = fromValue(value);
  return d ? showDecimal(row, d, unit) : "";
}

// ---- ranges ---------------------------------------------------------------------------------------------------

// Whether v lies outside the closed or open range {lo, hi, lo_open, hi_open} (null bounds are open ends).
export function outside(range, v) {
  if (!range) return false;
  if (range.lo !== null && range.lo !== undefined && (range.lo_open ? v <= range.lo : v < range.lo)) return true;
  return range.hi !== null && range.hi !== undefined && (range.hi_open ? v >= range.hi : v > range.hi);
}

export const rangeText = (range, unit) => {
  const u = unit && unit !== "1" ? ` ${unit}` : "";
  const lo = range.lo === null || range.lo === undefined ? "" : String(range.lo);
  const hi = range.hi === null || range.hi === undefined ? "" : String(range.hi);
  return `${range.lo_open ? "(" : "["}${lo || "–∞"}, ${hi || "∞"}${range.hi_open ? ")" : "]"}${u}`;
};

// "hard", "soft" or null for a value in the schema unit (a hard violation wins).
export function violation(row, v) {
  if (outside(row.hard, v)) return "hard";
  return outside(row.soft, v) ? "soft" : null;
}

// ---- sliders and steppers (§4.9) --------------------------------------------------------------------------------

export const SLIDER_STEPS = 1000;

const snap3 = (v) => Number(v.toPrecision(3));

// u ∈ [0, 1] → the track's value: lo + u (hi − lo), or lo (hi/lo)^u on a log track, snapped to 3 significant
// digits (integer rows: whole units).
export function sliderValue(u, track, scale, isInt) {
  const v = scale === "log" ? track.lo * (track.hi / track.lo) ** u : track.lo + u * (track.hi - track.lo);
  return Math.min(Math.max(isInt ? Math.round(v) : snap3(v), track.lo), track.hi);
}

export function sliderPosition(v, track, scale) {
  const u = scale === "log" ? Math.log(v / track.lo) / Math.log(track.hi / track.lo)
    : (v - track.lo) / (track.hi - track.lo);
  return Number.isFinite(u) ? Math.min(Math.max(u, 0), 1) : 0;
}

// The value one step from `d` (a decimal in the schema unit) as a decimal: ×2^(dir/4) on log rows (3 significant
// digits; integer rows whole, at least one whole unit away from `d` and at least 1), else ± one unit of the third
// significant digit (integer rows: at least 1). From 0 a step goes to the track's lower end (1 without a track).
export function stepDecimal(d, dir, row) {
  const isInt = row.type === "int";
  const x = toNumber(d);
  if (x === 0) {
    const first = row.track ? row.track.lo : 1;
    return fromNumber(dir > 0 ? first : -first);
  }
  if (row.scale === "log" && x > 0) {
    const y = x * 2 ** (dir / 4);
    if (!isInt) return fromNumber(snap3(y));
    const r = Math.round(snap3(y));                 // 3·2^(−1/4) rounds back to 3: move by one whole unit at least
    return fromNumber(Math.max(1, dir > 0 ? Math.max(r, x + 1) : Math.min(r, x - 1)));
  }
  const three = fromNumber(snap3(x));
  const unit = Math.max(isInt ? 0 : -400, magnitude(three) - 2);
  const e = Math.min(three.exp, unit);
  const a = (three.neg ? -three.digits : three.digits) * 10n ** BigInt(three.exp - e);
  const sum = a + BigInt(dir) * 10n ** BigInt(unit - e);
  return make(sum < 0n, sum < 0n ? -sum : sum, e);
}

// ---- lists ------------------------------------------------------------------------------------------------------

// A vector row's typed text (comma or space separated) → {text: the YAML flow list, value: [..]} or {error}.
export function listFromText(row, typed) {
  const items = typed.split(/[\s,]+/).filter((s) => s !== "");
  const base = row.type.replace("_list", "");
  if (row.length !== null && row.length !== "any" && items.length !== row.length) {
    return { error: `${row.length} values expected, ${items.length} given` };
  }
  if (!items.length && row.length === "any") return { text: "[]", value: [] };
  const out = [], values = [];
  for (const item of items) {
    if (base === "string") {
      out.push(JSON.stringify(item));
      values.push(item);
      continue;
    }
    const r = numberFromText({ type: base, unit: row.unit }, item);
    if (r.error) return { error: `${item}: ${r.error}` };
    out.push(r.text);
    values.push(r.value);
  }
  return { text: `[${out.join(", ")}]`, value: values };
}

export const showList = (row, value) => (Array.isArray(value) ? value.map((v) => (typeof v === "number" ?
  decimalText(fromNumber(v), "show") : String(v))).join(", ") : "");
