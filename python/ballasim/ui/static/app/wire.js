// Decoders of the binary wire formats (docs/design/phase3/ui.md §3.5; little-endian, version 1): BSEV block slices
// and worker frames, BSRT rate overviews, BSFR dense frames, BSMK region masks and BSTK timeline tracks. Every column
// body starts 8-aligned, so columns are typed-array views of the message, never copies. int64 values are BigInt.

export const WIRE_VERSION = 1;
const magic = (s) => [...s].reduceRight((v, c) => (v << 8) | c.charCodeAt(0), 0) >>> 0;
export const MAGIC = { BSEV: magic("BSEV"), BSRT: magic("BSRT"), BSFR: magic("BSFR"), BSMK: magic("BSMK"),
  BSTK: magic("BSTK") };
export const BSEV_FLAGS = { DEFLATE: 1, LOD: 2, BLOCK_END: 4, WORKER: 8 };
// BSEV column ids (§3.5); unknown ids are skipped.
export const COLUMN_NAMES = { 0: "ev", 16: "instance", 17: "semantic", 18: "depth", 19: "flow", 20: "t_cross",
  21: "flags", 22: "ds", 23: "dn", 24: "dl", 25: "l_first" };
// BSTK extra-column ids (§3.5).
export const TRACK_COLUMN_NAMES = { 1: "index", 2: "window" };
const DTYPES = { 1: Uint8Array, 2: Uint16Array, 3: Uint32Array, 4: Int32Array,
  5: globalThis.Float16Array ?? Uint16Array, 6: Float32Array, 7: BigInt64Array };

export class WireError extends Error {
  constructor(message, code = "BS-UI-WIRE") {
    super(message);
    this.code = code;
  }
}

function header(buf, name, minBytes) {
  if (buf.byteLength < minBytes) throw new WireError(`${name}: ${buf.byteLength} bytes, need ${minBytes}`);
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC[name]) throw new WireError(`not a ${name} message`);
  const version = dv.getUint16(4, true);
  if (version > WIRE_VERSION) {
    throw new WireError(`${name} version ${version}; this page reads ${WIRE_VERSION}: reload`, "BS-UI-WIRE-VERSION");
  }
  return dv;
}

function view(buf, Type, offset, count, what) {
  if (offset % 8 || offset + count * Type.BYTES_PER_ELEMENT > buf.byteLength) {
    throw new WireError(`${what}: ${count} values at offset ${offset} do not fit the message`);
  }
  return new Type(buf, offset, count);
}

// The 16-byte column table at `at` (BSEV, BSTK): {id, comps, values}, values = n·comps elements.
function columnTable(buf, dv, at, nColumns, n, names) {
  const out = {};
  for (let c = 0; c < nColumns; c++) {
    const e = at + 16 * c;
    const id = dv.getUint16(e, true), code = dv.getUint8(e + 2), comps = dv.getUint8(e + 3);
    const offset = dv.getUint32(e + 4, true), bytes = dv.getUint32(e + 8, true);
    const Type = DTYPES[code];
    if (!Type) throw new WireError(`column ${id}: unknown dtype ${code}`);
    if (bytes !== n * comps * Type.BYTES_PER_ELEMENT) throw new WireError(`column ${id}: ${bytes} bytes for ${n} rows`);
    const name = names[id];
    if (name) out[name] = { id, comps, values: view(buf, Type, offset, n * comps, `column ${name}`) };
  }
  return out;
}

async function inflateRaw(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Response(stream).arrayBuffer();
}

// One BSEV message: {flags, lod, blockEnd, worker, n, tBase, i0, nBlock, width, height, coverage, columns}, where
// columns.ev.values is the Uint32Array of interleaved (word 0, word 1) pairs.
export async function decodeBSEV(buf) {
  let dv = header(buf, "BSEV", 64);
  const flags = dv.getUint16(6, true);
  if (flags & BSEV_FLAGS.DEFLATE) {               // the header stays plain; offsets refer to the inflated message
    const rest = await inflateRaw(new Uint8Array(buf, 64));
    const whole = new Uint8Array(64 + rest.byteLength);
    whole.set(new Uint8Array(buf, 0, 64));
    whole.set(new Uint8Array(rest), 64);
    buf = whole.buffer;
    dv = new DataView(buf);
  }
  const n = dv.getUint32(12, true), nColumns = dv.getUint16(36, true);
  if (dv.getUint16(38, true) !== 64 + 16 * nColumns) throw new WireError("BSEV: header_bytes != 64 + 16 n_columns");
  return {
    flags, lod: Boolean(flags & BSEV_FLAGS.LOD), blockEnd: Boolean(flags & BSEV_FLAGS.BLOCK_END),
    worker: Boolean(flags & BSEV_FLAGS.WORKER), n, tBase: dv.getBigInt64(16, true), i0: dv.getUint32(24, true),
    nBlock: dv.getUint32(28, true), width: dv.getUint16(32, true), height: dv.getUint16(34, true),
    coverage: [dv.getUint32(40, true), dv.getUint32(44, true)],
    columns: columnTable(buf, dv, 64, nColumns, n, COLUMN_NAMES),
  };
}

// Word 0 of an `ev` pair: x | y<<14 | tag<<28 | p<<31 (tag: 0 without GT, else 1 + the S1 label).
export function unpackWord(w) {
  return { x: w & 0x3fff, y: (w >>> 14) & 0x3fff, tag: (w >>> 28) & 7, p: w >>> 31 };
}

// BSRT: {flags, allOnValid, t0, binNs, total, on}; on[i] = 0xFFFFFFFF where the ON count is not known yet.
export function decodeBSRT(buf) {
  const dv = header(buf, "BSRT", 32);
  const flags = dv.getUint16(6, true), n = dv.getUint32(12, true);
  if (buf.byteLength !== 32 + 8 * n) throw new WireError(`BSRT: ${buf.byteLength} bytes for ${n} bins`);
  return { flags, allOnValid: Boolean(flags & 1), t0: dv.getBigInt64(16, true), binNs: dv.getUint32(24, true),
    total: new Uint32Array(buf, 32, n), on: new Uint32Array(buf, 32 + 4 * n, n) };
}

// BSFR: one dense GT frame, H·W·comps values row-major after the 48-byte header.
export function decodeBSFR(buf) {
  const dv = header(buf, "BSFR", 48);
  const code = dv.getUint8(8), comps = dv.getUint8(9), width = dv.getUint16(12, true), height = dv.getUint16(14, true);
  const Type = DTYPES[code];
  if (!Type) throw new WireError(`BSFR: unknown dtype ${code}`);
  const count = width * height * comps;
  if (buf.byteLength !== 48 + count * Type.BYTES_PER_ELEMENT) throw new WireError("BSFR: body size");
  return { channel: dv.getUint16(6, true), comps, group: dv.getUint16(10, true), width, height,
    tNs: dv.getBigInt64(24, true), tTrueNs: dv.getBigInt64(32, true), values: view(buf, Type, 48, count, "BSFR") };
}

// BSMK: masks [m0, m0 + n) with their [t_lo, t_hi) intervals (interleaved) and ⌈W·H/32⌉ u32 words each.
export function decodeBSMK(buf) {
  const dv = header(buf, "BSMK", 32);
  const n = dv.getUint32(8, true), width = dv.getUint16(12, true), height = dv.getUint16(14, true);
  const wordsPerMask = Math.ceil((width * height) / 32);
  if (buf.byteLength !== 32 + 16 * n + 4 * n * wordsPerMask) throw new WireError("BSMK: body size");
  return { flags: dv.getUint16(6, true), n, width, height, m0: dv.getBigInt64(16, true), wordsPerMask,
    intervals: view(buf, BigInt64Array, 32, 2 * n, "BSMK intervals"),
    words: view(buf, Uint32Array, 32 + 16 * n, n * wordsPerMask, "BSMK words") };
}

// BSTK: {kind, n, t0, tNs (sensor clock), columns}. The column table (16 B per extra column, as BSEV's) follows the
// 32-byte header; t_ns i64[n] follows the table; extra column bodies lie at their table offsets.
export function decodeBSTK(buf) {
  const dv = header(buf, "BSTK", 32);
  const n = dv.getUint32(8, true), nExtra = dv.getUint32(12, true);
  return { kind: dv.getUint16(6, true), n, t0: dv.getBigInt64(16, true),
    tNs: view(buf, BigInt64Array, 32 + 16 * nExtra, n, "BSTK t_ns"),
    columns: columnTable(buf, dv, 32, nExtra, n, TRACK_COLUMN_NAMES) };
}
