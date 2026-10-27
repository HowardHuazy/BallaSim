// The API client (docs/design/phase3/ui.md §3.2, §3.3): a Bearer session secret kept in this origin's localStorage
// (no cookies), JSON in and out, binary bodies for the wire formats (§3.5). No third-party code.

export const SESSION = "bs_session";
const PROTOCOL_MAJOR = "1";
let onProtocolMismatch = () => {};

export class ApiError extends Error {
  constructor(status, code, message, body = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.body = body;                             // the answer's JSON (a 422 of /resolve holds diagnostics and missing)
  }
}

export function session() {
  try { return localStorage.getItem(SESSION); } catch { return null; }
}

// fn(): called once per response whose X-BallaSim-UI major version is not ours (BS-UI-WIRE-VERSION).
export function setProtocolWarning(fn) { onProtocolMismatch = fn; }

async function request(path, { method = "GET", body, cache = "no-store" } = {}) {
  const headers = {};
  const s = session();
  if (s) headers.Authorization = `Bearer ${s}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const r = await fetch(`/api/v1${path}`, {
    method, headers, cache, body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (r.headers.get("X-BallaSim-UI") !== PROTOCOL_MAJOR) onProtocolMismatch();
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new ApiError(r.status, data?.error?.code ?? `HTTP ${r.status}`, data?.error?.message ?? r.statusText, data);
  }
  return r;
}

// fetch /api/v1<path> with the session; JSON in and out; errors as ApiError {status, code, message}.
export async function api(path, options = {}) {
  const r = await request(path, options);
  return r.status === 204 ? null : r.json().catch(() => ({}));
}

// A binary body (BSEV, BSRT, …) as an ArrayBuffer. Slices of closed files are immutable (§3.3), so the browser's HTTP
// cache may answer them.
export async function apiBytes(path, { cache = "default" } = {}) {
  return (await request(path, { cache })).arrayBuffer();
}

// The Response of /api/v1<path> (errors as ApiError): for bodies read with their headers, such as PNG panels.
export async function apiResponse(path, options = {}) {
  return request(path, options);
}
