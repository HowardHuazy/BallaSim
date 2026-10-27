// The UI shell and the sign-in page (docs/design/phase3/ui.md §3.2, §3.3). No third-party code. The API takes a
// Bearer session secret kept in this origin's localStorage; there are no cookies.

import { SESSION, api, session, setProtocolWarning } from "./api.js";
import { checkBrowser, renderCheck } from "./check.js";

const REOPEN = "Not signed in: open the UI with  python -m ballasim ui --open  (or the sign-in URL that a " +
  "foreground server prints).";
const $ = (id) => document.getElementById(id);

function banner(kind, text) {
  const div = document.createElement("div");
  div.className = kind;
  div.textContent = text;
  $("banners")?.append(div);
}

let warnedProtocol = false;
setProtocolWarning(() => {
  if (warnedProtocol) return;
  warnedProtocol = true;
  banner("error", "BS-UI-WIRE-VERSION: the server speaks another protocol version; reload the page.");
});

// /login#once=<nonce>[&to=<route>] (redirect file) or /login#token=<server token> (terminal URL).
async function login() {
  const params = new URLSearchParams(location.hash.slice(1));
  history.replaceState(null, "", "/login");      // the secret leaves the address bar and the history first
  const once = params.get("once");
  const token = params.get("token");
  const to = /^\/[\w/.:%=&?-]*$/.test(params.get("to") ?? "") ? params.get("to") : "/";
  const status = $("status");
  if (!once && !token) {
    status.textContent = REOPEN;
    return;
  }
  try {
    const { session: secret } = await api("/session", { method: "POST", body: once ? { once } : { token } });
    localStorage.setItem(SESSION, secret);
    location.replace(`/#${to}`);
  } catch (e) {
    status.textContent = `${e.code ?? "error"}: ${e.message}`;
  }
}

let meta = null;
let browser = null;

function table(rows) {
  const t = document.createElement("table");
  for (const [k, v] of rows) {
    const tr = t.insertRow();
    const th = document.createElement("th");
    th.textContent = k;
    tr.append(th);
    tr.insertCell().textContent = v;
  }
  return t;
}

function showServer(view) {
  const h = document.createElement("h2");
  h.textContent = "Server";
  if (!meta) {
    view.replaceChildren(h, Object.assign(document.createElement("p"), { textContent: REOPEN }));
    return;
  }
  const gpu = meta.gpu;
  const mem = gpu.nvml ? `${gpu.nvml.used_mib} / ${gpu.nvml.total_mib} MiB used` : "unknown";
  const built = Object.entries(meta.build).filter(([, v]) => v === true).map(([k]) => k).join(", ");
  const rows = [
    ["Version", `${meta.version} (protocol ${meta.protocol.join(".")})`],
    ["Source",
      meta.git.commit ? `${meta.git.commit.slice(0, 12)}${meta.git.dirty ? " (modified)" : ""}` : "not a checkout"],
    ["Modules", `_core${built ? ", " + built : ""}; compiler ${meta.build.compiler}`],
    ["GPU", `${gpu.available === null ? "probing…" : gpu.available ? "available" : "not available"}; memory ${mem}`],
    ["GPU lock", gpu.lock ?? "none"],
    ["Jobs root", `${meta.jobs_root}${meta.dispatcher ? "" : " (another server dispatches its jobs)"}`],
    ["Bench features", meta.repo ? "available (source checkout)" : "unavailable (no --repo)"],
  ];
  const roots = table(meta.roots.map((r) => [r.id, `${r.path}${r.readonly ? " (read-only)" : ""}`]));
  const h3 = document.createElement("h3");
  h3.textContent = "Roots";
  view.replaceChildren(h, table(rows), h3, roots);
}

let routeSeq = 0;                                 // the latest route; a page still loading for an older one gives up
let leave = null;                                 // tears the shown page down (the viewer: playback, fetches, frames)

// #/view[?path=…]: the event viewer (UI-2), loaded on first use.
async function showViewer(view, query, seq) {
  const { mountViewer, unmountViewer } = await import("../view/views.js");
  if (seq !== routeSeq) return;
  mountViewer(view, { meta, browser, query });
  leave = unmountViewer;
}

// #/compare?ref=…&test=…: real vs sim (UI-8a), loaded on first use; it shares the viewer's device and pool.
async function showCompare(view, query, seq) {
  const { mountCompare, unmountCompare } = await import("../view/compare.js");
  if (seq !== routeSeq) return;
  mountCompare(view, { meta, browser, query });
  leave = unmountCompare;
}

// The runs and jobs screens (UI-7), loaded on first use: #/runs and #/jobs (the list and the roots), #/runs/<rid>
// (a run's results), #/jobs/<id> (a job's progress, plots and log); the setup screen (UI-5): #/setup (the form).
function showScreen(module, name, ids) {
  return async (view, query, seq, match) => {
    const mod = await import(`../screens/${module}.js`);
    if (seq !== routeSeq) return;
    leave = mod[name](view, { query, meta, [ids]: match[1] });
  };
}

const routes = [[/^\/$/, showServer], [/^\/check$/, (view) => renderCheck(view, browser)], [/^\/view$/, showViewer],
  [/^\/compare$/, showCompare], [/^\/setup$/, showScreen("setup", "mountSetup", "id")],
  [/^\/(?:runs|jobs)$/, showScreen("runs", "mountRuns", "id")],
  [/^\/runs\/([A-Za-z0-9_-]{1,4096})$/, showScreen("results", "mountResults", "rid")],
  [/^\/jobs\/([0-9a-f]{16})$/, showScreen("job", "mountJob", "id")]];

function route() {
  const seq = ++routeSeq;
  leave?.();
  leave = null;
  const view = $("view");
  view.className = "";
  delete view.dataset.state;
  delete view.dataset.status;
  const [path, query] = (location.hash.slice(1) || "/").split("?");
  let match = null;
  const show = routes.find(([re]) => (match = re.exec(path)))?.[1];
  if (show) {
    Promise.resolve(show(view, new URLSearchParams(query ?? ""), seq, match)).catch((e) => {
      view.replaceChildren(Object.assign(document.createElement("p"), { className: "error",
        textContent: `${e.code ?? "error"}: ${e.message}` }));
    });
  } else view.replaceChildren(Object.assign(document.createElement("p"), { textContent: "No such page." }));
}

async function shell() {
  browser = await checkBrowser();
  if (!browser.chromeOk) {
    banner("warn", `BS-UI-BROWSER: the UI is built for Chrome or Chromium ${browser.minChrome} or newer.`);
  }
  if (!browser.webgpu.adapter) {
    banner("warn", "BS-UI-WEBGPU: no WebGPU adapter; the viewer will show PNG panels rendered by the server " +
      "(see Browser check).");
  }
  if (!session()) banner("warn", REOPEN);
  else {
    try {
      meta = await api("/meta");
      $("server").textContent = `${meta.version} on port ${location.port}`;
      $("signout").hidden = false;
    } catch (e) {
      if (e.status === 401) {
        localStorage.removeItem(SESSION);
        banner("warn", REOPEN);
      } else banner("error", `${e.code}: ${e.message}`);
    }
  }
  $("signout").addEventListener("click", async () => {
    await api("/session", { method: "DELETE" }).catch(() => {});
    localStorage.removeItem(SESSION);
    location.reload();
  });
  window.addEventListener("hashchange", route);
  route();
  document.body.dataset.ready = "1";
}

if (document.body.dataset.page === "login") login();
else shell();
