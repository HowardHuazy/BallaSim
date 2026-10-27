// Browser and WebGPU check (docs/design/phase3/ui.md §6 BS-UI-BROWSER and BS-UI-WEBGPU, §8.6, Q2). Chrome or
// Chromium 147 or newer is expected; other browsers get a banner, not a block. The adapter is only requested, never
// turned into a device (requestDevice would consume it).

export const MIN_CHROME = 147;
const LIMITS = ["maxBufferSize", "maxStorageBufferBindingSize", "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup", "maxStorageBuffersPerShaderStage", "maxComputeWorkgroupsPerDimension"];
const FEATURES = ["timestamp-query", "shader-f16", "subgroups"];

export function chromeVersion() {
  const brands = navigator.userAgentData?.brands ?? [];
  const brand = brands.find((b) => b.brand === "Google Chrome" || b.brand === "Chromium");
  if (brand) return Number(brand.version);
  const m = /\b(?:Chrome|Chromium)\/(\d+)/.exec(navigator.userAgent);
  return m ? Number(m[1]) : null;
}

export async function checkBrowser(minChrome = MIN_CHROME) {
  const chrome = chromeVersion();
  const out = {
    userAgent: navigator.userAgent, chrome, minChrome, chromeOk: chrome !== null && chrome >= minChrome,
    secureContext: window.isSecureContext, crossOriginIsolated: window.crossOriginIsolated,
    webgpu: { api: "gpu" in navigator, adapter: false },
  };
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (a) {
        const info = a.info ?? {};
        Object.assign(out.webgpu, {
          adapter: true, vendor: info.vendor ?? "", architecture: info.architecture ?? "",
          description: info.description ?? "",
          isFallbackAdapter: Boolean(info.isFallbackAdapter ?? a.isFallbackAdapter),
          preferredCanvasFormat: navigator.gpu.getPreferredCanvasFormat(),
          limits: Object.fromEntries(LIMITS.map((k) => [k, a.limits[k]])),
          features: FEATURES.filter((f) => a.features.has(f)),
        });
      }
    } catch (e) {
      out.webgpu.error = String(e);
    }
  }
  return out;
}

const yesNo = (v) => (v ? "yes" : "no");

export function renderCheck(view, r) {
  const g = r.webgpu;
  const rows = [
    ["Browser", r.chrome === null ? "not Chrome or Chromium" : `Chrome/Chromium ${r.chrome}`,
      r.chromeOk ? "ok" : `BS-UI-BROWSER: ${r.minChrome} or newer expected`],
    ["Secure context", yesNo(r.secureContext), ""],
    ["Cross-origin isolated", yesNo(r.crossOriginIsolated),
      r.crossOriginIsolated ? "" : "SharedArrayBuffer unavailable"],
    ["WebGPU adapter",
      g.adapter ? `${g.vendor || "?"} / ${g.architecture || "?"}${g.isFallbackAdapter ? " (fallback)" : ""}` : "none",
      g.adapter ? "" : `BS-UI-WEBGPU: the viewer falls back to PNG panels${g.error ? ` (${g.error})` : ""}`],
    ["NVIDIA adapter", yesNo(g.adapter && /nvidia/i.test(g.vendor)), "the manual check (§8.6): yes on the lab desktop"],
  ];
  if (g.adapter) {
    rows.push(["Features", g.features.join(", ") || "none", ""], ["Canvas format", g.preferredCanvasFormat, ""]);
    for (const [k, v] of Object.entries(g.limits)) rows.push([k, String(v), ""]);
  }
  const table = document.createElement("table");
  for (const cells of rows) {
    const tr = table.insertRow();
    cells.forEach((text, i) => {
      const cell = document.createElement(i === 0 ? "th" : "td");
      cell.textContent = text;
      tr.append(cell);
    });
  }
  const pre = document.createElement("pre");
  pre.id = "check-json";
  pre.textContent = JSON.stringify(r, null, 1);
  const h = document.createElement("h2");
  h.textContent = "Browser check";
  view.replaceChildren(h, table, pre);
}
