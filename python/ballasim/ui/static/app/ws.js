// The WebSocket client (docs/design/phase3/ui.md §3.4): one socket per tab, opened with a single-use ticket from
// POST /ws-ticket (a browser cannot set headers on a WebSocket). Screens subscribe to topics ("jobs", "job:<id>",
// "run:<rid>", "server") with a handler; each message goes once to every subscription of a topic it belongs to. The
// hub acknowledges a `sub` ("subscribed"); from then on it publishes every change, so each subscription's `onSync`
// re-reads its state over HTTP then: nothing is missed between a screen's first read and its subscription. After a
// close (the hub closes a client 256 messages behind with 1013, or the server restarted) the socket reconnects with a
// new ticket (1 s, doubling to 16 s) and subscribes again, and the acknowledgement syncs every subscription. No
// third-party code.

import { api } from "./api.js";

const subs = new Map();                           // topic -> Set of {handler, onSync}
let ws = null, opening = false, retry = 1000, timer = null;

// The topics a hub message belongs to (§3.4).
export function topicsOf(msg) {
  switch (msg?.op) {
    case "job": return ["jobs", `job:${msg.job?.job_id}`];
    case "log": return [`job:${msg.job}`];
    case "progress":
    case "perf": return [`run:${msg.rid}`];
    case "server": return ["server"];
    case "resim": return [`resim:${msg.session}`];
    default: return [];
  }
}

function send(obj) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function dispatch(msg) {
  const ack = msg?.op === "subscribed", seen = new Set();
  for (const topic of ack ? msg.topics ?? [] : topicsOf(msg)) {
    for (const s of subs.get(topic) ?? []) {
      if (seen.has(s)) continue;
      seen.add(s);
      try {
        if (ack) s.onSync?.();
        else s.handler(msg);
      } catch (e) {
        console.error("ws handler", e);
      }
    }
  }
}

function schedule() {
  if (timer || !subs.size) return;
  timer = setTimeout(() => {
    timer = null;
    connect();
  }, retry);
  retry = Math.min(retry * 2, 16000);
}

async function connect() {
  if (opening || (ws && ws.readyState <= WebSocket.OPEN) || !subs.size) return;
  opening = true;
  try {
    const { ticket } = await api("/ws-ticket", { method: "POST" });
    const sock = new WebSocket(`ws://${location.host}/api/v1/ws?ticket=${encodeURIComponent(ticket)}`);
    ws = sock;
    sock.onopen = () => {
      retry = 1000;
      if (subs.size) send({ op: "sub", topics: [...subs.keys()] });
    };
    sock.onmessage = (e) => {
      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch {
        return;
      }
      dispatch(msg);
    };
    sock.onclose = () => {
      if (ws === sock) ws = null;
      schedule();
    };
  } catch {
    schedule();
  } finally {
    opening = false;
  }
}

// Subscribes `handler(msg)` to `topics`; returns the unsubscribe function. `onSync()`: called each time the hub
// acknowledges the topics (after this call, and after every reconnect), to re-read state over HTTP.
export function subscribe(topics, handler, { onSync = null } = {}) {
  const sub = { handler, onSync };
  for (const t of topics) {
    if (!subs.has(t)) subs.set(t, new Set());
    subs.get(t).add(sub);
  }
  send({ op: "sub", topics });                    // acknowledged also for topics the socket already holds
  connect();
  return () => {
    const gone = [];
    for (const t of topics) {
      const set = subs.get(t);
      if (!set?.delete(sub) || set.size) continue;
      subs.delete(t);
      gone.push(t);
    }
    if (gone.length) send({ op: "unsub", topics: gone });
  };
}

// Whether the socket is open (tests, the status line).
export const connected = () => ws?.readyState === WebSocket.OPEN;
