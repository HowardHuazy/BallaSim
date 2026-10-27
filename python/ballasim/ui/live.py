"""The WebSocket hub (docs/design/phase3/ui.md §3.4, §4.10): topics, per-socket send queues and the 250 ms tailers.

- **Topics:** ``jobs`` (every job's record when it changes), ``job:<id>`` (that job's record and its ``bridge`` logs),
  ``run:<rid>`` (the run's progress and perf records), ``server`` (GPU and dispatcher state), ``resim:<session>``
  (UI-11); at most 256 topics per socket. A run topic may name a run directory a job has yet to create. One tick
  every 250 ms polls by ``mtime_ns`` (progress.json, status.json) and byte offset (perf.jsonl and
  the logs, complete lines only for perf), so each topic gets at most 4 messages a second. A log message carries at
  most 64 kB; ``offset`` is the byte offset its text or records start at (a client stitches them to what it read over
  HTTP, whose ``X-Next-Offset`` is where those end).
- **Progress** (§4.10): ``fraction`` = (camera_index + window/windows)/cameras; ``eta_s`` = camera_elapsed_s/k·(n − k)
  for the current camera plus the median per-window time of the perf records seen (sample_s, which holds the plan
  and the render, + sense_s + gt_s + write_s) times the windows of the later cameras.
- **Isolation:** one run or job whose files cannot be read (another user's run, a malformed job) is skipped, logged
  once per exception type; the other topics' tailers go on.
- **Flow control:** each socket has a queue of 256 messages written with ``await write_message``; a client that falls
  further behind is closed with code 1013 and re-fetches its state over HTTP.
- **Acknowledgement** (UI-7): a ``sub`` is answered with ``{"op": "subscribed", "topics": [...]}``, the requested
  topics the socket now holds (already held ones included). Changes are published from then on, so a client re-reads
  its state over HTTP after the acknowledgement and misses nothing between its first read and the subscription.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import statistics

import tornado.websocket

from .fs import read_json, read_lines
from .jobs import MAX_LOG_BYTES
from .streams import StreamError

log = logging.getLogger("ballasim.ui")

TICK_S = 0.25
MAX_BEHIND = 256
#: Topics one socket may hold.
MAX_TOPICS = 256
MAX_PERF_TICK_BYTES = 1 << 20
_TOPIC_PREFIXES = ("job:", "run:", "resim:")
#: A window's wall time from its perf record: sampling (plan and render included: engine.h sample_ms, ENGINE.md), sense,
#: GT and write.
_WINDOW_TIMES = ("sample_s", "sense_s", "gt_s", "write_s")


class _RunTail:
    """The tail of one subscribed run: progress.json by mtime_ns, perf.jsonl from a byte offset. A run directory that
    does not exist yet (a job's run before its run step) is looked for at every tick, and its perf.jsonl is then read
    from its start; a rid the roots refuse otherwise raises StreamError."""

    def __init__(self, fs, rid: str):
        self.fs, self.rid, self.path, self.progress_mtime, self.perf_offset = fs, rid, None, None, 0
        self.window_s: list[float] = []
        self._locate(at_end=True)

    def _locate(self, *, at_end: bool = False) -> bool:
        if self.path is not None:
            return True
        try:
            path = self.fs.resolve(self.rid)
        except StreamError as e:
            if e.status != 404:
                raise
            return False
        perf = path / "logs" / "perf.jsonl"
        self.path, self.perf_offset = path, perf.stat().st_size if at_end and perf.exists() else 0
        return True

    def progress(self) -> dict | None:
        if not self._locate():
            return None
        try:
            mtime = (self.path / "progress.json").stat().st_mtime_ns
        except OSError:
            return None
        if mtime == self.progress_mtime:
            return None
        self.progress_mtime = mtime
        return read_json(self.path / "progress.json")

    def perf(self) -> tuple[list, int] | None:
        if not self._locate():
            return None
        with contextlib.suppress(OSError):          # a run that replaced the directory (overwrite): from its start
            if (self.path / "logs" / "perf.jsonl").stat().st_size < self.perf_offset:
                self.perf_offset = 0
        start = self.perf_offset
        data, self.perf_offset = read_lines(self.path / "logs" / "perf.jsonl", start, MAX_PERF_TICK_BYTES)
        if not data:
            return None
        records = []
        for line in data.splitlines():
            with contextlib.suppress(ValueError):
                records.append(json.loads(line))
        for r in records:
            if isinstance(r, dict):
                with contextlib.suppress(TypeError, ValueError):
                    self.window_s.append(sum(float(r.get(k) or 0.0) for k in _WINDOW_TIMES))
        del self.window_s[:-4096]
        return records, start


def progress_message(rid: str, p: dict, window_s: list[float]) -> dict:
    """§3.4's progress message with §4.10's fraction and ETA of a run's progress.json."""
    i, n_cams = int(p.get("camera_index") or 0), max(int(p.get("cameras") or 1), 1)
    k, n = int(p.get("window") or 0), int(p.get("windows") or 0)
    fraction = (i + k / n) / n_cams if n else None
    eta = None
    if n and k:
        elapsed = float(p.get("camera_elapsed_s", p.get("elapsed_s")) or 0.0)
        per_window = statistics.median(window_s) if window_s else elapsed / k
        eta = elapsed / k * (n - k) + per_window * n * (n_cams - i - 1)
    return {"op": "progress", "rid": rid, "progress": p, "camera_index": i, "cameras": n_cams, "fraction": fraction,
            "eta_s": eta}


class Hub:
    """The WebSocket hub of one server. ``server_message()`` is the ``server`` topic's message (sent when it
    changes)."""

    def __init__(self, fs, jobs, server_message, *, tick_s: float = TICK_S):
        self.fs, self.jobs, self.server_message, self.tick_s = fs, jobs, server_message, tick_s
        self.topics: dict[object, set[str]] = {}
        self.queues: dict[object, asyncio.Queue] = {}
        self.senders: dict[object, asyncio.Task] = {}
        self.runs: dict[str, _RunTail] = {}
        self.logs: dict[tuple[str, str], int] = {}
        self._server = None
        self._task: asyncio.Task | None = None
        self._failed: set[tuple[str, str]] = set()

    def _skip(self, item: str, e: Exception) -> None:
        key = (item, type(e).__name__)
        if key not in self._failed:
            self._failed.add(key)
            log.warning("%s: skipped by the tailers: %s: %s", item, type(e).__name__, e)

    # -- sockets -----------------------------------------------------------------------------------------------

    def add(self, ws) -> None:
        self.topics[ws], self.queues[ws] = set(), asyncio.Queue(MAX_BEHIND)
        self.senders[ws] = asyncio.ensure_future(self._sender(ws, self.queues[ws]))

    def remove(self, ws) -> None:
        self.topics.pop(ws, None)
        self.queues.pop(ws, None)
        task = self.senders.pop(ws, None)
        if task is not None:
            task.cancel()
        self._forget()

    async def _sender(self, ws, queue: asyncio.Queue) -> None:
        try:
            while True:
                msg = await queue.get()
                await ws.write_message(json.dumps(msg, ensure_ascii=False))
        except (tornado.websocket.WebSocketClosedError, asyncio.CancelledError):
            pass

    def send(self, ws, msg: dict) -> None:
        queue = self.queues.get(ws)
        if queue is None:
            return
        try:
            queue.put_nowait(msg)
        except asyncio.QueueFull:
            log.warning("a WebSocket client fell %d messages behind: closed", MAX_BEHIND)
            self.remove(ws)
            ws.close(1013, "too far behind: reconnect and re-fetch over HTTP")

    def publish(self, topic: str, msg: dict) -> None:
        for ws, topics in list(self.topics.items()):
            if topic in topics:
                self.send(ws, msg)

    def on_message(self, ws, msg) -> None:
        """``sub``, ``unsub`` and ``ping`` (answered with the ``server`` message)."""
        if not isinstance(msg, dict) or ws not in self.topics:
            return
        op, topics = msg.get("op"), msg.get("topics")
        if op == "ping":
            self.send(ws, self.server_message())
        elif op in ("sub", "unsub") and isinstance(topics, list):
            topics = [t for t in topics if isinstance(t, str) and (t in ("jobs", "server")
                                                                   or t.startswith(_TOPIC_PREFIXES))]
            if op == "sub":
                held = self.topics[ws]
                new = list(dict.fromkeys(t for t in topics if t not in held))[:MAX_TOPICS - len(held)]
                for t in new:
                    self._watch(t)
                held.update(new)
                self.send(ws, {"op": "subscribed", "topics": [t for t in dict.fromkeys(topics) if t in held]})
            else:
                self.topics[ws].difference_update(topics)
                self._forget()

    def _watch(self, topic: str) -> None:
        if topic.startswith("run:") and topic not in self.runs:
            with contextlib.suppress(Exception):
                self.runs[topic] = _RunTail(self.fs, topic[4:])
        elif topic.startswith("job:"):
            with contextlib.suppress(Exception):
                job = self.jobs.get(topic[4:])
                for stream in ("stdout", "stderr"):
                    path = job.dir / "bridge" / f"job_{stream}.log"
                    self.logs.setdefault((job.job_id, stream), path.stat().st_size if path.exists() else 0)

    def _forget(self) -> None:
        live = set().union(*self.topics.values()) if self.topics else set()
        for t in [t for t in self.runs if t not in live]:
            del self.runs[t]
        for key in [k for k in self.logs if f"job:{k[0]}" not in live]:
            del self.logs[key]

    # -- the tick ----------------------------------------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for task in self.senders.values():
            task.cancel()

    async def _loop(self) -> None:
        while True:
            self._safe_tick()
            await asyncio.sleep(self.tick_s)

    def _safe_tick(self) -> None:
        try:
            self.tick()
        except Exception:  # the tailers never stop the server
            log.exception("job and run tailers")

    def tick_soon(self) -> None:
        """A tick now (after a job's creation, cancel or resume), besides the 250 ms ones."""
        asyncio.get_running_loop().call_soon(self._safe_tick)

    def tick(self) -> None:
        try:
            recs = self.jobs.tick()
        except Exception as e:  # noqa: BLE001  the run tailers go on
            self._skip("jobs", e)
            recs = []
        for rec in recs:
            msg = {"op": "job", "job": rec}
            self.publish("jobs", msg)
            self.publish(f"job:{rec['job_id']}", msg)
        server = self.server_message()
        if server != self._server:
            self._server = server
            self.publish("server", server)
        for topic, tail in list(self.runs.items()):
            rid = topic[4:]
            try:
                p = tail.perf()
                if p is not None:
                    self.publish(topic, {"op": "perf", "rid": rid, "records": p[0], "offset": str(p[1])})
                progress = tail.progress()
                if progress is not None:
                    self.publish(topic, progress_message(rid, progress, tail.window_s))
            except Exception as e:  # noqa: BLE001  one unreadable run never stops the other tailers
                self._skip(topic, e)
        for (job_id, stream), offset in list(self.logs.items()):
            job = self.jobs.jobs.get(job_id)
            if job is None:
                continue
            try:
                data, _ = self.jobs.log(job_id, stream, offset)
            except Exception as e:  # noqa: BLE001
                self._skip(f"job:{job_id} {stream}", e)
                continue
            if not data:
                continue
            cut = data.rfind(b"\n") + 1 if len(data) == MAX_LOG_BYTES and b"\n" in data else len(data)
            self.logs[(job_id, stream)] = offset + cut
            self.publish(f"job:{job_id}", {"op": "log", "job": job_id, "stream": stream,
                                           "text": data[:cut].decode(errors="replace"), "offset": str(offset)})
