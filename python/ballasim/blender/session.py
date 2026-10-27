"""The Blender motion session, Python side (docs/design/phase3/blender.md §3.5, §3.6, §3.9, §5.1; BL-9).

``ballasim blender session --exchange DIR --config RUN.yaml [--set K=V ...]`` runs on the pipe pair of the add-on that
started it (stdin: add-on to engine, stdout: engine to add-on, ``-I -u`` so every frame reaches the pipe):

* **framing** (:class:`Wire`): one UTF-8 JSON header line of at most 64 KiB (counts and small scalars), then exactly
  ``bytes`` payload bytes, little-endian, C order; the engine flushes after every frame. ``hello`` both ways first
  (protocol ``ballasim.motion-oracle`` 1.x), then the engine leads: ``bodies``, ``set``, ``sample`` / ``poses``,
  ``mesh_probe`` / ``meshes``, ``progress`` (no reply), and last ``done`` (exit 0, or 2 when the scene is refused) or
  ``error`` (exit 1). ``done`` carries the report and the summary as JSON in its payload (a header is bounded).
* **the session** is C++ (:func:`ballasim._scene.motion_session`): this module only moves bytes between the pipe and
  its callables, then writes what the session decided (:func:`write_outputs`): ``motion/{session.json,
  trajectories.npz, oracle_log.npz, session_perf.json}``, ``conversion_report.json`` and the C1 bundle in
  ``cache/<key>/`` (SCENE.md "Blender exchanges").
* **replay** (``blender replay``): the same session over ``motion/oracle_log.npz`` instead of the pipe; the outputs and
  the cache key come out bitwise as the session's.

Nothing here prints to stdout once the wire is open: it is the add-on's pipe (the process's ``sys.stdout`` becomes
stderr, which the add-on keeps in ``bridge/session_stderr.log``).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .calibration import calibrations
from .convert import (
    OPTION_KEYS,
    assemble_moving,
    blender_cache_key,
    motion_key,
    motion_key_inputs,
    statics,
    trajectory_arrays,
)
from .exchange import Exchange, ExchangeError, read_exchange, texture_paths

PROTOCOL = "ballasim.motion-oracle"
VERSION = (1, 0)
MAX_HEADER = 64 * 1024
HEARTBEAT_S = 2.0       # the engine sends a progress at least this often while it computes (blender.md §3.5)
#: scene.blender.* keys a session reads from the YAML (the keys its cache key sees are the resolver's, _values_for_key).
SESSION_KEYS = ("motion_tol_px", "fps_base_snap", "cache_samples_per_frame")
_CODE = re.compile(r"^([A-Z]{2,4}-[A-Z0-9-]+): ")


class SessionError(Exception):
    """A protocol or session failure with its blender.md §6.1 code (BRG-PROTO, BRG-EOF, ...)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.detail = code, message


class AddonAbort(SessionError):
    """The add-on sent ``error`` or closed the pipe: the session ends without a reply."""


# --- framing ----------------------------------------------------------------------------------------------------------


class Wire:
    """Length-framed messages over a pipe pair (blender.md §3.5)."""

    def __init__(self, reader: BinaryIO, writer: BinaryIO):
        self._r, self._w = reader, writer
        self._lock = threading.Lock()      # the heartbeat thread also sends

    def send(self, op: str, payload: bytes = b"", **fields) -> None:
        head = json.dumps({"op": op, **fields, "bytes": len(payload)}, separators=(",", ":"), allow_nan=False)
        line = head.encode() + b"\n"
        if len(line) > MAX_HEADER:
            raise SessionError("BRG-PROTO", f"a {op} header of {len(line)} bytes exceeds {MAX_HEADER}")
        try:
            with self._lock:
                # under `python -u` stdout.buffer is a raw FileIO, whose write may take fewer bytes than offered
                view = memoryview(line + payload)
                while len(view):
                    n = self._w.write(view)
                    if n is None or n <= 0:
                        raise BrokenPipeError("the pipe made no write progress")
                    view = view[n:]
                self._w.flush()
        except (BrokenPipeError, ValueError) as e:
            raise AddonAbort("BRG-EOF", f"the add-on closed the pipe ({e})") from None

    def _exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._r.read(n - len(buf))
            if not chunk:
                raise AddonAbort("BRG-EOF", "the add-on closed the pipe")
            buf += chunk
        return bytes(buf)

    def recv(self, expect: str | None = None, errors: bool = True) -> tuple[dict, bytes]:
        """The next message; ``expect``: its op. An ``error`` from the add-on ends the session (AddonAbort) unless
        ``errors`` is off (the add-on's own reader of the engine's)."""
        line = self._r.readline(MAX_HEADER + 1)
        if not line:
            raise AddonAbort("BRG-EOF", "the add-on closed the pipe")
        if len(line) > MAX_HEADER or not line.endswith(b"\n"):
            raise SessionError("BRG-PROTO", "a header is longer than 64 KiB or unterminated")
        try:
            head = json.loads(line)
            if not isinstance(head, dict) or not isinstance(head.get("op"), str):
                raise ValueError("expected a message object")
            n = head.get("bytes", 0)
            op = head["op"]
        except (ValueError, KeyError, TypeError):
            raise SessionError("BRG-PROTO", "a header is not a JSON object with an op") from None
        if not (type(n) is int and 0 <= n <= 1 << 33):
            raise SessionError("BRG-PROTO", f"bytes {n!r} is not a payload size")
        payload = self._exact(n) if n else b""
        if op == "error" and errors:
            raise AddonAbort("BRG-ADDON", str(head.get("message", "the add-on stopped the session")))
        if expect is not None and op != expect:
            raise SessionError("BRG-PROTO", f"expected {expect}, got {op}")
        return head, payload


def _nbytes(*arrays: np.ndarray) -> int:
    return sum(a.nbytes for a in arrays)


class PipeOracle:
    """The session's view of the add-on: the callables of ``_scene.motion_session`` as messages."""

    def __init__(self, wire: Wire, progress_interval: float = 0.25):
        self.wire, self.rounds, self.meshes_asked = wire, 0, 0
        self._last, self._interval, self._stage, self._fraction = 0.0, progress_interval, None, 0.0

    def bodies(self, objects: np.ndarray) -> None:
        payload = np.asarray(objects, "<i4").tobytes()
        self.wire.send("bodies", payload, n=len(objects))
        self.wire.recv("ok")

    def simplify(self, on: bool) -> None:
        self.wire.send("set", simplify=bool(on))
        self.wire.recv("ok")

    def sample(self, F: np.ndarray, s: np.ndarray):
        n = len(F)
        payload = np.asarray(F, "<i4").tobytes() + np.asarray(s, "<f4").tobytes()
        self.wire.send("sample", payload, round=self.rounds, n=n)
        self.rounds += 1
        head, data = self.wire.recv("poses")
        B = head.get("bodies")
        if head.get("n") != n or not isinstance(B, int) or B < 0:
            raise SessionError("BRG-PROTO", f"a poses answer for {n} samples says n={head.get('n')}, bodies={B!r}")
        sizes = (8 * n, 4 * n * B * 12, n * B, 4 * n)
        if len(data) != sum(sizes):
            raise SessionError("BRG-PROTO", f"a poses payload of {len(data)} bytes, expected {sum(sizes)}")
        o = np.cumsum((0,) + sizes)
        c_final = np.frombuffer(data, "<f8", n, o[0])
        m = np.frombuffer(data, "<f4", n * B * 12, o[1]).reshape(n, B, 12)
        present = np.frombuffer(data, "u1", n * B, o[2]).reshape(n, B)
        unknown = np.frombuffer(data, "<u4", n, o[3])
        keys = head.get("unknown_keys")
        return c_final, m, present, unknown, json.dumps(keys) if keys else ""

    def mesh_probe(self, objects: np.ndarray, F: np.ndarray, s: np.ndarray):
        payload = np.asarray(objects, "<i4").tobytes() + np.asarray(F, "<i4").tobytes() + np.asarray(s, "<f4").tobytes()
        self.wire.send("mesh_probe", payload, round=self.meshes_asked, k=len(objects), n=len(F))
        self.meshes_asked += 1
        head, data = self.wire.recv("meshes")
        rows = head.get("rows")
        if not isinstance(rows, int) or rows < 0 or len(data) < 32 * rows or (len(data) - 32 * rows) % 4:
            raise SessionError("BRG-PROTO", f"a meshes payload of {len(data)} bytes for {rows!r} rows")
        table = np.frombuffer(data, "<i8", 4 * rows).reshape(rows, 4)
        co = np.frombuffer(data, "<f4", offset=32 * rows)
        return table, co

    def progress(self, fraction: float, stage: str) -> None:
        now = time.monotonic()
        self._fraction = float(fraction)
        if stage == self._stage and now - self._last < self._interval and fraction < 1.0:
            return
        self._stage, self._last = stage, now
        self.wire.send("progress", fraction=round(float(fraction), 4), stage=stage)

    def start_heartbeat(self, every: float = HEARTBEAT_S) -> None:
        """A thread repeating the last progress every ``every`` s: the engine sends one at least that often while it
        computes (blender.md §3.5), also through the long steps with no oracle call (the conversion, writing the
        bundle). The add-on's silence timeout only runs after its own reply, and 60 s is far more."""
        self._stop = threading.Event()

        def beat():
            while not self._stop.wait(every):
                if time.monotonic() - self._last >= every:
                    try:
                        self.wire.send("progress", fraction=round(self._fraction, 4), stage=self._stage or "working")
                    except (SessionError, OSError):
                        return

        self._beat = threading.Thread(target=beat, name="ballasim-session-heartbeat", daemon=True)
        self._beat.start()

    def stop_heartbeat(self) -> None:
        if getattr(self, "_stop", None) is not None:
            self._stop.set()
            self._beat.join(timeout=2.0)


# --- inputs -----------------------------------------------------------------------------------------------------------


class InputError(ValueError):
    """A configuration this session cannot use (exit 2)."""


@dataclass
class Inputs:
    """What ``--config`` and ``--set`` give the session: the scene.blender.* values (the keys that move the cache key),
    the conversion options, the session's own options and the cameras the run selects (``run.select`` for events,
    ``run.frames_select`` for frames only: the session carries both, PHASE3.md D-P3-12)."""

    values: dict = field(default_factory=dict)
    convert: dict = field(default_factory=dict)
    session: dict = field(default_factory=dict)
    select: list | None = None
    frames_select: list | None = None

    def record(self) -> dict:
        """What session.json keeps (and the cache key hashes): everything but the pool size, which changes no result."""
        return {"values": self.values, "convert": self.convert,
                "session": {k: v for k, v in self.session.items() if k != "workers"}, "select": self.select,
                "frames_select": self.frames_select}


def default_workers() -> int:
    """The pool of the pipelines' steps (``execution.scene_workers`` sets it); the result is bitwise independent of its
    size (T7), so session.json does not record it."""
    return max(1, min(4, (os.cpu_count() or 1) // 2))


def inputs_of(config: str | os.PathLike | None = None, sets: Iterable[str] = (), workers: int | None = None) -> Inputs:
    """The run configuration's ``scene.blender`` values, ``run.select`` and ``run.frames_select`` (read directly: the
    resolver takes these rows with BL-3b and FR-1b) with the ``--set`` items over them. ``scene.blender.<key>`` keys:
    the conversion options (:data:`OPTION_KEYS`), ``motion_tol_px`` and ``fps_base_snap``. InputError for anything
    else."""
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    from ballasim.config import parse_sets

    flat: dict[str, Any] = {}
    select = frames_select = None
    if config is not None:
        try:
            raw = YAML(typ="safe").load(Path(config).read_text())
        except YAMLError as e:
            raise InputError(f"{config}: {e}") from None
        if not isinstance(raw, dict):
            raise InputError(f"{config}: a run configuration is a YAML mapping")
        scene = raw.get("scene") or {}
        sub = scene.get("blender") or {}
        if not isinstance(sub, dict):
            raise InputError(f"{config}: scene.blender must be a mapping")
        flat.update({f"scene.blender.{k}": v for k, v in sub.items()})
        flat.update({f"scene.{k}": v for k, v in scene.items() if k.startswith("blender.")})
        run = raw.get("run") or {}
        select, frames_select = run.get("select"), run.get("frames_select")
        workers = workers if workers is not None else (raw.get("execution") or {}).get("scene_workers")
    try:
        cli = parse_sets(list(sets))
    except ValueError as e:
        raise InputError(str(e)) from None
    flat.update({k: v for k, v in cli.items() if k.startswith("scene.blender.")})
    if "run.select" in cli:
        select = cli["run.select"]
    if "run.frames_select" in cli:
        frames_select = cli["run.frames_select"]
    if "execution.scene_workers" in cli:
        workers = cli["execution.scene_workers"]
    out = Inputs(select=list(select) if select else None, frames_select=list(frames_select) if frames_select else None)
    for key, value in flat.items():
        name = key.removeprefix("scene.blender.")
        if name in OPTION_KEYS:
            out.convert[name] = value
        elif name in SESSION_KEYS:
            if name != "cache_samples_per_frame":   # BL-10's
                out.session[name] = value
        else:
            raise InputError(f"{key}: no such scene.blender key (known: "
                             f"{', '.join(sorted(OPTION_KEYS + SESSION_KEYS))})")
        out.values[f"scene.blender.{name}"] = value
    out.session["workers"] = int(workers) if workers else default_workers()
    carried = list(dict.fromkeys([*(out.select or []), *(out.frames_select or [])]))
    if carried:
        out.session["cameras"] = carried
    return out


def _values_for_key(inputs: Inputs, exchange_dir: Path) -> dict:
    """The mapping :func:`blender_cache_key` and :func:`load_blender` see for this exchange: the scene values a run on it
    resolves (``config.load`` with the session's ``scene.blender.*`` items over the rows' defaults, BL-3b), so the
    session's cache key and its motion key inputs equal the run's."""
    from ballasim import config

    doc = {"format": "ballasim.run/2", "output": {"dir": "motion"}}   # output.dir: required, moves no scene value
    spec = config.load(doc, sets=dict(inputs.values), scene_path=exchange_dir)
    return dict(spec.scene.values()["run"])


# --- outputs ----------------------------------------------------------------------------------------------------------


def canonical(doc) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def write_atomic(path: Path, data: str | bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(data.encode() if isinstance(data, str) else data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def log_arrays(log: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """``oracle_log.npz``'s arrays (blender.md §3.6): the body columns, every poses call as ``call.<k>.{F, s, c_final,
    m, present, unknown, unknown_keys}`` and every mesh probe as ``mesh.<k>.{objects, F, s, table, co}``."""
    out = {"bodies": np.asarray(log["bodies"], np.int32), "n_calls": np.array(len(log["calls"]), np.int64),
           "n_meshes": np.array(len(log["mesh_calls"]), np.int64)}
    for k, c in enumerate(log["calls"]):
        for name, dtype in (("F", np.int32), ("s", np.float32), ("c_final", np.float64), ("m", np.float32),
                            ("present", np.uint8), ("unknown", np.uint32)):
            out[f"call.{k}.{name}"] = np.ascontiguousarray(c[name], dtype)
        out[f"call.{k}.unknown_keys"] = np.array(c["unknown_keys"] or "")
    for k, c in enumerate(log["mesh_calls"]):
        for name, dtype in (("objects", np.int32), ("F", np.int32), ("s", np.float32), ("table", np.int64),
                            ("co", np.float32)):
            out[f"mesh.{k}.{name}"] = np.ascontiguousarray(c[name], dtype)
    return out


def read_log(path: Path) -> dict:
    """The log :func:`log_arrays` wrote, as ``_scene.motion_session(log=...)`` takes it."""
    with np.load(path, allow_pickle=False) as z:
        calls = [{**{n: z[f"call.{k}.{n}"] for n in ("F", "s", "c_final", "m", "present", "unknown")},
                  "unknown_keys": str(z[f"call.{k}.unknown_keys"])} for k in range(int(z["n_calls"]))]
        meshes = [{n: z[f"mesh.{k}.{n}"] for n in ("objects", "F", "s", "table", "co")}
                  for k in range(int(z["n_meshes"]))]
        return {"bodies": z["bodies"], "calls": calls, "mesh_calls": meshes}


def merged_report(st_report: dict, items: list[dict]) -> dict:
    """conversion_report.json (§3.12): the statics' report with the session's items after them, the summary
    recounted."""
    doc = dict(st_report)
    doc["items"] = list(st_report["items"]) + list(items)
    count = {"refused": "refuse", "warnings": "warn", "exact": "exact", "approx": "approx", "info": "info",
             "errors": "error"}
    doc["summary"] = {k: sum(1 for it in doc["items"] if it["severity"] == sev) for k, sev in count.items()}
    return doc


def write_outputs(ex: Exchange, res: Mapping[str, Any], inputs: Inputs, st: dict) -> dict:
    """Writes ``motion/`` and the report for a finished session (``res``: ``_scene.motion_session``'s result, ``st``:
    :func:`statics` about its origin) and returns the session document. The motion files go first, atomically; the key
    of the bundle cache hashes them."""
    import ballasim
    from ballasim import _scene

    motion = ex.root / "motion"
    motion.mkdir(parents=True, exist_ok=True)
    traj = trajectory_arrays(res["trajectories"])
    cams = [c["name"] for c in res["session"]["cameras"]]
    key_inputs = motion_key_inputs(_values_for_key(inputs, ex.root), ex, st, cams)
    doc = dict(res["session"])
    doc["engine"] = {"version": ballasim.__version__, "converter": int(_scene.blender_formats()["converter"])}
    doc["inputs"] = inputs.record()
    doc["options"] = {**inputs.values}
    doc["motion_key"] = motion_key(key_inputs)
    doc["motion_key_inputs"] = key_inputs
    doc["trajectories"] = [{"span_ns": list(t["span_ns"]), "K": t["K"], "controls": len(t["rot"]),
                            "pieces": 1 if t["pieces"] is None else len(t["pieces"])} for t in res["trajectories"]]
    np.savez(motion / ".trajectories.tmp.npz", **traj)
    os.replace(motion / ".trajectories.tmp.npz", motion / "trajectories.npz")
    np.savez(motion / ".oracle_log.tmp.npz", **log_arrays(res["log"]))
    os.replace(motion / ".oracle_log.tmp.npz", motion / "oracle_log.npz")
    write_atomic(motion / "session_perf.json", canonical(res["perf"]))
    write_atomic(motion / "session.json", canonical(doc))   # last: its presence marks a complete session
    return doc


def finish(ex: Exchange, res: dict, inputs: Inputs) -> dict:
    """Everything after the session: the statics about its origin, ``motion/``, the report and the cached bundle.
    Returns ``{status, report, summary, cache_key}`` (``done``'s content); ``status`` is ``refused`` when the session or
    the placement refused the scene."""
    values = _values_for_key(inputs, ex.root)
    summary = dict(res["summary"])
    if res["refused"]:
        report = res["conversion_report"]   # the session's own conversion (about the Blender origin) and its items
        write_atomic(ex.root / "conversion_report.json", canonical(report))
        summary["report_summary"] = report["summary"]
        return {"status": "refused", "report": report, "summary": summary, "cache_key": None}
    dynamic = [r["id"] for r in res["session"]["bodies"] if r["role"] != "rig" and r["trajectory"] >= 0]
    st = statics(ex, inputs.convert, origin=res["origin_world"], dynamic_ids=dynamic)
    report = merged_report(st["report"], res["report"])
    write_atomic(ex.root / "conversion_report.json", canonical(report))
    summary["report_summary"] = report["summary"]
    if st["refused"]:
        return {"status": "refused", "report": report, "summary": summary, "cache_key": None}
    doc = write_outputs(ex, res, inputs, st)
    traj = trajectory_arrays(res["trajectories"])
    try:
        bundle = assemble_moving(ex, st, doc, traj, values)
    except ValueError as e:   # the bundle validator's refusal (the 16 m rule, a span, ...)
        report["items"].append({"code": "BRG-BUNDLE", "severity": "refuse", "object": "", "message": str(e),
                                "bound": 0.0, "bound_kind": "none"})
        write_atomic(ex.root / "conversion_report.json", canonical(merged_report(report, [])))
        return {"status": "refused", "report": report, "summary": summary, "cache_key": None}
    key = blender_cache_key(values, ex)
    bundle.save(ex.root / "cache" / key)
    summary["cache_key"] = key
    summary["motion_key"] = doc["motion_key"]
    return {"status": "ok", "report": report, "summary": summary, "cache_key": key}


# --- the commands -----------------------------------------------------------------------------------------------------


def check_select(ex: Exchange, inputs: Inputs) -> None:
    """InputError for a ``run.select`` or ``run.frames_select`` camera the exchange does not have (C1 names,
    blender.md §3.4)."""
    from ballasim import _scene

    objects = ex.doc["objects"]
    names = _scene.blender_camera_names([objects[c["object"]]["name_full"] for c in ex.doc.get("cameras") or []])
    for row, chosen in (("run.select", inputs.select), ("run.frames_select", inputs.frames_select)):
        missing = [n for n in chosen or [] if n not in names]
        if missing:
            raise InputError(f"{row} names {', '.join(missing)}: the exchange's cameras are {', '.join(names)}")


def _session_args(ex: Exchange, inputs: Inputs) -> tuple:
    return (ex.doc, ex.arrays, ex.digest, inputs.convert, inputs.session)


def _extra(ex: Exchange) -> dict:
    return {"textures": texture_paths(ex), "calibrations": calibrations(ex.doc)}


def _error_of(e: BaseException) -> tuple[str, str]:
    """(code, message) of an exception from the engine: the §6.1 code its message starts with, else the type's name."""
    text = str(e)
    m = _CODE.match(text)
    return (m.group(1), text[m.end():]) if m else (type(e).__name__, text)


def run(exchange_dir: str | os.PathLike, config: str | os.PathLike | None, sets: Iterable[str], reader: BinaryIO,
        writer: BinaryIO) -> int:
    """The session on a pipe pair. Exit status: 0 done, 2 refused or a configuration or file error, 1 any other
    failure."""
    import ballasim
    from ballasim import _core, _scene

    wire = Wire(reader, writer)
    oracle = PipeOracle(wire)
    try:
        wire.send("hello", protocol=PROTOCOL, version=list(VERSION), engine=ballasim.__version__,
                  build=dict(_core.build_info()))
        head, _ = wire.recv("hello")
        version = head.get("version")
        if head.get("protocol") != PROTOCOL or not (isinstance(version, list) and version[:1] == [VERSION[0]]):
            wire.send("error", type="ProtocolError", code="BRG-PROTO",
                      message=f"the add-on speaks {head.get('protocol')} {head.get('version')}, this engine "
                              f"{PROTOCOL} {VERSION[0]}.x")
            return 2
        try:
            inputs = inputs_of(config, sets)
            ex = read_exchange(exchange_dir)
            check_select(ex, inputs)
        except (InputError, ExchangeError, FileNotFoundError) as e:
            wire.send("error", type=type(e).__name__, code="BRG-PROTO" if isinstance(e, ExchangeError) else "CONFIG",
                      message=str(e)[:8000])
            return 2
        oracle.progress(0.0, "convert")
        oracle.start_heartbeat()
        res = _scene.motion_session(*_session_args(ex, inputs), oracle=oracle.sample, mesh_oracle=oracle.mesh_probe,
                                    on_bodies=oracle.bodies, on_simplify=oracle.simplify,
                                    on_progress=oracle.progress, **_extra(ex))
        oracle.progress(0.97, "writing the motion")
        done = finish(ex, res, inputs)
        oracle.stop_heartbeat()
        body = json.dumps({"report": done["report"], "summary": done["summary"]}, allow_nan=False).encode()
        wire.send("done", body, status=done["status"], format="json")
        return 0 if done["status"] == "ok" else 2
    except AddonAbort as e:
        oracle.stop_heartbeat()
        print(f"ballasim blender session: {e}", file=sys.stderr, flush=True)
        return 1
    except (SessionError, ValueError, RuntimeError, OSError) as e:
        oracle.stop_heartbeat()
        code, message = (e.code, e.detail) if isinstance(e, SessionError) else _error_of(e)
        print(f"ballasim blender session: {code}: {message}", file=sys.stderr, flush=True)
        try:
            wire.send("error", type=type(e).__name__, code=code, message=message[:8000])
        except SessionError:
            pass
        return 2 if isinstance(e, (ValueError, OSError)) and not isinstance(e, SessionError) else 1


def command(args) -> int:
    """``blender session`` (the add-on's pipe) and ``blender replay`` (the recorded log)."""
    if args.blender_command == "replay":
        return replay(args.exchange, args.config, args.set or [])
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    return run(args.exchange, args.config, args.set or [], sys.stdin.buffer, out)


def replay(exchange_dir: str | os.PathLike, config: str | os.PathLike | None, sets: Iterable[str]) -> int:
    """``blender replay``: refits the recorded session of an exchange from ``motion/oracle_log.npz`` (Blender is not
    needed); the recorded session's options apply unless ``--config`` or ``--set`` name others. Prints a JSON
    summary."""
    from ballasim import _scene

    try:
        ex = read_exchange(exchange_dir)
        motion = ex.root / "motion"
        recorded = json.loads((motion / "session.json").read_text())
        log = read_log(motion / "oracle_log.npz")
        if config is None and not list(sets):
            rec = recorded["inputs"]
            inputs = Inputs(rec["values"], rec["convert"], {"workers": default_workers(), **rec["session"]},
                            rec["select"], rec.get("frames_select"))
        else:
            inputs = inputs_of(config, sets)
        check_select(ex, inputs)
        res = _scene.motion_session(*_session_args(ex, inputs), log=log, **_extra(ex))
        done = finish(ex, res, inputs)
    except (InputError, ExchangeError, FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as e:
        code, message = _error_of(e)
        print(f"error: {code}: {message}", file=sys.stderr)
        return 1 if code.startswith("BRG-") else 2
    print(json.dumps({"status": done["status"], "cache_key": done["cache_key"],
                      "motion_key": done["summary"].get("motion_key"), "report": done["report"]["summary"]}, indent=1))
    return 0 if done["status"] == "ok" else 2
