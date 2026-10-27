"""Roots, listings and runs (docs/design/phase3/ui.md §3.2, §3.3 ``/attach``, ``/fs``, ``/runs``; §4.10, §4.11).

- **Roots** are the ``--root`` directories, ``evimo2`` (read-only), ``jobs`` and the **attached** directories
  (``ext-<first 8 hex of SHA-256(realpath)>``): a job or run directory that ``POST /attach`` (``ballasim ui --open
  PATH``) registers read-only and records in the runtime file, from which a restarted server takes them back. One list
  of roots is shared with the event streams, so an attached run's events open at once. A path is **writable** only
  inside a writable root and outside every attached one: the server and its jobs never write into an attached root.
- **Listing** (``/fs``): one directory level, kinds by name, hidden lock files left out, entries whose real path leaves
  the roots left out. A run's summary is read from its run.json, cached by (path, mtime_ns, size); its liveness is not
  cached: ``live`` when a listed job owns the run or a run holds the run lock (``runlog.run_dir_locked``), ``stale``
  when run.json says ``running`` without either (a crash, a kill, a reboot), else ``done``.
- **Runs** (``/runs/{rid}``): run.json without ``resolved``, its files and capabilities, each ``true`` or the refusal
  code of §4.11 (format-1, supplied-bundle and dict-source runs with a relative source path); ``record``, ``progress``
  and ``perf`` (complete lines from a byte offset).
- **Tracks** (``/runs/{rid}/tracks``, BSTK; UI-7): a run that is a job's ``run/`` gets the job's
  ``preview/index.json`` (BL §3.3, written by the runner's preview step, inside the roots) as a ``preview`` track. Its
  frame times and windows are on the bundle clock (TimeMap); the track carries them on the preview camera's sensor
  clock, t + clock_offset_ns + llround(clock_drift_ppm 1e-6 t) with the run's resolved values (§4.6: the output cut's
  formula, no delay, no quantum; C++ ``ballasim._frames.sensor_time``, as the dense frames' ``/t_ns``). ``frames`` and
  ``imu`` tracks wait for Q19 (UI-15).
- **The OS's refusals** (a directory or file this user may not read) map as §6 says (``streams._path_errors``: 403
  BS-UI-PATH, a vanished path 404, any other OSError 422 BS-UI-SOURCE), never a 500.
- **Rebuilding a run's specification** (§3.6 ``{"run": rid}``) for a ``run`` job (and UI-4a's ``/resolve``):
  :func:`rebuild_spec`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ballasim.engine.runlog import run_dir_locked

from .streams import StreamError, _path_errors, bstk_message, decode_rid, encode_rid, resolve_path

log = logging.getLogger("ballasim.ui")

ATTACHED = "ext-"
#: Files a listing never shows: lock files (the run lock, ``.dispatch.lock``) and atomic-write temporaries.
_HIDDEN_SUFFIXES = (".lock", ".tmp")
_EVIMO2_EVENTS = "dataset_events_t.npy"
#: Path-typed keys that name a scene source: a dict-source run (no file) must give them absolute (§3.6).
SOURCE_PATH_KEYS = ("scene.path", "scene.evimo2.sequence", "scene.evimo2.objects_root", "scene.evimo2.raw_root")
#: Keys a rebuilt specification drops: every UI job writes its own run directory (stage none, no digest moves).
OUTPUT_KEYS = ("output.dir", "output.overwrite")
#: Run pages list at most this many files.
MAX_RUN_FILES = 2000
MAX_SUMMARIES = 4096
#: ``/perf`` answers at most this many bytes of complete lines per request.
MAX_PERF_BYTES = 4 << 20
#: Capabilities whose feature a later increment builds (the endpoint does not exist yet): BS-UI-NOT-FOUND.
_UNBUILT = ("cache", "resim")
#: A track holds at most this many rows (a preview of 2^20 Blender frames is far beyond any timeline).
MAX_TRACK_ROWS = 1 << 20


def api_error(status: int, code: str, message: str) -> StreamError:
    return StreamError(status, code, message)


@dataclass(frozen=True)
class Root:
    id: str
    path: Path
    readonly: bool

    def as_json(self) -> dict:
        return {"id": self.id, "path": str(self.path), "readonly": self.readonly}


def attached_root_id(real: str | os.PathLike) -> str:
    return ATTACHED + hashlib.sha256(os.fspath(real).encode()).hexdigest()[:8]


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def read_json(path: Path) -> dict | None:
    """A JSON object file, or None when it is absent, unreadable or not an object."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def run_format(doc: dict) -> int:
    """2 for a run.json with its ``resolved`` block (P0b), 1 before (§4.11)."""
    return 2 if isinstance(doc.get("resolved"), dict) else 1


def _nested(cfg, key: str):
    """``key``'s value in a format-2 configuration (nested mappings, or the dotted key itself at the top)."""
    if not isinstance(cfg, dict):
        return None
    if key in cfg:
        return cfg[key]
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _drop(cfg: dict, key: str) -> None:
    cfg.pop(key, None)
    head, _, last = key.rpartition(".")
    node = _nested(cfg, head)
    if isinstance(node, dict):
        node.pop(last, None)


def kind_refusal(doc: dict) -> tuple[str, str] | None:
    """The refusal of a run whose specification cannot be rebuilt (§4.11): a format-1 run (no ``resolved``), a
    supplied-bundle run (no scene keys) or a dict-source run with a relative source path; (code, message) or None."""
    if run_format(doc) == 1:
        return "BS-UI-RUN-FORMAT1", ("a format-1 run (run.json has no resolved block): migrate it with python "
                                     "tools/config/convert_v1.py --runs DIR")
    inputs = doc.get("inputs") or {}
    if (inputs.get("bundle") or {}).get("supplied"):
        return "BS-UI-RUN-SUPPLIED", "a supplied-bundle run: its configuration holds no scene keys"
    if inputs.get("file") is None:
        for key in SOURCE_PATH_KEYS:
            value = _nested(inputs.get("config"), key)
            if isinstance(value, str) and not os.path.isabs(os.path.expanduser(value)) and not value.startswith("$"):
                return "BS-UI-RUN-RELPATH", f"a dict-source run with the relative source path {key} = {value!r}"
    return None


def rebuild_spec(doc: dict) -> tuple[dict, Path | None, list[str], list[str], list[str]]:
    """A run's specification from its run.json ``inputs`` (§3.6 ``{"run": rid}``): ``(config, base_dir, items,
    profiles, warnings)``. ``config`` is the raw configuration without ``output.dir`` and ``output.overwrite``;
    ``base_dir`` the directory of its file (None for a dict source, whose source paths are absolute); ``items`` the
    recorded CLI layer as ``KEY=<JSON>`` items (JSON is YAML flow: null, numbers, bools, lists and strings such as
    ``'1e3'`` or ``'#'`` round-trip), the same two keys dropped; ``profiles`` the recorded ``--profile`` ids;
    ``warnings`` a line per recorded profile whose file changed since. The user's items go after ``items``, so they
    win. Raises the run's :func:`kind_refusal` as a StreamError (422)."""
    from ballasim.config import profile_path

    refusal = kind_refusal(doc)
    if refusal is not None:
        raise api_error(422, *refusal)
    inputs = doc["inputs"]
    config = copy.deepcopy(inputs.get("config") or {})
    for key in OUTPUT_KEYS:
        _drop(config, key)
    base = Path(inputs["file"]).parent if inputs.get("file") else None
    cli = inputs.get("cli") or {}
    items = [f"{k}={json.dumps(v)}" for k, v in (cli.get("set") or {}).items() if k not in OUTPUT_KEYS]
    warnings = []
    for p in inputs.get("profiles") or []:
        try:
            now = hashlib.sha256(profile_path(p["id"]).read_bytes()).hexdigest()
        except (OSError, KeyError, TypeError):
            now = None
        if isinstance(p, dict) and now != p.get("sha256"):
            warnings.append(f"profile {p.get('id')} changed since the run")
    return config, base, items, list(cli.get("profiles") or []), warnings


def _int64(v) -> bool:
    return type(v) is int and -(1 << 63) <= v < (1 << 63)


def camera_clock(doc: dict, camera) -> tuple[int, float] | None:
    """``(clock_offset_ns, clock_drift_ppm)`` the run resolved for ``camera`` (run.json ``resolved.cameras``; a drift
    the run did not record reads 0, as for the dense frames), or None: no such camera, a format-1 run, a malformed
    record (any level not an object), or a drift outside the sensor row's hard range (-1e4, 1e4) ppm."""
    def value(entry: dict, key: str, missing):
        if key not in entry:
            return missing
        e = entry[key]
        return e.get("value") if isinstance(e, dict) else None

    resolved = doc.get("resolved")
    cams = resolved.get("cameras") if isinstance(resolved, dict) else None
    entry = cams.get(camera) if isinstance(cams, dict) and isinstance(camera, str) else None
    if not isinstance(entry, dict):
        return None
    offset = value(entry, "sensor.clock_offset_ns", None)
    drift = value(entry, "sensor.clock_drift_ppm", 0.0)
    if not _int64(offset) or isinstance(drift, bool) or not isinstance(drift, (int, float)) or not -1e4 < drift < 1e4:
        return None
    return offset, float(drift)


def preview_rows(index: dict) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    """``(camera, window kind, t_ns, index, window)`` of a ``ballasim.preview-index`` document (the runner's preview
    step, BL §3.3), times on the bundle clock; ValueError when it is not one."""
    if index.get("format") != "ballasim.preview-index" or not isinstance(index.get("frames"), list):
        raise ValueError("not a ballasim.preview-index document")
    frames = index["frames"]
    if len(frames) > MAX_TRACK_ROWS:
        raise ValueError(f"{len(frames)} preview frames (at most {MAX_TRACK_ROWS})")
    t, idx, win = [], [], []
    for f in frames:
        w = f.get("window") if isinstance(f, dict) else None
        if not (isinstance(w, list) and len(w) == 2 and all(map(_int64, (f.get("t_ns"), *w)))
                and type(f.get("index")) is int and 0 <= f["index"] < (1 << 32)):
            raise ValueError(f"a malformed preview frame: {f!r:.200}")
        t.append(f["t_ns"])
        idx.append(f["index"])
        win.append(w)
    camera, kind = index.get("camera"), index.get("window")
    if not (isinstance(camera, str) and isinstance(kind, str) and camera and kind):
        raise ValueError("the preview index names no camera or window")
    return (camera, kind, np.array(t, np.int64), np.array(idx, np.uint32),
            np.array(win, np.int64).reshape(len(win), 2))


def sensor_clock(t_ns: np.ndarray, clock: tuple[int, float]) -> np.ndarray:
    """True times on a camera's sensor clock (§4.6): C++ ``_frames.sensor_time``, the readout formula without the
    delay and the quantum (the output cut's; UI-4b's ``_core.config.sensor_clock_ns`` is the same formula)."""
    from ballasim import _frames

    return np.asarray(_frames.sensor_time(np.ascontiguousarray(t_ns, np.int64), clock[0], clock[1]), np.int64)


class Fs:
    """The roots and the run and directory views of one server. ``roots`` is the server's list (shared with the
    streams); ``owned_runs()`` gives the real paths of the run directories that listed, non-final jobs own."""

    def __init__(self, roots: list, *, owned_runs=None):
        self.roots = roots
        self.owned_runs = owned_runs or (lambda: set())
        self._summaries: dict[str, tuple[tuple, dict]] = {}

    # -- roots ---------------------------------------------------------------------------------------------------

    def root(self, root_id: str):
        r = next((r for r in self.roots if r.id == root_id), None)
        if r is None:
            raise api_error(404, "BS-UI-NOT-FOUND", f"no root {root_id!r}")
        return r

    def attached(self) -> list[dict]:
        return [{"root_id": r.id, "path": str(r.path)} for r in self.roots if r.id.startswith(ATTACHED)]

    @staticmethod
    def attach_dir(path: Path) -> tuple[str, Path]:
        """The job or run directory ``path`` names (itself or its nearest parent holding ``job.json`` or
        ``run.json``), to attach (:meth:`add_attached`): ``(kind, real directory)``. 422 BS-UI-ATTACH otherwise."""
        try:
            real = Path(os.path.realpath(path))
            start = real if real.is_dir() else real.parent
            found = None
            for d in (start, *start.parents):
                if (d / "job.json").is_file():
                    found = ("job", d)
                elif (d / "run.json").is_file():
                    found = ("run", d)
                if found:
                    break
        except (OSError, ValueError):
            found = None
        if found is None:
            raise api_error(422, "BS-UI-ATTACH", f"{path}: neither a job directory (job.json) nor a run directory "
                                                 "(run.json), nor a file inside one")
        return found

    def add_attached(self, real: Path) -> str:
        root_id = attached_root_id(real)
        if not any(r.id == root_id for r in self.roots):
            self.roots.append(Root(root_id, Path(real), True))
        return root_id

    def restore(self, entries) -> None:
        """Attached roots recorded by an earlier server (its runtime file), those that still exist."""
        for e in entries or []:
            path = e.get("path") if isinstance(e, dict) else None
            if isinstance(path, str) and os.path.isdir(path) and (
                    os.path.isfile(os.path.join(path, "job.json")) or os.path.isfile(os.path.join(path, "run.json"))):
                self.add_attached(Path(os.path.realpath(path)))

    def rid(self, path: str | os.PathLike) -> str | None:
        """The rid of a real path: in the most specific root that holds it, or None outside the roots."""
        real = os.path.realpath(path)
        best = None
        for r in self.roots:
            rp = os.path.realpath(r.path)
            if _inside(real, rp) and (best is None or len(rp) > len(best[1])):
                best = (r, rp)
        if best is None:
            return None
        return encode_rid(best[0].id, os.path.relpath(real, best[1]) if real != best[1] else "")

    def writable(self, path: str | os.PathLike) -> bool:
        """Whether the server and its jobs may write at ``path`` (which need not exist): inside a writable root and
        outside every attached root."""
        real = os.path.realpath(path)
        inside = [r for r in self.roots if _inside(real, os.path.realpath(r.path))]
        return bool(inside) and not any(r.readonly or r.id.startswith(ATTACHED) for r in inside)

    def default_output_root(self) -> Path:
        """The first writable root (``--root`` order; ``jobs`` last)."""
        for r in self.roots:
            if not r.readonly and not r.id.startswith(ATTACHED):
                return Path(r.path)
        raise api_error(403, "BS-UI-PATH", "no writable root")

    def resolve(self, ref) -> Path:
        return resolve_path(self.roots, ref)

    def target(self, ref, what: str) -> Path:
        """A path to write (a run directory, a video), given as an absolute path or a rid; it need not exist, but it
        must be writable (:meth:`writable`)."""
        if not isinstance(ref, str) or not ref or "\0" in ref:
            raise api_error(400, "BS-UI-REQUEST", f"{what}: an absolute path or a resource id")
        if ref.startswith("/"):
            path = Path(os.path.normpath(ref))
        else:
            root_id, rel = decode_rid(ref)
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise api_error(403, "BS-UI-PATH", f"{what}: the path leaves its root")
            path = Path(os.path.normpath(os.path.join(self.root(root_id).path, rel)))
            if "\0" in rel:
                raise api_error(400, "BS-UI-REQUEST", f"{what}: not a valid path")
        if not self.writable(path):
            raise api_error(403, "BS-UI-PATH", f"{what}: {path} is not inside a writable root")
        return path

    # -- runs ----------------------------------------------------------------------------------------------------

    def liveness(self, run_dir: Path, status: str | None) -> str:
        if os.path.realpath(run_dir) in self.owned_runs() or run_dir_locked(run_dir):
            return "live"
        return "stale" if status == "running" else "done"

    def summary(self, run_dir: Path) -> dict | None:
        """The listing's run summary (§3.3 ``/fs``), cached by run.json's (path, mtime_ns, size); None without a
        readable run.json."""
        path = run_dir / "run.json"
        try:
            st = path.stat()
        except OSError:
            return None
        key, cached = str(path), self._summaries.get(str(path))
        stamp = (st.st_mtime_ns, st.st_size)
        if cached is None or cached[0] != stamp:
            doc = read_json(path)
            if doc is None:
                return None
            cams = doc.get("cameras") if isinstance(doc.get("cameras"), list) else None
            resolved = doc.get("resolved") or {}
            names = ([c.get("name") for c in cams] if cams is not None
                     else list((resolved.get("cameras") or {}).keys()))
            span = doc.get("span_ns")
            info = {"status": doc.get("status"), "format": run_format(doc),
                    "supplied": bool(((doc.get("inputs") or {}).get("bundle") or {}).get("supplied")),
                    "span_ns": [str(int(t)) for t in span] if isinstance(span, list) and len(span) == 2 else None,
                    "cameras": names, "wall_s": doc.get("wall_s"), "simulated_s": doc.get("simulated_s"),
                    "events": sum(int(c.get("events") or 0) for c in cams) if cams is not None else None,
                    "warnings": list(doc.get("warnings") or [])}
            if len(self._summaries) >= MAX_SUMMARIES:
                self._summaries.clear()
            self._summaries[key] = cached = (stamp, info)
        return {**cached[1], "live": self.liveness(run_dir, cached[1]["status"])}

    def run_dir(self, rid: str) -> tuple[Path, dict]:
        path = self.resolve(rid)
        doc = read_json(path / "run.json")
        if doc is None:
            raise api_error(404, "BS-UI-NOT-FOUND", "not a run directory (no readable run.json)")
        return path, doc

    @_path_errors()
    def run_page(self, rid: str, *, repo: bool = False) -> dict:
        """``GET /runs/{rid}``: run.json without ``resolved``, ``live``, ``files`` and ``capabilities`` (§4.11)."""
        path, doc = self.run_dir(rid)
        live = self.liveness(path, doc.get("status"))
        files = []
        for d, dirs, names in os.walk(path):
            dirs.sort()
            for name in sorted(names):
                if name.startswith(".") and name.endswith(_HIDDEN_SUFFIXES):
                    continue
                files.append(os.path.relpath(os.path.join(d, name), path))
                if len(files) >= MAX_RUN_FILES:
                    break
            if len(files) >= MAX_RUN_FILES:
                break
        page = {k: v for k, v in doc.items() if k != "resolved"}
        page.update(rid=rid, live=live, format=run_format(doc), files=files,
                    capabilities=self.capabilities(path, doc, live, files, repo))
        return page

    @staticmethod
    def capabilities(path: Path, doc: dict, live: str, files: list[str], repo: bool = False) -> dict:
        events = [f for f in files if f.startswith("events/") and f.endswith(".h5")]
        partial = [f for f in files if f.startswith("events/.") and f.endswith(".partial")]
        refusal = kind_refusal(doc)
        kind = refusal[0] if refusal else None
        caps = {"view": True if events else "BS-UI-PARTIAL" if partial else "BS-UI-NOT-FOUND"}
        caps["resolve"] = kind or True                  # UI-4a's /resolve rebuilds the run's specification (§3.6)
        for name in _UNBUILT:
            caps[name] = kind or "BS-UI-NOT-FOUND"
        dense = kind if kind != "BS-UI-RUN-RELPATH" else None       # dense reads the resolved, absolute values
        caps["dense"] = dense or ("BS-UI-RUN-LIVE" if live == "live" else True)
        # the bench level (UI-8b) needs a source checkout (--repo) and a run it can read: events, run.json's resolved
        # clock offset (not format 1), complete
        caps["bench"] = ("BS-UI-BENCH-UNAVAILABLE" if not repo else caps["view"] if caps["view"] is not True
                         else "BS-UI-RUN-FORMAT1" if run_format(doc) == 1
                         else "BS-UI-RUN-LIVE" if live == "live" else True)
        return caps

    @_path_errors()
    def record(self, rid: str) -> dict:
        _, doc = self.run_dir(rid)
        return {k: doc.get(k) for k in ("resolved", "inputs", "stage_digests")}

    @_path_errors()
    def progress(self, rid: str) -> dict:
        path = self.resolve(rid)
        doc = read_json(path / "progress.json")
        if doc is None:
            raise api_error(404, "BS-UI-NOT-FOUND", "the run has no progress.json")
        return doc

    @_path_errors()
    def perf(self, rid: str, since: int) -> tuple[bytes, int]:
        """Complete lines of ``logs/perf.jsonl`` from byte ``since`` (at most MAX_PERF_BYTES) and the next offset."""
        path = self.resolve(rid) / "logs" / "perf.jsonl"
        return read_lines(path, since, MAX_PERF_BYTES)

    # -- tracks (UI-7) -------------------------------------------------------------------------------------------

    def _preview_index(self, run_dir: Path) -> Path | None:
        """The preview index of the job whose ``run/`` ``run_dir`` is (BL §3.3), when it lies inside the roots (a run
        attached alone does not reach its job's files)."""
        job = run_dir.parent
        if run_dir.name != "run" or not (job / "job.json").is_file():
            return None
        try:
            return resolve_path(self.roots, str(job / "preview" / "index.json"))
        except StreamError:
            return None

    def _tracks(self, rid: str) -> list[tuple[dict, np.ndarray, list]]:
        """The run's tracks: (listing entry, t_ns on the sensor clock, BSTK extra columns). A track whose file is
        malformed, or whose camera the run did not resolve, is left out (logged)."""
        path, doc = self.run_dir(rid)
        out = []
        index_path = self._preview_index(path)
        if index_path is not None:
            try:
                index = read_json(index_path)
                if index is None:
                    raise ValueError("unreadable, or not a JSON object")
                camera, kind, t, idx, win = preview_rows(index)
                clock = camera_clock(doc, camera)
                if clock is None:
                    raise ValueError(f"the run records no resolved sensor clock of camera {camera!r}")
                t, win = sensor_clock(t, clock), sensor_clock(win.ravel(), clock).reshape(win.shape)
            except ValueError as e:
                log.warning("%s: no preview track: %s", index_path, e)
            else:
                entry = {"id": "preview", "kind": "preview", "label": f"Blender preview ({camera}, {kind} windows)",
                         "camera": camera, "window": kind, "source": self.rid(index_path), "n": len(t),
                         "t_range_ns": [str(int(t.min())), str(int(t.max()))] if len(t) else None}
                out.append((entry, t, [("index", idx), ("window", win)]))
        return out

    @_path_errors()
    def tracks(self, rid: str) -> list[dict]:
        """``GET /runs/{rid}/tracks`` (§3.3)."""
        return [entry for entry, _, _ in self._tracks(rid)]

    @_path_errors()
    def track(self, rid: str, track_id: str, t0_ns: int | None, t1_ns: int | None) -> bytes:
        """``GET /runs/{rid}/tracks/{id}?t0_ns&t1_ns``: BSTK (§3.5) of the track's rows with t0 <= t_ns < t1 on the
        sensor clock (default: all); its header's t0_ns is the request's (default: the earliest row's, else 0)."""
        found = next((tr for tr in self._tracks(rid) if tr[0]["id"] == track_id), None)
        if found is None:
            raise api_error(404, "BS-UI-NOT-FOUND", f"the run has no track {track_id!r}")
        entry, t, columns = found
        keep = np.ones(len(t), bool)
        if t0_ns is not None:
            keep &= t >= t0_ns
        if t1_ns is not None:
            keep &= t < t1_ns
        t0 = t0_ns if t0_ns is not None else int(t.min()) if len(t) else 0
        return bstk_message(entry["kind"], t0, t[keep], [(name, values[keep]) for name, values in columns])

    # -- listing -------------------------------------------------------------------------------------------------

    @_path_errors()
    def listing(self, root_id: str, rel: str) -> tuple[Path, list[dict]]:
        """``GET /fs``: the directory and one level of it (§3.3); a job's summary is added on the loop (the jobs'
        state lives there)."""
        self.root(root_id)
        directory = self.resolve(encode_rid(root_id, rel))
        if not directory.is_dir():
            raise api_error(422, "BS-UI-REQUEST", "not a directory")
        allowed = [os.path.realpath(r.path) for r in self.roots]
        out = []
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name)
        for e in entries:
            if e.name.startswith(".") and e.name.endswith(_HIDDEN_SUFFIXES):
                continue
            real = os.path.realpath(e.path)
            if not any(_inside(real, a) for a in allowed):
                continue                        # a symlink out of the roots is not shown (listing it: 403)
            try:
                st = os.stat(real)
            except OSError:
                continue
            entry = {"name": e.name, "kind": "file", "size": st.st_size, "mtime_ns": str(st.st_mtime_ns)}
            p = Path(real)
            if stat.S_ISDIR(st.st_mode):
                entry["kind"] = self._dir_kind(p)
                if entry["kind"] == "run":
                    entry["run"] = self.summary(p)
            else:
                entry["kind"] = file_kind(e.name, directory.name)
            out.append(entry)
        return directory, out

    @staticmethod
    def _dir_kind(p: Path) -> str:
        """A directory's kind by the files it holds; ``dir`` when this user may not search it."""
        try:
            if (p / "run.json").is_file():
                return "run"
            if (p / "job.json").is_file() or (p / "ui.json").is_file():
                return "job"
            if (p / _EVIMO2_EVENTS).is_file():
                return "evimo2_seq"
        except OSError:
            pass
        return "dir"


def file_kind(name: str, parent: str) -> str:
    """A file's kind by name (§3.3 ``/fs``)."""
    lower = name.lower()
    if name.startswith(".") and lower.endswith(".partial"):
        return "partial"
    if lower.endswith((".h5", ".hdf5")):
        return {"dense": "dense_h5", "signal_cache": "signal_cache"}.get(parent, "events_h5")
    if lower.endswith((".yaml", ".yml")):
        return "config"
    return "file"


def read_lines(path: Path, since: int, limit: int) -> tuple[bytes, int]:
    """The complete (``\\n``-terminated) lines of a file from byte ``since``, at most ``limit`` bytes (one longer
    line whole), and the next offset."""
    try:
        with open(path, "rb") as f:
            f.seek(since)
            data = f.read(limit)
            if len(data) == limit and b"\n" not in data:
                data += f.readline()
    except FileNotFoundError:
        return b"", since
    end = data.rfind(b"\n") + 1
    return data[:end], since + end
