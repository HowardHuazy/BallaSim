"""UI jobs: ``ballasim.job/1`` directories that BL-0's runner executes (docs/design/phase3/ui.md §3.7, §4.10, §6).

- **Kinds.** ``run``, ``evimo2``, ``bundle``, ``dense``, ``export_video`` and ``bench`` (``evb.py compare`` of a run
  against an evb span, with ``--repo`` only; ``signal_cache`` comes with UI-9). The server builds every configuration
  and argv itself (:meth:`Jobs.create`); the client sends a kind and validated arguments. A job is
  ``<jobs_root>/<YYYYmmdd-HHMMSS>-<slug>/`` with ``run.yaml`` (when a step reads it), ``ui.json`` and ``job.json``,
  written last and atomically, with the user's edits as ``sets`` (E1), so they stay in run.json's ``cli`` layer.
- **Dispatch.** Only the jobs root's dispatcher (the server holding ``.dispatch.lock``) spawns, and only directories
  whose ``ui.json`` names this jobs root (Blender and attached jobs are read-only). Two FIFO lanes by ``queue_seq``: one
  GPU job (``run``/``evimo2`` on the GPU, ``dense``) and one CPU job (``bundle``, ``export_video``, ``bench``) non-final
  at a time; ``gpu_lock`` keeps other producers off the GPU. A spawn writes the marker ``ui.json.spawn``, starts the
  runner detached (own session, logs in ``bridge/``, no wrapper) with ``--allow-exec`` (the server built the argv) and
  writes ``bridge/pid.json``; the ``Popen`` is kept and reaped by :meth:`Jobs.tick`.
- **States** (§3.7): ``queued``; ``held`` (queued under an earlier server, or a spawn marker without ``pid.json``
  whose runner never took ``runner.lock``), released by ``/jobs/queue/resume``; the runner's states from status.json;
  ``stalled`` (alive, heartbeat older than 30 s: kill is offered); ``interrupted`` (not alive and not final: then,
  and only then, ``bridge/interrupted`` is written, with the exit code when known). A runner is **alive** while this
  server's child has not exited, while ``bridge/runner.lock`` is held (E3; a ``LOCK_SH`` probe), or, for a runner
  without the lock, while ``pid.json``'s pid runs (not a zombie) with its start time and boot id.
- **Cancel** writes ``cancel``, sends SIGINT to the runner's process group (once the runner has written status.json:
  before that it has no SIGINT handler yet, and it reads ``cancel`` before its first step) and SIGKILL after the
  grace period; a queued or held job is cancelled in ``ui.json``, by the dispatcher only (BS-UI-DISPATCH elsewhere).
  **Kill** sends SIGKILL to the group. The runner signalled is this server's child, else, while ``runner.lock`` is
  held, ``pid.json``'s (or status.json's) pid, else a lockless runner that :func:`runner_alive` finds.
- **Other servers' writes:** a job's ``ui.json`` is read again whenever its (mtime_ns, size, inode) moves, so a
  listing server follows the dispatcher's spawns and cancels. One unreadable or malformed job directory is skipped
  (logged once) and never stops the others' records, ticks or dispatch.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import ballasim
from ballasim.engine.job import python_argv
from ballasim.engine.runlog import write_json_atomic

from .bench import Bench
from .fs import OUTPUT_KEYS, api_error, kind_refusal, read_json, rebuild_spec, run_format
from .streams import _path_errors, decode_rid, locate_source

log = logging.getLogger("ballasim.ui")

KINDS = ("run", "evimo2", "bundle", "dense", "export_video", "bench")
FINAL = ("complete", "failed", "cancelled")
#: States in which a job holds its lane.
_IDLE = FINAL + ("queued", "held", "interrupted")
STALL_S = 30.0
KILL_GRACE_S = 30.0
MAX_LOG_BYTES = 64 << 10
VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".gif", ".png")
_CAMERA = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_PROFILE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@v\d+")
#: A full key of a ``sets`` item: dotted names (camera names may hold '.' and '-'); no blank, '=', '#' or ':'.
_KEY = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?")


def job_id_of(real: str | os.PathLike) -> str:
    """First 16 hex digits of SHA-256(realpath(job dir)): the same for the UI and the add-on (§3.3)."""
    return hashlib.sha256(os.fspath(real).encode()).hexdigest()[:16]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def proc_stat(pid: int) -> tuple[str, int] | None:
    """(state, start time) of a process: fields 3 and 22 of ``/proc/<pid>/stat``; None when it is gone."""
    try:
        rest = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return rest[0], int(rest[19])
    except (OSError, IndexError, ValueError):
        return None


def runner_alive(info: dict | None) -> bool:
    """A runner from ``pid.json``: its pid runs (not a zombie) with the recorded start time and boot id."""
    if not isinstance(info, dict) or not isinstance(info.get("pid"), int):
        return False
    st = proc_stat(info["pid"])
    return st is not None and st[0] not in ("Z", "X") and st[1] == info.get("start_time") \
        and info.get("boot_id") == boot_id()


def lock_held(path: Path) -> bool:
    """Whether a process holds ``path`` locked: a ``LOCK_SH | LOCK_NB`` probe, released at once (never created)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _heartbeat_age(status: dict | None, path: Path, fallback: float) -> float:
    """Seconds since the runner's last heartbeat: status.json's ``heartbeat``, counted from the end of its second (the
    text has 1 s resolution, so a live runner never reads up to 1 s older than it is), else status.json's mtime, else
    ``fallback``."""
    now = time.time()
    if status is not None:
        with contextlib.suppress(TypeError, ValueError):
            return max(now - datetime.strptime(status.get("heartbeat"), "%Y-%m-%dT%H:%M:%S%z").timestamp() - 1.0, 0.0)
        with contextlib.suppress(OSError):
            return now - path.stat().st_mtime
    return now - fallback


def _seq(ui: dict | None) -> int:
    """ui.json's ``queue_seq`` (0 when absent or malformed)."""
    v = (ui or {}).get("queue_seq")
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _steps(doc: dict) -> list[dict]:
    steps = doc.get("steps")
    return [st for st in steps if isinstance(st, dict)] if isinstance(steps, list) else []


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _stamp(path: Path) -> str | None:
    """A file's mtime as local ISO 8601 text, or None."""
    t = _mtime(path)
    return None if t is None else datetime.fromtimestamp(t).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40].rstrip("-") or "job"


def _refuse(message: str, code: str = "BS-UI-JOB-KIND", status: int = 422):
    return api_error(status, code, message)


def _int(value, what: str) -> int:
    """An int64 given as an integer or a decimal string (§3.3)."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise _refuse(f"{what}: an integer (or its decimal string)")
    try:
        v = int(value)
    except ValueError:
        raise _refuse(f"{what}: an integer (or its decimal string)") from None
    if not -(1 << 63) <= v < (1 << 63):
        raise _refuse(f"{what}: outside int64")
    return v


def _decimal(value, what: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _refuse(f"{what}: a number")
    try:
        d = Decimal(str(value))
    except InvalidOperation:
        raise _refuse(f"{what}: a number") from None
    if not d.is_finite():
        raise _refuse(f"{what}: a finite number")
    return d


def _text(d: Decimal) -> str:
    """Exact plain decimal text (never an exponent)."""
    return format(d.normalize(), "f") if d != 0 else "0"


def ns_to_s(ns: int) -> str:
    return _text(Decimal(ns).scaleb(-9))


def ns_to_ms(ns: int) -> str:
    return _text(Decimal(ns).scaleb(-6))


def set_items(sets) -> list[str]:
    """The ``sets`` mapping {key: "value text"} as ``KEY=VALUE`` items (§3.6: a mapping would bypass the YAML parse);
    the run's own output keys are refused (every UI job writes its own run directory: ``out``, ``overwrite``)."""
    if sets is None:
        return []
    if not isinstance(sets, dict) or not all(isinstance(k, str) and k and isinstance(v, str) for k, v in sets.items()):
        raise _refuse("sets: a mapping of full keys to value text")
    for k in sets:
        if not _KEY.fullmatch(k):       # parse_sets strips a key and splits at the first '=': " output.dir" or
            raise _refuse(f"sets: {k!r} is not a full key")      # "output.dir=x #" would name output.dir
        if any(k == o or o.startswith(k + ".") or k.startswith(o + ".") for o in OUTPUT_KEYS):
            raise _refuse(f"sets: {k} is the job's own (give out and overwrite)")
    return [f"{k}={v}" for k, v in sets.items()]


def _profiles(value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(p, str) and _PROFILE.fullmatch(p) for p in value):
        raise _refuse("profiles: a list of profile ids (group/name@vN)")
    return list(value)


@dataclass
class Plan:
    """What a kind's builder decided: run.yaml (or None), the steps, the job.json ``sets`` and its lane."""

    steps: list
    gpu: bool
    nice: int = 0
    run_yaml: dict | None = None
    sets: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)       # ui.json fields: run_dir, source_run, bundle_source, warnings


@dataclass
class Job:
    dir: Path
    job_id: str
    attached: bool
    ui: dict | None
    popen: subprocess.Popen | None = None
    exit_code: int | None = None
    held: bool = False
    final: bool = False
    state: str | None = None
    published: str | None = None
    published_final: bool = False       # the tick published the job's final record (UI-7)
    kill_timer: object = None
    bundle_recorded: bool = False
    ui_stamp: tuple | None = None       # ui.json's (mtime_ns, size, inode) when last read


class Jobs:
    """The jobs of one server: its jobs root's and the attached ones. Every method runs on the server's loop thread;
    ``create``'s builders read files only. ``runner`` is the command before the job directory (default ``python -m
    ballasim job`` with this interpreter's import flags, ``-u``); tests give a stub runner, short stall and grace
    periods. ``device()`` is the device probe's answer (the lane of ``run.device: auto``)."""

    def __init__(self, fs, settings, *, runner: list[str] | None = None, stall_s: float = STALL_S,
                 grace_s: float = KILL_GRACE_S, device=lambda: None):
        self.fs = fs
        self.root = Path(os.path.realpath(settings.jobs_root))
        self.repo = None if settings.repo is None else Path(os.path.realpath(settings.repo))
        self.bench = None if self.repo is None else Bench(self.repo)
        self.gpu_lock = str(settings.gpu_lock) if settings.gpu_lock else None
        self.python = python_argv()
        self.runner = list(runner) if runner else [*self.python, "-m", "ballasim", "job"]
        self.stall_s, self.grace_s, self.device = stall_s, grace_s, device
        self.dispatcher = False
        self.before_gpu_spawn = lambda: None            # UI-11: release_gpu to the GPU worker
        self.jobs: dict[str, Job] = {}
        self._seq = 0
        self._scanned = False
        self._root_mtime = None
        self._pending: set[Path] = set()                # directories of the jobs root without job.json yet
        self._failed: set[tuple[str, str]] = set()      # (job, exception type) logged once

    # -- discovery ---------------------------------------------------------------------------------------------

    def _ours(self, ui: dict | None) -> bool:
        return isinstance(ui, dict) and ui.get("jobs_root") == str(self.root)

    def scan(self) -> None:
        """Finds new job directories in the jobs root (when its mtime moves) and among the attached roots. At the
        first scan, the queue an earlier server left is held."""
        dirs = []
        try:
            mtime = self.root.stat().st_mtime_ns
        except OSError:
            mtime = None
        if mtime != self._root_mtime:
            self._root_mtime = mtime
            with contextlib.suppress(OSError), os.scandir(self.root) as it:
                dirs += [(Path(e.path), False) for e in it if e.is_dir(follow_symlinks=False)
                         and not e.name.startswith(".")]
        dirs += [(d, False) for d in self._pending] + [(Path(a["path"]), True) for a in self.fs.attached()]
        for d, attached in dirs:
            real = Path(os.path.realpath(d))
            jid = job_id_of(real)
            if jid in self.jobs:
                continue
            try:
                listed = (real / "job.json").is_file() or (real / "ui.json").is_file()
            except OSError:                         # a directory this user may not search: never a job of ours
                listed = False
            if not listed:
                if not attached:
                    (self._pending.add if real.is_dir() else self._pending.discard)(real)
                continue
            self._pending.discard(real)
            job = Job(real, jid, attached, None)
            self._refresh_ui(job)
            ui = job.ui
            if self._ours(ui):
                self._seq = max(self._seq, _seq(ui))
                # the queue an earlier server left waits for the dispatcher's /jobs/queue/resume
                job.held = self.dispatcher and not self._scanned and not ui.get("spawn") and not ui.get("cancelled") \
                    and not any((real / f).exists() for f in ("status.json", "bridge/pid.json"))
            self.jobs[jid] = job
        self._scanned = True

    def _refresh_ui(self, job: Job) -> None:
        """Reads the job's ui.json again when its (mtime_ns, size, inode) moved (another server's spawn or cancel);
        attached jobs have none."""
        if job.attached:
            return
        try:
            st = (job.dir / "ui.json").stat()
        except OSError:
            return
        stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
        if stamp != job.ui_stamp:
            job.ui_stamp = stamp
            ui = read_json(job.dir / "ui.json")
            if ui is not None:
                job.ui = ui

    def _skip(self, job: Job, e: Exception) -> None:
        """A job directory whose files make its record fail: logged once per exception type, then skipped."""
        key = (job.job_id, type(e).__name__)
        if key not in self._failed:
            self._failed.add(key)
            log.warning("job %s (%s): skipped: %s: %s", job.job_id, job.dir, type(e).__name__, e)

    def summary(self, path: Path) -> dict:
        """A job directory's listing summary (``/fs``: {job_id, state, kind, producer, label, listed, run_dir}), also
        outside the jobs root and the attached roots (read-only there, never spawned). ``listed``: ``/jobs/{id}``
        knows the job (this jobs root's or an attached one); ``run_dir``: its run's rid, or null (UI-7: the runs
        screen opens an unlisted job's run)."""
        real = Path(os.path.realpath(path))
        job = self.jobs.get(job_id_of(real))
        listed = job is not None
        try:
            if job is not None:
                rec = self.record(job)
            else:                                   # elsewhere in a root: read only, no marker written
                job = Job(real, job_id_of(real), False, None)
                rec = self.record(job, mark=False)
        except Exception as e:  # noqa: BLE001  a malformed job directory is listed without its state
            self._skip(job, e)
            return {"job_id": job_id_of(real), "state": None, "kind": None, "producer": None, "label": real.name,
                    "listed": listed, "run_dir": None}
        return {**{k: rec[k] for k in ("job_id", "state", "kind", "producer", "label", "run_dir")}, "listed": listed}

    def get(self, job_id: str) -> Job:
        self.scan()
        job = self.jobs.get(job_id)
        if job is None:
            raise _refuse(f"no job {job_id}", "BS-UI-NOT-FOUND", 404)
        return job

    # -- state -------------------------------------------------------------------------------------------------

    def _alive(self, job: Job, pid: dict | None) -> bool:
        if job.popen is not None:
            return True
        return lock_held(job.dir / "bridge" / "runner.lock") or runner_alive(pid)

    def record(self, job: Job, *, mark: bool = True) -> dict:
        """The job's record (§3.7 ``GET /jobs/{id}``), from its files; reaps this server's runner. ``mark``: write
        ``bridge/interrupted`` and hold a spawn marker that never ran (jobs of this jobs root and attached ones)."""
        if job.popen is not None and (rc := job.popen.poll()) is not None:
            job.exit_code, job.popen = rc, None
        self._refresh_ui(job)
        d, ui = job.dir, job.ui or {}
        status = read_json(d / "status.json")
        pid = read_json(d / "bridge" / "pid.json")
        doc = read_json(d / "job.json") or {}
        spawn = ui.get("spawn")
        liveness, exit_code, error = "dead", job.exit_code, (status or {}).get("error")
        err = error if isinstance(error, dict) else {}
        alive = self._alive(job, pid)
        if ui.get("spawn_error"):
            state, error = "failed", {"type": "SpawnError", "message": ui["spawn_error"], "exit_code": None}
        elif status is not None and status.get("state") in FINAL:
            state = status["state"]
            if exit_code is None:                   # the runner's exit code (BL §3.3: 0 complete, 3 cancelled)
                exit_code = {"complete": 0, "cancelled": 3}.get(state, err.get("exit_code"))
            liveness = "alive" if alive else "dead"
        elif self._ours(job.ui) and not spawn and pid is None and status is None:
            state = "cancelled" if ui.get("cancelled") else "held" if job.held else "queued"
        elif alive:
            since = _mtime(d / "bridge" / "pid.json") or _mtime(d / "job.json") or time.time()
            liveness = "stalled" if _heartbeat_age(status, d / "status.json", since) > self.stall_s else "alive"
            state = "stalled" if liveness == "stalled" else (status or {}).get("state") or "starting"
        elif pid is None and status is None:
            if self._ours(job.ui) and self.dispatcher:  # a server died between the spawn marker and pid.json: never
                if mark:                            # ran (re-spawning is safe: E3), so it is queued again, held
                    job.ui.pop("spawn", None)
                    write_json_atomic(d / "ui.json", job.ui)
                    job.held = True
                state = "cancelled" if ui.get("cancelled") else "held"
            else:                                   # the add-on between its job.json and its pid.json, or the
                state = "starting"                  # dispatcher's spawn seen from another server
        else:
            state = "interrupted"
            marker = d / "bridge" / "interrupted"
            if mark and not marker.exists():
                with contextlib.suppress(OSError):
                    marker.write_text(json.dumps({"at": _now(), "exit_code": exit_code}) + "\n")
            if exit_code is None:
                exit_code = (read_json(marker) or {}).get("exit_code")
        run_dir = ui.get("run_dir") or ui.get("source_run")
        if not isinstance(run_dir, str):
            run_dir = str(d / "run") if any(st.get("step") == "run" for st in _steps(doc)) else None
        started = spawn.get("at") if isinstance(spawn, dict) else _stamp(d / "bridge" / "pid.json") if pid else None
        job.state, job.final = state, state in FINAL + ("interrupted",) and job.popen is None
        return {"job_id": job.job_id, "kind": ui.get("kind"), "producer": doc.get("producer"),
                "label": ui.get("label") or d.name, "state": state, "position": self.position(job, state),
                "gpu": bool(ui["gpu"]) if "gpu" in ui else _job_gpu(doc),
                "created": ui.get("created") or _stamp(d / "job.json"), "started": started,
                "finished": _stamp(d / "status.json") if state in FINAL and status is not None else None,
                "run_dir": self.fs.rid(run_dir) if run_dir else None, "status": status,
                "pid": (pid or {}).get("pid") or (status or {}).get("pid"), "exit_code": exit_code, "error": error,
                "liveness": liveness, "attached": job.attached, "dir": self.fs.rid(d),
                "warnings": ui.get("warnings") or [],
                "result": self._result_rid(ui)}

    def _result_rid(self, ui: dict) -> str | None:
        """The rid of a ``bench`` job's result JSON once the file exists (``ui.json`` names its path from creation)."""
        path = ui.get("result")
        return self.fs.rid(path) if isinstance(path, str) and os.path.isfile(path) else None

    def _queued(self, job: Job, state: str) -> bool:
        return state in ("queued", "held") and self._ours(job.ui)

    def position(self, job: Job, state: str) -> int | None:
        """1-based place among the queued and held jobs of the job's lane, in ``queue_seq`` order."""
        if not self._queued(job, state):
            return None
        lane, seq = bool(job.ui.get("gpu")), _seq(job.ui)
        return 1 + sum(1 for j in self.jobs.values() if j is not job and self._ours(j.ui) and not j.final
                       and bool(j.ui.get("gpu")) == lane and _seq(j.ui) < seq
                       and not j.ui.get("spawn") and not j.ui.get("cancelled")
                       and not (j.dir / "status.json").exists() and not (j.dir / "bridge" / "pid.json").exists())

    def records(self) -> list[dict]:
        self.scan()
        out = []
        for j in list(self.jobs.values()):
            try:
                out.append(self.record(j))
            except Exception as e:  # noqa: BLE001  one malformed job directory never hides the others
                self._skip(j, e)
        return sorted(out, key=lambda r: (str(r["created"] or ""), r["job_id"]), reverse=True)

    def owned_runs(self) -> set[str]:
        """Real paths of the run directories that this jobs root's non-final run and evimo2 jobs write."""
        out = set()
        for j in list(self.jobs.values()):          # also read off the loop (listings, job checks)
            if j.state not in FINAL + ("interrupted",) and j.ui and isinstance(j.ui.get("run_dir"), str):
                out.add(os.path.realpath(j.ui["run_dir"]))
        return out

    def _foreign_gpu(self) -> bool:
        """Without a GPU lock (no cross-producer exclusion): another producer's listed job is alive (also stalled) in
        a GPU step (its run or dense step), so the GPU lane waits for it (§5.3)."""
        def gpu_step(j: Job) -> bool:
            status = read_json(j.dir / "status.json") or {}
            return status.get("state") == "running" and status.get("step") in ("run", "dense")

        return self.gpu_lock is None and any(not self._ours(j.ui) and j.state in ("running", "stalled") and gpu_step(j)
                                             for j in list(self.jobs.values()))

    def gpu_busy(self) -> bool:
        """A UI GPU job is non-final (spawned), or :meth:`_foreign_gpu`."""
        return self._foreign_gpu() or any(
            self._ours(j.ui) and j.ui.get("gpu") and not j.final and (j.ui.get("spawn") or j.popen is not None)
            for j in list(self.jobs.values()))

    # -- the loop's tick ---------------------------------------------------------------------------------------

    def tick(self) -> list[dict]:
        """Reaps, refreshes and dispatches; returns the records that changed since the last tick (heartbeats and
        elapsed time aside)."""
        self.scan()
        changed, states = [], {}
        for j in list(self.jobs.values()):
            if j.final and j.published_final and j.published is not None:   # a final state an HTTP read saw first
                continue                                                      # is still published once
            if not j.dir.is_dir():
                del self.jobs[j.job_id]
                continue
            try:
                rec = self.record(j)
                states[j.job_id] = rec["state"]
                key = json.dumps({**rec, "status": {k: v for k, v in (rec["status"] or {}).items()
                                                    if k not in ("heartbeat", "elapsed_s")}}, sort_keys=True)
                if key != j.published:
                    j.published = key
                    changed.append(rec)
                j.published_final = j.final
                if rec["state"] == "complete" and j.ui and j.ui.get("bundle_source") and not j.bundle_recorded:
                    self._record_bundle(j)
            except Exception as e:  # noqa: BLE001  one malformed job directory never stops the tick
                self._skip(j, e)
        if self.dispatcher:
            changed += self._dispatch(states)
        return changed

    def _lane_busy(self, gpu: bool, states: dict) -> bool:
        if gpu and self._foreign_gpu():
            return True
        for j in self.jobs.values():
            if self._ours(j.ui) and bool(j.ui.get("gpu")) == gpu and not j.final:
                st = states.get(j.job_id)
                if st is None:
                    try:
                        st = self.record(j)["state"]
                    except Exception as e:  # noqa: BLE001
                        self._skip(j, e)
                        continue
                if st not in _IDLE:
                    return True
        return False

    def _dispatch(self, states: dict) -> list[dict]:
        out = []
        for gpu in (True, False):
            if self._lane_busy(gpu, states):
                continue
            queued = [j for j in self.jobs.values() if self._ours(j.ui) and bool(j.ui.get("gpu")) == gpu
                      and states.get(j.job_id) == "queued"]
            if not queued:
                continue
            job = min(queued, key=lambda j: _seq(j.ui))
            self._refresh_ui(job)                   # a cancel written since the record: never spawned
            if job.ui.get("cancelled"):
                job.published = None
                continue
            if gpu:
                self.before_gpu_spawn()
            try:
                self.spawn(job)
            except OSError as e:
                job.ui["spawn_error"] = str(e)
                write_json_atomic(job.dir / "ui.json", job.ui)
            job.published = None
            try:
                rec = self.record(job)
            except Exception as e:  # noqa: BLE001
                self._skip(job, e)
                continue
            states[job.job_id] = rec["state"]
            out.append(rec)
        return out

    def spawn(self, job: Job) -> None:
        """§3.7 spawning: the marker, the detached runner (``Popen`` kept), then ``pid.json``."""
        job.ui["spawn"] = {"at": _now()}
        write_json_atomic(job.dir / "ui.json", job.ui)
        bridge = job.dir / "bridge"
        bridge.mkdir(exist_ok=True)
        argv = [*self.runner, str(job.dir), *(["--allow-exec"] if job.ui.get("allow_exec") else [])]
        with open(bridge / "job_stdout.log", "ab") as out, open(bridge / "job_stderr.log", "ab") as err:
            job.popen = subprocess.Popen(argv, cwd=job.dir, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                         start_new_session=True, close_fds=True)
        st = proc_stat(job.popen.pid)
        write_json_atomic(bridge / "pid.json", {"pid": job.popen.pid, "argv": argv,
                                                "start_time": st[1] if st else None, "boot_id": boot_id()})

    def _dispatcher_only(self, what: str) -> None:
        if not self.dispatcher:
            raise _refuse(f"{what}: this server does not hold the jobs root's dispatch lock: the server that does "
                          "dispatches its queue", "BS-UI-DISPATCH", 409)

    def resume(self) -> int:
        """``/jobs/queue/resume``: the held jobs become queued; returns how many. The dispatcher's only."""
        self._dispatcher_only("resuming the queue")
        n = 0
        for j in self.jobs.values():
            if j.held:
                j.held, n = False, n + 1
        return n

    # -- cancel and kill ---------------------------------------------------------------------------------------

    def _runner_pid(self, job: Job) -> int | None:
        """The live runner's pid: this server's child; else, while ``runner.lock`` is held (E3: the runner lives, also
        without /proc), ``pid.json``'s pid, else status.json's (a server that died before writing pid.json); else, for
        a runner without the lock (before E3), ``pid.json``'s while :func:`runner_alive`. None when there is none."""
        if job.popen is not None and job.popen.poll() is None:
            return job.popen.pid
        info = read_json(job.dir / "bridge" / "pid.json")
        if lock_held(job.dir / "bridge" / "runner.lock"):
            for doc in (info, read_json(job.dir / "status.json")):
                pid = (doc or {}).get("pid")
                if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1 and pid != os.getpid():
                    return pid
            return None
        return info["pid"] if runner_alive(info) and info["pid"] > 1 and info["pid"] != os.getpid() else None

    def _signal(self, job: Job, signum: int) -> bool:
        """``signum`` to the runner's process group (:meth:`_runner_pid`); False when there is none."""
        pid = self._runner_pid(job)
        if pid is None:
            return False
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signum)
            else:
                os.kill(pid, signum)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def cancel(self, job_id: str, loop=None) -> dict:
        """``POST /jobs/{id}/cancel``: a queued or held job is cancelled in ui.json; a running one gets ``cancel``,
        SIGINT to its group and SIGKILL after the grace period."""
        job = self.get(job_id)
        state = self.record(job)["state"]
        if state in ("queued", "held"):
            self._dispatcher_only("cancelling a queued job")
            job.ui["cancelled"] = True
            write_json_atomic(job.dir / "ui.json", job.ui)
            job.published = None
            return {"state": "cancelled"}
        if state in FINAL + ("interrupted",):
            return {"state": state}
        with contextlib.suppress(OSError):
            (job.dir / "cancel").write_text("")
        if (job.dir / "status.json").exists():     # before its first status write the runner has no SIGINT handler
            self._signal(job, signal.SIGINT)        # (KeyboardInterrupt: interrupted); it reads cancel first
        if loop is not None and job.kill_timer is None:
            job.kill_timer = loop.call_later(self.grace_s, self._kill_late, job)
        return {"state": "cancelling"}

    def _kill_late(self, job: Job) -> None:
        job.kill_timer = None
        if self.record(job)["state"] not in FINAL + ("interrupted",):
            self._signal(job, signal.SIGKILL)

    def kill(self, job_id: str) -> dict:
        job = self.get(job_id)
        if not self._signal(job, signal.SIGKILL):
            raise _refuse("the job has no live runner", "BS-UI-REQUEST", 409)
        return {"state": "killing"}

    @_path_errors()
    def log(self, job_id: str, stream: str, since: int) -> tuple[bytes, int]:
        job = self.get(job_id)
        path = job.dir / "bridge" / f"job_{stream}.log"
        try:
            with open(path, "rb") as f:
                f.seek(since)
                data = f.read(MAX_LOG_BYTES)
        except FileNotFoundError:
            return b"", since
        return data, since + len(data)

    def close(self) -> None:
        for j in self.jobs.values():
            if j.kill_timer is not None:
                j.kill_timer.cancel()

    # -- bundles.json --------------------------------------------------------------------------------------------

    def _record_bundle(self, job: Job) -> None:
        """§4.12: a completed ``bundle`` job of a scene YAML records the YAML's (realpath, size, mtime_ns) in
        ``<jobs_root>/bundles.json``."""
        job.bundle_recorded = True
        path = os.path.realpath(job.ui["bundle_source"])
        try:
            st = os.stat(path)
        except OSError:
            return
        doc = read_json(self.root / "bundles.json") or {}
        entries = [e for e in doc.get("entries", []) if isinstance(e, dict) and e.get("path") != path]
        entries.append({"path": path, "size": st.st_size, "mtime_ns": str(st.st_mtime_ns)})
        write_json_atomic(self.root / "bundles.json", {"format": "ballasim.ui.bundles/1", "entries": entries})

    # -- creation ----------------------------------------------------------------------------------------------

    def prepare(self, body: dict) -> tuple:
        """``POST /jobs`` ``{kind, args, label, overwrite}``, the part that reads files (off the loop): checks and
        builds the job (422 BS-UI-JOB-KIND for an unknown kind or bad arguments, 409 BS-UI-RUN-EXISTS / BS-UI-RUN-LIVE
        for the output) in a new, empty job directory. Only the dispatcher creates jobs (409 BS-UI-DISPATCH)."""
        if not self.dispatcher:
            raise _refuse("this server does not hold the jobs root's dispatch lock: another server dispatches its "
                          "jobs", "BS-UI-DISPATCH", 409)
        kind, args = body.get("kind"), body.get("args")
        if kind not in KINDS:
            raise _refuse(f"kind: one of {', '.join(KINDS)}")
        if not isinstance(args, dict):
            raise _refuse("args: an object")
        label = body.get("label")
        if label is not None and not isinstance(label, str):
            raise _refuse("label: a string")
        overwrite = body.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise _refuse("overwrite: true or false")
        job_dir = self._new_dir(label or kind)
        try:
            return job_dir, kind, args, label or kind, getattr(self, f"_build_{kind}")(args, overwrite, job_dir)
        except BaseException:
            with contextlib.suppress(OSError):
                job_dir.rmdir()
            raise

    def commit(self, prepared: tuple) -> tuple[Job, dict]:
        """The rest of ``POST /jobs``, on the loop: ``queue_seq``, run.yaml, ui.json, then job.json, atomically."""
        job_dir, kind, args, label, plan = prepared
        try:
            self._seq += 1
            ui = {"kind": kind, "args": args, "label": label, "created": _now(), "jobs_root": str(self.root),
                  "queue_seq": self._seq, "gpu": plan.gpu, "allow_exec": True, **plan.extra}
            (job_dir / "bridge").mkdir()
            if plan.run_yaml is not None:
                (job_dir / "run.yaml").write_text(json.dumps(plan.run_yaml, indent=1) + "\n")
            write_json_atomic(job_dir / "ui.json", ui)
            write_json_atomic(job_dir / "job.json", {
                "format": "ballasim.job", "version": [1, 0], "producer": f"ballasim-ui {ballasim.__version__}",
                "config": "run.yaml" if plan.run_yaml is not None and any(s["step"] in ("validate", "run")
                                                                          for s in plan.steps) else None,
                "sets": plan.sets, "steps": plan.steps, "timeline": None, "preview": None, "gpu_lock": self.gpu_lock,
                "nice": plan.nice})
        except BaseException:
            for name in ("job.json", "ui.json", "run.yaml"):
                with contextlib.suppress(OSError):
                    (job_dir / name).unlink()
            with contextlib.suppress(OSError):
                (job_dir / "bridge").rmdir()
            with contextlib.suppress(OSError):
                job_dir.rmdir()
            raise
        job = Job(job_dir, job_id_of(job_dir), False, ui)
        self.jobs[job.job_id] = job
        return job, self.record(job)

    def _new_dir(self, label: str) -> Path:
        base = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(label)}"
        for n in range(1, 1000):
            d = self.root / (base if n == 1 else f"{base}-{n}")
            try:
                d.mkdir()
                return Path(os.path.realpath(d))
            except FileExistsError:
                continue
        raise _refuse("too many jobs with this name this second", "BS-UI-REQUEST", 409)

    def _out(self, args: dict, overwrite: bool, job_dir: Path) -> Path:
        """A run job's run directory: ``out`` (a writable path), default ``<first writable root>/runs/<job name>``;
        409 BS-UI-RUN-EXISTS when it holds anything and ``overwrite`` is false, BS-UI-RUN-LIVE while a run writes it."""
        out = args.get("out")
        path = self.fs.target(out, "out") if out is not None else self.fs.default_output_root() / "runs" / job_dir.name
        try:
            occupied = path.exists() and (not path.is_dir() or any(path.iterdir()))
        except OSError as e:
            raise _refuse(f"out: {e.strerror}", "BS-UI-PATH", 403) from None
        if occupied and not overwrite:
            raise _refuse(f"{path} exists; confirm overwrite to replace it", "BS-UI-RUN-EXISTS", 409)
        if self.fs.liveness(path, None) == "live":
            raise _refuse(f"{path} is being written by a run", "BS-UI-RUN-LIVE", 409)
        return path

    def _wants_gpu(self, items: list[str], config: dict) -> bool:
        """The lane of a run: ``run.device`` from the items (the last wins) or the configuration; ``auto`` (the
        default) is the GPU when the device probe found one (or has not answered)."""
        device = None
        for item in items:
            key, _, text = item.partition("=")
            if key.strip() == "run.device":
                device = text.strip().strip("\"'")
        if device is None:
            run = config.get("run") if isinstance(config.get("run"), dict) else {}
            device = config.get("run.device", run.get("device"))
        if device == "cpu":
            return False
        if device == "gpu":
            return True
        return self.device() is not False

    def _source_config(self, args: dict) -> tuple[dict, Path | None, list[str], list[str], list[str], str | None]:
        """The base configuration of a ``run`` or ``bundle`` job: ``config_path`` (a rid or an absolute path),
        ``config`` with ``base_dir``, or (``run`` only) ``run`` (§3.6's rebuild). Returns (config with every path key
        absolute, base_dir, recorded items, recorded profiles, warnings, the source's scene YAML or None)."""
        from ruamel.yaml import YAML, YAMLError

        from ballasim import config as cfg

        given = [k for k in ("config_path", "config", "run") if args.get(k) is not None]
        if len(given) != 1:
            raise _refuse("args: give exactly one of config_path, config (with base_dir) or run")
        items, profiles, warnings = [], [], []
        if given[0] == "config_path":
            ref = args["config_path"]
            path = self.fs.resolve(ref)                 # inside the roots (realpath containment)
            try:
                raw = YAML(typ="safe").load(path.read_text())
            except (OSError, YAMLError, UnicodeDecodeError) as e:
                raise _refuse(f"config_path: {e}") from None
            # relative paths start from the parent of the path as given, as `ballasim run` takes it (config.py:
            # path.parent.absolute()), not from the real path's: a symlinked configuration names the same scene
            if ref.startswith("/"):
                base = Path(ref).parent
            else:
                root_id, rel = decode_rid(ref)
                base = (Path(self.fs.root(root_id).path) / rel).absolute().parent
        elif given[0] == "config":
            raw, base = args["config"], args.get("base_dir")
            if not isinstance(base, str) or not os.path.isabs(base):
                raise _refuse("base_dir: the absolute directory the configuration's relative paths start from")
            base = Path(base)
        else:
            _, doc = self.fs.run_dir(args["run"])
            raw, base, items, profiles, warnings = rebuild_spec(doc)
        if not isinstance(raw, dict):
            raise _refuse("the configuration is not a mapping")
        try:
            raw = cfg._expand(raw, "", base or Path("/"))
        except ValueError as e:
            raise _refuse(str(e)) from None
        scene = raw.get("scene") if isinstance(raw.get("scene"), dict) else {}
        yaml_source = scene.get("path") or raw.get("scene.path")
        if not (isinstance(yaml_source, str) and yaml_source.lower().endswith((".yaml", ".yml"))):
            yaml_source = None
        return raw, base, items, profiles, warnings, yaml_source

    def _check_exchange(self, raw: dict, sets: list[str], job_dir: Path) -> None:
        """§4.12: a job whose scene is a Blender exchange this server may not write (an attached ``ext-*`` root, a
        read-only root, outside the roots) is refused (422 BS-UI-EXT-READONLY) when the exchange holds no converted
        bundle for the job's values (``cache/<blender_cache_key>/``): loading it would write that entry there. The
        values are the runner's (run.yaml with job.json's ``sets``, path-typed items against the job directory, the
        runner's working directory); when they cannot be computed, the runner fails the same way before any write."""
        from ballasim import config as cfg

        try:
            cli = {k: cfg.expand_path(os.fspath(v), job_dir, k) if k in cfg.path_keys() and isinstance(v, str) else v
                   for k, v in cfg.parse_sets(sets).items()}
            values = cfg.load(raw, sets=cli, base_dir=job_dir).scene.values()["run"]
        except Exception:  # noqa: BLE001  the job's validate step reports it
            return
        path = values.get("scene.path")
        if values.get("scene.evimo2.sequence") is not None or not isinstance(path, str) \
                or not os.path.isfile(os.path.join(path, "exchange.json")) or self.fs.writable(path):
            return
        try:
            from ballasim.blender.convert import blender_cache_key, read_exchange

            ex = read_exchange(path)
            missing = not (ex.root / "cache" / blender_cache_key(values, ex) / "bundle.json").is_file()
        except Exception:  # noqa: BLE001  load_blender fails the same way, before writing
            return
        if missing:
            raise _refuse(f"{path}: these values need a new bundle inside a Blender exchange this server may not "
                          "write: change scene.blender.* in Blender, or copy the job into a writable root",
                          "BS-UI-EXT-READONLY")

    def _build_run(self, args: dict, overwrite: bool, job_dir: Path) -> Plan:
        raw, _, items, profiles, warnings, _ = self._source_config(args)
        out = self._out(args, overwrite, job_dir)
        raw["profiles"] = list(raw.get("profiles") or []) + profiles + _profiles(args.get("profiles"))
        output = raw.get("output") if isinstance(raw.get("output"), dict) else {}
        for key in OUTPUT_KEYS:                     # a flat key would shadow the job's own
            raw.pop(key, None)
        raw["output"] = {**output, "dir": str(out), "overwrite": overwrite}
        sets = items + set_items(args.get("sets"))
        self._check_exchange(raw, sets, job_dir)
        return Plan([{"step": "validate"}, {"step": "run"}], self._wants_gpu(sets, raw), 0, raw, sets,
                    {"run_dir": str(out), "warnings": warnings})

    def _build_evimo2(self, args: dict, overwrite: bool, job_dir: Path) -> Plan:
        doc = self._evimo2_doc(args, str(self._out(args, overwrite, job_dir)))
        doc["profiles"] = list(doc.get("profiles") or []) + _profiles(args.get("profiles"))
        doc["output"]["overwrite"] = overwrite
        sets = set_items(args.get("sets"))
        return Plan([{"step": "validate"}, {"step": "run"}], self._wants_gpu(sets, doc), 0, doc, sets,
                    {"run_dir": doc["output"]["dir"]})

    def _evimo2_doc(self, args: dict, out: str) -> dict:
        from ballasim.config import ConfigError
        from ballasim.engine.cli import evimo2_doc

        sequence = self.fs.resolve(args.get("sequence"))
        if not sequence.is_dir():
            raise _refuse("sequence: an EVIMO2 sequence directory")
        camera = args.get("camera")
        if camera is not None and not isinstance(camera, str):
            raise _refuse("camera: a camera folder name")
        profile_set = args.get("profile_set")
        if profile_set is not None and not isinstance(profile_set, str):
            raise _refuse("profile_set: a set name")
        t0 = None if args.get("t0_ns") is None else _int(args["t0_ns"], "t0_ns")
        t1 = None if args.get("t1_ns") is None else _int(args["t1_ns"], "t1_ns")
        try:
            return evimo2_doc(str(sequence), camera=camera, t0_ns=t0, t1_ns=t1, out=out, profile_set=profile_set,
                              no_profiles=bool(args.get("no_profiles")))
        except ConfigError as e:
            raise _refuse(str(e)) from None

    def _build_bundle(self, args: dict, overwrite: bool, job_dir: Path) -> Plan:
        recorded: list[str] = []
        if args.get("sequence") is not None:
            raw, yaml_source = self._evimo2_doc(args, str(job_dir / "run")), None
            raw["profiles"] = list(raw.get("profiles") or []) + _profiles(args.get("profiles"))
        else:
            # a `run` base is the run's rebuilt specification (§3.6), so that the bundle the setup screen asks for after
            # a 202 of `/resolve {run}` is the one that resolve looks up
            raw, _, recorded, profiles, _, yaml_source = self._source_config(args)
            raw["profiles"] = list(raw.get("profiles") or []) + profiles + _profiles(args.get("profiles"))
            raw["output"] = {"dir": str(job_dir / "run")}       # a placeholder: scene build writes no run
        items = recorded + set_items(args.get("sets"))
        if args.get("sequence") is None:
            self._check_exchange(raw, items, job_dir)
        argv = [*self.python, "-m", "ballasim", "scene", "build", str(job_dir / "run.yaml")]
        for item in items:
            argv += ["--set", item]
        return Plan([{"step": "exec", "name": "scene build", "argv": argv, "cwd": None, "gpu": False}], False, 10,
                    raw, [], {"bundle_source": yaml_source} if yaml_source else {})

    def _build_dense(self, args: dict, overwrite: bool, job_dir: Path) -> Plan:
        from ballasim.engine.dense import CHANNELS, DEFAULT_CHANNELS

        run_dir, doc = self.fs.run_dir(args.get("run"))
        refusal = kind_refusal(doc)
        if refusal is not None and refusal[0] != "BS-UI-RUN-RELPATH":   # dense reads the resolved, absolute values
            raise _refuse(refusal[1], refusal[0])
        if self.fs.liveness(run_dir, doc.get("status")) == "live":
            raise _refuse(f"{run_dir} is being written by a run", "BS-UI-RUN-LIVE", 409)
        dt = _decimal(args.get("dt_ms", 20), "dt_ms")
        if dt <= 0:
            raise _refuse("dt_ms: positive")
        channels = args.get("channels") or list(DEFAULT_CHANNELS)
        if not isinstance(channels, list) or not channels or not set(channels) <= set(CHANNELS):
            raise _refuse(f"channels: a non-empty list of {', '.join(CHANNELS)}")
        # a run this server may not write: <default output root>/dense/<run name>-<first 8 hex of SHA-256(realpath)>,
        # unique per run (every add-on job's run directory is named "run")
        out = run_dir / "dense" if self.fs.writable(run_dir) else self.fs.default_output_root() / "dense" / \
            f"{run_dir.name}-{hashlib.sha256(os.fsencode(os.path.realpath(run_dir))).hexdigest()[:8]}"
        argv = [*self.python, "-m", "ballasim", "dense", str(run_dir), "--dt-ms", _text(dt)]
        for key, flag in (("t0_ns", "--t0"), ("t1_ns", "--t1")):
            if args.get(key) is not None:
                argv += [flag, ns_to_s(_int(args[key], key))]
        argv += ["--channels", ",".join(channels)]
        camera = args.get("camera")
        if camera is not None:
            if not isinstance(camera, str) or not _CAMERA.fullmatch(camera):
                raise _refuse("camera: a camera name")
            argv += ["--camera", camera]
        argv += ["--out", str(out)]
        return Plan([{"step": "exec", "name": "dense", "argv": argv, "cwd": None, "gpu": True}], True, 0, None, [],
                    {"source_run": str(run_dir), "out": str(out)})

    def _build_bench(self, args: dict, overwrite: bool, job_dir: Path) -> Plan:
        """``bench`` (§3.7): ``evb.py compare SPAN RUN_DIR`` of the bench in ``--repo``, on the CPU lane. The result is
        ``<job>/bench/ui-<job_id>/<span tag>.json``, shown unchanged. The span and the run are checked here (the job
        would fail on them): 422 BS-UI-SPAN, BS-UI-GT-BAD (unless ``allow_bad``), BS-UI-SOURCE (no EVIMO2 sequence at
        the span's directory; a run whose first camera's events are not ``<run>/events/<camera>.h5`` or are missing)."""
        if self.bench is None:
            raise _refuse("bench jobs need a source checkout (start the server with --repo)",
                          "BS-UI-BENCH-UNAVAILABLE", 501)
        run_dir, doc = self.fs.run_dir(args.get("run"))
        if run_format(doc) == 1:
            raise _refuse(kind_refusal(doc)[1], "BS-UI-RUN-FORMAT1")      # the bench needs the resolved clock offset
        if self.fs.liveness(run_dir, doc.get("status")) == "live":
            raise _refuse(f"{run_dir} is being written by a run", "BS-UI-RUN-LIVE", 409)
        span = args.get("span")
        if not isinstance(span, str) or not span or span.startswith("-") or not span.isprintable():
            raise _refuse("span: an evb span id (a key of spans.yaml, or <camera>/<category>/<split>/<sequence>@<t0_s>-"
                          "<t1_s>)")
        allow_bad = args.get("allow_bad", False)
        if not isinstance(allow_bad, bool):
            raise _refuse("allow_bad: true or false")
        if self.repo in job_dir.parents:        # evb.py writes its output outside the source checkout
            raise _refuse(f"the jobs root is inside the source checkout {self.repo}: the bench writes its output "
                          "outside it (give --jobs-root elsewhere)")
        self.bench.span(span, None, allow_bad)
        self.bench.run_info(run_dir)
        label = f"ui-{job_id_of(job_dir)}"
        argv = [*self.python, str(self.bench.script), "--out", str(job_dir / "bench"), "compare", span,
                str(run_dir), "--label", label]
        if allow_bad:
            argv.append("--allow-bad")
        result = self.bench.result_path(job_dir, label, span)
        return Plan([{"step": "exec", "name": "evb compare", "argv": argv, "cwd": None, "gpu": False}], False, 10,
                    None, [], {"source_run": str(run_dir), "result": str(result), "span": span})

    def _build_export_video(self, args: dict, overwrite: bool, job_dir: Path) -> Plan:
        from ballasim.viz.compare import KINDS as VIZ_KINDS
        from ballasim.viz.compare import THIRD_PANELS

        streams = args.get("streams")
        if not isinstance(streams, list) or not 1 <= len(streams) <= 2 or not all(isinstance(s, dict)
                                                                                    for s in streams):
            raise _refuse("streams: one or two {source, offset_ns} (viz compare takes at most two)")
        argv = [*self.python, "-m", "ballasim.viz", "compare"]
        for role, s in zip(("real", "sim"), streams):
            path, _ = locate_source(self.fs.roots, s.get("source"))
            argv += [f"--{role}", str(path), f"--{role}-offset-ms", ns_to_ms(_int(s.get("offset_ns", 0), "offset_ns"))]
        for key, flag in (("t0_ns", "--t0"), ("t1_ns", "--t1")):
            if args.get(key) is not None:
                argv += [flag, ns_to_s(_int(args[key], key))]
        kind = args.get("kind", "events")
        if kind not in VIZ_KINDS:
            raise _refuse(f"kind: one of {', '.join(VIZ_KINDS)}")
        panels = args.get("extra_panels") or "none"
        if isinstance(panels, list):
            panels = ",".join(panels) or "none"
        names = panels.split(",") if isinstance(panels, str) else [None]
        if panels != "none" and (not all(p in THIRD_PANELS[1:] for p in names) or len(set(names)) != len(names)):
            raise _refuse(f"extra_panels: none, or distinct values of {', '.join(THIRD_PANELS[1:])}")
        out = args.get("out")
        path = self.fs.target(out, "out") if out is not None else \
            self.fs.default_output_root() / "videos" / f"{job_dir.name}.mp4"
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise _refuse(f"out: a {', '.join(VIDEO_SUFFIXES)} file")
        if path.exists() and not overwrite:
            raise _refuse(f"{path} exists; confirm overwrite to replace it", "BS-UI-RUN-EXISTS", 409)
        argv += ["--kind", kind, "--third", panels, "--out", str(path)]
        return Plan([{"step": "exec", "name": "export video", "argv": argv, "cwd": None, "gpu": False}], False, 10,
                    None, [], {"out": str(path)})


def _job_gpu(doc: dict) -> bool:
    return any(st.get("step") in ("run", "dense") or (st.get("step") == "exec" and st.get("gpu")) for st in _steps(doc))

