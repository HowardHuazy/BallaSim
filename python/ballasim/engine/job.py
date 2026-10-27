"""The job runner of ``ballasim.job/1`` (docs/design/phase3/blender.md §3.3, §5.1; ui.md §3.7 E1-E3):
``python -m ballasim job JOBDIR [--allow-exec]`` runs the ordered steps of ``JOBDIR/job.json`` and reports them in
``JOBDIR/status.json``.

- **Threads.** The main thread supervises: its SIGINT and SIGTERM handlers (installed also when SIGINT was inherited
  as ignored) only note the signal; its loop turns a signal or the ``cancel`` file into the cancel (:class:`Cancel`,
  which also sets the engine's C++ ``CancelToken`` while the run step holds one), writes ``status.json`` every second
  (the heartbeat) and waits. A worker thread runs the steps.
- **Steps.** ``validate`` resolves the configuration in this process (exit 2 on a ConfigError). ``run`` runs the
  engine in the worker thread (BL-0b: ``engine.run(on_window=, cancel=)``; each window updates status.json, and a
  cancel stops the C++ engine within a pixel batch or refinement round, run.json then reading ``cancelled``).
  ``dense`` runs the engine's CLI in a child process, ``preview`` the viewer's ``frames`` command, ``exec`` an external
  program.
- **Children.** Each child runs in its own process group, so a client's SIGINT to the runner's group reaches only the
  runner, and a cancel sends SIGINT once to the child's group; an engine child (``dense``, ``preview``) gets it again
  every second until it exits, since one can be lost before :func:`child_main`'s handler is installed. Engine children
  run through :func:`child_main`, which turns the first SIGINT or SIGTERM into KeyboardInterrupt (raised again when
  Python swallows it) and ignores later ones, so the step's cleanup completes; they die with the runner
  (PR_SET_PDEATHSIG). ``exec`` programs run under the supervisor :mod:`.job_exec`, which kills the program's whole
  group when the runner dies.
- **Locks.** ``bridge/runner.lock`` keeps a second runner off the job (E3). The GPU lock (job.json ``gpu_lock``, relative
  to the job directory, else ``$BALLASIM_GPU_LOCK``, an absolute path) is taken around each GPU step (``run``,
  ``dense``, an ``exec`` with ``gpu``). While it is busy the state is ``waiting``.
- **Exit codes:** 0 complete, 1 failed, 2 configuration or file error, 3 cancelled.
"""

from __future__ import annotations

import _thread
import contextlib
import fcntl
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

log = logging.getLogger(__name__)

FORMAT, VERSION = "ballasim.job", (1, 0)
PROTOCOL = f"{FORMAT}/{VERSION[0]}"
STATUS_FORMAT = "ballasim.job-status"
EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_CANCELLED = 0, 1, 2, 3
FINAL_STATES = ("complete", "failed", "cancelled")
#: Built-in steps of the frames area (frames.md §3.3), added to this runner at FR-5.
FRAMES_STEPS = ("frames_plan", "frames_ingest", "frames_reference")
STEPS = ("validate", "run", "preview", "dense", "exec")
#: Modules a child started through child_main may run, by their ``python -m`` name.
CHILD_MAINS = {"ballasim": "ballasim.engine.cli", "ballasim.viz": "ballasim.viz.cli"}
#: The windows a preview frame offers (blender.md §3.3).
PREVIEW_WINDOWS = ("prev", "exposure")
#: The exposure window of frame f, [f + a, f + b) in frames, by motion_blur.position, for the shutter σ (§4.2).
_EXPOSURE = {"CENTER": lambda s: (-s / 2, s / 2), "START": lambda s: (0.0, s), "END": lambda s: (-s, 0.0)}
_EXEC_SUPERVISOR = Path(__file__).resolve().with_name("job_exec.py")
_LOCK_RETRY_S, _LOCK_TRIES = 0.1, 10
_POLL_S = 0.1
#: An engine child's SIGINT again, after a cancel, until it exits.
_INTERRUPT_AGAIN_S = 1.0


class JobError(Exception):
    """A step or job.json failure: ``kind`` is status.json's ``error.type``."""

    def __init__(self, kind: str, message: str, exit_code: int = EXIT_FAILED):
        super().__init__(message)
        self.kind, self.exit_code = kind, exit_code


class _Cancelled(Exception):
    pass


class Cancel(threading.Event):
    """The job's cancel: an Event for the runner's own waits, mirrored into the engine's C++ ``CancelToken`` of the run
    step (:meth:`token`) once set. :meth:`set` runs on the supervisor's loop, never in a signal handler."""

    def __init__(self):
        super().__init__()
        self._guard = threading.Lock()
        self._tokens: list = []

    def set(self) -> None:
        with self._guard:
            super().set()
            for token in self._tokens:
                token.set()

    @contextlib.contextmanager
    def token(self):
        """A ``ballasim._engine.CancelToken`` set with this cancel (also when it was set before), for one engine run."""
        from ballasim import _engine

        token = _engine.CancelToken()
        with self._guard:
            self._tokens.append(token)
            if self.is_set():
                token.set()
        try:
            yield token
        finally:
            with self._guard:
                self._tokens.remove(token)


def write_json_atomic(path: Path, doc: dict) -> None:
    """:func:`.runlog.write_json_atomic`, imported on first use: runlog imports NumPy, which an engine child imports
    only after :func:`child_main` has installed its interrupt handler (an import is where Python swallows one)."""
    from .runlog import write_json_atomic as write

    write(path, doc)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def python_argv() -> list[str]:
    """This interpreter for a child: -I, or this one's -E and -s plus always -P (children run ``-m`` in the job
    directory, which must never shadow a module), this one's -S, and unbuffered (-u)."""
    f = sys.flags
    flags = ["-I"] if f.isolated else ["-E"] * f.ignore_environment + ["-s"] * f.no_user_site + ["-P"]
    return [sys.executable, *flags, *["-S"] * f.no_site, "-u"]


def exec_allow_list() -> set[str]:
    """The executables ``exec`` steps may run without --allow-exec: ``$XDG_CONFIG_HOME/ballasim/exec_allow.json``
    (``~/.config`` when unset or relative, as the XDG spec says), ``{"executables": [path, ...]}``, compared by real
    path."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    root = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".config"
    try:
        doc = json.loads((root / "ballasim" / "exec_allow.json").read_text())
        return {os.path.realpath(p) for p in doc.get("executables", []) if isinstance(p, str)}
    except (OSError, ValueError, AttributeError):
        return set()


def _executable(argv0: str, cwd: Path) -> str | None:
    path = shutil.which(argv0) if os.sep not in argv0 else str(cwd / argv0)
    return os.path.realpath(path) if path and os.access(path, os.X_OK) else None


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _config_error(message: str) -> JobError:
    return JobError("JobError", message, EXIT_CONFIG)


def _preview_frames(frames, timeline: dict) -> tuple[int, int, int]:
    """(f0, f1, step) of a preview step's ``frames``: "span" (the timeline's frame_start to frame_end) or
    [f0, f1, step] with f0 <= f1 and step > 0 (f1 included)."""
    if frames == "span":
        f0, f1, step = timeline.get("frame_start"), timeline.get("frame_end"), 1
        if not (_is_int(f0) and _is_int(f1) and f0 <= f1):
            raise _config_error(f"timeline frame_start {f0!r} and frame_end {f1!r} must be frames, start <= end")
        return f0, f1, step
    if isinstance(frames, list) and len(frames) == 3 and all(map(_is_int, frames)) and frames[2] > 0 \
            and frames[0] <= frames[1]:
        return tuple(frames)
    raise _config_error(f"preview frames must be 'span' or [f0, f1, step] with f0 <= f1 and step > 0, "
                        f"got {frames!r}")


def _exposure_ends(timeline: dict) -> tuple[float, float]:
    """The exposure window of frame f as [f + a, f + b) (§4.2): motion_blur's position and shutter σ."""
    blur = timeline.get("motion_blur") or {}
    if not isinstance(blur, dict):
        raise _config_error(f"timeline motion_blur must be an object or null, got {blur!r}")
    sigma, position = blur.get("shutter", 0.5), blur.get("position", "CENTER")
    if not _is_number(sigma) or sigma < 0:
        raise _config_error(f"timeline motion_blur.shutter {sigma!r}")
    if position not in _EXPOSURE:
        raise _config_error(f"timeline motion_blur.position {position!r}")
    return _EXPOSURE[position](float(sigma))


def _preview_args(args: dict, timeline: dict) -> None:
    window = args.get("window", "prev")
    if window not in PREVIEW_WINDOWS:
        raise _config_error(f"preview window must be 'prev' or 'exposure', got {window!r}")
    if not isinstance(args.get("camera"), (str, type(None))):
        raise _config_error(f"preview camera must be a name or null, got {args['camera']!r}")
    _preview_frames(args.get("frames", "span"), timeline)
    _exposure_ends(timeline)


def _dense_args(args: dict) -> None:
    dt_ms = args.get("dt_ms", 20)
    if not _is_number(dt_ms) or dt_ms <= 0:
        raise _config_error(f"dense dt_ms must be a positive number, got {dt_ms!r}")
    for key in ("t0_s", "t1_s"):
        if args.get(key) is not None and not _is_number(args[key]):
            raise _config_error(f"dense {key} must be a number or null, got {args[key]!r}")
    channels = args.get("channels")
    if channels is not None and (not isinstance(channels, list) or not all(isinstance(c, str) for c in channels)):
        raise _config_error(f"dense channels must be a list of names or null, got {channels!r}")
    if not isinstance(args.get("camera"), (str, type(None))):
        raise _config_error(f"dense camera must be a name or null, got {args['camera']!r}")


def read_job(job_dir: Path, *, allow_exec: bool) -> dict:
    """job.json, checked as a whole before any step runs (JobError with exit 2)."""
    def bad(message: str) -> JobError:
        return JobError("JobError", f"{job_dir / 'job.json'}: {message}", EXIT_CONFIG)

    try:
        doc = json.loads((job_dir / "job.json").read_text())
    except FileNotFoundError:
        raise bad("missing (an add-on or the UI writes it last, when the job is complete)") from None
    except ValueError as e:
        raise bad(f"not JSON ({e})") from None
    version = doc.get("version") if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or doc.get("format") != FORMAT or not isinstance(version, list) or len(version) != 2:
        raise bad(f"not a {FORMAT} document")
    if version[0] != VERSION[0]:
        raise bad(f"version {version}: this runner reads {FORMAT} {VERSION[0]}.x")
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps or not all(isinstance(s, dict) for s in steps):
        raise bad("steps must be a non-empty list of objects")
    for s in steps:
        name = s.get("step")
        if name in FRAMES_STEPS:
            raise bad(f"step {name!r} is not in this build (frames area, FR-5)")
        if name not in STEPS:
            raise bad(f"unknown step {name!r} (known: {', '.join(STEPS)})")
    config = doc.get("config")
    if config is not None and not isinstance(config, str):
        raise bad("config must be a path or null")
    if config is None and any(s["step"] in ("validate", "run") for s in steps):
        raise bad("config is null, but a validate or run step reads it (null is for exec-only jobs)")
    sets = doc.get("sets") or []
    if not isinstance(sets, list) or not all(isinstance(i, str) and "=" in i for i in sets):
        raise bad("sets must be a list of KEY=VALUE items")
    nice = doc.get("nice")
    if nice is not None and (not _is_int(nice) or not 0 <= nice <= 19):
        raise bad(f"nice must be 0-19 or null, got {nice!r}")
    if not isinstance(doc.get("gpu_lock"), (str, type(None))):
        raise bad(f"gpu_lock must be a path or null, got {doc['gpu_lock']!r}")
    timeline = doc.get("timeline")
    for s in steps:
        args = s.get("args")
        if args is not None and not isinstance(args, dict):
            raise bad(f"step {s['step']!r}: args must be an object or null")
        try:
            if s["step"] == "preview":
                if not isinstance(timeline, dict):
                    raise JobError("JobError", "the preview step needs job.json's timeline (the session's "
                                   "done.summary)", EXIT_CONFIG)
                _preview_args(args or {}, timeline)
            elif s["step"] == "dense":
                _dense_args(args or {})
        except JobError as e:
            raise bad(str(e)) from None
    allowed = None
    for s in steps:
        if s["step"] != "exec":
            continue
        for key in ("name", "cwd", "progress"):
            if not isinstance(s.get(key), (str, type(None))):
                raise bad(f"exec step {s.get('name')!r}: {key} must be a string or null, got {s[key]!r}")
        if not isinstance(s.get("gpu"), (bool, type(None))):
            raise bad(f"exec step {s.get('name')!r}: gpu must be true, false or null")
        argv, cwd = s.get("argv"), s.get("cwd") or "."
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise bad(f"exec step {s.get('name')!r}: argv must be a non-empty list of strings")
        exe = _executable(argv[0], job_dir / cwd)
        if exe is None:
            raise bad(f"exec step {s.get('name')!r}: {argv[0]} is not an executable")
        if not allow_exec:
            allowed = exec_allow_list() if allowed is None else allowed
            if exe not in allowed:
                raise JobError("ExecNotAllowed", f"exec step {s.get('name')!r}: {exe} is not in the allow-list "
                               "$XDG_CONFIG_HOME/ballasim/exec_allow.json, and the runner was started without "
                               "--allow-exec", EXIT_CONFIG)
    return {**doc, "sets": sets}


class Status:
    """status.json (``ballasim.job-status`` 1.0), shared by the two threads and written atomically."""

    def __init__(self, path: Path):
        self.path, self._lock, self.started = path, threading.Lock(), time.monotonic()
        self.doc = {"format": STATUS_FORMAT, "version": [1, 0], "state": "starting", "step": None, "stage": "",
                    "fraction": None, "camera": None, "window": None, "windows": None, "events": None,
                    "elapsed_s": 0.0, "eta_s": None, "pid": os.getpid(), "heartbeat": _now(), "error": None}

    def update(self, *, write: bool = False, **fields) -> None:
        with self._lock:
            self.doc.update(fields)
            if write:
                self._write()

    def heartbeat(self) -> None:
        with self._lock:
            self._write()

    def _write(self) -> None:
        self.doc.update(heartbeat=_now(), elapsed_s=round(time.monotonic() - self.started, 3))
        write_json_atomic(self.path, self.doc)


def preview_times(time_map, timeline: dict, frames, span_ns) -> list[dict]:
    """The preview frames (blender.md §3.3): frame f at t(f) = ``time_map.to_ns(f)``, its ``prev`` window
    [t(f - 1), t(f)) (the events since the previous frame) and its ``exposure`` window (§4.2: CENTER, START or END
    with the shutter σ of the timeline's ``motion_blur``), both clipped to the run span. ``frames``: "span" (the
    timeline's frame_start to frame_end) or [f0, f1, step] (f1 included). Frame f has index f - f0, the number of its
    PNG, so the add-on's image-sequence offset puts every image on its frame also when step > 1."""
    f0, f1, step = _preview_frames(frames, timeline)
    ends = _exposure_ends(timeline)
    s0, s1 = span_ns

    def clip(a: int, b: int) -> list[int]:
        a, b = min(max(a, s0), s1), min(max(b, s0), s1)
        return [a, max(a, b)]

    out = []
    for f in range(f0, f1 + 1, step):
        t = int(time_map.to_ns(f))
        out.append({"index": f - f0, "frame": f, "t_ns": t, "prev": clip(int(time_map.to_ns(f - 1)), t),
                    "exposure": clip(int(time_map.to_ns(f + ends[0])), int(time_map.to_ns(f + ends[1])))})
    return out


def _time_map(timeline: dict):
    """The engine's TimeMap of a timeline record (blender.md §4.2), bound at BL-6 as ``_scene.TimeMap``."""
    from ballasim import _scene

    if not hasattr(_scene, "TimeMap"):
        raise JobError("NotBuilt", "the preview step needs the engine's TimeMap (ballasim._scene.TimeMap, BL-6), "
                       "which this build does not have")
    return _scene.TimeMap.from_record(timeline)


class Runner:
    """The steps of one job. :func:`main` adds the locks, the signals and the supervisor thread."""

    def __init__(self, job_dir: Path, job: dict, status: Status, cancel: Cancel, gpu_lock: Path | None = None):
        self.job_dir, self.job, self.status, self.cancel = job_dir, job, status, cancel
        self.run_dir = job_dir / "run"
        self.cameras: list[str] = []
        self.gpu_lock = gpu_lock

    def run_steps(self) -> int:
        """Runs every step; the final state, error and exit code go into status.json."""
        try:
            for step in self.job["steps"]:
                if self.cancel.is_set():
                    raise _Cancelled
                getattr(self, f"_step_{step['step']}")(step)
        except _Cancelled:
            self.status.update(state="cancelled", stage="cancelled", eta_s=None, write=True)
            return EXIT_CANCELLED
        except JobError as e:
            return self.fail(e.kind, str(e), e.exit_code)
        except SystemExit as e:
            return self.fail("SystemExit", str(e), EXIT_FAILED)
        except Exception as e:  # noqa: BLE001  every failure ends the job in status.json
            from ballasim.config import ConfigError

            config = isinstance(e, (ConfigError, FileNotFoundError, FileExistsError))
            if not config:
                traceback.print_exc()
            return self.fail(type(e).__name__, str(e), EXIT_CONFIG if config else EXIT_FAILED)
        self.status.update(state="complete", stage="complete", fraction=1.0, eta_s=None, write=True)
        return EXIT_OK

    def fail(self, kind: str, message: str, exit_code: int) -> int:
        log.error("job failed: %s: %s", kind, message)
        self.status.update(state="failed", eta_s=None, error={"type": kind, "message": message,
                                                               "exit_code": exit_code}, write=True)
        return exit_code

    # ---- steps ----

    def _step_validate(self, step: dict) -> None:
        from ballasim import config

        from .engine import _check_run_dir
        from .scene import load_source, scene_facts, render_bundle

        self.status.update(state="loading", step="validate", stage="validate: resolving the configuration",
                           fraction=None, write=True)
        spec = config.load(self.job["config"], sets=self.job["sets"])
        values = spec.scene.values()["run"]
        source = load_source(values)
        resolved = config.resolve(spec, render_bundle(source, values), supplied=False,
                                  **scene_facts(values, source))
        for w in resolved.warnings:
            log.warning("%s", w)
        self.run_dir = _check_run_dir(Path(spec.get("output.dir")), spec.get("output.overwrite"))
        self.cameras = list(resolved.cameras)

    def _step_run(self, step: dict) -> None:
        """BL-0b: the engine in this (the worker) thread, with the job's cancel token; each window updates
        status.json."""
        from ballasim import _engine, config

        from .cli import _report
        from .engine import run

        spec = config.load(self.job["config"], sets=self.job["sets"])
        self.run_dir = Path(spec.get("output.dir"))
        started = [0.0]     # after the GPU lock wait

        def resolved(r) -> None:
            self.cameras = list(r.cameras)

        def on_window(cam: str, k: int, n: int, events: int, t_ns: int) -> None:
            i = self.cameras.index(cam) if cam in self.cameras else 0
            fraction = (i * n + k) / (max(len(self.cameras), 1) * n)
            elapsed = time.monotonic() - started[0]
            self.status.update(camera=cam, window=k, windows=n, events=events, stage=f"camera {cam}: window {k}/{n}",
                               fraction=fraction, eta_s=elapsed * (1 - fraction) / fraction if fraction else None,
                               write=True)

        with self._gpu("run"):
            self.status.update(state="running", step="run", stage="run: starting", fraction=0.0, write=True)
            started[0] = time.monotonic()
            with self.cancel.token() as token:
                try:
                    summary = run(spec, report=resolved, on_window=on_window, cancel=token)
                except BaseException as e:
                    # The run's frames hold the engine and its device memory: freed here, inside the GPU lock.
                    traceback.clear_frames(e.__traceback__)
                    if isinstance(e, _engine.Cancelled):
                        raise _Cancelled from None
                    raise
        print(_report(summary), flush=True)

    def _step_dense(self, step: dict) -> None:
        a = step.get("args") or {}
        argv = [*self._engine_child("ballasim"), "dense", str(self.run_dir), "--dt-ms", str(a.get("dt_ms", 20))]
        for key, flag in (("t0_s", "--t0"), ("t1_s", "--t1")):
            if a.get(key) is not None:
                argv += [flag, str(a[key])]
        if a.get("channels"):
            argv += ["--channels", ",".join(a["channels"])]
        if a.get("camera"):
            argv += ["--camera", a["camera"]]
        with self._gpu("dense"):
            self.status.update(state="running", step="dense", stage="dense frames", fraction=None, write=True)
            self._child(argv, "dense", again=True)

    def _step_preview(self, step: dict) -> None:
        a = step.get("args") or {}          # checked with the timeline by read_job
        window, timeline = a.get("window", "prev"), self.job["timeline"]
        doc = json.loads((self.run_dir / "run.json").read_text())
        if doc.get("status") != "complete":
            raise JobError("JobError", f"{self.run_dir}: the run is {doc.get('status')}, not complete")
        camera = a.get("camera") or doc["cameras"][0]["name"]
        self.status.update(state="previewing", step="preview", stage="preview: frame times", fraction=None,
                           write=True)
        frames = preview_times(_time_map(timeline), timeline, a.get("frames", "span"), doc["span_ns"])
        out = self.job_dir / "preview"
        out.mkdir(exist_ok=True)
        write_json_atomic(out / "times.json", {"format": "ballasim.preview-times", "version": [1, 0],
                                               "camera": camera, "frames": frames})
        name = "events_{index:05d}.png"
        self.status.update(stage=f"preview: {len(frames)} frames of camera {camera}", write=True)
        self._child([*self._engine_child("ballasim.viz"), "frames", "--sim",
                     str(self.run_dir / "events" / f"{camera}.h5"), "--times-ns", str(out / "times.json"),
                     "--win", window, "--flat", "--name", name, "--mask", "none", "--out", str(out)], "preview",
                    again=True)
        write_json_atomic(out / "index.json", {
            "format": "ballasim.preview-index", "version": [1, 0], "camera": camera, "window": window,
            "frames": [{"index": f["index"], "frame": f["frame"], "t_ns": f["t_ns"], "window": f[window],
                        "file": name.format(index=f["index"])} for f in frames]})

    def _step_exec(self, step: dict) -> None:
        name = step.get("name") or "exec"
        cwd = self.job_dir / (step.get("cwd") or ".")
        progress = step.get("progress")
        progress = None if progress is None else self.job_dir / progress

        last = [None]

        def report() -> None:
            try:
                p = json.loads(progress.read_text())
                done, total = int(p["done"]), int(p["total"])
            except (OSError, ValueError, KeyError, TypeError):
                return
            if (done, total) != last[0]:
                last[0] = done, total
                self.status.update(fraction=done / total if total else None, stage=f"{name}: {done}/{total}",
                                   write=True)

        ctx = self._gpu(name) if step.get("gpu") else contextlib.nullcontext()
        with ctx:
            self.status.update(state="exec", step="exec", stage=f"{name}: starting", fraction=None, write=True)
            self._child([sys.executable, "-I", "-S", str(_EXEC_SUPERVISOR), str(os.getpid()), "--", *step["argv"]],
                        name, report if progress else None, cwd=cwd)

    # ---- children and the GPU lock ----

    @staticmethod
    def _engine_child(module: str) -> list[str]:
        return [*python_argv(), "-m", __name__, "child", str(os.getpid()), module, "--"]

    def _child(self, argv: list[str], name: str, report=None, cwd: Path | None = None, *, again: bool = False) -> None:
        """Runs a child in its own process group to its end; a cancel sends SIGINT to that group once and, ``again``
        (an engine child, :func:`child_main`), to the child alone every second until it exits: Python can swallow
        the first while the child starts, before child_main's handler is installed (the default handler's
        KeyboardInterrupt in an import's weakref callback), and child_main ignores the repeats once it has one. Exit 3
        (or a death by SIGINT) is a cancel, any other non-zero exit a JobError (exit 2 stays 2)."""
        log.info("step %s: %s", name, " ".join(argv))
        proc = subprocess.Popen(argv, cwd=cwd or self.job_dir, stdin=subprocess.DEVNULL, process_group=0)
        interrupted = None      # when the last SIGINT went
        try:
            while True:
                try:
                    rc = proc.wait(_POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if self.cancel.is_set():
                    if interrupted is None:
                        _signal_group(proc, signal.SIGINT)
                        interrupted = time.monotonic()
                    elif again and time.monotonic() - interrupted >= _INTERRUPT_AGAIN_S:
                        proc.send_signal(signal.SIGINT)     # not reaped yet, so the pid is still the child's
                        interrupted = time.monotonic()
                if report is not None:
                    report()
        finally:
            if proc.poll() is None:          # an error in this thread: never leave the child behind
                _signal_group(proc, signal.SIGKILL)
                proc.wait()
        if report is not None:
            report()
        if rc == EXIT_CANCELLED or (self.cancel.is_set() and rc in (-signal.SIGINT, 128 + signal.SIGINT)):
            raise _Cancelled
        if rc != 0:
            code = EXIT_CONFIG if rc == EXIT_CONFIG else EXIT_FAILED
            raise JobError("StepError", f"{name}: {argv[0]} exited with {rc}", code)

    @contextlib.contextmanager
    def _gpu(self, step: str):
        """Holds the GPU lock (flock LOCK_EX) for one step; waits in state ``waiting``, honouring cancel."""
        if self.gpu_lock is None:
            yield
            return
        try:
            fd = os.open(self.gpu_lock, os.O_RDWR | os.O_CREAT, 0o666)
        except OSError as e:
            raise JobError("GpuLockError", f"gpu_lock {self.gpu_lock}: {e}") from None
        try:
            waiting = False
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not waiting:
                        waiting = True
                        self.status.update(state="waiting", step=step, stage=f"{step}: waiting for the GPU lock "
                                           f"{self.gpu_lock}", fraction=None, write=True)
                    if self.cancel.wait(1.0):
                        raise _Cancelled from None
            yield
        finally:
            os.close(fd)


def _signal_group(proc: subprocess.Popen, signum: int) -> None:
    """A signal to a child's process group (the child and what it started), once; nothing when it is gone."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signum)


def _runner_lock(path: Path) -> int | None:
    """E3: ``bridge/runner.lock`` held for the runner's lifetime (non-inheritable); None when another runner has it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    for _ in range(_LOCK_TRIES):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            time.sleep(_LOCK_RETRY_S)
    os.close(fd)
    return None


def main(job_dir: str | os.PathLike, *, allow_exec: bool = False) -> int:
    """``ballasim job``: the supervisor (this, the main thread) and one worker thread running the steps."""
    job = Path(job_dir).absolute()
    if not job.is_dir():
        print(f"error: {job} is not a job directory", file=sys.stderr)
        return EXIT_CONFIG
    if _runner_lock(job / "bridge" / "runner.lock") is None:
        print(f"error: another runner holds {job / 'bridge' / 'runner.lock'}", file=sys.stderr)
        return EXIT_FAILED
    signalled: list[int] = []

    def on_signal(signum, _frame) -> None:
        # Only a note: this runs on the main thread between any two bytecodes, also inside cancel.set() or a status
        # write, whose locks are not reentrant. The supervisor loop sets the token.
        signalled.append(signum)

    signums = [getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGBREAK") if hasattr(signal, name)]
    for signum in signums:
        signal.signal(signum, on_signal)
    try:
        return _supervise(job, allow_exec, signalled)
    finally:
        for signum in signums:      # the job has its final state: later signals change nothing, also during exit
            signal.signal(signum, signal.SIG_IGN)


def _supervise(job: Path, allow_exec: bool, signalled: list[int]) -> int:
    os.chdir(job)
    cancel = Cancel()
    status = Status(job / "status.json")
    status.update(write=True)
    try:
        doc = read_job(job, allow_exec=allow_exec)
        runner = Runner(job, doc, status, cancel, _gpu_lock(job, doc))
    except Exception as e:  # noqa: BLE001  every failure ends the job in status.json
        kind, code = (e.kind, e.exit_code) if isinstance(e, JobError) else (type(e).__name__, EXIT_CONFIG)
        if not isinstance(e, JobError):
            traceback.print_exc()
        status.update(state="failed", error={"type": kind, "message": str(e), "exit_code": code}, write=True)
        print(f"error: {e}", file=sys.stderr)
        return code
    if doc.get("nice") and doc["nice"] > os.getpriority(os.PRIO_PROCESS, 0):
        os.setpriority(os.PRIO_PROCESS, 0, doc["nice"])   # this thread; the worker and the children inherit it
    result = [EXIT_FAILED]
    worker = threading.Thread(target=lambda: result.__setitem__(0, runner.run_steps()), name="job-worker")
    cancel_file = job / "cancel"

    def check_cancel() -> None:
        if not cancel.is_set() and (signalled or cancel_file.exists()):
            if signalled:
                log.warning("job: %s, cancelling", signal.Signals(signalled[0]).name)
            cancel.set()

    check_cancel()                       # cancelled before it started: no step runs
    worker.start()
    beat = 0.0
    while worker.is_alive():
        check_cancel()
        if time.monotonic() >= beat:
            beat = time.monotonic() + 1.0
            if status.doc["state"] not in FINAL_STATES:
                status.heartbeat()
        worker.join(_POLL_S)
    return result[0]


def _gpu_lock(job_dir: Path, doc: dict) -> Path | None:
    """The GPU lock file: job.json's ``gpu_lock`` (relative to the job directory), else ``$BALLASIM_GPU_LOCK``, which
    must be absolute: one lock shared by every job, never one per job directory."""
    if doc.get("gpu_lock"):
        return job_dir / doc["gpu_lock"]
    env = os.environ.get("BALLASIM_GPU_LOCK")
    if not env:
        return None
    if not os.path.isabs(env):
        raise JobError("GpuLockError", f"$BALLASIM_GPU_LOCK must be an absolute path, got {env!r}", EXIT_CONFIG)
    return Path(env)


class _ChildInterrupt:
    """child_main's SIGINT and SIGTERM handling: the first signal raises KeyboardInterrupt and later ones are ignored,
    so the step's cleanup runs once and completes. A KeyboardInterrupt that Python swallows, because the handler ran in
    a context that cannot raise (a weakref callback or ``__del__``, e.g. during the child's imports: "Exception ignored
    in ..."), reaches :meth:`unraisable`, which has the signal delivered again outside that context. Without it the
    runner's one SIGINT was lost and the step ran to its end (BL-0b; the add-on's test_cancel_on_quit found it)."""

    ARMED, RAISED, CAUGHT = 0, 1, 2

    def __init__(self):
        self.state, self.signum = self.ARMED, signal.SIGINT
        self._again = None   # the delivery thread's queue (SimpleQueue.put is reentrant), from the first swallow on

    def handler(self, signum, frame) -> None:
        if self.state != self.ARMED:
            return
        if self._in_hook(frame):     # raised in the hook, it would be swallowed too
            self.deliver_again(signum)
            return
        self.state, self.signum = self.RAISED, signum
        raise KeyboardInterrupt

    def unraisable(self, unraisable) -> None:
        sys.__unraisablehook__(unraisable)
        if self.state == self.RAISED and issubclass(unraisable.exc_type, KeyboardInterrupt):
            self.state = self.ARMED
            self.deliver_again(self.signum)

    @classmethod
    def _in_hook(cls, frame) -> bool:
        code = cls.unraisable.__code__
        while frame is not None and frame.f_code is not code:
            frame = frame.f_back
        return frame is not None

    def deliver_again(self, signum: int) -> None:
        """The signal again, to the main thread's handler, 1 ms later from a delivery thread (the handler defers
        again while the main thread is still in the hook)."""
        if self._again is None:
            self._again = queue.SimpleQueue()
            threading.Thread(target=self._deliver, args=(self._again,), name="child-interrupt", daemon=True).start()
        self._again.put(signum)

    @staticmethod
    def _deliver(requests) -> None:
        while True:
            signum = requests.get()
            time.sleep(0.001)
            _thread.interrupt_main(signum)

    def install(self) -> None:
        signal.signal(signal.SIGINT, self.handler)
        signal.signal(signal.SIGTERM, self.handler)
        sys.unraisablehook = self.unraisable


def child_main(argv: list[str]) -> int:
    """``python -m ballasim.engine.job child PARENT MODULE -- ARGS``: runs ``python -m MODULE ARGS`` (MODULE in
    CHILD_MAINS) as a job step. The first SIGINT or SIGTERM raises KeyboardInterrupt and later ones are ignored, so
    the step's cleanup runs once and completes; KeyboardInterrupt exits 3. One that Python swallows is raised again
    (:class:`_ChildInterrupt`); one lost before this handler is installed (the interpreter's start-up and this
    module's standard-library imports) is covered by the runner's SIGINT every second (``Runner._child``). On Linux
    the child dies with the runner thread that started it (PR_SET_PDEATHSIG)."""
    parent, module, rest = int(argv[0]), argv[1], argv[3:]
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    if os.getppid() != parent:
        print(f"{module}: the job runner {parent} is gone", file=sys.stderr)
        return EXIT_FAILED
    interrupt = _ChildInterrupt()
    interrupt.install()
    from importlib import import_module

    try:
        return import_module(CHILD_MAINS[module]).main(rest)
    except KeyboardInterrupt:
        interrupt.state = interrupt.CAUGHT
        print(f"{module}: cancelled", file=sys.stderr)
        return EXIT_CANCELLED


if __name__ == "__main__":
    if sys.argv[1:2] != ["child"] or len(sys.argv) < 5 or sys.argv[4] != "--" or sys.argv[3] not in CHILD_MAINS:
        sys.exit("usage: python -m ballasim.engine.job child PARENT_PID {ballasim,ballasim.viz} -- ARGS")
    sys.exit(child_main(sys.argv[2:]))
