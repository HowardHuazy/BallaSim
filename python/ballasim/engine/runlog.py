"""Run directory bookkeeping: run.json, logs/perf.jsonl and progress.json (docs/dev/ENGINE.md), and the run
directory's lock.

The lock (docs/design/phase3/ui.md §4.10) is the sibling file ``<parent>/.<name>.lock``, which removing the run
directory (``output.overwrite``) cannot remove. A run takes it ``LOCK_EX`` from the check of its directory to its end
(:func:`lock_run_dir`); a viewer asks whether a run holds it with a ``LOCK_SH`` probe released at once
(:func:`run_dir_locked`), so a run's ``running`` status without the lock means a crashed run. A run that cannot create
the lock file (a parent directory it may not write) runs unlocked, with a warning, as before the lock existed; one
that fails before its run directory holds a run.json removes the lock file and the parent directories it created."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: The run lock is retried every 100 ms for up to 1 s, so a probe's momentary LOCK_SH is not taken for a run.
RUN_LOCK_RETRY_S, RUN_LOCK_TRIES = 0.1, 10


class RunDirInUse(FileExistsError):
    """Another run holds the run directory's lock (the CLI exits 2)."""


def run_lock_path(run_dir: str | os.PathLike) -> Path:
    """``<parent>/.<name>.lock`` of a run directory."""
    run_dir = Path(run_dir).absolute()
    return run_dir.parent / f".{run_dir.name}.lock"


class RunLock:
    """A run directory's lock held by this process (:func:`lock_run_dir`). ``created``: the lock file and the parent
    directories that taking it created (the deepest first), which :meth:`release` with ``discard`` removes."""

    def __init__(self, fd: int, run_dir: Path, path: Path, created: list[Path]):
        self.fd, self.run_dir, self.path, self.created = fd, run_dir, path, created

    def release(self, *, discard: bool = False) -> None:
        """Releases the lock; ``discard`` (a run that failed before its run directory held a run.json) first removes
        the lock file and the directories taking it created (those that are empty), while still holding it: a run
        waiting on the removed file sees it is no longer the lock and takes the new one."""
        if self.fd is None:
            return
        try:
            if discard:
                for p in self.created:
                    with contextlib.suppress(OSError):
                        p.unlink() if p == self.path else p.rmdir()
        finally:
            os.close(self.fd)
            self.fd = None


def lock_run_dir(run_dir: str | os.PathLike) -> RunLock | None:
    """Takes the run directory's lock (``LOCK_EX | LOCK_NB``, retried every 100 ms for up to 1 s), creating the parent
    directory if needed; its file descriptor is not inherited by children. :class:`RunDirInUse` when another run holds
    it; None, with a warning, when the lock file cannot be created or opened (e.g. a parent this user may not write):
    the run then runs unlocked and fails, or not, where it did before the lock existed."""
    run_dir = Path(run_dir).absolute()
    path = run_lock_path(run_dir)
    created = []
    for attempt in range(RUN_LOCK_TRIES):
        try:
            missing = []
            for d in (path.parent, *path.parent.parents):
                if d.exists():
                    break
                missing.append(d)
            path.parent.mkdir(parents=True, exist_ok=True)
            created += [d for d in missing if d not in created]
            try:
                fd, mine = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_EXCL, 0o666), True
            except FileExistsError:
                fd, mine = os.open(path, os.O_RDONLY), False
        except FileNotFoundError:           # removed between the two opens by a run that failed early: again
            continue
        except OSError as e:
            log.warning("%s: cannot create or open the run lock %s (%s); the run is not locked", run_dir, path,
                        e.strerror or type(e).__name__)
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            if attempt + 1 < RUN_LOCK_TRIES:
                time.sleep(RUN_LOCK_RETRY_S)
            continue
        try:                                # the file locked must still be the lock (not one a failed run removed)
            st, own = os.stat(path), os.fstat(fd)
            current = (st.st_dev, st.st_ino) == (own.st_dev, own.st_ino)
        except FileNotFoundError:
            current = False
        if current:
            return RunLock(fd, run_dir, path, ([path] if mine else []) + created)
        os.close(fd)
    raise RunDirInUse(f"{run_dir}: run directory in use (another run holds {path})")


def run_dir_locked(run_dir: str | os.PathLike) -> bool:
    """Whether a run holds the run directory's lock: a ``LOCK_SH | LOCK_NB`` probe, released at once; no lock file is
    created."""
    try:
        fd = os.open(run_lock_path(run_dir), os.O_RDONLY)
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


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None   # JSON has no inf/NaN: null
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(_jsonable(doc), f, indent=1, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def git_commit() -> dict:
    """Commit of the source tree the package runs from (None outside a git checkout)."""
    import ballasim

    root = Path(ballasim.__file__).resolve().parent
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip() != ""
        return {"commit": head, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def versions() -> dict:
    import h5py
    import hdf5plugin
    import ruamel.yaml
    import scipy

    import ballasim
    from ballasim import _core

    return {"ballasim": ballasim.__version__, "build": dict(_core.build_info()), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version, "hdf5plugin": hdf5plugin.version,
            "ruamel.yaml": ruamel.yaml.__version__}


class RunLog:
    """perf.jsonl (one record per window) and progress.json of a run directory. progress.json's ``camera_index`` and
    ``cameras`` (the camera's place among the run's cameras and their number) and ``camera_elapsed_s`` (since the
    camera's start, :meth:`start_camera`) let a viewer compute the run's fraction and ETA; ``elapsed_s`` counts from
    the run's start."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.perf_path = run_dir / "logs" / "perf.jsonl"
        self.progress_path = run_dir / "progress.json"
        self.perf_path.write_text("")
        self.started = self.camera_started = time.monotonic()
        self.camera_index, self.cameras = 0, 1

    def start_camera(self, name: str, selected) -> None:
        """A camera's run starts (engine.run_camera): ``selected``, the run's cameras in order (resolved.cameras)."""
        names = list(selected)
        if name in names:
            self.camera_index = names.index(name)
        self.cameras, self.camera_started = max(len(names), 1), time.monotonic()

    def window(self, record: dict) -> None:
        with open(self.perf_path, "a") as f:
            f.write(json.dumps(_jsonable(record), allow_nan=False) + "\n")

    def progress(self, **fields) -> None:
        now = time.monotonic()
        write_json_atomic(self.progress_path, {**fields, "camera_index": self.camera_index, "cameras": self.cameras,
                                               "elapsed_s": now - self.started,
                                               "camera_elapsed_s": now - self.camera_started,
                                               "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
