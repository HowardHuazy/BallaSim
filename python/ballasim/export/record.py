"""``EXPORT.json``, contract C6 (docs/dev/EXPORT.md; docs/design/phase3/outputs.md §3.6): one per export directory.

It records the source run (run.json's SHA-256; the input files by size and mtime, not hashed), every option as
resolved, the clock (epoch, origin, the cameras' windows on their sensor clocks, the time rule of each output, the
dropped sub-us parts), the units, the IMU state of each camera, the outputs (size, SHA-256 hashed in C++ at close,
streams) and the warnings. ``created`` is its only field that is not reproducible.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

FORMAT = "ballasim.export/1"
#: The clock strings of every record (§4.7).
IMU_STAMPS = "camera sensor clock"
FRAME_TIMES = "true clock mapped by the camera sensor clock (offset + drift)"


def input_entry(run_dir: Path, path: Path) -> dict:
    """An input file of the export, relative to the run directory, by size and mtime."""
    st = path.stat()
    return {"path": path.relative_to(run_dir).as_posix(), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def make_record(*, run_dir: Path, run_json_sha256: str, inputs: list[dict], options: dict, clock: dict, units: dict,
                frames: dict, imu: dict, outputs: list[dict], warnings: list[str]) -> dict:
    import ballasim
    from ballasim.engine.runlog import git_commit

    return {"format": FORMAT, "ballasim": ballasim.__version__, "git": git_commit(),
            "source": {"run_dir": str(run_dir), "run_json_sha256": run_json_sha256, "inputs": inputs},
            "options": options, "clock": {**clock, "imu_stamps": IMU_STAMPS, "frame_times": FRAME_TIMES},
            "units": units, "frames": frames, "imu": imu, "outputs": outputs, "warnings": list(warnings),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def write_record(path: Path, record: dict) -> None:
    from ballasim.engine.runlog import write_json_atomic

    write_json_atomic(path, record)


def read_record(path: str | Path) -> dict:
    """An EXPORT.json, refused (ExportError) unless its format is ``ballasim.export/1`` (any minor)."""
    from .errors import ExportError

    doc = json.loads(Path(path).read_text())
    name, _, version = str(doc.get("format", "")).partition("/")
    if name != FORMAT.split("/")[0] or version.split(".")[0] != "1":
        raise ExportError(f"{path}: format {doc.get('format')!r} is not {FORMAT}")
    return doc
