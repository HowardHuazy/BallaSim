"""Per-camera clock offsets for EVIMO2 replays (DESIGN §5.5, PHASE1 "Clocks", G1): the inputs read here (G1's table,
the raw extrinsics' line 3); the offset itself is the C++ build's (src/scene/evimo2/build.cpp, S4), recorded in the
bundle meta's ``clocks``.

C1 defines ``t_sensor = t_true + clock_offset_ns``. The true clock of a replay bundle is the
re-timed Vicon clock. Two sensor clocks are meaningful (``reference``):

- ``"npz"`` (default): the clock of the released event timestamps (``dataset_events_t.npy``), so
  simulated and real events compare directly. EVIMO already subtracted the raw ``extrinsics.txt``
  line 3 (L3) from these stamps (t_npz = t_raw - L3 - t_bag0, G1 §2.1); what remains is G1's
  residual delta, defined by t_vicon = t_npz + delta, hence offset = -delta. L3 is therefore not
  an offset on this clock: applying it again would shift events by 2-63 ms (the imo_ll values).
- ``"raw"``: the camera driver's clock relative to bag start, t_raw = t_npz + L3, hence
  offset = L3 - delta. It needs L3, which exists only in the raw archives (``imo_ll`` here).

delta comes from G1's table (91 camera-sequences, SE median 0.25 ms); when a sequence is not
tabulated, delta = 0 and a warning says so. 17 rows, the sequences whose first GT frame id is 1 or 2
(sfm/train and sfm_ll/eval, one Samsung imo/train), were re-measured on 2026-10-10: G1 had keyed
their mask/depth archives by the raw frame id instead of id - first id, a GT frame (16.7 ms) late
per unit of first id (the table's _source and each row's "remeasured" field;
docs/dev/EVIMO2_BENCH.md). Rows with se_ms null are not identified (the two sfm_ll scene17_d_02 rows,
left imo_ll scene16_d_dyn_test_04): the build uses them with a warning. Drift priors (G1 t6, :data:`DRIFT_PPM`) are
reported, not applied: C1 has a constant offset only.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .dataset import CAMERA_FOLDERS, Sequence

DEFAULT_RAW_ROOT = "/data/evimo2_raw"
#: Sensor clock rate relative to the pose clock on the NPZ event clock (G1 t6, ppm): the meta's ``clocks.*.drift_ppm``
#: (the build's copy is src/scene/evimo2/build.cpp ``sensor_of``; a test keeps them equal).
DRIFT_PPM = {"left_camera": 10.8, "right_camera": 18.0, "samsung_mono": -2.0}
_G1_TABLE = Path(__file__).with_name("g1_offsets.json")


def raw_root() -> Path:
    return Path(os.environ.get("EVIMO2_RAW_ROOT", DEFAULT_RAW_ROOT))


def raw_extrinsics_path(directory: str | os.PathLike, root: Path | None = None) -> Path | None:
    """The raw archive's ``extrinsics.txt`` of a sequence folder (``<camera>/<category>/<split>/<name>``, read from the
    path alone: the bundle cache key holds it), or None for a path outside a camera folder. ``root``: the raw archive
    (scene.evimo2.raw_root); default raw_root()."""
    parts = Path(directory).resolve().parts
    cams = [i for i, p in enumerate(parts) if p in CAMERA_FOLDERS]
    if not cams:
        return None
    i = cams[-1]
    split = parts[i + 2] if len(parts) > i + 3 else ""
    recording = re.sub(r"_\d{6}$", "", Path(directory).name)
    return (root or raw_root()) / parts[i + 1] / split / recording / parts[i] / "extrinsics.txt"


def raw_line3_s(seq: Sequence, root: Path | None = None) -> float | None:
    """Line 3 of the raw ``extrinsics.txt`` (camera - Vicon time offset EVIMO applied, s), or None. ``root``: the raw
    archive (scene.evimo2.raw_root); default raw_root(). The file is :func:`raw_extrinsics_path`'s (the cache key's)."""
    path = raw_extrinsics_path(seq.directory, root)
    if path is None or not path.exists():
        return None
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: fewer than 3 lines")
    return float(lines[2].split()[0])


def g1_delta_ms(seq: Sequence) -> tuple[float, float | None] | None:
    """G1's residual offset (delta, SE) in ms for this camera-sequence, or None if not measured."""
    table = json.loads(_G1_TABLE.read_text())["offsets"]
    rec = table.get(f"{seq.camera}/{seq.category}/{seq.name}")
    return None if rec is None else (float(rec["delta_ms"]), rec["se_ms"])
