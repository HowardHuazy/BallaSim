"""``python -m ballasim imu`` and :func:`simulate_imu` (docs/dev/IMU.md; docs/design/phase3/outputs.md §3.4, §3.5, §4.11):
the IMU of a finished run's cameras, written as contract C5 ``<run>/imu/<camera>.h5`` with the record
``<run>/imu/<camera>.json`` (``ballasim.imu_run/1``).

The IMU is a post-process: it reads run.json (never modified) and the recorded resolved values, overlays new ``imu.*``
layers on them in C++ (:func:`ballasim._imu.resolve_imu`: ``--profile``, ``--camera-profile``, ``--set``), takes the rig
trajectory of the run's bundle (digest-checked) and steps ``ballasim._imu.ImuSimulator`` in 10 s blocks. Python reads
and writes files only: the model, the validation and the overlay are C++.

Files. C5 is written as ``.<camera>.h5.partial`` and renamed, with chunks of 2^14 samples and Fletcher-32 only (no
compression filter: it reads without hdf5plugin, unlike the event files). A rerun for one camera rewrites only that
camera's two files; the other cameras' files and run.json are untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from ruamel.yaml.error import YAMLError

from .errors import ExportError

log = logging.getLogger(__name__)

#: The simulator is stepped, and C5 written, in blocks of this length (§5.3: bounded memory, about 2.3 MB per block).
BLOCK_NS = 10_000_000_000
#: C5 chunk length in samples (§3.5).
CHUNK = 1 << 14
#: World gravity when neither imu.gravity_mps2 nor the bundle meta gives one (§4.1): g_n, +Z up (authored scenes).
DEFAULT_GRAVITY = (0.0, 0.0, -9.80665)
GRAVITY_META = "gravity_world_mps2"
_PROFILE_ID = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+@v[0-9]+")


def parse_camera_profiles(items: Iterable[str]) -> dict[str, list[str]]:
    """``--camera-profile CAMERA=ID`` items as {camera: [ids]} in the given order; ExportError for another shape."""
    out: dict[str, list[str]] = {}
    for item in items:
        camera, sep, pid = str(item).partition("=")
        if not sep or not camera.strip() or not pid.strip():
            raise ExportError(f"--camera-profile expects CAMERA=ID, got {item!r}")
        out.setdefault(camera.strip(), []).append(pid.strip())
    return out


def simulate_imu(run_dir, *, cameras: Sequence[str] | None = None, sets: Iterable[str] | Mapping = (),
                 profiles: Iterable[str] = (), camera_profiles: Mapping[str, str | Sequence[str]] | None = None,
                 bundle=None, overwrite: bool = False) -> list[Path]:
    """Simulates the IMU of the cameras of a finished run and writes ``<run>/imu/<camera>.h5`` (C5) and
    ``<run>/imu/<camera>.json``; returns the C5 files.

    ``cameras``: default the cameras whose resolved ``imu.enabled`` is true. The ``imu.*`` values are the run's recorded
    ones under, lowest first: ``profiles`` (root-mounted profile ids), ``camera_profiles`` ({camera: id or ids}; mount
    camera) and ``sets`` (``--set KEY=VALUE`` items or {full key: value}: ``cameras.<camera>.imu.*`` or
    ``cameras.default.imu.*``). A key outside ``imu.*`` is refused. ``bundle`` replaces loading the run's scene (it must
    have the run's digest, and is required when the run's bundle was supplied). ``overwrite`` replaces existing files.
    Refusals raise :class:`ExportError`; run.json is never modified."""
    from ballasim import _imu, config
    from ballasim.engine.scene import bundle_digest, load_bundle

    from .formats import _read_run

    run_dir = Path(run_dir)
    doc, run_sha = _read_run(run_dir)
    rec = doc["resolved"]
    run_names = list(rec["cameras"])
    stages = doc.get("stage_digests") or {}
    for c in run_names:
        if "sensor" not in stages.get(c, {}):
            raise ExportError(f"{run_dir}: run.json has no stage_digests of camera {c} (a run written by ballasim run "
                              f"has them)")
    recorded = {"cameras": {c: {k: e for k, e in v.items() if k.startswith("imu.")} for c, v in rec["cameras"].items()},
                "digests": {c: stages[c]["sensor"] for c in run_names}}
    roots = [str(p) for p in profiles]
    per_camera = {c: [ids] if isinstance(ids, str) else [str(i) for i in ids]
                  for c, ids in (camera_profiles or {}).items()}
    unknown = [c for c in per_camera if c not in run_names]
    if unknown:
        raise ExportError(f"--camera-profile {unknown[0]}: not a camera of the run ({', '.join(run_names)})")
    docs = _profile_docs([*roots, *(i for ids in per_camera.values() for i in ids)])
    try:
        sets_dict = config.parse_sets(sets)
    except (ValueError, YAMLError) as e:
        raise ExportError(f"--set: {e}") from None
    res = _imu.resolve_imu(recorded, roots, per_camera, sets_dict, docs, run_names)
    enabled = [c for c in run_names if res["cameras"][c]["imu.enabled"]["value"]]
    selected = _select(cameras, enabled, run_names)
    out_h5 = {c: run_dir / "imu" / f"{c}.h5" for c in selected}
    for c in selected:
        for path in (out_h5[c], out_h5[c].with_suffix(".json")):
            if path.exists() and not overwrite:
                raise ExportError(f"imu/{path.name} exists: pass --overwrite (overwrite=True) to replace this "
                                  f"camera's IMU files")

    if bundle is None:
        if doc.get("inputs", {}).get("bundle", {}).get("supplied"):
            raise ExportError("the run's bundle was supplied: pass bundle= (the scene keys do not give it)")
        bundle = load_bundle({k: e["value"] for k, e in rec["run"].items()})
    if bundle_digest(bundle) != doc["bundle"]["digest"]:
        raise ExportError(f"{run_dir}: the scene has changed since the run (bundle digest differs)")
    names = [c.name for c in bundle.cameras]
    missing = [c for c in selected if c not in names]
    if missing:
        raise ExportError(f"cameras {missing} are not in the bundle (it has {names})")
    rigs = {}
    for c in selected:
        rigs[c] = bundle.trajectories[bundle.cameras[names.index(c)].rig_trajectory]

    seed = int(rec["run"]["run.sensor_seed"]["value"])
    t_s, t_e = (int(v) for v in doc["span_ns"])
    evimo2 = rec["run"].get("scene.evimo2.sequence", {}).get("value") is not None or \
        isinstance(bundle.meta.get("source"), dict) and "category" in bundle.meta["source"]
    layers = {"profiles": roots, "camera_profiles": per_camera,
              "sets": {k: v for k, v in sets_dict.items()}}
    # Overlay warnings that name no selected camera (a key set on a camera whose IMU is off, a profile's block for a
    # camera the run lacks) are in no camera's record: log them here so none is lost.
    claimed = {w for c in selected for w in _camera_warnings(res["warnings"], c)}
    for w in res["warnings"]:
        if w not in claimed:
            log.warning("IMU overlay: %s", w)
    (run_dir / "imu").mkdir(exist_ok=True)
    written = []
    for c in selected:
        values = {k: e["value"] for k, e in res["cameras"][c].items()}
        try:
            written.append(_simulate_camera(
                run_dir, c, bundle, names.index(c), rigs[c], values, res, rec, seed, (t_s, t_e), run_sha, evimo2, layers))
        except BaseException as e:
            from ballasim.engine.runlog import write_json_atomic

            # Includes failures before the writer opens (for example an invalid mount). Completed cameras survive.
            out_h5[c].with_name(f".{out_h5[c].name}.partial").unlink(missing_ok=True)
            write_json_atomic(out_h5[c].with_suffix(".json"), {
                "format": _imu.RUN_FORMAT, "status": "failed", "camera": c, "run_json_sha256": run_sha,
                "error": f"{type(e).__name__}: {e}",
            })
            raise
    return written


def _select(cameras: Sequence[str] | None, enabled: list[str], run_names: list[str]) -> list[str]:
    if cameras is None:
        if not enabled:
            raise ExportError("no camera has imu.enabled: set cameras.<camera>.imu.enabled=true (--set, or an IMU "
                              "profile with --camera-profile CAMERA=ID)")
        return enabled
    selected = list(dict.fromkeys(cameras))
    unknown = [c for c in selected if c not in run_names]
    if unknown or not selected:
        raise ExportError(f"cameras {unknown or selected}: the run has {', '.join(run_names)}")
    for c in selected:
        if c not in enabled:
            raise ExportError(f"--camera {c}: its IMU is off: set cameras.{c}.imu.enabled=true (--set, or an IMU "
                              f"profile with --camera-profile {c}=ID)")
    return selected


def _profile_docs(ids: Iterable[str]) -> dict:
    """The profile files the layers name, as plain documents (the resolver checks them)."""
    from ballasim import config

    docs: dict = {}
    for pid in ids:
        if pid in docs:
            continue
        if not _PROFILE_ID.fullmatch(pid):    # before any path is built from it (the resolver checks it again)
            raise ExportError(f"profile id {pid!r} is not <group>/<name>@v<N>")
        path = config.profile_path(pid)
        if not path.is_file():
            raise FileNotFoundError(f"profile {pid}: no file {path}")
        docs[pid] = config.read_profile(pid)
    return docs


def _step_join_warnings(pieces, span: tuple[int, int]) -> list[str]:
    """One warning per step join (a pose jump, join -1) of a piecewise rig inside the run's span (§4.3.4, A11): no
    specific force or rate is defined across it, the samples after it carry IMPACT, and a filtered sensor
    re-initializes there (WARMUP) and forgets what came before. C0 and C1 joins are real dynamics (an impact), so they
    are not warned about."""
    if pieces is None:
        return []
    t_s, t_e = span
    return [f"the rig trajectory has a step join (a pose jump) at t = {int(begin)} ns: the samples after it carry "
            f"the IMPACT flag, and a filtered sensor re-initializes there (WARMUP; outputs.md §4.3.4)"
            for begin, join in zip(pieces[1:, 0], pieces[1:, 5]) if join == -1 and t_s < int(begin) <= t_e]


def _gravity(values: dict, bundle) -> tuple[list[float], str]:
    """World gravity by priority (§4.1): imu.gravity_mps2, the bundle meta gravity_world_mps2, else (0, 0, -9.80665)."""
    g = values["imu.gravity_mps2"]
    if g is not None:
        return [float(x) for x in g], "parameter"
    meta = bundle.meta.get(GRAVITY_META)
    if meta is not None:
        try:
            vec = [float(x) for x in meta]
        except (TypeError, ValueError):
            vec = []
        if len(vec) != 3 or not all(math.isfinite(x) for x in vec):
            raise ExportError(f"bundle meta {GRAVITY_META} = {meta!r}: expected three finite numbers (m/s^2)")
        return vec, "bundle meta"
    return list(DEFAULT_GRAVITY), "default"


def _t_offset_us(run_dir: Path, camera: str) -> int:
    import h5py

    from ballasim import _imu

    path = run_dir / "events" / f"{camera}.h5"
    if not path.is_file():
        raise ExportError(f"{run_dir}: camera {camera} has no event file events/{camera}.h5 (its /t_offset sets the "
                          f"first IMU stamp)")
    with h5py.File(path, "r") as f:
        _imu.check_format(str(f.attrs.get("format", "")), "ballasim.events/1")
        return int(f["t_offset"][()])


def _simulate_camera(run_dir: Path, camera: str, bundle, index: int, traj, values: dict, res: dict, rec: dict,
                     sensor_seed: int, span: tuple[int, int], run_sha: str, evimo2: bool, layers: dict) -> Path:
    import ballasim
    from ballasim import _imu
    from ballasim.engine.runlog import write_json_atomic

    sensor = {k: e["value"] for k, e in rec["cameras"][camera].items()}
    clock = _imu.camera_clock(sensor["sensor.readout_quantum_s"], sensor["sensor.clock_offset_ns"],
                              sensor["sensor.clock_drift_ppm"], sensor_seed, index)
    t_offset_us = _t_offset_us(run_dir, camera)
    g_world, g_source = _gravity(values, bundle)
    cam = bundle.cameras[index]
    trajectory = {"K": traj.K, "t0_ns": traj.t0_ns, "h_ns": traj.h_ns, "rot": traj.rot, "pos": traj.pos,
                  "span_ns": traj.span_ns, "pieces": traj.pieces}
    t_s, t_e = span
    sim = _imu.ImuSimulator(trajectory, np.ascontiguousarray(cam.T_rig_cam, np.float64), values, clock,
                            np.array(g_world, np.float64), index, t_s, t_e, t_offset_us)
    entries = res["cameras"][camera]
    warnings = _camera_warnings(res["warnings"], camera)
    if g_source == "default" and evimo2:
        warnings.append("imu.gravity_mps2 is null on an EVIMO2 bundle: its world is the tabletop frame, so the gravity "
                        "(0, 0, -9.80665) is a placeholder; estimate it with tools/imu_bench/evimo2_gravity.py "
                        "--bundle <the bundle's directory> (the Vicon-world vector it prints without --bundle is in "
                        "another frame) or set imu.gravity_mps2")
    warnings += _step_join_warnings(traj.pieces, span)
    info = _ideal_info(values)
    change = res["rotation_change"][camera]
    out = run_dir / "imu" / f"{camera}.h5"
    partial = out.with_name(f".{out.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        stats = _write_c5(partial, sim, camera, values, entries, res["digests"][camera], g_world, t_offset_us, span)
        warm = stats["warmup_samples"]
        if warm:
            warnings.append(f"{warm} samples carry the WARMUP flag: the rig span starts less than 20 tau_c + d before "
                            f"the run (outputs.md §4.3.3)")
        partial.replace(out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    for w in warnings:
        log.warning("IMU of camera %s: %s", camera, w)
    for line in info:
        log.info("IMU of camera %s: %s", camera, line)
    record = {"format": _imu.RUN_FORMAT, "status": "complete", "camera": camera, "ballasim": ballasim.__version__,
              "run_json_sha256": run_sha, "layers": layers,
              "imu": {k: {"value": e["value"], "status": e["status"], "source": e["source"], "layer": e["layer"]}
                      for k, e in entries.items()},
              "imu_digest": res["digests"][camera],
              "gravity": {"source": g_source, "value": g_world},
              "t_cam_imu": {"max_rotation_change": change},
              "warnings": warnings, "info": info, "stats": stats,
              "file": {"path": f"imu/{out.name}", "bytes": out.stat().st_size, "sha256": _sha256(out)},
              "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json_atomic(out.with_suffix(".json"), record)
    return out


def _ideal_info(values: dict) -> list[str]:
    """The record's info line when ``imu.enabled`` is the only value that is not its default: an ideal IMU (outputs.md
    §3.1 notes, Q15), so a user who meant a real device (an IMU profile) sees from the record that none was applied."""
    from ballasim import _imu

    if any(values[k] != v for k, v in _imu.default_values().items() if k != "imu.enabled"):
        return []
    return ["every imu.* value but imu.enabled is at its default: an ideal IMU (gyro = omega, accel = specific force; "
            "no noise, bias, errors, filter or quantization); set cameras.<camera>.imu.* or use an IMU profile for a "
            "real device"]


def _camera_warnings(warnings: list[str], camera: str) -> list[str]:
    """The overlay's warnings that concern this camera (they name ``cameras.<camera>``, a profile's block of another
    camera excepted)."""
    pat = re.compile(rf"cameras\.{re.escape(camera)}(?![\w-])")
    return [w for w in warnings if pat.search(w)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class _Datasets:
    """The resizable C5 datasets, appended block by block."""

    def __init__(self, f, spec: Mapping[str, tuple[type, tuple[int, ...]]]):
        self.sets = {name: f.create_dataset(name, shape=(0, *tail), maxshape=(None, *tail),
                                            chunks=(CHUNK, *tail), dtype=dtype, fletcher32=True)
                     for name, (dtype, tail) in spec.items()}
        self.n = 0

    def append(self, columns: Mapping[str, np.ndarray]) -> None:
        n = len(next(iter(columns.values())))
        if n == 0:
            return
        for name, ds in self.sets.items():
            ds.resize(self.n + n, axis=0)
            ds[self.n:self.n + n] = columns[name]
        self.n += n


def _write_c5(path: Path, sim, camera: str, values: dict, entries: dict, digest: str, g_world: list[float],
              t_offset_us: int, span: tuple[int, int]) -> dict:
    """C5 (§3.5) of the whole span into ``path``; returns the simulator's stats."""
    import h5py

    from ballasim import _imu

    counts_dtype = np.int16 if values["imu.adc_bits"] <= 16 else np.int32
    spec: dict[str, tuple[type, tuple[int, ...]]] = {
        "t_ns": (np.int64, ()), "gyro": (np.float64, (3,)), "accel": (np.float64, (3,))}
    for name, sensor in (("counts_gyro", "gyro"), ("counts_accel", "accel")):
        if values[f"imu.{sensor}.lsb"] is not None:
            spec[name] = (counts_dtype, (3,))
    spec.update({"flags": (np.uint8, ()), "truth/t_true_ns": (np.int64, ()), "truth/omega": (np.float64, (3,)),
                 "truth/specific_force": (np.float64, (3,)), "truth/bias_gyro": (np.float64, (3,)),
                 "truth/bias_accel": (np.float64, (3,)), "truth/q_world_imu": (np.float64, (4,)),
                 "truth/p_world_imu": (np.float64, (3,))})
    source = {"truth/t_true_ns": "t_true_ns", "truth/omega": "omega_true", "truth/specific_force": "f_true",
              "truth/bias_gyro": "bias_gyro", "truth/bias_accel": "bias_accel", "truth/q_world_imu": "q_W_I",
              "truth/p_world_imu": "p_W_I"}
    profile_ids = sorted({e["layer"][len("profile:"):] for e in entries.values() if e["layer"].startswith("profile:")})
    with h5py.File(path, "w") as f:
        a = f.attrs
        a["format"] = _imu.FILE_FORMAT
        a["camera"] = camera
        a["rate_hz"] = np.float64(values["imu.rate_hz"])
        a["T_cam_imu"] = np.array(values["imu.T_cam_imu"], np.float64).reshape(4, 4)
        a["gravity_world_mps2"] = np.array(g_world, np.float64)
        a["imu_seed"] = np.int64(values["imu.seed"])
        a["profile_ids"] = json.dumps(profile_ids)
        a["params"] = json.dumps(values, sort_keys=True, separators=(",", ":"))
        a["imu_digest"] = digest
        f["t_offset"] = np.int64(t_offset_us)
        out = _Datasets(f, spec)
        t, t_e = span
        while t < t_e:
            t = min(t + BLOCK_NS, t_e)
            block = sim.advance_to(t)
            out.append({name: block[source.get(name, name)] for name in spec})
        stats = sim.stats()
        a["dropped_before_t_offset"] = np.int64(stats["dropped_before_t_offset"])
    return stats
