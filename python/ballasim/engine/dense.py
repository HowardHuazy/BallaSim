"""Dense frames of a run (PHASE1 "Dense API"): every pixel's centre-ray depth, instance, semantic and flow,
and its pixel-area irradiance per light group, at regular times or at given ones, for the viewer and for pairing
with the frames file (docs/dev/ENGINE.md "Dense frames"; docs/design/phase3/frames.md).

The frames come from the C++ renderer (``ballasim._render``'s ``Renderer.dense``) with the run's resolved render
values (run.json ``resolved``, docs/dev/P0_PARAMS.md §2.6) and backend, the code path of the per-event ground truth
(C2), on the bundle the run used (its content digest is checked). They are written to
``<run>/dense/<camera>.h5``, or into another directory (``out_dir``, ``dense --out``; e.g. when the run lies in a
directory the caller must not write, docs/design/phase3/ui.md §3.7):

- ``/t_ns`` (i64): frame times on the camera's sensor clock, clock_c(t_true) = t_true + the run's
  sensor.clock_offset_ns + llround(sensor.clock_drift_ppm 1e-6 t_true) (``ballasim._frames.sensor_time``, the readout
  formula without delay and quantum), the clock of the event file, so the viewer pairs frames and events directly;
  ``/t_true_ns`` on the bundle clock. Files written before the attribute ``t_ns_clock = "sensor"`` existed hold
  t_true + offset only (no drift; PHASE3 D-P3-8);
- ``/depth`` f32 (n, H, W) camera z in m (NaN: no geometry), ``/instance`` u32 and ``/semantic`` u16
  (n, H, W), ``/flow`` f32 (n, H, W, 2) in px/s, ``/irradiance`` f32 (n, H, W, G) per light group in renderer
  units, without the waveforms (C2: the pixel model applies them), and ``/shadow_factor`` f32 (n, H, W, G): per
  group the ratio of the irradiance to the irradiance with every proxy-light shadow factor set to 1 (scanned rooms
  with proxy lights, docs/dev/SCANS.md; 1 where that is 0, and everywhere in a scene without baked proxies), from
  the same pass as ``/irradiance`` (opt-in: not among the default channels);
- attributes ``format``, ``camera``, ``resolution = [W, H]``, ``dt_ns`` (the frame spacing, the viewer's pairing
  tolerance; with ``times_ns``, the largest gap between consecutive frames, or for one frame the bundle span's length,
  at least 1), ``clock_offset_ns``, ``clock_drift_ppm``, ``t_ns_clock`` (``"sensor"``) and ``light_groups`` (names,
  JSON).

One chunk per frame and channel, Blosc (zstd, bitshuffle); written as ``.<camera>.h5.partial`` and renamed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

from ballasim.scene.bundle import SceneBundle

from .scene import bundle_digest, load_bundle

log = logging.getLogger(__name__)

FORMAT = "ballasim.dense/1"
CHANNELS = ("depth", "instance", "semantic", "flow", "irradiance", "shadow_factor")
DEFAULT_CHANNELS = ("depth", "instance", "irradiance")
_DTYPES = {"depth": np.float32, "instance": np.uint32, "semantic": np.uint16, "flow": np.float32,
           "irradiance": np.float32, "shadow_factor": np.float32}


def make_renderer(bundle: SceneBundle, values: dict):
    """The renderer of a run's resolved values (``values``: {run: {...}, cameras: {name: {...}}}), built: every
    SceneSettings and RendererOptions value applied in C++ through the parameter rows, as the engine applies them
    (``run.device`` gpu: OptiX, cpu: Embree)."""
    from ballasim import _render

    renderer = _render.Renderer.from_values(bundle, values)
    renderer.build()
    return renderer


def recorded_values(run: dict) -> tuple[dict, str]:
    """The resolved values a run recorded in its run.json (``resolved``) and the renderer backend it ran (the GT's),
    which the values' ``run.device`` then names; a run rendered on OptiX falls back to Embree (``run.device`` cpu),
    with a warning, where OptiX is unavailable (its frames may differ from OptiX's in the last bits). Format-1 run
    directories (a ``config`` block, no ``resolved``) are refused until migrated (tools/config/convert_v1.py --runs),
    and so are migrated ones whose scene no longer gives the run's bundle (``resolved.bundle_match`` false)."""
    from ballasim import _render

    if "resolved" not in run:
        raise ValueError("a format-1 run directory (run.json has no resolved block): migrate it with python "
                         "tools/config/convert_v1.py --runs DIR (docs/dev/P0_PARAMS.md §3.8)")
    rec = run["resolved"]
    if rec.get("bundle_match") is False:
        raise ValueError("a migrated format-1 run whose scene no longer gives its bundle (resolved.bundle_match is "
                         "false): the scene has changed since the run")
    values = {"run": {k: e["value"] for k, e in rec["run"].items()},
              "cameras": {c: {k: e["value"] for k, e in v.items()} for c, v in rec["cameras"].items()}}
    backend = run["renderer"]["backend"]
    if backend == "optix" and "optix" not in _render.backends():
        log.warning("the run rendered on OptiX, which is unavailable here; its dense frames render on Embree")
        backend = "embree"
    values["run"]["run.device"] = "gpu" if backend == "optix" else "cpu"
    return values, backend


def frame_times(span_ns: tuple[int, int], dt_ns: int, t0_ns: int | None = None,
                t1_ns: int | None = None) -> np.ndarray:
    """``t0 + k dt`` inside ``[t0, t1)``, the run span by default (bundle clock)."""
    if dt_ns <= 0:
        raise ValueError("dt_ns must be positive")
    t0 = span_ns[0] if t0_ns is None else max(int(t0_ns), span_ns[0])
    t1 = span_ns[1] if t1_ns is None else min(int(t1_ns), span_ns[1])
    if t1 <= t0:
        raise ValueError(f"no frame time in [{t0}, {t1}) ns (run span {list(span_ns)})")
    return np.arange(t0, t1, dt_ns, dtype=np.int64)


def given_times(times_ns, span_ns: tuple[int, int]) -> np.ndarray:
    """``times_ns`` as a strictly increasing int64 array inside ``span_ns`` = [t_start, t_end] (the bundle's span);
    floats are refused, so no time is truncated."""
    times = np.asarray(times_ns)
    if times.ndim != 1 or times.size == 0 or not np.issubdtype(times.dtype, np.integer):
        raise ValueError("times_ns must be a non-empty one-dimensional integer array")
    times = times.astype(np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("times_ns must increase strictly")
    if times[0] < span_ns[0] or times[-1] > span_ns[1]:
        raise ValueError(f"times_ns must lie inside the bundle span {list(span_ns)} ns, got "
                         f"[{int(times[0])}, {int(times[-1])}]")
    return times


def export_dense(run_dir: str | Path, dt_ns: int | None = None, *, times_ns=None, t0_ns: int | None = None,
                 t1_ns: int | None = None, channels: Sequence[str] = DEFAULT_CHANNELS,
                 cameras: Sequence[str] | None = None, bundle: SceneBundle | None = None,
                 out_dir: str | Path | None = None) -> list[Path]:
    """Writes the dense frames of a run's cameras (default: those it simulated) into ``out_dir`` (default
    ``<run>/dense``, created with its parents); returns the files.

    The frames are at ``t0 + k dt_ns`` inside the run span (``frame_times``), or at ``times_ns`` (true clock, strictly
    increasing, inside the bundle span; e.g. a frames file's ``t_true_mid_ns``), exactly one of the two.
    ``bundle`` replaces loading the run's scene (it must have the run's digest)."""
    if (dt_ns is None) == (times_ns is None):
        raise ValueError("export_dense needs exactly one of dt_ns and times_ns")
    if times_ns is not None and (t0_ns is not None or t1_ns is not None):
        raise ValueError("t0_ns and t1_ns apply to dt_ns, not to times_ns")
    run_dir = Path(run_dir)
    doc = json.loads((run_dir / "run.json").read_text())
    try:
        values, _backend = recorded_values(doc)
    except ValueError as e:
        raise ValueError(f"{run_dir}: {e}") from None
    unknown = set(channels) - set(CHANNELS)
    if unknown or not channels:
        raise ValueError(f"channels must be a non-empty subset of {CHANNELS}, got {list(channels)}")
    if bundle is None:
        bundle = load_bundle(values["run"])
    if bundle_digest(bundle) != doc["bundle"]["digest"]:
        raise ValueError(f"{run_dir}: the scene has changed since the run (bundle digest differs)")
    names = [c.name for c in bundle.cameras]
    selected = list(cameras) if cameras is not None else list(values["cameras"])
    missing = [c for c in selected if c not in names]
    if missing:
        raise ValueError(f"cameras {missing} are not in the bundle (it has {names})")
    if times_ns is None:
        times = frame_times(tuple(doc["span_ns"]), dt_ns, t0_ns, t1_ns)
        spacing = int(dt_ns)
    else:
        times = given_times(times_ns, (bundle.t_start_ns, bundle.t_end_ns))
        # The viewer pairs a frame with event times within dt_ns; one frame covers the whole bundle span.
        spacing = (int(np.diff(times).max()) if times.size > 1
                   else max(int(bundle.t_end_ns) - int(bundle.t_start_ns), 1))
    renderer = make_renderer(bundle, values)
    out = run_dir / "dense" if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in selected:
        cam_index = names.index(name)
        cam_values = values["cameras"].get(name, {})
        offset = int(cam_values.get("sensor.clock_offset_ns", bundle.cameras[cam_index].clock_offset_ns))
        drift = float(cam_values.get("sensor.clock_drift_ppm", 0.0))
        paths.append(_write_camera(out / f"{name}.h5", renderer, bundle, cam_index, times, spacing, tuple(channels),
                                   offset, drift))
        log.info("dense frames of %s: %d from %d ns, spacing %d ns -> %s", name, len(times), times[0], spacing,
                 paths[-1])
    return paths


def _write_camera(path: Path, renderer, bundle: SceneBundle, cam_index: int, times: np.ndarray, dt_ns: int,
                  channels: tuple[str, ...], clock_offset_ns: int, clock_drift_ppm: float) -> Path:
    from ballasim import _frames

    cam = bundle.cameras[cam_index]
    h, w, g = cam.height, cam.width, len(bundle.light_groups)
    tmp = path.with_name(f".{path.name}.partial")
    shapes = {"depth": (h, w), "instance": (h, w), "semantic": (h, w), "flow": (h, w, 2), "irradiance": (h, w, g),
              "shadow_factor": (h, w, g)}
    comp = dict(hdf5plugin.Blosc(cname="zstd", clevel=1, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
    with h5py.File(tmp, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["camera"] = cam.name
        f.attrs["resolution"] = np.array([w, h], np.int64)
        f.attrs["dt_ns"] = np.int64(dt_ns)
        f.attrs["clock_offset_ns"] = np.int64(clock_offset_ns)
        f.attrs["clock_drift_ppm"] = np.float64(clock_drift_ppm)
        f.attrs["t_ns_clock"] = "sensor"
        f.attrs["light_groups"] = json.dumps([lg.name for lg in bundle.light_groups])
        f.create_dataset("t_ns", data=_frames.sensor_time(times, clock_offset_ns, clock_drift_ppm))
        f.create_dataset("t_true_ns", data=times)
        sets = {c: f.create_dataset(c, shape=(len(times), *shapes[c]), dtype=_DTYPES[c], chunks=(1, *shapes[c]),
                                    **comp) for c in channels}
        for k, t in enumerate(times):
            frame = renderer.dense(cam_index, int(t), channels=channels)
            for c, ds in sets.items():
                ds[k] = frame[c]
    tmp.replace(path)
    return path
