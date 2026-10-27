"""BallaSim's own frames of a finished run (docs/design/phase3/frames.md §3.9, §4.4, §5.1 "reference"):
``export_reference(run_dir, cameras=None, sets=())`` writes ``<run>/frames/<camera>.h5`` (``ballasim.frames/1``,
mode ``reference``) for the cameras of ``run.select`` and ``run.frames_select`` with BallaSim's exposure reference
(``ballasim._frames.ExposureReference``, C++): the exposure integral of the run's own N-sample irradiance over each
read-row window, the waveforms applied exactly. Command line: ``python -m ballasim frames reference RUN_DIR [--camera
C ...] [--set cameras.<c>.frames.K=V ...]``.

The frames values are the run's own layers plus the ``--set`` items (:mod:`ballasim.frames.resolve`), recorded in
``<run>/frames/resolved.json``. A camera named by ``--camera`` whose ``frames.mode`` is ``off`` gets reference
frames (the command makes them); one whose mode is ``matched`` or ``full`` is refused (FRM-VALUE: its file is
Blender's). Without ``--camera`` the cameras are those whose ``frames.mode`` is ``reference``.

One frame is one sub-render: ``sub_offsets`` is 0..n, the sub-window is the frame's, ``m_mean_sub`` equals
``m_mean`` and the factorization test runs at ``n_sub = 1``. ``mono`` is F_ref (renderer units, waveform-weighted), also stored as
``reference`` when ``frames.outputs`` names it; ``frames.outputs`` defaults to ``[mono]`` and may add ``reference`` and
``aps`` (the DAVIS-like APS model, :mod:`ballasim.frames.aps`, run on the stored ``mono``; it needs ``mono`` and an
events camera). ``rgb`` and ``groups`` are Blender's and refused. The uniform timeline starts at ``frames.t0_ns`` (null:
the run's start) with period ``frames.period_ns`` (null: round_half_even(1e9 / frames.fps)) and keeps the frames whose
exposure lies inside the bundle's span; the exposure is ``frames.exposure_ns`` (null: 10 ms). A Blender source's null
exposure is its authored shutter and its null timeline ``blender`` (from the timeline record, which this command does
not read, and the Blender session): such a run takes ``frames.exposure_ns`` and ``frames.timeline=uniform`` explicitly
(FRM-VALUE). Stamps ``t_*_ns`` are on the camera's sensor clock (offset and drift as resolved), the ``t_true_*_ns`` on
the bundle clock. ``frames_digest_base`` (§3.6) covers the format, the bundle digest, the camera's sensor-stage digest (a
frames-only camera: the render-stage digest, its clock offset and the resolved ``t0_ns``, which the render stage does not
hold when ``frames.t0_ns`` is null) and the canonical frames values; ``frames_digest`` adds the APS parameters (none
without APS output).

Every refusal that needs no render (FRM-VALUE, FRM-APS-SCALE: ``aps`` in the outputs of a camera the run did not
simulate for events) comes before the first camera is rendered, so that a refusal leaves no file behind.

A display sequence ``<run>/frames/<camera>/display/%05d.png`` (8-bit, a fixed sRGB curve of x / p99 of the first
frame) and ``index.json`` (``{frame: [k, t_true_mid_ns, t_mid_ns]}``) are for display only.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import aps as aps_module
from . import resolve as fr
from .io import FORMAT, write_frames

log = logging.getLogger(__name__)

DEFAULT_EXPOSURE_NS = 10_000_000       # frames.md §4.1: a source without Blender's authored shutter
UNITS = "C2 renderer irradiance, exposure-integrated and waveform-weighted (BallaSim's reference)"


def _requested(res: fr.FramesResolution, cameras: Iterable[str] | None) -> list[str]:
    """The cameras to write: ``cameras`` (each in a selection: FRM-CAMERA), else those with frames.mode reference."""
    if cameras is not None:
        names = []
        for name in cameras:
            res.camera(name)
            if name not in names:
                names.append(name)
        return names
    names = [c["name"] for c in res.cameras if c["values"]["frames.mode"]["value"] == "reference"]
    if not names:
        raise fr.refusal("FRM-CAMERA", "no camera has frames.mode: reference; name one with --camera, or set "
                                       "cameras.<camera>.frames.mode=reference")
    return names


def _camera_digest(doc: dict, c: dict, t0_ns: int) -> str:
    """frames_digest_base's camera part (§3.6): an events camera's sensor-stage digest; a frames-only camera has none,
    and takes the run-wide render-stage digest with its clock offset and the first frame's start ``t0_ns`` (the
    resolved ``frames.t0_ns``: a null one is the run's start, which the render stage does not hold)."""
    digests = doc["stage_digests"]
    if c["events"]:
        return digests[c["name"]]["sensor"]
    render = next(iter(digests.values()))["render"]
    return f"{render}\nclock_offset_ns={c['clock_offset_ns']}\nt0_ns={t0_ns}"


def _srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def write_display(out: Path, mono: np.ndarray, t_true_mid: np.ndarray, t_mid: np.ndarray) -> Path:
    """The display sequence of ``mono`` (n, H, W): 8-bit PNGs of a fixed sRGB curve of x / p99 of the first frame, and
    ``index.json`` (``{frame: [k, t_true_mid_ns, t_mid_ns]}``), in ``out/display/``."""
    from ballasim.viz.video import write_png

    folder = out / "display"
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("*.png"):         # a rerun with fewer frames leaves none of the earlier ones
        stale.unlink()
    finite = mono[0][np.isfinite(mono[0])]
    p99 = float(np.percentile(finite, 99)) if finite.size else 0.0
    scale = p99 if p99 > 0 else (float(finite.max()) if finite.size and finite.max() > 0 else 1.0)
    index = {}
    for k in range(mono.shape[0]):
        grey = np.rint(255.0 * _srgb(mono[k] / scale)).astype(np.uint8)
        write_png(folder / f"{k:05d}.png", np.repeat(grey[..., None], 3, axis=2))
        index[f"{k:05d}"] = [k, int(t_true_mid[k]), int(t_mid[k])]
    (folder / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return folder


INT32_MAX, INT64_MAX = 2**31 - 1, 2**63 - 1


def _settings(res: fr.FramesResolution, c: dict) -> dict:
    """The settings of one camera's reference frames from its resolved values, every refusal of the command that does
    not need a render (FRM-VALUE, FRM-APS-SCALE): :func:`export_reference` takes them for every camera before the first
    render, so that a refusal leaves no file behind."""
    name = c["name"]
    v = {k: e["value"] for k, e in c["values"].items()}
    cam = res.bundle.cameras[c["index"]]
    roi = [int(x) for x in v["frames.roi_px"]]
    outputs = list(v["frames.outputs"]) if v["frames.outputs"] is not None else ["mono"]
    if "aps" in outputs and "mono" not in outputs:
        raise fr.refusal("FRM-VALUE", f"cameras.{name}.frames.outputs: aps needs mono (it runs on the stored mono)")
    if "aps" in outputs:
        aps_module.photocurrent(res, name)       # FRM-APS-SCALE: a frames-only camera has no photocurrent scale
    if v["frames.timeline"] == "blender":
        raise fr.refusal("FRM-VALUE", f"cameras.{name}.frames.timeline = 'blender': the frames of a Blender timeline "
                                      f"come from the Blender session (frames plan); reference frames use 'uniform'")
    if res.spec.inputs.get("source_kind") == "blender":
        # A Blender source's null timeline is 'blender' and its null exposure the authored shutter (timeline record,
        # frames plan): this command does not read that record, so it takes explicit values.
        if v["frames.timeline"] is None:
            raise fr.refusal("FRM-VALUE", f"cameras.{name}.frames.timeline is null, which is 'blender' for a Blender "
                                          f"source; reference frames use 'uniform': set cameras.{name}.frames.timeline"
                                          f"=uniform")
        if v["frames.exposure_ns"] is None:
            raise fr.refusal("FRM-VALUE", f"cameras.{name}.frames.exposure_ns is null, which is the authored Blender "
                                          f"shutter for a Blender source, a value frames reference does not read: set "
                                          f"cameras.{name}.frames.exposure_ns")
    exposure_ns = int(v["frames.exposure_ns"]) if v["frames.exposure_ns"] is not None else DEFAULT_EXPOSURE_NS
    exposure_source = "set" if v["frames.exposure_ns"] is not None else "default"
    from_fps = v["frames.period_ns"] is None
    if from_fps and not math.isfinite(1e9 / v["frames.fps"]):
        raise fr.refusal("FRM-VALUE", f"cameras.{name}.frames.fps = {v['frames.fps']}: the period exceeds 2^63 ns")
    period_ns = int(round(1e9 / v["frames.fps"])) if from_fps else int(v["frames.period_ns"])
    if not 0 < period_ns <= INT64_MAX:
        what = f"frames.fps = {v['frames.fps']}: the period rounds to {period_ns} ns" if from_fps else \
               f"frames.period_ns = {period_ns}"
        raise fr.refusal("FRM-VALUE", f"cameras.{name}.{what}, outside (0, 2^63) ns")
    count = v["frames.count"]
    if count is not None and int(count) > INT32_MAX:
        raise fr.refusal("FRM-VALUE", f"cameras.{name}.frames.count = {count}: at most 2^31 - 1 frames")
    t0_ns = int(v["frames.t0_ns"]) if v["frames.t0_ns"] is not None else int(res.doc["span_ns"][0])
    shutter = v["frames.shutter"]
    return {"width": int(cam.width), "height": int(cam.height), "roi": roi, "outputs": outputs,
            "exposure_ns": exposure_ns, "exposure_source": exposure_source, "period_ns": period_ns,
            "count": None if count is None else int(count), "t0_ns": t0_ns, "shutter": shutter,
            "readout": v["frames.readout"], "row_dt_ns": int(v["frames.row_dt_ns"]) if shutter == "rolling" else 0,
            "h_px": float(v["frames.ref.h_px"]), "probe_tol": float(v["frames.ref.probe_tol"]),
            "k_max": int(v["frames.ref.k_max"])}


def _write_reference(res: fr.FramesResolution, c: dict, s: dict, *, threads: int) -> Path:
    from ballasim import _frames

    doc, bundle = res.doc, res.bundle
    name = c["name"]
    width, height, roi, outputs = s["width"], s["height"], s["roi"], s["outputs"]
    exposure_ns, period_ns, t0_ns = s["exposure_ns"], s["period_ns"], s["t0_ns"]
    shutter, readout, row_dt_ns = s["shutter"], s["readout"], s["row_dt_ns"]
    sched = _frames.Schedule.uniform(height, exposure_ns, t0_ns, period_ns, s["count"],
                                     (int(bundle.t_start_ns), int(bundle.t_end_ns)), row_dt_ns=row_dt_ns,
                                     shutter=shutter, readout=readout)
    times = sched.times()
    n = sched.size
    ref = _frames.ExposureReference(bundle, res.rvalues, name, roi=roi)
    h, w, g = roi[3] - roi[1], roi[2] - roi[0], ref.num_groups
    groups = [lg.name for lg in bundle.light_groups] if len(bundle.light_groups) == g else ["default"]
    mono = np.empty((n, h, w), np.float32)
    m_mean = np.empty((n, h, g), np.float64)
    stat = {k: np.empty(n, np.int32 if k.startswith("k_") else np.float32)
            for k in ("k_ref", "k_ref_mask", "probe_p999", "probe_max", "e_fact_p99")}
    warnings = [[code, message] for code, message in sched.warnings]
    for k in range(n):
        f = ref.frame(sched, k, h_px=s["h_px"], probe_tol=s["probe_tol"], k_max=s["k_max"], n_sub=1)
        mono[k], m_mean[k] = f["mono"], f["m_mean"]
        stat["k_ref"][k], stat["k_ref_mask"][k] = f["k_used"], f["k_mask"]
        stat["probe_p999"][k], stat["probe_max"][k] = f["probe_p999"], f["probe_max"]
        stat["e_fact_p99"][k] = f["fact_p99"]
        warnings += [[code, f"frame {k}: {message}"] for code, message in f["warnings"]]
        log.info("reference frame %d/%d of %s: K %d (%d on %d probe pixels)", k + 1, n, name, f["k_used"], f["k_mask"],
                 f["mask_px"])
    soe, eoe, mid = times["soe"], times["eoe"], times["mid"]
    offset, drift = int(c["clock_offset_ns"]), float(c["clock_drift_ppm"])
    s_soe, s_eoe = _frames.sensor_time(soe, offset, drift), _frames.sensor_time(eoe, offset, drift)
    s_mid = s_soe + (s_eoe - s_soe) // 2
    data = {"t_soe_ns": s_soe, "t_eoe_ns": s_eoe, "t_mid_ns": s_mid, "t_true_soe_ns": soe, "t_true_eoe_ns": eoe,
            "t_true_mid_ns": mid, "t_req_true_mid_ns": mid, "sub_offsets": np.arange(n + 1, dtype=np.int64),
            "sub_t_true_soe_ns": soe, "sub_t_true_eoe_ns": eoe, "sub_blender_c": np.full(n, np.nan), **stat,
            "m_mean": m_mean, "m_mean_sub": m_mean}
    if "mono" in outputs:
        data["mono"] = mono
    if "reference" in outputs:
        data["reference"] = mono
    values = res.rvalues
    cam_values = next(iter(values["cameras"].values()))   # render-stage values are run-wide (D10)
    base = _frames.frames_digest_base(FORMAT, doc["bundle"]["digest"], _camera_digest(doc, c, t0_ns),
                                      c["digest_values"])
    attrs = {"camera": name, "resolution": [width, height], "roi_px": roi, "mode": "reference", "light_groups": groups,
             "route": "none", "route_bound": {}, "supersample": 1, "shutter": shutter, "readout": readout,
             "timeline": "uniform", "row_dt_ns": row_dt_ns, "exposure_ns": exposure_ns,
             "exposure_source": s["exposure_source"], "period_ns": period_ns, "clock_offset_ns": offset,
             "clock_drift_ppm": drift, "dropped_frames": list(sched.dropped),
             "rgb_weights": cam_values["sensor.rgb_weights"], "aperture_px": cam_values["sensor.aperture_px"],
             "blur_sigma_px": cam_values["optics.blur_sigma_px"],
             "truncation_sigmas": values["run"]["accuracy.psf_truncation_sigmas"],
             "n_subsamples": values["run"]["accuracy.n_subsamples"], "units": UNITS, "plan_digest": "",
             "bundle_digest": doc["bundle"]["digest"], "frames_digest_base": base,
             "frames_digest": _frames.frames_digest(base, None), "warnings": warnings}
    path = write_frames(res.run_dir / "frames" / f"{name}.h5", data, attrs)
    write_display(res.run_dir / "frames" / name, mono, mid, s_mid)
    if "aps" in outputs:
        aps_module.apply(res, name, threads=threads)
    return path


def export_reference(run_dir: str | Path, cameras: Iterable[str] | None = None, sets: Iterable[str] = (), *,
                     bundle=None, threads: int = 0) -> list[Path]:
    """Writes the reference frames of ``cameras`` (default: those whose frames.mode is reference) of a finished run and
    returns their files. ``sets``: ``--set`` items (cameras.<c>.frames.K=V, cameras.default.frames.K=V,
    execution.frames_device, execution.frames_texture_mb, run.frames_select); ``bundle`` replaces loading the run's
    scene (it must have the run's digest); ``threads`` bounds the APS workers (<= 0: the hardware's)."""
    run_dir = Path(run_dir)
    doc = fr.read_run(run_dir)
    sets = list(sets)
    res = fr.resolve(run_dir, sets, bundle=bundle, doc=doc)
    names = _requested(res, None if cameras is None else list(cameras))
    # The command makes the reference frames of the cameras it writes: their mode off becomes reference, below the
    # --set items (which can still set another mode, and then refuse).
    off = {f"cameras.{n}.frames.mode": "reference" for n in names
           if res.camera(n)["values"]["frames.mode"]["value"] == "off"}
    if off:
        res = fr.resolve(run_dir, sets, bundle=res.bundle, implicit=off, doc=doc)
    for n in names:
        mode = res.camera(n)["values"]["frames.mode"]["value"]
        if mode != "reference":
            raise fr.refusal("FRM-VALUE", f"cameras.{n}.frames.mode = '{mode}': frames reference writes the reference "
                                          f"frames of a camera, whose file in that mode is Blender's (frames plan, "
                                          f"frames ingest)")
    settings = [_settings(res, res.camera(n)) for n in names]       # every refusal before the first file
    for w in res.warnings:
        log.warning("%s", w)
    fr.write_resolved(res)
    return [_write_reference(res, res.camera(n), s, threads=threads) for n, s in zip(names, settings)]


def main(args) -> int:
    """``frames reference``: prints the files written."""
    for path in export_reference(args.run_dir, args.camera or None, args.set, threads=args.threads):
        print(path)
    return 0
