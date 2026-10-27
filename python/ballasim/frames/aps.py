"""APS frames from a stored frames file (docs/design/phase3/frames.md §4.10, §3.9): ``aps(run_dir, camera, sets=())``
re-runs the DAVIS-like APS model (C++: ``ballasim._frames.aps_frames``, the entry point ingest also calls) on the
stored ``mono`` without re-rendering, and rewrites ``aps_dn``, the ``aps`` attribute, the file's FRM-APS-* warnings
and ``frames_digest``. Command line: ``python -m ballasim frames aps RUN_DIR --camera C [--set
cameras.<c>.frames.aps.K=V ...]``.

Inputs, as the run simulated the camera (run.json): its photocurrent scale ``cameras[].illuminance_scale`` (a camera
the run did not simulate has none: FRM-APS-SCALE), ``optics.veiling_glare``, and its gain map when
``optics.vignetting`` resolves on (the engine's map: the camera's ``optics.vignetting_law`` and
``optics.relative_illumination`` on the run's bundle camera, digest-checked); the bundle
camera index keys the draws and ``run.sensor_seed`` is the default seed. The frames file gives the ROI, the full
resolution, the exposure, the row delay, the shutter, the readout and each frame's ``t_true_soe_ns``.

APS values: the ``frames.aps.*`` rows (§3.2), resolved through :func:`ballasim.frames.resolve.resolve` as the run's
layers with the ``--set`` items on top (``cameras.default.frames.aps.K=V``, then ``cameras.<camera>.frames.aps.K=V``;
any ``cameras.<c>.frames.K``, ``execution.frames_*`` and ``run.frames_select`` item is accepted, and only the APS rows
are read). A camera the bundle lacks is FRM-CAMERA, any other key FRM-SET-SCOPE, a value outside its row FRM-VALUE; a
camera that only gets frames (``run.frames_select``) has no photocurrent scale (FRM-APS-SCALE). The ``aps`` attribute
records the resolved parameters, all nine resolved values, the ``--set`` items, the photometry inputs and the per-frame
veil means.

``frames_digest`` = ``_frames.frames_digest(frames_digest_base, params)`` (§3.6): the writer records the base, over
§3.6's parts without the APS. A file without a base (written before FR-1b's writer) gets no ``frames_digest``: the
file alone lacks the frames values the base covers, and one it had no longer holds for the new APS output, so it is
removed (caches then miss rather than serve a stale file).
"""

from __future__ import annotations

from pathlib import Path

from . import resolve as fr
from .io import FramesFile, rewrite_frames

_TIMING_ATTRS = ("exposure_ns", "row_dt_ns", "shutter", "readout")
FULL_MODE_NOTE = "full mode: the APS sees Blender's global illumination, the DVS does not"
_APS_PREFIX = "frames.aps."


def aps_values(res: fr.FramesResolution, camera: str) -> dict:
    """The ``frames.aps.*`` values of ``camera`` as {key below frames.aps: value} (the nine rows, resolved)."""
    return {k[len(_APS_PREFIX):]: v for k, v in res.values(camera).items() if k.startswith(_APS_PREFIX)}


def aps(run_dir: str | Path, camera: str, sets=(), *, bundle=None, threads: int = 0) -> Path:
    """Rewrites the APS output of ``<run_dir>/frames/<camera>.h5`` from its ``mono`` and returns the path. ``bundle``
    replaces loading the run's scene for the gain map (it must have the run's digest)."""
    return apply(fr.resolve(run_dir, sets, bundle=bundle), camera, threads=threads)


def photocurrent(res: fr.FramesResolution, camera: str) -> tuple[dict, dict]:
    """``camera``'s run.json summary (``cameras[]``: index, illuminance_scale) and resolved values (``resolved.cameras``)
    as the run simulated it; FRM-APS-SCALE when it has no photocurrent scale (a camera of ``run.frames_select``, or one
    the run recorded none for). ``frames reference`` asks before it renders, :func:`apply` before it models."""
    summary = next((c for c in res.doc.get("cameras", []) if c.get("name") == camera), None)
    record = res.doc.get("resolved", {}).get("cameras", {}).get(camera)
    if not res.camera(camera)["events"] or summary is None or record is None or not summary.get("illuminance_scale"):
        raise fr.refusal("FRM-APS-SCALE", f"camera {camera}: the run recorded no photocurrent scale for it (run.json "
                                          f"cameras[].illuminance_scale; frames-only cameras have none)")
    return summary, record


def apply(res: fr.FramesResolution, camera: str, *, threads: int = 0) -> Path:
    """:func:`aps` on resolved frames values (``frames reference`` calls it for a camera whose outputs name aps)."""
    from ballasim import _frames
    from ballasim.engine.scene import bundle_digest

    run_dir, doc = res.run_dir, res.doc
    values = aps_values(res, camera)
    path = run_dir / "frames" / f"{camera}.h5"
    with FramesFile(path) as f:
        attrs = dict(f.attrs)
        missing = [k for k in ("mono", "t_true_soe_ns") if k not in f.names] + \
                  [k for k in _TIMING_ATTRS if k not in attrs]
        if missing:
            raise ValueError(f"{path}: the APS needs {missing}, which the file lacks")
        mono, soe = f.read("mono"), f.read("t_true_soe_ns")
    if attrs.get("bundle_digest", doc["bundle"]["digest"]) != doc["bundle"]["digest"]:
        raise ValueError(f"{path}: its bundle digest differs from the run's ({run_dir / 'run.json'})")
    summary, record = photocurrent(res, camera)
    cam = {k: e["value"] for k, e in record.items()}
    seed = doc["resolved"]["run"]["run.sensor_seed"]["value"]
    roi = [int(v) for v in attrs["roi_px"]]
    gain = None
    if _frames.vignetting_on(cam.get("optics.vignetting"), cam.get("photometry.mode", "relative")):
        if bundle_digest(res.bundle) != doc["bundle"]["digest"]:
            raise ValueError(f"{run_dir}: the scene has changed since the run (bundle digest differs)")
        gain = _frames.vignetting_gain(res.bundle, res.rvalues, summary["index"], roi,
                                       law=cam.get("optics.vignetting_law", "cos4"),
                                       relative_illumination=list(cam.get("optics.relative_illumination") or []))
    veil = float(cam.get("optics.veiling_glare") or 0.0)
    out = _frames.aps_frames(mono, soe, run_sensor_seed=seed, camera=summary["index"],
                             resolution=tuple(int(v) for v in attrs["resolution"]), roi=roi,
                             exposure_ns=int(attrs["exposure_ns"]), illuminance_scale=summary["illuminance_scale"],
                             values=values, row_dt_ns=int(attrs["row_dt_ns"]), shutter=attrs["shutter"],
                             readout=attrs["readout"], veiling_glare=veil, gain=gain, threads=threads)
    base = attrs.get("frames_digest_base")
    record = {**out["params"], "values": values, "sets": list(res.sets),
              "illuminance_scale_fA": summary["illuminance_scale"], "vignetting": gain is not None,
              "veiling_glare": veil, "veil_mean": out["veil_mean"].tolist(),
              "normal_fallbacks": out["normal_fallbacks"]}
    if attrs.get("mode") == "full":
        record["note"] = FULL_MODE_NOTE
    warnings = [list(w) for w in attrs.get("warnings", []) if not str(w[0]).startswith("FRM-APS")]
    digest = _frames.frames_digest(base, out["params"]) if base else None  # None removes a stale one
    return rewrite_frames(path, {"aps_dn": out["dn"]},
                          {"aps": record, "warnings": warnings + [list(w) for w in out["warnings"]],
                           "frames_digest": digest})


def main(args) -> int:
    """``frames aps``: prints the rewritten file."""
    print(aps(args.run_dir, args.camera, args.set, threads=args.threads))
    return 0
