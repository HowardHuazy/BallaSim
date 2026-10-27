"""P1-M1 validation: re-render EVIMO2 depth and instance masks and compare with the release.

For selected GT frames the bodies of an EVIMO2 bundle are posed, merged in the camera frame and
ray-cast with Embree (one ray per pixel position, unprojected by the ``_core`` camera model):

- ``frame`` geometry: object poses T_C<-O from ``meta['frames'][k]``, exactly those EVIMO
  rasterized; this isolates meshes, Kabsch registration, camera model and ray casting;
- ``bundle`` geometry: poses from the bundle's fitted trajectories at the re-timed frame time;
  this adds re-timing, recomposition into the world frame, smoothing and holds.

EVIMO's rasterizer truncates projected vertices to int and fills triangles inclusively, so for
dense meshes it marks every pixel whose cell [x, x+1) x [y, y+1) the object touches (G2 §2.4:
centroid shift (-0.47, -0.47) px plus +0.36..0.62 px dilation). Three sampling variants:

- ``float``: rays through pixel centres (x, y), BallaSim's convention;
- ``shift``: rays through (x + 0.5, y + 0.5), removing the mean shift only;
- ``cell``: an n x n grid over the cell (``grid``, default 4), closest hit wins (z-buffer),
  emulating truncation and the inclusive fill; depth is then taken from the cell-centre ray when it
  sees the same object. A finer grid catches thinner slivers of the inclusive fill, which separates
  emulation error from geometry error on small objects.

IoU is reported pooled over all labelled pixels (dominated by the tabletop) and over the other
objects only: their mean per-object IoU, the lowest one, and their pooled IoU.

Only objects present in the frame's metadata are rendered (EVIMO skips absent ones) and marker
spheres are left out (EVIMO's GT has none). Depth errors are in mm on pixels where both sides
show the same object. Stillness: every trajectory is evaluated densely inside its holds and must
be bitwise constant there.

Run: ``python -m ballasim.evimo2.validate SEQUENCE_DIR [...] --out DIR``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ballasim import _core
from ballasim.scene.raycast import cell_grid, combine_min_depth, merge_in_camera, pixel_rays
from ballasim.scene.transforms import invert_transform, pose_to_transform, rotation_angle
from ballasim.traj.evaluate import evaluate

from .adapter import TABLETOP_ID, Evimo2Build, build_bundle
from .dataset import entry_pose

DEFAULT_OUT = "/data/ballasim_out/scene"
VARIANTS = ("float", "shift", "cell")


def hold_stillness(build: Evimo2Build, samples_per_hold: int = 257) -> dict:
    """Pose change and speeds inside every hold, evaluated by ``_core`` (ideally exactly 0).

    Positions and velocities come out bitwise constant/zero. Rotations carry round-off of the
    core's increment d_c = Log(R_{c-1}^T R_c), which is ~1e-17 rather than exactly 0 for identical
    control rotations (fused multiply-adds in the quaternion product), i.e. |omega| ~ 1e-14 rad/s."""
    out = {}
    names = {b.trajectory: b.name for b in build.bundle.bodies if b.trajectory is not None}
    names[build.bundle.cameras[0].rig_trajectory] = "rig"
    for i, traj in enumerate(build.bundle.trajectories):
        rec = {"holds": len(traj.holds), "held_s": float((traj.holds[:, 1] - traj.holds[:, 0]).sum() * 1e-9),
               "max_dp_m": 0.0, "max_v_mps": 0.0, "max_dtheta_rad": 0.0, "max_omega_radps": 0.0}
        for a, b in traj.holds:
            t = np.unique(np.linspace(a, b - 1, samples_per_hold).astype(np.int64))
            k = evaluate(traj, t)
            rec["max_dp_m"] = max(rec["max_dp_m"], float(np.abs(k["p"] - k["p"][0]).max()))
            rec["max_v_mps"] = max(rec["max_v_mps"], float(np.abs(k["v"]).max()))
            rec["max_dtheta_rad"] = max(rec["max_dtheta_rad"], float(rotation_angle(k["q"], k["q"][:1]).max()))
            rec["max_omega_radps"] = max(rec["max_omega_radps"], float(np.linalg.norm(k["omega"], axis=1).max()))
        out[names.get(i, f"trajectory_{i}")] = rec
    out["static_bodies"] = [b.name for b in build.bundle.bodies if b.trajectory is None]
    return out


class _Rays:
    """All ray sets of one camera, concatenated so each frame needs a single Embree call."""

    def __init__(self, cam, grid: int):
        sets = [(0.0, 0.0), (0.5, 0.5)] + list(cell_grid(grid))
        rays, valid = zip(*(pixel_rays(cam, o) for o in sets))
        self.n = cam.width * cam.height
        self.shape = (cam.height, cam.width)
        self.count = len(sets)
        self.dirs = np.ascontiguousarray(np.where(np.concatenate(valid)[:, None], np.concatenate(rays), 0.0),
                                         np.float32)
        self.valid = np.concatenate(valid)

    def cast(self, vertices, indices, prim_id):
        hits = _core.embree_intersect(vertices, indices, np.zeros_like(self.dirs), self.dirs)
        hit = self.valid & np.isfinite(hits["t"])
        depth = np.where(hit, hits["t"], 0.0).astype(np.float32)
        inst = np.zeros(len(hit), np.uint32)
        inst[hit] = prim_id[hits["prim"][hit]]
        d = depth.reshape(self.count, self.n)
        i = inst.reshape(self.count, self.n)
        cell_d, cell_i = combine_min_depth([(d[k], i[k]) for k in range(2, self.count)])
        # Cell depth: the cell-centre ray when it sees the same object, else the closest sub-sample.
        cell_d = np.where(i[1] == cell_i, d[1], cell_d)
        return {"float": (d[0], i[0]), "shift": (d[1], i[1]), "cell": (cell_d, cell_i)}


def _frame_geometry(build: Evimo2Build, frame: dict, ids: list[int]):
    meshes, faces, poses, out_ids = [], [], [], []
    for b in build.bundle.bodies:
        if b.id in ids:
            q, p = entry_pose(frame[str(b.id)])
            meshes.append(b.mesh.positions)
            faces.append(b.mesh.indices)
            poses.append(pose_to_transform(q, p) @ invert_transform(build.T_body_subject[b.id]))
            out_ids.append(b.id)
    return merge_in_camera(meshes, faces, poses, out_ids)


def _bundle_geometry(build: Evimo2Build, t_ns: int, ids: list[int]):
    bundle = build.bundle
    cam = bundle.cameras[0]
    k = evaluate(bundle.trajectories[cam.rig_trajectory], np.array([t_ns]))
    t_cw = invert_transform(pose_to_transform(k["q"][0], k["p"][0]) @ cam.T_rig_cam)
    meshes, faces, poses, out_ids = [], [], [], []
    for b in bundle.bodies:
        if b.id not in ids:
            continue
        if b.trajectory is None:
            t_wb = b.T_world_body
        else:
            kb = evaluate(bundle.trajectories[b.trajectory], np.array([t_ns]))
            t_wb = pose_to_transform(kb["q"][0], kb["p"][0])
        meshes.append(b.mesh.positions)
        faces.append(b.mesh.indices)
        poses.append(t_cw @ t_wb)
        out_ids.append(b.id)
    return merge_in_camera(meshes, faces, poses, out_ids)


def _compare(inst: np.ndarray, depth: np.ndarray, gt_inst: np.ndarray, gt_depth_mm: np.ndarray, ids):
    per = {}
    for i in ids:
        a, b = inst == i, gt_inst == i
        per[i] = (int(np.sum(a & b)), int(np.sum(a | b)))
    same = (inst == gt_inst) & (gt_inst > 0)
    pooled = (int(same.sum()), int(np.sum((inst > 0) | (gt_inst > 0))))
    both = same & (gt_depth_mm > 0) & (depth > 0)
    dz = depth[both].astype(np.float64) * 1e3 - gt_depth_mm[both].astype(np.float64)
    return per, pooled, dz


_PALETTE = np.array([[0, 0, 0]] + [[(37 * k) % 255, (91 * k + 60) % 255, (151 * k + 120) % 255] for k in range(1, 64)],
                    np.uint8)


def _colorize(inst: np.ndarray) -> np.ndarray:
    return _PALETTE[inst % len(_PALETTE)]


def _depth_image(d_mm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.clip((d_mm - lo) / max(hi - lo, 1e-9), 0, 1)
    img = (255 * (1 - v)).astype(np.uint8)
    img[d_mm <= 0] = 0
    return np.repeat(img[..., None], 3, axis=2)


def _panel(gt_inst, inst, gt_d, d_mm, title_path: Path):
    diff = np.zeros(gt_inst.shape + (3,), np.uint8)
    diff[(gt_inst > 0) & (inst != gt_inst)] = [255, 64, 64]     # released only / other object
    diff[(inst > 0) & (gt_inst == 0)] = [64, 160, 255]           # BallaSim only
    diff[(inst == gt_inst) & (gt_inst > 0)] = [70, 70, 70]
    valid = gt_d > 0
    lo, hi = (np.percentile(gt_d[valid], [1, 99]) if valid.any() else (0, 1))
    err = np.zeros_like(diff)
    both = valid & (d_mm > 0) & (inst == gt_inst)
    e = np.clip(np.abs(d_mm - gt_d) / 5.0, 0, 1)                 # 5 mm = full scale
    err[both] = (np.stack([e, 1 - e, np.zeros_like(e)], -1)[both] * 255).astype(np.uint8)
    top = np.concatenate([_colorize(gt_inst), _colorize(inst), diff], axis=1)
    bottom = np.concatenate([_depth_image(gt_d, lo, hi), _depth_image(d_mm, lo, hi), err], axis=1)
    Image.fromarray(np.concatenate([top, bottom], axis=0)).save(title_path)


def _object_ious(per: dict[int, list[int]], names: dict[int, str]) -> dict:
    """IoU over the objects other than the tabletop, which dominates the pooled number."""
    objs = {o: s for o, s in per.items() if o != TABLETOP_ID}
    if not objs:
        return {"mean": None, "min": None, "pooled": None}
    iou = {o: s[0] / max(s[1], 1) for o, s in objs.items()}
    worst = min(iou, key=iou.get)
    return {"mean": float(np.mean(list(iou.values()))), "min": {names[worst]: iou[worst]},
            "pooled": sum(s[0] for s in objs.values()) / max(sum(s[1] for s in objs.values()), 1)}


def validate_sequence(sequence_dir: str | Path, *, max_frames: int = 30, out_dir: str | Path = DEFAULT_OUT,
                      png_frames: int = 3, build: Evimo2Build | None = None, workers: int = 1,
                      grid: int = 4) -> dict:
    t0 = time.time()
    if build is None:
        from ballasim.engine.scene import evimo2_lab_options

        build = build_bundle(sequence_dir, evimo2_lab_options(sequence_dir, workers=workers, markers=False))
    seq = build.sequence
    cam = build.bundle.cameras[0]
    rays = _Rays(cam, grid)
    frames = {key: (t, f) for key, t, f in seq.gt_frames()}
    times = dict(build.gt_frame_times())
    keys = sorted(k for k in frames if build.bundle.t_start_ns <= times[k] <= build.bundle.t_end_ns)
    pick = keys if len(keys) <= max_frames else \
        [keys[round(i)] for i in np.linspace(0, len(keys) - 1, max_frames)]
    body_ids = {b.id for b in build.bundle.bodies}
    out = Path(out_dir) / f"{seq.camera}_{seq.category}_{seq.name}"
    out.mkdir(parents=True, exist_ok=True)
    acc = {g: {v: {"per": {}, "pooled": [0, 0], "dz": []} for v in VARIANTS} for g in ("frame", "bundle")}
    png_keys = {pick[round(i)] for i in np.linspace(0, len(pick) - 1, min(png_frames, len(pick)))}
    for key in pick:
        frame = frames[key][1]
        ids = sorted(int(k) for k in frame if k.isdigit() and int(k) in body_ids)
        gt_inst = (seq.mask(key) // 1000).astype(np.uint32)
        gt_depth = seq.depth(key)
        geometries = {"frame": _frame_geometry(build, frame, ids), "bundle": _bundle_geometry(build, times[key], ids)}
        for geom, merged in geometries.items():
            res = rays.cast(*merged)
            for v in VARIANTS:
                d, i = res[v]
                d, i = d.reshape(rays.shape), i.reshape(rays.shape)
                per, pooled, dz = _compare(i, d, gt_inst, gt_depth, ids)
                a = acc[geom][v]
                for oid, (inter, union) in per.items():
                    s = a["per"].setdefault(oid, [0, 0])
                    s[0] += inter
                    s[1] += union
                a["pooled"][0] += pooled[0]
                a["pooled"][1] += pooled[1]
                a["dz"].append(dz)
                if key in png_keys and v == "cell":
                    _panel(gt_inst, i, gt_depth.astype(np.float64), d.astype(np.float64) * 1e3,
                           out / f"{geom}_frame{key:05d}_cell.png")
                if key in png_keys and v == "float" and geom == "frame":
                    _panel(gt_inst, i, gt_depth.astype(np.float64), d.astype(np.float64) * 1e3,
                           out / f"{geom}_frame{key:05d}_float.png")
    names = {b.id: b.name for b in build.bundle.bodies}
    summary = {}
    for geom, variants in acc.items():
        summary[geom] = {}
        for v, a in variants.items():
            dz = np.concatenate(a["dz"]) if a["dz"] else np.zeros(0)
            adz = np.abs(dz)
            summary[geom][v] = {
                "pooled_iou": a["pooled"][0] / max(a["pooled"][1], 1),
                "objects_without_table_iou": _object_ious(a["per"], names),
                "per_object_iou": {names[o]: s[0] / max(s[1], 1) for o, s in sorted(a["per"].items())},
                "depth_mm": {"pixels": len(dz), "median_abs": float(np.median(adz)) if len(dz) else None,
                             "p90_abs": float(np.percentile(adz, 90)) if len(dz) else None,
                             "p99_abs": float(np.percentile(adz, 99)) if len(dz) else None,
                             "mean": float(dz.mean()) if len(dz) else None,
                             "frac_below_1mm": float(np.mean(adz < 1.0)) if len(dz) else None,
                             "frac_below_5mm": float(np.mean(adz < 5.0)) if len(dz) else None}}
    report = {"sequence": str(seq.directory), "camera": seq.camera, "frames": pick, "num_frames": len(pick),
              "cell_grid": grid, "summary": summary, "stillness": hold_stillness(build),
              "bundle_meta": build.bundle.meta, "seconds": time.time() - t0}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sequences", nargs="+")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1, help="processes for the trajectory fits")
    ap.add_argument("--cell-grid", type=int, default=4, help="n for the n x n sub-samples of the cell variant")
    args = ap.parse_args(argv)
    for s in args.sequences:
        r = validate_sequence(s, max_frames=args.max_frames, out_dir=args.out, workers=args.workers,
                              grid=args.cell_grid)
        print(f"== {r['sequence']} ({r['num_frames']} frames, {r['seconds']:.0f} s)")
        for geom, variants in r["summary"].items():
            for v, m in variants.items():
                o = m["objects_without_table_iou"]
                objs = "" if o["mean"] is None else f" (objects: mean {o['mean']:.4f}, pooled {o['pooled']:.4f})"
                print(f"  {geom:6s} {v:5s} pooled IoU {m['pooled_iou']:.4f}{objs}  depth |dz| median "
                      f"{m['depth_mm']['median_abs']:.2f} mm p90 {m['depth_mm']['p90_abs']:.2f} mm  "
                      + " ".join(f"{k}:{x:.3f}" for k, x in m["per_object_iou"].items()))
        held = {k: v for k, v in r["stillness"].items() if isinstance(v, dict) and v["holds"]}
        print("  holds:", {k: f"{v['held_s']:.2f}s dp={v['max_dp_m']:.0e} v={v['max_v_mps']:.0e} "
                              f"dth={v['max_dtheta_rad']:.0e} w={v['max_omega_radps']:.0e}" for k, v in held.items()},
              "| static bodies:", r["stillness"]["static_bodies"])


if __name__ == "__main__":
    main()
