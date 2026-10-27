"""Reference dense depth and instance frames of a scene bundle by CPU ray casting.

Bodies are posed at one instant with the bundle's trajectories (evaluated by ``ballasim._core``),
merged into one mesh in the camera frame and intersected with Embree (``_core.embree_intersect``).
Pixel rays come from the camera model's exact unprojection in ``_core`` (Newton on the monotone
radtan branch, so geometry outside the injective domain never folds back into the image, G3 t13).
Rays are (x_n, y_n, 1), so the hit distance is the camera z-depth.

This is a validation tool (P1-M1) and a reference for the renderer's ``dense`` API, not a fast
path: Embree's BVH is rebuilt per call.
"""

from __future__ import annotations

import numpy as np

from ballasim import _core
from ballasim.traj.evaluate import evaluate

from .bundle import Camera, SceneBundle
from .transforms import invert_transform, pose_to_transform


def cell_grid(n: int) -> tuple[tuple[float, float], ...]:
    """Sub-pixel offsets of an n x n grid of cell centres over the pixel cell [x, x+1) x [y, y+1)."""
    return tuple(((i + 0.5) / n, (j + 0.5) / n) for j in range(n) for i in range(n))


def camera_model(cam: Camera, support_px: float = 1.0):
    """The ``_core`` camera of a C1 camera; its injective domain covers pixel centres +- support."""
    if cam.model == "radtan":
        k1, k2, p1, p2, k3 = (float(x) for x in cam.dist)
        return _core.RadtanCamera(cam.fx, cam.fy, cam.cx, cam.cy, k1, k2, p1, p2, k3,
                                  width=cam.width, height=cam.height, support_px=support_px)
    k1, k2, k3, k4 = (float(x) for x in cam.dist)
    return _core.KannalaBrandtCamera(cam.fx, cam.fy, cam.cx, cam.cy, k1, k2, k3, k4,
                                     width=cam.width, height=cam.height, support_px=support_px)


def pixel_rays(cam: Camera, offset: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Rays through pixel positions (x + ox, y + oy), row-major (H*W, 3), plus validity."""
    model = camera_model(cam, support_px=max(1.0, abs(offset[0]) + 0.5, abs(offset[1]) + 0.5))
    ys, xs = np.mgrid[0:cam.height, 0:cam.width]
    uv = np.stack([xs.ravel() + offset[0], ys.ravel() + offset[1]], axis=1).astype(np.float64)
    rays, valid = model.unproject(uv)
    if cam.model == "kb":   # KB unprojection returns unit rays; scale to z = 1 for z-depth
        rays = rays / np.where(np.abs(rays[:, 2:3]) > 1e-12, rays[:, 2:3], np.nan)
    return rays, valid


def body_poses(bundle: SceneBundle, t_ns: int) -> list[np.ndarray]:
    """T_world_body(t) for every body (static bodies: their fixed pose)."""
    out = []
    for b in bundle.bodies:
        if b.trajectory is None:
            out.append(b.T_world_body)
        else:
            k = evaluate(bundle.trajectories[b.trajectory], np.array([t_ns]))
            out.append(pose_to_transform(k["q"][0], k["p"][0]))
    return out


def camera_pose(bundle: SceneBundle, cam_index: int, t_ns: int) -> np.ndarray:
    """T_world_cam(t) = T_world_rig(t) T_rig_cam (M2 C12)."""
    cam = bundle.cameras[cam_index]
    k = evaluate(bundle.trajectories[cam.rig_trajectory], np.array([t_ns]))
    return pose_to_transform(k["q"][0], k["p"][0]) @ cam.T_rig_cam


def merge_in_camera(meshes: list[np.ndarray], indices: list[np.ndarray], t_cam_body: list[np.ndarray],
                    ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posed vertices (fp64 transform, then float32 camera-relative), indices and per-triangle ids."""
    verts, tris, prim_id = [], [], []
    offset = 0
    for x, f, t, i in zip(meshes, indices, t_cam_body, ids):
        verts.append((x.astype(np.float64) @ t[:3, :3].T + t[:3, 3]).astype(np.float32))
        tris.append(f.astype(np.uint32) + np.uint32(offset))
        prim_id.append(np.full(len(f), i, np.uint32))
        offset += len(x)
    return np.concatenate(verts), np.concatenate(tris), np.concatenate(prim_id)


def cast(vertices: np.ndarray, indices: np.ndarray, prim_id: np.ndarray, rays: np.ndarray,
         valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest hits from the camera centre: (z-depth float32, 0 on a miss; instance uint32)."""
    n = len(rays)
    dirs = np.where(valid[:, None], rays, 0.0).astype(np.float32)
    hits = _core.embree_intersect(vertices, indices, np.zeros((n, 3), np.float32), np.ascontiguousarray(dirs))
    hit = valid & np.isfinite(hits["t"])
    depth = np.where(hit, hits["t"], 0.0).astype(np.float32)
    inst = np.zeros(n, np.uint32)
    inst[hit] = prim_id[hits["prim"][hit]]
    return depth, inst


def combine_min_depth(samples: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel, the closest hit over several sub-sample casts (z-buffer over the cell)."""
    depth = np.stack([np.where(d > 0, d, np.inf) for d, _ in samples])
    inst = np.stack([i for _, i in samples])
    k = np.argmin(depth, axis=0)
    best = np.take_along_axis(depth, k[None], 0)[0]
    return np.where(np.isfinite(best), best, 0.0).astype(np.float32), np.take_along_axis(inst, k[None], 0)[0]


def render_dense(bundle: SceneBundle, cam_index: int, t_ns: int, *,
                 offsets: tuple[tuple[float, float], ...] = ((0.0, 0.0),),
                 bodies: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dense (depth m, instance) frames (H, W) of camera ``cam_index`` at time t_ns.

    One offset is the BallaSim convention (pixel centres at integer coordinates); several offsets
    are combined per pixel by minimum depth (e.g. ``cell_grid(4)`` approximates "the object covers
    any part of the pixel cell")."""
    cam = bundle.cameras[cam_index]
    t_wc = camera_pose(bundle, cam_index, t_ns)
    t_cw = invert_transform(t_wc)
    poses = body_poses(bundle, t_ns)
    sel = range(len(bundle.bodies)) if bodies is None else bodies
    v, f, ids = merge_in_camera([bundle.bodies[i].mesh.positions for i in sel],
                                [bundle.bodies[i].mesh.indices for i in sel],
                                [t_cw @ poses[i] for i in sel], [bundle.bodies[i].id for i in sel])
    casts = [cast(v, f, ids, *pixel_rays(cam, o)) for o in offsets]
    depth, inst = combine_min_depth(casts) if len(casts) > 1 else casts[0]
    return depth.reshape(cam.height, cam.width), inst.reshape(cam.height, cam.width)
