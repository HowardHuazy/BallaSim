"""Static-interval (hold) detection and hold constraints (DESIGN §5.4 step 9, §5.6; M2 Lemma B1).

A hold is an interval where a body is exactly still: every control point acting on it is tied to
one pose, so the spline's derivatives vanish there identically and Vicon wander never reaches the
renderer. In replays the holds come from the gated raw samples and the unconstrained pre-fit:

1. A body is *moving* where the pre-fit's bounding-sphere speed bound |v_c| + rho |omega|
   exceeds eps_on = 2 cm/s for at least T_hold = 30 ms (the hysteresis rule of M2 §D.5).
   Candidates are the maximal runs between moving stretches. Only sustained motion splits a
   candidate: Vicon wander and glitches on a large body (the 1.8 m tabletop wanders 0.1-0.4 mrad
   at 3-10 Hz, with ~10 ms jumps of 0.4 mrad, i.e. brief corner speeds of 1-3 cm/s) must not
   fragment it.
2. Interior edges (where the body starts or stops moving; not the ends of the data, where the
   smoother's boundary behaviour dominates) are trimmed to where the speed falls below
   min(eps_on / 2, max(3 x the candidate's median speed, 1 mm/s)), i.e. to that body's noise
   level. Edge placement matters: a hold that starts while the body still decelerates pins the
   spline (value and four derivatives) against moving data, and a stiff smoothing spline answers
   with a ringing misfit several times larger than the remaining motion (a hold 45 ms early on a
   smooth stop gave 0.47 mm).
3. The trimmed run must last >= T_hold and every gated sample in it must lie within 1 mm
   (bounding-sphere displacement) of the run's robust mean pose. A run that fails is a slow drift
   and is not held at all: splitting it would turn smooth drift into a staircase of holds.
4. Motion bouts shorter than T_hold between two holds are merged into one hold when the union
   still passes the displacement test.

That is hold_rule 'sphere' (scene.trajectory.hold_rule, the default and the v0 profiles' rule). Its 1 mm and 2 cm/s
are object-space numbers: about the rig origin (rho ~ 0.2 m) they are 5 mrad and 0.1 rad/s, so a real pan slower than
~1-2 deg/s for ~200 ms is held and its ~3 px of sweep frozen (F090). hold_rule 'image' measures both in the pixels of
the bundle's cameras (:class:`HoldView`): a bearing turning at |omega|, or a point at Euclidean distance z moving at
|v| (its bearing turns at <= |v| / z), moves in the image at most f (1 + r^2) times that, (1 + r^2) = 1 / cos^2 the
projection's radial stretch at normalised radius r and f the largest singular value of the pixel Jacobian over the
field (the focal length times the lens distortion's largest stretch; r^2 also the field's largest, undistorted):
- speed: a camera-carrying body (the rig) f (1 + r^2) (|omega| + |v_cam| / z), z the scene's near-distance bound;
  another body f (1 + r^2) (|v_c| + rho |omega|) / z, z its nearest point's distance to a camera;
- excursion: the same bounds on the pose differences (angle, displacement) from the run's robust mean pose, on the
  unconstrained pre-fit (the raw samples' white Vicon noise would dominate a sub-pixel bound).
The bound is scene.trajectory.hold_max_disp_px (0.5 px), the moving speed eps_on = that / T_hold (a motion that leaves
the bound within one shortest hold; 16.7 px/s) and the edge floor eps_on / 20 (the sphere rule's ratio). A candidate
whose excursion exceeds the bound is split once at its creeps: stretches faster than max(3 x its median speed, the
floor) for >= T_hold, or, when no piece passes there (a slow pan filling most of the candidate sets the median), 3 x its
20th percentile (its quiet speed; the median's edges, when they work, sit nearer the motion's onset). The pieces
between them are tested again and neighbours whose union passes are merged; a piece that fails, and a candidate without
a creep (a uniform drift), is not held (reason 'drift'): a split needs motion well above the candidate's quiet speed,
so smooth drift never becomes a staircase of holds. A piece shorter than the two blends a hold costs, 2 (K - 1) h
(50 ms at EVIMO2's 5 ms knots), is not held either (reason 'short'): it would turn wander into steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ballasim.scene.transforms import quat_rotate, robust_mean_pose, rotation_angle

from .basis import KnotGrid


def sphere_displacement(q_a, p_a, q_b, p_b, center: np.ndarray, radius: float) -> np.ndarray:
    """Upper bound on the displacement of any point of the bounding sphere (centre in the body
    frame) between poses a and b: |delta centre| + 2 rho sin(theta / 2)."""
    ca = quat_rotate(q_a, center) + p_a
    cb = quat_rotate(q_b, center) + p_b
    theta = rotation_angle(q_a, q_b)
    return np.linalg.norm(ca - cb, axis=-1) + 2.0 * radius * np.sin(0.5 * theta)


def sphere_speed(v: np.ndarray, omega_body: np.ndarray, q: np.ndarray, center: np.ndarray,
                 radius: float) -> np.ndarray:
    """Bound s = |v_c| + rho |omega| on the speed of the sphere's points (M2 D13)."""
    v_c = v + quat_rotate(q, np.cross(omega_body, center))
    return np.linalg.norm(v_c, axis=-1) + radius * np.linalg.norm(omega_body, axis=-1)


#: hold_rule 'image': when no piece passes between a failing candidate's creeps above 3x its median speed, it is split
#: again at 3x this percentile of its speed (its quiet speed; a pan below eps_on filling most of it set the median).
CREEP_PERCENTILE = 20.0


@dataclass(frozen=True)
class HoldView:
    """A body as the image-space hold rule sees it (hold_rule 'image'): per camera the largest singular value f_k of
    the pixel Jacobian over its field (px/rad; the focal length times the distortion's largest stretch) and the largest
    squared normalised (undistorted) radius r_k^2, and per input sample a lower bound on the Euclidean distance (m) of
    the surfaces the bound covers. For the camera-carrying body (the rig) ``cameras_body_m`` holds the camera centres
    in its frame and the distance is the scene's nearest surface; for another body it is None and the distance is from
    the nearest camera to the body's nearest point."""

    focal_px: np.ndarray
    corner_r2: np.ndarray
    depth_m: np.ndarray
    cameras_body_m: np.ndarray | None = None

    def _cams(self):
        return zip(self.focal_px * (1.0 + self.corner_r2), self.cameras_body_m)

    def speed(self, ev: dict, center: np.ndarray, radius: float, idx: np.ndarray) -> np.ndarray:
        """Bound on the image speed (px/s) at the pre-fit's values ``ev`` of input samples ``idx``."""
        z = self.depth_m[idx]
        if self.cameras_body_m is None:
            g = float(np.max(self.focal_px * (1.0 + self.corner_r2)))
            return g * sphere_speed(ev["v"], ev["omega"], ev["q"], center, radius) / z
        w = np.linalg.norm(ev["omega"], axis=-1)
        out = np.zeros(len(z))
        for g, c in self._cams():
            v = np.linalg.norm(ev["v"] + quat_rotate(ev["q"], np.cross(ev["omega"], c)), axis=-1)
            out = np.maximum(out, g * (w + v / z))
        return out

    def excursion(self, q, p, center: np.ndarray, radius: float, idx: np.ndarray) -> float:
        """Bound on the image displacement (px) of poses (q, p) of input samples ``idx`` from their robust mean."""
        qm, pm = robust_mean_pose(q, p)
        qa, pa = np.broadcast_to(qm, q.shape), np.broadcast_to(pm, p.shape)
        z = self.depth_m[idx]
        if self.cameras_body_m is None:
            g = float(np.max(self.focal_px * (1.0 + self.corner_r2)))
            return float((g * sphere_displacement(qa, pa, q, p, center, radius) / z).max())
        theta = rotation_angle(qa, q)
        out = np.zeros(len(z))
        for g, c in self._cams():
            out = np.maximum(out, g * (theta + sphere_displacement(qa, pa, q, p, c, 0.0) / z))
        return float(out.max())


@dataclass(frozen=True)
class StaticInterval:
    i0: int                    # first sample index (inclusive)
    i1: int                    # last sample index (inclusive)
    t_begin_ns: int
    t_end_ns: int
    max_displacement: float    # m (hold_rule sphere) or px (image)


def detect_static(t_ns: np.ndarray, q: np.ndarray, p: np.ndarray, speed: np.ndarray, *, center: np.ndarray,
                  radius: float, max_disp: float = 1e-3, min_duration_ns: int = 30_000_000,
                  moving_speed: float = 0.02, edge_speed_floor: float = 1e-3, edge_noise_factor: float = 3.0,
                  excursion: Callable[[int, int], float] | None = None, split: bool = False,
                  min_piece_ns: int = 0) -> tuple[list[StaticInterval], list[dict]]:
    """Static intervals from gated raw samples (q, p) and the pre-fit's speed at the same times. ``excursion``
    (i0, i1) -> the run's largest displacement (default: the bounding-sphere displacement of the samples from their
    robust mean pose, m); ``split``: split creeping candidates (hold_rule image) into pieces of >= min_piece_ns.
    Returns (accepted intervals, rejected candidates with their reason)."""
    t = np.asarray(t_ns, np.int64)
    speed = np.asarray(speed, np.float64)
    quiet = (min_duration_ns, edge_speed_floor, edge_noise_factor)
    rejected: list[dict] = []

    def check(i0, i1):
        qm, pm = robust_mean_pose(q[i0:i1 + 1], p[i0:i1 + 1])
        disp = sphere_displacement(np.broadcast_to(qm, (i1 - i0 + 1, 4)), np.broadcast_to(pm, (i1 - i0 + 1, 3)),
                                   q[i0:i1 + 1], p[i0:i1 + 1], center, radius)
        return float(disp.max())

    check = excursion or check

    def reject(i0, i1, d, reason):
        rejected.append({"t_begin_ns": int(t[i0]), "t_end_ns": int(t[i1]), "max_displacement": d, "reason": reason})

    def split_at(i0, i1, pct):
        """Pieces of a creeping candidate between its stretches faster than 3x its speed's pct-th percentile, each
        >= T_hold; neighbouring passing pieces whose union passes are one hold (the rule's own criterion). Returns
        (passing pieces of >= min_piece_ns, failing ones, shorter ones)."""
        creep = max(edge_noise_factor * float(np.percentile(speed[i0:i1 + 1], pct)), edge_speed_floor)
        pieces, failed = [], []
        for a, b in _quiet_runs(t, speed, i0, i1, creep, *quiet):
            e = check(a, b)
            if e > max_disp:
                failed.append((a, b, e))
                pieces.append(None)
            elif pieces and pieces[-1] and (u := check(pieces[-1][0], b)) <= max_disp:
                pieces[-1] = (pieces[-1][0], b, u)
            else:
                pieces.append((a, b, e))
        pieces = [x for x in pieces if x]
        return ([x for x in pieces if t[x[1]] - t[x[0]] >= min_piece_ns], failed,
                [x for x in pieces if t[x[1]] - t[x[0]] < min_piece_ns])

    accepted: list[list] = []
    for i0, i1 in _quiet_runs(t, speed, 0, len(t) - 1, moving_speed, *quiet):
        d = check(i0, i1)
        pieces = [(i0, i1, d)]
        if d > max_disp:
            if not split:
                reject(i0, i1, d, "displacement")
                continue
            # Creeping: split once at its median speed's creeps, or, when no piece passes there (a slow pan filling
            # most of the candidate sets its median), at its quiet speed's (CREEP_PERCENTILE). A uniform drift has no
            # creep at either: its one piece fails again, and it is not held.
            pieces, failed, short = split_at(i0, i1, 50.0)
            if not pieces:
                pieces, failed, short = split_at(i0, i1, CREEP_PERCENTILE)
            for x in failed or ([] if pieces or short else [(i0, i1, d)]):
                reject(*x, "drift")
            for x in short:
                reject(*x, "short")
        for a, b, d in pieces:
            if accepted and t[a] - t[accepted[-1][1]] < min_duration_ns:
                d_union = check(accepted[-1][0], b)
                if d_union <= max_disp:
                    accepted[-1] = [accepted[-1][0], b, d_union]
                    continue
            accepted.append([a, b, d])
    return ([StaticInterval(i0, i1, int(t[i0]), int(t[i1]), d) for i0, i1, d in accepted], rejected)


def detect_static_image(t_ns: np.ndarray, ev: dict, view: HoldView, idx: np.ndarray, *, center: np.ndarray,
                        radius: float, min_duration_ns: int, max_disp_px: float = 0.5,
                        min_piece_ns: int = 0) -> tuple[list[StaticInterval], list[dict]]:
    """hold_rule 'image' (F090): holds by image speed and excursion (px) on the pre-fit's values ``ev`` (q, p, v,
    omega) at the kept samples, ``idx`` their input indices (``view``'s depths are per input sample); the pieces of a
    split candidate last >= min_piece_ns (fit_smoothing: the two blends a hold costs, 2 (K - 1) h)."""
    q, p = ev["q"], ev["p"]
    eps_on = max_disp_px / (min_duration_ns * 1e-9)

    def excursion(i0, i1):
        return view.excursion(q[i0:i1 + 1], p[i0:i1 + 1], center, radius, idx[i0:i1 + 1])

    return detect_static(t_ns, q, p, view.speed(ev, center, radius, idx), center=center, radius=radius,
                         max_disp=max_disp_px, min_duration_ns=min_duration_ns, moving_speed=eps_on,
                         edge_speed_floor=eps_on / 20, excursion=excursion, split=True, min_piece_ns=min_piece_ns)


def _quiet_runs(t: np.ndarray, speed: np.ndarray, i0: int, i1: int, moving_speed: float, min_duration_ns: int,
                edge_floor: float, edge_factor: float) -> list[tuple[int, int]]:
    """Candidates in samples i0..i1: the runs between sustained motion (speed > moving_speed for >= min_duration_ns),
    interior edges trimmed to the run's noise level, at least min_duration_ns long."""
    s = speed[i0:i1 + 1]
    moving = np.zeros(len(s), bool)
    for a, b in _runs(s > moving_speed):
        if t[i0 + b] - t[i0 + a] >= min_duration_ns:
            moving[a:b + 1] = True
    runs = []
    last = len(s) - 1
    for a, b in _runs(~moving):
        edge = min(0.5 * moving_speed, max(edge_factor * float(np.median(s[a:b + 1])), edge_floor))
        inside = np.flatnonzero(s[a:b + 1] <= edge)
        if not len(inside):
            continue
        # Trim only edges where the body starts or stops moving; at the ends of the range (the data's, where the
        # pre-fit's speed reflects the smoother's boundary behaviour, or a candidate's, already trimmed) keep them.
        runs.append((i0 + (a if a == 0 else a + int(inside[0])), i0 + (b if b == last else a + int(inside[-1]))))
    return [(a, b) for a, b in runs if t[b] - t[a] >= min_duration_ns]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


@dataclass
class TieMap:
    """Control points c -> free parameters f(c); each tied group (c0, c1), inclusive, shares one
    parameter. f is monotone with unit steps, so a banded normal matrix stays banded after tying."""

    num_ctrl: int
    groups: list[tuple[int, int]]

    def __post_init__(self):
        f = np.arange(self.num_ctrl)
        for c0, c1 in self.groups:
            f[c0:c1 + 1] = c0
        _, self.free_index = np.unique(f, return_inverse=True)
        self.num_free = int(self.free_index.max()) + 1

    def matrix(self, block: int = 1):
        """Sparse G (block N x block N_free) with x = G x_free."""
        rows = (np.arange(self.num_ctrl)[:, None] * block + np.arange(block)[None, :]).ravel()
        cols = (self.free_index[:, None] * block + np.arange(block)[None, :]).ravel()
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.num_ctrl * block, self.num_free * block))


def hold_groups(grid: KnotGrid,
                intervals: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Tie groups for closed hold intervals [t_a, t_b]. Groups that share a control point are merged
    (they must hold one pose); adjacent groups stay separate, the spline then blends their two poses
    over the K - 1 segments between them.

    Returns (groups as control ranges, exact still intervals [t_begin, t_end) on the knot grid)."""
    groups: list[list[int]] = []
    for ta, tb in sorted(intervals):
        c0, c1 = grid.controls_touching(int(ta), int(tb))
        if groups and c0 <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], c1)
        else:
            groups.append([c0, c1])
    out = [(a, b) for a, b in groups if b - a + 1 >= grid.order]
    return out, [grid.still_interval(a, b) for a, b in out]
