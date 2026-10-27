"""Evaluation of fitted trajectories through the shared C++ core (``ballasim._core``).

The fitter and every consumer evaluate splines here, so Python never re-implements the K = 6
cumulative B-spline: results are the fp64 reference path of the C++ header the renderer uses.
"""

from __future__ import annotations

import numpy as np

from ballasim import _core
from ballasim.scene.bundle import Trajectory
from ballasim.scene.transforms import pose_to_transform


def evaluate(traj: Trajectory, t_ns: np.ndarray) -> dict[str, np.ndarray]:
    """Pose and derivatives at int64 ns times: q (w, x, y, z), p (relative to origin_world),
    omega and omega_dot (body frame), v and a (world frame). Times outside the support raise.

    A piecewise trajectory (``traj.pieces``) is evaluated piece by piece through the C++ piece lookup
    (``core/traj/pieces.h``, bound as ``_core.spline_eval_pieces_f64``) that the renderer's ``body_state`` and the
    planner use too (BL-5b): a time at a breakpoint takes the right piece, and each piece's nodes come from its own
    rows, so the values equal the renderer's bit for bit."""
    if traj.pieces is None:
        return evaluate_controls(traj.K, traj.t0_ns, traj.h_ns, traj.rot, traj.pos, t_ns)
    t = np.ascontiguousarray(np.asarray(t_ns, dtype=np.int64).ravel())
    return _core.spline_eval_pieces_f64(traj.K, np.asarray(traj.pieces, dtype=np.int64),
                                        np.ascontiguousarray(traj.rot, dtype=np.float64),
                                        np.ascontiguousarray(traj.pos, dtype=np.float64), t)


def evaluate_controls(order: int, t0_ns: int, h_ns: int, rot: np.ndarray, pos: np.ndarray,
                      t_ns: np.ndarray) -> dict[str, np.ndarray]:
    """As :func:`evaluate`, for the control points of one spline that are not yet wrapped in a Trajectory."""
    t = np.ascontiguousarray(np.asarray(t_ns, dtype=np.int64).ravel())
    return _core.spline_eval_f64(order, int(t0_ns), int(h_ns), np.ascontiguousarray(rot, dtype=np.float64),
                                 np.ascontiguousarray(pos, dtype=np.float64), t)


def pose_matrices(traj: Trajectory, t_ns: np.ndarray) -> np.ndarray:
    """T_world_body(t) as (n, 4, 4) matrices (positions relative to origin_world)."""
    k = evaluate(traj, t_ns)
    return pose_to_transform(k["q"], k["p"])
