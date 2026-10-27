"""Rotations and rigid transforms in the BallaSim conventions (DESIGN §5.1, CONTRACTS.md).

- Hamilton quaternions, active, stored as (w, x, y, z); never JPL. SciPy's scalar-last order is
  converted only inside this module.
- ``T_A_B`` is a 4x4 matrix that maps coordinates in frame B to frame A.
- The rotation logarithm is sign-invariant (|phi| <= pi, taken through the quaternion with w >= 0),
  so q and -q give the same residual; it and the exponential come from ``ballasim._core``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ballasim import _core


def _as_rows(x: np.ndarray, width: int) -> tuple[np.ndarray, tuple[int, ...]]:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != width:
        raise ValueError(f"expected trailing dimension {width}, got shape {x.shape}")
    return np.ascontiguousarray(x.reshape(-1, width)), x.shape[:-1]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def canonical_quat(q: np.ndarray) -> np.ndarray:
    """Unit quaternions on the w >= 0 hemisphere (a unique representative of each rotation)."""
    q = normalize_quat(q)
    return np.where(q[..., :1] < 0.0, -q, q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    rows, lead = _as_rows(q, 4)
    return Rotation.from_quat(rows[:, [1, 2, 3, 0]]).as_matrix().reshape(lead + (3, 3))


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrices -> unit quaternions (w, x, y, z) with w >= 0."""
    r = np.asarray(r, dtype=np.float64)
    lead = r.shape[:-2]
    q = Rotation.from_matrix(r.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
    return canonical_quat(q).reshape(lead + (4,))


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply the rotation q to vectors v (broadcasting over leading dimensions)."""
    return np.einsum("...ij,...j->...i", quat_to_matrix(q), np.asarray(v, dtype=np.float64))


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rotation vectors -> unit quaternions (w, x, y, z), fp64 (``_core.so3_exp``)."""
    rows, lead = _as_rows(phi, 3)
    return np.asarray(_core.so3_exp(rows)).reshape(lead + (4,))


def so3_log(q: np.ndarray) -> np.ndarray:
    """Quaternions -> rotation vectors with |phi| <= pi, invariant under q -> -q (``_core.so3_log``)."""
    rows, lead = _as_rows(q, 4)
    return np.asarray(_core.so3_log(rows)).reshape(lead + (3,))


def relative_rotvec(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Log(R_a^T R_b): the body-frame rotation from a to b, sign-invariant."""
    return so3_log(quat_multiply(quat_conjugate(q_a), q_b))


def rotation_angle(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotations, in [0, pi]; robust near 0 and pi."""
    dot = np.abs(np.sum(normalize_quat(q_a) * normalize_quat(q_b), axis=-1))
    rel = quat_multiply(quat_conjugate(q_a), q_b)
    return 2.0 * np.arctan2(np.linalg.norm(rel[..., 1:], axis=-1), np.minimum(dot, 1.0))


def hat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def _jacobian_coefficients(phi: np.ndarray):
    theta2 = np.sum(phi * phi, axis=-1)
    theta = np.sqrt(theta2)
    small = theta2 < 1e-8
    safe = np.where(small, 1.0, theta)
    return theta, theta2, small, safe


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """J_r(phi) = I - (1 - cos t)/t^2 [phi]x + (t - sin t)/t^3 [phi]x^2 (series below 1e-4 rad)."""
    phi = np.asarray(phi, dtype=np.float64)
    _, theta2, small, safe = _jacobian_coefficients(phi)
    a = np.where(small, 0.5 - theta2 / 24.0, (1.0 - np.cos(safe)) / safe**2)
    b = np.where(small, 1.0 / 6.0 - theta2 / 120.0, (safe - np.sin(safe)) / safe**3)
    k = hat(phi)
    return np.eye(3) - a[..., None, None] * k + b[..., None, None] * (k @ k)


def right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """J_r(phi)^-1 = I + [phi]x/2 + (1/t^2 - (1 + cos t)/(2 t sin t)) [phi]x^2."""
    phi = np.asarray(phi, dtype=np.float64)
    _, theta2, small, safe = _jacobian_coefficients(phi)
    c = np.where(small, 1.0 / 12.0 + theta2 / 720.0,
                 1.0 / safe**2 - (1.0 + np.cos(safe)) / (2.0 * safe * np.sin(safe)))
    k = hat(phi)
    return np.eye(3) + 0.5 * k + c[..., None, None] * (k @ k)


# Rigid transforms ---------------------------------------------------------------------------


def make_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros(r.shape[:-2] + (4, 4))
    out[..., :3, :3] = r
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    return out


def pose_to_transform(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return make_transform(quat_to_matrix(q), p)


def transform_to_pose(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64)
    return matrix_to_quat(t[..., :3, :3]), t[..., :3, 3].copy()


def invert_transform(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    rt = np.swapaxes(t[..., :3, :3], -1, -2)
    return make_transform(rt, -np.einsum("...ij,...j->...i", rt, t[..., :3, 3]))


def transform_points(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) @ t[:3, :3].T + t[:3, 3]


def is_rigid(t: np.ndarray, tol: float = 1e-6) -> bool:
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (4, 4) or not np.all(np.isfinite(t)):
        return False
    r = t[:3, :3]
    return (np.allclose(r.T @ r, np.eye(3), atol=tol) and abs(np.linalg.det(r) - 1.0) < tol
            and np.allclose(t[3], [0.0, 0.0, 0.0, 1.0], atol=0.0))


def rotation_about(axis: str, angle: float) -> np.ndarray:
    return Rotation.from_euler(axis, angle).as_matrix()


# Registration and robust averaging ----------------------------------------------------------


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares rigid (R, t) with R a_i + t ~ b_i (no reflection, no scale).

    Returns (R, t, per-point residual norms). Points are corresponding rows of a and b.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError(f"kabsch needs two (N >= 3, 3) arrays of equal shape, got {a.shape}, {b.shape}")
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = cb - r @ ca
    return r, t, np.linalg.norm(a @ r.T + t - b, axis=1)


def geometric_median(x: np.ndarray, *, iters: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Weiszfeld's geometric median of the rows of x (robust to up to half outliers)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.median(x, axis=0)
    for _ in range(iters):
        d = np.linalg.norm(x - m, axis=1)
        w = 1.0 / np.maximum(d, 1e-15)
        m_new = (w[:, None] * x).sum(axis=0) / w.sum()
        if np.linalg.norm(m_new - m) <= tol * max(1.0, np.linalg.norm(m)):
            return m_new
        m = m_new
    return m


def rotation_median(q: np.ndarray, *, iters: int = 200, tol: float = 1e-13) -> np.ndarray:
    """Geodesic L1 mean (Weiszfeld on SO(3)) of rotations given as quaternions; returns w >= 0."""
    q = canonical_quat(q)
    # Start from the chordal mean, which is close for any reasonable cluster.
    m = matrix_to_quat(Rotation.from_quat(q[:, [1, 2, 3, 0]]).mean().as_matrix())
    for _ in range(iters):
        v = relative_rotvec(np.broadcast_to(m, q.shape), q)
        d = np.linalg.norm(v, axis=1)
        w = 1.0 / np.maximum(d, 1e-15)
        step = (w[:, None] * v).sum(axis=0) / w.sum()
        m = canonical_quat(quat_multiply(m, so3_exp(step)))
        if np.linalg.norm(step) < tol:
            break
    return m


def robust_mean_pose(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust mean pose: geodesic median rotation and geometric-median position (SCN-03)."""
    return rotation_median(q), geometric_median(p)
