"""Trajectory fitting: robust smoothing of measured poses and exact interpolation of keyframes.

Both modes produce the C1 :class:`~ballasim.scene.bundle.Trajectory`: a split SO(3) x R^3
cumulative B-spline of order K (default 6) on uniform int64-ns knots (DESIGN §5.3-5.4, M2 §C).

Smoothing (measured data, M2 §C.0-C.6, Phase 1 subset):
  E = sum_n rho(s_n) + lambda_p/sigma_p^2 R_p + lambda_R/sigma_R^2 R_R,  s_n = |r_rot|^2/sigma_R^2 + |r_pos|^2/sigma_p^2
  - r_rot = Log(R~_n^T R(t_n)) is sign-invariant (q and -q are the same rotation);
  - rho is Cauchy with c^2 = chi2_6(0.99) = 16.81, minimized by IRLS;
  - R_p = sum_s int |d^3 p/du^3|^2 du, the exact jerk penalty in banded form (M2 C4);
  - R_R = (D_2 d)^T Gamma (D_2 d) with d_c = Log(R_{c-1}^T R_c): the same banded form on the
    per-knot rotation increments. It is the jerk penalty of the unrolled rotation vector, exact
    for rotations about a fixed axis and bi-invariant; for general motion it differs from
    int |omega_B''|^2 by Lie-bracket terms of order |omega| |omega'| (the order that separates
    the bi-invariant and local-chart WNOJ priors, M2 §C.3);
  - given the IRLS weights the position block is linear and the rotation block is solved by
    Levenberg-Marquardt on the manifold (R_c <- R_c Exp(delta_c)); the Cauchy weights couple them;
  - lambda per block by GCV (exact for positions, on the Gauss-Newton linearization for
    rotations), refined jointly with the IRLS weights (AR(1) REML is Phase 2);
  - outliers: model-free gating before the fit (G-a), Mahalanobis > 8 after it (G-b), refit once;
  - holds: static stretches detected on the gated samples become hard equality constraints
    (tied control points), so the spline is exactly still there (M2 Lemma B1).

Interpolation (authored keyframes, M2 §C.7): the same roughness objective with hard constraints,
KKT for positions and SQP for rotations, holds for runs of identical keys; |r_n| < 1e-9.

Splines are evaluated by ``ballasim._core`` throughout; Python only builds the linear systems.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from math import gcd

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial.transform import Rotation, Slerp

from ballasim import _core
from ballasim.scene.bundle import Trajectory
from ballasim.scene.transforms import (
    canonical_quat,
    quat_multiply,
    quat_to_matrix,
    relative_rotvec,
    right_jacobian,
    right_jacobian_inv,
    rotation_angle,
    so3_exp,
)

from .banded import BandedSPD, selected_inverse_band, trace_of_product
from .basis import KnotGrid, cumulative_basis, difference_gram, difference_matrix, roughness_matrix
from .evaluate import evaluate_controls
from .gating import ALIAS_RAD, gate_poses
from .holds import (
    HoldView,
    TieMap,
    detect_static,
    detect_static_image,
    hold_groups,
    sphere_displacement,
    sphere_speed,
)

log = logging.getLogger(__name__)
_default = _core.config.default


class TrajectoryError(ValueError):
    """The input cannot be represented faithfully (e.g. rotation aliasing between samples)."""


@dataclass
class PoseSamples:
    """Timed poses T_world_body: t_ns int64 strictly increasing, q (w, x, y, z), p metres."""

    t_ns: np.ndarray
    q: np.ndarray
    p: np.ndarray

    def __post_init__(self):
        self.t_ns = np.asarray(self.t_ns, np.int64)
        self.q = np.asarray(self.q, np.float64)
        self.p = np.asarray(self.p, np.float64)
        n = len(self.t_ns)
        if self.t_ns.ndim != 1 or self.q.shape != (n, 4) or self.p.shape != (n, 3):
            raise ValueError("PoseSamples needs t_ns (n,), q (n, 4), p (n, 3)")
        if n and (np.any(np.diff(self.t_ns) <= 0)):
            raise ValueError("sample times must be strictly increasing")
        if not (np.all(np.isfinite(self.q)) and np.all(np.isfinite(self.p))):
            raise ValueError("samples contain NaN or Inf")
        if n and np.abs(np.linalg.norm(self.q, axis=1) - 1.0).max() > 1e-3:
            raise ValueError("sample quaternions must have unit norm")
        self.q = self.q / np.linalg.norm(self.q, axis=1, keepdims=True)

    def __len__(self):
        return len(self.t_ns)


#: The SmoothingOptions fields that are schema keys scene.trajectory.<field> (D15a; docs/dev/P0_PARAMS.md §3.4): their
#: defaults are the schema's, and a run sets them from its resolved values (ballasim.engine.scene.evimo2_options).
SCHEMA_FIELDS = ("order", "loss_c2", "gate", "gate_sigma_k", "gate_rot_floor_rad", "gate_pos_floor_m", "mahalanobis_max",
                 "lambda_pos", "lambda_rot", "gcv_log10_range", "detect_holds", "hold_max_disp_m", "hold_min_ns",
                 "hold_moving_mps", "hold_rule", "hold_max_disp_px", "branch_jump_rad", "branch_rate_factor")


def _d(name: str):
    return _default(f"scene.trajectory.{name}")


@dataclass
class SmoothingOptions:
    order: int = _d("order")
    h_ns: int | None = None                    # default: round(median sample period)
    span_ns: tuple[int, int] | None = None     # default: first..last kept sample; beyond data = hold
    loss_c2: float = _d("loss_c2")             # Cauchy c^2 = chi2_6(0.99)
    gate: bool = _d("gate")
    gate_sigma_k: float = _d("gate_sigma_k")
    gate_rot_floor_rad: float = _d("gate_rot_floor_rad")   # math.radians(3.0)
    gate_pos_floor_m: float = _d("gate_pos_floor_m")
    mahalanobis_max: float = _d("mahalanobis_max")
    lambda_pos: float | None = _d("lambda_pos")            # u-units (penalty per segment in u); None = GCV
    lambda_rot: float | None = _d("lambda_rot")
    gcv_log10_range: tuple[float, float] = tuple(_d("gcv_log10_range"))
    detect_holds: bool = _d("detect_holds")
    hold_max_disp_m: float = _d("hold_max_disp_m")
    hold_min_ns: int = _d("hold_min_ns")
    hold_moving_mps: float = _d("hold_moving_mps")         # eps_on (DESIGN §5.6): faster is clearly moving
    hold_rule: str = _d("hold_rule")                       # sphere (object space) or image (px; F090, holds.py)
    hold_max_disp_px: float = _d("hold_max_disp_px")       # hold_rule image: largest image excursion in a hold
    hold_view: HoldView | None = None                      # the cameras, for hold_rule image (the adapter's)
    branch_jump_rad: float = _d("branch_jump_rad")         # gate G-a's orientation-branch split (gating.minority_branches)
    branch_rate_factor: float = _d("branch_rate_factor")
    center_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3))  # bounding sphere
    radius_m: float = 0.1
    sigma_floor_pos_m: float = 1e-6
    sigma_floor_rot_rad: float = 1e-6
    max_irls: int = 60


@dataclass
class FitResult:
    trajectory: Trajectory
    diagnostics: dict                          # JSON-serializable summary
    keep: np.ndarray                           # bool per input sample (gating G-a and G-b)
    weights: np.ndarray                        # final IRLS weights per input sample (0 if rejected)


# ---------------------------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------------------------


def check_rotation_aliasing(q: np.ndarray, *, warn_rad: float = 0.6 * math.pi,
                            fail_rad: float = ALIAS_RAD) -> int:
    """Residuals use the shortest rotation, so a true step above pi between samples would be
    reconstructed backwards (DESIGN §5.3). Raise above 0.9 pi; return the count above 0.6 pi."""
    if len(q) < 2:
        return 0
    steps = rotation_angle(q[:-1], q[1:])
    if steps.max() > fail_rad:
        i = int(np.argmax(steps))
        raise TrajectoryError(f"consecutive samples {i} and {i + 1} differ by {math.degrees(steps[i]):.1f} deg "
                              "(> 162 deg): the rotation direction is ambiguous; add intermediate samples")
    return int(np.sum(steps > warn_rad))


def _slerp_init(t_ns: np.ndarray, q: np.ndarray, at_ns: np.ndarray) -> np.ndarray:
    """Control rotations at the Greville abscissae by slerp of the samples (clamped at the ends)."""
    if len(t_ns) == 1:
        return np.tile(q[0], (len(at_ns), 1))
    rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    tc = np.clip(at_ns, t_ns[0], t_ns[-1]).astype(np.float64)
    out = Slerp(t_ns.astype(np.float64), rot)(tc).as_quat()[:, [3, 0, 1, 2]]
    return _align_hemisphere(out)


def _align_hemisphere(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions have non-negative dot products (same rotations)."""
    q = np.array(q, np.float64)
    if len(q) > 1:
        flips = np.concatenate([[1.0], np.where(np.sum(q[1:] * q[:-1], axis=1) < 0, -1.0, 1.0)])
        q *= np.cumprod(flips)[:, None]
    return q


class _Spline:
    """Control points on a knot grid with tied groups, plus the fitter's linearizations."""

    def __init__(self, grid: KnotGrid, rot: np.ndarray, pos: np.ndarray, tie: TieMap,
                 fixed: dict[int, tuple[np.ndarray, np.ndarray]] | None = None):
        self.grid = grid
        self.k = grid.order
        self.m = grid.order // 2
        self.n = grid.num_ctrl
        self.tie = tie
        # fixed: free index -> (q, p) held exactly (interpolation-mode holds on key poses).
        self.fixed = fixed or {}
        self.var = np.array([f for f in range(tie.num_free) if f not in self.fixed], np.int64)
        self.rot = _align_hemisphere(rot)
        self.pos = np.array(pos, np.float64)
        self._project()
        g1 = tie.matrix(1)
        g3 = tie.matrix(3)
        self.g1_var = g1[:, self.var]
        self.g3_var = g3[:, (self.var[:, None] * 3 + np.arange(3)).ravel()]
        self.penalty_pos = roughness_matrix(self.n, self.k, self.m)
        self.d_rot = sp.kron(difference_matrix(self.n - 1, self.m - 1), sp.identity(3), format="csr")
        self.gamma3 = sp.kron(difference_gram(self.n, self.k, self.m), sp.identity(3), format="csr")

    def _project(self):
        """Make tied control points bit-identical (and fixed ones equal to their pose)."""
        f = self.tie.free_index
        first = np.zeros(self.tie.num_free, np.int64)
        first[f[::-1]] = np.arange(self.n)[::-1]
        for fi, (q, p) in self.fixed.items():
            self.rot[first[fi]] = q
            self.pos[first[fi]] = p
        self.rot = self.rot[first[f]].copy()
        self.pos = self.pos[first[f]].copy()

    def evaluate(self, t_ns):
        return evaluate_controls(self.k, self.grid.t0_ns, self.grid.h_ns, self.rot, self.pos, t_ns)

    def increments(self) -> np.ndarray:
        return relative_rotvec(self.rot[:-1], self.rot[1:])

    def rotation_penalty_value(self) -> float:
        e = self.d_rot @ self.increments().ravel()
        return float(e @ (self.gamma3 @ e))

    def rotation_penalty(self):
        """e = D_{m-1} d (stacked xyz) and its Jacobian w.r.t. right perturbations of the controls."""
        d = self.increments()                               # (n-1, 3), d_c for c = 1..n-1
        e = self.d_rot @ d.ravel()
        jri = right_jacobian_inv(d)                         # dd_c/de_c ; dd_c/de_{c-1} = -Jr^-1(d)^T
        c = np.arange(1, self.n)
        rows = (3 * (c - 1))[:, None, None] + np.arange(3)[None, :, None]
        cols_c = (3 * c)[:, None, None] + np.arange(3)[None, None, :]
        rr = np.broadcast_to(rows, jri.shape)
        data = np.concatenate([jri.ravel(), -np.swapaxes(jri, 1, 2).ravel()])
        r_idx = np.concatenate([rr.ravel(), rr.ravel()])
        c_idx = np.concatenate([np.broadcast_to(cols_c, jri.shape).ravel(),
                                np.broadcast_to(cols_c - 3, jri.shape).ravel()])
        lin = sp.csr_matrix((data, (r_idx, c_idx)), shape=(3 * (self.n - 1), 3 * self.n))
        return e, self.d_rot @ lin

    def rotation_jacobian(self, t_ns: np.ndarray, r_res: np.ndarray) -> sp.csr_matrix:
        """d r_n / d eps_c for r_n = Log(R~_n^T R(t_n)) and R_c <- R_c Exp(eps_c) (3n x 3N)."""
        s, u = self.grid.locate(t_ns)
        k, n = self.k, len(t_ns)
        lam = cumulative_basis(u, k)                        # (n, K-1): lambda_j, j = 1..K-1
        d_all = np.zeros((self.n, 3))
        d_all[1:] = self.increments()
        d = d_all[s[:, None] + np.arange(1, k)[None, :]]    # (n, K-1, 3)
        a = quat_to_matrix(so3_exp(lam[..., None] * d))     # A_j
        q = np.empty((n, k, 3, 3))                          # Q_j = A_{j+1} ... A_{K-1}
        q[:, k - 1] = np.eye(3)
        for j in range(k - 2, -1, -1):
            q[:, j] = a[:, j] @ q[:, j + 1]
        qt = np.swapaxes(q, -1, -2)
        mj = lam[..., None, None] * right_jacobian(lam[..., None] * d)   # lambda_j J_r(lambda_j d_j)
        jri = right_jacobian_inv(d)
        jli = np.swapaxes(jri, -1, -2)
        blk = np.empty((n, k, 3, 3))
        blk[:, 0] = qt[:, 0] - qt[:, 1] @ mj[:, 0] @ jli[:, 0]
        for j in range(1, k - 1):
            blk[:, j] = qt[:, j] @ mj[:, j - 1] @ jri[:, j - 1] - qt[:, j + 1] @ mj[:, j] @ jli[:, j]
        blk[:, k - 1] = mj[:, k - 2] @ jri[:, k - 2]
        blk = right_jacobian_inv(r_res)[:, None] @ blk
        rows = (3 * np.arange(n))[:, None, None, None] + np.arange(3)[None, None, :, None]
        cols = (3 * (s[:, None] + np.arange(k)[None, :]))[:, :, None, None] + np.arange(3)[None, None, None, :]
        rows = np.broadcast_to(rows, blk.shape)
        cols = np.broadcast_to(cols, blk.shape)
        return sp.csr_matrix((blk.ravel(), (rows.ravel(), cols.ravel())), shape=(3 * n, 3 * self.n))

    def apply_rotation_step(self, delta_var: np.ndarray) -> None:
        full = self.g3_var @ delta_var
        self.rot = _align_hemisphere(canonical_quat(quat_multiply(self.rot, so3_exp(full.reshape(-1, 3)))))
        self._project()

    def set_positions(self, x_var: np.ndarray, x_fixed_full: np.ndarray | None = None) -> None:
        self.pos = self.g1_var @ x_var
        if x_fixed_full is not None:
            self.pos = self.pos + x_fixed_full
        self._project()

    def fixed_positions_full(self) -> np.ndarray | None:
        """Contribution of fixed groups to the full control positions (zero elsewhere)."""
        if not self.fixed:
            return None
        g1 = self.tie.matrix(1)
        x = np.zeros((self.tie.num_free, 3))
        for fi, (_, p) in self.fixed.items():
            x[fi] = p
        return g1 @ x


def _rotation_residuals(q_meas: np.ndarray, q_spline: np.ndarray) -> np.ndarray:
    return relative_rotvec(q_meas, q_spline)


def _gcv_search(evaluate, lo: float, hi: float) -> tuple[float, dict]:
    """Minimize GCV over log10(lambda): whole decades, then quarter decades, then a parabola."""
    coarse = np.arange(lo, hi + 1e-9, 1.0)
    vc = evaluate(10.0**coarse)
    i = int(np.nanargmin(vc))
    fine = np.arange(max(lo, coarse[i] - 1.0), min(hi, coarse[i] + 1.0) + 1e-9, 0.25)
    vf = evaluate(10.0**fine)
    j = int(np.nanargmin(vf))
    best = fine[j]
    if 0 < j < len(fine) - 1:
        x, y = fine[j - 1:j + 2], vf[j - 1:j + 2]
        a = np.polyfit(x, y, 2)
        if a[0] > 0:
            best = float(np.clip(-a[1] / (2 * a[0]), x[0], x[2]))
    at_edge = best <= lo + 1e-9 or best >= hi - 1e-9
    return 10.0**best, {"log10_grid": np.concatenate([coarse, fine]).round(4).tolist(),
                        "gcv": np.concatenate([vc, vf]).tolist(), "at_grid_edge": bool(at_edge)}


def _solve_and_trace(mats: list[sp.spmatrix], rhs: list[np.ndarray], m: sp.spmatrix,
                     u: int) -> tuple[list[np.ndarray], list[float]]:
    """Solve A_l x_l = rhs_l and return tr(A_l^-1 M), one band Cholesky per A_l and a single
    batched selected inversion (its sequential loop dominates, so the batch shares it)."""
    chol = [BandedSPD(a, u) for a in mats]
    sols = [c.solve(b) for c, b in zip(chol, rhs)]
    traces = trace_of_product(selected_inverse_band(np.stack([c.factor for c in chol])), m)
    return sols, traces.tolist()


# ---------------------------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------------------------


@dataclass
class _RotSystem:
    rr: np.ndarray            # (n, 3) residuals Log(R~^T R)
    jac: sp.csr_matrix        # d rr / d free parameters
    jtwj: sp.csr_matrix
    dtgd: sp.csr_matrix       # Gauss-Newton Hessian of the rotational penalty
    grad_data: np.ndarray
    grad_pen: np.ndarray
    penalty: float


class _Smoother:
    def __init__(self, samples: PoseSamples, keep: np.ndarray, opts: SmoothingOptions, grid: KnotGrid):
        self.opts = opts
        self.samples = samples
        self.keep = keep.copy()
        self.grid = grid
        self.sigma_p = 1e-3
        self.sigma_r = 1e-3
        self.mu = 1e-8
        self.tie = TieMap(grid.num_ctrl, [])
        kept = np.flatnonzero(keep)
        rot0 = _slerp_init(samples.t_ns[kept], samples.q[kept], grid.greville_ns())
        self.spline = _Spline(grid, rot0, np.zeros((grid.num_ctrl, 3)), self.tie)
        self._prepare()

    def _prepare(self):
        idx = np.flatnonzero(self.keep)
        self.idx = idx
        self.t = self.samples.t_ns[idx]
        self.qm = self.samples.q[idx]
        self.pm = self.samples.p[idx]
        self.b = self.grid.design_matrix(self.t)

    def set_ties(self, groups):
        self.tie = TieMap(self.grid.num_ctrl, groups)
        self.spline = _Spline(self.grid, self.spline.rot, self.spline.pos, self.tie)

    def drop(self, mask_kept: np.ndarray):
        """Reject kept samples (G-b) given a boolean mask over the currently kept samples."""
        self.keep[self.idx[mask_kept]] = False
        self._prepare()

    # -- residuals and weights --------------------------------------------------------------

    def residuals(self):
        ev = self.spline.evaluate(self.t)
        return _rotation_residuals(self.qm, ev["q"]), ev["p"] - self.pm

    def weights(self, rr, rp):
        s = np.sum(rr**2, 1) / self.sigma_r**2 + np.sum(rp**2, 1) / self.sigma_p**2
        return 1.0 / (1.0 + s / self.opts.loss_c2), s

    # -- position block (linear) ------------------------------------------------------------

    def _pos_system(self, w):
        g = self.spline.g1_var
        bw = self.b.multiply(w[:, None]).tocsr()
        btwb = (g.T @ (self.b.T @ bw) @ g).tocsr()
        pen = (g.T @ self.spline.penalty_pos @ g).tocsr()
        rhs = g.T @ (bw.T @ self.pm)
        return btwb, pen, rhs

    def solve_positions(self, lam, w):
        btwb, pen, rhs = self._pos_system(w)
        self.spline.set_positions(BandedSPD(btwb + lam * pen).solve(rhs))

    def gcv_positions(self, w):
        btwb, pen, rhs = self._pos_system(w)
        u = self.grid.order - 1
        nd = len(self.t)

        def rss_edf(lams):
            sols, tr = _solve_and_trace([btwb + lam * pen for lam in lams], [rhs] * len(lams), btwb, u)
            rss = [float(np.sum(w[:, None] * (self.b @ (self.spline.g1_var @ x) - self.pm) ** 2)) for x in sols]
            return np.array(rss), np.array(tr)

        def gcv(lams):
            rss, tr = rss_edf(lams)
            return nd * rss / np.maximum(nd - tr, 1e-9) ** 2

        lam, info = _gcv_search(gcv, *self.opts.gcv_log10_range)
        rss, tr = rss_edf([lam])
        info.update(edf=float(tr[0]), rss=float(rss[0]))
        return lam, info

    # -- rotation block (Gauss-Newton / LM on the manifold) -----------------------------------

    def _rot_system(self, w):
        """Gauss-Newton pieces of the rotation block at the current controls (free parameters)."""
        ev = self.spline.evaluate(self.t)
        rr = _rotation_residuals(self.qm, ev["q"])
        jac = self.spline.rotation_jacobian(self.t, rr) @ self.spline.g3_var
        w3 = np.repeat(w, 3)
        jtw = jac.T.multiply(w3[None, :]).tocsr()
        jtwj = (jtw @ jac).tocsr()
        e, de = self.spline.rotation_penalty()
        de = de @ self.spline.g3_var
        gamma = self.spline.gamma3
        dtgd = (de.T @ gamma @ de).tocsr()
        return _RotSystem(rr, jac, jtwj, dtgd, jtw @ rr.ravel(), de.T @ (gamma @ e), float(e @ (gamma @ e)))

    def _rot_objective(self, w, lam):
        rr, _ = self.residuals()
        return float(np.sum(w * np.sum(rr**2, 1)) + lam * self.spline.rotation_penalty_value())

    def rotation_steps(self, lam, w, steps: int = 3) -> float:
        """Up to `steps` LM iterations at fixed weights; returns the last max |delta|."""
        last = 0.0
        u = 3 * self.grid.order - 1
        for _ in range(steps):
            rs = self._rot_system(w)
            h = (rs.jtwj + lam * rs.dtgd).tocsr()
            g = rs.grad_data + lam * rs.grad_pen
            f0 = float(np.sum(w * np.sum(rs.rr**2, 1)) + lam * rs.penalty)
            rot_saved = self.spline.rot.copy()
            diag = sp.diags(h.diagonal())
            accepted = False
            for _ in range(12):
                delta = BandedSPD(h + self.mu * diag, u).solve(-g)
                self.spline.apply_rotation_step(delta)
                f1 = self._rot_objective(w, lam)
                if f1 <= f0 * (1 + 1e-12) + 1e-300:
                    self.mu = max(self.mu / 3.0, 1e-12)
                    accepted = True
                    break
                self.spline.rot = rot_saved.copy()
                self.mu *= 10.0
            last = float(np.abs(delta).max()) if accepted else 0.0
            if not accepted or last < 1e-10:
                break
        return last

    def gcv_rotations(self, w):
        """GCV on the Gauss-Newton linearization at the current fit (M2 §C.6, nonlinear block)."""
        rs = self._rot_system(w)
        u = 3 * self.grid.order - 1
        nd = 3 * len(self.t)
        w3 = np.repeat(w, 3)

        def rss_edf(lams):
            sols, tr = _solve_and_trace([(rs.jtwj + lam * rs.dtgd).tocsr() for lam in lams],
                                        [-(rs.grad_data + lam * rs.grad_pen) for lam in lams], rs.jtwj, u)
            rss = [float(np.sum(w3 * (rs.rr.ravel() + rs.jac @ delta) ** 2)) for delta in sols]
            return np.array(rss), np.array(tr)

        def gcv(lams):
            rss, tr = rss_edf(lams)
            return nd * rss / np.maximum(nd - tr, 1e-9) ** 2

        lam, info = _gcv_search(gcv, *self.opts.gcv_log10_range)
        rss, tr = rss_edf([lam])
        info.update(edf=float(tr[0]), rss=float(rss[0]))
        return lam, info

    # -- IRLS ---------------------------------------------------------------------------------

    def irls(self, lam_p, lam_r) -> int:
        """Alternate Cauchy weights, the exact position solve and one LM rotation step until the
        control points move by less than 1e-2 of the noise (sigma_p, sigma_R), floored at 1e-9.

        Cauchy IRLS converges linearly (rate ~0.9 when many samples sit near the loss knee, as on
        hand-held drones), so the tolerance is tied to the statistical resolution of the fit (its
        own standard error is ~0.3 sigma) rather than to round-off; 1e-2 sigma is <= 0.002 px for
        EVIMO2 noise levels."""
        tol_p = max(1e-2 * self.sigma_p, 1e-9)
        tol_r = max(1e-2 * self.sigma_r, 1e-9)
        for it in range(self.opts.max_irls):
            rr, rp = self.residuals()
            w = self.weights(rr, rp)[0]
            pos_old = self.spline.pos.copy()
            self.solve_positions(lam_p, w)
            step = self.rotation_steps(lam_r, w, steps=1)
            if it and step < tol_r and np.abs(self.spline.pos - pos_old).max() < tol_p:
                return it + 1
        return self.opts.max_irls

    def update_sigmas(self, info_p, info_r):
        n = len(self.t)
        self.sigma_p = max(math.sqrt(info_p["rss"] / max(3 * (n - info_p["edf"]), 1.0)), self.opts.sigma_floor_pos_m)
        self.sigma_r = max(math.sqrt(info_r["rss"] / max(3 * n - info_r["edf"], 1.0)), self.opts.sigma_floor_rot_rad)

    def current_weights(self):
        rr, rp = self.residuals()
        return self.weights(rr, rp)


def fit_smoothing(samples: PoseSamples, options: SmoothingOptions | None = None) -> FitResult:
    """Robust smoothing fit of measured poses (see module docstring)."""
    opts = options or SmoothingOptions()
    n = len(samples)
    if n < 2:
        raise TrajectoryError("smoothing needs at least two samples")
    if opts.hold_rule not in ("sphere", "image") or (opts.hold_rule == "image" and opts.hold_view is None):
        raise ValueError(f"hold_rule {opts.hold_rule!r}: sphere, or image with a HoldView (the cameras)")
    if opts.hold_rule == "image" and not (opts.hold_min_ns > 0 and opts.hold_max_disp_px > 0):
        raise ValueError("hold_rule image needs scene.trajectory.hold_min_ns > 0 and hold_max_disp_px > 0 "
                         f"(eps_on = hold_max_disp_px / hold_min_ns); got {opts.hold_min_ns}, {opts.hold_max_disp_px}")
    diag: dict = {"mode": "smoothing", "num_samples": n}
    period = float(np.median(np.diff(samples.t_ns)))
    if opts.gate:
        gate = gate_poses(samples.t_ns, samples.q, samples.p, period_ns=period, sigma_k=opts.gate_sigma_k,
                          rot_floor_rad=opts.gate_rot_floor_rad, pos_floor_m=opts.gate_pos_floor_m,
                          branch_jump_rad=opts.branch_jump_rad, branch_rate_factor=opts.branch_rate_factor)
        keep = gate.keep.copy()
        diag["gating"] = gate.summary() | {"rejected_t_ns": samples.t_ns[~keep].tolist()}
    else:
        keep = np.ones(n, bool)
    kept = np.flatnonzero(keep)
    if len(kept) < 2:
        raise TrajectoryError("fewer than two samples survive gating")
    diag["rotation_alias_warnings"] = check_rotation_aliasing(samples.q[kept])

    h = int(opts.h_ns) if opts.h_ns is not None else round(period)
    t_first, t_last = int(samples.t_ns[kept[0]]), int(samples.t_ns[kept[-1]])
    span = (t_first, t_last) if opts.span_ns is None else (int(opts.span_ns[0]), int(opts.span_ns[1]))
    if span[0] > t_first or span[1] < t_last:
        raise ValueError("span_ns must contain every kept sample")
    grid = KnotGrid.covering(span[0], span[1], h, opts.order)
    # Presence fallback. Outside its kept samples a body's controls are set by the jerk penalty alone until the
    # presence holds tie them at the end. On a data-free stretch of L controls the system there is lambda P, and the
    # smallest eigenvalue of P's leading minors falls as L^-6 (L^-2m, m = 3); past about 700 controls (3.5 s at
    # 200 Hz) it is below the rounding of P's float64 entries, a leading minor is indefinite and the band Cholesky
    # fails (P alone on 1500 controls: minor 861). A7's drone_01, a rejected orientation branch, leaves 668 free. Such
    # a fit refits once with the presence groups tied from round 0, the hold its result gets anyway; a fit that
    # factorizes keeps its path bit for bit.
    try:
        r = _robust_fit(samples, keep, opts, grid, span, t_first, t_last)
    except np.linalg.LinAlgError as e:
        outside = [(a, b) for a, b in ((span[0], t_first), (t_last, span[1])) if a < b]
        early, _ = hold_groups(grid, outside)
        if not early:
            raise
        r = _robust_fit(samples, keep, opts, grid, span, t_first, t_last, early)
        fallback = {"error": str(e), "groups": [list(g) for g in early]}
        log.warning("smoothing fit: %s with %s outside the kept samples free; refitted with the presence holds tied "
                    "from round 0", e, " and ".join(f"{(b - a) * 1e-9:.2f} s" for a, b in outside))
    else:
        fallback = None
    fit, lam_p, lam_r, info_p, info_r = r.fit, r.lam_p, r.lam_r, r.info_p, r.info_r
    irls_iters, gb_rejected, static, rejected, presence, whole, still = (
        r.irls_iters, r.gb_rejected, r.static, r.rejected, r.presence, r.whole, r.still)

    traj = _make_trajectory(fit.spline, still, span)
    _check_increments(traj)

    w_final, _ = fit.current_weights()
    weights = np.zeros(n)
    weights[fit.idx] = w_final
    rr, rp = fit.residuals()
    disp_still = _still_displacement(traj, opts) if still else 0.0
    m = grid.order // 2
    h_s = h * 1e-9
    diag.update({
        "order": grid.order, "h_ns": h, "t0_ns": grid.t0_ns, "num_ctrl": grid.num_ctrl,
        "span_ns": list(span),
        "lambda_pos_u": lam_p, "lambda_rot_u": lam_r,
        "lambda_pos_si": lam_p * h_s ** (2 * m - 1), "lambda_rot_si": lam_r * h_s ** (2 * m - 1),
        "q_pos": fit.sigma_p**2 / (lam_p * h_s ** (2 * m - 1)), "q_rot": fit.sigma_r**2 / (lam_r * h_s ** (2 * m - 1)),
        "cutoff_hz_pos": _cutoff_hz(lam_p * h_s ** (2 * m - 1), period * 1e-9, m),
        "cutoff_hz_rot": _cutoff_hz(lam_r * h_s ** (2 * m - 1), period * 1e-9, m),
        "edf_pos": info_p["edf"], "edf_rot": info_r["edf"],
        "gcv_pos": {k: v for k, v in info_p.items() if k in ("log10_grid", "gcv", "at_grid_edge")},
        "gcv_rot": {k: v for k, v in info_r.items() if k in ("log10_grid", "gcv", "at_grid_edge")},
        "sigma_pos_m": fit.sigma_p, "sigma_rot_rad": fit.sigma_r,
        "rms_pos_m": float(np.sqrt(np.mean(np.sum(rp**2, 1)))),
        "rms_rot_rad": float(np.sqrt(np.mean(np.sum(rr**2, 1)))),
        "max_pos_m": float(np.linalg.norm(rp, axis=1).max()), "max_rot_rad": float(np.linalg.norm(rr, axis=1).max()),
        "irls_iterations": irls_iters, "min_weight": float(w_final.min()), "mean_weight": float(w_final.mean()),
        "num_kept": len(fit.t), "gb_rejected_t_ns": gb_rejected,
        "hold_rule": opts.hold_rule, "holds_ns": traj.holds.tolist(),
        "static_intervals": [s.__dict__ for s in static],
        "rejected_static_candidates": rejected, "presence_holds_ns": [list(p) for p in presence],
        "still_max_displacement_m": disp_still,
        "max_increment_rad": float(rotation_angle(traj.rot[:-1], traj.rot[1:]).max()),
        "gaps_ns": _gaps(fit.t, period),
        "static": bool(whole),
    })
    if fallback is not None:
        diag["presence_fallback"] = fallback
    log.info("smoothing fit: %d/%d samples kept, lambda_p %.3g (%.1f Hz), lambda_R %.3g (%.1f Hz), %d holds%s",
             diag["num_kept"], n, lam_p, diag["cutoff_hz_pos"], lam_r, diag["cutoff_hz_rot"], len(traj.holds),
             ", static" if whole else "")
    return FitResult(traj, diag, fit.keep, weights)


@dataclass
class _Robust:
    fit: _Smoother
    lam_p: float
    lam_r: float
    info_p: dict
    info_r: dict
    irls_iters: list
    gb_rejected: list
    static: list
    rejected: list
    presence: list
    whole: bool
    still: list


def _robust_fit(samples: PoseSamples, keep: np.ndarray, opts: SmoothingOptions, grid: KnotGrid, span, t_first: int,
                t_last: int, early: list[tuple[int, int]] | None = None) -> _Robust:
    """fit_smoothing's robust part on one grid (round 0, the robust rounds, G-b, holds); early: tie groups held from
    round 0 (the presence fallback), none on the normal path."""
    fit = _Smoother(samples, keep, opts, grid)
    if early:
        fit.set_ties(early)

    # Round 0: unweighted fit with lambda by GCV, then sigma from its residuals.
    w1 = np.ones(len(fit.t))
    lam_p, info_p = (opts.lambda_pos, None) if opts.lambda_pos is not None else fit.gcv_positions(w1)
    fit.solve_positions(lam_p, w1)
    lam_r = opts.lambda_rot if opts.lambda_rot is not None else lam_p
    fit.rotation_steps(lam_r, w1, steps=20)
    if opts.lambda_rot is None:
        lam_r, info_r = fit.gcv_rotations(w1)
        fit.rotation_steps(lam_r, w1, steps=20)
    if info_p is None:
        info_p = _fixed_lambda_info(fit, lam_p, w1, "pos")
    if opts.lambda_rot is not None:
        info_r = _fixed_lambda_info(fit, lam_r, w1, "rot")
    fit.update_sigmas(info_p, info_r)

    # Robust rounds: IRLS at fixed lambda, then GCV with the converged weights; repeat only while
    # lambda still moves by more than a quarter decade (warm starts keep the repeats short).
    irls_iters = []
    for _ in range(3):
        irls_iters.append(fit.irls(lam_p, lam_r))
        w, _ = fit.current_weights()
        new_p, info_p = (lam_p, _fixed_lambda_info(fit, lam_p, w, "pos")) if opts.lambda_pos is not None \
            else fit.gcv_positions(w)
        new_r, info_r = (lam_r, _fixed_lambda_info(fit, lam_r, w, "rot")) if opts.lambda_rot is not None \
            else fit.gcv_rotations(w)
        fit.update_sigmas(info_p, info_r)
        moved = abs(math.log10(new_p / lam_p)) > 0.25 or abs(math.log10(new_r / lam_r)) > 0.25
        lam_p, lam_r = new_p, new_r
        if not moved:
            break
    irls_iters.append(fit.irls(lam_p, lam_r))

    # Stage G-b on the unconstrained robust fit: residuals there are measurement noise. (After
    # holds they also contain the coloured Vicon wander that a hold deliberately ignores.)
    rr, rp = fit.residuals()
    maha = np.sqrt(np.sum(rr**2, 1) / fit.sigma_r**2 + np.sum(rp**2, 1) / fit.sigma_p**2)
    bad = maha > opts.mahalanobis_max
    gb_rejected = fit.t[bad].tolist()
    if bad.any() and (~bad).sum() >= 2:
        fit.drop(bad)
        irls_iters.append(fit.irls(lam_p, lam_r))

    # Holds: static stretches on the gated samples, plus presence holds outside the data.
    static, rejected, presence = [], [], []
    if opts.detect_holds and opts.hold_rule == "image":
        static, rejected = detect_static_image(fit.t, fit.spline.evaluate(fit.t), opts.hold_view, fit.idx,
                                               center=opts.center_body_m, radius=opts.radius_m,
                                               min_duration_ns=opts.hold_min_ns, max_disp_px=opts.hold_max_disp_px,
                                               min_piece_ns=2 * (grid.order - 1) * grid.h_ns)
    elif opts.detect_holds:
        ev = fit.spline.evaluate(fit.t)
        speed = sphere_speed(ev["v"], ev["omega"], ev["q"], opts.center_body_m, opts.radius_m)
        static, rejected = detect_static(fit.t, fit.qm, fit.pm, speed, center=opts.center_body_m,
                                         radius=opts.radius_m, max_disp=opts.hold_max_disp_m,
                                         min_duration_ns=opts.hold_min_ns, moving_speed=opts.hold_moving_mps)
    intervals = [(s.t_begin_ns, s.t_end_ns) for s in static]
    whole = len(static) == 1 and static[0].i0 == 0 and static[0].i1 == len(fit.t) - 1
    if whole:
        # Static throughout (SCN-03): one hold over the whole span.
        intervals = [(span[0], span[1])]
    else:
        if span[0] < t_first:
            presence.append((span[0], t_first))
        if span[1] > t_last:
            presence.append((t_last, span[1]))
    groups, still = hold_groups(grid, intervals + presence)
    if groups:
        fit.set_ties(groups)
        irls_iters.append(fit.irls(lam_p, lam_r))
    return _Robust(fit, lam_p, lam_r, info_p, info_r, irls_iters, gb_rejected, static, rejected, presence, whole,
                   still)


def _fixed_lambda_info(fit: _Smoother, lam: float, w, block: str) -> dict:
    """RSS and edf at a user-fixed lambda (so sigma can still be estimated)."""
    if block == "pos":
        btwb, pen, rhs = fit._pos_system(w)
        a = btwb + lam * pen
        u = fit.grid.order - 1
        (x,), (tr,) = _solve_and_trace([a], [rhs], btwb, u)
        r = fit.b @ (fit.spline.g1_var @ x) - fit.pm
        return {"edf": tr, "rss": float(np.sum(w[:, None] * r**2)), "at_grid_edge": False}
    rs = fit._rot_system(w)
    u = 3 * fit.grid.order - 1
    _, (tr,) = _solve_and_trace([(rs.jtwj + lam * rs.dtgd).tocsr()], [-(rs.grad_data + lam * rs.grad_pen)], rs.jtwj, u)
    return {"edf": tr, "rss": float(np.sum(w * np.sum(rs.rr**2, 1))), "at_grid_edge": False}


def _cutoff_hz(lam_si: float, period_s: float, m: int) -> float:
    """-6 dB frequency of the equivalent Wiener filter, (2 pi f_c)^(2m) = 1 / (lambda T) (M2 §C.6)."""
    return float((1.0 / (lam_si * period_s)) ** (1.0 / (2 * m)) / (2 * math.pi))


def _gaps(t: np.ndarray, period: float) -> list[list[int]]:
    dt = np.diff(t)
    big = np.flatnonzero(dt > 1.5 * period)
    return [[int(t[i]), int(t[i + 1])] for i in big]


def _still_displacement(traj: Trajectory, opts: SmoothingOptions) -> float:
    """Max sphere displacement inside every hold, evaluated on the fitted spline (must be 0)."""
    worst = 0.0
    for a, b in traj.holds:
        t = np.linspace(a, b, 64).astype(np.int64)
        t = t[t < b]
        k = evaluate_controls(traj.K, traj.t0_ns, traj.h_ns, traj.rot, traj.pos, t)
        d = sphere_displacement(np.broadcast_to(k["q"][0], k["q"].shape), np.broadcast_to(k["p"][0], k["p"].shape),
                                k["q"], k["p"], opts.center_body_m, opts.radius_m)
        worst = max(worst, float(d.max()))
    return worst


def _make_trajectory(spline: _Spline, still: list[tuple[int, int]], span: tuple[int, int]) -> Trajectory:
    holds = np.array(sorted(still), np.int64).reshape(-1, 2)
    return Trajectory(spline.k, spline.grid.t0_ns, spline.grid.h_ns, spline.rot.copy(), spline.pos.copy(),
                      holds, (int(span[0]), int(span[1])))


def _check_increments(traj: Trajectory) -> None:
    worst = float(rotation_angle(traj.rot[:-1], traj.rot[1:]).max())
    if worst > math.pi / 2:
        raise TrajectoryError(f"per-knot rotation increment {math.degrees(worst):.1f} deg exceeds 90 deg; "
                              "halve the knot spacing (M2 §C.5)")


# ---------------------------------------------------------------------------------------------
# Interpolation (authored keyframes)
# ---------------------------------------------------------------------------------------------


def _same_pose(q_a, p_a, q_b, p_b) -> np.ndarray:
    """Bitwise-identical poses (q and -q are the same rotation). Exact comparison on purpose: a
    computed angle of R_a^T R_b is ~1e-17, not 0, for identical inputs."""
    q_a, q_b = np.atleast_2d(q_a), np.atleast_2d(q_b)
    same_q = np.all(q_a == q_b, axis=1) | np.all(q_a == -q_b, axis=1)
    return np.all(np.atleast_2d(p_a) == np.atleast_2d(p_b), axis=1) & same_q


def identical_key_runs(keys: PoseSamples) -> list[tuple[int, int]]:
    """Runs of consecutive identical keys (index ranges, inclusive); a run is a hold (DESIGN §5.6:
    "a hold is a run of identical keys, nothing else")."""
    same = _same_pose(keys.q[1:], keys.p[1:], keys.q[:-1], keys.p[:-1])
    runs, i = [], 0
    while i < len(same):
        if same[i]:
            j = i
            while j < len(same) and same[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def default_keyframe_spacing(t_ns: np.ndarray, max_h_ns: int = 20_000_000) -> int:
    """Knot spacing for keys: the largest divisor of the gcd of key offsets that is <= max_h_ns, so
    keys sit on knots and holds are knot-aligned. Keys off any common lattice of at least max_h/4
    fall back to min(max_h, smallest key gap)."""
    diffs = np.diff(np.asarray(t_ns, np.int64))
    g = 0
    for d in diffs:
        g = gcd(g, int(d))
    k = -(-g // max_h_ns)
    while g % k and g // k >= max_h_ns // 4:
        k += 1
    if g % k == 0 and g // k >= max_h_ns // 4:
        return int(g // k)
    return int(min(max_h_ns, diffs.min()))


class _GridTooCoarse(Exception):
    """The knot spacing cannot represent the keys: an increment > pi/2, two holds sharing nodes,
    or keys next to a hold that outnumber the free control points there (singular KKT; within K-1
    knots of a hold most of a segment's control points are tied)."""


def fit_interpolating(keys: PoseSamples, *, order: int = 6, h_ns: int | None = None,
                      span_ns: tuple[int, int] | None = None, max_h_ns: int = 20_000_000,
                      tol: float = 1e-9, max_iterations: int = 60) -> FitResult:
    """Minimum-roughness spline through the keys, exactly still over runs of identical keys and,
    when span_ns extends beyond the keys, at the first/last key pose outside them.

    Positions: KKT [[P, B^T], [B, 0]]; rotations: SQP on the same penalty with linearized key
    constraints (M2 §C.7). The spline passes every key to |r| < tol (rad, m). The knot spacing is
    halved until every per-knot increment is <= pi/2 and distinct holds get distinct nodes."""
    if len(keys) < 2:
        raise TrajectoryError("interpolation needs at least two keys")
    check_rotation_aliasing(keys.q)
    span = (int(keys.t_ns[0]), int(keys.t_ns[-1])) if span_ns is None else (int(span_ns[0]), int(span_ns[1]))
    if span[0] > keys.t_ns[0] or span[1] < keys.t_ns[-1]:
        raise ValueError("span_ns must contain every key")
    lattice = np.concatenate([[span[0]], keys.t_ns])
    h = int(h_ns) if h_ns is not None else default_keyframe_spacing(lattice[lattice >= span[0]], max_h_ns)
    for _ in range(12):
        try:
            return _interpolate(keys, order, h, span, tol, max_iterations)
        except _GridTooCoarse:
            if h < 2:
                break
            h //= 2
    raise TrajectoryError("no knot spacing represents these keys (holds too close or turns too fast)")


def _interpolate(keys: PoseSamples, order: int, h: int, span, tol: float, max_iterations: int) -> FitResult:
    n = len(keys)
    grid = KnotGrid.covering(span[0], span[1], h, order)
    runs = identical_key_runs(keys)
    in_hold = np.zeros(n, bool)
    holds = []                                     # (closed interval, anchor key index)
    for a, b in runs:
        in_hold[a:b + 1] = True
        holds.append(((int(keys.t_ns[a]), int(keys.t_ns[b])), a))
    if span[0] < keys.t_ns[0]:
        holds.append(((span[0], int(keys.t_ns[0])), 0))
        in_hold[0] = True
    if span[1] > keys.t_ns[-1]:
        holds.append(((int(keys.t_ns[-1]), span[1]), n - 1))
        in_hold[-1] = True
    groups, still = hold_groups(grid, [iv for iv, _ in holds])
    tie = TieMap(grid.num_ctrl, groups)
    fixed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for (ta, _), i in holds:
        fi = int(tie.free_index[grid.controls_touching(ta, ta)[0]])
        if fi in fixed and not _same_pose(fixed[fi][0], fixed[fi][1], keys.q[i], keys.p[i])[0]:
            raise _GridTooCoarse
        fixed[fi] = (keys.q[i], keys.p[i])
    spline = _Spline(grid, _slerp_init(keys.t_ns, keys.q, grid.greville_ns()), np.zeros((grid.num_ctrl, 3)),
                     tie, fixed)
    free_keys = np.flatnonzero(~in_hold)
    t_free = keys.t_ns[free_keys]
    nv = len(spline.var)
    iters = 0
    if nv:
        try:
            _interpolate_positions(spline, grid.design_matrix(t_free), keys.p[free_keys])
            for iters in range(1, max_iterations + 1):
                step, r_max = _sqp_rotation_step(spline, t_free, keys.q[free_keys])
                # An ill-conditioned (nearly singular) KKT shows up as a huge or non-finite step.
                if not np.all(np.isfinite(step)) or np.abs(step).max() > math.pi:
                    raise _GridTooCoarse
                spline.apply_rotation_step(step)
                if np.abs(step).max() < 1e-13 and r_max < tol:
                    break
        except RuntimeError as e:   # SuperLU: "Factor is exactly singular"
            raise _GridTooCoarse from e
        if not np.all(np.isfinite(spline.pos)):
            raise _GridTooCoarse
    traj = _make_trajectory(spline, still, span)
    if rotation_angle(traj.rot[:-1], traj.rot[1:]).max() > math.pi / 2:
        raise _GridTooCoarse
    ev = evaluate_controls(order, grid.t0_ns, h, traj.rot, traj.pos, keys.t_ns)
    err_r = rotation_angle(ev["q"], keys.q)
    err_p = np.linalg.norm(ev["p"] - keys.p, axis=1)
    if err_r.max() >= tol or err_p.max() >= tol:
        raise _GridTooCoarse
    diag = {"mode": "interpolating", "order": order, "h_ns": h, "t0_ns": grid.t0_ns, "num_ctrl": grid.num_ctrl,
            "num_keys": n, "span_ns": list(span), "holds_ns": traj.holds.tolist(), "sqp_iterations": iters,
            "max_key_error_rad": float(err_r.max()), "max_key_error_m": float(err_p.max()),
            "max_increment_rad": float(rotation_angle(traj.rot[:-1], traj.rot[1:]).max())}
    return FitResult(traj, diag, np.ones(n, bool), np.ones(n))


def _interpolate_positions(spline: _Spline, b: sp.csr_matrix, y: np.ndarray) -> None:
    """min x^T P x subject to B x = y, x = G_var x_var + x_fixed (KKT, exact)."""
    gv = spline.g1_var
    x_fix = spline.fixed_positions_full()
    pen = (gv.T @ spline.penalty_pos @ gv).tocsc()
    rhs = np.zeros((gv.shape[1], 3))
    if x_fix is not None:
        rhs = -(gv.T @ (spline.penalty_pos @ x_fix))
        y = y - b @ x_fix
    if len(y):
        bv = (b @ gv).tocsc()
        kkt = sp.bmat([[pen, bv.T], [bv, None]], format="csc")
        sol = spla.splu(kkt).solve(np.vstack([rhs, y]))[: gv.shape[1]]
    else:
        sol = spla.splu(pen).solve(rhs)
    spline.set_positions(sol, x_fix)


def _sqp_rotation_step(spline: _Spline, t_ns: np.ndarray, q_keys: np.ndarray) -> tuple[np.ndarray, float]:
    """One Gauss-Newton SQP step: min |e + D delta|^2_Gamma s.t. r + J delta = 0."""
    e, de = spline.rotation_penalty()
    de = de @ spline.g3_var
    hess = (de.T @ spline.gamma3 @ de).tocsc()
    grad = de.T @ (spline.gamma3 @ e)
    if len(t_ns) == 0:
        return spla.splu(hess).solve(-grad), 0.0
    r = _rotation_residuals(q_keys, spline.evaluate(t_ns)["q"])
    jac = (spline.rotation_jacobian(t_ns, r) @ spline.g3_var).tocsc()
    kkt = sp.bmat([[hess, jac.T], [jac, None]], format="csc")
    step = spla.splu(kkt).solve(np.concatenate([-grad, -r.ravel()]))[: jac.shape[1]]
    return step, float(np.linalg.norm(r, axis=1).max())
