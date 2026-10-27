"""Model-free outlier gating of pose samples, stage G-a (M2 §C.2, DESIGN §5.4 step 3).

Each sample is predicted leave-one-out from its +-W neighbours by robust (Tukey-biweight IRLS)
quadratics in time, fitted to rotation vectors in the tangent chart at the window medoid and to
positions. A sample is flagged when its prediction error exceeds max(8 sigma, 3 deg) or
max(8 sigma, 10 mm), sigma = 1.4826 MAD over the track. This rejects 180-degree flips and marker
swaps without the speed thresholds that falsely flag genuine wheel spins (M2 §C.2).
All windows are processed at once (vectorized), so a 60 s track gates in well under a second.

A flip that persists for longer than the window (a symmetric marker set solved the other way up for
a while, common for EVIMO2's drones and small objects) predicts itself and survives that test. A last
step therefore splits the kept samples at jumps between orientation branches and keeps the largest
branch (:func:`minority_branches`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ballasim import _core
from ballasim.scene.transforms import quat_conjugate, quat_multiply, rotation_angle, so3_exp, so3_log

TUKEY_C = 4.685
# The defaults of the tunables are the schema's (scene.trajectory.*, docs/dev/P0_PARAMS.md §3.4, D15a).
_default = _core.config.default
#: Branch jumps: a turn by more than this between consecutive kept samples ...
BRANCH_JUMP_RAD = _default("scene.trajectory.branch_jump_rad")
#: ... at more than this factor times the track's 95th-percentile turn rate between kept samples ...
BRANCH_RATE_FACTOR = _default("scene.trajectory.branch_rate_factor")
#: ... or by more than this at any rate: the fit cannot tell the direction of such a turn
#: (fit.check_rotation_aliasing refuses it), so the track is split there instead.
ALIAS_RAD = 0.9 * np.pi


@dataclass(frozen=True)
class GateResult:
    keep: np.ndarray           # bool[n]
    err_rot: np.ndarray        # rad, leave-one-out prediction error
    err_pos: np.ndarray        # m
    threshold_rot: float
    threshold_pos: float
    half_window: int
    sigma_rot: float           # 1.4826 MAD of err_rot
    sigma_pos: float
    num_branch_rejected: int = 0   # samples of minority orientation branches (persistent flips)

    def summary(self) -> dict:
        return {"num_rejected": int((~self.keep).sum()), "threshold_rot_rad": self.threshold_rot,
                "threshold_pos_m": self.threshold_pos, "half_window": self.half_window,
                "sigma_rot_rad": self.sigma_rot, "sigma_pos_m": self.sigma_pos,
                "num_branch_rejected": self.num_branch_rejected}


def _masked_median(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median over the last axis of the entries where `valid` (0 where none is valid)."""
    srt = np.sort(np.where(valid, x, np.inf), axis=-1)       # valid entries first
    c = valid.sum(axis=-1, keepdims=True)
    lo = np.take_along_axis(srt, np.maximum(c - 1, 0) // 2, axis=-1)[..., 0]
    hi = np.take_along_axis(srt, np.minimum(c // 2, x.shape[-1] - 1), axis=-1)[..., 0]
    return np.where(c[..., 0] > 0, 0.5 * (lo + hi), 0.0)


def _tukey_quadratics(x: np.ndarray, y: np.ndarray, valid: np.ndarray, iterations: int = 10) -> np.ndarray:
    """Batched robust quadratic fits y ~ c0 + c1 x + c2 x^2 per row. x: (n, m); y: (n, m, 3).
    Returns c0 (n, 3), the leave-one-out prediction at x = 0."""
    n = x.shape[0]
    a = np.stack([np.ones_like(x), x, x * x], axis=-1)        # (n, m, 3)
    w = valid.astype(np.float64)
    frozen = np.zeros(n, bool)
    coef = np.zeros((n, 3, 3))
    for _ in range(iterations):
        aw = a * w[..., None]
        ata = np.einsum("nmi,nmj->nij", aw, a) + 1e-18 * np.eye(3)
        coef = np.linalg.solve(ata, np.einsum("nmi,nmk->nik", aw, y))
        r = np.linalg.norm(y - np.einsum("nmi,nik->nmk", a, coef), axis=-1)
        s = 1.4826 * _masked_median(r, valid) + 1e-12
        uu = r / (TUKEY_C * s[:, None])
        w_new = np.where(valid & (uu < 1.0), (1.0 - uu * uu) ** 2, 0.0)
        # A window whose support collapses keeps the plain least-squares fit (as in M2 gating.py).
        collapse = (w_new > 0).sum(axis=1) < 3
        frozen |= collapse
        w = np.where(frozen[:, None], valid.astype(np.float64), w_new)
    if frozen.any():
        aw = a * valid[..., None]
        ata = np.einsum("nmi,nmj->nij", aw, a) + 1e-18 * np.eye(3)
        coef[frozen] = np.linalg.solve(ata, np.einsum("nmi,nmk->nik", aw, y))[frozen]
    return coef[:, 0, :]


def _loo_errors(t: np.ndarray, q: np.ndarray, p: np.ndarray, targets: np.ndarray, pool: np.ndarray, half: int):
    """Leave-one-out prediction errors of samples `targets` from their +-half nearest neighbours in
    `pool` (sorted sample indices, may contain the targets themselves, which are skipped).
    Returns (err_rot, err_pos, enough)."""
    pos = np.searchsorted(pool, targets)
    own = np.isin(targets, pool).astype(np.int64)
    slots = np.concatenate([pos[:, None] - np.arange(half, 0, -1)[None, :],
                            pos[:, None] + own[:, None] + np.arange(half)[None, :]], axis=1)
    valid = (slots >= 0) & (slots < len(pool))
    idx = pool[np.clip(slots, 0, len(pool) - 1)]
    n = len(targets)
    enough = valid.sum(axis=1) >= 4
    # Window medoid as the chart centre: smallest median geodesic distance to the others.
    qw = q[idx]                                               # (n, m, 4)
    dist = rotation_angle(qw[:, :, None, :], qw[:, None, :, :])
    med = np.where(valid, _masked_median(dist, np.broadcast_to(valid[:, None, :], dist.shape)), np.inf)
    ref = qw[np.arange(n), np.argmin(med, axis=1)]          # (n, 4)
    phi = so3_log(quat_multiply(quat_conjugate(ref)[:, None, :], qw))
    x = (t[idx] - t[targets][:, None]) * 1e-9
    c_rot = _tukey_quadratics(x, phi, valid)
    c_pos = _tukey_quadratics(x, p[idx] - p[targets][:, None, :], valid)
    pred = quat_multiply(ref, so3_exp(c_rot))
    err_rot = np.where(enough, rotation_angle(pred, q[targets]), 0.0)
    err_pos = np.where(enough, np.linalg.norm(c_pos, axis=1), 0.0)
    return err_rot, err_pos, enough


def minority_branches(t_ns: np.ndarray, q: np.ndarray, keep: np.ndarray, *, jump_rad: float = BRANCH_JUMP_RAD,
                      rate_factor: float = BRANCH_RATE_FACTOR) -> np.ndarray:
    """Kept samples outside the largest orientation branch (bool over all samples).

    A jump is a step between consecutive kept samples that turns by more than jump_rad at more
    than rate_factor times the 95th-percentile turn rate of those steps (a genuine spin stays far
    below it: a 7 rev/s wheel turns 13 deg per 5 ms sample, and so does a genuine turn across a short
    Vicon gap), or by more than ALIAS_RAD, whose direction no fit can tell. The runs between jumps form
    branches in order: a run continues the branch whose latest
    orientation is nearest to its first sample if that is within jump_rad, else it starts a new
    branch. Every branch but the one with the most samples is rejected; without jumps nothing is."""
    t = np.asarray(t_ns, np.int64)
    kept = np.flatnonzero(keep)
    out = np.zeros(len(t), bool)
    if len(kept) < 3:
        return out
    steps = rotation_angle(q[kept[:-1]], q[kept[1:]])
    dt_s = np.diff(t[kept]) * 1e-9
    rate95 = float(np.percentile(steps / dt_s, 95))
    jump = ((steps > jump_rad) & (steps > rate_factor * rate95 * dt_s)) | (steps > ALIAS_RAD)
    if not jump.any():
        return out
    starts = np.concatenate([[0], np.flatnonzero(jump) + 1])
    ends = np.concatenate([starts[1:], [len(kept)]])
    latest: list[np.ndarray] = []     # orientation of each branch's latest sample
    counts: list[int] = []
    label = np.empty(len(kept), np.int64)
    for a, b in zip(starts, ends):
        d = rotation_angle(np.array(latest), q[kept[a]][None, :]) if latest else np.empty(0)
        j = int(np.argmin(d)) if len(d) else -1
        if j < 0 or d[j] > jump_rad:
            j = len(latest)
            latest.append(q[kept[b - 1]])
            counts.append(0)
        latest[j] = q[kept[b - 1]]
        counts[j] += int(b - a)
        label[a:b] = j
    out[kept[label != int(np.argmax(counts))]] = True
    return out


def gate_poses(t_ns: np.ndarray, q: np.ndarray, p: np.ndarray, *, period_ns: float,
               sigma_k: float = _default("scene.trajectory.gate_sigma_k"),
               rot_floor_rad: float = _default("scene.trajectory.gate_rot_floor_rad"),
               pos_floor_m: float = _default("scene.trajectory.gate_pos_floor_m"),
               branch_jump_rad: float = BRANCH_JUMP_RAD, branch_rate_factor: float = BRANCH_RATE_FACTOR) -> GateResult:
    """Stage G-a. A second pass re-predicts every flagged sample from unflagged neighbours only, so
    the neighbours of a gross outlier, whose windows contained it, are not rejected with it (with
    the minimum window of +-3 samples one 176-degree flip otherwise drags its neighbours along).
    Finally the minority orientation branches of persistent flips are rejected (minority_branches,
    with branch_jump_rad and branch_rate_factor), and the samples flagged next to them are judged again
    from windows without them."""
    t = np.asarray(t_ns, np.int64)
    q = np.asarray(q, np.float64)
    p = np.asarray(p, np.float64)
    n = len(t)
    if n < 5:
        z = np.zeros(n)
        return GateResult(np.ones(n, bool), z, z, np.inf, np.inf, 0, 0.0, 0.0)
    rates = rotation_angle(q[:-1], q[1:]) / (np.diff(t) * 1e-9)
    w95 = float(np.percentile(rates, 95))
    half = int(np.clip(np.floor(np.pi / (4.0 * max(w95, 1e-6) * period_ns * 1e-9)), 3, 5))
    everyone = np.arange(n)
    err_rot, err_pos, enough = _loo_errors(t, q, p, everyone, everyone, half)
    sr = 1.4826 * float(np.median(err_rot[enough])) if enough.any() else 0.0
    sp_ = 1.4826 * float(np.median(err_pos[enough])) if enough.any() else 0.0
    thr_r = max(sigma_k * sr, rot_floor_rad)
    thr_p = max(sigma_k * sp_, pos_floor_m)
    bad = (err_rot > thr_r) | (err_pos > thr_p)
    if bad.any() and (~bad).sum() >= 5:
        flagged = np.flatnonzero(bad)
        er2, ep2, ok2 = _loo_errors(t, q, p, flagged, np.flatnonzero(~bad), half)
        err_rot[flagged], err_pos[flagged] = er2, ep2
        bad[flagged] = ok2 & ((er2 > thr_r) | (ep2 > thr_p)) | (~ok2 & bad[flagged])
    branch = minority_branches(t, q, ~bad, jump_rad=branch_jump_rad, rate_factor=branch_rate_factor)
    if branch.any() and bad.any():
        # The neighbours of a flipped run were judged from windows that held it: judge them again without it.
        flagged = np.flatnonzero(bad)
        er3, ep3, ok3 = _loo_errors(t, q, p, flagged, np.flatnonzero(~(bad | branch)), half)
        err_rot[flagged], err_pos[flagged] = er3, ep3
        bad[flagged] = ok3 & ((er3 > thr_r) | (ep3 > thr_p)) | (~ok3 & bad[flagged])
    return GateResult(~(bad | branch), err_rot, err_pos, thr_r, thr_p, half, sr, sp_, int(branch.sum()))
