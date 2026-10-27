"""Re-timing of Vicon stamps onto their uniform frame grid (G1 rec. 2, M2 §C.1, DESIGN §5.4 step 2).

vicon_bridge stamps are the output of a second-order loop filter (poles 0.9 +- 0.3i) and ring at
10.24 Hz with 0.02-1.9 ms errors. The capture instants are uniform, so the frame index n is
recovered from the rounded stamp differences and t = a + b n is fitted by Huber regression. The
filter has unit DC gain, so the fit is unbiased; the raw stamps are kept for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Retiming:
    t_raw_ns: np.ndarray       # int64, as recorded
    frame_index: np.ndarray    # int64, 0 at the first stamp; jumps > 1 are gaps
    t_ns: np.ndarray           # int64, re-timed: round(a + b n)
    a_ns: float                # intercept (ns)
    b_ns: float                # fitted frame period (ns)
    residual_std_ns: float     # std of raw - re-timed stamps
    num_gaps: int

    def to_retimed(self, t_raw_ns: np.ndarray) -> np.ndarray:
        """Map other times on the raw stamp clock (e.g. EVIMO2 GT frame times, which were rendered
        from poses interpolated on raw stamps) onto the re-timed clock by subtracting the stamp
        error g(t) = raw - re-timed, interpolated linearly between samples (G1 rec. 2)."""
        g = (self.t_raw_ns - self.t_ns).astype(np.float64)
        t = np.asarray(t_raw_ns, np.int64)
        return t - np.rint(np.interp(t.astype(np.float64), self.t_raw_ns.astype(np.float64), g)).astype(np.int64)

    def summary(self) -> dict:
        return {"a_ns": self.a_ns, "b_ns": self.b_ns, "residual_std_ns": self.residual_std_ns,
                "num_gaps": self.num_gaps, "num_samples": len(self.t_ns)}


def retime_stamps(t_raw_ns: np.ndarray, nominal_period_ns: int = 5_000_000, *,
                  huber_k: float = 1.345, iterations: int = 20) -> Retiming:
    """Fit t_raw = a + b n robustly; n from cumulative round(diff / nominal period)."""
    t = np.asarray(t_raw_ns, dtype=np.int64)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("re-timing needs at least two stamps")
    steps = np.rint(np.diff(t) / float(nominal_period_ns)).astype(np.int64)
    if np.any(steps < 1):
        raise ValueError("stamps must be strictly increasing by at least half the nominal period")
    n = np.concatenate([[0], np.cumsum(steps)])
    # Work relative to the first stamp so the fp64 regression keeps sub-ns precision.
    y = (t - t[0]).astype(np.float64)
    x = n.astype(np.float64)
    design = np.stack([np.ones_like(x), x], axis=1)
    w = np.ones_like(x)
    coef = np.zeros(2)
    for _ in range(iterations):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]
        r = y - design @ coef
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3  # 1e-3 ns floor for exact grids
        w_new = np.minimum(1.0, huber_k * scale / np.maximum(np.abs(r), 1e-12))
        if np.allclose(w_new, w, rtol=0.0, atol=1e-12):
            break
        w = w_new
    fit = design @ coef
    t_hat = t[0] + np.rint(fit).astype(np.int64)
    return Retiming(t, n, t_hat, float(t[0] + coef[0]), float(coef[1]),
                    float(np.std(y - fit)), int(np.sum(steps > 1)))
