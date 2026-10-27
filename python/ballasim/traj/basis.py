"""Uniform B-spline bases, knot grids and the exact roughness penalty (M2 §A.1, §A.4, §C.3, §C.5).

These are the linear-algebra building blocks of the fitter: design matrices that map control
points to samples and the banded penalty matrices. Evaluating a fitted trajectory is not done
here; it goes through ``ballasim._core`` (see :mod:`ballasim.traj.evaluate`).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import comb, factorial

import numpy as np
import scipy.sparse as sp


@cache
def _blend_exact(order: int) -> tuple[tuple[Fraction, ...], ...]:
    """M^(K) of M2 (A3): b(u) = M [1, u, ..., u^(K-1)]^T, exact rationals."""
    k = order
    rows = []
    for j in range(k):
        row = []
        for n in range(k):
            acc = sum((-1) ** (l - j) * comb(k, l - j) * (k - 1 - l) ** (k - 1 - n) for l in range(j, k))
            row.append(Fraction(comb(k - 1, n) * acc, factorial(k - 1)))
        rows.append(tuple(row))
    return tuple(rows)


def blend_matrix(order: int) -> np.ndarray:
    return np.array([[float(v) for v in row] for row in _blend_exact(order)])


def cumulative_matrix(order: int) -> np.ndarray:
    """M~^(K) of M2 (A4): lambda_j(u) = sum_{l >= j} b_l(u)."""
    m = _blend_exact(order)
    return np.array([[float(sum((m[l][n] for l in range(j, order)), Fraction(0))) for n in range(order)]
                     for j in range(order)])


def _powers(u: np.ndarray, order: int, deriv: int) -> np.ndarray:
    out = np.zeros((len(u), order))
    for n in range(deriv, order):
        out[:, n] = factorial(n) // factorial(n - deriv) * u ** (n - deriv)
    return out


def basis(u: np.ndarray, order: int, deriv: int = 0) -> np.ndarray:
    """b_j^(r)(u) for j = 0..K-1 (derivatives in u), shape (len(u), K)."""
    return _powers(np.asarray(u, np.float64), order, deriv) @ blend_matrix(order).T


def cumulative_basis(u: np.ndarray, order: int, deriv: int = 0) -> np.ndarray:
    """lambda_j^(r)(u) for j = 1..K-1, shape (len(u), K - 1)."""
    return _powers(np.asarray(u, np.float64), order, deriv) @ cumulative_matrix(order)[1:].T


@cache
def segment_gram(order: int, m: int) -> np.ndarray:
    """G_jj' = int_0^1 N_{K-m}(u + K-1-j) N_{K-m}(u + K-1-j') du for j, j' = m..K-1 (exact).

    The integrand is a polynomial of degree 2(K-m-1); K Gauss-Legendre nodes integrate it exactly.
    For K = 6, m = 3 this is [[6, 13, 1], [13, 54, 13], [1, 13, 6]] / 120 (M2 C4).
    """
    x, w = np.polynomial.legendre.leggauss(order)
    u, w = 0.5 * (x + 1.0), 0.5 * w
    b = basis(u, order - m)
    return (b * w[:, None]).T @ b


def difference_matrix(n: int, order: int) -> sp.csr_matrix:
    """Rows c = order..n-1 of the order-th backward difference: (D x)_c = Delta^order x_c."""
    if order == 0:
        return sp.identity(n, format="csr")
    coeff = [(-1) ** k * comb(order, k) for k in range(order + 1)]
    rows = n - order
    data = np.repeat(np.array(coeff[::-1], np.float64)[None, :], rows, axis=0).ravel()
    cols = (np.arange(rows)[:, None] + np.arange(order + 1)[None, :]).ravel()
    ptr = np.arange(rows + 1) * (order + 1)
    return sp.csr_matrix((data, cols, ptr), shape=(rows, n))


def difference_gram(num_ctrl: int, order: int, m: int) -> sp.csr_matrix:
    """Gamma: the banded Gram of the m-th differences, so that the roughness in u-units is
    sum_s int_0^1 |d^m x/du^m|^2 du = (D_m x)^T Gamma (D_m x)  (M2 C4, Lemma B2)."""
    g = segment_gram(order, m)
    k = order - m
    n_seg = num_ctrl - order + 1
    rows = num_ctrl - m
    # Segment s uses differences c = s+m .. s+K-1, i.e. rows s .. s+k-1 of D_m.
    r = (np.arange(n_seg)[:, None, None] + np.arange(k)[None, :, None]).repeat(k, axis=2)
    c = (np.arange(n_seg)[:, None, None] + np.arange(k)[None, None, :]).repeat(k, axis=1)
    v = np.broadcast_to(g, (n_seg, k, k))
    return sp.coo_matrix((v.ravel(), (r.ravel(), c.ravel())), shape=(rows, rows)).tocsr()


def roughness_matrix(num_ctrl: int, order: int, m: int) -> sp.csr_matrix:
    """P with x^T P x = sum_s int_0^1 |d^m p/du^m|^2 du; the SI integral is h^(1-2m) x^T P x."""
    d = difference_matrix(num_ctrl, m)
    return (d.T @ difference_gram(num_ctrl, order, m) @ d).tocsr()


@dataclass(frozen=True)
class KnotGrid:
    """Uniform knots t_c = t0 + c h (int64 ns); segment s covers [t0 + s h, t0 + (s+1) h)."""

    t0_ns: int
    h_ns: int
    order: int
    num_ctrl: int

    def __post_init__(self):
        if self.order not in (4, 6):
            raise ValueError("spline order must be 4 or 6")
        if not 1 <= self.h_ns < 2**31:
            raise ValueError(f"knot spacing {self.h_ns} ns outside [1, 2^31)")
        if self.num_ctrl < self.order:
            raise ValueError("need at least `order` control points")

    @classmethod
    def covering(cls, t_begin_ns: int, t_end_ns: int, h_ns: int, order: int) -> KnotGrid:
        """The grid with t0 = t_begin whose closed support [t0, t0 + N_seg h] covers t_end."""
        n_seg = max(1, -((t_begin_ns - t_end_ns) // h_ns))
        return cls(int(t_begin_ns), int(h_ns), order, n_seg + order - 1)

    @property
    def num_segments(self) -> int:
        return self.num_ctrl - self.order + 1

    @property
    def end_ns(self) -> int:
        return self.t0_ns + self.num_segments * self.h_ns

    def locate(self, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Segment index and u in [0, 1] in exact integer arithmetic; the span is closed on the right."""
        t = np.asarray(t_ns, dtype=np.int64)
        d = t - self.t0_ns
        s = d // self.h_ns
        r = d - s * self.h_ns
        end = (s == self.num_segments) & (r == 0)
        s = np.where(end, s - 1, s)
        r = np.where(end, self.h_ns, r)
        if np.any((s < 0) | (s >= self.num_segments)):
            raise ValueError("sample time outside the knot grid's support")
        return s, r.astype(np.float64) / self.h_ns

    def design_matrix(self, t_ns: np.ndarray, deriv: int = 0) -> sp.csr_matrix:
        """B with (B c)_n = sum_j b_j(u_n) c_{s_n + j} (derivative in u for deriv > 0)."""
        s, u = self.locate(t_ns)
        k = self.order
        data = basis(u, k, deriv).ravel()
        cols = (s[:, None] + np.arange(k)[None, :]).ravel()
        ptr = np.arange(len(s) + 1) * k
        return sp.csr_matrix((data, cols, ptr), shape=(len(s), self.num_ctrl))

    def greville_ns(self) -> np.ndarray:
        """Control-point centres tau_c = t0 + (c - K/2 + 1) h (M2 A5)."""
        return self.t0_ns + (np.arange(self.num_ctrl) - self.order // 2 + 1) * self.h_ns

    def controls_touching(self, t_begin_ns: int, t_end_ns: int) -> tuple[int, int]:
        """Control points whose basis function is non-zero somewhere on the closed [t_begin, t_end].

        Basis c is non-zero on the open interval (t0 + (c-K+1) h, t0 + (c+1) h), so c ranges over
        floor(x_b) .. ceil(x_e) + K - 2 with x = (t - t0)/h, clipped to the grid.
        """
        h, t0, k = self.h_ns, self.t0_ns, self.order
        c0 = (t_begin_ns - t0) // h
        c1 = -((t0 - t_end_ns) // h) + k - 2
        return max(0, int(c0)), min(self.num_ctrl - 1, int(c1))

    def still_interval(self, c0: int, c1: int) -> tuple[int, int]:
        """The knot-aligned [t_begin, t_end) on which the spline is exactly constant when control
        points c0..c1 are equal: segments whose K nodes all lie in c0..c1 (M2 Lemma B1)."""
        s0, s1 = c0, c1 - self.order + 1
        if s1 < s0:
            raise ValueError("fewer than K tied control points: the spline is not still anywhere")
        return self.t0_ns + s0 * self.h_ns, self.t0_ns + (s1 + 1) * self.h_ns
