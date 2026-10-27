"""Banded symmetric positive-definite algebra for the fitter (M2 §C.4, §C.6).

Normal matrices of B-spline least squares are banded (bandwidth K-1 control points), so every
solve is a LAPACK band Cholesky. GCV needs tr(A^-1 M) for many lambda; it comes from the band of
A^-1 by Takahashi's selected inversion, run for a batch of factors at once so the sequential
recursion is paid once per batch rather than once per lambda.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_solve_banded, cholesky_banded


def bandwidth(a: sp.spmatrix) -> int:
    coo = a.tocoo()
    return int(np.abs(coo.row - coo.col).max()) if coo.nnz else 0


def to_upper_band(a: sp.spmatrix, u: int) -> np.ndarray:
    """LAPACK upper band storage: ab[u + i - j, j] = a[i, j] for i <= j <= i + u."""
    csr = sp.csr_matrix(a, copy=True)
    csr.sum_duplicates()
    coo = csr.tocoo()
    keep = (coo.col >= coo.row) & (coo.col - coo.row <= u)
    ab = np.zeros((u + 1, a.shape[0]))
    ab[u + coo.row[keep] - coo.col[keep], coo.col[keep]] = coo.data[keep]
    return ab


class BandedSPD:
    """Cholesky factor of a sparse SPD matrix stored in band form."""

    def __init__(self, a: sp.spmatrix, u: int | None = None):
        self.u = bandwidth(a) if u is None else u
        self.ab = to_upper_band(a, self.u)
        self.factor = cholesky_banded(self.ab, lower=False, check_finite=True)

    def solve(self, b: np.ndarray) -> np.ndarray:
        return cho_solve_banded((self.factor, False), b, check_finite=False)

    def logdet(self) -> float:
        return 2.0 * float(np.sum(np.log(self.factor[self.u])))


def selected_inverse_band(factors: np.ndarray) -> np.ndarray:
    """Band of A^-1 from upper band Cholesky factors (A = U^T U), batched over the first axis.

    factors: (L, u+1, n) in LAPACK upper storage. Returns S with S[l, d, i] = (A_l^-1)[i, i+d]
    for d = 0..u (zero beyond the matrix). Takahashi recursion from U Sigma = U^-T, rows n-1 .. 0:
    Sigma_ij = (delta_ij / U_ii - sum_{k=i+1}^{i+u} U_ik Sigma_kj) / U_ii, off-diagonals first.
    The loop is sequential in i, so all L systems advance together.
    """
    lcount, ub, n = factors.shape
    u = ub - 1
    inv_d = (1.0 / factors[:, u, :]).T.copy()            # (n, L)
    pad = np.zeros((lcount, ub, n + u))
    pad[:, :, :n] = factors
    kk = np.arange(1, u + 1)
    # urows[i, l, k-1] = U_l[i, i+k], stored at ab[u - k, i + k]; columns past n are zero.
    urows = np.stack([pad[:, u - k, k:k + n] for k in kk], axis=-1).transpose(1, 0, 2).copy()
    s = np.zeros((n + u, ub, lcount))                    # s[row, d, l] = Sigma_l[row, row + d]
    dmat = np.abs(kk[:, None] - kk[None, :])             # Sigma[i+k, i+j] = s[i + min(k, j), |k - j|]
    base = np.minimum(kk[:, None], kk[None, :])
    for i in range(n - 1, -1, -1):
        ur = urows[i]
        off = np.einsum("lk,kjl->jl", ur, s[i + base, dmat]) * -inv_d[i]
        s[i, 1:] = off
        s[i, 0] = (inv_d[i] - np.einsum("lk,kl->l", ur, off)) * inv_d[i]
    return s[:n].transpose(2, 1, 0)


def trace_of_product(s_band: np.ndarray, m: sp.spmatrix) -> np.ndarray:
    """tr(A_l^-1 M) for each l, given the band of A_l^-1 and a symmetric M within that band."""
    _, ub, n = s_band.shape
    u = ub - 1
    mb = to_upper_band(m, u)                              # mb[u - d, i + d] = M[i, i+d]
    total = np.einsum("li,i->l", s_band[:, 0, :], mb[u])
    for d in range(1, u + 1):
        total += 2.0 * np.einsum("li,i->l", s_band[:, d, : n - d], mb[u - d, d:])
    return total
