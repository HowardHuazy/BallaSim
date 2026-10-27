// Rotations as the Python scene side computes them (docs/dev/CPP_PORT.md item 5, E2): SciPy 1.18's Rotation and
// Slerp (Cython backend) and ballasim.scene.transforms, operation for operation.
//
// Quaternions are stored (w, x, y, z) as everywhere in BallaSim; SciPy's functions take and return them in that
// layout but sum in SciPy's (x, y, z, w) order where it matters (its normalization). The SO(3) maps are the core's
// (core/math/so3.h, what ballasim._core.so3_exp / so3_log call).
#pragma once

#include <span>
#include <utility>
#include <vector>

#include "core/math/quat.h"
#include "core/math/vec.h"

namespace balla::scene::rot {

using Q = Quat<double>;
using V = Vec3<double>;
using M = Mat3<double>;

// --- scipy.spatial.transform.Rotation ---------------------------------------------------------------------------
// Rotation.from_quat: q / sqrt(((x^2 + y^2) + z^2) + w^2).
Q sp_normalize(Q q);
// Rotation.as_matrix of the stored quaternion.
M sp_as_matrix(Q q);
// (p * q) of two stored quaternions, normalized as Rotation.__mul__ does.
Q sp_compose(Q p, Q q);
// Rotation.as_rotvec / Rotation.from_rotvec (raw, not normalized).
V sp_as_rotvec(Q q);
Q sp_from_rotvec(V v);
// Rotation.from_matrix (Shepperd's method, normalized); throws for a non-positive determinant or a matrix that is
// not orthogonal to 1e-12 (SciPy then projects it by SVD, which this port does not reproduce).
Q sp_from_matrix(const M& m);
// Rotation.mean() of stored quaternions: the eigenvector of np.dot(1 * q^T, q) (cblas_dgemm) with the largest
// eigenvalue (np.linalg.eigh), raw.
Q sp_mean(std::span<const Q> q);

// --- ballasim.scene.transforms ----------------------------------------------------------------------------------
Q normalize_quat(Q q);
Q canonical_quat(Q q);
Q quat_multiply(Q a, Q b);
Q quat_conjugate(Q q);
M quat_to_matrix(Q q);
// quat_rotate: np.einsum("...ij,...j->...i", R, v), whose 3-term sums NumPy pairs as (r0 v0 + r2 v2) + r1 v1.
V quat_rotate(Q q, V v);
V relative_rotvec(Q a, Q b);
double rotation_angle(Q a, Q b);
// so3_exp as ballasim._core.so3_exp (refuses |phi|^2 > 4e12) and so3_log.
Q so3_exp(V phi);
V so3_log(Q q);
// np.cross of two 3-vectors.
V cross(V a, V b);
// np.linalg.norm(v, axis=-1): sqrt(0 + x^2 + y^2 + z^2); norm_dot: np.linalg.norm(v) of a 1-D array (ddot).
double norm_axis(V v);
double norm_dot(V v);
// right_jacobian / right_jacobian_inv (k @ k through cblas_dgemm as np.matmul).
M right_jacobian(V phi);
M right_jacobian_inv(V phi);
// robust_mean_pose: rotation_median (Weiszfeld on SO(3) from the chordal mean) and geometric_median.
Q rotation_median(std::span<const Q> q, int iters = 200, double tol = 1e-13);
V geometric_median(std::span<const V> x, int iters = 200, double tol = 1e-12);
std::pair<Q, V> robust_mean_pose(std::span<const Q> q, std::span<const V> p);
// fit._align_hemisphere: flip signs so consecutive dot products are non-negative.
void align_hemisphere(std::vector<Q>& q);
// fit._slerp_init: control rotations at times at_ns by SciPy's Slerp of the samples (clamped), hemisphere-aligned.
std::vector<Q> slerp_init(std::span<const int64_t> t_ns, std::span<const Q> q, std::span<const int64_t> at_ns);

}  // namespace balla::scene::rot
