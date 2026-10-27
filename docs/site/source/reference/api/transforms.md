# Rigid-transform methods

Use these helpers when assembling [scene records](scene.md) or measured
[trajectories](../../guide/trajectories.md).

## Rigid transforms

Functions in `ballasim.scene.transforms` use `float64`, scalar-first
quaternions, and radians. Quaternion and rotation functions preserve leading
array dimensions.

(scene-normalize-quat)=
### Unit normalization — `normalize_quat`

`normalize_quat(q)` returns `float64 (..., 4)` values divided by their Euclidean
norm. Use it for non-zero scalar-first Hamilton quaternions. A zero quaternion
does not define a rotation; this helper does not provide a replacement pose.

(scene-canonical-quat)=
### Quaternion representative — `canonical_quat`

`canonical_quat(q)` normalizes `(..., 4)` quaternions and flips negative-W
representatives onto the non-negative-W hemisphere. It preserves the represented
rotation. At exactly 180° the sign remains ambiguous because W is zero.

(scene-quat-to-matrix)=
### Rotation matrices — `quat_to_matrix`

`quat_to_matrix(q)` converts `(..., 4)` scalar-first quaternions into
`float64 (..., 3, 3)` active rotation matrices. Use matrix multiplication to
map body vectors into world vectors when `q` denotes body-to-world rotation.

(scene-matrix-to-quat)=
### Quaternion conversion — `matrix_to_quat`

`matrix_to_quat(r)` converts `(..., 3, 3)` rotation matrices into normalized
`float64 (..., 4)` quaternions with non-negative W. Supply rotations rather than
general scaled transforms; use `transform_to_pose()` to extract a rigid pose.

(scene-quat-multiply)=
### Rotation composition — `quat_multiply`

`quat_multiply(a, b)` returns the Hamilton product with NumPy broadcasting over
leading dimensions. Its rotation is `R(a) @ R(b)`, so `b` acts first on a vector.
Input and output trailing dimensions are four; the helper does not normalize
the product.

(scene-quat-conjugate)=
### Quaternion conjugation — `quat_conjugate`

`quat_conjugate(q)` keeps W and negates XYZ in a `float64 (..., 4)` array.
For unit quaternions this is the inverse rotation. It is not the general
algebraic inverse of a non-unit quaternion.

(scene-quat-rotate)=
### Vector rotation — `quat_rotate`

`quat_rotate(q, v)` applies `(..., 4)` quaternion rotations to `(..., 3)` vectors
with broadcasting and returns `float64 (..., 3)`. It rotates directions or
positions about the origin without adding translation.

(scene-so3-exp)=
### Rotation-vector exponential — `so3_exp`

`so3_exp(phi)` converts `float64 (..., 3)` rotation vectors in radians into
unit quaternions `(..., 4)` through the shared core. Vector direction gives the
axis and norm gives the rotation angle. Use it for local rotation increments.

(scene-so3-log)=
### Principal rotation logarithm — `so3_log`

`so3_log(q)` returns the shortest rotation vector `float64 (..., 3)` with norm
at most pi, invariant to quaternion sign. It cannot recover the number of turns
from a rotation alone; retain a sampled trajectory for multi-turn motion.

(scene-relative-rotvec)=
### Relative orientation — `relative_rotvec`

`relative_rotvec(q_a, q_b)` returns `Log(R(a).T @ R(b))` as a broadcast
`(..., 3)` vector in frame A, in radians. It supplies a local residual that is
unchanged when either input quaternion changes sign.

(scene-rotation-angle)=
### Angular separation — `rotation_angle`

`rotation_angle(q_a, q_b)` returns the geodesic separation in `[0, pi]` with
the broadcast leading shape. It is a magnitude, so it discards rotation axis
and direction; use `relative_rotvec()` when those are required.

(scene-hat)=
### Cross-product matrix — `hat`

`hat(v)` maps `(..., 3)` vectors to `float64 (..., 3, 3)` skew matrices satisfying
`hat(v) @ x = cross(v, x)`. It is useful when building pose Jacobians from
analytic angular quantities.

(scene-right-jacobian)=
### Rotation differential — `right_jacobian`

`right_jacobian(phi)` returns `float64 (..., 3, 3)` SO(3) right Jacobians for
rotation vectors in radians. It maps rotation-vector perturbations to local
right perturbations. A small-angle series avoids cancellation near zero.

(scene-right-jacobian-inv)=
### Inverse rotation differential — `right_jacobian_inv`

`right_jacobian_inv(phi)` returns the inverse right Jacobian with the same
shapes and units. Use it for local residual linearization within a consistent
principal rotation chart, rather than across an unresolved branch cut.

(scene-make-transform)=
### Matrix assembly — `make_transform`

`make_transform(r, t)` combines `(..., 3, 3)` rotations and compatible
`(..., 3)` translations into `float64 (..., 4, 4)` homogeneous transforms.
It writes the last row as `[0, 0, 0, 1]` without checking orthonormality.

(scene-pose-to-transform)=
### Pose assembly — `pose_to_transform`

`pose_to_transform(q, p)` converts scalar-first `(..., 4)` quaternions and
`(..., 3)` positions to `float64 (..., 4, 4)` transforms. The resulting mapping
follows the frame meaning of its input pose; it does not add the scene origin.

(scene-transform-to-pose)=
### Pose extraction — `transform_to_pose`

`transform_to_pose(t)` returns `(q, p)` from rigid `(..., 4, 4)` transforms.
Quaternions are normalized and canonical; translation is copied into a
`(..., 3)` array. It does not decompose scale or shear as physical body motion.

(scene-invert-transform)=
### Rigid inversion — `invert_transform`

`invert_transform(t)` returns `float64 (..., 4, 4)` using the transposed rotation
and translated inverse. The input must be rigid: the operation relies on
`R.T` being the rotation inverse and is not a general matrix inverse.

(scene-transform-points)=
### Point transformation — `transform_points`

`transform_points(t, x)` applies one `(4, 4)` transform to point rows ending in
dimension three and returns `float64` points. Unlike direction rotation, it
adds `t[:3, 3]`; use it to place body-frame vertices in a chosen external frame.

(scene-is-rigid)=
### Rigidity predicate — `is_rigid`

`is_rigid(t, tol=1e-6)` returns a boolean for a finite `(4, 4)` transform with
orthonormal positive-determinant rotation and homogeneous last row. It checks
rigidity without modifying the matrix; failure does not project the matrix to
the nearest rotation.

(scene-rotation-about)=
### Axis rotation — `rotation_about`

`rotation_about(axis, angle)` returns a `float64 (3, 3)` rotation using the
SciPy Euler-axis convention. Supply an axis such as `"x"`, `"y"`, or `"z"`
and an angle in radians. Use explicit quaternion or matrix composition when
combining frame changes.

(scene-kabsch)=
### Rigid registration — `kabsch`

`kabsch(a, b)` finds the least-squares proper rotation and translation satisfying
`R @ a[i] + t ≈ b[i]`. Both arrays must have matching `(N, 3)` shape with at
least three corresponding points; invalid shapes raise `ValueError`.
Returns `(R, t, residual_norms)` and fits neither scale nor reflection.

(scene-geometric-median)=
### Robust position — `geometric_median`

`geometric_median(x, *, iters=200, tol=1e-12)` iterates a Weiszfeld geometric
median over point rows. It returns one point minimizing summed distances,
reducing the influence of distant observations compared with a mean. The
iteration limit and relative tolerance control convergence work.

(scene-rotation-median)=
### Robust orientation — `rotation_median`

`rotation_median(q, *, iters=200, tol=1e-13)` iterates a geodesic L1 orientation
estimate from `(N, 4)` quaternions and returns one canonical quaternion. It uses
local SO(3) increments rather than averaging quaternion components, preserving
sign invariance. Widely separated rotation modes should be rejected or separated
before interpreting one central orientation.

(scene-robust-mean-pose)=
### Robust central pose — `robust_mean_pose`

`robust_mean_pose(q, p)` returns `(q_median, p_median)` using the rotation and
geometric medians independently. Inputs describe corresponding pose rows;
the returned `(4,)` and `(3,)` arrays define a robust central pose used by
stationary tests. This is a point estimate rather than a pose covariance.

| Functions | Operation |
| --- | --- |
| `normalize_quat(q)`, `canonical_quat(q)` | Normalize quaternions; canonical form has non-negative W. |
| `quat_to_matrix(q)`, `matrix_to_quat(r)` | Convert `(..., 4)` quaternions and `(..., 3, 3)` matrices. |
| `quat_multiply(a, b)`, `quat_conjugate(q)`, `quat_rotate(q, v)` | Quaternion composition, conjugation, and vector rotation. |
| `so3_exp(phi)`, `so3_log(q)` | Convert rotation vectors `(..., 3)` and quaternions. |
| `relative_rotvec(q_a, q_b)`, `rotation_angle(q_a, q_b)` | Relative rotation vector and geodesic angle. |
| `hat(v)`, `right_jacobian(phi)`, `right_jacobian_inv(phi)` | Skew matrix and SO(3) right Jacobian or inverse. |
| `make_transform(r, t)`, `pose_to_transform(q, p)`, `transform_to_pose(t)` | Build or decompose rigid transforms. |
| `invert_transform(t)`, `transform_points(t, x)` | Invert transforms; apply one transform to point rows. |
| `is_rigid(t, tol=1e-6)`, `rotation_about(axis, angle)` | Check a `(4, 4)` transform; construct an axis rotation. |
| `kabsch(a, b)` | Fit `R a + t ≈ b` for corresponding `(N >= 3, 3)` points; return `(R, t, residual_norms)`. |
| `geometric_median(x, *, iters=200, tol=1e-12)` | Geometric median of point rows. |
| `rotation_median(q, *, iters=200, tol=1e-13)` | Geodesic L1 mean quaternion. |
| `robust_mean_pose(q, p)` | Return median rotation and position as `(q, p)`. |
