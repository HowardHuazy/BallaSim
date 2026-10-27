# Rotation and spline primitives

These `ballasim._core` batch routines expose the rotation and cumulative-spline
implementation used by trajectory evaluation. Most callers should fit and query
a [Trajectory](trajectories.md) through `ballasim.traj`.

## Rotation maps

### `_core.so3_exp(phi)`

Converts axis-angle rotation vectors `float64 (N,3)` into scalar-first unit
quaternions `float64 (N,4)`. A vector's direction is its rotation axis and its
length is the angle in radians; zero gives the identity. This is the SO(3)
exponential, not componentwise exponentiation or a sequence of Euler rotations.
The batch wrapper rejects nonfinite vectors and magnitudes above `2e6` radians.

### `_core.so3_log(q)`

Converts scalar-first quaternions `float64 (N,4)` to principal rotation vectors
`float64 (N,3)` with magnitude at most π. Supply valid unit quaternions. The
implementation selects the nonnegative-scalar hemisphere and uses an `atan2`
form that remains accurate near zero and π. Rotations beyond half a turn do not
retain their winding through this principal logarithm.

## Uniform cumulative splines

### `_core.spline_eval_f64`

```python
spline_eval_f64(order, t0_ns, h_ns, q, p, t_ns)
```

Evaluates a split rotation/translation cumulative B-spline in `float64`.
`order=4` is cubic and C²; `order=6` is quintic and C⁴ within a uniform piece.
The input rows are control poses, not interpolation samples: the curve does
not generally pass through each control. Use the fitting API when measurements
or authored keyframes must be interpolated.

On segment `s`, the rotation is a product of weighted relative rotations and
translation is a weighted sum of position increments:

```{math}
R(t)=R_s\prod_{j=1}^{K-1}\operatorname{Exp}
\bigl(\lambda_j(u)\operatorname{Log}(R_{s+j-1}^{T}R_{s+j})\bigr).
```

Here `u=(t-t0_ns-s*h_ns)/h_ns` and `lambda_j` are cumulative basis functions.
Quaternion controls are normalized during node construction; their relative
increments use the principal logarithm. Provide intermediate controls when an
intended turn cannot be represented by consecutive shortest rotation increments.
Equal controls over a segment's support give an exactly held pose.

### Controls and support

Controls are `float64 q (M,4)` and `p (M,3)`, with scalar-first quaternions and
positions in meters. At least `order` rows are required; both arrays need the
same row count. There are `M-order+1` segments. Knot spacing satisfies
`1 <= h_ns < 2**31` nanoseconds.

Queries are exact `int64 (N,)`; other dtypes raise `TypeError`. Segment `s`
uses controls `s:s+order` over `[t0_ns+s*h_ns, t0_ns+(s+1)*h_ns)`.
The final endpoint is accepted at `u=1`; times outside the spline's full support
raise `ValueError`. This low-level routine does not apply a trajectory record's
separate hold or valid-span metadata.

### Returned kinematics

All evaluators return a dictionary containing analytic pose and derivatives:

| Field | Shape | Meaning and frame |
| --- | --- | --- |
| `q` | `(N,4)` | Body-to-parent unit quaternion `(w,x,y,z)` |
| `p` | `(N,3)` | Position in the parent frame, meters |
| `omega` | `(N,3)` | Body-frame angular velocity, rad/s |
| `omega_dot` | `(N,3)` | Body-frame angular acceleration, rad/s² |
| `v` | `(N,3)` | Parent-frame linear velocity, m/s |
| `a` | `(N,3)` | Parent-frame linear acceleration, m/s² |

The angular convention is `[omega]×=R.T*dR/dt`. Derivatives include the knot
spacing conversion from nanoseconds to seconds. They are derivatives of the
represented curve, not finite differences of returned poses or independently
fitted velocity measurements.

### `_core.spline_eval_f32`

```python
spline_eval_f32(order, t0_ns, h_ns, q, p, origin, t_ns)
```

Evaluates the fp32 execution path while accepting the same fp64 controls.
Nodes and control differences are first constructed in fp64, then cast.
`origin` is `float64 (3,)`; returned positions are relative to it. Subtracting
the origin before casting preserves local position precision. Rotations and
time derivatives retain their frames and units.

Times are converted to signed 32-bit offsets from a knot-aligned base chosen
for the batch. Split a query batch when its offsets cannot fit that range.
The fp64 CPU evaluator uses absolute int64 times and has no such per-batch
offset restriction.

## Piecewise trajectories

### `_core.spline_eval_pieces_f64`

```python
spline_eval_pieces_f64(order, pieces, q, p, t_ns)
```

Evaluates independent uniform spline pieces with shared control arrays.
`pieces` is exact `int64 (P,6)`, with rows
`(begin_ns, t0_ns, h_ns, ctrl_begin, ctrl_end, join)`. Begins must increase
strictly; each knot origin must be at or before its begin. Consecutive control
ranges must cover all rows of `q` and `p`, with at least `order` controls per
piece.

A query selects the last piece whose `begin_ns` is no later than the query.
Before the first begin it selects piece zero, provided the time lies in that
piece's support. A query exactly at a breakpoint selects the right piece.
The result has the same `float64` fields as `spline_eval_f64`.

`join` carries trajectory continuity metadata but is not read or enforced by
this evaluator. Different pieces can therefore encode a step or a derivative
discontinuity. A smooth spline within each piece does not guarantee continuity
across their boundary; fitting and scene validation must enforce the intended
join conditions.

## Motion bounds

### `_core.spline_segment_bounds(order, h_ns, q, p)`

Returns `float64` arrays `omega`, `omega_dot`, `v` and `a`, each with one scalar
upper bound per uniform segment. The values bound derivative magnitudes over
the whole segment, rather than measuring them at a particular timestamp.
The scheduler uses such bounds to limit image motion between render samples.

Velocity bounds use maximum control increments divided by knot spacing;
acceleration bounds include increment differences and rotation-composition
terms. They include floating-point allowances and are generally conservative.
Finite positions and finite nonzero quaternions are required. Bounds of an
exact hold remain exactly zero.

## CUDA evaluation

### `_cuda.spline_eval_f32` and `_cuda.spline_eval_f64`

The optional CUDA module evaluates uniform splines with the corresponding CPU
function's signature and returns CPU NumPy arrays. Its fp32 variant also takes
`origin`. Both device variants use signed 32-bit query offsets from one
knot-aligned base; batch the query times accordingly. No piecewise evaluator
is exposed by this module.

The shared math targets matching host/device results at the same precision;
fp32 is not interchangeable with fp64. `_cuda.device_count()` reports visible
CUDA devices. Device calls require the CUDA build, a usable driver and a GPU.

## Callable reference

| Callable | Output dtype or result |
| --- | --- |
| `so3_exp(phi)` | `float64 (N,4)` unit quaternions |
| `so3_log(q)` | `float64 (N,3)` rotation vectors |
| `spline_eval_f64(order, t0_ns, h_ns, q, p, t_ns)` | `float64` kinematics dictionary |
| `spline_eval_f32(order, t0_ns, h_ns, q, p, origin, t_ns)` | `float32` kinematics dictionary |
| `spline_eval_pieces_f64(order, pieces, q, p, t_ns)` | `float64` kinematics dictionary |
| `spline_segment_bounds(order, h_ns, q, p)` | Four `float64` per-segment bound arrays |

## References and implementation

Cumulative-spline derivatives follow Sommer et al.,
[Efficient Derivative Computation for Cumulative B-Splines on Lie Groups](https://arxiv.org/abs/1911.08860).
Quaternion conventions follow the Hamilton product discussed in
[Why and How to Avoid the Flipped Quaternion Multiplication](https://arxiv.org/abs/1801.07478).
The split representation, floating-point motion bounds and piece-selection
contracts are BallaSim implementation choices.

Bindings are in `src/bindings/core_traj.cpp`; node construction, evaluation,
bounds and piece selection are in `src/core/traj/`. The higher-level
[trajectory API](trajectories.md) supplies fitting and trajectory metadata.
