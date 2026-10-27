# Fitting and evaluation

`ballasim.traj` converts timed poses into the `Trajectory` records consumed by
scene bundles. Interpolation treats keys as constraints; smoothing treats
measurements as noisy observations. Both use the same cumulative-rotation
representation and return analytic pose derivatives through the shared core.
See [Trajectories](../../guide/trajectories.md) for a complete example.

(pose-samples)=
## Pose samples — `PoseSamples`

```python
PoseSamples(t_ns, q, p)
```

Collects poses of one body in a common world frame. Input timestamps must
increase strictly, and the position refers to the body origin rather than the
mesh centroid unless those origins coincide. Construct one instance per track;
use the same clock and world frame for tracks that enter one scene.

| Argument / attribute | Shape and dtype | Meaning |
| --- | --- | --- |
| `t_ns` | `int64 (N,)` | Measurement times in ns. |
| `q` | `float64 (N, 4)` | Active body-to-world Hamilton quaternions, `(w, x, y, z)`. |
| `p` | `float64 (N, 3)` | World-frame body positions in m. |

Arrays are converted to these dtypes. Quaternions within `1e-3` of unit norm
are normalized. Invalid shapes, non-finite poses, non-increasing times, or
larger norm errors raise `ValueError`. `len(samples)` returns `N`; fitting
requires at least two usable samples.

(fit-interpolating)=
## Keyframe interpolation — `fit_interpolating`

```python
fit_interpolating(keys, *, order=6, h_ns=None, span_ns=None,
                  max_h_ns=20_000_000, tol=1e-9, max_iterations=60)
```

Returns a `FitResult` whose spline passes through `keys: PoseSamples` within
`tol` in meters and radians. It minimizes roughness subject to those pose
constraints. Use it for authored keyframes whose locations and orientations
must be preserved; it does not estimate a measurement-noise model.

- `order` selects cubic (`4`) or quintic (`6`) position and rotation splines.
- `h_ns` sets the initial knot spacing. With `None`, the fitter selects a spacing
  from the key-time lattice, initially no larger than `max_h_ns`.
- `span_ns=(begin, end)` must contain all keys. `None` uses the first and last
  key times; an extension holds the corresponding endpoint pose.
- `tol` limits the largest translation and rotation residual at the keys.
  `max_iterations` bounds each constrained rotation solve.

Consecutive identical keys create exact holds, including quaternions that differ
only in sign. The fitter halves knot spacing if rotations or closely spaced
holds cannot be represented on the current grid. Controls are spline
coefficients; they are not the interpolated keyframes.

Raises `TrajectoryError` for fewer than two keys, ambiguous inter-key turns,
or failure to represent the constraints. A span excluding a key raises
`ValueError`.

(fit-smoothing)=
## Measured-pose smoothing — `fit_smoothing`

```python
fit_smoothing(samples, options=None)
```

Returns a robust `FitResult` from `samples: PoseSamples`; `options=None` uses
`SmoothingOptions()`. The fit combines normalized pose residuals with a
roughness penalty, iterates Cauchy weights, and optionally rejects outliers and
imposes exact holds. Position is solved with banded linear algebra; rotation
is optimized on SO(3), preserving unit rotations.

Use `fit.keep`, `fit.weights`, and `fit.diagnostics` to inspect data rejection
and fit quality. A smooth curve alone does not establish measurement accuracy:
regularization and holds intentionally remove some observed variation.

Raises `TrajectoryError` for insufficient retained data or ambiguous turns,
`ValueError` for inconsistent options, and may propagate numerical-solver
errors when a fit cannot be solved.

(smoothing-options)=
## Smoothing controls — `SmoothingOptions`

A dataclass accepting keyword arguments. Schema-backed defaults and their
individual effects are documented in [Trajectory settings](../parameters/trajectories.md).
The remaining arguments describe the sampling grid, object extent, and solver.

### Grid and interval — `order`, `h_ns`, `span_ns`

`order` is `4` or `6`. `h_ns=None` uses the median input sample period;
smaller spacing permits more local variation but increases control count.
`span_ns=None` uses the first and last retained samples. An explicit span must
contain those samples; outside their range the endpoint poses are held.

### Robust fitting and regularization

`loss_c2` controls Cauchy downweighting; `gate`, `gate_sigma_k`,
`gate_rot_floor_rad`, `gate_pos_floor_m`, and `mahalanobis_max` control rejection.
`branch_jump_rad` and `branch_rate_factor` identify persistent orientation flips.
See the {ref}`initial rejection <scene-trajectory-gate>`
and {ref}`regularization <scene-trajectory-lambda-pos>`
entries before changing them.

`lambda_pos` and `lambda_rot` are independent roughness weights; `None` selects
each with generalized cross-validation over `gcv_log10_range`. Larger weights
smooth more strongly; they are not bounds on speed or acceleration.

### Stationary support — `center_body_m`, `radius_m`

Specify a body-frame sphere for the spatial hold test: `center_body_m` is a
`float64 (3,)` vector, default `[0, 0, 0]`, and `radius_m` defaults to `0.1` m.
The test includes both translation and rotation of this support. A larger
sphere makes a given rotation produce a larger displacement bound; use an
extent representative of the object's geometry.

### Projected holds — `hold_view`

With `hold_rule="image"`, supply a `ballasim.traj.holds.HoldView` describing
camera projection and per-sample distances. The EVIMO2 adapter supplies this
information automatically. `hold_max_disp_px` and positive `hold_min_ns`
control the projected excursion and its associated motion-speed threshold.
The sphere rule instead uses `hold_max_disp_m` and `hold_moving_mps`.
`detect_holds=False` omits interior stationary detection.

### Noise floors — `sigma_floor_pos_m`, `sigma_floor_rot_rad`

Bound estimated residual scales below by `1e-6` m and `1e-6` rad by default.
They prevent nearly exact tracks from creating a singular normalized weighting
problem. Raising them reduces the influence assigned to a track's apparent
precision; these floors do not add noise to the output poses.

### Solver iterations — `max_irls`

Limits robust fitting to 60 IRLS iterations by default. Additional iterations
may help a slowly stabilizing weight/regularization solution at additional
build cost. Review the reported diagnostics when the fit remains unsettled;
a larger limit does not repair ambiguous or invalid measurements.

(hold-view)=
## Projected-motion support — `HoldView`

```python
from ballasim.traj.holds import HoldView

HoldView(focal_px, corner_r2, depth_m, cameras_body_m=None)
```

Supplies conservative camera geometry for `SmoothingOptions(hold_rule="image")`.
`focal_px: (C,)` contains camera angular gains in px/rad, including projection
stretch; `corner_r2: (C,)` contains maximum squared normalized image radius.
`depth_m: (N,)` gives a positive lower distance bound for each original pose
sample. These are bounds over the relevant surfaces, not an arbitrary center
depth. For a camera rig, `cameras_body_m: (C, 3)` additionally supplies camera
centers in rig coordinates; use `None` for an observed object's view.

(hold-view-speed)=
### Projected speed — `HoldView.speed`

`view.speed(ev, center, radius, idx)` returns a per-sample image-speed bound
in px/s. `ev` contains evaluated pose derivatives, `center: (3,)` and `radius`
describe body support, and `idx` maps those samples to the original distance
array. Rig evaluation includes the camera lever arms; object evaluation bounds
the projected bounding-sphere motion.

(hold-view-excursion)=
### Projected displacement — `HoldView.excursion`

`view.excursion(q, p, center, radius, idx)` returns a scalar maximum image
displacement bound in px relative to the robust central pose of the supplied
samples. Quaternion/position rows use `(N, 4)`/`(N, 3)`; `idx` selects matching
distance bounds. The hold detector compares this bound with its pixel tolerance.

(fit-result)=
## Fit output — `FitResult`

Stores `trajectory`, JSON-serializable `diagnostics`, and arrays `keep: bool (N,)`
and `weights: float64 (N,)` corresponding to the original input rows. Rejected
samples have zero weight; interpolation returns all rows retained with unit
weights. Preserve this row correspondence when reviewing outliers.

`trajectory` contains the actual controls, support, and exact holds used by
rendering. `diagnostics` records the mode, grid, fit/rejection information,
and hold intervals; its detailed fields depend on fit mode. The result does
not contain an independent pose-uncertainty covariance estimate.

(evaluate-trajectory)=
## Pose and derivatives — `evaluate`

```python
evaluate(traj, t_ns)
```

Evaluates a `ballasim.scene.Trajectory` at timestamps converted to a flat
contiguous `int64` array. Returns a dictionary of `float64` arrays in query
order. The implementation is shared with the renderer; it does not estimate
derivatives from neighboring requested times.

| Key | Shape | Frame and units |
| --- | --- | --- |
| `q` | `(N, 4)` | Body-to-world quaternion, scalar first. |
| `p` | `(N, 3)` | Position relative to the bundle origin, m. |
| `v`, `a` | `(N, 3)` | World-frame velocity (m/s) and acceleration (m/s²). |
| `omega`, `omega_dot` | `(N, 3)` | Body-frame angular velocity (rad/s) and angular acceleration (rad/s²). |

Times outside the serving spline's support raise `ValueError`. Piecewise
trajectories select the right-hand piece at a breakpoint, so a derivative at
that time is the right-hand derivative. Exact holds return constant poses and
zero derivatives.

(pose-matrices)=
## Rigid pose matrices — `pose_matrices`

```python
pose_matrices(traj, t_ns)
```

Returns `float64 (N, 4, 4)` body-to-world transforms using `evaluate()` and the
same timestamp conversion and support checks. The translation remains relative
to `origin_world`; add that origin only when converting to an absolute external
world convention. Use this interface to pose geometry when derivatives are
not required.

(retime-stamps)=
## Uniform-clock fit — `retime_stamps`

```python
retime_stamps(t_raw_ns, nominal_period_ns=5_000_000, *,
              huber_k=1.345, iterations=20)
```

Fits a uniform capture clock to a one-dimensional recorded `int64` timestamp
array and returns `Retiming`. Frame increments are obtained by rounding each
recorded difference divided by `nominal_period_ns`; a robust line then fits
`time = a_ns + b_ns * frame_index` with Huber residual weights. Gaps retain
integer jumps in frame index.

Use this only when capture was physically uniform and timestamp variation is
recording jitter. It cannot infer arbitrary capture-rate changes.
`huber_k` controls robust regression weighting; `iterations` bounds its update
count. Fewer than two stamps or non-positive rounded frame increments raise
`ValueError`.

(retiming)=
## Clock-fit output — `Retiming`

An immutable dataclass holding the recorded and corrected `int64 (N,)` arrays
`t_raw_ns` and `t_ns`, acquisition `frame_index`, fitted `a_ns`/`b_ns`,
`residual_std_ns`, and `num_gaps`. The first frame index is zero; indices jump
across missing captures. Corrected times are rounded to integer ns.

(retiming-to-retimed)=
### Associated timestamps — `Retiming.to_retimed`

```python
retiming.to_retimed(t_raw_ns)
```

Returns corrected `int64` timestamps with the input shape. It linearly
interpolates the measured `raw - corrected` stamp error and subtracts that
error from associated times, such as ground-truth render timestamps. Outside
the fit's sample range the endpoint correction is retained; this is not a
new clock-drift extrapolation.

(retiming-summary)=
### Clock diagnostics — `Retiming.summary`

```python
retiming.summary()
```

Returns a JSON-serializable dictionary with fitted intercept, period, residual
standard deviation, gap count, and sample count. Use it to record the correction
alongside exported data; it contains summary values rather than the full
per-stamp correction arrays.

(trajectory-error)=
## Fitting error — `TrajectoryError`

A `ValueError` subclass for pose sequences that cannot be fitted faithfully,
including ambiguous rotations or insufficient retained data. Turns above 162°
between usable input poses are rejected because the shortest-rotation direction
is ambiguous. Add intermediate measurements rather than relaxing a numerical
fit tolerance.

## References

- Sommer et al., [Efficient Derivative Computation for Cumulative B-Splines on Lie Groups](https://arxiv.org/abs/1911.08860),
  CVPR 2020. Rotation evaluation and derivatives.
- Ovrén and Forssén, [Trajectory Representation and Landmark Projection for Continuous-Time Structure from Motion](https://arxiv.org/abs/1805.02543),
  IJRR 2019. Split position/rotation representations.
- {download}`Fitting implementation <../../../../../python/ballasim/traj/fit.py>`.
