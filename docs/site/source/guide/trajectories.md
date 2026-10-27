# Trajectories

BallaSim represents body and camera-rig poses as continuous-time splines. Fit
authored keyframes with interpolation; fit measured poses with robust smoothing.
Both paths return a `FitResult` containing the trajectory and fit diagnostics.

Position and rotation use separate splines on R³ and SO(3). Rotation is composed
from local relative rotations, preserving a rigid pose throughout evaluation.
Analytic derivatives provide velocity and acceleration for sampling bounds,
optical flow, and inertial output without finite differencing queried poses.

## Input conventions

`PoseSamples(t_ns, q, p)` accepts strictly increasing `int64` timestamps `(N,)`,
unit quaternions `(N, 4)` in `(w, x, y, z)` order, and positions `(N, 3)` in
meters. Pose arrays use `float64`. Each fit requires at least two samples.

Positions describe the body origin in world coordinates. Once attached to a
scene bundle, they are relative to `origin_world`. Camera-rig trajectories follow
the same convention; `T_rig_cam` supplies each camera's fixed extrinsic transform.

## Interpolate keyframes

```python
import numpy as np
from ballasim.traj import PoseSamples, fit_interpolating

keys = PoseSamples(
    t_ns=np.array([0, 500_000_000, 1_000_000_000], dtype=np.int64),
    q=np.array([[1, 0, 0, 0]] * 3, dtype=np.float64),
    p=np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0, 0]], dtype=np.float64),
)
fit = fit_interpolating(keys)
trajectory = fit.trajectory
```

The fit passes through every key within `tol` (default `1e-9` m and rad).
Consecutive identical poses define exact holds. In this example, the object
remains stationary from `0.5` to `1.0` s. Extending `span_ns` beyond the key range
holds the first and last poses outside that range.

The default spline order is `6`; order `4` is also supported. `max_h_ns` limits
the initial knot spacing, which is refined when required by the motion or holds.

See {ref}`Fitting and evaluation <fit-interpolating>`
for interpolation arguments and failure conditions.

### Scene YAML

Bodies and rigs accept the same keyframe syntax:

```yaml
keyframes:
  - {t_s: 0, t_m: [0, 0, 0], rpy_deg: [0, 0, 0], hold_s: 0.25}
  - {t_s: 1, t_m: [0.2, 0, 0], q_wxyz: [1, 0, 0, 0]}
knot_spacing_ms: 20
```

Use `t_ns` for exact integer times. `hold_s` inserts an identical key at the end
of the hold; later keys must remain strictly increasing. Rotations may use
`q_wxyz` or extrinsic XYZ `rpy_deg`. Camera poses may use `eye_m`, `target_m`, and
`up` instead.

Optional `spin` and `shake` apply rotations in the body frame over `t0_s` to
`t1_s`. `ramp_s` controls their transition; `perturbation_key_ms` sets the
sampling interval used to refit the perturbed motion. A spin retains its final
angle after its window.

## Smooth measured poses

```python
from ballasim.traj import SmoothingOptions, fit_smoothing

fit = fit_smoothing(samples, SmoothingOptions(detect_holds=True))
```

Here, `samples` is a `PoseSamples` instance containing measured poses. Smoothing
uses robust outlier rejection and regularized fitting. `lambda_pos=None` and
`lambda_rot=None` select regularization by generalized cross-validation.

| Option | Purpose |
| --- | --- |
| `h_ns` | Knot spacing; defaults to the median sample period. |
| `span_ns` | Valid interval; must include all retained samples. |
| `gate` | Reject pose outliers and inconsistent orientation branches. |
| `lambda_pos`, `lambda_rot` | Override position and rotation regularization. |
| `detect_holds` | Detect stationary intervals and constrain them to exact holds. |
| `hold_rule` | `sphere` for object-space displacement; `image` for projected displacement, requiring a `HoldView`. |

Inspect `fit.keep` and `fit.weights` for per-sample rejection and robust weights.
`fit.diagnostics` records fit settings, residuals, and detected holds. EVIMO2 runs
expose the corresponding settings through `scene.trajectory`.

[Trajectory settings](../reference/parameters/trajectories.md) describe how
outlier gates, regularization, and stationary-interval tests affect the fit.

## Evaluate motion

```python
from ballasim.traj import evaluate, pose_matrices

times = np.array([100_000_000, 750_000_000], dtype=np.int64)
state = evaluate(trajectory, times)
transforms = pose_matrices(trajectory, times)
```

| Result | Shape | Frame and units |
| --- | --- | --- |
| `q`, `p` | `(N, 4)`, `(N, 3)` | Body-to-world rotation; position in m |
| `v`, `a` | `(N, 3)` | World-frame velocity in m/s and acceleration in m/s² |
| `omega`, `omega_dot` | `(N, 3)` | Body-frame angular velocity in rad/s and acceleration in rad/s² |
| `pose_matrices()` | `(N, 4, 4)` | Body-to-world rigid transforms |

Evaluation uses the shared C++ implementation and rejects times outside spline
support. Holds give constant poses and zero derivatives. Piecewise trajectories
select the right-hand piece at a breakpoint.

## Time and rotation constraints

`retime_stamps()` fits a uniform capture clock to irregular recorded timestamps.
Its `Retiming` result contains corrected times and can map associated timestamps
with `to_retimed()`. Use it only when the acquisition process is known to be
uniform; EVIMO2 applies this correction to its motion-capture clock.

Fits reject consecutive input rotations above 162° because their direction is
ambiguous. Add intermediate samples for rapid motion. Consecutive spline control
rotations must differ by at most 90°; interpolation refines its knots as needed.

See the [trajectory API](../reference/api/trajectories.md) for signatures and
return types.

## References

- Sommer et al., [Efficient Derivative Computation for Cumulative B-Splines on Lie Groups](https://arxiv.org/abs/1911.08860),
  CVPR 2020. Rotation-spline evaluation and analytic derivatives.
- Ovrén and Forssén, [Trajectory Representation and Landmark Projection for Continuous-Time Structure from Motion](https://arxiv.org/abs/1805.02543),
  IJRR 2019. Split position and rotation trajectory representations.

```{toctree}
:hidden:
:maxdepth: 1

Fitting and evaluation <../reference/api/trajectories>
Trajectory settings <../reference/parameters/trajectories>
Rotation and spline primitives <../reference/api/geometry>
Rigid-transform methods <../reference/api/transforms>
```
