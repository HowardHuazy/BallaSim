# Trajectory fitting

Scene scope: `scene.trajectory.*`. These controls apply to measured-pose
fitting in the EVIMO2 adapter. Authored YAML keyframes use interpolation;
see [Trajectories](../../guide/trajectories.md).


## Parameter details

(scene-trajectory-order)=
### Spline order — `scene.trajectory.order`

Selects order 4 (cubic) or 6 (quintic) for the position and cumulative-rotation
splines. Higher order supplies smoother derivatives and a wider local support;
order 6 is the default for motion and acceleration evaluation. The value is an
order, so `5` is rejected despite lying inside the numeric bounds.

(scene-trajectory-loss-c2)=
### Robust loss — `scene.trajectory.loss_c2`

Sets the squared Cauchy scale for joint normalized pose residuals. Samples with
large residuals receive smaller iteratively reweighted least-squares weights,
limiting the influence of motion-capture errors that survive gating. Larger
values approach ordinary least squares; smaller values downweight deviations
more strongly. This changes fitting weights, not the geometric outlier floors.

(scene-trajectory-gate)=
### Initial rejection — `scene.trajectory.gate`

Enables model-free pose rejection before the smoothing fit. Local robust
predictions identify isolated errors, and orientation-branch checks reject
persistent flips of ambiguous marker configurations. Rejected samples receive
zero fit weight; the spline bridges their times. Disable only when the supplied
measurements should bypass this preprocessing; Cauchy weighting and post-fit
residual checks still belong to the smoother.

(scene-trajectory-gate-sigma-k)=
### Outlier multiplier — `scene.trajectory.gate_sigma_k`

Multiplies the robust scale estimated from local prediction residuals. A sample
is rejected when its rotation or position residual exceeds the corresponding
`max(multiplier * scale, floor)`. Raising the multiplier retains more samples;
lowering it increases sensitivity to local tracking errors. It applies with
`gate: true`.

(scene-trajectory-gate-rot-floor-rad)=
### Rotation rejection floor — `scene.trajectory.gate_rot_floor_rad`

Sets the minimum angular discrepancy eligible for initial rejection, measured
by the shortest relative rotation. The default is 3°. This floor prevents
nearly noise-free tracks from producing an arbitrarily tight gate. Increasing
it tolerates larger orientation errors without altering the scale multiplier.

(scene-trajectory-gate-pos-floor-m)=
### Position rejection floor — `scene.trajectory.gate_pos_floor_m`

Sets the minimum translation residual eligible for initial rejection. The
default is 1 cm. The actual limit is the larger of this value and the robust
scale limit; changing it is useful when a track's absolute spatial precision
differs from the default motion-capture assumption.

(scene-trajectory-mahalanobis-max)=
### Post-fit rejection — `scene.trajectory.mahalanobis_max`

Limits the normalized six-dimensional pose residual after the unconstrained
robust fit. Samples beyond the limit are removed and the track is fitted again
before holds are imposed. Smaller limits reject more residual outliers; larger
limits leave their treatment to robust weighting. Applying this check before
holds avoids treating the wander suppressed by a hold as a tracking outlier.

(scene-trajectory-lambda-pos)=
### Position regularization — `scene.trajectory.lambda_pos`

Balances position residuals against the spline roughness penalty. Larger
weights suppress rapid variation in the fitted path and its derivatives;
smaller weights follow measurements more closely. `null` chooses the weight
per track using generalized cross-validation. Values use normalized segment
coordinates, so they are not an SI jerk-noise density.

(scene-trajectory-lambda-rot)=
### Rotation regularization — `scene.trajectory.lambda_rot`

Controls smoothing of the incremental rotation sequence independently of
position smoothing. The order-6 penalty acts on second differences of control
rotation increments, a jerk-like regularizer that is exact for fixed-axis
rotation. Increasing the weight reduces angular variation; `null` selects it
with generalized cross-validation. It is not a direct angular-acceleration cap.

(scene-trajectory-gcv-log10-range)=
### Regularization search — `scene.trajectory.gcv_log10_range`

Sets the base-10 exponent interval searched when either smoothing weight is
`null`. The search evaluates a decade grid, refines around its best candidate,
and interpolates the local minimum. Widen the interval if a chosen weight lies
at a boundary; a wider interval costs additional fits. It must increase from
lower to upper bound and contain no more than 61 whole-decade grid points.

(scene-trajectory-detect-holds)=
### Exact stationary intervals — `scene.trajectory.detect_holds`

Identifies sufficiently quiet intervals in the preliminary fit and retained
measurements, then ties their active controls to one pose. The resulting hold
has exactly zero translational and angular derivatives. This removes stationary
tracking wander that would otherwise create image motion. Disabling detection
preserves the smoothed residual motion instead of imposing interior holds.

(scene-trajectory-hold-rule)=
### Hold criterion — `scene.trajectory.hold_rule`

`sphere` bounds motion of the body's bounding sphere in meters. `image`
projects the pose change through camera geometry and uses pixel displacement,
allowing the tolerance to follow distance and focal length. Image-space fitting
requires a camera/depth view; the EVIMO2 adapter constructs it. The choice
changes hold detection, not the spline representation or event thresholds.

(scene-trajectory-hold-max-disp-m)=
### Spatial hold tolerance — `scene.trajectory.hold_max_disp_m`

Limits every retained pose's bounding-sphere displacement from a candidate
hold's robust central pose. Under the sphere rule, exceeding the limit rejects
the hold rather than converting a slow drift into a staircase of constant
poses. The EVIMO2 static-tabletop test also uses this value under either hold
rule. Increasing it accepts more residual object motion as stationary.

(scene-trajectory-hold-min-ns)=
### Minimum stationary duration — `scene.trajectory.hold_min_ns`

Sets how long a quiet interval must persist before becoming a hold. Shorter
values admit briefer rests but make transient slowdowns easier to classify as
stationary. Image-space detection also divides `hold_max_disp_px` by this
duration in seconds to obtain its active-motion speed threshold; its duration
must therefore be positive.

(scene-trajectory-hold-moving-mps)=
### Active-motion speed — `scene.trajectory.hold_moving_mps`

Under the sphere rule, bounds the speed of the body's support using translation
and angular motion. Runs above this threshold identify active motion; quieter
runs remain candidates until their duration and displacement checks pass.
Raising the threshold broadens the candidate intervals, but does not override
`hold_max_disp_m`. The image rule derives its own pixel-speed threshold.

(scene-trajectory-hold-max-disp-px)=
### Projected hold tolerance — `scene.trajectory.hold_max_disp_px`

Bounds a hold's largest projected excursion under the image rule. Camera
projection and distance convert translation and rotation into image travel.
Increasing the value admits more image motion as stationary; it also raises
the speed threshold derived from `hold_min_ns`. Uniform drift that exceeds the
excursion bound remains moving.

(scene-trajectory-branch-jump-rad)=
### Orientation-branch separation — `scene.trajectory.branch_jump_rad`

Sets the minimum turn between retained samples considered a branch jump, and
the orientation distance used to join subsequent runs into branches. A jump
normally must also exceed the relative rate threshold below. The branch with
the most retained samples is kept. Steps above the fixed 162° aliasing limit
are treated as ambiguous regardless of this setting.

(scene-trajectory-branch-rate-factor)=
### Orientation-branch rate — `scene.trajectory.branch_rate_factor`

Multiplies the track's 95th-percentile inter-sample turn rate. A candidate above
`branch_jump_rad` is split when it also exceeds this relative rate, separating
abrupt tracking flips from sustained fast rotation. Lower values make branch
rejection more sensitive; higher values require more abrupt jumps. It applies
with initial gating enabled.

## Fit settings lookup

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `scene.trajectory.order` | `int` | `6` | Spline order; accepts `4` or `6`. Range: [4, 6]. |
| `scene.trajectory.loss_c2` | `float` | `16.81` | Squared Cauchy scale for robust fitting. Range: >0. |
| `scene.trajectory.gate` | `bool` | `true` | Reject pose outliers before fitting. |
| `scene.trajectory.gate_sigma_k` | `float` | `8.0` | Robust standard-deviation multiplier for the initial outlier gate. Range: >0. |
| `scene.trajectory.gate_rot_floor_rad` | `float · rad` | `0.05235987755982989` | Minimum rotation residual rejected by the initial gate. Range: ≥0. |
| `scene.trajectory.gate_pos_floor_m` | `float · m` | `0.01` | Minimum position residual rejected by the initial gate. Range: ≥0. |
| `scene.trajectory.mahalanobis_max` | `float` | `8.0` | Mahalanobis residual limit for the post-fit rejection/refit pass. Range: >0. |
| `scene.trajectory.lambda_pos` | `float?` | `null` | Position regularization in normalized segment units; `null` selects GCV per track. Range: >0. |
| `scene.trajectory.lambda_rot` | `float?` | `null` | Rotation regularization in normalized segment units; `null` selects GCV per track. Range: >0. |
| `scene.trajectory.gcv_log10_range` | `float[2]` | `[-4.0, 12.0]` | Base-10 logarithmic interval for GCV regularization search. Range: [-30, 30]. |

## Hold settings lookup

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `scene.trajectory.detect_holds` | `bool` | `true` | Detect stationary intervals and constrain them to exact holds. |
| `scene.trajectory.hold_max_disp_m` | `float · m` | `0.001` | Maximum object-space hold displacement; also used by the static-tabletop test. Range: ≥0. |
| `scene.trajectory.hold_min_ns` | `int · ns` | `30000000` | Minimum hold duration. Range: ≥0. |
| `scene.trajectory.hold_moving_mps` | `float · m/s` | `0.02` | Object-space speed above which motion is active. Sphere hold rule only. Range: ≥0. |
| `scene.trajectory.hold_rule` | `enum` | `"sphere"` | Use bounding-sphere displacement or projected image displacement for hold detection. Choices: `sphere`, `image`. |
| `scene.trajectory.hold_max_disp_px` | `float · px` | `0.5` | Maximum projected excursion in an image-space hold. Range: >0. |
| `scene.trajectory.branch_jump_rad` | `float · rad` | `1.5707963267948966` | Minimum inter-sample rotation considered an orientation-branch jump. Range: >0. |
| `scene.trajectory.branch_rate_factor` | `float` | `10.0` | Branch-jump rate threshold relative to the track's 95th-percentile turn rate. Range: >0. |

## Constraints

Spline order accepts only `4` or `6`. `gcv_log10_range` must be an increasing
finite interval with at most 61 whole-decade grid points. Image-space holds
require positive `hold_min_ns` and `hold_max_disp_px`; their motion threshold
is the allowed image displacement divided by the hold duration in seconds.

Inputs require increasing timestamps and unit scalar-first quaternions. Fits
reject rotation steps above 162°; add intermediate measurements for rapid turns.

## References

- Sommer et al., [Efficient Derivative Computation for Cumulative B-Splines on Lie Groups](https://arxiv.org/abs/1911.08860),
  CVPR 2020. Cumulative rotation splines and analytic derivatives.
- Eilers and Marx, [Flexible Smoothing with B-splines and Penalties](https://projecteuclid.org/journals/statistical-science/volume-11/issue-2/Flexible-smoothing-with-B-splines-and-penalties/10.1214/ss/1038425655.full),
  1996. Penalized spline fitting.
- Wahba, [A Comparison of GCV and GML for Choosing the Smoothing Parameter](https://projecteuclid.org/journals/annals-of-statistics/volume-13/issue-4/A-Comparison-of-GCV-and-GML-for-Choosing-the-Smoothing/10.1214/aos/1176349743.full),
  1985. Data-driven regularization selection.
