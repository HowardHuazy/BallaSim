# Accuracy

```{toctree}
:hidden:
:maxdepth: 1

profiles
```

Run scope: top-level `accuracy.*`. Tables list **schema base defaults**. The
selected [accuracy profile](profiles.md) replaces applicable defaults before
explicit run/profile overrides are applied; the default profile is `fidelity`.

Adaptive-scheduling settings apply when `run.fixed_rate_hz` is `null`.
Stationary-probe settings apply with `initial_state: stationary`.

Numerical settings determine how closely the simulation follows its selected
image-formation and pixel models. They do not calibrate a physical sensor.
Spatial integration controls the finite-pixel estimate; temporal scheduling
controls when that estimate is evaluated; sensor tolerances control integration
and crossing searches between those samples.

Start with an [accuracy profile](profiles.md). Tightening one tolerance cannot
compensate for error in a different stage: extra spatial samples do not resolve
an unsampled visibility change, and a finer temporal lattice does not restore
missing interreflection.

## Preset and design inputs

(accuracy-profile)=
### `accuracy.profile`

Select a coherent set of numerical defaults before explicit overrides are
applied. `fidelity` derives several values from the configured thresholds, noise,
and readout. `balanced` relaxes the initial motion schedule and bounds refinement;
`reference` uses a denser comparison schedule; `phase1` preserves the frozen
baseline. A reference run remains a numerical approximation to the implemented
model, rather than a claim of physical ground truth.

(accuracy-design-contrast)=
### `accuracy.design_contrast`

Set the natural-log edge contrast used to derive the spatial sample count in
fidelity and balanced. A finite-pixel edge moves across one subsample at a time;
higher contrast makes each unresolved step more consequential relative to the
event threshold. Raising this design value can therefore increase the derived
count. It does not clamp or alter actual scene contrast.

(accuracy-design-threshold)=
### `accuracy.design_threshold`

Represent the threshold that spatial-sampling design should resolve. Normally
this is the minimum of `theta - 3*sigma_theta` across selected cameras and both
polarities, floored at `0.10`. The three-sigma margin accounts for threshold
mismatch without designing around the hot-pixel tail. Frozen EVIMO2 calibrations
use the smallest mean threshold instead. This derived design quantity does not
replace the per-pixel firing thresholds.

## Spatial integration

(accuracy-n-subsamples)=
### `accuracy.n_subsamples`

Set the base number of rays used to integrate a pixel's spatial response at one
sample time. Owen-scrambled Sobol prefixes give nested spatial patterns, so
powers of two retain the intended low-discrepancy structure. Increasing the
count improves the finite-pixel estimate near silhouettes, texture, and shadow
boundaries, at greater tracing cost; it does not change exposure or event rate
directly. The profile's design formula and cap are listed in
[Accuracy profiles](profiles.md#derived-values).

(accuracy-psf-truncation-sigmas)=
### `accuracy.psf_truncation_sigmas`

Truncate the Gaussian pixel-response tail at this many standard deviations,
beyond half the box aperture. The support radius is
`aperture_px/2 + truncation_sigmas*blur_sigma_px`. Larger support retains more of
the Gaussian tail but requires valid rays farther outside edge pixels and can
increase planning reach. This changes the represented spatial response, unlike
merely increasing its integration sample count.

(accuracy-warp-cells)=
### `accuracy.warp_cells`

Set the number of cells in the tabulated inverse CDF used to map uniform samples
to the box-convolved-Gaussian response. More cells reduce interpolation error in
sample positions. This is a table-resolution control, not another blur radius
or another ray count. The same tabulation defines the host and device sampling
paths at their respective precision.

(accuracy-lut-samples-per-px)=
### `accuracy.lut_samples_per_px`

Set the ray lookup-table resolution per image axis. The renderer interpolates
camera rays from this table instead of solving unprojection for every subsample.
Increasing the resolution reduces interpolation error where lens distortion
curves rapidly, while increasing table memory. The camera's calibrated
intrinsics remain unchanged.

(accuracy-pixel-integral)=
### `accuracy.pixel_integral`

Choose how surface terms are estimated when the spatial sample count increases.
`phase1` averages all terms over the selected count. Above 64 samples, `fixed64`
keeps point, normal, and texture estimates on the first 64 samples while additional
samples contribute visibility and shadow information. This limits variation in
smooth surface estimates when a pixel switches sampling bands. At counts of 64
or fewer the two modes agree; `fixed64` is an estimator convention, not a complete
high-sample evaluation of every shading term.

A cluster with no member in the first 64 samples falls back to all of its
members, so that a surface absent from the prefix still has shading terms.

(accuracy-n-band-max)=
### `accuracy.n_band_max`

Use a higher spatial count near predicted silhouettes and shadow edges, while
retaining the base count elsewhere. A non-null value must be a compatible power
of two greater than the base count. Pixels whose rendered edge classification
requires more sampling are escalated and their window is rendered again at the
higher count. This concentrates work on difficult boundaries; `null` gives every
pixel the same count. Compatibility rules include reset mode and the selected
pixel-integral convention.

(accuracy-band-growth)=
### `accuracy.band_growth`

Choose the shape of the predicted high-sample region: complete tiles (`tile`) or
individual pixel-reach regions (`pixel`). Pixel growth can avoid tracing a whole
tile for a small boundary, but requires sufficient pixel-response support for
the escalation rules. The resolver falls back to tile growth for a derived
ineligible case and rejects an explicitly incompatible pixel map.

(accuracy-band-reach)=
### `accuracy.band_reach`

Scale the camera-motion and moving-body contributions used to grow a pixel-band
seed, in that order. Larger reach predicts a broader high-sample region before
rendering; smaller reach relies more on subsequent edge detection and escalation.
These coefficients affect prediction cost, rather than changing an object's
trajectory or the optical support. They apply only with an active band and
`band_growth: pixel`.

## Time sampling and refinement

(accuracy-d-px)=
### `accuracy.d_px`

Limit the predicted image displacement between samples on the initial lattice.
The planner combines motion bounds with geometry to choose per-tile rates.
Smaller values produce a finer starting schedule and reduce dependence on later
midpoint or visibility refinement. This is a displacement budget in pixels,
not an event-position uncertainty or a maximum allowed object speed.

(accuracy-refine-signature)=
### `accuracy.refine_signature`

Split a temporal interval when its endpoint visibility signatures differ.
Signatures summarize changes such as geometry or shadow membership that a smooth
irradiance interpolation can miss. Refinement localizes these transitions before
the pixel model is integrated. Equal signatures alone do not prove constant
irradiance, which is why the midpoint test is a separate control.

(accuracy-refine-tol-log)=
### `accuracy.refine_tol_log`

Bound the midpoint deviation from the endpoint irradiance interpolation using a
relative, log-domain error estimate. For one unmodulated group it compares the
midpoint departure to the minimum irradiance plus the equivalent dark term;
modulated groups also account for waveform bounds. Smaller values refine curved
brightness changes more often. `null` disables this test, leaving motion and
signature rules. The criterion is local; it is not a global bound on event-time
error.

(accuracy-max-refine-levels)=
### `accuracy.max_refine_levels`

Cap the number of refinement rounds below the initial lattice. This bounds the
work generated by repeated midpoint or signature flags. Reaching the cap may
leave a flagged interval unresolved, so a larger cap can be necessary when
starting from a coarse schedule. Refinement is also limited independently by
`min_dt_ns` and any rule-specific stopping criterion.

(accuracy-mid-levels)=
### `accuracy.mid_levels`

Limit how many levels a midpoint flag may refine below the starting lattice.
`null` allows every level permitted by the global cap. A finite value can stop
smooth-intensity refinement early while letting signature-driven localization
continue. This is the economy schedule's separation of smooth variation from
visibility jumps; it does not disable the midpoint samples already taken.

(accuracy-min-dt-ns)=
### `accuracy.min_dt_ns`

Set the smallest dyadic spacing admitted by scheduling and refinement. The value
must be a power of two and is expressed on the true clock. A smaller spacing
allows finer localization at increased sampling cost. Threshold roots are solved
inside the sensor's continuous-time model, so this spacing is distinct from both
the stored event timestamp quantum and the sensor's crossing solution.

(accuracy-tau-jump-ns)=
### `accuracy.tau_jump_ns`

Set the target time scale for signature-jump localization. A signature interval
splits only while its length is at least
`2*tau_jump_ns*max(1, delta_ref/eta_J)`, where `eta_J` is the endpoint jump
estimate. With `delta_ref=0`, all jumps use the same absolute scale. `null`
removes this additional stopping rule and leaves the level and minimum-spacing
limits. This controls render sampling, not readout quantization.

(accuracy-delta-ref)=
### `accuracy.delta_ref`

Set the jump magnitude below which the localization time target is relaxed.
Jumps smaller than this value stop at a proportionally wider bracket. This can
save samples on low-amplitude visibility changes, but such changes may still
matter to a pixel already near threshold. The default `0` avoids amplitude-based
relaxation. It applies with a non-null `tau_jump_ns`.

(accuracy-half-ratio)=
### `accuracy.half_ratio`

After a midpoint flag, refine a half-interval only if its endpoint jump reaches
this fraction of the larger half's jump. Positive values concentrate work on the
stronger half; `0` keeps both. This heuristic can omit slower changes on the
other half and should be assessed separately from the numerical integration
tolerance. It affects midpoint-driven refinement, not the physical threshold.

(accuracy-eps-jump)=
### `accuracy.eps_jump`

Require a signature change to have an endpoint jump estimate above this value
before splitting it. Geometry can change while radiance barely changes, so the
gate avoids refining some photometrically negligible signature changes. `null`
refines every signature flag. Raising the gate may skip small changes that would
matter near threshold; it is not a noise filter on the final event stream.

(accuracy-max-samples-per-window)=
### `accuracy.max_samples_per_window`

Cap the planned initial-lattice sample count in one window. Above the cap, the
finest planned levels are clamped. This deliberately relaxes the requested
schedule and is recorded in run counters. `null` retains the requested levels.
It is different from `execution.device_budget_mb`, which changes batching while
preserving requested work.

(accuracy-level-policy)=
### `accuracy.level_policy`

Choose whether initial rates remain local (`tile`), use the largest active tile
level everywhere (`global`), or impose that level on dynamic tiles while retaining
certified static regions (`global_certified`). Global policies reduce variation
in initial temporal sampling between tiles, usually at higher cost. They retain
the same subsequent refinement rules.

## Visibility and shading conventions

(accuracy-unlit-mask)=
### `accuracy.unlit_mask`

Treat a shadow bit as lit when that light's direct contribution is zero. A
visibility change behind a surface or outside a light's useful contribution then
need not trigger refinement solely through that bit. This changes signature and
class metadata, not the computed irradiance itself.

(accuracy-texture-shadow)=
### `accuracy.texture_shadow`

Choose how a textured shading cluster couples albedo with visibility. `pooled`
uses a shared shadow ratio for the cluster; `weighted` weights each member's
contribution by its own albedo. Weighting preserves correlations such as a shadow
covering a bright texel rather than a dark one. It changes a shading estimator,
not the scene texture or the ray intersection.

(accuracy-shading-patches)=
### `accuracy.shading_patches`

Choose the surfaces allowed to share a shading cluster. `material` uses material
identity; `crease` also separates shells and geometric or normal discontinuities.
Separating these patches prevents averaging surface terms across incompatible
faces, at the cost of more clusters. A change in clustering can affect both
irradiance and boundary signatures.

(accuracy-mover-ao)=
### `accuracy.mover_ao`

Approximate ambient occlusion by other moving bodies. `proxy` uses fitted sphere
sets; `traced` uses reference hemisphere rays; `off` omits this term. Occlusion
attenuates ambient illumination near movers without adding reflected bounce
light. The proxy is an approximation to the represented body geometry, and
static geometry is not added to the ambient occlusion term by this switch.

(accuracy-held-world)=
### `accuracy.held_world`

Place bodies certified still over a window into the static tracing structure.
This reduces repeated moving-body work while preserving their held poses.
Intersection arithmetic can differ in its last bits from the moving-body path;
the setting therefore participates in render reproducibility. It must not be
used to freeze a body that actually moves.

(accuracy-shadow-band)=
### `accuracy.shadow_band`

Apply the high moving-shadow rate bound mainly near predicted shadow boundaries
and retained difficult regions. Other affected regions keep a coarser schedule.
This reduces conservative cone-wide sampling, but is a prediction heuristic:
thin geometry, multiple movers, borders, and related cases retain extra work.
It does not remove shadows from rendering.

(accuracy-shadow-obliquity)=
### `accuracy.shadow_obliquity`

With `projected`, account for the receiver surface's obliquity in the predicted
image speed of a shadow edge. Near-grazing receivers can magnify that speed, so
the implementation caps its projection factors and budgets the raised lattice
levels. The bound uses plan-pass facet normals; it is not a bound for unresolved
subpixel surface relief. `off` retains the baseline shadow-speed treatment.

(accuracy-localization)=
### `accuracy.localization`

`bisect` localizes signature changes by temporal subdivision. `shadow` additionally
runs predict-and-verify shadow diagnostics where supported, while retaining
bisection's output. In this build it is a diagnostic comparison, not an alternate
production event-localization algorithm.

## Sensor integration and noise resolution

(accuracy-eps-l)=
### `accuracy.eps_L`

Set the log-domain tolerance used to split and approximate the deterministic
cascade integration between irradiance samples. Smaller values reduce that
integration approximation at additional substep cost. This is distinct from the
renderer midpoint tolerance: one controls the supplied photocurrent curve, the
other how the sensor follows it. It is inactive for the instantaneous
photoreceptor.

(accuracy-noise-leaf-level)=
### `accuracy.noise_leaf_level`

Set the temporal resolution of leaves in the stochastic noise tree. Leaf length
is `2**(16-level)` ns, so increasing the level gives shorter leaves and more work
inside each fixed noise cell. The tree refines the same keyed stochastic process
rather than drawing unrelated noise each time the renderer adds a sample.
This helps keep noise realizations independent of render scheduling.

(accuracy-noise-margin-z)=
### `accuracy.noise_margin_z`

Set the stochastic exclusion margin in standard deviations used to discard
intervals unlikely to contain a missed threshold crossing. Larger values are
more conservative and retain more search work. The margin concerns Gaussian
noise-tail exclusion within the model; it is not a deterministic guarantee that
no rare crossing can be missed.

(accuracy-noise-diffusion)=
### `accuracy.noise_diffusion`

Choose the input-node noise diffusion approximation within a fixed noise cell.
`frozen` holds the local diffusion coefficient at its cell approximation;
`transient` adjusts it to reproduce the transient input-node variance at the
nodes. This changes the modeled stochastic approximation during changing
illumination. Frozen EVIMO2 profiles retain the approximation under which they
were calibrated.

(accuracy-sensor-precision)=
### `accuracy.sensor_precision`

Select `fp64` or `fp32` arithmetic for the CPU sensor. The CUDA sensor uses
`fp32` regardless of this CPU setting. Precision changes rounding in integration
and crossing decisions; it is separate from the time unit, which remains integer
nanoseconds. Use the recorded precision when comparing or restoring low-level
sensor states.

## Initialization

(accuracy-initial-state)=
### `accuracy.initial_state`

Choose how to initialize the comparator reference. `phase` uses
`sensor.initial_phase` without preceding comparator history. `stationary` samples
from a stationary probe under the held starting illumination, reducing artificial
startup behavior associated with an arbitrary phase. Stationarity here concerns
the initial pixel model, not a claim that the moving scene is stationary.
Instantaneous or noiseless models retain the phase initialization law.

(accuracy-stationary-burnin-ns)=
### `accuracy.stationary_burnin_ns`

Set the held-input history used by the stationary probe. `0` derives
`1.5/R_L`, clamped to 60–600 s, from the temperature-adjusted median leak rate.
For this automatically derived duration, an enabled high-pass filter additionally
caps the history at `max(1, 10*tau_hp)` s; an explicit duration is not capped.
The history is rounded to noise cells. This is a synthetic comparator-history
calculation; it does not require the scene to contain that much motion history.

(accuracy-stationary-probe-pixels)=
### `accuracy.stationary_probe_pixels`

Set how many representative pixels the stationary probe evolves. Other pixels
receive a comparator snapshot from a nearest representative. More probe pixels
can represent parameter-map variation more closely but increase initialization
cost. The probe is an initialization approximation, rather than an additional
render or a change in sensor resolution.

(accuracy-warmup)=
### `accuracy.warmup`

Select held illumination (`static`) or available preceding scene motion
(`motion`) before the first output window. Warm-up evolves the sensor while
suppressing its events, allowing temporal filters and comparator state to settle.
It complements stationary reference initialization: a stationary comparator probe
does not reconstruct the real preceding scene motion.

(accuracy-warmup-ns)=
### `accuracy.warmup_ns`

Set the requested warm-up duration. Static warm-up requires at least 50 ms.
Motion warm-up rounds to whole windows and uses covered scene history; when the
requested history is unavailable, static completion extends initialization to
at least 50 ms. Fully covered motion warm-up can be shorter, including fidelity's
two windows (16.777216 ms). The run's output interval remains unchanged.

## Implementation and related methods

These numerical choices are defined by the resolver and engine, rather than by
universal sensor constants. See {download}`sampling.h <../../../../../src/engine/sampling.h>`,
{download}`engine_impl.h <../../../../../src/engine/engine_impl.h>`, and
{download}`sensor_model.cpp <../../../../../src/sensor/sensor_model.cpp>` for the
implemented rules. Spatial low-discrepancy sampling follows the principles in
[pbrt, Sampling and Reconstruction](https://pbr-book.org/4ed/Sampling_and_Reconstruction).
The [sensor model](../../guide/sensors.md) identifies the physical mechanisms
whose numerical approximation these controls govern.

## Quick reference

### Profile and design

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `accuracy.profile` | `enum` | `"fidelity"` | Accuracy preset applied before explicit parameter overrides. Choices: `phase1`, `fidelity`, `balanced`, `reference`. |
| `accuracy.design_contrast` | `float · log-e` | `1.0` | Natural-log edge contrast used to derive the base sample count. Fidelity/balanced only. Range: >0. |
| `accuracy.design_threshold` | `float · log-e` | derived | Derived design contrast threshold across selected cameras and polarities. Fidelity/balanced only. Range: >0. |

### Spatial sampling

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `accuracy.n_subsamples` | `int` | `64` | Subpixel samples per pixel-time query; powers of two preserve nested sampling patterns. Range: [1, 1024]. |
| `accuracy.psf_truncation_sigmas` | `float` | `3.0` | Gaussian response truncation radius in standard deviations, beyond half the aperture width. Range: >0. |
| `accuracy.warp_cells` | `int` | `256` | Cells in the pixel-response inverse-CDF table. Range: [2, 65535]. |
| `accuracy.lut_samples_per_px` | `int · 1/px` | `2` | Ray lookup-table nodes per pixel axis. Range: ≥1. |
| `accuracy.pixel_integral` | `enum` | `"phase1"` | `phase1` integrates all quantities over N samples; `fixed64` fixes point, normal, and texture estimates to the first 64. Choices: `phase1`, `fixed64`. |
| `accuracy.n_band_max` | `int?` | `null` | Higher sample count near predicted silhouettes/shadow edges; `null` uses one count. Range: [64, 1024]. |
| `accuracy.band_growth` | `enum` | `"tile"` | Grow higher-sample regions by tiles or individual pixel reach. Choices: `tile`, `pixel`. |
| `accuracy.band_reach` | `float[2]` | `[1.0, 1.0]` | Camera and mover motion scales `[sigma_cam, sigma_mov]` for pixel-band reach. Range: ≥0. |

### Adaptive scheduling

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `accuracy.refine_signature` | `bool` | `true` | Refine intervals whose visibility or shadow signatures change. |
| `accuracy.refine_tol_log` | `float? · log-e` | `0.002` | Midpoint log-intensity error tolerance; `null` disables midpoint refinement. Range: >0. |
| `accuracy.max_refine_levels` | `int` | `4` | Maximum refinement rounds below the initial lattice. Range: [0, 13]. |
| `accuracy.mid_levels` | `int?` | `null` | Midpoint-driven refinement depth; `null` uses every allowed refinement level. Range: [0, 13]. |
| `accuracy.tau_jump_ns` | `int? · ns` | `null` | Target jump-localization time scale; `null` uses the refinement-level limit alone. Range: [1, 8388608]. |
| `accuracy.delta_ref` | `float · log-e` | `0.0` | Jump magnitude for full localization; `0` applies the same time target to all jumps. Range: ≥0. |
| `accuracy.half_ratio` | `float` | `0.0` | Relative jump threshold for refining a midpoint-flagged half; `0` refines both halves. Range: [0, 1]. |
| `accuracy.eps_jump` | `float? · log-e` | `null` | Minimum endpoint jump for signature refinement; `null` refines every signature change. Range: ≥0. |
| `accuracy.min_dt_ns` | `int · ns` | `1024` | Minimum lattice/refinement interval; must be a power of two. Range: [1, 8388608]. |
| `accuracy.d_px` | `float · px` | `0.25` | Maximum predicted image displacement per initial sampling interval. Range: >0. |
| `accuracy.max_samples_per_window` | `int?` | `33554432` | Initial sample-count cap per window; `null` disables clamping. Range: ≥1. |
| `accuracy.level_policy` | `enum` | `"tile"` | Per-tile levels, one global level, or a global level only on uncertified dynamic regions. Choices: `tile`, `global`, `global_certified`. |

### Visibility and shading

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `accuracy.unlit_mask` | `bool` | `false` | Ignore shadow-signature changes for lights with zero direct contribution; irradiance is unchanged. |
| `accuracy.texture_shadow` | `enum` | `"pooled"` | Pool shadow ratios, or weight each subpixel's shadow contribution by its own albedo. Choices: `pooled`, `weighted`. |
| `accuracy.shading_patches` | `enum` | `"material"` | Group samples by material alone, or split at material, shell, and crease boundaries. Choices: `material`, `crease`. |
| `accuracy.mover_ao` | `enum` | `"off"` | Moving-body ambient occlusion: disabled, sphere proxies, or traced subpixel rays. Choices: `off`, `proxy`, `traced`. |
| `accuracy.held_world` | `bool` | `true` | Merge bodies held over a window into static trace geometry; may change last-bit results. |
| `accuracy.shadow_band` | `bool` | `false` | Restrict moving-shadow rate bounds to predicted boundary regions. |
| `accuracy.shadow_obliquity` | `enum` | `"off"` | Include capped receiver-projection factors in shadow-edge speed bounds. Choices: `off`, `projected`. |
| `accuracy.localization` | `enum` | `"bisect"` | Bisection localization, or shadow diagnostics alongside bisection with identical outputs. Choices: `bisect`, `shadow`. |

### Sensor integration and initialization

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `accuracy.eps_L` | `float · log-e` | `0.002` | Log-intensity integration tolerance for the cascade; inactive for instantaneous photoreceptors. Range: (0, 0.1]. |
| `accuracy.noise_leaf_level` | `int` | `5` | Noise-tree leaf level; leaf duration is `2**(16 - level)` ns. Range: [0, 10]. |
| `accuracy.noise_margin_z` | `float` | `6.0` | Noise exclusion margin in standard deviations. Range: >0. |
| `accuracy.noise_diffusion` | `enum` | `"frozen"` | Input-node diffusion approximation: frozen within cells, or transient with exact node variance. Choices: `frozen`, `transient`. |
| `accuracy.initial_state` | `enum` | `"phase"` | Initialize from the configured phase, or from a sampled stationary comparator state. Choices: `phase`, `stationary`. |
| `accuracy.stationary_burnin_ns` | `int · ns` | `0` | Stationary-probe history; `0` derives the duration from leak rate and high-pass filtering. Range: ≥0. |
| `accuracy.stationary_probe_pixels` | `int` | `32` | Representative pixels used to initialize stationary comparator states. Range: ≥1. |
| `accuracy.sensor_precision` | `enum` | `"fp64"` | CPU sensor arithmetic; the CUDA sensor always uses `fp32`. Choices: `fp64`, `fp32`. |
| `accuracy.warmup_ns` | `int · ns` | `50331648` | Warm-up duration; generated warm-up events are discarded. Range: ≥8388608. |
| `accuracy.warmup` | `enum` | `"static"` | Hold the first frame, or use available motion windows followed by static completion. Choices: `static`, `motion`. |
