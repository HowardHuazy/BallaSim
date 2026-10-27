# Accuracy profiles

`accuracy.profile` selects numerical defaults. It does not select a physical
sensor model. Explicit profile-file, run-configuration, and command-line values
override the preset, and dependent defaults are recomputed from resolved inputs.

| Profile | Purpose |
| --- | --- |
| `fidelity` | Default adaptive settings derived from sensor thresholds, noise, and readout timing. |
| `balanced` | Larger motion intervals with bounded jump refinement. |
| `reference` | Uniform high spatial sampling and finer visibility refinement. |
| `phase1` | Frozen baseline settings for reproducibility. |

## Effective defaults

### `fidelity`

Use sensor-dependent numerical targets for a general run. Spatial sample design
accounts for the lowest effective threshold, while integration tolerances track
configured noise and temporal spacing tracks readout resolution. Eligible edge
regions use a higher spatial count. This avoids applying one arbitrary numerical
table to sensors with substantially different sensitivity, but does not replace
validation for a particular scene.

### `balanced`

Retain fidelity's spatial design and sensor integration while using a coarser
initial displacement schedule and bounded temporal refinement. Midpoint testing
and jump localization still detect changes after the initial plan. This trades
conservative sampling work for an economy policy; it does not make the sensor
less noisy or change its contrast thresholds. Compare important results against
a finer schedule when visibility changes dominate.

### `reference`

Use 256 spatial samples, a smaller displacement target than the balanced policy,
and deeper visibility refinement for numerical comparisons. This is a
conformance schedule for the implemented renderer and sensor. It retains their
physical approximations and does not add effects such as general interreflection.

### `phase1`

Preserve the fixed baseline table used by earlier runs. It retains the original
spatial estimator, noise diffusion, phase initialization, and static warm-up.
Choose it when reproducing that baseline rather than assuming it is the current
recommended policy for a new sensor. Explicit overrides can still change the
resulting configuration.

## Profile comparison

`N` is the resolved `accuracy.n_subsamples`. `epsilon` and `dt_min` are defined
below. Values shown here replace the schema base defaults; other settings retain
their base defaults unless explicitly overridden.

| Parameter | `fidelity` | `balanced` | `reference` | `phase1` |
| --- | --- | --- | --- | --- |
| `accuracy.n_subsamples` | Derived; see below | Derived | `256` | `64` |
| `accuracy.d_px` | `2 / sqrt(N)` | `min(8 / sqrt(N), 1)` | `0.25` | `0.25` |
| `accuracy.eps_L` | `epsilon` | `epsilon` | `epsilon` | `0.002` |
| `accuracy.refine_tol_log` | `epsilon` | `epsilon` | `epsilon` | `0.002` |
| `accuracy.min_dt_ns` | `dt_min` | `dt_min` | `dt_min` | `1024` |
| `accuracy.max_samples_per_window` | `null` | `null` | `null` | `33554432` |
| `accuracy.max_refine_levels` | `4` | `13` | `13` | `4` |
| `accuracy.mid_levels` | `null` | `4` | `null` | `null` |
| `accuracy.tau_jump_ns` | `null` | `4096` | `null` | `null` |
| `accuracy.eps_jump` | `epsilon`, with profile exception below | `epsilon` | `null` | `null` |
| `accuracy.warmup` | `motion` | `motion` | `motion` | `static` |
| `accuracy.warmup_ns` | Two windows; six for static warm-up | Same | Same | `50331648` |
| `accuracy.initial_state` | `stationary` | `stationary` | `stationary` | `phase` |
| `accuracy.noise_diffusion` | `transient`, with profile exception below | Same | Same | `frozen` |
| `accuracy.pixel_integral` | `fixed64`, with profile exception below | Same | `phase1` | `phase1` |
| `accuracy.n_band_max` | `256` when eligible; otherwise `null` | `null` | `null` | `null` |
| `accuracy.band_growth` | `pixel` when eligible; otherwise `tile` | `tile` | `tile` | `tile` |
| `accuracy.band_reach` | `[0.25, 0]` for pixel growth; otherwise `[1, 1]` | `[1, 1]` | `[1, 1]` | `[1, 1]` |
| `accuracy.shadow_band` | `false` | `true` | `false` | `false` |

## Derived values

The design threshold is the minimum `theta - 3 * sigma_theta` over selected
cameras and polarities, floored at `0.10`. The base count is the smallest
`4**k`, with `k >= 2`, satisfying:

```text
N >= (exp(design_contrast) - 1) / (exp(design_threshold / 4) - 1)
```

It is capped at 256, with a warning when the design requirement exceeds the cap.
Fidelity raises a derived count of 16 to 64 when higher-sample bands are enabled.

`epsilon = 0.002 * min(1, sigma_min / 0.02)`, where `sigma_min` is the smallest
stationary comparator-noise standard deviation among noisy cameras at the
reference input pole of `1e9` s⁻¹. Without noisy cameras the multiplier is one.
`refine_tol_log` derives from the same noise criterion; explicitly changing
`eps_L` does not change it.

`dt_min = max(1024, pow2ceil(q_min))`, where `q_min` is the smallest positive
readout quantum in ns; without a quantum it is 1024 ns. One engine window is
`2**23` ns, so two warm-up windows are `16777216` ns.

## Bands and calibrated profiles

Fidelity enables a band count of 256 only with a compatible power-of-two base
count below 256, physical reset, and the `fixed64` integral. An explicit base
count must be at least 64. An explicit `phase1` integral disables this default.

Pixel-band growth additionally requires a response reach of at least 0.75 px
in every selected camera:

```text
reach = 0.5 * sensor.aperture_px
        + min(accuracy.psf_truncation_sigmas, 3) * optics.blur_sigma_px
```

EVIMO2 v0/v1 sensor profiles preserve their calibrated `frozen` noise diffusion
and `phase1` pixel integral. Fidelity also disables its default bands and jump
gate for these profiles and uses their smallest mean contrast threshold for
design calculations.

Fidelity and balanced derive `optics.vignetting_law: jacobian` when lens gain is
active, except under these calibrated EVIMO2 profiles. Other cases use `cos4`.
