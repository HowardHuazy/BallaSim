# Frames and APS

Frame parameters control exposure timing, BallaSim's exposure reference and an
optional APS readout model. Use `cameras.<camera>.frames.*`, or
`cameras.default.frames.*` for shared values. These settings do not change event
simulation; frame files carry their own resolved values and digests.

The available commands are `frames reference` and `frames aps`.
`matched` and `full` describe Blender planning interfaces; their complete
render/ingest workflow is not available in this build. Settings in the Blender
planning sections below are recorded schema choices, not controls for the
reference renderer.

See [Outputs & ground truth](../../guide/outputs.md#reference-frames-and-aps)
for commands and [Output API](../api/frames.md#frames) for Python interfaces.

## Timeline and exposure

The timeline places exposure anchors on the true clock. Exposure duration sets
the interval averaged by each row; frame period sets how often that measurement
is repeated. Global and rolling shutters share this definition, while rolling
shutters offset successive rows. Cropping changes stored image extent without
resetting the full camera's row timing.

| Parameter | Type / unit | Default |
| --- | --- | --- |
| {ref}`frames.mode <frames-mode>` | `enum` | `off` |
| {ref}`frames.timeline <frames-timeline>` | nullable `enum` | `null` |
| {ref}`frames.stride <frames-stride>` | `int ≥ 1` | `1` |
| {ref}`frames.t0_ns <frames-t0-ns>` | nullable `int`, ns | `null` |
| {ref}`frames.period_ns <frames-period-ns>` | nullable `int > 0`, ns | `null` |
| {ref}`frames.fps <frames-fps>` | `float > 0`, Hz | `30.0` |
| {ref}`frames.count <frames-count>` | nullable `int ≥ 1` | `null` |
| {ref}`frames.exposure_ns <frames-exposure-ns>` | nullable `int ≥ 0`, ns | `null` |
| {ref}`frames.shutter <frames-shutter>` | `enum` | `global` |
| {ref}`frames.row_dt_ns <frames-row-dt-ns>` | `int ≥ 0`, ns | `0` |
| {ref}`frames.readout <frames-readout>` | `enum` | `top_down` |
| {ref}`frames.roi_px <frames-roi-px>` | nullable `int[4]`, px | `null` |
| {ref}`frames.outputs <frames-outputs>` | nullable `string[]` | `null` |

(frames-mode)=
### Frame mode (`frames.mode`)

Select `off`, `reference`, `matched` or `full`. `reference` renders
exposure-averaged irradiance with BallaSim's renderer for any scene source.
`matched` is intended to reproduce that response in Blender; `full` retains
Blender's authored light transport. The latter two require a Blender exchange
and currently have no end-to-end writer. Other frame settings are inert under
`off`; explicitly naming a camera to `frames reference` enables reference output.

(frames-timeline)=
### Timeline source (`frames.timeline`)

Select equally spaced exposures (`uniform`) or authored Blender frame times
(`blender`). `null` chooses Blender timing for exchanges and uniform timing
otherwise. The current reference command requires `uniform`; set it explicitly
when rendering a Blender-derived run.

(frames-stride)=
### Timeline stride (`frames.stride`)

Retain every nth authored Blender frame. Increasing the stride reduces planned
frame output without changing exposure duration or scene motion. It has no
effect on the current uniform reference timeline.

(frames-t0-ns)=
### Initial exposure time (`frames.t0_ns`)

Set the true-clock start of exposure of frame zero's first-read row. `null`
uses the event run's aligned start. This is an exposure anchor, rather than a
midpoint or reported event timestamp; rolling-shutter rows start later according
to `row_dt_ns` and `readout`.

(frames-period-ns)=
### Exact frame period (`frames.period_ns`)

Set the integer interval between uniform exposure anchors. A non-null value
takes precedence over `fps`, allowing exact timing without floating-point rate
conversion. Exposures may overlap when their duration exceeds this period.

(frames-fps)=
### Frame rate (`frames.fps`)

Set the uniform frame rate when `period_ns` is null. The period is
`round_half_even(1e9 / fps)` ns; integer rounding may make the realized rate
slightly different. This changes frame spacing, not exposure duration.

(frames-count)=
### Requested frame count (`frames.count`)

Limit the requested frame sequence. `null` extends the uniform sequence to the
end of scene support. Candidate exposures outside the supported trajectory
interval are dropped and recorded; the renderer does not shorten them to fit.

(frames-exposure-ns)=
### Exposure duration (`frames.exposure_ns`)

Set each row's exposure duration. `null` denotes the authored Blender shutter
for an exchange, or 10 ms for other sources. The current reference command cannot
read an authored shutter, so Blender-derived runs need an explicit value.
`0` evaluates an instantaneous image. Longer exposures increase motion blur and
APS photo-electron counts while leaving `mono` in average irradiance units.

(frames-shutter)=
### Shutter model (`frames.shutter`)

`global` exposes all rows simultaneously. `rolling` offsets row exposures and
therefore samples different scene motion in each row. With zero `row_dt_ns`,
rolling timing reduces to simultaneous row timing. The full exposure, including
row offsets, must lie within scene support.

(frames-row-dt-ns)=
### Row delay (`frames.row_dt_ns`)

Set the delay between consecutive rows in read order. For row `y`, start time
is `t0 + rho(y)*row_dt_ns`, where `rho(y)` is `y` for top-down readout and
`H-1-y` for bottom-up readout. `H` is the full image height; cropping preserves
the original row times. The setting is used only for a rolling shutter.

(frames-readout)=
### Row order (`frames.readout`)

Select `top_down` or `bottom_up` rolling-shutter row order. Reversing it reverses
the temporal scan through the image and can reverse motion skew. It does not
reverse image storage or pixel coordinates.

(frames-roi-px)=
### Image region (`frames.roi_px`)

Render the half-open rectangle `[x0, y0, x1, y1)`. `null` uses `run.crop_px`, or
the full image when no crop is configured. The rectangle must be nonempty and
inside the camera resolution. Arrays use ROI dimensions; metadata retains the
full resolution and origin. APS random draws remain keyed by full-frame pixels.

(frames-outputs)=
### Stored products (`frames.outputs`)

Choose stored products from `rgb`, `mono`, `groups`, `reference` and `aps`.
The current reference writer accepts `mono`, `reference` and `aps`, defaults to
`mono`, and rejects `rgb` and `groups`. `reference` stores the exposure reference
under that dataset name. `aps` requires `mono` and an event camera's recorded
photocurrent scale. Blender planning defaults to `rgb` and `mono`.

## Reference integration

The stored reference is the mean over each row's exposure, rather than a
photo-electron count. For light group `g`, the renderer samples irradiance
`I_g` while applying the exact waveform mean over each integration subinterval:

```{math}
F(x)=\sum_g\frac{1}{K}\sum_{j=0}^{K-1}
       \overline{m}_{g,j}\,I_g(x,t_j).
```

Geometry and visibility still require numerical time sampling. A denser probe
checks estimated error; it does not constitute an analytic error bound.

| Parameter | Type / unit | Default |
| --- | --- | --- |
| {ref}`frames.ref.h_px <frames-ref-h-px>` | `float > 0`, px | `0.25` |
| {ref}`frames.ref.k_max <frames-ref-k-max>` | `int`, `[2, 4096]` | `256` |
| {ref}`frames.ref.probe_tol <frames-ref-probe-tol>` | `float > 0`, log-intensity | `0.005` |

(frames-ref-h-px)=
### Initial sample spacing (`frames.ref.h_px`)

Control initial time-sample spacing from estimated image travel `D` during an
exposure. For `D ≥ 4 px`, the initial sample count is approximately
`ceil(1.25*D/h_px)`; smaller displacements use a reduced effective spacing.
Decreasing `h_px` increases initial rendering work. A static scene uses one
geometry sample with exact waveform averaging.

(frames-ref-k-max)=
### Temporal sample cap (`frames.ref.k_max`)

Cap the all-pixel integration level. The probe can use a denser level on its
mask, up to three times that count, and those values supply the masked output
pixels. Counts are also limited by integer exposure duration. Reaching the cap
with excessive estimated error emits `FRM-REF-K`; it does not silently certify
the frame.

(frames-ref-probe-tol)=
### Probe tolerance (`frames.ref.probe_tol`)

Set the accepted 99.9th percentile of estimated log-intensity error on the probe
mask. The mask includes temporally changing pixels and a regular spatial grid.
The estimator compares `K` and `3K` sampling with a small intensity floor;
failing the tolerance triples the all-pixel level until the cap. Lower values
increase refinement and may expose a sample-cap warning.

## Blender render planning

These choices define the planned matched/full rendering behavior. They do not
alter `frames reference` or enable the unavailable render/ingest workflow.

| Parameter | Type / unit | Default |
| --- | --- | --- |
| {ref}`frames.fitted_poses <frames-fitted-poses>` | `enum` | `physics` |
| {ref}`frames.keep_raw <frames-keep-raw>` | `enum` | `none` |
| {ref}`frames.route <frames-route>` | `enum` | `auto` |
| {ref}`frames.supersample <frames-supersample>` | nullable `int`, `{2, 4, 8}` | `null` |
| {ref}`frames.spp <frames-spp>` | nullable `int ≥ 1` | `null` |
| {ref}`frames.seed <frames-seed>` | `int ≥ 0` | `0` |
| {ref}`frames.matched.render_groups <frames-matched-render-groups>` | `enum` | `separate` |
| {ref}`frames.matched.texture_mode <frames-matched-texture-mode>` | `enum` | `float_linear` |
| {ref}`frames.motion_tol_px <frames-motion-tol-px>` | `float > 0`, px | `0.05` |
| {ref}`frames.factor_tol <frames-factor-tol>` | `float > 0` | `0.005` |

(frames-fitted-poses)=
### Fitted motion (`frames.fitted_poses`)

Choose which fitted trajectories would replace authored motion in a saved
Blender copy: `none`, physics bodies (`physics`), or bodies, camera rigs and
light carriers (`all`). Fitted poses reduce differences caused by Blender's
interpolation of physics caches. Transforms with non-diagonal constant scale
cannot be represented by ordinary location/rotation/scale keys and require a
reported fallback.

(frames-keep-raw)=
### Intermediate retention (`frames.keep_raw`)

Specify retention of intermediate Blender passes after successful ingest:
`none`, decoded `npy`, or decoded arrays plus `npy_exr`. Raw passes support
inspection of frame conversion and increase storage. This setting has no effect
on reference HDF5 or display PNG output.

(frames-route)=
### Camera-response route (`frames.route`)

Select the planned camera-response route: native film filtering (`film`), a
high-resolution pinhole render followed by resampling (`supersample`), or a
generated OSL camera (`osl`). `auto` chooses an applicable route. Film filtering
is limited to compatible undistorted responses; supersampling approximates the
pixel integral and has pinhole-coverage and rolling-warp limits. OSL is intended
to sample the specified camera and pixel response directly.

(frames-supersample)=
### Spatial supersampling (`frames.supersample`)

Choose linear resolution factor `2`, `4` or `8` for the supersampling route.
The number of source pixels grows approximately with the square of the factor;
higher factors reduce resampling error but increase render and memory cost.
`null` specifies `8` for identity-map matched mode and `4` otherwise.

(frames-spp)=
### Cycles sample count (`frames.spp`)

Set Cycles samples per source pixel, independently of temporal integration.
Higher values reduce Monte Carlo noise rather than temporal or resampling error.
Planned defaults are 512 for matched film/OSL, 32 for matched 8× supersampling,
128 for matched 4× supersampling, and 128 for full mode. An explicit value
replaces the route default.

(frames-seed)=
### Cycles realization (`frames.seed`)

Set the Cycles sampling seed. Planned full-mode comparison halves use `seed`
and `seed+1` to estimate Monte Carlo disagreement. This seed does not select
BallaSim's pixel sample pattern or the APS noise realization.

(frames-matched-render-groups)=
### Light-group passes (`frames.matched.render_groups`)

`separate` specifies one render per light group. `together` specifies a shared
render with light-group passes. Shared rendering reduces pass count, but divides
sampling effort among groups and can increase noise in weakly illuminated
regions. Group separation is required to apply each temporal waveform correctly.

(frames-matched-texture-mode)=
### Texture interpolation (`frames.matched.texture_mode`)

`float_linear` specifies engine-decoded linear-light float textures with linear
interpolation. `closest` retains byte textures with nearest-neighbor lookup,
which is coarser near texel boundaries. The planned float-texture path falls back
to Closest when `execution.frames_texture_mb` is exceeded.

(frames-motion-tol-px)=
### Motion tolerance (`frames.motion_tol_px`)

Bound the planned image-space error from Cycles motion interpolation and time
resolution. Smaller values request more motion steps or exposure subdivisions.
This controls how Blender follows fitted motion, independently of
`frames.ref.probe_tol`, which checks BallaSim's reference integration.

(frames-factor-tol)=
### Factorization tolerance (`frames.factor_tol`)

Limit the planned relative error from separating temporal light modulation from
motion-dependent irradiance, and from freezing moving lights in a sub-render.
A smaller value requests shorter sub-exposures or row bands. The reference API
reports the associated `fact_p99` diagnostic, but the current reference command
does not generate a Blender subdivision plan.

## Full-render calibration planning

These schema values describe comparison of authored Blender light transport
against the direct-light reference. The calibration and correction writer is
not available in this build.

| Parameter | Type / unit | Default |
| --- | --- | --- |
| {ref}`frames.full.bounces <frames-full-bounces>` | nullable `int ≥ 0` | `null` |
| {ref}`frames.full.correction <frames-full-correction>` | `enum` | `off` |
| {ref}`frames.full.calib_frames <frames-full-calib-frames>` | nullable `int[]` | `null` |
| {ref}`frames.full.calib_spp <frames-full-calib-spp>` | `int ≥ 16` | `64` |
| {ref}`frames.full.contrast_ref <frames-full-contrast-ref>` | nullable `float > 0` | `null` |

(frames-full-bounces)=
### Light-transport depth (`frames.full.bounces`)

Specify the Cycles light-bounce limit for full renders; `null` preserves the
authored scene value. Additional bounces admit indirect illumination absent from
the event renderer. Raising this value therefore changes the modeled light
transport, not just image quality.

(frames-full-correction)=
### Smooth correction (`frames.full.correction`)

Choose `off` or the planned `lowpass` multiplicative correction for a full
render. The correction is intended to remove smooth radiometric differences
while retaining the raw comparison and correction map. It cannot establish
agreement at shadows, reflections or other transport features absent from the
reference.

(frames-full-calib-frames)=
### Calibration selection (`frames.full.calib_frames`)

Select zero-based frames used to estimate radiometric scale and assess full-mode
disagreement. `null` denotes first, middle and last. Frames should represent the
lighting and visibility of the sequence; a scale fitted to a narrow subset does
not account for changing indirect light.

(frames-full-calib-spp)=
### Calibration samples (`frames.full.calib_spp`)

Set Cycles samples per pixel for the planned direct-only calibration passes.
Increasing it reduces noise in scale estimates and disagreement diagnostics.
It is independent of the main full render's `spp` and cannot correct differences
between the two light-transport models.

(frames-full-contrast-ref)=
### Disagreement scale (`frames.full.contrast_ref`)

Set the log-contrast scale used to classify frame disagreement relative to an
event threshold. `null` uses the camera's smaller mean ON/OFF threshold. This
changes the interpretation of a comparison, not the rendered intensity or the
event sensor thresholds.

## APS readout

The APS converts mean irradiance `F` into expected signal electrons using the
event camera's photocurrent scale, exposure duration and static gain map.
It then applies shot noise, fixed-pattern errors, full-well saturation, read
noise and ADC conversion:

```{math}
\begin{aligned}
\mu_e&=\max(0,(1+p_x)S_x)+I_{\mathrm{dark}}T,\\
n_e&=\min(\operatorname{Poisson}(\mu_e)+d_x,\mathrm{FW})
       +\mathcal N(0,\sigma_r^2),\\
\mathrm{DN}&=\operatorname{clip}\!\left(
 \operatorname{round}_{\mathrm{even}}(K n_e+y_0),0,2^b-1\right).
\end{aligned}
```

Here `p_x` and `d_x` are fixed pixel maps. Photon shot noise remains present even
when read noise and non-uniformity are zero. Cropped veiling glare uses an ROI
mean as an approximation to full-frame glare. APS currently requires an event
camera's recorded scale; a frame-only camera is refused with `FRM-APS-SCALE`.

| Parameter | Type / unit | Default | Basis |
| --- | --- | --- | --- |
| {ref}`frames.aps.bits <frames-aps-bits>` | `int`, `[8, 16]` | `10` | DAVIS-class prior |
| {ref}`frames.aps.full_well_e <frames-aps-full-well-e>` | `float > 0`, e⁻ | `30900` | Derived device prior |
| {ref}`frames.aps.read_noise_e <frames-aps-read-noise-e>` | `float ≥ 0`, e⁻ RMS | `55` | Device prior |
| {ref}`frames.aps.dark_current_e_s <frames-aps-dark-current-e-s>` | `float ≥ 0`, e⁻/s | `18000` | Uncertain placeholder |
| {ref}`frames.aps.prnu <frames-aps-prnu>` | `float ≥ 0` | `0.042` | Uncertain placeholder |
| {ref}`frames.aps.dsnu_e <frames-aps-dsnu-e>` | `float ≥ 0`, e⁻ | `20` | Assumed placeholder |
| {ref}`frames.aps.gain_dn_per_e <frames-aps-gain-dn-per-e>` | nullable `float > 0`, DN/e⁻ | `null` | Derived when null |
| {ref}`frames.aps.offset_dn <frames-aps-offset-dn>` | `float ≥ 0`, DN | `16` | Prior |
| {ref}`frames.aps.seed <frames-aps-seed>` | nullable `int ≥ 0` | `null` | Run choice |

(frames-aps-bits)=
### ADC width (`frames.aps.bits`)

Set the unsigned ADC range `0 … 2**bits-1`. The HDF5 array remains `uint16` for
all supported bit depths. Changing the bit depth changes quantization and,
when gain is automatic, DN per electron; it does not change full-well capacity.

(frames-aps-full-well-e)=
### Charge capacity (`frames.aps.full_well_e`)

Set the charge-capacity ceiling applied before read noise. Larger values defer
charge saturation and reduce automatic conversion gain. The default is inferred
from a DAVIS346 dynamic-range/read-noise specification; it is not a calibrated
capacity for an arbitrary camera.

(frames-aps-read-noise-e)=
### Read noise (`frames.aps.read_noise_e`)

Set independent per-read Gaussian noise in electrons RMS, added after full-well
clipping. It affects dark and illuminated pixels alike. `0` disables this term;
it does not disable Poisson shot noise. The default is a DAVIS346 prior.

(frames-aps-dark-current-e-s)=
### Dark current (`frames.aps.dark_current_e_s`)

Add `dark_current_e_s * exposure_s` electrons to the Poisson mean. Longer
exposures increase both dark signal and its shot noise. The default is an
uncertain placeholder: device documentation reports conflicting values of
18000 and 1800 e⁻/s. Set it from the intended sensor and temperature.

(frames-aps-prnu)=
### Response non-uniformity (`frames.aps.prnu`)

Set the standard deviation of the fixed multiplicative response error
`p_x ~ Normal(0, prnu**2)`. It scales signal electrons before shot noise and
remains constant across frames. The default interprets a reported 4.2% fixed
pattern figure as PRNU; that interpretation is uncertain.

(frames-aps-dsnu-e)=
### Dark-signal non-uniformity (`frames.aps.dsnu_e`)

Set the standard deviation of the fixed additive pixel error
`d_x ~ Normal(0, dsnu_e**2)`, in electrons. It is added after the Poisson draw and
before full-well clipping. This model treats DSNU as a per-read charge offset,
not an exposure-scaled dark-current map. The default is assumed.

(frames-aps-gain-dn-per-e)=
### Conversion gain (`frames.aps.gain_dn_per_e`)

Set linear conversion gain from electrons to DN. `null` uses
`(2**bits-1)/full_well_e`. Higher explicit gain improves DN resolution but reaches
the ADC rail at less charge. Charge saturation and ADC clipping are separate;
read noise and black offset can clip even before a nominal full-well signal.

(frames-aps-offset-dn)=
### Black level (`frames.aps.offset_dn`)

Add the black-level offset after gain and before integer rounding. Increasing
it leaves room for negative noise excursions around black but reduces positive
ADC headroom. It is a digital offset and contributes no shot noise.

(frames-aps-seed)=
### APS realization (`frames.aps.seed`)

Select the APS fixed maps and temporal noise realization; `null` uses
`run.sensor_seed`. Draws depend on seed, bundle camera index, full-frame pixel
and true-clock row exposure start. The same inputs reproduce the APS array
independently of worker count. Changing the seed changes both fixed and temporal
errors.

## References

- [iniVation DAVIS346 documentation](https://docs.inivation.com/hardware/current-products/davis346.html): device priors; the uncertain entries above remain tunable assumptions.
- [Haessig et al., Bio-inspired Polarization Event Camera](https://arxiv.org/abs/2112.01933): PDAVIS346 APS/ADC context, rather than a characterization of every configured default.
- The readout follows the EMVA 1288 Poisson–Gaussian model; BallaSim's fixed-pattern placement, glare approximation and defaults are stated above.

```{toctree}
:hidden:
:maxdepth: 1

Frame-generation methods <../api/frames>
```
