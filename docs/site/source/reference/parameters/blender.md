# Blender conversion and motion

Scene scope: `scene.blender.*`. These controls apply only to Blender exchange
sources. Unsupported features that have no permitted approximation remain
errors under either policy; inspect the conversion report before running.


## Parameter details

(scene-blender-policy)=
### Conversion policy — `scene.blender.policy`

Controls whether supported approximations are accepted when Blender features
cannot be represented exactly. `strict` rejects cases without an acceptable
conversion; `lenient` permits the converter's defined approximations and reports
them. It does not enable arbitrary shaders, deformation, or missing render
features. Inspect each report's code, severity, and stated bound.

(scene-blender-lm-per-w)=
### Light-power conversion — `scene.blender.lm_per_w`

Converts Blender light power into photometric intensity or radiance before
camera photocurrent is computed. It represents luminous efficacy of the light
spectrum, rather than a generic exposure multiplier. Physical photometry
requires an explicit positive value; 683 lm/W is the monochromatic 555 nm
reference, not a default for white lighting. Other modes use a factor of one
when the value is `null`.

(scene-blender-world-as-ambient)=
### World illumination — `scene.blender.world_as_ambient`

Also applies the converted constant world color to surface illumination as an
unshadowed ambient term. By default, world radiance supplies the background
without this extra surface-lighting approximation. Enabling it brightens
surfaces independently of geometric occlusion and can over-light contact
regions. It does not add environment-map sampling or indirect illumination.

(scene-blender-crop-radius-m)=
### Static-geometry crop — `scene.blender.crop_radius_m`

Clips static geometry to a ball around the chosen world origin before bundle
assembly. This admits a local region of a larger Blender scene within the 16 m
coordinate-reach limit. Smaller radii reduce geometry and memory, but may remove
visible surfaces or shadow blockers. `null` preserves the source geometry and
rejects static geometry beyond the supported reach.

(scene-blender-fps-base-snap)=
### Frame-rate rationalization — `scene.blender.fps_base_snap`

Replaces Blender's stored frame-rate base with a nearby rational whose
denominator is 1, 1000, or 1001. This can recover intended video timing from a
rounded floating-point value. Leave it disabled when the exact stored base is
intentional. Changing it changes the frame-to-nanosecond map, so previously
sampled motion must be regenerated.

(scene-blender-max-triangles)=
### Geometry budget — `scene.blender.max_triangles`

Limits the converted triangle count after instancing. It rejects oversized
scenes before they enter the renderer; it does not simplify meshes or reduce
their resolution. Increase it only when the available scene-memory budget can
accommodate the additional geometry. Cropping or reducing source geometry can
lower the count without changing this limit.

(scene-blender-rigidity-tol-m)=
### Rigid-mesh check — `scene.blender.rigidity_tol_m`

Bounds vertex displacement measured in the object's local frame during motion
sampling. Rigid transforms leave these coordinates unchanged; larger departures
identify deformation that a rigid-body trajectory cannot reproduce. Increasing
the tolerance accepts more geometric discrepancy rather than adding deformable
simulation. The check uses metric units after Blender scale conversion.

(scene-blender-calib-tol-px)=
### Calibration mismatch — `scene.blender.calib_tol_px`

Sets the image-corner displacement that triggers a warning when a supplied
calibration disagrees with Blender's pinhole intrinsics. The imported
calibration remains authoritative. Lower values expose smaller differences;
changing the warning threshold does not alter the camera model or fit the
calibration.

(scene-blender-fisheye-fit-tol-px)=
### Fisheye approximation — `scene.blender.fisheye_fit_tol_px`

Limits the pixel error when a Blender lens-polynomial fisheye is represented by
the supported Kannala–Brandt model. A fit exceeding the tolerance is rejected.
Increasing the value accepts a less accurate image mapping; it does not expand
the converter's accepted camera families or bypass injectivity checks.

(scene-blender-motion-tol-px)=
### Motion accuracy — `scene.blender.motion_tol_px`

Sets the image-space tolerance used for adaptive sampling and trajectory
validation. Translation and rotation are converted to pixel displacement using
camera angular gain, object extent, and conservative scene distances. Smaller
values require more samples or finer knots. Blender's time and transform
precision impose a floor, so refinement cannot recover detail absent from the
evaluated source.

(scene-blender-cache-samples-per-frame)=
### Transform-cache rate — `scene.blender.cache_samples_per_frame`

States the samples per authored frame for transform caches evaluated in
frame-state mode, such as Alembic or USD motion. Match the cache's actual rate
so the importer samples its available states and detects changes between them.
Increasing this value does not create temporal information missing from the
cache. It applies to cached motion rather than ordinary continuously evaluated
animation.

(scene-blender-detect-k-sigma)=
### Breakpoint significance — `scene.blender.detect_k_sigma`

Sets the standard-error multiplier for classifying value or slope differences
around candidate breakpoints. The detector compares one-sided local fits in
pixel-scaled pose coordinates. Lower values admit less significant changes;
higher values suppress uncertain candidates. A confirmed pose step remains
unsupported even when the threshold is relaxed.

(scene-blender-detect-burst-samples)=
### Localization samples — `scene.blender.detect_burst_samples`

Requests this many samples on each side of a candidate during one localization
burst. At least three are needed for the one-sided quadratic fits. Increasing
the count provides more local fit evidence while increasing Blender evaluation
work. It does not change the smallest temporal window allowed below.

(scene-blender-detect-burst-rounds)=
### Localization rounds — `scene.blender.detect_burst_rounds`

Limits repeated localization bursts around a candidate. Each round narrows the
sampling window by a factor of four to refine a slope-change time. More rounds
can improve localization when the source time grid permits it; zero disables
these additional bursts. The minimum burst window still bounds refinement.

(scene-blender-detect-burst-floor-ns)=
### Localization time floor — `scene.blender.detect_burst_floor_ns`

Sets the smallest window used for burst refinement. A smaller value focuses
additional samples more tightly around a breakpoint, but cannot overcome
Blender's representable frame-time spacing. A larger floor avoids sampling
scales finer than the source warrants and leaves a wider localization interval.

(scene-blender-detect-merge-ns)=
### Breakpoint merging — `scene.blender.detect_merge_ns`

Combines candidates whose times lie closer than this interval. When candidates
conflict, a step takes precedence over a kink; otherwise the larger jump or
slope change wins. Larger values reduce duplicate detections but may combine
nearby distinct changes. Zero disables proximity-based merging.

(scene-blender-impact-k-sigma)=
### Frame-state impact significance — `scene.blender.impact_k_sigma`

Detects unusually large second pose differences in frame-state motion using a
robust scale estimate. These candidates identify abrupt changes such as impacts
for piecewise fitting. Lower values increase sensitivity to smaller changes;
higher values suppress detections caused by ordinary variation. The absolute
displacement floor is applied alongside this relative threshold.

(scene-blender-impact-floor-m)=
### Frame-state impact floor — `scene.blender.impact_floor_m`

Sets the minimum second-difference displacement eligible as an impact. It keeps
rounding-scale deviations from becoming significant when the robust scale is
very small. Raising it ignores small discontinuities; lowering it exposes
smaller candidates to the significance test. It does not cap physical
acceleration.

(scene-blender-fit-lambda-rel)=
### Motion regularization — `scene.blender.fit_lambda_rel`

Scales the small roughness penalty in each piece's weighted least-squares fit.
The actual weight is normalized by sample weights, sample count, and control
count separately for position and rotation. Larger values favor smoother
controls; smaller values follow sampled Blender poses more closely. Final
acceptance is still determined by `motion_tol_px` and validation probes.

(scene-blender-validation-probes)=
### Independent fit probes — `scene.blender.validation_probes`

Adds seeded pose evaluations inside each fitted piece beyond the samples used
for fitting and the required breakpoint probes. These checks can expose error
between fit samples and trigger knot refinement. More probes increase coverage
and evaluation cost. Zero omits the additional probes but retains breakpoint
validation.

(scene-blender-max-halvings)=
### Fit refinement limit — `scene.blender.max_halvings`

Limits knot-spacing halvings when a piece fails its error or rotation-increment
checks. More halvings allow additional controls to represent difficult motion,
at increased fitting and storage cost. A piece still failing after the limit
raises a motion-fit refusal. Increasing the limit cannot make deformation or
pose steps representable by the supported rigid-motion model.

## Conversion lookup

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `scene.blender.policy` | `enum` | `"strict"` | `strict` rejects unsupported features; `lenient` permits supported approximations with diagnostics. Choices: `strict`, `lenient`. |
| `scene.blender.lm_per_w` | `float? · lm/W` | `null` | Luminous efficacy for Blender light power; required by physical photometry. Otherwise `null` uses 1. Range: >0. |
| `scene.blender.world_as_ambient` | `bool` | `false` | Also apply world color as unshadowed ambient illumination. |
| `scene.blender.crop_radius_m` | `float? · m` | `null` | Crop static geometry around the scene origin; `null` rejects geometry beyond the 16 m reach limit. Range: (0, 16]. |
| `scene.blender.fps_base_snap` | `bool` | `false` | Snap the frame-rate base to a rational with denominator 1, 1000, or 1001. |
| `scene.blender.max_triangles` | `int` | `20000000` | Triangle budget after instancing. Range: ≥1. |
| `scene.blender.rigidity_tol_m` | `float · m` | `1e-05` | Object-frame vertex-motion limit before deformation is rejected. Range: >0. |
| `scene.blender.calib_tol_px` | `float · px` | `0.01` | Image-corner tolerance for camera-calibration mismatch warnings. Range: >0. |
| `scene.blender.fisheye_fit_tol_px` | `float · px` | `0.05` | Maximum image error when fitting a polynomial fisheye to Kannala–Brandt. Range: >0. |

## Motion settings lookup

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `scene.blender.motion_tol_px` | `float · px` | `0.01` | Image-space tolerance for fitted motion. Range: >0. |
| `scene.blender.cache_samples_per_frame` | `int` | `1` | Transform-cache sampling rate in frame-state mode; use the source cache's rate. Range: ≥1. |
| `scene.blender.detect_k_sigma` | `float` | `6.0` | Standard-error threshold for candidate value or slope discontinuities. Range: >0. |
| `scene.blender.detect_burst_samples` | `int` | `8` | Samples on each side of a breakpoint-localization burst. Range: ≥3. |
| `scene.blender.detect_burst_rounds` | `int` | `4` | Localization bursts per candidate; each reduces the window by four. Range: ≥0. |
| `scene.blender.detect_burst_floor_ns` | `int · ns` | `64000` | Minimum localization-burst window. Range: ≥1. |
| `scene.blender.detect_merge_ns` | `int · ns` | `100000` | Merge breakpoint candidates closer than this interval. Range: ≥0. |
| `scene.blender.impact_k_sigma` | `float` | `30.0` | Robust acceleration-difference threshold for frame-state impact detection. Range: >0. |
| `scene.blender.impact_floor_m` | `float · m` | `1e-06` | Minimum second-difference displacement counted as an impact. Range: ≥0. |
| `scene.blender.fit_lambda_rel` | `float` | `1e-10` | Relative jerk regularization for motion fitting. Range: ≥0. |
| `scene.blender.validation_probes` | `int` | `32` | Seeded validation probes per fitted piece, in addition to breakpoint probes. Range: ≥0. |
| `scene.blender.max_halvings` | `int` | `5` | Maximum knot-spacing refinements before fit refusal. Range: ≥0. |

## Applicability

Motion controls change the sampled/fitted trajectory and invalidate the exchange's
motion cache. Animated scenes require a compatible motion session from the
Blender add-on. Static scenes can convert without one. Physical photometry
requires a positive `lm_per_w`; specify the luminous efficacy appropriate to the
light spectrum.

```{toctree}
:hidden:
:maxdepth: 1

Exchange and calibration methods <../api/blender>
```
