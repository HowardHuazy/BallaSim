# EVIMO2

Scene controls live under top-level `scene.evimo2.*`. Dataset controls live
under `dataset.evimo2.*`. They apply to EVIMO2 sources; the adapter's ordinary
and low-light branches have different required lighting fields.


## Source and geometry details

(scene-evimo2-sequence)=
### Sequence directory — `scene.evimo2.sequence`

Selects one released camera sequence containing its metadata, motion-capture
tracks, and ground-truth archives. The adapter registers catalog meshes,
retimes pose stamps, and builds the common scene bundle. Use the recording's
camera/category/split directory layout so the camera identity and measured clock
offset can be determined. This is an alternative to `scene.path`.

(scene-evimo2-camera)=
### Primary camera — `scene.evimo2.camera`

Selects the recording's `left_camera`, `right_camera`, or `samsung_mono` source
when resolving the requested sequence. `null` preserves the camera identified
by the sequence path. Camera selection determines calibration and clock facts;
it does not substitute one sensor's intrinsics for another camera's images.

(scene-evimo2-objects-root)=
### Object catalog — `scene.evimo2.objects_root`

Locates the catalog's meshes, marker layouts, and registrations. By default it
is derived from `$EVIMO2_ROOT/.tools/evimo/evimo/objects`, falling back to the
configured machine location. An explicit root permits a local corrected copy
without altering the sequence archive. Imported geometry is cached by source
content, including its dependencies.

(scene-evimo2-raw-root)=
### Raw archive — `scene.evimo2.raw_root`

Locates the original recording metadata used by `clock_reference: raw`.
The default comes from `EVIMO2_RAW_ROOT` or the machine location. Raw timing
needs the recording's `extrinsics.txt` offset; the released NPZ files alone do
not provide it. Use an explicit root when raw recordings are stored elsewhere.

(scene-evimo2-clock-reference)=
### Event timestamp convention — `scene.evimo2.clock_reference`

`npz` matches the released event timestamps after their preprocessing offset;
`raw` matches the original driver clock relative to recording start. The bundle
uses corrected motion-capture time as true time and stores the corresponding
sensor offset. Raw mode additionally needs the original extrinsics offset.
Applying that raw correction again to NPZ timestamps would shift the stream
twice, so select the convention of the real data being compared.

(scene-evimo2-start-trim-ns)=
### Initial span trim — `scene.evimo2.start_trim_ns`

Moves the bundle start after the first rig pose by the stated duration.
It excludes the initial recording interval while leaving the underlying fitted
tracks available. This is distinct from `accuracy.warmup_ns`, which initializes
the sensor and discards generated initialization events. Decreasing the trim
exposes earlier scene time; the run still needs complete supported windows.

(scene-evimo2-objects)=
### Object subset — `scene.evimo2.objects`

Selects EVIMO catalog IDs to include as scene bodies. `null` includes all
objects with available catalog geometry and sufficient track samples.
Excluding an object removes its visible surface, occlusion, and shadows rather
than merely hiding its output label. The tabletop track may still be read to
establish the scene frame and lighting anchor.

(scene-evimo2-extra-cameras)=
### Additional cameras — `scene.evimo2.extra_cameras`

Adds other camera folders from the same recording to the bundle. Each camera
retains its own calibration and sensor-clock offset while sharing geometry and
the corrected motion clock. The run's camera selection then determines which
streams are simulated. Use synchronized folders of the same recording rather
than unrelated sequences.

(scene-evimo2-min-samples)=
### Track-length filter — `scene.evimo2.min_samples`

Skips objects with fewer recorded source-track rows than the threshold before
the expensive build. Raising it removes short or poorly observed tracks;
lowering it admits more objects but does not relax the fitter's requirement for
enough valid poses. Skipped objects are reported in build warnings.

(scene-evimo2-markers)=
### Marker geometry — `scene.evimo2.markers`

Adds sphere meshes at the catalog's motion-capture marker locations.
These contribute visible geometry, occlusion, and shading like other parts of
the body. Disabling them can simplify the reconstruction when marker images are
irrelevant, but also changes the rendered scene. Marker reflectance is a
Lambertian approximation rather than a retroreflection model.

(scene-evimo2-marker-subdivisions)=
### Marker mesh resolution — `scene.evimo2.marker_subdivisions`

Sets the icosphere subdivision level for generated marker spheres. Higher
levels reduce faceting and increase triangle count; they leave the specified
radius and marker placement unchanged. This affects imported geometry only
when markers are enabled.

(scene-evimo2-marker-radius-m)=
### Marker radius — `scene.evimo2.marker_radius_m`

Overrides the catalog sphere radius in meters; `null` keeps the catalog
value. Changing the radius changes silhouette coverage and local occlusion as
well as the marker's rendered area. Use this for known marker dimensions rather
than compensating for brightness errors.

(scene-evimo2-marker-albedo)=
### Marker reflectance — `scene.evimo2.marker_albedo`

Sets the gray Lambertian reflectance assigned to generated markers. It controls
their direct and ambient response within `[0, 1]`, while geometry is controlled
separately by marker radius and subdivisions. This required value is an explicit
approximation to retroreflective material, so it should be stated for the
recording or supplied through a calibrated profile.

(scene-evimo2-untextured-albedo)=
### Untextured-object reflectance — `scene.evimo2.untextured_albedo`

Supplies a gray material for catalog objects whose scans have no usable
texture. It changes surface brightness and contrast on those objects while
preserving their geometry. Textured objects keep their imported materials.
The value is required because an untextured white scan is not a measurement of
the real object's reflectance.

(scene-evimo2-static-tabletop)=
### Static tabletop — `scene.evimo2.static_tabletop`

Permits a tabletop track to be replaced by a fixed robust central pose when
enough samples satisfy the stationary test. This suppresses tracking wander or
an inconsistent minority solution in the scene's reference surface. Disable it
when genuine tabletop motion should remain represented. Ordinary fully static
fits can still become static bodies independently of this special test.

(scene-evimo2-static-tabletop-fraction)=
### Tabletop inlier fraction — `scene.evimo2.static_tabletop_fraction`

Sets the minimum share of gated tabletop samples consistent with its robust
central pose. Consistency is measured with the bounding-sphere displacement
limit `scene.trajectory.hold_max_disp_m`. A larger fraction requires stronger
evidence before fixing the tabletop; a smaller fraction tolerates more divergent
samples. It applies when `static_tabletop` is enabled.

(scene-evimo2-pin-static-px-s)=
### Projected-motion pinning — `scene.evimo2.pin_static_px_s`

Optionally freezes body intervals whose measured image-speed estimate remains
below this limit. The estimate uses the diameter of pose changes over 100 ms,
body extent, focal length, and camera-relative depth; it is not a noisy
single-sample finite difference. Quiet intervals become exact holds, with
transitions affecting the neighboring spline support. Increasing the limit
can suppress real low-amplitude motion; `null` disables this additional pinning.

## Lighting details

(scene-evimo2-lights)=
### Lab-light assembly — `scene.evimo2.lights`

Enables the adapter's point-plus-ambient lighting approximation. Ordinary
recordings use a Fourier-modulated mains group; low-light recordings use a
strobe synchronized to corrected capture timing. Disabling it creates no
adapter lights and makes the associated lighting controls inactive. Their
parameters describe an approximation, not a recovered full room light field.

(scene-evimo2-low-light)=
### Recording light mode — `scene.evimo2.low_light`

Identifies low-light categories ending in `_ll` and selects their strobe branch.
It is a derived sequence fact, so assigning it in a run configuration is
rejected. Use the ordinary or low-light lighting parameters appropriate to the
selected recording.

(scene-evimo2-light-height-m)=
### Point-light position — `scene.evimo2.light_height_m`

Places the point light above its anchor along the tabletop normal. Its intensity
is scaled by height squared, keeping unit direct irradiance at the anchor while
the height changes spatial falloff and shadow geometry. It is therefore not a
simple brightness control. A positive value is required with adapter lighting.

(scene-evimo2-light-color)=
### Light color — `scene.evimo2.light_color`

Sets non-negative linear RGB multipliers shared by the point and ambient
lighting. The camera's spectral weights turn these channels into monochrome
irradiance, so changing color can alter intensity even with unchanged scalar
light settings. Values describe linear light, not display-encoded sRGB.

(scene-evimo2-light-shadows)=
### Light occlusion — `scene.evimo2.light_shadows`

Enables hard shadow rays for the adapter point light. Turning it off removes
light-blocker occlusion from direct shading but preserves object silhouettes
and camera-ray occlusion. The ambient term remains unshadowed. This choice
changes both image contrast and the temporal sampling needed at shadow edges.

(scene-evimo2-ambient)=
### Ordinary ambient term — `scene.evimo2.ambient`

Sets the unshadowed ambient radiance term for ordinary recordings. Surface
shading is `rho * (E / pi + A)`, so `A` is added beside the direct irradiance
divided by pi; it is not a percentage of the point light. Raising it brightens
shadowed surfaces and reduces direct-shadow contrast. `ambient_flickers`
determines whether it shares the mains modulation.

(scene-evimo2-low-light-ambient)=
### Low-light ambient term — `scene.evimo2.low_light_ambient`

Sets constant ambient radiance between and during strobe flashes. Raising it
increases the baseline signal and reduces the flash-to-background contrast.
It uses the constant light group even when the point light is pulsed. This
parameter is required only in the enabled low-light branch.

(scene-evimo2-background-albedo)=
### Unscanned backdrop — `scene.evimo2.background_albedo`

Adds background radiance `background_albedo * ambient` on camera rays missing
all scanned geometry, using the selected branch's ambient value and group.
This approximates an illuminated unscanned room without assigning it depth or
an object ID. Zero leaves those misses dark. Increasing it reduces silhouette
contrast against empty regions without changing scan materials.

(scene-evimo2-ambient-flickers)=
### Ambient flicker — `scene.evimo2.ambient_flickers`

Assigns the ordinary ambient term to the mains group when enabled, or to the
constant group otherwise. The point light remains in the mains group in either
case. This controls whether shaded regions with little direct light still
carry periodic modulation. It does not apply to low-light ambient illumination.

(scene-evimo2-mains-a)=
### Mains amplitudes — `scene.evimo2.mains_a`

Supplies up to eight harmonic amplitudes for the ordinary illumination waveform.
The light multiplier is `1 + sum(a[k] * cos(2*pi*(k+1)*f0*t + phi[k]))`.
The sum of absolute amplitudes must not exceed one, keeping illumination
non-negative. Larger amplitudes increase periodic contrast at the same mean
light level.

(scene-evimo2-mains-f0-hz)=
### Mains fundamental — `scene.evimo2.mains_f0_hz`

Sets the waveform's fundamental frequency; harmonic `k` uses `(k + 1) * f0`.
Changing it changes the physical illumination timing and associated events,
rather than only a viewer diagnostic. Use `dataset.evimo2.mains_hz` to change
the frequency reported to analysis tools without changing the rendered lights.

(scene-evimo2-mains-phi-rad)=
### Mains phases — `scene.evimo2.mains_phi_rad`

Sets one phase in radians for each Fourier amplitude. The list length must
match `mains_a`. Phases shift the periodic illumination relative to the scene's
true clock, changing where threshold crossings fall without changing the
waveform's mean or harmonic frequencies.

(scene-evimo2-strobe-width-s)=
### Strobe duration — `scene.evimo2.strobe_width_s`

Sets the full width at half maximum of each trapezoidal flash. The period comes
from the corrected motion-capture clock, and duty is `width / period`.
The waveform has unit mean: changing width also changes its peak, rather than
holding peak intensity fixed. The width must accommodate the rise and fall
ramps within one period.

(scene-evimo2-strobe-rise-s)=
### Strobe ramps — `scene.evimo2.strobe_rise_s`

Sets equal linear rise and fall durations. The plateau duration is
`strobe_width_s - strobe_rise_s`, because width is measured at half height.
Ramps must be at least 1 ns; plateau duration must be non-negative and both
ramps plus the plateau must fit the period. Longer ramps soften temporal
contrast without changing the specified full width at half maximum.

(scene-evimo2-strobe-floor)=
### Inter-flash level — `scene.evimo2.strobe_floor`

Sets the pulse multiplier between flashes in `[0, 1)`. Unit-mean normalization
gives a peak of `floor + (1 - floor) / duty`. Increasing the floor reduces
flash modulation while preserving the mean multiplier. The independent
low-light ambient term can also provide constant baseline illumination.

(scene-evimo2-strobe-phase-offset-s)=
### Flash alignment — `scene.evimo2.strobe_phase_offset_s`

Offsets the start of each rising ramp from the corrected capture-sample time.
The assembler combines it with the fitted capture-clock intercept and reduces
the result modulo the period. Change it to align flash-induced events with the
recording; it does not alter camera clock offsets or motion timestamps.

## Dataset interpretation details

(dataset-evimo2-mains-hz)=
### Analysis mains frequency — `dataset.evimo2.mains_hz`

States the expected mains frequency for viewer statistics and comparison tools.
It does not drive the simulated illumination. `null` allows the scene's light
groups to supply the value; the versioned EVIMO2 release profile states its
recording-specific frequency explicitly.

(dataset-evimo2-strobe-hz)=
### Analysis strobe frequency — `dataset.evimo2.strobe_hz`

States the expected strobe frequency for periodicity statistics. The simulation
instead derives strobe timing from the retimed capture period. Use a non-null
value when analyzing data without corresponding scene light groups; changing
this field does not modify generated event times.

(dataset-evimo2-gt-lookup-tol-s)=
### Ground-truth lookup — `dataset.evimo2.gt_lookup_tol_s`

Limits the time difference accepted when matching a requested timestamp to a
released ground-truth frame. A larger tolerance permits more distant frames
but can compare a different pose during fast motion. A smaller tolerance leaves
more requests unmatched. `null` derives the tolerance from frame spacing.

(dataset-evimo2-samsung-on-polarity)=
### Samsung polarity — `dataset.evimo2.samsung_on_polarity`

States which stored binary polarity value denotes an ON event. This is an
input-format convention rather than a sensor threshold setting. The schema's
null fallback is 1; the included EVIMO2 release profile explicitly uses 0 for
the Samsung recordings. The viewer uses that release profile when opening
EVIMO2 streams automatically.

## Source and geometry lookup

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `scene.evimo2.sequence` | `path` | required | EVIMO2 camera sequence directory; alternative to `scene.path`. |
| `scene.evimo2.camera` | `string?` | `null` | Camera folder: `left_camera`, `right_camera`, or `samsung_mono`; `null` uses the sequence camera. |
| `scene.evimo2.markers` | `bool` | `true` | Include motion-capture marker spheres in the scene. |
| `scene.evimo2.marker_subdivisions` | `int` | `4` | Icosphere subdivision level for marker meshes. Range: [0, 8]. |
| `scene.evimo2.clock_reference` | `enum` | `"npz"` | Use released NPZ timestamps or original raw-event timestamps. Choices: `npz`, `raw`. |
| `scene.evimo2.start_trim_ns` | `int · ns` | `50000000` | Trim the scene start after the first rig sample. Range: ≥0. |
| `scene.evimo2.objects` | `int[]?` | `null` | Object IDs to include; `null` selects all objects with catalog meshes. |
| `scene.evimo2.extra_cameras` | `string[]` | `[]` | Other camera folders of the same recording to include in the bundle. |
| `scene.evimo2.min_samples` | `int` | `10` | Minimum motion-capture samples required to retain an object. Range: ≥0. |
| `scene.evimo2.static_tabletop` | `bool` | `true` | Make the tabletop static when the stationary-sample fraction exceeds its threshold. |
| `scene.evimo2.static_tabletop_fraction` | `float` | `0.5` | Fraction within `scene.trajectory.hold_max_disp_m` required for a static tabletop. Range: [0, 1]. |
| `scene.evimo2.pin_static_px_s` | `float? · px/s` | `null` | Freeze intervals below this projected motion speed; `null` disables pinning. Range: >0. |
| `scene.evimo2.marker_albedo` | `float` | required | Lambertian marker reflectance approximation; must be specified. Range: [0, 1]. |
| `scene.evimo2.untextured_albedo` | `float` | required | Fallback gray reflectance for untextured catalog objects; must be specified. Range: [0, 1]. |
| `scene.evimo2.marker_radius_m` | `float? · m` | `null` | Marker radius; `null` uses the catalog value. Range: >0. |
| `scene.evimo2.objects_root` | `path` | derived | Object-catalog root; derived from `EVIMO2_ROOT` unless explicitly set. |
| `scene.evimo2.raw_root` | `path` | derived | Raw-archive root; derived from `EVIMO2_RAW_ROOT`. Used for the raw clock reference. |

## Lighting lookup

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `scene.evimo2.lights` | `bool` | `true` | Enable approximate lab lighting; otherwise the adapter adds no lights. |
| `scene.evimo2.low_light` | `bool` | derived | Read-only sequence fact identifying strobe-lit low-light recordings. |
| `scene.evimo2.ambient` | `float` | required | Ambient surface-radiance term. Required for enabled ordinary lighting. Range: ≥0. |
| `scene.evimo2.low_light_ambient` | `float` | required | Constant ambient radiance beside the strobe. Required for enabled low-light lighting. Range: ≥0. |
| `scene.evimo2.background_albedo` | `float` | `0.0` | Backdrop albedo on geometry misses, lit by the selected ambient term; `0` disables it. Range: [0, 1]. |
| `scene.evimo2.light_height_m` | `float · m` | required | Point-light height above its anchor along the tabletop normal. Required while lights are enabled. Range: >0. |
| `scene.evimo2.light_color` | `float[3]` | required | Linear RGB light color. Required while lights are enabled. Range: ≥0. |
| `scene.evimo2.light_shadows` | `bool` | required | Enable light shadows. Required while lights are enabled. |
| `scene.evimo2.ambient_flickers` | `bool` | required | Place ambient illumination in the mains waveform group. Required for enabled ordinary lighting. |
| `scene.evimo2.mains_a` | `float[]` | required | Fourier harmonic amplitudes. Required for enabled ordinary lighting. Range: [-1, 1]. |
| `scene.evimo2.strobe_width_s` | `float · s` | required | Strobe full width at half maximum. Required for enabled low-light lighting. Range: >0. |
| `scene.evimo2.strobe_floor` | `float` | required | Relative light level between strobe flashes. Required for enabled low-light lighting. Range: [0, 1). |
| `scene.evimo2.mains_f0_hz` | `float · Hz` | required | Mains fundamental frequency. Required for enabled ordinary lighting. Range: >0. |
| `scene.evimo2.mains_phi_rad` | `float[] · rad` | required | Fourier harmonic phases, one per amplitude. Required for enabled ordinary lighting. |
| `scene.evimo2.strobe_rise_s` | `float · s` | required | Equal rise and fall durations. Required for enabled low-light lighting. Range: ≥1e-09. |
| `scene.evimo2.strobe_phase_offset_s` | `float · s` | required | Strobe-rise offset from each retimed capture sample. Required for enabled low-light lighting. |

## Dataset interpretation lookup

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `dataset.evimo2.mains_hz` | `float? · Hz` | `null` | Mains frequency for viewer statistics; `null` uses the scene light groups. Range: >0. |
| `dataset.evimo2.strobe_hz` | `float? · Hz` | `null` | Strobe frequency for viewer statistics; `null` uses the scene light groups. Range: >0. |
| `dataset.evimo2.gt_lookup_tol_s` | `float? · s` | `null` | Ground-truth frame lookup tolerance; `null` uses the frame spacing. Range: >0. |
| `dataset.evimo2.samsung_on_polarity` | `int?` | `null` | Raw polarity value for Samsung ON events; `null` uses `1`. The release profile states `0`. Range: [0, 1]. |

## Lighting constraints

With lights enabled, ordinary sequences require the mains fields and
`ambient_flickers`; low-light sequences require the strobe fields. The light
height, color, and shadow switch are required for both. Disabled lighting
leaves these controls inactive.

Mains amplitudes contain at most eight harmonics, have one phase per amplitude,
and satisfy `sum(abs(a)) <= 1`. Strobe rise/fall duration must fit inside the
full-width-at-half-maximum interval and the retimed capture period. Ambient
values are radiance terms in `rho * (E / pi + A)`.

`low_light` is derived from the recording and cannot be set. Object and raw
roots derive from their environment variables, with machine defaults when
unset. Use explicit paths for portable run configurations.

## References

- Burner et al., [EVIMO2](https://arxiv.org/pdf/2205.03467), 2022.
  Dataset geometry, calibration, and motion-capture context.
- [EVIMO2 download and format information](https://better-flow.github.io/evimo/download_evimo_2.html).
  Released camera sequences and raw archives.

```{toctree}
:hidden:
:maxdepth: 1

Reconstruction and dataset methods <../api/evimo2>
```
