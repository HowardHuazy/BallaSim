# Frame generation and readout methods

Use these methods with the [frame and APS settings](../parameters/frames.md).
Reference integration computes an exposure mean; APS conversion turns that
signal into charge and a digital readout.

## Frames

Frame files store exposure timing, waveform means and selected image products.
The high-level writer currently generates reference frames and optional APS
output. Matched/full Blender rendering and ingestion are not available as a
complete workflow.

(frames-file)=
### `FramesFile`

```python
from ballasim.frames import FramesFile

with FramesFile("run/frames/camera.h5") as frames:
    mono = frames.read("mono", slice(0, 4))
```

Open a `ballasim.frames/1` file without loading every image. `attrs` contains
decoded metadata and `names` lists datasets. `read(name, index=None)` returns
one dataset, one row or a slice along its first axis; absent datasets raise
`KeyError`. `close()` releases the file and is called by the context manager.
The format name must match or construction raises `ValueError`.

(read-frames)=
### `read_frames`

`ballasim.frames.read_frames(path, names=None)` returns `(attributes, datasets)`.
Use `names` to load only selected products. This convenience function closes
the file after reading, making it appropriate for small frame sequences;
use `FramesFile` for bounded-memory row access.

(write-frames)=
### `write_frames`

`ballasim.frames.write_frames(path, datasets, attrs)` validates and writes a
frames file through a partial file, returning its `Path`. Required metadata is
`camera`, full `resolution`, `roi_px`, `mode` and `light_groups`. Required timing,
sub-render and waveform arrays are listed in `ballasim.frames.io.REQUIRED`;
`DATASETS` defines accepted names, shapes and dtypes.

Image products use `(N, H, W)` within the ROI, RGB uses `(N, H, W, 3)` and group
products add a light-group dimension. Missing fields, unsafe dtype casts and
inconsistent dimensions raise `ValueError`. The function validates a file
contract; it does not infer missing exposure metadata.

(rewrite-frames)=
### `rewrite_frames`

`ballasim.frames.rewrite_frames(path, datasets, attrs)` replaces or adds the
specified datasets and metadata while preserving other stored data and HDF5
filters. An attribute value of `None` removes that attribute. Required attributes
cannot be removed and `format` cannot be replaced. The update is written and
renamed atomically; return the file `Path`.

(export-reference)=
### `export_reference`

```python
from ballasim.frames.reference import export_reference

export_reference(run_dir, cameras=None, sets=(), *, bundle=None, threads=0)
```

Render exposure-averaged irradiance and return the written `list[Path]`.
`cameras=None` selects cameras with `frames.mode="reference"`; an explicitly
selected camera with mode `off` is enabled for reference output. Selected
`matched` or `full` cameras are refused. Cameras may belong to `run.select` or
`run.frames_select`, but APS output requires a recorded event-camera scale.

`sets` contains full `KEY=VALUE` frame overrides. The function rebuilds the
recorded layers, rejects changed profiles or scene content, and writes
`frames/resolved.json`. `threads` controls APS workers; nonpositive values use
the hardware default. The reference renderer uses the run's recorded renderer
values. Supported products and accuracy settings are described under
[Frames and APS](../parameters/frames.md).

(aps)=
### `aps`

```python
from ballasim.frames.aps import aps

aps(run_dir, camera, sets=(), *, bundle=None, threads=0)
```

Recompute APS output from a frames file's stored `mono` and exposure timing.
Return the updated file `Path`. This changes `aps_dn`, APS metadata, warnings
and the frame digest without rendering again. Frame-section overrides are
accepted by the common resolver; only resolved `frames.aps.*` values affect
this operation.

The camera's scale, gain map and glare settings come from the recorded event
run. A frame-only camera lacks that scale and raises `FRM-APS-SCALE`. The
[APS model](../parameters/frames.md#aps-readout) specifies shot noise, fixed
pattern, saturation and quantization. Worker count does not change the result.

(frames-resolve)=
### `frames.resolve.resolve` and `FramesResolution`

```python
from ballasim.frames.resolve import resolve

resolve(run_dir, sets=(), *, bundle=None, implicit=None, doc=None)
```

Resolve frame values from the run's original configuration and profiles, with
frame overrides on top. Return `FramesResolution`, containing the bundle,
recorded renderer values, selected camera records, warnings and run settings.
`.names` lists cameras; `.camera(name)` returns one record; `.values(name)`
returns its plain frame values. `implicit` supplies command-made values below
explicit `sets`; integrations normally leave it unset.

Frame-only cameras are resolved without requiring event-sensor values.
Changed profile hashes or inconsistent recorded frame values are refused.
`_frames.FramesError` is a `ValueError` with a `code` attribute; examples include
`FRM-CONFIG-CHANGED`, `FRM-CAMERA`, `FRM-VALUE` and `FRM-SET-SCOPE`.

## Native frame evaluation

`ballasim._frames` exposes schedules and array operations without file I/O.
Its inputs still use the recorded scene and camera conventions. Native schedules
operate on the true clock; clock mapping is a separate operation.

(frame-schedule)=
### `_frames.Schedule`

```python
Schedule.uniform(height, exposure_ns, t0_ns, period_ns, count, support, *,
                 row_dt_ns=0, shutter="global", readout="top_down")
Schedule.blender(height, exposure_ns, frame_ns, position, support, *,
                 row_dt_ns=0, shutter="global", readout="top_down")
```

Construct a row-exposure schedule within `(begin_ns, end_ns)` support. Uniform
anchors are `t0_ns + k*period_ns`; `count=None` includes all supported frames.
The Blender form takes an `int64` frame-time array and `START`, `CENTER` or
`END` anchoring. Both drop unsupported complete exposures rather than clipping
individual rows. Rolling timing always refers to full-image rows.

#### Schedule queries

`times()` returns kept-frame `soe`, `eoe`, `mid` and original `candidate` indices.
`soe(k, row)` and `eoe(k, row)` return a row's true-clock exposure bounds and
raise `IndexError` outside the schedule. `shifted(shift_ns)` returns a schedule
with one time shift per kept frame. Properties `size`, `dropped`, `warnings`,
`exposure_ns` and `readout_span_ns` describe the resulting timeline.

(exposure-reference)=
### `_frames.ExposureReference`

```python
ExposureReference(bundle, values, camera, roi=None, waveforms=None)
```

Construct an exposure renderer from a bundle and resolved plain values.
`camera` is a bundle index or name. `roi=None` uses the full image; optional
`waveforms` replaces the bundle's light-group waveforms. Properties `roi`,
`num_groups` and `backend` report the constructed renderer.

The model computes a temporal mean of pixel-integrated irradiance weighted by
exact waveform means. It retains the renderer's direct-light and scene
approximations; it does not introduce Blender global illumination.

#### `ExposureReference.frame`

```python
frame(schedule, k, *, h_px=0.25, probe_tol=0.005, k_max=256,
      k_fixed=0, n_sub=1, reuse_budget_mb=1024.0)
```

Evaluate frame `k`. Adaptive sampling uses `h_px`, `probe_tol` and `k_max` as
specified by the [reference integration settings](../parameters/frames.md#reference-integration).
`k_fixed>0` instead requests a fixed sample count and disables the probe.
`n_sub` divides the exposure for the waveform–motion factorization diagnostic;
it is in `[1, 64]`. `reuse_budget_mb` limits reusable sample storage and changes
rendering work, not the output frame.

| Result | Type / shape | Meaning |
| --- | --- | --- |
| `mono`, `per_group` | `float32 (H, W)`, `(H, W, G)` | Exposure-averaged signal. |
| `m_mean` | `float64 (H, G)` | Exact waveform mean per ROI row and group. |
| `depth`, `instance` | `(H, W)` | Center-ray geometry at frame midpoint. |
| `k_used`, `k_mask`, `mask_px`, `rendered` | Integers | All-pixel/probe levels, mask size and rendered sample count. |
| `probe_p999`, `probe_max`, `fact_p99` | Scalars | Estimated temporal error and factorization diagnostic. |
| `warnings` | `(code, message)` pairs | Caps or unsupported accuracy conditions. |

The probe estimates error from denser samples; success is not a universal
analytic bound on an arbitrary visibility signal.

#### `ExposureReference.motion`

`motion(schedule, k)` returns `d_max_px`, `any_motion`, `light_motion`,
`light_rate_per_s` and trajectory `breakpoints` for the exposure. Use these facts
to inspect temporal complexity or construct an external render plan; they do
not render the frame.

#### `ExposureReference.visibility`

`visibility(t_ns)` returns `float32 (H, W, G)` ratios of shadowed irradiance to a
copy with shadow casting disabled, with denominator floor `1e-12`. It is an
instantaneous radiance ratio for comparison, distinct from an exposure mean or
a binary geometric visibility mask.

(waveform-mean)=
### `_frames.waveform_mean`

`waveform_mean(waveform, t_a, t_b)` returns exact `float64` means for equal-length
`int64` endpoint arrays. It accepts a light-group waveform dictionary and
returns its instantaneous value at `t_a` when `t_b <= t_a`. Use this primitive
to integrate flicker without increasing geometry sample density.

(frame-sensor-time)=
### `_frames.sensor_time`

`sensor_time(t_ns, clock_offset_ns=0, clock_drift_ppm=0.0)` maps an `int64` true-time
array to `t + offset + round(drift*1e-6*t)`. It adds neither readout delay nor
counter quantization. This is the clock used for frame observations and dense
query stamps, not the event readout operation.

(aps-params)=
### `_frames.aps_params`

`aps_params(values, run_sensor_seed)` resolves APS values into device parameters.
`values=None` uses defaults. Null gain is derived from bit depth and full well;
a null APS seed uses the run sensor seed. Return the complete parameters used
by `aps_frames`; invalid values raise `FRM-VALUE`.

(aps-frames)=
### `_frames.aps_frames`

```python
aps_frames(mono, frame_soe_ns, *, run_sensor_seed, camera, resolution, roi,
           exposure_ns, illuminance_scale, values=None, row_dt_ns=0,
           shutter="global", readout="top_down", veiling_glare=0.0,
           gain=None, threads=0)
```

Convert `float32 mono (N, H, W)` into APS DN using `int64 frame_soe_ns (N,)`
true-clock anchors. `resolution=(W_full, H_full)` and end-exclusive `roi`
retain full-frame row timing and random keys. `illuminance_scale` is the event
camera's fA per renderer unit; `gain` is an optional ROI `(H, W)` pixel-gain map.

Return `dn: uint16 (N, H, W)`, per-frame `veil_mean`, resolved `params`,
`normal_fallbacks` and warnings. `mono` is converted to electrons using exposure;
Poisson shot noise remains even when Gaussian noise parameters are zero.
A missing photocurrent scale is refused. Exposure compatibility warnings do not
change the nanosecond timing used for the model.

(vignetting-helpers)=
### `_frames.vignetting_on` and `_frames.vignetting_gain`

`vignetting_on(vignetting, photometry_mode)` returns the explicit Boolean, or
`True` for null under physical/reference photometry.
`vignetting_gain(bundle, values, camera, roi=None, law="cos4",
relative_illumination=())` returns the engine's `float32 (H, W)` static gains.
Use the recorded camera's `cos4` or `jacobian` law and relative-illumination
coefficients to reproduce its photometry; a new gain law changes the sensor.

(frame-digests)=
### `_frames.frames_digest_base` and `_frames.frames_digest`

`frames_digest_base(format, bundle_digest, camera_digest, values="",
plan_digest="", manifest_sha256="")` hashes scene, camera and non-APS frame
inputs. `frames_digest(base, params)` adds resolved APS parameters, or the
absence of APS when `params=None`. The two-level record lets APS updates change
the product digest without pretending the base image was rendered again.

(native-resolve-frames)=
### `_frames.resolve_frames`

`resolve_frames(inputs, bundle, sets=None, check=True)` resolves plain
`RunSpec.inputs` against `config.bundle_info(...)`, with restricted frame edits.
Return camera value records, frame run settings and warnings. `check=False`
collects the recorded event-camera rows without frame-workflow refusals; it is
for run-record reconstruction. Use the Python resolver for profile-hash and
scene-file checks.
