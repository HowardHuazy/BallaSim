# SensorModel API

`ballasim._sensor` converts per-pixel irradiance histories into events. It
exposes the same pixel response used by the window engine without rendering a
scene. For complete simulations use `ballasim.run`; for response semantics see
[Event sensors](../../guide/sensors.md).

## `SensorParams`

```python
from ballasim import _sensor

params = _sensor.SensorParams.preset("prophesee_gen3")
params.theta_on = 0.25
params.validate()
```

Holds the model's physical and numerical values. Attributes use native field
names, such as `theta_on`, `illuminance_scale`, `eps_L` and `label_share_cut`,
rather than full run-configuration keys. Fields use SI units except where their
names specify another unit. See the [sensor settings](../parameters/sensor.md)
for meanings and the complete descriptor for field-to-key mappings.

`SensorParams()` has no usable family prior: required floating-point fields are
unset and validation fails until they are populated. Start from a preset or
resolved values. Renderer/resolver settings such as `sensor.rgb_weights`,
`aperture_px`, `qe`, `kappa` and `nir_fraction` are not all direct `SensorParams`
attributes; rendering and photometry apply them before sensor integration.

### `SensorParams.preset(name)`

Returns a complete parameter object for a built-in
[family](../parameters/families.md). `_sensor.preset_names()` lists accepted
names. The low-level preset uses neutral `illuminance_scale=1`; choose a
photocurrent scale appropriate to the supplied irradiance before modeling.

### `SensorParams.of_values(values, camera, photometry=None)`

Builds validated parameters from `Resolved.values()` for the named camera.
Run-level numerical values are applied along with camera-level sensor values.
Without `photometry`, this accepts only relative photometry without vignetting.

For physical/reference photometry or enabled vignetting, pass
`photometry=(scale, gain)` from the corresponding `CameraEngine`: its
`illuminance_scale` after `start()` and its `gain_map()` (`None` when unvignetted).
Pass the same pair to `SensorModel.set_photometry` before initialization.
`of_values` stores the scalar, but the gain map belongs to the model.

### `SensorParams.validate()`

Checks field ranges and cross-parameter rules and raises `ValueError` for an
invalid combination. Setting an attribute checks its descriptor; call
`validate()` after a group of edits to check relationships such as refractory
limits, pole separation and readout causality.

## `SensorModel`

```python
SensorModel(params, width, height, seed, light_groups, camera,
            num_threads=0, precision=None, overrides=None,
            device="cpu", roi=None)
```

Constructs one camera's pixel array. `params` is a preset name or `SensorParams`;
`overrides` maps native parameter-field names to values. `light_groups` is a
list of constant, Fourier or pulse waveform dictionaries, or records containing
a `waveform` dictionary. Irradiance samples have one column per group, before
its time-dependent multiplier is applied.

`camera` is the camera's bundle index and, with `seed` and full-frame pixel ID,
keys random draws. Distinct cameras need distinct indices. CPU execution uses
`fp64` by default and accepts `fp32`; `device="cuda"` requires `fp32`.
`num_threads=0` selects hardware concurrency for CPU work.
`_sensor.cuda_available()` reports CUDA sensor and visible-device availability.

`roi=(x0, y0, x1, y1)` constructs a rectangular subset with exclusive upper
bounds. Pixel IDs, event coordinates and random keys remain full-frame values;
initial frames and stationary probes still require the full frame. Model calls
from Python threads are serialized. Do not mutate sample arrays during a call.

## Initialization and input

(sensor-init)=
### `init(t0_ns, irradiance0)`

Initializes analog filters to steady state at `t0_ns`, with comparator state
selected by `params.initial_state`. `irradiance0` is finite `float32 (H*W, G)`
in renderer units, including pixels outside an ROI. Values below zero or above
`model.irr_max` are clamped and counted; nonfinite values are rejected.

Initialization does not replay scene motion or discard a warm-up interval.
The run engine performs those operations separately. Reinitialization replaces
the previous temporal state and establishes the current model time.

(sensor-advance)=
### `advance(pixel_ids, t_ns, irradiance, t_end_ns, counts=None, offsets=None)`

Advances every pixel to `t_end_ns`. Listed pixels follow their sample history;
unlisted pixels hold their last irradiance. Pixel IDs must be unique, in range
and within the model's ROI. A pixel ID is `y*W+x`.

| Argument | Padded form | Ragged form |
| --- | --- | --- |
| `pixel_ids` | Exact `uint32 (N,)` | Exact `uint32 (N,)` |
| `t_ns` | Exact `int64 (N, M)` | Exact `int64 (S,)` |
| `irradiance` | `float32 (N, M, G)` | `float32 (S, G)` |
| `counts` | Optional exact `int32 (N,)` valid row lengths | `None` |
| `offsets` | `None` | Exact `int64 (N+1,)`, from zero to `S` |

Integer input arrays must be C-contiguous and are not cast implicitly.
Irradiance may be converted to contiguous `float32`. Times increase strictly
within each valid row, begin no earlier than `time_ns` and do not exceed
`t_end_ns`. Ragged offsets are nondecreasing. The last sample is held to the
advance end; irradiance clamping follows `init`. Each group's irradiance is
linearly interpolated between samples, starting from its carried value, then
multiplied by the continuous group waveform.

The result is a dictionary of arrays `(E,)`, ordered by `(t_cross_ns, pixel)`:

| Fields | Dtype | Meaning |
| --- | --- | --- |
| `x`, `y` | `uint16` | Full-frame coordinates |
| `p` | `uint8` | Polarity: `1` ON, `0` OFF |
| `t_cross_ns`, `t_reported_ns` | `int64` | True crossing time and delayed, quantized sensor time |
| `label` | `uint8` | Raw label: `0` signal, `1` noise, `2` leak, `3` hot |
| `flags` | `uint16` | Pixel-history and approximation flags |
| `ds`, `dn`, `dl` | `float32` | Signal, noise and leak increments retained for relabeling |
| `l_first` | `float32` | Leak since initialization for a first event; zero otherwise |

These raw labels differ from the saved event file's five-class convention.
Readout can merge distinct crossing times into timestamp ties; use `rederive`
to obtain final reported-time and coordinate order. A partially failed advance
invalidates temporal state; reinitialize or restore a saved state before
continuing.

## Stationary state

### `stationary_probe(t0_ns, irradiance0)`

Returns a byte blob describing a stationary comparator-start probe, without
changing the model. It uses the full `float32 (H*W, G)` initial frame and the
configured representative pixels and burn-in history. This samples a held-input
history; it is not a stored rendering or a steady-state assertion for moving
illumination. `params.initial_state` must be `stationary`; noiseless models
produce an empty probe payload and retain the phase law.

### `stationary_init(t0_ns, irradiance0, probe)`

Initializes using a compatible probe. Inputs must match the probe's sensor,
frame and initialization conditions; readout and label-cut differences are
permitted. Incompatible blobs raise `ValueError`. A matching cached probe
reproduces the ordinary stationary initialization.

Properties `stationary_ms`, `stationary_probes`, `stationary_history_ns` and
`stationary_phase_kept` report initialization work, selected probe count,
history duration and pixels that retained the phase law because no compatible
probe snapshot fit inside their levels.

## Photometry and full-frame context

### `set_photometry(illuminance_scale, gain=None)`

Sets the photocurrent scale in fA per renderer unit and a positive finite gain
per full-frame pixel. `gain` may have any shape with `H*W` elements; `None`
selects unit gain. Set it before `init`; the next initialization applies it to
the pixel states. This interface carries lens gain and a solved reference scale
that are not fully represented by parameter values alone.

### `gain_map()`

Returns static pixel gains as `float32 (H, W)`, or the ROI's height and width
for an ROI model. Gains default to one. This is a spatial photocurrent factor,
not a temporal response or a threshold-mismatch map.

### `set_frame_means(means)`

Supplies full-frame context for ROI veiling glare as
`float64 (calls+1, G)`: row zero belongs to initialization and row `k+1` to the
end of advance `k`. A crop alone cannot recover irradiance outside its ROI, so
replay must preserve these means to reproduce glare.

### `set_track_frame_means(on)`

Enables full-frame mean tracking even when glare is disabled. The current
`glare_frame_mean` property reports one mean per group. Each advance updates
means from the input values held at its end; replay schedules must preserve
the corresponding full-frame context.

## State and diagnostics

### `save_state()`, `load_state(state)`, `state_bytes()`

`save_state()` returns temporal state as bytes; `state_bytes()` reports its
size. `load_state(state)` restores a state created with compatible model
parameters and geometry. The state header is checked, not every payload value;
use blobs produced by the model. Restoring a compatible state also recovers
from an advance that failed partway.

### `sensor_maps()`

Returns frozen per-pixel maps with shape `(H, W)`, or ROI shape:
`theta_on`, `theta_off`, `leak`, `dark_current_fA`, `phi0`, `hot`,
`hot_target_hz` and `tau_hp`. Most numerical maps use `float32`;
`hot_target_hz` uses `float64` and `hot` uses `uint8` (`0` ordinary, `1` leak
type, `2` threshold type). `leak` is log-reference drift per second, rather
than an observed event rate.

### `hpf_states()`

Returns `(memory, noise_memory)` as `float64` arrays, one entry per modeled
pixel. They describe the high-pass reference memory and its temporal-noise
component. Arrays are empty when the filter kernel is inactive; `hpf_kernel`
reports whether it is active.

### `stats()`, `reset_stats()`

Read or clear work counters since construction or the last reset. Counters
include numerical substeps, crossing-search work, clamped inputs and CUDA
launch diagnostics where applicable. Resetting counters does not reset sensor
state or random-number history.

Other read-only properties are `params`, `width`, `height`, `groups`, `roi`,
`time_ns`, `num_threads`, `device`, `precision`, `irr_max`, `median_dark_fA` and
`readout_phase_ns`. `median_dark_fA` includes the configured temperature factor;
`readout_phase_ns` is the camera's counter-grid phase.

## Readout and label recomputation

### `_sensor.rederive`

```python
rederive(events, width, height, seed, camera, quantum_ns, delay_ns,
         clock_offset_ns, clock_drift_ppm, label_cut, relabel=True)
```

Recomputes reported times, labels and final output order from raw `advance`
events, leaving crossings and analog history unchanged. The event dictionary
must contain `x`, `y`, `p`, `t_cross_ns`, `label`, `flags`, `ds`, `dn`, `dl` and
`l_first` with the exact contiguous dtypes listed above.

Returns `order: uint32 (E,)`, plus `t_reported_ns: int64 (E,)` and
`label_s1: uint8 (E,)` in input order. Apply `order` to every column to obtain
reported-time output order. The new counter phase is drawn from the seed,
camera and quantum. This does not apply event-rate limiting.

Relabeling uses stored `float32` increments. For an `fp64` session, use
`relabel=False` to preserve the sensor's original decisions; its increments
were rounded on export to the raw array. Ordinary HDF5 event files do not store
these increment columns and cannot serve as complete `rederive` input.

## Pixel-response routines

### `_sensor.cascade_trace`

```python
cascade_trace(t_ns, current_fA, query_ns, k=1e16, tau_o=25e-6,
              tau_sf=100e-6, eps_L=0.002, precision="fp64")
```

Evaluates the deterministic cascade from piecewise-linear current in fA,
starting in steady state. Input and query times are exact `int64` arrays;
the result is `(s3, substeps)`. It does not generate threshold events or temporal
noise. `precision` selects `fp64` or `fp32` arithmetic.

### `z_step(z0, I0, I1, h, k, precision="fp64")`

Evaluates the first-stage closed form for linear current `I0` to `I1` across
duration `h`. Array inputs are one-dimensional `float64`; current is in fA,
`h` in seconds and this routine's `k` in `(fA s)^-1`, unlike `SensorParams.k`
in `(A s)^-1`. Returns one state value per input row.

### `special(name, x, precision="fp64")`

Evaluates the kernel implementations of `exp`, `log`, `expm1`, `log1p`, `erfcx`
or `dawson` on a one-dimensional `float64` array. `erfcx` requires nonnegative
arguments. These hooks support numerical integration comparisons; they are not
sensor configuration controls.
