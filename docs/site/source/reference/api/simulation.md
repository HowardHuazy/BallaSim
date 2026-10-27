# Run and window methods

Use `ballasim.run` for a complete recorded simulation. The native window engine
and host planning methods support custom integrations that need to control
advancement explicitly. Rendering and pixel-model methods are documented with
[Scenes & assets](../../guide/scenes.md) and [Event sensors](../../guide/sensors.md).

## Complete runs

### `ballasim.run`

```python
ballasim.run(spec, *, bundle=None, report=None, on_window=None, cancel=None)
```

Load a configuration path or use an existing `RunSpec`, construct its scene and
engine, and write one event stream per selected camera. Return a `RunSummary`.
An in-memory `bundle` replaces the configured scene source. Resolution and engine
construction precede output-directory creation; per-camera initialization,
reference photometry, warm-up, and advancement follow it.

`report(resolved)` observes final configuration resolution before engine
construction. `on_window(camera, window, windows, events, t_ns)` observes completed
windows on the calling thread: `window` is the number completed, `events` the
cumulative crop-kept crossing count before rate limiting and final tail release,
and `t_ns` the true-clock end reached. This count can differ from written event
rows. These callbacks report
progress; they do not supply new physics or change a window's samples.

Pass `_engine.CancelToken` as `cancel` for cooperative interruption. Invalid
configuration raises `ConfigError`; occupied or locked run directories are
refused. Cancellation raises `_engine.Cancelled`, records the outcome when a
run exists, and removes the active partial event file. Completed camera files
remain. Enabled IMUs are generated after the event run completes, with separate
failure records.

### `RunSummary`

`ballasim.engine.RunSummary` records `run_dir`, `backend`, `t_start_ns`,
`t_end_ns`, `cameras`, and `wall_s`. `simulated_s` is the aligned span duration,
not wall-clock processing time. Use the recorded interval when calculating
throughput; requested start/end values may have been rounded inward.

### `CameraSummary`

Each camera summary records `name`, bundle `index`, file `path`, event and window
counts, pixel samples, rays, render calls, clamps, timings, and counters. Sample
and ray counts describe different costs: one pixel-time sample can trace many
spatial rays and shadow rays. They should not be interpreted as event counts or
fixed-rate image frames.

## Native window engine

### `_engine.Engine`

```python
engine = _engine.Engine(bundle, resolved.values())
camera = engine.camera(index)
```

Construct the renderer and sensors from a bundle and fully resolved plain values.
`camera(index)` uses a bundle camera index and returns `CameraEngine`; it does not
create a new camera. Cameras share the engine's renderer and work buffers, and
calls are serialized. `backend` reports `optix` or `embree`; `budget_bytes`
reports the working-memory budget, independently of accuracy settings.

`_engine.has_device` reports compiled device support. `_engine.device_available()`
also checks the visible CUDA device and usable OptiX runtime. Use the latter to
reason about automatic backend selection.

### `CameraEngine.start`

```python
camera.start(t_s, warmup_ns, motion=False)
```

Render the starting input, solve reference photometry when selected, initialize
the sensor, and advance through warm-up without releasing events. Motion warm-up
uses covered complete windows; missing history is completed statically according
to the {ref}`warm-up rules <accuracy-warmup-ns>`.
Call this before advancing windows.

The result records warm-up and probe cost, motion windows, rendered work, held
geometry, solved `illuminance_scale`, and optional full-frame glare means. Baked
scenes also report proxy blockage and split-geometry diagnostics. These are
initialization facts, rather than additional event streams.

### `CameraEngine.step`

```python
columns, counters = camera.step(t_w)
```

Plan, sample, sense, and annotate `[t_w, t_w + WINDOW_NS)`. Process contiguous
windows beginning at the sensor's current time. The returned columns contain
only events final in reported-time output order; delayed events may be held for
a later step or `flush`. With ground truth enabled, columns include instance,
semantic, depth, flow, label, physical crossing time, and flags.

Counters distinguish planning, rendering, sampling, sensing, and output work.
The number returned in one step need not equal the number physically crossing
inside that window because the readout stage orders reported times.

### `CameraEngine.window`

```python
raw, counters = camera.window(t_w, pixel_ids, levels, rank)
```

Advance one window using caller-selected pixels and lattice levels. Pixel IDs
must be unique; levels and rank have one entry per pixel. `levels=None` is valid
only for a configured fixed-rate grid. This bypasses the full step's automatic
planning/output stage and returns raw sensor-order events, sorted by physical
crossing time and pixel.

Raw labels differ from saved file labels. This method also omits the attribution
increment columns returned by `SensorModel.advance`; its result is therefore
not a complete input for arbitrary relabeling. Use `step` when writing a normal
recorded event stream.

### `CameraEngine.flush`

Release remaining delayed events in final output order after the last window.
The columns match `step`. Flush also completes any rate-limiter bin waiting for
later events. Omitting it loses a valid readout tail even though physical
simulation has reached its requested end.

### `CameraEngine.gt`

`gt(x, y, t_ns)` queries center-ray instance, semantic, camera-z depth, and flow
at true-clock times. This geometric query does not infer a surface from the
pixel's blurred irradiance mixture. For image-shaped queries and detailed
renderer flags, use [rendering methods](rendering.md).

### Clock and gain inspection

`sensor_clock_ns(t_ns)` maps true time to the sensor clock without readout delay
or quantization. `gain_map()` returns `float32 (H, W)` static photocurrent gains.
Properties `readout_quantum_ns`, `readout_delay_ns`, `device`, and
`illuminance_scale` expose the selected sensor conversion; a reference scale is
available after `start`. Clock mapping and photocurrent gain serve different
stages and should not be combined into a timestamp correction.

### `CameraEngine.limiter`

Return cumulative `kept`, `dropped`, `capped_bins`, `held`, and timing counters,
or `None` when limiting is off. Drops occur after a pixel has fired and reset.
They affect the observed stream without undoing that pixel's internal state.
The held count reaches zero after a complete flush.

### Cancellation

`CancelToken.set()` requests cancellation; `is_set()` reads it. Tokens are not
resettable. `camera.set_cancel(token)` attaches one, or accepts `None` to clear
the attachment. Start, batches, refinement rounds, and pre-sensing boundaries
check it. A canceled camera is on an abort path and must be started again before
reuse; cancellation is not a checkpoint.

### Sample diagnostics

`last_runs()` exposes the most recent window's pixel IDs, ragged offsets, true
times, irradiance, and optional band counts as the sensor consumed them.
`runs_digest()` returns host audit hashes for the start frame and last runs.
These are diagnostic hooks for stage comparisons, not portable serialized
sensor states. See [sensor state methods](sensor.md) when state restoration is
required.

## Motion and planning

### `CameraGeometry`

`ballasim.engine.camera.CameraGeometry(cam, support_px=...)` precomputes pixel
rays, Morton ranks, tile cones, and image-speed factors for one calibrated camera.
These geometric quantities let the planner convert bounded scene motion to
sampling rates. They do not render irradiance or choose physical thresholds.

### `SceneMotion`

`ballasim.engine.kinematics.SceneMotion(bundle)` supplies `camera_window`,
`camera_pose`, `body_windows`, `light_windows`, and `body_moving_at`. Window
queries bound the spline's translational and angular speeds over every segment
touched; a held flag certifies exact stillness over the window. Use window bounds
for conservative scheduling and point-time poses for actual geometry evaluation.

### `MeshDistance`, `BodyWindow`, and `LightWindow`

`MeshDistance(mesh).distance(q)` provides a bounded mesh-distance estimate for
motion planning. `thickness(cap)` estimates a bounded thickness measure, and
`cone(q)` encloses the body's vertices as seen from a point. `BodyWindow` and
`LightWindow` combine geometric placement with bounded motion over a window.
These inputs help choose sampling rates; they do not replace the renderer's
triangle intersections.

### `Planner.plan`

```python
Planner(geometry, cam_index, motion, d_px, refine_tol_log, min_dt_ns,
        crop, plan_pass, ...).plan(t_w)
```

Return a `TilePlan` containing tile-shaped `static`, `level`, `v_hat`, and `stats`.
The image-speed bound `v_hat` and displacement target select a dyadic lattice
level. Certified static tiles need no new samples. `plan_pass(t_ns)` supplies
true-clock center-ray depth (`NaN` on misses); optional instance, shadow,
normal, and class data enable the corresponding prediction modes.

The plan is an initial schedule. Later radiance and visibility tests may still
refine it, so a tile's lattice level is not a complete count of rendered samples.

### Planning helpers

`dilate_by_reach(value, radius)` expands tile values over their reach.
`moving_light_edge_speed(x, light, bodies)` bounds a moving-light edge at a
receiver. `obliquity_budget(groups, max_samples_per_window=None)` budgets
receiver-projection refinement. These implement the same planning policies used
by the native engine; inspect [accuracy settings](../parameters/accuracy.md)
before overriding them.

## Sampling and current runs

### `MidpointTest`

Construct with `(tol, dark, m_min, m_max)` or
`for_groups(tol, dark, waveforms)`. For `float32 (N, G)` endpoint and midpoint
irradiances, `eta(ea, em, eb)` estimates relative interpolation error;
`exceeds(ea, em, eb)` compares it to tolerance. `jump(ea, eb)` estimates the
endpoint change used in jump rules. The dark term avoids interpreting low-signal
relative change without the sensor's additive current floor.

### `sample_window`

```python
sample_window(t_w, pixel_ids, levels, rank, carried_irr, carried_sig, render,
              *, refine_signature, midpoint, max_refine_levels, min_dt_ns, ...)
```

Build the initial lattice, request missing samples, refine flagged intervals,
and return `WindowRuns`. The render callback receives pixel IDs and true times
and must return irradiance and signatures in request order. Carried values
supply the already-known window-start samples; they must belong to this
camera and current window boundary.

### `fixed_rate_window`

`fixed_rate_window(t_w, pixel_ids, rank, period_ns, carried_irr, carried_sig,
render)` uses a uniform grid without adaptive refinement. Its result has the
same ragged run format, so the downstream sensor can consume either schedule.
`waveform_range(waveform)` supplies all-time light bounds; `level_cap(levels,
budget, l_min)` computes a deliberate initial-lattice budget cap.

### `WindowRuns`

The return contains `pixel_ids: uint32 (N,)`, `t_ns: int64 (S,)`,
`irradiance: float32 (S, G)`, and `offsets: int64 (N+1,)`. Times increase within
each pixel run. `last_irradiance`, `last_signature`, and `stats` retain boundary
values and work counts. Pass the ragged arrays to
[`SensorModel.advance`](sensor.md); do not reinterpret them as uniformly spaced
image frames.

### `_engine.limit`

`limit(events, *, family="prophesee_gen3", sensor={}, seed=1, camera=0,
width, cuts=None)` returns a keep mask and limiter counters for events sorted
by reported time. `cuts` identifies call boundaries up to which the stream is
complete. The token bucket carries state between cuts; only its rate cap resets
at an epoch. This helper evaluates readout-side drops independently of pixel
integration.
