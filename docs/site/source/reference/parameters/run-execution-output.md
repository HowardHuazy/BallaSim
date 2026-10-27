# Run and execution settings

Top-level `run.*` and `execution.*` controls select the workload and allocate
resources. The entries below distinguish changes to requested work from changes
to batching or thread counts.

## Run selection and time

(run-select)=
### `run.select`

Choose the scene cameras that produce event streams. `null` selects every bundle
camera; a supplied list must contain unique, existing names and cannot be empty.
The resolver returns them in bundle order. Camera calibration comes from the
scene, so adding a name here cannot create a new sensor pose or change its
resolution.

(run-frames-select)=
### `run.frames_select`

Select additional cameras for frame output without simulating their event
streams. These names must be disjoint from `run.select`. Their frame settings
are resolved separately, allowing an RGB-style view of the same scene without
paying for an event sensor on that view. Current APS conversion requires a
recorded event-camera photocurrent scale and therefore does not support these
frame-only cameras.

(run-t-start-ns)=
### `run.t_start_ns`

Set the requested start on the scene's true clock. `null` uses the bundle start.
The engine intersects the requested interval with scene support, then rounds
inward to complete windows aligned to time zero. Consequently the first output
window may begin after the requested timestamp. Initialization may evaluate
available earlier history, but its events are discarded.

(run-t-end-ns)=
### `run.t_end_ns`

Set the requested true-clock end, or use the bundle end with `null`. Together
with the start, it must leave at least one complete `2**23` ns window. Physical
crossings are cut on this simulated interval; readout delay can place their
reported timestamps after its end. Inspect the recorded `span_ns` rather than
assuming the requested end was an aligned output boundary.

(run-crop-px)=
### `run.crop_px`

Restrict event simulation to `[x0, y0, x1, y1]`, with exclusive upper bounds.
Pixel coordinates and random-number identities retain the full sensor origin,
so a crop is not a new camera calibration. The region must fit every selected
camera. Scene visibility is still evaluated against the full geometry, and
veiling glare still uses full-frame illumination statistics.

(run-device)=
### `run.device`

Choose the complete simulation backend. `gpu` requires OptiX and the CUDA sensor;
`cpu` uses Embree and the CPU sensor. `auto` tries the usable device engine and
falls back to CPU. Compiled CUDA kernels alone do not provide the full GPU
engine. Backend selection can change floating-point rounding, so record it
alongside precision when comparing runs.

(run-fixed-rate-hz)=
### `run.fixed_rate_hz`

Replace adaptive planning with a uniform temporal grid on every active pixel.
The rate must correspond to a positive integer period in nanoseconds. This is
useful for comparisons against a fixed-sampling approximation of the same
sensor model. It is not a frame camera: events still come from threshold
crossings of the supplied current curve. `null` enables the adaptive planner;
planner displacement and refinement controls are inactive under fixed sampling.

## Random realizations

(run-sensor-seed)=
### `run.sensor_seed`

Choose the event sensor's frozen parameter maps, stochastic noise realization,
and readout phase. Philox draws are keyed by their purpose and camera/pixel
identities rather than work order. Keep this seed fixed when isolating the
effect of a numerical or physical parameter; changing it intentionally samples
a different sensor realization. APS uses it only when its own seed is `null`;
IMU has a separate seed.

(run-render-seed)=
### `run.render_seed`

Choose the scrambling of the renderer's spatial subsample pattern. The pattern
remains associated with its pixel and is reused over time, avoiding unrelated
spatial redraws at every temporal query. This seed governs finite-sample image
integration, independently of the sensor's temporal noise and mismatch maps.

## CPU work and batching

(execution-sensor-threads)=
### `execution.sensor_threads`

Set the CPU sensor worker count; `0` uses hardware concurrency. This controls
parallel work distribution while preserving the configured pixel model and keyed
random draws. It does not choose GPU execution or change the sampling schedule.
The setting is relevant when the CPU sensor performs the work.

(execution-render-threads)=
### `execution.render_threads`

Set the host renderer's worker count. `0` chooses at least one worker, normally
hardware concurrency minus one, leaving capacity for the caller and other
stages. This changes how rendering requests are processed, not their ray or
sample definitions.

(execution-scene-workers)=
### `execution.scene_workers`

Set parallel workers for EVIMO2 per-body trajectory fitting. Each body's fit is
independent; changing the count redistributes preparation work without choosing
a different fit objective or regularization. It does not control simulation
threads or Blender motion-request sampling.

(execution-max-rays-per-launch)=
### `execution.max_rays_per_launch`

Limit rays in one renderer launch, including shadow work. Smaller launches use
less transient work per batch but add launch overhead. Larger launches amortize
that overhead at the cost of larger work units. This partitions the requested
rays; it does not reduce the spatial count or impose an accuracy cap.

(execution-pose-cache-mb)=
### `execution.pose_cache_mb`

Budget the renderer's cached pose tables. Entries avoid re-evaluating transforms
at repeated query times; the table is flushed when its limit is exceeded.
A smaller cache can require recomputation but does not simplify trajectories.
Despite the configuration suffix, one MB here is `2**20` bytes (MiB).

(execution-device-budget-mb)=
### `execution.device_budget_mb`

Budget sample, plan, and output working buffers. `0` selects 70% of memory
available after scene construction: free device memory on GPU, available host
memory on CPU. The engine batches requested work into that budget and refuses
a workload that cannot fit its minimum work unit. It does not relax sampling
tolerances; use `accuracy.max_samples_per_window` only when deliberate lattice
clamping is intended.

## Diagnostic work

(execution-census)=
### `execution.census`

Count subsamples that a proposed probe-reuse rule could avoid and compare their
hits. The counters describe potential work reuse; enabling them does not make
the renderer skip those samples. They add diagnostic cost and belong in
`perf.jsonl` interpretation, not physical sensor calibration.

(execution-pv-outputs)=
### `execution.pv_outputs`

Request projected-visibility scratch codes and tables for supported render
calls with at least 256 subsamples. These support inspection and cost accounting
for visibility prediction. They do not replace the production render result or
change event crossings. Unsupported material/device combinations may restrict
which diagnostic outputs are available.

(execution-jump-counters)=
### `execution.jump_counters`

Count the behavior of jump and sibling-refinement rules, including localization
brackets. This records diagnostic alternatives alongside the selected schedule;
it does not silently adopt those alternatives. Enable it when inspecting why
samples were taken, rather than to request finer event precision.

## Blender frame execution

(execution-frames-device)=
### `execution.frames_device`

Select the device for Blender frame rendering: `optix`, `cuda`, or `cpu`.
`auto` tries them in that order; OSL requires OptiX or CPU. This belongs to the
Blender frame-planning interfaces and is independent of the asynchronous event
engine's `run.device`. The current checkout has no complete matched/full
render-and-ingest writer.

(execution-frames-texture-mb)=
### `execution.frames_texture_mb`

Budget decoded float textures used by matched Blender planning. Above the
budget, byte textures use nearest interpolation, changing the intended texture
comparison rather than the event renderer's texture storage. Treat this as a
frame-planning compatibility choice. It has no effect on the reference-frame
writer's BallaSim rendering.

The implementation separates these execution controls from output-affecting
settings in its stage digests. See
{download}`engine.h <../../../../../src/engine/engine.h>` and
{download}`param_fields.h <../../../../../src/render/param_fields.h>` for batching
and cache units.

## Quick reference

### Run

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `run.sensor_seed` | `int` | `1` | Seed for event-sensor maps, noise, and readout phase. Range: ≥0. |
| `run.render_seed` | `int` | `0` | Seed for scrambled subpixel sampling patterns. Range: ≥0. |
| `run.device` | `enum` | `"auto"` | `gpu` uses OptiX/CUDA; `cpu` uses Embree; `auto` chooses a usable GPU first. Choices: `auto`, `gpu`, `cpu`. |
| `run.fixed_rate_hz` | `float? · Hz` | `null` | Fixed sampling rate for every crop pixel; requires an integer period in ns. `null` enables adaptive scheduling. Range: >0. |
| `run.crop_px` | `int[4]? · px` | `null` | Half-open `[x0, y0, x1, y1]` pixel region; `null` selects the full image. Range: ≥0. |
| `run.t_start_ns` | `int? · ns` | `null` | True-clock start bound; `null` uses the bundle start. Only complete aligned windows are retained. |
| `run.t_end_ns` | `int? · ns` | `null` | True-clock end bound; `null` uses the bundle end. |
| `run.select` | `string[]?` | `null` | Camera names producing events; `null` selects all bundle cameras. |
| `run.frames_select` | `string[]` | `[]` | Additional cameras producing frames without events; must be disjoint from the event selection. |

### Execution

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `execution.sensor_threads` | `int` | `0` | CPU sensor threads; `0` uses hardware concurrency. Range: ≥0. |
| `execution.render_threads` | `int` | `0` | Embree threads; `0` uses hardware concurrency minus one. Range: ≥0. |
| `execution.max_rays_per_launch` | `int` | `16777216` | Ray budget per renderer launch, including shadow rays. Range: ≥65536. |
| `execution.pose_cache_mb` | `int · MB` | `256` | Pose-table memory limit in MiB; exceeding it flushes the cache. Range: [0, 1.0995116e+12]. |
| `execution.census` | `bool` | `false` | Record diagnostic counters for potential subpixel probe reuse. |
| `execution.pv_outputs` | `bool` | `false` | Write projected-visibility scratch tables at sample counts of at least 256; records counters. |
| `execution.device_budget_mb` | `float · MB` | `0.0` | Working-buffer memory budget; `0` uses 70% of device memory free after scene construction. Range: ≥0. |
| `execution.jump_counters` | `bool` | `false` | Record diagnostic jump-rule and localization-bracket counters. |
| `execution.scene_workers` | `int` | `1` | Threads for EVIMO2 per-body trajectory fits. Range: ≥1. |
| `execution.frames_device` | `enum` | `"auto"` | Blender frame-render device; `auto` tries OptiX, CUDA, then CPU. OSL requires OptiX or CPU. Choices: `auto`, `optix`, `cuda`, `cpu`. |
| `execution.frames_texture_mb` | `int · MB` | `4096` | Matched-mode Blender float-texture budget; exceeding it falls back to byte textures with nearest interpolation. Range: ≥0. |


## Selection and interval constraints

`run.select` must be non-empty and contain unique camera names. `run.frames_select`
contains additional unique cameras and must be disjoint from event selection.
A crop must be non-empty and fit every selected camera.

The requested time interval is intersected with the scene and rounded inward to
complete `2**23` ns windows aligned to time zero. At least one window is required.
A fixed sample rate must have a positive integer period in ns. Adaptive
refinement and planner displacement controls are inactive under fixed sampling.

`gpu` requires usable CUDA and OptiX support; an explicit unavailable GPU is an
error. The CPU sensor's arithmetic is selected with `accuracy.sensor_precision`.
