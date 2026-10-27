# Architecture

BallaSim separates scene preparation, rendering, temporal sampling, sensor
simulation, and output. Python handles files and user interfaces; native modules
perform the numerical work.

## Pipeline

| Stage | Responsibility | Main implementation |
| --- | --- | --- |
| Scene | Load geometry, calibration, light groups, and fitted rigid motion into a scene bundle. | `python/ballasim/scene`, `src/scene` |
| Configuration | Resolve layers, validate values, derive settings, and record provenance. | `python/ballasim/config.py`, `src/config` |
| Render | Evaluate posed geometry, visibility, materials, and per-group irradiance at requested pixel times. | `src/render`, `src/backend` |
| Schedule | Select temporal samples from motion bounds and refine radiance or visibility changes. | `src/engine` |
| Sensor | Integrate the photocurrent response, locate threshold crossings, and apply readout. | `src/sensor` |
| Output | Write events, ground truth, logs, frame products, IMU samples, and format exports. | `python/ballasim/engine/writer.py`, `src/frames`, `src/imu`, `src/export` |

The engine processes fixed windows of `2^23 ns` (8.388608 ms). Sampling times
within a window lie on a dyadic lattice. Pose evaluation uses integer times;
device kernels receive integer offsets from an integer-nanosecond base.

## Backends and precision

The CPU path uses Embree for visibility and a CPU sensor model. The GPU path
uses OptiX for visibility and CUDA for rendering, scheduling, and sensor work.
`run.device: auto` selects the GPU when the device engine is usable.

Shared camera, trajectory, random-number, and mathematical routines live in
`src/core`. Host and device code share these definitions. The CPU sensor supports
`fp64` and `fp32`; the GPU sensor uses `fp32`. Backend and precision choices are
recorded with the run.

## Native modules

| Module | Interface |
| --- | --- |
| `_core` | Cameras, SO(3), splines, sampling, parameter schema, resolver, and browser wire format. |
| `_scene` | Asset processing, trajectory fitting, EVIMO2 construction, and Blender conversion. |
| `_render` | Scene rendering and dense ground-truth queries. |
| `_sensor` | Pixel-model integration and sensor replay primitives. |
| `_engine` | Window planning, sampling, sensor integration, output ordering, and cancellation. |
| `_frames` | Frame schedules, exposure integration, resampling, and APS response. |
| `_imu` | Inertial truth, clock, noise, filters, and quantization. |
| `_export` | AEDAT4, DAT, EVT, ROS, MCAP, and calibration writers. |
| `_cuda`, `_optix` | Optional CUDA spline and OptiX intersection interfaces. |

Nanobind exposes the native modules. `ballasim._core.build_info()` reports which
modules were built. Optional components are imported by the interfaces that use
them.

## Data and reproducibility

Scene bundles store array data separately from metadata and declare optional
features explicitly. A bundle contains scene data; run settings remain in the
run configuration.

The configuration resolver associates each parameter with its earliest affected
stage: `scene`, `render`, `schedule`, `sensor`, or `output`. Stage digests include
the preceding stages. Source and engine digests identify the inputs used to
produce a run.

Philox supplies counter-based random draws, and Owen-scrambled Sobol sequences
supply pixel subsamples. Render, sensor, APS, and IMU seeds are configurable.
Record the resolved configuration, input assets, and build alongside results.

## User interfaces

`ballasim.run` and `ballasim.engine` expose the simulation workflow. The command
line uses the same configuration resolver. `ballasim.viz` reads event files for
rendering and statistics. The optional Tornado UI serves a local browser client;
jobs run in worker processes and publish status files for progress and
cancellation.

See the topic-specific methods in the [simulation manual](../guide/scenes.md) and
[Outputs & ground truth](../guide/outputs.md) for file contracts.
