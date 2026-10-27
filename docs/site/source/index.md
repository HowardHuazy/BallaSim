# Overview

BallaSim simulates asynchronous event cameras from 3D geometry, lighting, and
continuous camera and object trajectories. The renderer samples each pixel at
adaptive times. The sensor model converts the resulting photocurrent into
timestamped ON and OFF events.

Simulation runs use a YAML configuration and execute on the CPU with Embree or
on an NVIDIA GPU with OptiX and CUDA. Python provides scene preparation,
configuration, simulation, and analysis interfaces.

## Documentation

| Start here | Description |
| --- | --- |
| [Installation](getting-started/installation.md) | Set up the toolchain and install BallaSim. |
| [Quickstart](getting-started/quickstart.md) | Create a small scene and write an event stream. |
| [Scenes & assets](guide/scenes.md) | Author a scene, import geometry, and work with scene objects. |
| [Trajectories](guide/trajectories.md) | Fit motion and evaluate poses and derivatives. |
| [Cameras & optics](guide/cameras.md) | Calibrate projection, photometry, and the pixel response. |
| [Event sensors](guide/sensors.md) | Understand photoreceptors, thresholds, noise, and readout. |
| [Running simulations](guide/simulations.md) | Configure execution, sampling accuracy, and progress. |
| [Outputs & ground truth](guide/outputs.md) | Read events, generate frames and IMU, and export results. |
| [Browser interface](guide/browser-ui.md) | Configure jobs and inspect event streams locally. |

Each manual topic brings together its workflow, Python methods, and settings.
Begin with the overview, use the named entries to understand individual controls,
and consult the tables for defaults and ranges.

## Inputs and outputs

**Inputs:** triangle meshes, materials, light groups, camera calibration, and
rigid-body trajectories. Scene sources include authored YAML, scene bundles,
Blender exchanges, and EVIMO2 sequences.

**Outputs:** events with integer-nanosecond timestamps and optional per-event
instance IDs, semantic IDs, depth, optical flow, noise labels, and moving flags.
Additional interfaces provide dense ground truth, reference frames, APS frames,
and IMU samples. See [Outputs & ground truth](guide/outputs.md) for file formats
and current workflow support.

## Conventions

Distances use meters, angles use radians, and simulation timestamps use signed
64-bit integer nanoseconds. Sensor timestamps include the configured clock and
readout model. Quaternions use scalar-first order `(w, x, y, z)`; image coordinates
use `(x, y)` with the origin at the upper left.

BallaSim is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Third-party assets retain their own licenses.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Getting started

self
getting-started/installation
getting-started/quickstart
reference/python-api
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Simulation manual

guide/scenes
guide/trajectories
guide/cameras
guide/sensors
guide/configuration
guide/simulations
guide/outputs
guide/browser-ui
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

development/architecture
development/contributing
```
