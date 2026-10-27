# Running simulations

Run a [configuration](configuration.md) from the command line or Python. Each
selected camera produces an event file and shares the run's resolved configuration.

## Run a configuration

```bash
python -m ballasim resolve run.yaml
python -m ballasim run run.yaml
```

`resolve` loads the scene and validates the configuration without simulating
events. Use `--all` to inspect every resolved value or `--json` to save the full
record. Both commands accept repeated `--profile ID` and `--set KEY=VALUE` options.

```bash
python -m ballasim run run.yaml --set run.device=cpu
python -m ballasim run run.yaml --set cameras.default.sensor.theta_on=0.2
```

The run directory is `output.dir`. An existing run requires `--overwrite` or
`output.overwrite: true`. A nonempty directory without `run.json`, or a directory
held by another run, is refused. Scene loading, resolution, and engine construction complete
before the run directory is created or replaced.

## Select the device and cameras

| Parameter | Behavior |
| --- | --- |
| `run.device: auto` | Use OptiX and the CUDA sensor when available; otherwise use Embree and the CPU sensor. |
| `run.device: gpu` | Require the device engine: OptiX and the CUDA sensor in fp32. |
| `run.device: cpu` | Use Embree and the CPU sensor at `accuracy.sensor_precision`. |
| `run.select` | Camera names to simulate; `null` selects all bundle cameras. |
| `run.crop_px` | Restrict events to `[x0, y0, x1, y1)`. Coordinates retain the full sensor origin. |

Use `python -m ballasim probe --json` to inspect the build and supported protocols.
See [Accuracy and sampling](../reference/parameters/accuracy.md) for numerical
controls and [Run and execution settings](../reference/parameters/run-execution-output.md)
for selection, batching, memory, and diagnostics.

## Time and initialization

`run.t_start_ns` and `run.t_end_ns` crop the scene's true-clock span. The engine
simulates complete windows of `2²³ ns`, aligned to time zero. The recorded
`span_ns` is the requested span rounded inward to these boundaries.

The sensor warms up before the first output window; warm-up events are discarded.
`accuracy.warmup` selects a held initial frame (`static`) or available scene motion
(`motion`). `accuracy.initial_state` selects phase-based or stationary comparator
initialization.

Event timestamps use each camera's sensor clock, including clock offset, drift
and readout. Per-event ground truth uses the physical firing time on the true
clock. See [Outputs & ground truth](outputs.md#timestamps).

## Run an EVIMO2 sequence

```bash
python -m ballasim evimo2 /data/evimo2/left_camera/imo/eval/scene13_dyn_test_00_000000 \
    --t0 0.5 --t1 1.0 --out runs/scene13
```

The command builds or reuses the scene bundle and applies the sequence camera's
sensor profile and the default lighting profile. `--t0` and `--t1` are seconds on
the bundle's true clock. Use `--camera` to select another camera folder of the same
recording. Profile selection and scene import are described in
[Configuration](configuration.md) and [Scenes & assets](scenes.md).

## Python

```python
import ballasim

summary = ballasim.run("run.yaml")
for camera in summary.cameras:
    print(camera.name, camera.events, camera.path)
```

`run` also accepts a `RunSpec` from `ballasim.config.load`. Pass `bundle=` to use an
in-memory scene. `RunSummary` records the backend, simulated span and per-camera
counts and timings.

For progress reporting, pass
`on_window(camera, window, windows, events, t_ns)`. The callback runs after each
completed window; `window` is the number completed, `events` counts crop-kept
crossings before output limiting, and `t_ns` is the true-clock
end time. A `ballasim._engine.CancelToken` passed as `cancel=` can be set from
another thread. Cancellation raises `ballasim._engine.Cancelled`.

## Monitor a run

| File | Content |
| --- | --- |
| `run.json` | Inputs, resolved values, provenance, warnings, stage digests and final camera summaries. |
| `progress.json` | Current camera, completed windows, event count and simulation time. |
| `perf.jsonl` | Per-window planning, rendering, sensing and output counters. |

`run.json` reports `running`, `complete`, `failed` or `cancelled`. Completed camera
files are retained after a failure or cancellation; the active camera's partial
event file is removed. Enabled IMUs are generated after the event run completes.
An IMU failure has its own record and leaves the event run marked complete.

The [run and window methods](../reference/api/simulation.md) expose the complete
Python call and the native step interface. Use the
[accuracy profiles](../reference/parameters/profiles.md) to choose a coherent
starting schedule before changing individual tolerances.

## Browser interface

Install the `ui` extra, then open a local run:

```bash
python -m ballasim ui --open runs/scene13
```

The interface provides run browsing, event views, comparisons, parameter forms,
job progress and provenance. Use `--root NAME=PATH` to add a browsing root.
The server binds to loopback only (`127.0.0.1` or `::1`). `ui --status` reports the
running server; `ui --stop` stops it. Use Chrome or Chromium 147 or later for the
browser interface.

## Jobs

`python -m ballasim job JOBDIR` executes the ordered steps in `JOBDIR/job.json`.
The document uses `format: "ballasim.job"`, `version: [1, 0]`, a configuration
path and a nonempty `steps` array. For example:

```json
{
  "format": "ballasim.job",
  "version": [1, 0],
  "config": "run.yaml",
  "steps": [{"step": "validate"}, {"step": "run"}]
}
```

Supported steps are `validate`, `run`, `dense`, `preview` and `exec`. Preview steps
require timeline metadata. External `exec` steps require an allow-listed
executable or `--allow-exec`. `status.json` records the active step and outcome;
a `cancel` file or an interrupt requests cancellation. Job exit codes are
`0` complete, `1` failed, `2` configuration or file error, and `3` canceled.

```{toctree}
:hidden:
:maxdepth: 1

Run and window methods <../reference/api/simulation>
Accuracy and sampling <../reference/parameters/accuracy>
Run and execution settings <../reference/parameters/run-execution-output>
Command line <../reference/cli>
```
