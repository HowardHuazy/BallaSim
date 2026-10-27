# CLI reference

```text
ballasim [-v] COMMAND ...
python -m ballasim [-v] COMMAND ...
python -m ballasim.viz COMMAND ...
```

`ballasim` and `python -m ballasim` are equivalent. Place `-v` or `--verbose`
before the command to enable debug logging. Use `COMMAND --help` for its complete
option list.

## Commands

| Command | Operation |
| --- | --- |
| `run CONFIG` | Simulate a format-2 run configuration |
| `evimo2 SEQUENCE --out DIR` | Build or reuse an EVIMO2 scene and simulate it |
| `resolve CONFIG` | Resolve configuration values without simulating |
| `params [KEY]` | List parameter descriptors, optionally filtered by key or prefix |
| `profiles` | List installed profiles and their mount points |
| `probe` | Report the version, build, interpreter and supported protocols |
| `scene build CONFIG` | Build and cache only the scene bundle |
| `dense RUN` | Sample dense ground truth from a recorded run |
| `frames reference RUN` | Render exposure-integrated reference frames |
| `frames aps RUN --camera NAME` | Apply the APS model to stored mono frames |
| `imu RUN` | Simulate IMU tracks for a recorded run |
| `export RUN --format FORMAT` | Export events, frames, IMU or calibration |
| `ui` | Start the local browser interface |
| `job DIR` | Execute the steps in a `ballasim.job/1` job directory |
| `blender check --exchange DIR` | Inspect a Blender exchange and conversion report |
| `blender session --exchange DIR` | Serve motion requests over the calling process's pipes |
| `blender replay --exchange DIR` | Refit a recorded motion session without Blender |
| `blender camera-settings SOURCE` | Convert camera calibration to Blender settings |
| `scan report TARGET` | Report scan import, texture, geometry and scale diagnostics |

`RUN` denotes a directory containing `run.json`. `params`, `profiles` and `probe`
accept `--json`. See [Running simulations](../guide/simulations.md) and
[Outputs & ground truth](../guide/outputs.md) for workflows.

## Configuration and runs

`run`, `resolve`, `scene build` and `evimo2` accept repeatable `--profile ID` and
`--set KEY=VALUE` options. Values use YAML scalar or flow-list syntax. Camera keys
must include their mount, such as `cameras.default.sensor.theta_on`.

```bash
ballasim resolve run.yaml --all
ballasim resolve run.yaml --json
ballasim run run.yaml --set cameras.default.sensor.theta_on=0.25
ballasim params sensor
```

`resolve --all` includes values supplied by defaults and derivations. `run` and
`evimo2` accept `--overwrite` to replace an existing run.

| EVIMO2 option | Meaning |
| --- | --- |
| `--camera NAME` | Select `left_camera`, `right_camera` or `samsung_mono`; inferred from the path by default |
| `--t0 S`, `--t1 S` | Start and end in seconds on the bundle clock |
| `--profile-set SET` | Choose `v0` (default) or `v1` defaults |
| `--camera-profile CAMERA=ID` | Append a profile for the selected camera; repeatable |
| `--no-profiles` | Omit default profiles; supply the required sensor and lighting values explicitly |

The `v1` profile set requires `EVIMO2_COLORFIX_ROOT` and normal-light sequences.
`--profile-set` and `--no-profiles` are mutually exclusive.

## Dense, frame and IMU outputs

| Command | Options |
| --- | --- |
| `dense RUN` | `--dt-ms` (default `20`), `--t0`, `--t1`, repeatable `--camera`, `--channels`, `--out` |
| `frames reference RUN` | Repeatable `--camera`, `--set`; `--threads` (default `0`, hardware-selected) |
| `frames aps RUN` | Required `--camera`; repeatable `--set`; `--threads` (default `0`) |
| `imu RUN` | Repeatable `--camera`, `--profile`, `--camera-profile CAMERA=ID`, `--set`; `--overwrite` |

Dense channels are `depth`, `instance`, `semantic`, `flow`, `irradiance` and
`shadow_factor`. The default is `depth,instance,irradiance`; output defaults to
`RUN/dense`.

Frame reference overrides accept camera `frames.*` values,
`run.frames_select`, `execution.frames_device` and `execution.frames_texture_mb`.
APS overrides accept camera `frames.aps.*` values. IMU profiles and overrides may
set only `imu.*` values; `--set` uses full camera-mounted keys.

## Export

```bash
ballasim export runs/example --format mcap
ballasim export runs/example --format aedat4 --frame-map srgb8
```

| Option | Values or default |
| --- | --- |
| `--format` | Required: `raw-evt3`, `raw-evt2`, `raw-evt21`, `dat`, `aedat4`, `rosbag1`, `mcap`, `kalibr` |
| `--camera NAME` | Repeatable; defaults to the run's cameras |
| `--out DIR` | Defaults to `RUN/exports/FORMAT` |
| `--overwrite` | Replace an existing export |
| `--t0 S`, `--t1 S` | Bundle-clock cuts in seconds |
| `--epoch-ns N` | Added timestamp epoch; a multiple of 1000; default 1 s for ROS containers, otherwise 0 |
| `--no-imu`, `--no-frames` | Omit the corresponding streams |
| `--compression` | Format-dependent; defaults to `bz2` for rosbag1 and `zstd` for MCAP and AEDAT4 |
| `--threads N` | Compression workers, 1–256; default `4` |
| `--event-msg` | rosbag1: `dvs` (default) or `evt3`; MCAP: `evt3` |
| `--packet-us N` | ROS or AEDAT4 packet duration; default 1000 µs for ROS, 10000 µs for AEDAT4 |
| `--topic STREAM:CAMERA=NAME` | Repeatable ROS or Kalibr topic mapping; streams: `events`, `imu`, `image_raw` |
| `--imu-covariance` | `model` (default) or `zero` |
| `--evt2-time-high` | `openeb` (default) or `change` for RAW EVT2 |
| `--frame-source` | `auto` (default), `aps`, `rgb`, `mono` |
| `--frame-map` | `none` (default), `linear8`, `srgb8`; AEDAT4 frames require an 8-bit mapping |
| `--frame-scale F` | Scale before frame mapping; default `1` |
| `--kalibr-inflate F` | Multiply Kalibr noise densities; default `1` |

## Browser interface and jobs

```bash
ballasim ui --root runs=/path/to/runs --no-browser
ballasim ui --open runs/example
ballasim job /path/to/job
```

The UI binds to `127.0.0.1` by default; `--bind ::1` selects IPv6 loopback. It
uses an available port unless `--port N` is given. `--root NAME=PATH` is repeatable.
`--status` reports the running server; `--stop` stops it.

Additional options select the browser (`--browser CMD`), EVIMO2 data
(`--evimo2-root`), job storage (`--jobs-root`), source checkout (`--repo`), shared
GPU lock (`--gpu-lock`), event cache (`--cache-mb`, default `512`) and sensor
replay device (`--resim-device auto|gpu|cpu`). `--dev` enables heuristic parameter
editing and disables static caching.

`job DIR` reads `job.json` and writes `status.json`. External execution steps
require an allow-listed program or `--allow-exec`.

## Blender and scan tools

`blender check` accepts `--config FILE`, repeatable `--set KEY=VALUE` and `--json`.
Overrides require a configuration. A completed check exits successfully even
when its report contains refused features; inspect the report items.

`blender session` and `blender replay` accept `--config FILE` and repeatable
`--set`. These override `scene.blender.*`, `run.select`, `run.frames_select` or
`execution.scene_workers`. Replay reads `motion/oracle_log.npz` in the exchange.

`blender camera-settings` requires one source: `--preset NAME`, `--kalibr FILE`
or `--opencv FILE`. Optional arguments are `--camera NAME`, `--sensor-width MM`
and `--json`. Presets are `evimo2_left`, `evimo2_right` and `evimo2_samsung`.

`scan report` takes a mesh path or `SCENE.yaml#ASSET`. For a path, import options
are `--format gltf|obj|ply|replica`, `--scale`, `--shading auto|lit|baked`,
`--max-texture-size`, `--license` and `--scale-basis`. OBJ and PLY require a scale.
`--theta` defaults to `0.25`; `--weights R G B` replaces the sensor RGB weights.
Use `--out FILE` to save the JSON report.

## Event visualization

```text
python -m ballasim.viz {info,frames,compare,html,stats} ...
```

The `viz` extra is required. All commands accept `--real PATH`, `--sim PATH` or
both. Sources may be EVIMO2 sequence directories, HDF5 event files, NumPy files or
directories of NumPy columns.

| Command | Output |
| --- | --- |
| `info` | JSON stream metadata |
| `frames --out DIR` | PNG panels for each time window |
| `compare --out PATH [PATH ...]` | Video, GIF, PNG strip or image directory |
| `html --out FILE.html` | Browser viewer with a time slider |
| `stats --out DIR` | Event-rate, polarity, spectrum and interval statistics |

`--t0` and `--t1` select seconds on the common clock. Per-source options are
`--real-label`, `--real-offset-ms`, `--real-gt-offset-ms` and `--real-polarity`,
with corresponding `--sim-*` forms. The default interval is the streams' overlap.

Rendering commands share `--dt-ms` (default `10`), `--win-ms` (defaults to the
step), `--kind events|polarity|timesurface|counts`, `--theme dark|white`,
`--mask valid|labels|none`, `--clip`, `--gamma`, `--tau-ms`, `--workers` and
`--third PANEL[,PANEL]`. Extra panels are `diff`, `ratio`, `depth` and `irradiance`;
`--sigma-px` sets difference or ratio smoothing.

`compare` additionally accepts `--layout h|v`, `--scale`, `--fps` and
`--strip-frames`. `html` accepts `--fps`, `--format`, `--frames-dir` and `--title`.
`frames` accepts `--times-ns FILE`, `--win prev|exposure`, `--flat` and `--name`
for recorded preview schedules and naming.

`stats` provides `--rate-bin-ms`, `--on-fraction-bin-ms`, `--spectrum-bin-us`,
`--segment-s`, `--fmin`, `--fmax`, `--max-interval-events` and `--mask none|valid`.

## Exit status

Successful commands return `0`. Argument and reported configuration or export
errors return `2`; `ui --status` returns `1` when no server is running. Canceled
jobs return `3`; other job failures return `1` or `2`. Verbose mode includes
tracebacks for reported run errors.
