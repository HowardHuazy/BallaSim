# Outputs & ground truth

A run stores events, configuration provenance and performance records beneath
`output.dir`. Dense ground truth, reference frames and exports are generated
separately. Enabled IMUs are generated automatically after the event run.

Use events for asynchronous observations, dense queries for geometry at chosen
times, and reference frames for exposure-averaged images. APS adds frame-camera
readout effects to those images; IMU output samples the rig's physical motion.
These products have different measurement and timestamp conventions, described
below.

[Output settings](../reference/parameters/output.md) select the run directory,
overwrite policy, event ground truth and label threshold.

```text
run/
  run.json
  progress.json
  perf.jsonl
  events/<camera>.h5
  dense/<camera>.h5          # dense ground truth
  frames/<camera>.h5         # exposure reference and optional APS output
  imu/<camera>.h5
  imu/<camera>.json
  exports/<format>/EXPORT.json
```

## Event files

Each camera writes a DSEC-compatible HDF5 file with format
`ballasim.events/1`. All event columns have length `N`. Events are ordered by
reported time, then `x`, then `y`; ties at one pixel retain firing-time order.

| Dataset | Type | Description |
| --- | --- | --- |
| `/events/x`, `/events/y` | `uint16` | Raw, distorted pixel coordinates. |
| `/events/p` | `uint8` | Polarity: `1` ON, `0` OFF. |
| `/events/t` | `int64` | Reported time in microseconds relative to `/t_offset`. |
| `/events/t_ns_rem` | `uint16` | Nanosecond remainder in `[0, 999]`. |
| `/t_offset` | `int64` scalar | Timestamp origin in microseconds on the sensor clock. |
| `/ms_to_idx` | `uint64` | Entry `k` is the first event with `/events/t >= 1000*k`. |

Root attributes include `format`, `camera` and `resolution = [W, H]`. Event
columns use Blosc compression and Fletcher-32 checksums. Import `hdf5plugin`
before reading these datasets directly with `h5py`.

### Timestamps

Recover the exact reported timestamp with integer arithmetic:

```python
t_reported_ns = 1000 * (t_offset + t) + t_ns_rem
```

Reported timestamps include sensor clock offset, drift and readout. Ground-truth
`t_cross_ns` is the physical firing time on the scene's true clock. Readout may
place reported timestamps after the run's true-clock end.

Pair sensor-clock observations using their reported timestamps. Compare against
physical scene state using `t_cross_ns`, or the explicit true-time fields of
frames and IMU data. Subtracting only the camera clock offset does not remove
readout quantization, filter latency or clock drift.

### Per-event ground truth

Set `output.gt: true` to include `/gt`. Each row corresponds to the same row in
`/events`. Geometry is evaluated along the pixel's center ray at `t_cross_ns`.
The geometry describes the visible surface at firing time, including for noise
events; it is not an attribution of which subpixel surface caused a crossing.

| Dataset | Type | Description |
| --- | --- | --- |
| `/gt/instance` | `uint32` | Instance ID; `0` when no geometry is hit. |
| `/gt/semantic` | `uint16` | Semantic class ID. |
| `/gt/depth` | `float32` | Camera-frame z depth in meters; `NaN` on a miss. |
| `/gt/flow_x`, `/gt/flow_y` | `float32` | Instantaneous motion field in raw pixel coordinates, px/s; `NaN` on a miss. |
| `/gt/label` | `uint8` | `0` SIGNAL, `1` MIXED (reserved), `2` NOISE, `3` LEAK, `4` HOT. |
| `/gt/t_cross_ns` | `int64` | Physical firing time, ns. |
| `/gt/flags` | `uint16` | Bit field described below. |

Flag bits are `0` MOVING, `5` HOT_PIXEL, `9` QUARANTINED,
`10` SOLVER_FALLBACK and `12` FIRST_AFTER_INIT. MOVING describes the hit body's
motion state; it does not indicate whether camera motion produced the event.

Labels separate deterministic signal, leakage and noise using their comparator
increments and `output.label_cut`. They are an operational model annotation,
not a counterfactual test of whether an event would exist without noise.
`MIXED` is reserved in the file format; the current engine emits SIGNAL, NOISE,
LEAK and HOT. FIRST_AFTER_INIT identifies crossings whose label handles the
initial comparator reference separately.

## Dense ground truth

```bash
python -m ballasim dense runs/example --dt-ms 20 \
    --channels depth,instance,semantic,flow,irradiance
```

This writes `dense/<camera>.h5` with format `ballasim.dense/1`, using the run's
scene and recorded renderer settings. The bundle content digest must match the
run. By default, the channels are `depth`, `instance` and `irradiance`.

| Channel | Shape per frame | Units |
| --- | --- | --- |
| `depth`, `instance`, `semantic` | `(H, W)` | Meters, instance ID, class ID. |
| `flow` | `(H, W, 2)` | px/s in raw coordinates. |
| `irradiance` | `(H, W, G)` | Renderer units per light group, before temporal waveforms. |
| `shadow_factor` | `(H, W, G)` | Proxy-shadow attenuation; `1` without baked proxies. |

`t_true_ns` stores true-clock query times; `t_ns` stores their sensor-clock mapping
with offset and drift, without readout delay or quantization. Use `--camera NAME`
to select cameras and `--out DIR` to select another output directory. The Python
API also accepts explicit query times.

Dense queries trace the current scene state without integrating an exposure.
They therefore preserve sharp instantaneous motion and can be requested even
where no event fired. `shadow_factor` describes baked proxy shadows, not general
scene visibility. Use {ref}`export_dense <export-dense>`
to request exact times or a different channel set.

## Reference frames and APS

Reference frames average BallaSim's irradiance over each row's exposure, with
temporal light waveforms applied. They use the event renderer's scene, spatial
pixel response and camera model. Motion blur is integrated numerically; waveform
means are evaluated analytically. They contain no APS noise or display tone
mapping until those transformations are requested.

```bash
python -m ballasim frames reference runs/example --camera camera \
    --set cameras.camera.frames.timeline=uniform \
    --set cameras.camera.frames.exposure_ns=10000000
```

Replace `camera` with a camera in `run.select` or `run.frames_select`. Without
`--camera`, the command selects cameras with `frames.mode: reference`. Available
reference outputs are `mono`, `reference` and `aps`; `aps` requires `mono` and an
event camera. `matched` and `full` frame modes are not generated by this command.

Files use `ballasim.frames/1`. Image arrays have shape `(N, H, W)` within `roi_px`;
`mono` and `reference` are `float32`, and `aps_dn` is `uint16`. The file records
exposure start, end and midpoint on both true and sensor clocks, the shutter
settings and the resolved frame parameters. Display PNGs are written beneath
`frames/<camera>/display/`.

Set {ref}`exposure duration <frames-exposure-ns>` to
control blur and {ref}`shutter timing <frames-shutter>`
to choose simultaneous or staggered rows. Frame `mono` remains an average in
renderer units when exposure changes. It is not an accumulated charge image.
Integration accuracy is controlled separately by the
[reference sampling settings](../reference/parameters/frames.md#reference-integration).
Inspect `probe_p999` and warnings when the temporal sample cap is reached.

To change the APS response using the stored `mono` frames:

```bash
python -m ballasim frames aps runs/example --camera camera \
    --set cameras.camera.frames.aps.bits=10
```

This updates the APS datasets and record without rendering the scene again.

APS multiplies the stored mean by the recorded photocurrent scale and exposure,
then applies Poisson shot noise, fixed pixel response errors, full-well
saturation, Gaussian read noise and ADC conversion. Longer exposures collect
more signal and dark charge; changing ADC bits affects digital resolution.
[APS parameters](../reference/parameters/frames.md#aps-readout) distinguish
device priors from uncertain defaults. APS requires an event camera's recorded
scale and uses an ROI mean to approximate veiling glare for cropped images.
See {ref}`export_reference <export-reference>` and
{ref}`aps <aps>` for programmatic use.

## IMU

Set `cameras.<camera>.imu.enabled: true` to generate an IMU after the event run.
The default enabled device is ideal. Noise, bias, filtering, quantization and
other device properties must be specified explicitly or supplied by a profile.

For an existing completed run:

```bash
python -m ballasim imu runs/example \
    --set cameras.default.imu.enabled=true
```

Only `imu.*` keys can be changed by this command. Use `--overwrite` to replace
existing IMU files. Each camera writes `ballasim.imu/1` HDF5 data and a JSON
record containing the resolved parameters, warnings, file hash and outcome.

| Dataset | Description |
| --- | --- |
| `/t_ns` | `int64` sensor-clock timestamps, ns, sharing the event clock. |
| `/gyro`, `/accel` | `float64 (N, 3)` in IMU axes: rad/s and specific force in m/s². |
| `/counts_gyro`, `/counts_accel` | Integer ADC counts, present when quantization is enabled. |
| `/flags` | Gyroscope rail bits `0–2`, accelerometer rail bits `3–5`, WARMUP bit `6`, IMPACT bit `7`. |
| `/truth/*` | True time, angular rate, specific force, biases, position and orientation. |

`imu.T_cam_imu` places the IMU in the camera frame. Gravity is specified in world
axes by `imu.gravity_mps2`, otherwise taken from bundle metadata or
`[0, 0, -9.80665] m/s²`. EVIMO2's tabletop world frame requires the corresponding
gravity vector.

The ideal gyro is the rig's angular velocity rotated into the IMU frame. The
accelerometer subtracts world gravity from translational acceleration and
includes acceleration from the mount's lever arm. A stationary level IMU
therefore reports zero angular rate and the opposite of gravity as specific
force.

The [IMU parameters](../reference/parameters/imu.md) add calibration errors,
continuous bias processes, filtering, sampling-clock variation and ADC effects.
Filter delay changes the measured response; timestamp-latch delay changes its
stamp. These effects should be configured separately. WARMUP and IMPACT flags
identify limited filter history and trajectory joins. The
{ref}`simulate_imu API <simulate-imu>` supports IMU-only
profile overlays without changing event files.

## Export

```bash
python -m ballasim export runs/example --format mcap
python -m ballasim export runs/example --format aedat4 --frame-map srgb8
```

Exports are written to `exports/<format>/`, with an `EXPORT.json` record of
options, timestamp mappings, stream counts, warnings and file hashes.

| Format | Event time | Other streams |
| --- | --- | --- |
| `raw-evt3`, `raw-evt2`, `raw-evt21`, `dat` | Relative integer µs. | IMU CSV sidecars. |
| `aedat4` | Integer µs. | IMU and 8-bit frames. |
| `rosbag1` | ns ROS stamps; event encoding selectable. | IMU and images. |
| `mcap` | ns ROS 2 stamps; event packets encode µs. | IMU and images. |
| `kalibr` | Parameter export. | IMU parameters and camera–IMU calibration YAML. |

`--t0` and `--t1` crop in seconds on the true clock. Without `--t1`, all stored
events are exported, including the readout tail. `--epoch-ns` sets the output
epoch and must be a multiple of `1000`; its default is zero, or one second for
ROS formats. Microsecond formats discard the nanosecond remainder and record
the loss in `EXPORT.json`.

Use `--no-imu` or `--no-frames` to omit those streams. Frame selection defaults
to `aps_dn`, then `rgb`, then `mono`; AEDAT4 frames require `--frame-map linear8`
or `srgb8`. A finished export is replaced only with `--overwrite`. See
[CLI reference](../reference/cli.md) for format options and limits.

`linear8` normalizes and clips intensities to an 8-bit range; `srgb8` additionally
applies the sRGB transfer curve. These are display conversions rather than
radiometric calibration. Preserve HDF5 float images or unmapped ROS image
streams when linear values are required. The
{ref}`export API <export>` records the selected mapping,
time origin and any discarded submicrosecond precision in `EXPORT.json`.

## View and compare

```bash
python -m ballasim.viz html --sim runs/example/events/camera.h5 --out viewer.html
python -m ballasim.viz compare --real /data/sequence \
    --sim runs/example/events/camera.h5 --out compare.mp4
python -m ballasim.viz stats --sim runs/example/events/camera.h5 --out stats
```

The viewer accepts BallaSim/DSEC HDF5, EVIMO2 sequences and NumPy event arrays.
Available views include signed events, polarity counts, time surfaces, activity
ratios, depth and irradiance. Dense files are discovered beside a run's event
files. Statistics include rate, ON fraction, spectra, intervals and per-pixel
activity. Use `--mask valid` to restrict statistics to available ground-truth
geometry.

Viewer `--t0` and `--t1` use the streams' comparison clock. Use the event and
ground-truth offset options explicitly when comparing streams on different
clocks. For interactive browsing, open the run with
`python -m ballasim ui --open runs/example`.

```{toctree}
:hidden:
:maxdepth: 1

Event-output settings <../reference/parameters/output>
Event files, dense queries, and export <../reference/api/outputs>
Frames and APS <../reference/parameters/frames>
IMU <../reference/parameters/imu>
Reading and visualization <../reference/api/visualization>
```
