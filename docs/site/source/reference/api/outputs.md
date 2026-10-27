# Event files, dense queries, and export

Use these interfaces to derive observations from recorded runs and inspect the
resulting streams. Scene-changing edits require a new event run. Frame, IMU and
export functions instead use the recorded scene and configuration, with the
restricted overlays documented below.

| Task | Entry point |
| --- | --- |
| Query geometry at chosen times | {ref}`export_dense <export-dense>` |
| Render exposure-averaged frames | {ref}`export_reference <export-reference>` |
| Change APS readout without rendering | {ref}`aps <aps>` |
| Simulate rig-mounted inertial measurements | {ref}`simulate_imu <simulate-imu>` |
| Convert completed observations to external formats | {ref}`export <export>` |
| Read, render or summarize events | {ref}`open_events <open-events>`, {ref}`Renderer <viz-renderer>`, {ref}`compute_stats <compute-stats>` |

Paths are relative to a run directory where stated. Python time arguments use
integer nanoseconds. File layouts and clock conventions are specified in
[Outputs & ground truth](../../guide/outputs.md).

## Dense ground truth

(export-dense)=
### `export_dense`

```python
from ballasim.engine.dense import export_dense

export_dense(run_dir, dt_ns=None, *, times_ns=None, t0_ns=None, t1_ns=None,
             channels=("depth", "instance", "irradiance"), cameras=None,
             bundle=None, out_dir=None)
```

Render instantaneous ground truth using the recorded scene and renderer
settings. Return the written `list[Path]`, one `ballasim.dense/1` file per
selected camera. This samples geometry and irradiance directly; it does not
reconstruct images or motion from events.

Supply exactly one of positive `dt_ns` or `times_ns`. Regular sampling uses
`t0_ns + k*dt_ns` inside `[t0_ns, t1_ns)`, cropped to the run span. Explicit times
must be a nonempty, strictly increasing integer array within bundle support;
`t0_ns` and `t1_ns` do not apply to them. `cameras=None` selects event-run cameras
and `out_dir=None` selects `dense/`. A supplied bundle must match the run digest.

| Channel | Shape per query | Meaning |
| --- | --- | --- |
| `depth`, `instance`, `semantic` | `(H, W)` | Camera z in m, instance ID and semantic class. |
| `flow` | `(H, W, 2)` | Instantaneous raw-coordinate motion field, px/s. |
| `irradiance` | `(H, W, G)` | Pixel-integrated renderer signal before temporal waveforms. |
| `shadow_factor` | `(H, W, G)` | Baked-proxy shadow attenuation; one without baked proxies. |

Invalid times, channels or changed scenes raise `ValueError`. When an original
OptiX backend is unavailable, dense export warns and uses Embree; final bits may
differ between backends.

## Event writer

(event-writer)=
### `EventWriter`

```python
from ballasim.engine.writer import EventWriter

EventWriter(path, *, camera, width, height, t_offset_us, gt)
```

Write a single camera's DSEC-compatible `ballasim.events/1` file. This is the
streaming writer used by `ballasim.run`; it does not simulate events. `gt=True`
requires the complete ground-truth column set on each append. Data is written
to a neighboring partial file until successful close.

#### `EventWriter.append`

`append(x, y, p, t_reported_ns, gt=None)` appends aligned event columns.
Timestamps must be nondecreasing across calls and at least `1000*t_offset_us`.
The writer splits exact nanoseconds into microseconds and a remainder, and
constructs the millisecond index. It checks timestamp ordering but leaves
correct spatial tie ordering to the caller.

#### `EventWriter.close`

`close(t_end_reported_ns)` flushes pending rows, extends the time index to the
given sensor-clock bound and atomically renames the completed file. Return its
`Path`. Include the readout tail when choosing the end bound.

#### `EventWriter.abort`

`abort()` closes and removes the partial file. Use it on failure before close;
it prevents a truncated event file from appearing under the final name.

## Format export

(export)=
### `export`

```python
from ballasim.export.formats import export

export(run_dir, fmt, *, out=None, cameras=None, epoch_ns=None,
       t0_ns=None, t1_ns=None, overwrite=False, threads=4, bundle=None, **options)
```

Convert completed observations to a standard container or calibration format;
return the output directory `Path`. Formats are `raw-evt3`, `raw-evt2`,
`raw-evt21`, `dat`, `aedat4`, `rosbag1`, `mcap` and `kalibr`. `out=None` selects
`exports/<fmt>/` and `cameras=None` selects the run's event cameras.
The optional native `_export` module must be built.

`t0_ns` and `t1_ns` are true-clock bounds mapped to each camera clock before
selection. Omitting `t1_ns` preserves the stored readout tail. `epoch_ns` must be
a multiple of 1000; its default is zero except for ROS containers, which use one
second. `threads` is in `[1, 256]`. An existing destination requires
`overwrite=True` and an `EXPORT.json`; the run directory and its ancestors
cannot be replaced.

Use `bundle` for Kalibr export when the original run supplied its scene in
memory. Other formats read observations directly. Invalid options, incompatible
clocks and format limits raise `ExportError` before final output is committed.

### Stream and encoding options

`imu` and `frames` default to `True` and include available streams where the
format supports them. `topics` contains `STREAM:CAMERA=NAME` overrides for ROS
or Kalibr. `event_msg="dvs"` selects nanosecond event messages in rosbag1;
`"evt3"` selects microsecond EVT packets and is required for MCAP.
`packet_us` defaults to 1000 for ROS and 10000 for AEDAT4. Packet size changes
message grouping, rather than event generation.

`compression` selects a format-supported codec. Defaults are bz2 for rosbag1
and zstd for MCAP/AEDAT4; RAW, DAT and Kalibr accept only `none`.
`evt2_time_high="openeb"` follows OpenEB's dense TIME_HIGH policy;
`"change"` selects the more compact EVT 2.0 policy with keepalives.

### Image and calibration options

`frame_source="auto"` chooses `aps_dn`, then `rgb`, then `mono`; explicit
choices are `aps`, `rgb` and `mono`. `frame_map="none"` preserves supported
pixel types. `linear8` or `srgb8` normalizes, scales by `frame_scale` and clips to
8-bit; sRGB also applies its transfer curve. AEDAT4 frames require 8-bit mapping.
`frame_scale` defaults to `1.0` and changes stored display values, not photometry.

`imu_covariance="model"` uses additive white-noise variance; `"zero"` records
unknown covariance. `kalibr_inflate=1.0` leaves configured continuous-time noise
parameters unchanged; a larger value scales exported density and random-walk
parameters. It does not rerun or add noise to IMU measurements.

## Native export interfaces

`ballasim._export` encodes caller-supplied ordered arrays. Its writers do not
perform the high-level driver's scene checks, pre-scanning, cross-stream
interleaving or atomic directory handling. Use `export` for recorded runs;
use native writers when implementing a container integration.

Event columns are exact contiguous `uint16 x/y`, `uint8 p` and `int64 t_ns`.
IMU values are `float64 (N, 3)`. Native `close()` methods finish packet/index
state and return `path`, `bytes`, `sha256` and stream counts/times. Call close
after the last append to obtain a complete container.

### `_export.RawWriter`

`RawWriter(path, format, width, height, origin_us, epoch_ns,
evt2_time_high="openeb", t_ns_rem_min=0, t_ns_rem_max=0, version="")` writes one
camera's `evt2`, `evt21` or `evt3` stream. `append(x, y, p, t_ns)` accepts events
in nondecreasing reported time. Dimensions are at most 2048; timestamps become
microseconds relative to `origin_us`. Pre-scanned remainder bounds record the
precision discarded by the format.

### `_export.DatWriter`

`DatWriter(path, width, height, origin_us, epoch_ns, t_ns_rem_min=0,
t_ns_rem_max=0, version="")` exposes the same `append` and `close` lifecycle
for DAT. Dimensions are at most 16384 and relative microsecond timestamps must
fit `uint32`. It preserves event order and duplicates but cannot retain
nanosecond remainders.

### `_export.ImuCsvWriter`

`ImuCsvWriter(path, origin_us, epoch_ns, counts_gyro, counts_accel)` writes the
RAW/DAT inertial sidecar. `append(t_ns, gyro, accel, counts_gyro=None,
counts_accel=None)` requires `int32 (N, 3)` count arrays exactly for channels
enabled in the constructor. Physical values use rad/s and m/s² and retain
round-trip float64 text; timestamps use the shared relative microsecond origin.

### `_export.Rosbag1Writer`

`Rosbag1Writer(path, compression, origin_us, epoch_ns, threads=4,
chunk_bytes=786432)` writes a ROS1 bag. `connection(topic, msg)` registers a
`dvs`, `evt3`, `imu` or `image` stream and returns its handle. `event_arrays`
writes nanosecond `dvs_msgs/EventArray` events; `event_packets_evt3` writes
microsecond EVT payloads. `imu` and `images` write measurement streams.

Event calls take `(conn, frame_id, width, height, x, y, p, t_ns, packet_ns)`;
`packet_ns` fixes message grouping on the microsecond grid. Header and bag-record
time are the first event's mapped stamp. `compression` accepts `none`, `bz2` or
ROS-compatible `lz4`; compression worker count does not change container bytes.

### `_export.McapWriter`

`McapWriter(path, compression, origin_us, epoch_ns, threads=4,
chunk_bytes=1048576, library="BallaSim")` writes one ROS2-profile MCAP file.
`channel(topic, msg)` registers `evt3`, `imu` or `image`; subsequent
`event_packets_evt3`, `imu` and `images` calls use that channel handle.
Log/publish time equals the mapped message stamp. Event payloads retain the
EVT microsecond precision even though message stamps use nanoseconds.
Compression is `none`, `lz4` or `zstd`.

### `_export.Aedat4Writer`

`Aedat4Writer(path, compression, origin_us, epoch_ns, threads=4,
packet_us=10000)` writes event, IMU and frame streams in one AEDAT4 file.
Declare streams with `add_events_stream(camera, width, height)`,
`add_imu_stream(camera)` or `add_frame_stream(camera, width, height)` before
writing items. Methods `events`, `imu` and `frames` take the returned stream ID.

Dimensions are at most 32767. Input IMU SI values are converted to degrees/s
and g for the container; frame data must be mapped to 8-bit. Exposure midpoint
is stored as the frame timestamp, while the index uses exposure start, matching
dv-processing's frame lookup. Compression accepts `none`, `lz4`, `lz4-high`,
`zstd` or `zstd-high`.

### Native image and IMU append methods

ROS `imu(handle, frame_id, t_ns, gyro, accel, covariance="model",
white_var=(0.0, 0.0))` writes one sample message per row. `white_var` supplies
gyro and accelerometer additive-noise variances. AEDAT4 uses
`imu(stream, t_ns, gyro, accel)` and converts physical units itself.

ROS `images` and AEDAT4 `frames` accept a handle, optional ROS `frame_id`, five
sensor-clock time arrays (midpoint, frame start/end, exposure start/end), and
`pixels: (N, H, W[, 3])`. Keyword options `bits=0`, `x0=0`, `y0=0`,
`exposure_ns=-1`, `frame_map="none"` and `scale=1.0` specify ADC significance,
ROI origin, per-row exposure and image conversion. Time order and image layout
are validated before writing.

### `_export.frame_map8`

`frame_map8(pixels, bits=0, frame_map="none", scale=1.0, bgr=False)` returns
`uint8 (N, H, W[, 3])` using the writers' normalization, clipping and optional
sRGB transfer curve. `uint16` input uses `bits` significant bits (`0` means 16).
NaN maps to zero; `bgr=True` reorders RGB for AEDAT4. This is a display conversion,
not a calibrated exposure or light-transport operation.

### `_export.camera_times`

`camera_times(clock_offset_ns, clock_drift_ppm, readout_delay_s,
readout_quantum_s, t_end_ns, t0_ns=None, t1_ns=None)` returns reported selection
bounds and the file's readout-tail bound. Missing selection endpoints remain
open. The tail bound includes the sensor readout delay and quantum rounded as
the event model rounds them.

### `_export.write_kalibr`

`write_kalibr(imu_path, camchain_path, imu, readout_delay_s, inflate, model,
dist, fx, fy, cx, cy, width, height, camera, imu_topic, image_topic, version)`
writes IMU and camera-chain YAML and returns resolved calibration `values` and
two file records. Camera calibration uses radtan or equidistant distortion;
nonzero radtan k3 is refused because Kalibr's radtan model has four coefficients.
The time-shift estimate excludes illumination-dependent photoreceptor latency.

### `_export.ros1_msgdefs` and `_export.ros2_msgdefs`

`ros1_msgdefs()` returns message type names, MD5s and definitions for ROS1
writers. `ros2_msgdefs()` returns type names and schema text for MCAP.
These definitions support independent readers and container validation; they
are not a running ROS interface.

### `_export.compress`, `_export.decompress` and `_export.decompress_upto`

`compress(data, codec, lz4_frame="standard")` returns one self-contained codec
frame. `_export.CODECS` lists `none`, `lz4`, `lz4-high`, `zstd`, `zstd-high` and
`bz2`; the ROS LZ4 variant is selected by `lz4_frame="roslz4"`.
`decompress(data, codec, size)` requires the exact decoded size;
`decompress_upto(data, codec, max_size)` permits any size up to a limit. Both
reject corrupt, trailing or inconsistent data rather than silently ignoring it.

### `_export.crc32`

`crc32(data, crc=0)` computes or continues the MCAP-compatible CRC-32:
`crc32(b, crc32(a)) == crc32(a+b)`. It is a container-integrity primitive,
separate from the SHA-256 recorded for completed export files.
