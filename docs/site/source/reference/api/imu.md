# IMU generation methods

The [IMU settings](../parameters/imu.md) describe motion truth, calibration,
clock, noise, filters, and digitization. These methods generate and inspect
the corresponding observations.

## IMU

(simulate-imu)=
### `simulate_imu`

```python
from ballasim.export.imu import simulate_imu

simulate_imu(run_dir, *, cameras=None, sets=(), profiles=(),
             camera_profiles=None, bundle=None, overwrite=False)
```

Sample a completed run's rig trajectory and write one IMU HDF5 file and JSON
record per enabled camera. Return the HDF5 `list[Path]`. `cameras=None` selects
cameras with resolved `imu.enabled=True`; explicitly selected cameras must also
be enabled. The call leaves `run.json` and event data unchanged.

Overlays apply in order: root `profiles`, camera-mounted `camera_profiles`, then
`sets`. Camera profiles map names to an ID or sequence of IDs. `sets` accepts
full `KEY=VALUE` strings or a mapping; only `imu.*` keys may change. Supply a
matching `bundle` when the original run used an in-memory scene.
`overwrite=True` replaces only selected cameras' IMU files.

The [IMU model](../parameters/imu.md) defines rigid-body truth, calibration,
noise, delay and quantization. A per-camera failure removes its partial data and
writes a failed record. Refusals raise `ballasim.export.ExportError`.

(imu-truth)=
### `_imu.imu_truth`

```python
imu_truth(trajectory, T_rig_cam, T_cam_imu, g_W, t_ns)
```

Evaluate ideal rigid-body inertial truth without device errors. `trajectory`
is the rig's bundle trajectory as a dictionary, including optional `pieces`;
`t_ns` is an exact `int64 (N,)` array. Extrinsics are `float64 (4, 4)` and
`g_W` is a world-frame gravity vector. Return `omega`, `f`, `q` and `p`, where
angular rate and specific force are in IMU axes, and pose is IMU-to-world.

The acceleration includes angular acceleration and centripetal lever-arm terms.
Quaternions use `(w, x, y, z)`. Queries outside trajectory support are refused.

(imu-simulator)=
### `_imu.ImuSimulator`

```python
ImuSimulator(trajectory, T_rig_cam, imu, clock, g_W, camera_index,
             t_begin_ns, t_end_ns, t_offset_us)
```

Construct a stateful device simulation over `[t_begin_ns, t_end_ns)` inside rig
support. `imu` contains every resolved `imu.*` key, obtainable from
`default_values()`. `clock` is the camera counter mapping; `camera_index` is the
bundle index and must be below 256. `t_offset_us` is the event-file origin.

Calibration, optional continuous filtering, bias processes and quantization are
applied in the order documented under [IMU parameters](../parameters/imu.md).
Piecewise rigs include C0 accelerometer impulses, C1 cuts and filter resets at
pose steps. This inertial evaluation can represent steps even though an event
run rejects pose discontinuities.

#### `ImuSimulator.advance_to`

`advance_to(t_end_ns)` returns the next samples before the requested true time,
without exceeding the configured end. Advancing in several chunks gives the
same concatenated samples as one call. Stamps earlier than
`1000*t_offset_us` are dropped after updating internal processes.

| Result | Type / shape | Meaning |
| --- | --- | --- |
| `t_true_ns`, `t_ns` | `int64 (N,)` | Physical sample instant and camera-clock stamp. |
| `gyro`, `accel` | `float64 (N, 3)` | Published rad/s and m/s² values. |
| `counts_gyro`, `counts_accel` | `int32 (N, 3)`, optional | Signed ADC counts when that sensor has an LSB. |
| `omega_true`, `f_true` | `float64 (N, 3)` | Ideal truth before calibration and device errors. |
| `bias_gyro`, `bias_accel` | `float64 (N, 3)` | Turn-on plus random-walk and instability biases. |
| `q_W_I`, `p_W_I` | `(N, 4)`, `(N, 3)` | Scalar-first quaternion and IMU world position. |
| `flags` | `uint8 (N,)` | Rail bits, WARMUP and IMPACT. |

#### `ImuSimulator.stats`

`stats()` returns dropped-origin, warm-up, impact and step-reset counts, plus
per-axis pre-filter clipping counts. These diagnostics distinguish initialization
and trajectory-join effects from ordinary stochastic noise.

(imu-default-values)=
### `_imu.default_values`

`default_values()` returns all 37 `imu.*` values for an ideal, disabled IMU.
Use the complete mapping as the base for native simulations; missing, unknown
or mistyped keys are refused rather than filled implicitly.

(imu-camera-clock)=
### `_imu.camera_clock`

`camera_clock(readout_quantum_s, clock_offset_ns, clock_drift_ppm,
sensor_seed, camera_index)` constructs the same timestamp counter phase used by
the event camera. Return `quantum_ns`, `phase_ns`, `offset_ns` and `drift_ppm`.
The camera sensor seed determines counter phase; `imu.seed` controls the IMU
sample clock and device noise separately.

(imu-sensor-clock)=
### `_imu.sensor_clock_of`

`sensor_clock_of(clock)` returns a copy with zero counter quantum and phase.
Offset and drift remain. Use it for frame-clock mapping when counter rounding
would incorrectly quantize exposure times.

(imu-reported-time)=
### `_imu.reported_time`

`reported_time(clock, delay_ns, t_true_ns)` adds delay, floors to the camera's
free-running counter and then applies offset and drift. `t_true_ns` is an exact
`int64` array. A timestamp overflow or invalid counter configuration raises
`ExportError`; no floating-point timestamp truncation is performed.

(imu-white-variance)=
### `_imu.white_variance`

`white_variance(imu)` returns `(gyro_variance, accel_variance)` for additive
measurement noise: `sigma**2*rate_hz` in white mode, or
`sigma**2/(2*tau_c)` in shaped mode. ROS covariance export uses these values.
They exclude bias processes, calibration errors and quantization, and therefore
are not a complete measurement-error covariance.

(imu-resolve)=
### `_imu.resolve_imu`

`resolve_imu(recorded, root_profiles, camera_profiles, sets, profile_docs,
cameras)` applies an IMU-only overlay in the native resolver. Return per-camera
value/provenance records, warnings, digests and mount rotation corrections.
`profile_docs` supplies profile documents already read by the caller. For normal
file workflows, `simulate_imu` performs the necessary profile and scene handling.

(imu-check-format)=
### `_imu.check_format`

`check_format(value, expected=FILE_FORMAT)` accepts the same format name and
major version, with or without a minor suffix. Other names or major versions
raise `ExportError`. `_imu.FILE_FORMAT` is `ballasim.imu/1` and
`RUN_FORMAT` is `ballasim.imu_run/1`.
