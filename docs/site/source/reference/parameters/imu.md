# IMU

The IMU samples the camera rig's continuous trajectory and publishes angular
rate and specific force in IMU axes. Its mount is composed as
`T_rig_imu = T_rig_cam * T_cam_imu`; the accelerometer includes tangential and
centripetal acceleration from the mount's lever arm.

Set `cameras.<camera>.imu.*`, or use `cameras.default.imu.*` for shared settings.
The default disabled IMU contributes no output. Enabling it without additional
values selects an ideal device: no noise, calibration error, filtering,
quantization or saturation. It still uses the configured sample phase and
camera timestamp counter.

Use [IMU output](../../guide/outputs.md#imu) for the file contract and
{ref}`simulate_imu <simulate-imu>` for post-processing a completed run.

## Sampling and timestamps

The sampling clock determines when the physical signal is measured. The camera
clock determines the reported timestamp. `imu.clock_adev` changes the first;
`sensor.clock_offset_ns`, `sensor.clock_drift_ppm` and the shared camera counter
change the second. Filter delay changes the measured signal, rather than moving
its timestamp backward.

| Parameter | Type / unit | Default |
| --- | --- | --- |
| {ref}`imu.enabled <imu-enabled>` | `bool` | `false` |
| {ref}`imu.rate_hz <imu-rate-hz>` | `float`, `(0, 100000]` Hz | `1000` |
| {ref}`imu.start_phase <imu-start-phase>` | `random` or `zero` | `random` |
| {ref}`imu.clock_adev <imu-clock-adev>` | `float`, `[0, 0.01)` | `0` |
| {ref}`imu.T_cam_imu <imu-t-cam-imu>` | row-major `float[16]` | Identity |
| {ref}`imu.gravity_mps2 <imu-gravity-mps2>` | nullable `float[3]`, m/s² | `null` |
| {ref}`imu.stamp_offset_ns <imu-stamp-offset-ns>` | `int`, `(-10000000, 10000000)` ns | `0` |
| {ref}`imu.noise_mode <imu-noise-mode>` | `white` or `shaped` | `white` |
| {ref}`imu.clip_stage <imu-clip-stage>` | `output` or `both` | `output` |
| {ref}`imu.adc_bits <imu-adc-bits>` | `int`, `[8, 32]` | `16` |
| {ref}`imu.seed <imu-seed>` | `int ≥ 0` | `1` |

(imu-enabled)=
### Enable output (`imu.enabled`)

Generate `imu/<camera>.h5` after a successful event run. IMU values are output
settings and do not change rendered signals or events. `ballasim imu RUN` can
also enable or regenerate the stream from a completed run. An IMU failure is
recorded separately and does not change the event run's completed status.

(imu-rate-hz)=
### Sampling rate (`imu.rate_hz`)

Set the nominal rate on the true clock. With an ideal sample clock, successive
samples are separated by `1/rate_hz` seconds. Raising the rate samples the same
trajectory more densely; it does not improve the fitted motion's bandwidth.
White per-sample noise increases with the square root of rate for a fixed
continuous-time noise density. Rates outside 50–8000 Hz produce a soft warning.

(imu-start-phase)=
### First sample phase (`imu.start_phase`)

`random` places the first sample at `t_start + U/rate_hz`, with one seeded
uniform draw per camera. `zero` places it at the run start. Use `zero` for a
known synchronized sample phase; use `random` when the IMU free-runs relative to
the requested run interval.

(imu-clock-adev)=
### Sample-clock variation (`imu.clock_adev`)

Set the approximate fractional-frequency Allan-deviation plateau of the sample
clock. `0` gives uniform sampling. The model sums three stationary Gauss–Markov
processes with fixed time constants to approximate flicker frequency noise over
roughly 0.1–10 s. It is not a general oscillator phase-noise model and does not
set a constant clock drift; reported drift remains a camera sensor setting.

(imu-t-cam-imu)=
### Mount transform (`imu.T_cam_imu`)

Map IMU coordinates into camera coordinates: `X_cam = T_cam_imu * X_imu`.
Translation is in meters; rotation defines the gyro/IMU axes. A displaced mount
changes specific force through rotational lever-arm terms. The transform must
be rigid, with last row exactly `[0, 0, 0, 1]`; accepted rotation rounding is
removed by polar re-orthonormalization and recorded in the IMU report.

(imu-gravity-mps2)=
### Gravity (`imu.gravity_mps2`)

Specify the world-frame gravity vector. `null` uses bundle
`gravity_world_mps2`, otherwise `[0, 0, -9.80665]` m/s². The accelerometer reports
specific force, `R_world_imu.T * (acceleration - gravity)`, including lever-arm
acceleration. A stationary level device therefore measures gravity's opposite.
EVIMO2 uses a tabletop world frame; provide gravity in that frame rather than
assuming negative world Z.

(imu-stamp-offset-ns)=
### Timestamp latch delay (`imu.stamp_offset_ns`)

Add this signed delay to the true sample instant before camera-counter
quantization, offset and drift. It models a timestamp-latch offset, independently
of the gyro or accelerometer filter delay. Positive values report the sample
later. Samples stamped before the event file's origin are omitted and counted;
the internal sensor processes still advance through them.

(imu-noise-mode)=
### Noise spectrum (`imu.noise_mode`)

`white` draws independent noise with standard deviation `sigma/sqrt(dt)` at each
realized sample interval. `shaped` uses the stationary output of white noise
through the configured first-order low-pass model, giving variance
`sigma**2/(2*tau_c)` and correlation time `tau_c`. Shaped mode requires
`filter: first_order` on both gyro and accelerometer. Bias processes are added
separately in either mode.

(imu-clip-stage)=
### Saturation placement (`imu.clip_stage`)

`output` saturates only the ADC counts. `both` also clips the calibrated signal
at the physical rail before low-pass filtering, modeling recovery from an
analog overload. A channel needs `lsb` to define its rail; an unquantized channel
remains unsaturated. `both` is refused when neither channel has an LSB.
Pre-filter clipping does not clip the subsequent additive noise and bias terms.

(imu-adc-bits)=
### ADC width (`imu.adc_bits`)

Set the signed count range
`[-2**(adc_bits-1), 2**(adc_bits-1)-1]` for each quantized channel. Physical range
is count range multiplied by that channel's `lsb`. Increasing bits with a fixed
LSB expands the range; it does not reduce the quantization step. Channels with
`lsb: null` ignore this setting.

(imu-seed)=
### Random realization (`imu.seed`)

Select turn-on bias, measurement noise, bias drift, start phase and clock
variation. Philox draws are keyed by camera and sample index, so output does
not depend on write chunk size or event-simulation order. Changing one noise
amplitude retains the underlying standard-normal realization, which supports
paired comparisons of device settings.

## Sensor calibration and noise

Each suffix below exists under both `imu.gyro.*` and `imu.accel.*`. Gyroscope
output uses rad/s; accelerometer output uses m/s². Calibration maps an ideal
IMU-frame signal `x` through

```{math}
M=\operatorname{Exp}(\phi)\left(I+\operatorname{diag}(s)+
\begin{bmatrix}0&n_{xy}&n_{xz}\\n_{xy}&0&n_{yz}\\n_{xz}&n_{yz}&0\end{bmatrix}\right).
```

The fixed turn-on bias enters with `M*x` before optional clipping and filtering.
Bias random walk, bias instability and measurement noise are added after the
filter. Axes have independent noise processes; correlated cross-axis noise and
temperature-dependent calibration are not modeled by these settings.

| Suffix | Gyroscope default / unit | Accelerometer default / unit |
| --- | --- | --- |
| {ref}`noise_density <imu-gyro-noise-density>` | `0`, rad/s/√Hz | `0`, m/s²/√Hz |
| {ref}`random_walk <imu-gyro-random-walk>` | `0`, rad/s²/√Hz | `0`, m/s³/√Hz |
| {ref}`bias_instability <imu-gyro-bias-instability>` | `0`, rad/s | `0`, m/s² |
| {ref}`bias_instability_tau_s <imu-gyro-bias-instability-tau-s>` | `10`, s | `10`, s |
| {ref}`bias <imu-gyro-bias>` | `[0, 0, 0]`, rad/s | `[0, 0, 0]`, m/s² |
| {ref}`bias_sd <imu-gyro-bias-sd>` | `[0, 0, 0]`, rad/s | `[0, 0, 0]`, m/s² |
| {ref}`scale <imu-gyro-scale>` | `[0, 0, 0]`, fractional | `[0, 0, 0]`, fractional |
| {ref}`nonorth <imu-gyro-nonorth>` | `[0, 0, 0]`, fractional | `[0, 0, 0]`, fractional |
| {ref}`misalign_deg <imu-gyro-misalign-deg>` | `[0, 0, 0]`, degrees | `[0, 0, 0]`, degrees |
| {ref}`filter <imu-gyro-filter>` | `none` | `none` |
| {ref}`cutoff_hz <imu-gyro-cutoff-hz>` | `92`, Hz | `99`, Hz |
| {ref}`group_delay_s <imu-gyro-group-delay-s>` | `0.0039`, s | `0.00288`, s |
| {ref}`lsb <imu-gyro-lsb>` | `null`, rad/s/count | `null`, m/s²/count |

(imu-gyro-noise-density)=
(imu-accel-noise-density)=
### White-noise density

`imu.gyro.noise_density` and `imu.accel.noise_density` set nonnegative
continuous-time densities in the Kalibr convention. In white mode,
`n_k = sigma*z_k/sqrt(dt_k)`; in shaped mode, the same density drives the
low-pass noise process. Increasing sample rate therefore increases independent
per-sample RMS in white mode. Convert one-sided datasheet spectral densities
to the intended convention before using them.

(imu-gyro-random-walk)=
(imu-accel-random-walk)=
### Bias random walk

`imu.gyro.random_walk` and `imu.accel.random_walk` set nonnegative diffusion
amplitudes for a bias starting at zero:
`b_k = b_(k-1) + sigma_b*sqrt(dt_k)*z_k`. This slowly accumulating offset is
separate from the fixed turn-on bias. Its Allan deviation has the expected
positive half-slope; `sigma_b` equals that curve's value at 3 s under the
continuous model.

(imu-gyro-bias-instability)=
(imu-accel-bias-instability)=
### Bias instability

`imu.gyro.bias_instability` and `imu.accel.bias_instability` set nonnegative
coefficients `B`. The simulator uses a stationary Gauss–Markov surrogate with
standard deviation `0.664*B/0.617`. Its Allan deviation reaches `0.664*B` near
the requested averaging time. This approximates a local Allan plateau; a single
Gauss–Markov process does not reproduce broadband flicker bias at all timescales.

(imu-gyro-bias-instability-tau-s)=
(imu-accel-bias-instability-tau-s)=
### Bias-instability timescale

`imu.gyro.bias_instability_tau_s` and `imu.accel.bias_instability_tau_s` specify
the Allan averaging time `tau_B`, in `(0, 100000]` s. The underlying process
correlation time is `T = tau_B/1.89`. The two times are not interchangeable:
`tau_B` places the Allan peak/plateau approximation, whereas `T` governs
exponential autocorrelation. The value has no effect when `B=0`.

(imu-gyro-bias)=
(imu-accel-bias)=
### Fixed turn-on bias

`imu.gyro.bias` and `imu.accel.bias` set finite three-axis offsets in the sensor's
output units. The realized constant is `bias + bias_sd*z` and enters before
the filter. Use this term for deterministic calibration offsets; use
`random_walk` or `bias_instability` for time-varying bias.

(imu-gyro-bias-sd)=
(imu-accel-bias-sd)=
### Turn-on bias spread

`imu.gyro.bias_sd` and `imu.accel.bias_sd` give nonnegative three-axis standard
deviations for the random turn-on bias. One Gaussian draw per axis is held for
the entire simulated run. This changes device-to-device or start-to-start
offsets, rather than adding new measurement noise at each sample.

(imu-gyro-scale)=
(imu-accel-scale)=
### Scale error

`imu.gyro.scale` and `imu.accel.scale` set fractional gain errors on the three
axes, each in `(-0.5, 0.5)`. `0.01` multiplies the corresponding ideal component
by `1.01` before the remaining calibration rotation and fixed turn-on bias.
Additive output noise is not multiplied by this gain.

(imu-gyro-nonorth)=
(imu-accel-nonorth)=
### Axis non-orthogonality

`imu.gyro.nonorth` and `imu.accel.nonorth` set symmetric cross-axis couplings
`[xy, xz, yz]`, each in `(-0.1, 0.1)`. An `xy` entry couples both X into Y and
Y into X in the symmetric calibration factor. These values are dimensionless
couplings, not Euler angles.

(imu-gyro-misalign-deg)=
(imu-accel-misalign-deg)=
### Sensor-axis misalignment

`imu.gyro.misalign_deg` and `imu.accel.misalign_deg` specify a rotation vector
in degrees, with components in `(-10, 10)`. The exponential map rotates the
scaled, non-orthogonal ideal signal; it is not an XYZ Euler sequence. The IMU
frame is conventionally the gyro frame, so gyro misalignment is normally zero
and accelerometer misalignment expresses the relative sensor orientation.

## Filtering and ADC

The first-order device approximation has transfer function

```{math}
H(s)=\frac{e^{-d s}}{1+\tau_c s},\qquad
\tau_c=\frac{1}{2\pi f_c},\qquad d=\tau_{\mathrm{group}}-\tau_c.
```

This separates bandwidth from total DC group delay. It approximates a device
DLPF rather than reproducing a manufacturer's internal digital filter exactly.
The default frequencies and delays are MPU-9250 DLPF_CFG 2 priors and remain
inactive until the corresponding filter is enabled.

(imu-gyro-filter)=
(imu-accel-filter)=
### Filter model

`imu.gyro.filter` and `imu.accel.filter` select `none` or `first_order`.
`none` evaluates the ideal signal at each sample instant. `first_order`
integrates the delayed, calibrated input continuously between samples. When
trajectory history is insufficient, startup samples carry WARMUP; constant
inputs remain exact. Piecewise velocity changes generate the corresponding
filtered accelerometer impulse and IMPACT flags.

(imu-gyro-cutoff-hz)=
(imu-accel-cutoff-hz)=
### Filter bandwidth

`imu.gyro.cutoff_hz` and `imu.accel.cutoff_hz` set positive −3 dB frequencies.
Lower values suppress faster signal changes and reduce shaped-noise variance.
An active cutoff must be below `imu.rate_hz`; at or above half the sample rate
it produces an aliasing warning. The model does not synthesize aliased
high-frequency noise beyond its stated approximation.

(imu-gyro-group-delay-s)=
(imu-accel-group-delay-s)=
### DC group delay

`imu.gyro.group_delay_s` and `imu.accel.group_delay_s` specify total low-frequency
delay in nonnegative seconds, including the first-order pole's `tau_c`. An active filter
requires `group_delay_s ≥ tau_c`; the remainder becomes a nonnegative pure delay.
Changing this value delays the measurement response without changing the
sample's reported timestamp or the separate `stamp_offset_ns`.

(imu-gyro-lsb)=
(imu-accel-lsb)=
### ADC quantum

`imu.gyro.lsb` and `imu.accel.lsb` set positive physical units per signed count.
`null` disables both quantization and saturation for that channel. Otherwise,
values are rounded to nearest integer, ties away from zero, and clamped to the
`adc_bits` range; published measurements equal `count*lsb`. A smaller LSB
improves resolution while reducing physical range at fixed bit depth.

## Profiles and interpretation

The opt-in profiles `evimo2/prophesee_gen3_imu_left@v1` and
`evimo2/prophesee_gen3_imu_right@v1` combine measured EVIMO2 values with stated
priors and choices. They are not included in the event-sensor profile sets.
Their stochastic-process values and gravity assumptions require the same unit
and frame conventions as the explicit parameters above.

IMU output contains the simulated truth and separate bias arrays. ROS covariance
export includes additive white/shaped noise only; it excludes bias processes,
calibration error and quantization. Use those fields to distinguish configured
measurement uncertainty from a complete error model.

## References

- [Kalibr IMU Noise Model](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model): continuous-time density and bias-random-walk conventions.
- [Kalibr YAML formats](https://github.com/ethz-asl/kalibr/wiki/yaml-formats): `T_cam_imu` and exported calibration fields.
- [InvenSense MPU-9250 register map](https://cdn.sparkfun.com/assets/learn_tutorials/5/5/0/MPU-9250-Register-Map.pdf): device filter terminology. The configured 92/99 Hz and 3.9/2.88 ms defaults follow the later revision 1.6 tables recorded in the repository; earlier accelerometer tables differ.

```{toctree}
:hidden:
:maxdepth: 1

IMU-generation methods <../api/imu>
```
