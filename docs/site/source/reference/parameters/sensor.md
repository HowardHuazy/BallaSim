# Event sensor

Camera scope: `cameras.<camera>.sensor.*`. Set shared values under
`cameras.default.sensor`. Natural-log intensity units are denoted `log-e`.
Family defaults are listed in [Sensor families](families.md).

The entries below explain each setting and its interaction with the pixel
model; the [quick reference](#quick-reference) lists types, defaults and ranges.
For the signal chain and modeling assumptions, see
[Event sensors](../../guide/sensors.md).

## Parameter behavior

### Response and bandwidth

(sensor-family)=
#### `sensor.family`

Selects the prior table used to fill unspecified sensor values. It must be
explicit even when the scene names a camera model. `prophesee_gen3` and
`samsung_gen3` are approximate device families; `ideal` is a noise-free reference.
See [Family selection](families.md#family-selection) before using their defaults
as a device calibration.

(sensor-photoreceptor)=
#### `sensor.photoreceptor`

`cascade` models a nonlinear photoreceptor, output node and source follower.
`instantaneous` applies log current directly and bypasses all three stages and
their noise. Reset, threshold mismatch, leak and readout remain independently
configurable; changing this field alone does not create the full `ideal` family.

(sensor-k)=
#### `sensor.k`

Sets the input-stage response rate `a_input=k*I`, with total current `I` in
amperes. Higher `k` reduces input-stage lag but also raises stationary input-node
log-noise variance `k*q_e`. It cannot tune response speed independently of shot
noise; downstream time constants and photocurrent must be considered together.

(sensor-tau-o)=
#### `sensor.tau_o`

Sets the output-node low-pass time constant. Larger values reduce bandwidth and
increase analog lag; this stage also receives the `sigma_o` noise source.
Its pole is independent of illumination in the adopted cascade. Validation
requires it to remain separated from `tau_sf`.

(sensor-tau-sf)=
#### `sensor.tau_sf`

Sets the source-follower low-pass time constant after the output node. Increasing
it smooths signal changes and upstream noise before the comparator. This is a
fixed downstream bandwidth limit, unlike the current-dependent input pole.

(sensor-sigma-o)=
#### `sensor.sigma_o`

Sets the stationary log-noise amplitude of the output node before source-follower
filtering. Increasing it raises temporal comparator noise even in bright light.
Zero removes this noise source while retaining input shot noise when `noise`
is enabled; it does not remove threshold mismatch.

(sensor-noise)=
#### `sensor.noise`

Enables the cascade's temporal noise processes. Disabling it leaves the
deterministic filters, dark current, fixed mismatch and leakage active. Hot
pixels then use the leak mechanism because threshold-type hot pixels require
signal-path noise. The switch is inactive for `instantaneous` response.

### Contrast, reset and initial phase

(sensor-theta-on)=
#### `sensor.theta_on`

Sets the mean positive log-current change required for an ON crossing. In the
instantaneous noise-free case, threshold `theta` corresponds to a current ratio
`exp(theta)`. A lower ON threshold increases positive-change sensitivity and
usually increases both signal events and noise crossings.

(sensor-theta-off)=
#### `sensor.theta_off`

Sets the magnitude of the negative log-current change required for an OFF
crossing. An ideal OFF crossing corresponds to a ratio `exp(-theta_off)`.
Different ON and OFF means model polarity asymmetry; both must exceed
`theta_floor`.

(sensor-sigma-theta-on)=
#### `sensor.sigma_theta_on`

Sets the standard deviation of the frozen Gaussian ON-threshold map. It changes
spatial sensitivity, rather than adding event-to-event jitter. Increasing it
creates more low-threshold pixels as well as less-sensitive pixels; drawn
thresholds are clamped to `theta_floor`.

(sensor-sigma-theta-off)=
#### `sensor.sigma_theta_off`

Sets the corresponding frozen OFF-threshold spread. ON and OFF spreads can be
different. Their joint variation is controlled by `theta_onoff_corr`, and zero
spread makes that polarity's threshold uniform outside hot-pixel overrides.

(sensor-theta-onoff-corr)=
#### `sensor.theta_onoff_corr`

Correlates the Gaussian ON and OFF mismatch draws of a pixel. Zero gives
independent draws; positive values tend to make both thresholds high or low
together, while negative values favor opposite deviations. It describes fixed
pixel variation, not correlation between successive events.

(sensor-theta-floor)=
#### `sensor.theta_floor`

Clamps sampled thresholds away from zero, including thresholds assigned to
noise-driven hot pixels. This bounds pathological sensitivity in distribution
tails. Changing the floor can therefore affect hot-pixel rates even when the
mean thresholds remain fixed.

(sensor-reset-mode)=
#### `sensor.reset_mode`

`physical` stores the signal level at refractory release as the new reference,
plus `eps_ci`; changes during the blind interval are lost. `ideal` adds the
signed crossed threshold to the existing reference, preserving residual
contrast. The ideal mode is a simulator convention rather than the adopted
physical reset model.

(sensor-refractory-s)=
#### `sensor.refractory_s`

Sets the interval in which a pixel cannot report another crossing. Longer
intervals limit event rate and suppress changes arriving during physical reset.
Under ideal reset it remains a dead time but does not replace the threshold
increment rule. Zero removes this interval where the selected model permits it.

(sensor-eps-ci)=
#### `sensor.eps_ci`

Adds a charge-injection offset to the reference at physical refractory release.
A positive offset initially moves the comparator toward the OFF side; a negative
offset moves it toward ON. It is a reset-model correction, not a timestamp
delay, and must satisfy `abs(eps_ci) < theta_floor/2`.

(sensor-initial-phase)=
#### `sensor.initial_phase`

Chooses the initial comparator-reference offset: `uniform` samples the ON
interval, `symmetric` samples the OFF–ON interval, and `zero` aligns the
reference with the initial signal. Uniform phase uses the filtered leak-cycle
law when the high-pass filter is active. Stationary initialization can replace
this phase with a sampled comparator history.

### Dark current, leakage and high-pass response

(sensor-dark-current-fa)=
#### `sensor.dark_current_fA`

Sets the median positive dark current at the reference temperature. It is added
to signal current before the logarithmic response, reducing low-light contrast
and setting a finite input pole in darkness. Dark current is not itself an
independent stream of dark events.

(sensor-dark-sigma-ln)=
#### `sensor.dark_sigma_ln`

Sets the natural-log spread of the frozen dark-current map:
`I_dark,p = median * exp(sigma*Z)`, before temperature scaling. Zero gives the
same dark current to every pixel. The configured current is a median, so
increasing this spread also increases the map's arithmetic mean.

(sensor-dark-ea-ev)=
#### `sensor.dark_ea_eV`

Sets the activation energy in the adopted Arrhenius dark-current law:
`factor=exp[-Ea/k_B*(1/T-1/T_ref)]`. Positive energy makes dark current rise
with temperature. This phenomenological temperature law and its family prior
must be calibrated for a particular sensor.

(sensor-temperature-k)=
#### `sensor.temperature_K`

Sets the operating die temperature used to scale dark current and ordinary
leakage. Raising it affects low-light contrast and leak-driven activity through
their separate activation energies. The model does not simulate thermal
evolution or retune all analog biases with temperature.

(sensor-reference-temperature-k)=
#### `sensor.reference_temperature_K`

Defines the temperature at which `dark_current_fA` and `leak_rate_hz` are quoted.
Set it to the calibration temperature rather than changing the measured values
to compensate for a different operating temperature. Equal operating and
reference temperatures give unit Arrhenius factors.

(sensor-leak-rate-hz)=
#### `sensor.leak_rate_hz`

Sets the median nominal ON leak rate at the reference temperature. The model
turns it into reference drift `ell=theta_on,p*R_leak,p`; reset and high-pass
filtering determine the resulting event rate. It is not a total background
activity rate and must not include temporal-noise events.

(sensor-leak-sigma-log10)=
#### `sensor.leak_sigma_log10`

Sets the standard deviation of base-10 log leak rates across pixels. A draw
`Z` multiplies the nominal rate by `10**(sigma*Z)`. This persistent spread uses
decades, whereas dark-current and high-pass spreads use natural logarithms.

(sensor-leak-ea-ev)=
#### `sensor.leak_ea_eV`

Sets the ordinary leakage Arrhenius activation energy independently of the
dark-current energy. Larger values increase leakage's sensitivity to operating
temperature. Explicit hot-pixel target rates use their own assigned mechanism
and should not be interpreted as a calibrated thermal law.

(sensor-tau-hp)=
#### `sensor.tau_hp`

Sets the reference-relaxation time in `dr/dt=-ell+(x3-r)/tau_hp`. Smaller positive
values suppress slower illumination changes more strongly; zero disables the
filter. It can also prevent ordinary leak crossings when drift cannot reach
threshold. The model is behavioral: no universal bias-register-to-time-constant
conversion is assumed.

(sensor-tau-hp-sigma-ln)=
#### `sensor.tau_hp_sigma_ln`

Sets a frozen log-normal spread around `tau_hp`. Individual time constants are
clamped to `max(1 ms, 10*tau_o, 10*tau_sf)` to preserve the model's pole
separation. Zero gives a uniform time constant; the setting is inactive when
high-pass filtering is off.

### Hot-pixel population

(sensor-hot-fraction)=
#### `sensor.hot_fraction`

Sets the probability that a pixel belongs to the hot population. Membership is
drawn once from the sensor seed and full-frame pixel ID; it is not a fluctuating
event probability. Zero disables this population. Selected pixels still use
the comparator and reset model.

(sensor-hot-rate-law)=
#### `sensor.hot_rate_law`

Chooses the distribution of target hot-pixel rates. `log_uniform` assigns equal
probability to equal logarithmic intervals between the rate bounds;
`log_normal` draws around `hot_rate_median_hz` with natural-log spread and clamps
to those bounds. These are population assumptions, not universal device laws.

(sensor-hot-rate-min-hz)=
#### `sensor.hot_rate_min_hz`

Sets the lower target-rate bound. It is the lower endpoint for the log-uniform
law and the lower clamp for the log-normal law. Increasing it removes slower
members of the configured hot population without changing `hot_fraction`.

(sensor-hot-rate-max-hz)=
#### `sensor.hot_rate_max_hz`

Sets the upper target-rate bound for either law. It must not exceed what the
refractory interval allows: enabled hot pixels require
`hot_rate_max_hz*refractory_s < 1`. Actual threshold-type rates remain dependent
on current, filtering and the threshold floor.

(sensor-hot-rate-median-hz)=
#### `sensor.hot_rate_median_hz`

Sets the pre-clamp median of the log-normal target-rate distribution. It is
inactive for `log_uniform`. Increasing it shifts that population toward higher
activity; bounds can change the realized median when many draws are clamped.

(sensor-hot-rate-sigma-ln)=
#### `sensor.hot_rate_sigma_ln`

Sets natural-log rate spread for the log-normal hot population. Zero gives one
target rate, subject to the bounds; larger values produce a wider range and
more clamped draws. It is inactive for the log-uniform law.

(sensor-hot-leak-fraction)=
#### `sensor.hot_leak_fraction`

Splits selected hot pixels into leak-driven and noise-driven mechanisms.
Leak-type pixels receive reference drift; threshold-type pixels receive a low
threshold inferred from a stationary crossing-rate approximation. Without
temporal noise, every hot pixel becomes leak-type regardless of this fraction.

(sensor-hot-reference-current-fa)=
#### `sensor.hot_reference_current_fA`

Sets the current at which the noise-driven hot-pixel threshold is inferred from
its target rate. That rate is a calibration target at this current, not an
illumination-independent event generator. Changing current or bandwidth can
change the realized activity after thresholds are assigned.

### Readout and sensor clock

(sensor-readout-quantum-s)=
#### `sensor.readout_quantum_s`

Sets the counter-grid spacing. Delayed crossing times are floored to a
free-running grid with one phase drawn per camera; zero bypasses quantization.
Coarser spacing creates shared timestamps and loses timing precision without
changing analog threshold crossings.

(sensor-readout-delay-s)=
#### `sensor.readout_delay_s`

Adds a fixed reporting delay before counter quantization. It represents an
additional pipeline delay; analog latency already emerges from the filters.
With quantization enabled it must be at least one quantum so floor rounding
cannot report a crossing early.

(sensor-clock-offset-ns)=
#### `sensor.clock_offset_ns`

Adds a constant offset after readout quantization. The default comes from the
scene camera. Use it to express reported events on a calibrated sensor clock;
it does not move geometry, change crossings, or change ground-truth query time.

(sensor-clock-drift-ppm)=
#### `sensor.clock_drift_ppm`

Sets the linear clock-rate error: `t_sensor=t+offset+round(drift*1e-6*t)`,
where `t` is the delayed, quantized time. Positive drift makes reported intervals
longer. The drift acts from time zero, rather than restarting at the requested
run start.

### Event-rate limiting

(sensor-limiter)=
#### `sensor.limiter`

`epoch` applies a token bucket after crossings are sorted by reported time;
`off` keeps every event. Dropped events have already reset their pixel, so
limiting cannot be reproduced by lowering the sensor's sensitivity. It is a
driver approximation; it does not simulate a shared address-event bus.

(sensor-limiter-rate0-hz)=
#### `sensor.limiter_rate0_hz`

Sets the refill-rate cap at the beginning of each epoch. It is a camera-wide
event rate, not a per-pixel limit. The bucket permits bursts according to
`limiter_depth`; the initial bucket is full.

(sensor-limiter-rate1-hz)=
#### `sensor.limiter_rate1_hz`

Sets the cap just before the next epoch. The cap decreases linearly from
`rate0` to `rate1`, then resets to `rate0`; equal endpoints give a constant-rate
bucket. `rate1` must not exceed `rate0`.

(sensor-limiter-period-s)=
#### `sensor.limiter_period_s`

Sets the repeating interval of the rate-cap waveform on the sensor clock.
The rate resets each epoch, while bucket tokens carry continuously. It controls
periodic throughput structure without altering the pixel's analog evolution.

(sensor-limiter-depth)=
#### `sensor.limiter_depth`

Sets stored token capacity in events, allowing short bursts above the refill
rate. A shallow bucket can clip each timestamp bin, so validation requires
capacity greater than a saturated bin's refill. Increasing capacity preserves
larger bursts but does not raise the sustained refill rate.

(sensor-limiter-phase-s)=
#### `sensor.limiter_phase_s`

Sets the cap-reset position modulo the epoch on the sensor clock. `null` draws
one repeatable phase per camera; an explicit value aligns the limiter waveform
to known recording timing. It is independent of the timestamp counter's
quantization phase.

### Spectral and spatial response

(sensor-nir-fraction)=
#### `sensor.nir_fraction`

Sets each light group's near-infrared fraction of detected photoelectrons.
The renderer mixes material response as `(1-f)*visible + f*NIR`; this is not a
fraction of emitted optical power. Different illumination spectra can therefore
produce different material contrast. `[]` selects visible-only response.

(sensor-rgb-weights)=
#### `sensor.rgb_weights`

Converts linear visible RGB material response to a monochrome value. The
weights approximate sensor-and-illuminant spectral sensitivity; they are not
sRGB display-luminance coefficients. Selected event cameras share these weights.
Physical photometry expects their sum to be one and warns otherwise.

(sensor-aperture-px)=
#### `sensor.aperture_px`

Sets the width of the box response integrated across each pixel, in pixels.
The renderer convolves this spatial response with optical Gaussian blur.
It changes edge contrast and sampling support, not lens f-number or exposure
time; selected event cameras must use the same width.

(sensor-qe)=
#### `sensor.qe`

Sets effective pixel-area quantum efficiency for physical photocurrent
conversion. Microlens and fill-factor effects are already represented in this
effective value; do not multiply by a second fill factor. It is inactive under
relative and reference photometry, which use their own scale.

(sensor-kappa)=
#### `sensor.kappa`

Sets detected-electron conversion per lux, square micrometer and second at unit
QE. Physical photometry combines it with `qe`, pixel area and lens irradiance.
It depends on the assumed illuminant spectrum; the default is a white-LED prior,
not a universal conversion from lux to photons.


## Quick reference

### Response

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| {ref}`sensor.family <sensor-family>` | `enum` | required | Family supplying the sensor priors; must be specified explicitly. Choices: `prophesee_gen3`, `samsung_gen3`, `ideal`. |
| {ref}`sensor.photoreceptor <sensor-photoreceptor>` | `enum` | family | `cascade` uses the three-stage photoreceptor; `instantaneous` uses log intensity without cascade noise. Choices: `cascade`, `instantaneous`. |
| {ref}`sensor.k <sensor-k>` | `float · 1/(A s)` | family | Photoreceptor gain controlling the input pole and input-node noise. Cascade only. Range: >0. |
| {ref}`sensor.tau_o <sensor-tau-o>` | `float · s` | family | Output-node time constant. Cascade only. Range: >0. |
| {ref}`sensor.tau_sf <sensor-tau-sf>` | `float · s` | family | Source-follower time constant. Cascade only. Range: >0. |
| {ref}`sensor.sigma_o <sensor-sigma-o>` | `float · log-e` | family | Output-node noise amplitude in natural-log intensity units. Cascade only. Range: ≥0. |
| {ref}`sensor.noise <sensor-noise>` | `bool` | family | Enable in-path cascade noise; disabling it makes hot pixels leak-type. Cascade only. |

### Contrast and reset

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| {ref}`sensor.theta_on <sensor-theta-on>` | `float · log-e` | family | Mean ON contrast threshold, in natural-log intensity. Range: >0. |
| {ref}`sensor.theta_off <sensor-theta-off>` | `float · log-e` | family | Mean OFF contrast threshold, in natural-log intensity. Range: >0. |
| {ref}`sensor.sigma_theta_on <sensor-sigma-theta-on>` | `float · log-e` | family | Per-pixel standard deviation of ON threshold mismatch. Range: ≥0. |
| {ref}`sensor.sigma_theta_off <sensor-sigma-theta-off>` | `float · log-e` | family | Per-pixel standard deviation of OFF threshold mismatch. Range: ≥0. |
| {ref}`sensor.theta_onoff_corr <sensor-theta-onoff-corr>` | `float` | family | Correlation between ON and OFF threshold mismatch. Range: [-1, 1]. |
| {ref}`sensor.theta_floor <sensor-theta-floor>` | `float · log-e` | family | Lower bound for sampled thresholds. Range: >0. |
| {ref}`sensor.reset_mode <sensor-reset-mode>` | `enum` | family | `physical` resets after the blind refractory interval; `ideal` increments the reference by the signed threshold. Choices: `physical`, `ideal`. |
| {ref}`sensor.refractory_s <sensor-refractory-s>` | `float · s` | family | Refractory interval; also the dead time under ideal reset. Range: [0, 10). |
| {ref}`sensor.eps_ci <sensor-eps-ci>` | `float · log-e` | family | Reference offset applied at refractory release. |
| {ref}`sensor.initial_phase <sensor-initial-phase>` | `enum` | family | Initial reference phase: uniform ON interval, symmetric OFF–ON interval, or zero. Choices: `uniform`, `symmetric`, `zero`. |

### Dark current, leak, and filtering

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| {ref}`sensor.dark_current_fA <sensor-dark-current-fa>` | `float · fA` | family | Median dark current at `reference_temperature_K`. Range: >0. |
| {ref}`sensor.dark_sigma_ln <sensor-dark-sigma-ln>` | `float` | family | Log-normal spread of per-pixel dark currents. Range: ≥0. |
| {ref}`sensor.dark_ea_eV <sensor-dark-ea-ev>` | `float · eV` | family | Dark-current activation energy for temperature scaling. Range: ≥0. |
| {ref}`sensor.temperature_K <sensor-temperature-k>` | `float · K` | family | Die temperature used for dark-current and leak scaling. Range: >0. |
| {ref}`sensor.reference_temperature_K <sensor-reference-temperature-k>` | `float · K` | family | Reference temperature of the dark-current and leak priors. Range: >0. |
| {ref}`sensor.leak_rate_hz <sensor-leak-rate-hz>` | `float · Hz` | family | Median ON leak rate at the reference temperature. Range: ≥0. |
| {ref}`sensor.leak_sigma_log10 <sensor-leak-sigma-log10>` | `float` | family | Standard deviation of base-10 log leak rates across pixels. Range: ≥0. |
| {ref}`sensor.leak_ea_eV <sensor-leak-ea-ev>` | `float · eV` | family | Leak-rate activation energy for temperature scaling. Range: ≥0. |
| {ref}`sensor.tau_hp <sensor-tau-hp>` | `float · s` | family | Reference high-pass time constant; `0` disables the filter. Cascade only. Range: [0, 100]. |
| {ref}`sensor.tau_hp_sigma_ln <sensor-tau-hp-sigma-ln>` | `float` | `0.0` | Log-normal per-pixel spread of `tau_hp`; inactive when the filter is off. Range: ≥0. |

### Hot pixels

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| {ref}`sensor.hot_fraction <sensor-hot-fraction>` | `float` | family | Fraction of hot pixels; `0` disables hot-pixel simulation. Range: [0, 1]. |
| {ref}`sensor.hot_rate_law <sensor-hot-rate-law>` | `enum` | family | Distribution of target hot-pixel rates: log-uniform, or clamped log-normal. Choices: `log_uniform`, `log_normal`. |
| {ref}`sensor.hot_rate_min_hz <sensor-hot-rate-min-hz>` | `float · Hz` | family | Minimum target hot-pixel rate. Range: >0. |
| {ref}`sensor.hot_rate_max_hz <sensor-hot-rate-max-hz>` | `float · Hz` | family | Maximum target hot-pixel rate. Range: >0. |
| {ref}`sensor.hot_rate_median_hz <sensor-hot-rate-median-hz>` | `float · Hz` | family | Median target rate for `hot_rate_law: log_normal`. Range: >0. |
| {ref}`sensor.hot_rate_sigma_ln <sensor-hot-rate-sigma-ln>` | `float` | family | Natural-log rate spread for `hot_rate_law: log_normal`. Range: ≥0. |
| {ref}`sensor.hot_leak_fraction <sensor-hot-leak-fraction>` | `float` | family | Fraction of hot pixels driven by leak; others use threshold noise. All are leak-type without noise. Range: [0, 1]. |
| {ref}`sensor.hot_reference_current_fA <sensor-hot-reference-current-fa>` | `float · fA` | family | Photocurrent at which threshold-type hot-pixel rates are calibrated. Range: >0. |

### Readout and clock

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| {ref}`sensor.readout_quantum_s <sensor-readout-quantum-s>` | `float · s` | family | Timestamp quantization interval; `0` disables quantization. Range: [0, 1). |
| {ref}`sensor.readout_delay_s <sensor-readout-delay-s>` | `float · s` | family | Constant readout delay before timestamp quantization. Range: [0, 1). |
| {ref}`sensor.clock_offset_ns <sensor-clock-offset-ns>` | `int · ns` | derived | Additive sensor-clock offset; defaults to the bundle camera's offset. |
| {ref}`sensor.clock_drift_ppm <sensor-clock-drift-ppm>` | `float · ppm` | family | Sensor-clock drift relative to the run epoch. Range: (-10000, 10000). |

### Rate limiter

Settings other than `limiter` apply only with `limiter: epoch`.

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| {ref}`sensor.limiter <sensor-limiter>` | `enum` | family | Post-sort token bucket with a periodically resetting rate cap. Choices: `off`, `epoch`. |
| {ref}`sensor.limiter_rate0_hz <sensor-limiter-rate0-hz>` | `float · Hz` | family | Rate cap at the start of each limiter epoch. Range: [1, 1e+10]. |
| {ref}`sensor.limiter_rate1_hz <sensor-limiter-rate1-hz>` | `float · Hz` | family | Rate cap at the end of each epoch; the cap decreases linearly. Range: [1, 1e+10]. |
| {ref}`sensor.limiter_period_s <sensor-limiter-period-s>` | `float · s` | family | Limiter epoch length on the sensor clock. Range: [1e-06, 100]. |
| {ref}`sensor.limiter_depth <sensor-limiter-depth>` | `int` | family | Token-bucket capacity in events. Range: [1, 1e+09]. |
| {ref}`sensor.limiter_phase_s <sensor-limiter-phase-s>` | `float? · s` | `null` | Reset time modulo the epoch; `null` draws a phase per camera. Range: [0, 100). |

### Spectral response and aperture

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| {ref}`sensor.nir_fraction <sensor-nir-fraction>` | `float[]` | `[]` | Near-infrared photoelectron fraction per light group; `[]` means zero in every group. Range: [0, 1]. |
| {ref}`sensor.rgb_weights <sensor-rgb-weights>` | `float[3]` | family | Linear RGB weights for monochrome irradiance; selected cameras must agree. Range: ≥0. |
| {ref}`sensor.aperture_px <sensor-aperture-px>` | `float · px` | family | Box width of the pixel response; selected cameras must agree. Range: ≥0. |
| {ref}`sensor.qe <sensor-qe>` | `float` | family | Effective quantum efficiency, including pixel-area effects. Physical photometry only. Range: (0, 1]. |
| {ref}`sensor.kappa <sensor-kappa>` | `float · 1/(s um2 lx)` | `7800.0` | Photoelectrons per second, square micrometer, and lux at unit QE. Physical photometry only. Range: >0. |

### Cross-parameter constraints

- `theta_on` and `theta_off` must exceed `theta_floor`; `abs(eps_ci) < theta_floor / 2`.
- The larger of `tau_o` and `tau_sf` must be at least 1.1 times the smaller.
  With noise enabled and positive `sigma_o`, `tau_o >= 1e-6` s.
- Positive `tau_hp` requires the cascade and must be at least
  `max(0.001, 10 * tau_o, 10 * tau_sf)` s.
- Hot rates require `hot_rate_min_hz <= hot_rate_max_hz`; enabled hot pixels also
  require `hot_rate_max_hz * refractory_s < 1`.
- With quantization enabled, `readout_delay_s >= readout_quantum_s`.
- Epoch limiting requires `rate1 <= rate0`, a phase in `[0, period)`, and a bucket
  depth of at least `ceil(rate0 * max(readout_quantum_s, 1e-6)) + 1`.
- `nir_fraction` is empty or has one entry per light group. RGB weights and aperture
  width must agree across selected cameras.

## Model basis

The implemented signal path follows the adopted cascade in
`src/sensor/stage1.h`, `pixel.h` and `noise.h`; frozen maps are defined in
`maps.cpp`, and readout and limiting in `readout.h` and `limiter.h`.
The [sensor guide](../../guide/sensors.md) links the primary measurements behind
the response and noise model. Distribution choices, thermal extrapolation,
high-pass reference relaxation and the token bucket are explicit implementation
approximations and should be calibrated separately from those measurements.
