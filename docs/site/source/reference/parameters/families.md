# Sensor families

Family priors fill only fields marked `family`. They are starting values, not
device calibrations. Versioned profiles and explicit settings take precedence.

## Family selection

Choose a family for its response model, then inspect the physical assumptions
that matter to the recording. A family does not identify a serial-numbered
camera or reproduce arbitrary hardware bias settings. The full
[sensor settings](sensor.md#parameter-behavior) explain each parameter's effect.

(family-prophesee-gen3)=
### `prophesee_gen3`

Provides the three-stage cascade, filtered temporal noise, physical reset and
asynchronous timestamp model used as the base prior. The default mean ON/OFF
threshold is `0.25` log-e, refractory time is `300 µs`, and timestamp quantum
is `1 µs`. Input-stage gain and output-node noise draw on the multi-stage pixel
analysis in [Graça et al.](https://arxiv.org/abs/2304.04706).

These values combine literature priors, engineering choices and assumptions;
they are not a measured Gen3 parameter set. In particular, dark current is an
adopted radiometric prior, population spreads include placeholders, and no
published universal high-pass bias conversion is assumed. Hot pixels,
high-pass filtering and the driver rate limiter are off until enabled.

(family-samsung-gen3)=
### `samsung_gen3`

Overrides the base thresholds to `0.20` log-e, dark-current prior to `0.08 fA`,
refractory time to `196 µs` and additional readout delay to `100 µs`. The dark
current is derived by scaling the base pixel area; effective QE `0.5` is an
assumption. Other unlisted fields inherit the Prophesee prior and are unvalidated
for Samsung.

Samsung Gen3 uses a frame-like readout, described in the
[vendor presentation](https://rpg.ifi.uzh.ch/docs/CVPR19workshop/CVPRW19_Eric_Ryu_Samsung.pdf).
This family approximates it with mean blind time and reporting delay in an
asynchronous model. It does not reproduce the sensor's complete global-hold,
column-scan timing. Use a recording-specific profile when those timings matter.

(family-ideal)=
### `ideal`

Selects instantaneous log response, ideal threshold-increment reset, zero
refractory time and `0.20` log-e thresholds. It disables temporal noise,
threshold/dark-current mismatch, leakage, hot pixels, readout delay,
quantization and limiting. Use it to inspect geometry-driven contrast events
without the physical cascade's latency and noise.

The response still uses `ln(I_signal + I_dark)`, with a positive `0.01 fA`
dark-current floor. Pixel aperture, optical blur and scene sampling also remain
active. Inherited `sigma_o` and hot-rate values are stored but inactive; setting
`photoreceptor: cascade` reactivates the listed fast cascade constants and should
be treated as a new model choice.

## Priors, fitted profiles and variation

Families populate unspecified fields. A versioned EVIMO2 sensor profile instead
records a fitted setup, including its illumination, readout and model assumptions.
Applying such a profile does not turn its fitted values into general family
constants. Explicit run values override both sources.

The threshold and dark/leak distributions are drawn once for each sensor seed,
camera index and pixel identity. Temporal-noise draws evolve in time. Repeatable
spatial maps therefore do not imply noiseless event timing. See
[Noise and pixel variation](../../guide/sensors.md#noise-and-pixel-variation).

## Prior table

The table contains every family-default key. `*` marks a placeholder; `†` marks
a value inherited from the base family. Other statuses and source citations are
available in the [complete descriptor](../parameters.md).

| Parameter | Prophesee Gen3 | Samsung Gen3 | Ideal |
| --- | --- | --- | --- |
| {ref}`sensor.photoreceptor <sensor-photoreceptor>` | `"cascade"` | `"cascade"`† | `"instantaneous"` |
| {ref}`sensor.k <sensor-k>` | `1e+16` | `1e+16`† | `1e+18` |
| {ref}`sensor.tau_o <sensor-tau-o>` | `2.5e-05` | `2.5e-05`† | `1e-06` |
| {ref}`sensor.tau_sf <sensor-tau-sf>` | `0.0001` | `0.0001`† | `2e-06` |
| {ref}`sensor.sigma_o <sensor-sigma-o>` | `0.06` | `0.06`† | `0.06`† |
| {ref}`sensor.noise <sensor-noise>` | `true` | `true`† | `false` |
| {ref}`sensor.dark_current_fA <sensor-dark-current-fa>` | `0.21` | `0.08` | `0.01` |
| {ref}`sensor.dark_sigma_ln <sensor-dark-sigma-ln>` | `0.3`* | `0.3`*† | `0.0` |
| {ref}`sensor.dark_ea_eV <sensor-dark-ea-ev>` | `0.78` | `0.78`† | `0.78`† |
| {ref}`sensor.temperature_K <sensor-temperature-k>` | `298.15` | `298.15`† | `298.15`† |
| {ref}`sensor.reference_temperature_K <sensor-reference-temperature-k>` | `298.15` | `298.15`† | `298.15`† |
| {ref}`sensor.theta_on <sensor-theta-on>` | `0.25` | `0.2` | `0.2` |
| {ref}`sensor.theta_off <sensor-theta-off>` | `0.25` | `0.2` | `0.2` |
| {ref}`sensor.sigma_theta_on <sensor-sigma-theta-on>` | `0.03` | `0.03`† | `0.0` |
| {ref}`sensor.sigma_theta_off <sensor-sigma-theta-off>` | `0.03` | `0.03`† | `0.0` |
| {ref}`sensor.theta_onoff_corr <sensor-theta-onoff-corr>` | `0.0` | `0.0`† | `0.0`† |
| {ref}`sensor.theta_floor <sensor-theta-floor>` | `0.01` | `0.01`† | `0.01`† |
| {ref}`sensor.reset_mode <sensor-reset-mode>` | `"physical"` | `"physical"`† | `"ideal"` |
| {ref}`sensor.refractory_s <sensor-refractory-s>` | `0.0003` | `0.000196` | `0.0` |
| {ref}`sensor.eps_ci <sensor-eps-ci>` | `0.0` | `0.0`† | `0.0`† |
| {ref}`sensor.initial_phase <sensor-initial-phase>` | `"uniform"` | `"uniform"`† | `"uniform"`† |
| {ref}`sensor.leak_rate_hz <sensor-leak-rate-hz>` | `0.01` | `0.01` | `0.0` |
| {ref}`sensor.leak_sigma_log10 <sensor-leak-sigma-log10>` | `0.1`* | `0.1`*† | `0.0` |
| {ref}`sensor.leak_ea_eV <sensor-leak-ea-ev>` | `0.78` | `0.78`† | `0.78`† |
| {ref}`sensor.tau_hp <sensor-tau-hp>` | `0.0` | `0.0` | `0.0` |
| {ref}`sensor.hot_fraction <sensor-hot-fraction>` | `0.0` | `0.0`† | `0.0` |
| {ref}`sensor.hot_rate_law <sensor-hot-rate-law>` | `"log_uniform"`* | `"log_uniform"`*† | `"log_uniform"`*† |
| {ref}`sensor.hot_rate_min_hz <sensor-hot-rate-min-hz>` | `10.0`* | `10.0`*† | `10.0`*† |
| {ref}`sensor.hot_rate_max_hz <sensor-hot-rate-max-hz>` | `2000.0` | `2000.0`† | `2000.0`† |
| {ref}`sensor.hot_rate_median_hz <sensor-hot-rate-median-hz>` | `100.0`* | `100.0`*† | `100.0`*† |
| {ref}`sensor.hot_rate_sigma_ln <sensor-hot-rate-sigma-ln>` | `1.5`* | `1.5`*† | `1.5`*† |
| {ref}`sensor.hot_leak_fraction <sensor-hot-leak-fraction>` | `0.5`* | `0.5`*† | `0.5`*† |
| {ref}`sensor.hot_reference_current_fA <sensor-hot-reference-current-fa>` | `100.0`* | `100.0`*† | `100.0`*† |
| {ref}`sensor.readout_quantum_s <sensor-readout-quantum-s>` | `1e-06` | `1e-06`† | `0.0` |
| {ref}`sensor.readout_delay_s <sensor-readout-delay-s>` | `0.0003` | `0.0001` | `0.0` |
| {ref}`sensor.clock_drift_ppm <sensor-clock-drift-ppm>` | `0.0` | `0.0`† | `0.0`† |
| {ref}`sensor.limiter <sensor-limiter>` | `"off"` | `"off"` | `"off"` |
| {ref}`sensor.limiter_rate0_hz <sensor-limiter-rate0-hz>` | `6000000.0` | `6000000.0`† | `6000000.0`† |
| {ref}`sensor.limiter_rate1_hz <sensor-limiter-rate1-hz>` | `6000000.0` | `6000000.0`† | `6000000.0`† |
| {ref}`sensor.limiter_period_s <sensor-limiter-period-s>` | `0.0656` | `0.0656`† | `0.0656`† |
| {ref}`sensor.limiter_depth <sensor-limiter-depth>` | `32`* | `32`*† | `32`*† |
| {ref}`sensor.rgb_weights <sensor-rgb-weights>` | `[0.52, 0.37, 0.11]` | `[0.52, 0.37, 0.11]`† | `[0.52, 0.37, 0.11]`† |
| {ref}`sensor.aperture_px <sensor-aperture-px>` | `1.0` | `1.0`† | `1.0`† |
| {ref}`sensor.qe <sensor-qe>` | `0.37` | `0.5`* | `0.37`† |

Hot pixels are disabled in neutral family priors because no general device rate
is established. Enable them explicitly or use an appropriate measured sensor
profile. The included EVIMO2 profiles describe those recording setups.
