# Event sensors

An event sensor converts each pixel's photocurrent history into ON and OFF
threshold crossings. BallaSim models the analog response before generating
events, so filtering, temporal noise, leakage and reset share the same pixel
history.

Choose a [sensor family](../reference/parameters/families.md), then set measured
or task-specific values under `cameras.<name>.sensor`. The
[sensor settings](../reference/parameters/sensor.md) explain every parameter;
the [SensorModel API](../reference/api/sensor.md) accepts photocurrent-driving
irradiance samples directly.

## Photocurrent and contrast

The renderer supplies linear irradiance per light group. Photometry converts
it to signal current; each pixel adds its own positive dark current. The
photoreceptor therefore responds to total current:

$$
I(t)=I_{\mathrm{signal}}(t)+I_{\mathrm{dark}}.
$$

The comparator measures the change in filtered log current from a stored
reference. Positive changes produce ON events; negative changes produce OFF
events. Thresholds use natural-log units: `theta=0.2` requires a current ratio
of approximately `exp(0.2)=1.22` for an ideal ON crossing. Dark current compresses
the effective contrast of dim signals.

Absolute current matters even when only relative contrast is of interest: it
changes response speed and noise. See
[Photometry and optics](../reference/parameters/photometry-optics.md) for
`relative`, `physical` and `reference` scaling.

## Photoreceptor response

`sensor.photoreceptor: cascade` uses a nonlinear input stage followed by an
output-node low-pass and a source-follower low-pass. Around constant current,
their rates are:

$$
a_{\mathrm{input}}=kI,\qquad
a_{\mathrm{output}}=1/\tau_o,\qquad
a_{\mathrm{sf}}=1/\tau_{\mathrm{sf}}.
$$

Here `I` is in amperes and `k` is in `(A s)^-1`. Dim current slows the input
stage; bright current shifts the bandwidth limit toward the fixed downstream
poles. Increasing `k` changes both bandwidth and shot-noise strength. It is
therefore distinct from changing the readout delay.

This cascade is BallaSim's adopted approximation to the multi-stage pixel
response described by [Graça and Delbruck](https://arxiv.org/abs/2109.08640)
and [Graça et al.](https://arxiv.org/abs/2304.04706). It assumes separated real
poles and does not simulate a complete transistor circuit or underdamped
ringing.

`instantaneous` bypasses the cascade and its temporal noise and compares log
current directly. The `ideal` family selects this response together with ideal
reset and zero dead time. Its positive dark-current floor and spatial pixel
response remain present.

## Noise and pixel variation

| Component | Implementation | Effect |
| --- | --- | --- |
| Temporal noise | Filtered input shot noise and output-node noise | Shifts, completes or creates threshold crossings |
| Threshold mismatch | One Gaussian ON/OFF threshold draw per pixel, clamped to a floor | Persistent spatial sensitivity differences |
| Dark-current variation | Positive log-normal current map | Pixel-dependent low-light contrast and bandwidth |
| Leakage | Reference drift toward an ON crossing | Repeated ON events under steady input |
| Hot pixels | Selected tails of leak or threshold distributions | Persistent unusually active pixels |

Temporal noise uses a linear-noise approximation around the deterministic
cascade. At constant current, the input-node log variance is `k*q_e`, with
`q_e` the elementary charge. The downstream filters determine how much reaches
the comparator. Consequently, event noise is neither independent of illumination
nor an event stream added after signal generation.

Threshold mismatch is frozen for the sensor seed and pixel identity. It is
not redrawn at each crossing. Set mismatch spreads to zero to remove spatial
threshold variation; set `sensor.noise: false` to remove temporal cascade noise.
Neither switch removes leakage or dark current.

The distinction between temporal noise and fixed mismatch follows the pixel
measurements discussed by [McReynolds et al.](https://arxiv.org/abs/2404.07656v1).
The hot-pixel mixture and its rate distributions are configurable modeling
assumptions; the neutral families disable hot pixels by default.

## Reset, refractory time and filtering

After a physical crossing, the comparator is blind for `refractory_s`. At
release it stores the current filtered signal plus `eps_ci` as the new
reference. Contrast accumulated during the blind interval is lost.

Ideal reset instead advances the reference by the crossed threshold. This
preserves residual contrast and supports textbook DVS comparisons. A nonzero
`refractory_s` still imposes dead time in ideal mode.

Leakage lowers the reference at a rate proportional to the pixel's ON
threshold. An optional high-pass model, `tau_hp > 0`, relaxes the reference
toward the signal and suppresses sufficiently slow illumination changes. Smaller
positive `tau_hp` gives stronger high-pass filtering; zero disables it. This is
a behavioral approximation to the documented
[Prophesee high-pass bias](https://docs.prophesee.ai/stable/hw/manuals/biases.html),
not a published circuit model or a calibrated bias-register conversion.

## Initialization and timestamps

Initialization sets the analog filters to steady state for the first input.
`accuracy.initial_state: phase` uses `sensor.initial_phase` for the comparator;
`stationary` estimates a noisy cascade's comparator state from representative
pixels under held input; noiseless models retain the phase law. Warm-up then
discards startup events before the requested run. These
controls describe the start history; see
[Accuracy settings](../reference/parameters/accuracy.md).

Crossing time and reported time are separate. Readout adds a fixed delay,
floors time to a free-running counter grid, then applies clock offset and
drift. Analog latency emerges from the photoreceptor and threshold dynamics;
`readout_delay_s` represents the additional reporting delay.

The optional rate limiter drops events from the time-sorted reported stream.
Pixels have already updated their reference when those events are dropped.
Its token-bucket rate cap can fall across a repeating epoch; it is an adopted
driver-level approximation, not a full model of bus arbitration.

## Labels and interpretation

Signal, noise and leak can contribute to the same crossing. BallaSim attributes
events using increments from that pixel's actual history and a configurable
label-share cut. These are operational labels, not a claim that a physical event
has one uniquely observable cause. Signal-path interactions and alternating
noise pairs are documented in
[McReynolds et al.](https://arxiv.org/abs/2304.03494).

The native sensor's four raw labels differ from the saved event file's
five-class convention. See {ref}`SensorModel results <sensor-advance>`
and [Outputs & ground truth](outputs.md) before interpreting label values.

```{toctree}
:hidden:
:maxdepth: 1

Sensor families <../reference/parameters/families>
Pixel-model settings <../reference/parameters/sensor>
Sensor integration methods <../reference/api/sensor>
```
