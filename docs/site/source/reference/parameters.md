# Settings conventions

The run schema contains 287 parameters: 141 camera, 77 scene, 65 run, and four
dataset parameters. Settings are documented within the manual topic that uses
them. Entries use descriptor names such as `sensor.theta_on`;
camera parameters are written under `cameras.<name>`, or under `cameras.default`
to provide shared values.

```yaml
cameras:
  default:
    sensor: {family: prophesee_gen3, theta_on: 0.25}
    photometry: {mode: relative, illuminance_scale: 400}
accuracy: {profile: fidelity}
```

## Reading the tables

| Notation | Meaning |
| --- | --- |
| Literal default | Used when no explicit value, profile value, or applicable derivation replaces it. |
| `family` | Supplied by the selected [sensor family](parameters/families.md). |
| `required` | Must be specified while the parameter applies. |
| `derived` | Computed from the scene, camera, environment, or other parameters. |
| `float[3]`, `int[]` | Fixed-length or variable-length lists. |
| `?` | Accepts YAML `null`; each entry describes its effect. |
| `[a, b]`, `(a, b)` | Closed and open hard bounds; list bounds apply per element. |

Tables list schema **base defaults**. The default `fidelity` accuracy profile
replaces several accuracy settings; consult [Accuracy profiles](parameters/profiles.md)
for resolved behavior. Explicit values take precedence. Fields used outside their
applicable mode may be reported as inactive.

All numeric values must be finite. Hard-range violations are errors. Soft
physical ranges produce warnings and are included with provenance, applicability,
and cache-stage metadata in the
{download}`complete parameter descriptor <../_static/data/params.json>`.

Use `ballasim params PREFIX` to inspect descriptors, and `ballasim resolve RUN.yaml`
to inspect the resolved values for a particular run. See
[Configuration](../guide/configuration.md) for layers, profiles, and overrides.

## Find a setting

| Manual topic | Settings |
| --- | --- |
| [Scenes & assets](../guide/scenes.md) | Scene sources, importers, Blender, and EVIMO2. |
| [Trajectories](../guide/trajectories.md) | Smoothing, rejection, and stationary intervals. |
| [Cameras & optics](../guide/cameras.md) | Photometry, vignetting, blur, and diffraction. |
| [Event sensors](../guide/sensors.md) | Sensor families, thresholds, noise, and readout. |
| [Running simulations](../guide/simulations.md) | Accuracy, execution, selection, and memory budgets. |
| [Outputs & ground truth](../guide/outputs.md) | Event labels, ground truth, frames, APS, and IMU. |
