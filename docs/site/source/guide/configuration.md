# Configuration

A run configuration is a YAML mapping with `format: ballasim.run/2`. It selects
a scene source, camera models, simulation settings, and an output directory.

```yaml
format: ballasim.run/2
scene:
  path: scene.yaml
cameras:
  default:
    sensor:
      family: prophesee_gen3
    photometry:
      mode: relative
      illuminance_scale: 400.0
run:
  device: auto
accuracy:
  profile: fidelity
output:
  dir: runs/example
```

The camera names come from the scene. `cameras.default` supplies settings to
every camera; a named block overrides those settings within the same layer.
Use `run.select` to simulate a subset of the scene's cameras.

## Sections

| Section | Contents |
| --- | --- |
| `scene` | Source path, import settings, background, and trajectory fitting. |
| `cameras.<name>.sensor` | Sensor family, thresholds, noise, clock, and readout. |
| `cameras.<name>.photometry` | Conversion from rendered radiance to photocurrent. |
| `cameras.<name>.optics` | Blur, aperture transmission, vignetting, and diffraction. |
| `cameras.<name>.frames` | Frame timing, reference integration, and APS response. |
| `cameras.<name>.imu` | IMU timing, extrinsics, noise, filtering, and quantization. |
| `run` | Time interval, cameras, device, crop, and random seeds. |
| `accuracy` | Sampling and numerical tolerances. |
| `execution` | Thread counts and memory budgets. |
| `output` | Destination, overwrite policy, and event ground truth. |
| `dataset.evimo2` | Metadata used to interpret EVIMO2 recordings. |

Each namespace is documented with its corresponding manual topic. See
[Settings conventions](../reference/parameters.md) for scope, default notation,
and a map from configuration namespaces to those topics.

## Sensor families and photometry

Set `sensor.family` explicitly for each selected event camera, directly or
through `cameras.default` or a profile. Supported families are `prophesee_gen3`,
`samsung_gen3`, and `ideal`. Family values are model priors; use measured values
or versioned profiles for a particular device.

| `photometry.mode` | Required input | Behavior |
| --- | --- | --- |
| `relative` | Positive `illuminance_scale` | Scale the renderer output to photocurrent. |
| `physical` | Calibrated lights, pixel pitch, and lens f-number | Derive the scale from lens transmission, quantum efficiency, and illuminant response. Blender sources also require `scene.blender.lm_per_w`. |
| `reference` | Positive `reference_current_fA` | Solve the scale from the median signal current of the first frame's reference pixels. |

`photometry.reference_instance` restricts reference pixels to a named body.
An empty name selects all pixels that hit geometry. The physical and reference
modes derive `illuminance_scale`; an explicit scale is inert in those modes.

## Accuracy profiles

| Profile | Use |
| --- | --- |
| `fidelity` | Default. Derive sampling settings from the selected sensors. |
| `balanced` | Use a coarser displacement schedule and bounded jump refinement. |
| `reference` | Use uniform spatial sampling and stricter temporal refinement for conformance comparisons. |
| `phase1` | Reproduce the frozen Phase 1 settings. |

An explicit accuracy key overrides its profile value. The resolver warns when
an override is looser than the derived setting. Inspect the resolved values
when changing a sensor family or using a frozen EVIMO2 profile.

## Profiles and precedence

Profiles provide versioned values with provenance. Root profiles apply first;
camera profiles apply to their named camera. Later entries in a profile list
take precedence.

```yaml
profiles: [evimo2/lab_lighting@v0]
scene:
  evimo2:
    sequence: ${EVIMO2_ROOT}/left_camera/imo/eval/scene13_dyn_test_00_000000
cameras:
  left_camera:
    profiles: [evimo2/prophesee_gen3_left@v0]
```

Add these sections to a format-2 configuration with an output directory.
Profile IDs identify YAML files packaged under `ballasim/profiles`.

Values resolve in the following order, from lowest to highest precedence:

1. Schema defaults, sensor-family priors, and accuracy-profile values.
2. Root profiles, followed by camera profiles.
3. The run configuration.
4. Command-line `--set KEY=VALUE` overrides.

`--profile ID` appends to the root profile list. Within each layer, named camera
values override `cameras.default`. Lists replace earlier lists; `null` is a
value and does not restore a default.

## Paths and overrides

Path values in a YAML file are relative to that file. Paths supplied by
`--set` are relative to the working directory. Paths support `~`, `$NAME`, and
`${NAME}`; an unset environment variable is an error.

```bash
ballasim resolve run.yaml
ballasim run run.yaml --set cameras.default.sensor.theta_off=0.30
ballasim run run.yaml --set 'run.select=[left_camera]'
```

Override values use YAML scalar or flow-list syntax. Integer parameters require
integers; for example, use `1024`, rather than `1024.0`, for nanosecond spacing.

## Validation and provenance

Unknown keys, invalid types, values outside hard bounds, and incompatible
settings are rejected. Values outside soft bounds produce warnings. The
resolver also reports required values and settings that are inert in the
selected mode.

`ballasim resolve` loads the scene and prints the resolved configuration without
simulating events. Completed runs record resolved values, source layers,
profile hashes, warnings, and stage digests in `run.json`.
The [loading and resolution methods](../reference/api/configuration.md) expose the same
information programmatically.

```{toctree}
:hidden:
:maxdepth: 1

Loading and resolution <../reference/api/configuration>
Settings conventions <../reference/parameters>
```
