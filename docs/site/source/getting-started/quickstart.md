# Quickstart

This example simulates a small room under flickering illumination. It uses
procedural geometry and requires no external assets. Complete
[Installation](installation.md) first, then save the following files in one
directory.

## Create a scene

Save as `scene.yaml`:

```yaml
schema: ballasim.scene/1
name: quickstart
time: {t_start_s: 0.0, t_end_s: 0.1}
materials:
  wall: {base_color: [0.7, 0.7, 0.7]}
  floor: {base_color: [0.2, 0.2, 0.2]}
room:
  id: 1
  size_m: [4.0, 4.0, 3.0]
  walls: wall
  floor: floor
  ceiling: wall
rigs:
  rig:
    pose: {eye_m: [0.0, -1.0, 1.0], target_m: [0.0, 0.0, 1.0]}
cameras:
  - name: cam
    rig: rig
    model: radtan
    width: 64
    height: 48
    fx: 60.0
    fy: 60.0
    cx: 31.5
    cy: 23.5
    dist: [0, 0, 0, 0, 0]
    sensor: ideal
    pixel_pitch_m: 15.0e-6
light_groups:
  flicker: {type: fourier, f0_hz: 100.0, a: [0.5], phi: [0.0]}
lights:
  - {kind: ambient, group: flicker, intensity: 1.0}
```

The camera remains stationary. Illumination changes generate ON and OFF events.
Scene lengths are in meters; camera intrinsics are in pixels.

## Configure the run

Save as `run.yaml`:

```yaml
format: ballasim.run/2
scene: {path: scene.yaml}
run: {device: cpu}
accuracy: {profile: balanced}
cameras:
  default:
    sensor: {family: ideal}
    photometry: {illuminance_scale: 400.0}
output: {dir: runs/quickstart}
```

The `ideal` family disables sensor noise and uses an ideal threshold-reset model.
`illuminance_scale` sets the photocurrent per unit of rendered irradiance in
relative photometry mode.

## Resolve and simulate

```bash
python -m ballasim resolve run.yaml
python -m ballasim run run.yaml
```

`resolve` loads the scene and reports the selected values without simulating.
`run` writes `runs/quickstart/run.json`, performance logs and
`runs/quickstart/events/cam.h5`. The simulated interval contains the complete
engine windows within the requested scene interval.

To replace this run, add `--overwrite`. To select an available GPU engine, add
`--set run.device=gpu`.

## View the events

With the `viz` extra installed:

```bash
python -m ballasim.viz info --sim runs/quickstart/events/cam.h5
python -m ballasim.viz html --sim runs/quickstart/events/cam.h5 --out quickstart.html
```

Open `quickstart.html` in a browser. For interactive run inspection, use
`python -m ballasim ui --open runs/quickstart` with the `ui` extra installed.

Continue with [Scenes & assets](../guide/scenes.md),
[Configuration](../guide/configuration.md) or
[Outputs & ground truth](../guide/outputs.md).
