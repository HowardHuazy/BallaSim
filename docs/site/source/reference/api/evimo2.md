# EVIMO2 reconstruction methods

The [EVIMO2 settings](../parameters/evimo2.md) describe clock, asset, motion,
and lighting choices. These methods expose reconstruction and released-data access.

(scene-build-evimo2)=
## EVIMO2 reconstruction — `build_bundle`

```python
from ballasim.evimo2 import build_bundle, relight

build_bundle(sequence_dir, options, *, name=None, lit=True)
```

`build_bundle()` returns `Evimo2Build` for one camera sequence; `options` is an
`Evimo2Options` instance. Its keyword-only constructor requires
`marker_albedo` and `untextured_albedo`; `lighting: Evimo2Lighting` is required
while `lights=True`. Other fields control object selection, marker geometry,
clock reference, trimming, smoothing, static-body detection, dataset roots,
additional cameras, and worker count. Defaults correspond to `scene.evimo2.*`
and `scene.trajectory.*` parameters.

The build returns registered scanned bodies, a common corrected pose clock,
fitted rig/object trajectories, and per-camera calibration. `lit=False` defers
light assembly while preserving the sidecar needed for later relighting; it
does not create a different geometric reconstruction. Missing catalog objects
or insufficient tracks can produce warnings and skipped bodies, so inspect the
returned warnings before using its ground truth.

(scene-evimo2-options)=
### Adapter controls — `Evimo2Options`

Construct this keyword-only dataclass to specify the reconstruction assumptions.
`smoothing` is a `SmoothingOptions` object; `extra_cameras` contains sequence
directory paths for this Python interface. `warmup_ns` maps to the scene's start
trim and does not request sensor initialization. Worker count changes build
concurrency without changing fitted output. See
[EVIMO2 settings](../parameters/evimo2.md) for the individual controls' effects.

| `Evimo2Options` fields | Purpose |
| --- | --- |
| `markers`, `marker_subdivisions`, `marker_albedo`, `marker_radius_m` | Marker geometry and appearance. |
| `objects`, `min_samples`, `untextured_albedo` | Object selection, minimum track length, and fallback reflectance. |
| `static_tabletop`, `static_tabletop_fraction`, `pin_static_px_s` | World-frame and static-body selection. |
| `clock_reference`, `warmup_ns`, `smoothing` | Clock reference, start trim, and `SmoothingOptions`. |
| `extra_cameras`, `objects_root`, `raw_root` | Additional sequence directories and dataset locations. |
| `lights`, `lighting`, `workers` | Enable lighting, supply `Evimo2Lighting`, and select build threads. |

(scene-evimo2-lighting)=
### Lab-light controls — `Evimo2Lighting`

Represents the ordinary and low-light point-plus-ambient models together so the
same object can be used with either recording branch. Unlike conditional YAML
resolution, its constructor requires both branches' fields. `Evimo2Lighting`
requires `ambient`, `low_light_ambient`, `light_height_m`,
`light_color`, `light_shadows`, `ambient_flickers`, `mains_a`, `mains_f0_hz`,
`mains_phi_rad`, `strobe_width_s`, `strobe_floor`, `strobe_rise_s`, and
`strobe_phase_offset_s`; `background_albedo` defaults to `0.0`.
`Evimo2Lighting.from_values(values)` constructs it from resolved configuration
values. Invalid adapter and lighting parameters raise `ValueError`.

(scene-evimo2-lighting-from-values)=
#### Resolved lighting — `Evimo2Lighting.from_values`

`Evimo2Lighting.from_values(values)` accepts the run's flat resolved
`{parameter_key: value}` mapping and returns an `Evimo2Lighting` instance.
It translates the `scene.evimo2.*` names into dataclass fields and preserves
optional-field defaults for older records. It does not resolve profile files
or supply missing required lighting values.

(scene-evimo2-build)=
### Reconstruction output — `Evimo2Build`

`Evimo2Build` exposes `bundle`, `sequence`, `retiming`, `fits`, `warnings`,
`T_world_c0`, `T_body_subject`, and `sidecar`. `gt_frame_times()` returns
`(archive_key, corrected_time_ns)` pairs. `relight()` returns a validated bundle
with rebuilt lighting; geometry, trajectories, and cameras are shared.

`T_world_c0` is a `float64 (4, 4)` transform from the sequence's initial-camera
frame to the reconstructed world. `T_body_subject` maps catalog object IDs to
body-from-motion-capture-subject transforms. `fits` retains per-track diagnostics;
`sidecar` retains the temporal and spatial anchor required for light assembly.

(scene-evimo2-gt-frame-times)=
#### Ground-truth clock mapping — `Evimo2Build.gt_frame_times`

`build.gt_frame_times()` returns `(archive_key, corrected_time_ns)` pairs for
released depth/mask frames. It maps their raw pose-stamp times through the
build's `Retiming`; use these corrected times for bundle pose queries.
The archive key selects stored arrays and is not necessarily the original
frame ID in metadata.

(scene-evimo2-relight)=
### Lighting replacement — `relight`

`relight(bundle, sidecar, lighting)` returns a validated `SceneBundle` with
rebuilt lights, groups, and lighting metadata. Geometry, trajectories, and camera
objects are shared with the input. `lighting=None` removes the lights; otherwise
the sidecar must contain the anchor created by a build with lighting enabled.
It changes spatial/temporal illumination without fitting the measured motion
again.

(scene-evimo2-sequence-reader)=
### Released data — `Sequence`

`ballasim.evimo2.dataset.Sequence(directory)` reads released sequence data:

The constructor reads `dataset_info.npz` and identifies its camera from the
directory layout. Missing metadata raises `FileNotFoundError`; a path without a
recognized camera folder raises `ValueError`. It exposes released measurements
without reconstructing a `SceneBundle`. Camera dimensions, extrinsics, object
IDs, and low-light category are available as properties.

(scene-sequence-track)=
#### Recorded poses — `Sequence.track`

`sequence.track(key)` returns `PoseTrack(rows, t_raw_ns, q, p)` for `"cam"` or
an object-ID string. `rows: int64 (N,)` indexes the full trajectory;
`q: float64 (N, 4)` and `p: float64 (N, 3)` use source pose conventions. Camera
poses map the current camera to its initial frame; object poses map object to
camera. They require the adapter's frame composition and clock correction
before becoming world trajectories.

(scene-sequence-gt-frames)=
#### Frame catalog — `Sequence.gt_frames`

`sequence.gt_frames()` returns `(archive_key, raw_time_ns, metadata)` for each
frame with depth/mask data. Keys are formed relative to the first frame ID.
Times remain on the recorded pose-stamp clock; use `Evimo2Build.gt_frame_times()`
for corrected scene-query times.

(scene-sequence-depth)=
#### Released depth — `Sequence.depth`

`sequence.depth(key)` reads a `uint16 (H, W)` camera Z-depth image in millimeters
from `dataset_depth.npz`. Zero means no geometry. Convert units when comparing
with meter-valued BallaSim depth; a missing archive or key raises the underlying
file/lookup error.

(scene-sequence-mask)=
#### Released segmentation — `Sequence.mask`

`sequence.mask(key)` returns a `uint16 (H, W)` image containing object ID times
1000, with zero for no object. Decode the scaling before comparing it with
unscaled BallaSim instance IDs. The key is the archive key returned by
`gt_frames()`, not an arbitrary timestamp.

(scene-sequence-radtan)=
#### Released intrinsics — `Sequence.radtan`

`sequence.radtan()` returns `(fx, fy, cx, cy, dist)` from sequence metadata.
`dist: float64 (5,)` uses `[k1, k2, p1, p2, k3]`. Unsupported distortion models
and non-zero rational-radtan terms raise `ValueError`; no alternate calibration
file is silently substituted.

| Member | Description |
| --- | --- |
| `width`, `height`, `T_rig_cam`, `object_ids`, `low_light` | Camera and sequence metadata. |
| `track(key)` | `PoseTrack(rows, t_raw_ns, q, p)` for `"cam"` or an object ID string. |
| `gt_frames()` | `(archive_key, raw_time_ns, frame_metadata)` tuples. |
| `depth(key)` | `uint16 (H, W)` camera Z-depth in mm; zero on misses. |
| `mask(key)` | `uint16 (H, W)` encoded object ID times 1000. |
| `radtan()` | `(fx, fy, cx, cy, distortion)` from sequence metadata. |


## References

- [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).
  Imported geometry and material conventions.
- Burner et al., [EVIMO2](https://arxiv.org/pdf/2205.03467), 2022.
  Released measured-pose and scanned-object data.
- {download}`Scene-bundle implementation <../../../../../python/ballasim/scene/bundle.py>`.
  Record validation and persistence rules.
