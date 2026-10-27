# Scenes & assets

A scene defines geometry, materials, rigid-body motion, cameras, and lighting.
Run configurations select the scene through `scene.path` or `scene.evimo2.sequence`.

Importing assembles these inputs into one `SceneBundle`. Geometry is stored in
each body's local frame; a fixed transform or continuous-time trajectory places
that body in the world when it is queried. Cameras and lights reference the same
scene, so visibility, illumination, and ground truth use consistent poses.

| Source | Input |
| --- | --- |
| Authored scene | YAML file with `schema: ballasim.scene/1` |
| Scene bundle | Directory containing `bundle.json` and `arrays.npz` |
| Blender exchange | Directory containing `exchange.json`, exported by the separate Blender add-on |
| EVIMO2 sequence | Camera sequence directory selected with `scene.evimo2.sequence` |

## Coordinate conventions

Lengths are in meters; the world is right-handed with +Z up. Cameras use +X right,
+Y down, and +Z forward. Pixel centers have integer coordinates. Quaternions use
`(w, x, y, z)` order; `T_a_b` maps coordinates from frame `b` to frame `a`.

Scene YAML positions are absolute world coordinates. Bundle positions are relative
to `origin_world`; choose an origin within 16 m of all geometry and trajectory
control positions. Time is represented internally as integer nanoseconds.

## Author a scene

Save the following as `scene.yaml`. Asset paths are relative to this file.

```yaml
schema: ballasim.scene/1
name: moving_object
time: {t_start_s: 0, t_end_s: 1}
assets:
  object: {path: object.glb, license: CC0-1.0}
bodies:
  - name: object
    id: 1
    asset: object
    keyframes:
      - {t_s: 0, t_m: [0, 0, 0]}
      - {t_s: 1, t_m: [0.2, 0, 0]}
rigs:
  camera_rig:
    pose: {eye_m: [0, -2, 1], target_m: [0, 0, 0]}
cameras:
  - {name: camera, preset: evimo2_left, rig: camera_rig}
lights:
  - {kind: ambient, intensity: 1}
```

Each body requires a unique positive `id`, an asset, and exactly one of `pose` or
`keyframes`. Add `semantic_classes: {object: 1}` and `semantic: object` to assign a
class; class `0` is unlabeled. Body `origin` selects `asset` (default), `bottom`,
or the area-weighted `centroid` before placement.

Use `variables` and `${NAME}` for reusable paths. Environment variables override
variables defined in the scene. Unknown scene keys raise `SceneSpecError`.

```python
from ballasim.scene.authoring import load_scene

bundle = load_scene("scene.yaml")
bundle.save("scene_bundle")
```

See [Trajectories](trajectories.md) for keyframes and holds, and
[Configuration](configuration.md) for selecting the sensor and photometry.
The [scene settings](../reference/parameters/scene.md) describe source selection,
background radiance, and captured-light scaling.

## Import assets

`ballasim.scene.assets.load_asset()` returns an `Asset` with `mesh`, `materials`,
and import diagnostics in `info`.

| Format | Import options and constraints |
| --- | --- |
| glTF 2.0 / GLB | Meters; node transforms are applied. `node` selects a subtree; `pbr: true` imports metallic-roughness parameters. |
| OBJ / MTL | Explicit `scale` required. Reads material colors and base-color textures. |
| PLY | Explicit `scale` required. Reads normals, UVs, and supported texture atlases. |
| Replica directory | `format: replica`; meters, +Z up, baked radiance. `tile_texels` accepts `auto`, `4`, `8`, or `16`. |

`scale` converts file units to meters. `up_axis` sets the source axis for OBJ/PLY;
`uv_origin` selects plain PLY texture orientation. `max_texture_size` limits
ordinary texture dimensions by downsampling in linear light.

Importers remove degenerate and duplicate triangles, generate missing normals,
and record repairs in `info`. Unsupported glTF skins, compressed geometry, sparse
accessors, and varying vertex colors are rejected. Keep external buffers, MTL
files, and textures with the source asset.

Imported assets are cached by content and import options beneath
`$BALLASIM_CACHE/v1/assets`. Preserve source licenses and attribution in the
scene's asset entries. Replica entries require `license: Replica-Research`.

See {ref}`import methods <scene-load-asset>` for import options,
cache queries, and scene-bundle persistence.

## Materials and lighting

Ordinary materials use linear RGB base color, optional sRGB textures, and
emission. Lit materials also support GGX `roughness`, `metallic`, and `specular`
parameters in `[0, 1]`; zero metallic and specular give Lambertian reflection.
Near-infrared albedo is selected with `nir_form`, `nir_params`, or `nir_texture`;
its contribution is controlled per camera by `sensor.nir_fraction`.

`material_overrides` maps material indices to replacement properties. The
procedural `room` uses named materials for its walls, floor, and ceiling.

| Light kind | Meaning |
| --- | --- |
| `point`, `spot` | Positioned source; a spot also requires inner and outer cone angles. |
| `directional` | Distant source with a fixed direction. |
| `rect`, `disk` | One-sided area emitter; requires `size_m` and an in-plane `axis` perpendicular to `direction`. |
| `ambient` | Constant radiance term on surfaces. |
| `background` | Radiance seen by rays that miss geometry. |

Area-light records can be imported and saved, but the current renderer refuses
`rect` and `disk` lights.

Lights may attach to a body by name. Light group `default` is constant; named
groups accept constant, Fourier, or pulse waveforms. Materials and lights use
relative units unless the run selects physical photometry. Under physical
photometry, point/spot intensities are in cd, directional intensity in lx, and
area, ambient, background, and emission radiance in cd/m².

The renderer does not model general interreflection. Runs reject specular
roughness below `0.2`, or below `0.5` for metals, unless
`scene.allow_missing_reflections: true` explicitly accepts the approximation.

The {ref}`reflection setting <scene-allow-missing-reflections>`
describes this acceptance policy and its recorded limits.

## Scanned rooms

Use `shading: baked` for textures that already contain captured illumination.
`shading: auto` detects glTF `KHR_materials_unlit`; other materials remain lit.
Baked textures are interpreted as radiance and receive no additional lighting.

A baked asset requires `scale_basis: metric-source`, `measured`, or `unverified`.
Mark captured room bodies with `scan: true`; scan bodies must be static and use
baked materials. `baked: {scale: 1, groups: {default: 1}}` sets radiance scale and
light-group shares. Shares must sum to one.

Static lights with `proxy: true` approximate the captured illumination for
inserted objects and their shadows on the scan. Existing scan shadows remain in
the texture. Proxy lights require a scan body.

## Blender and EVIMO2

Point `scene.path` at a Blender exchange directory. Static exchanges convert
directly; animated exchanges require a completed motion-sampling session from
the add-on. Inspect conversion reports with:

```bash
ballasim blender check --exchange scene.bsx --json
```

Reports identify unsupported geometry, material graphs, camera models, and
animation. The Blender converter currently rejects area lights, animated
intrinsics, animated light properties, and visibility changes. Physical
photometry additionally requires `scene.blender.lm_per_w`.

Use [Blender settings](../reference/parameters/blender.md) to control conversion
policy, motion accuracy, and camera-fit tolerances.

For EVIMO2, select a sequence with `scene.evimo2.sequence`. The adapter loads scanned
objects, camera calibration, clock offsets, and fitted motion-capture tracks.
Object assets and any raw archive required by the selected clock reference must
be available locally. Configure adapter choices under `scene.evimo2` and fitting
under `scene.trajectory`.

[EVIMO2 settings](../reference/parameters/evimo2.md) explain mesh selection,
timestamp conventions, and the point-plus-ambient lighting approximation.

See [scene data and persistence](../reference/api/scene.md) for bundle persistence and import
interfaces.

## References

- [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).
  Asset, texture, and material conventions.
- Burner et al., [EVIMO2](https://arxiv.org/pdf/2205.03467), 2022.
  Scanned geometry and measured motion used by the dataset adapter.

```{toctree}
:hidden:
:maxdepth: 1

Scene data and persistence <../reference/api/scene>
Asset import and authoring <../reference/api/assets>
Scene settings <../reference/parameters/scene>
Blender import <../reference/parameters/blender>
EVIMO2 reconstruction <../reference/parameters/evimo2>
Rendering methods <../reference/api/rendering>
```
