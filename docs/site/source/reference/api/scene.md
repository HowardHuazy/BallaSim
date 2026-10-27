# Scene data and persistence

Scene objects are exported from `ballasim.scene`. Importers, authoring helpers,
and dataset adapters live in their respective submodules. See
[Scenes & assets](../../guide/scenes.md) for input formats and examples.

(scene-bundle)=
## Scene assembly — `SceneBundle`

```python
SceneBundle(name, t_start_ns, t_end_ns, origin_world, bodies, trajectories,
            cameras, light_groups, lights, meta={})
```

`t_start_ns < t_end_ns` defines the scene interval. `origin_world` is
`float64 (3,)`; `bodies`, `trajectories`, `cameras`, `light_groups`, and `lights`
contain the records below. `meta` must be JSON-serializable.

The bundle is the common geometry, calibration, lighting, and motion input to
rendering. Its list references are indices, while body IDs are persistent
ground-truth labels. Construct the dataclass first, then validate or save it;
construction alone does not repair inconsistent arrays or references.

(scene-bundle-validate)=
### Contract checks — `SceneBundle.validate`

`bundle.validate()` returns the same `SceneBundle` after checking dtypes,
shapes, rigid transforms, IDs, indices, trajectory support, waveforms, and
origin-relative reach. Every referenced trajectory must cover the scene span;
dynamic geometry must also remain within the 16 m support bound. Invalid input
raises `BundleError`. Validation does not triangulate or repair raw meshes;
those operations belong to the asset importer.

(scene-bundle-save)=
### Persistence — `SceneBundle.save`

`bundle.save(directory)` validates and returns a `pathlib.Path` to the written
bundle. It writes through a temporary sibling directory, then replaces an
existing bundle directory. An existing directory without `bundle.json` raises
`FileExistsError`. Use this to reuse a completed import or fit without rebuilding
its original source.

(scene-bundle-load)=
### Restoration — `SceneBundle.load`

`SceneBundle.load(directory, *, validate=True)` reconstructs arrays, textures,
and dataclasses from disk. It always checks format and feature declarations;
`validate=False` skips the final content checks and should be reserved for data
already trusted by the caller. It returns a `SceneBundle`; unsupported or
inconsistent format features raise `BundleError`.

(scene-bundle-as-dict)=
### Binding representation — `SceneBundle.as_dict`

`bundle.as_dict()` returns nested dictionaries and lists using the native
binding's field names. NumPy arrays remain shared with the bundle instead of
being copied. Use the result for APIs accepting the plain bundle representation;
mutating one of its arrays also changes the original bundle data.

(scene-bundle-features)=
### Format capabilities — `SceneBundle.features`

`bundle.features()` returns the sorted feature names required by its non-default
fields, such as baked radiance, piecewise trajectories, or PBR data. The writer
uses these declarations to select the format version and prevent an older
reader from silently ignoring unsupported content.

| Method | Returns | Description |
| --- | --- | --- |
| `validate()` | `SceneBundle` | Validate types, shapes, transforms, indices, support, and origin-relative reach; return `self`. |
| `save(directory)` | `pathlib.Path` | Validate and write a bundle, replacing an existing bundle directory. An existing non-bundle directory raises `FileExistsError`. |
| `SceneBundle.load(directory, *, validate=True)` | `SceneBundle` | Read the bundle and check its format/features; validate content unless disabled. |
| `as_dict()` | `dict` | Nested binding input; arrays are shared with the bundle. |
| `features()` | `list[str]` | Non-default format features used by the bundle. |

Validation failures raise `BundleError`, a `ValueError` subclass. On-disk bundles
contain `bundle.json`, `arrays.npz`, and textures. Version 2 declares optional
features; readers reject unknown or inconsistent feature declarations.

## Scene records

Construct these dataclasses with explicit arrays. Field dtypes are validated;
they are not implicitly converted. Indices refer to lists in the bundle.

(scene-mesh)=
### Triangle geometry — `Mesh`

```python
Mesh(positions, indices, normals=None, uvs=None, material_ids=None)
```

Stores triangles in one body's coordinate frame, allowing rigid motion without
rewriting vertex positions. Optional normals affect shading; UVs sample
material textures; per-triangle material indices select entries from the
containing body's material list. A missing normal or UV array is distinct from
an explicitly supplied array of zero values.

| Field | Type and shape | Description |
| --- | --- | --- |
| `positions` | `float32 (V, 3)` | Body-frame vertices in m. |
| `indices` | `uint32 (T, 3)` | Triangle vertex indices. |
| `normals` | `float32 (V, 3)`, optional | Shading normals. |
| `uvs` | `float32 (V, 2)`, optional | Texture coordinates; V increases down the image. |
| `material_ids` | `uint16 (T,)`, optional | Material index per triangle; `None` selects material 0. |

(scene-mesh-validate)=
#### Mesh checks — `Mesh.validate`

`mesh.validate(where="mesh", num_materials=None, *, carrier=False)` checks
array dtypes, lengths, index ranges, and non-empty geometry. Supplying
`num_materials` checks material indices against the containing list. Only a
validated moving light carrier may use an empty mesh. Errors raise
`BundleError`; `where` identifies the object in the error message.

(scene-mesh-used-materials)=
#### Referenced materials — `Mesh.used_materials`

`mesh.used_materials()` returns sorted unique `int64` material indices.
Without a material-ID array it returns `[0]`. Use this when applying material
operations only to surfaces present in the mesh, rather than unused material
slots.

(scene-material)=
### Surface response — `Material`

```python
Material(base_color, texture=None, emissive=..., shading="lit",
         baked_scale=1.0, baked_groups=None, texture_linear=None, texture_tile=0,
         roughness=1.0, metallic=0.0, specular=0.0,
         nir_form="assumed", nir_params=None, nir_texture=None)
```

A lit material represents reflectance under scene lights; a baked material
represents already illuminated radiance. Choose the latter for captured room
textures to avoid lighting them twice. Color multipliers and emission are
linear RGB; ordinary byte textures are decoded from sRGB before shading.

| Fields | Type and meaning |
| --- | --- |
| `base_color`, `emissive` | `float32 (3,)`, linear RGB. |
| `texture` | Optional `uint8 (H, W, 3)` sRGB texture. |
| `shading` | `lit` or `baked`; baked color represents radiance. |
| `baked_scale`, `baked_groups` | Non-negative scale and optional `float32 (G,)` shares summing to one. |
| `texture_linear`, `texture_tile` | Optional `float16 (H, W, 3)` baked texture and per-face tile pitch; linear and sRGB textures are mutually exclusive. |
| `roughness`, `metallic`, `specular` | GGX parameters in `[0, 1]`; lit materials only. |
| `nir_form`, `nir_params`, `nir_texture` | NIR form (`assumed`, `scale`, `affine`, `scalar`, `texture`), optional `float64 (2,)` parameters, and optional `uint8 (H, W)` linear albedo; lit materials only. |

The omitted emission default is a zero `float32 (3,)` array.
`has_lobe` reports whether metallic or specular is non-zero. `texture_size`
returns `(height, width)` or `None`.

(scene-material-validate)=
#### Material checks — `Material.validate`

`material.validate(where="material", num_groups=None)` checks color/texture
types, parameter bounds, and mutually exclusive texture representations. Baked
materials keep the default specular and NIR settings because their texture is
radiance, not albedo. A supplied `num_groups` checks baked-share length.
Invalid combinations raise `BundleError`.

(scene-material-has-lobe)=
#### Specular classification — `Material.has_lobe`

A boolean property indicating positive metallic or specular response. It
identifies whether the material has a GGX lobe independently of its roughness.
The renderer's reflection policy makes a separate decision about whether that
lobe is acceptable without secondary reflection rays.

(scene-material-texture-size)=
#### Texture extent — `Material.texture_size`

A property returning `(height, width)` for the active byte or linear texture,
or `None` when untextured. Use it without assuming which color encoding the
material stores; tile atlases are still reported by their packed image extent.

(scene-body)=
### Scene instance — `Body`

```python
Body(id, name, semantic, mesh, materials, trajectory, T_world_body,
     scan=False, normal_prefilter_m=0.0)
```

`id` is unique in `[1, 0xFFFFFFFF)`; `semantic` is in `[0, 0xFFFF)`, with `0`
unlabeled. `trajectory=None` selects the rigid `float64 (4, 4)` static transform;
otherwise it indexes `bundle.trajectories` and `T_world_body` must be identity.
`scan=True` marks static captured geometry with baked materials.
`normal_prefilter_m` sets the specular normal-filter radius in m.

The body joins geometry, material slots, motion, and output identity. Assign
distinct IDs even when bodies reuse an asset; semantic IDs may be shared by
several instances of one class. `scan` distinguishes captured geometry from
inserted blockers for proxy-shadow handling. A non-zero normal-filter radius
reduces the effect of small-scale scan-normal variation on specular shading.

(scene-body-validate)=
#### Body checks — `Body.validate`

`body.validate(where, num_trajectories, num_groups=None, *, carries_light=False)`
checks IDs, material slots, transforms, and trajectory references. It also
enforces that captured scan bodies are static with baked materials.
`carries_light=True` permits the special moving meshless carrier case; a bundle
sets this from actual light references. Returns `None` or raises `BundleError`.

(scene-trajectory)=
### Spline storage — `Trajectory`

```python
Trajectory(K, t0_ns, h_ns, rot, pos, holds, span_ns, pieces=None)
```

`K` is `4` or `6`; `h_ns` is an integer in `[1, 2**31)`. Controls are
`rot: float64 (C, 4)` unit quaternions and `pos: float64 (C, 3)` positions relative
to `origin_world`. Adjacent control rotations differ by at most 90°.
`holds: int64 (H, 2)` stores disjoint `[begin, end)` intervals of exact stillness.
`span_ns` is the valid scene interval.

Optional `pieces: int64 (P, 6)` uses rows
`(begin_ns, t0_ns, h_ns, ctrl_begin, ctrl_end, join)`; control ranges are
half-open. Join kinds are `-1` (step), `0` (C0), and `1` (C1); the first piece uses
`0`. Pieces are
right-continuous. The engine rejects pose steps.

Controls describe the spline and generally do not lie on its evaluated path.
Use [fitting methods](trajectories.md) to produce them from poses. The declared
`span_ns` identifies the scene interval, while spline support may extend beyond
it for evaluation and boundary handling.

(scene-trajectory-num-segments)=
#### Segment count — `Trajectory.num_segments`

Returns `len(rot) - K + 1` for a single spline. It raises `ValueError` for
piecewise storage because each piece has its own controls and knot spacing.

(scene-trajectory-end-ns)=
#### Support endpoint — `Trajectory.end_ns`

Returns the closed right endpoint of the final serving spline. For a single
spline it is `t0_ns + num_segments * h_ns`. This endpoint need not equal the
declared scene end if the grid was extended to cover it.

(scene-trajectory-piece-support)=
#### Piece interval — `Trajectory.piece_support`

`traj.piece_support(p)` returns `(begin_ns, end_ns)` for the closed support of
piece index `p`. This is the piece's full evaluable knot interval, which may be
wider than the time interval assigned to it by adjacent breakpoints.

(scene-trajectory-hold-ranges)=
#### Controls under a hold — `Trajectory.hold_ranges`

`traj.hold_ranges(t_begin, t_end)` returns inclusive control-row ranges whose
bases affect the half-open interval. There is one range per serving piece.
Tying those controls to identical poses enforces exact stillness; changing only
sampled poses would not guarantee zero derivatives between samples.

(scene-trajectory-validate)=
#### Trajectory checks — `Trajectory.validate`

`traj.validate(where="trajectory")` checks supported order/spacing, unit
control rotations, adjacent increments, support, and declared holds. Piecewise
tables must cover their intervals with contiguous control ranges. It returns
`None` or raises `BundleError`; it does not fit or normalize arbitrary controls.

| Property or method | Meaning |
| --- | --- |
| `num_segments` | Segment count for a single-piece trajectory; raises `ValueError` for multiple pieces. |
| `end_ns` | Closed right endpoint of spline support. |
| `piece_support(p)` | Closed support of piece `p`. |
| `hold_ranges(t_begin, t_end)` | Inclusive control-index ranges affecting the half-open interval, one per piece. |
| `validate(where="trajectory")` | Check trajectory invariants; raise `BundleError` on failure. |

Use the [trajectory API](trajectories.md) to fit and evaluate trajectories.

(scene-camera)=
### Camera calibration — `Camera`

```python
Camera(name, model, width, height, fx, fy, cx, cy, dist, rig_trajectory,
       T_rig_cam, clock_offset_ns, sensor, pixel_pitch_m, f_number=2.0)
```

`model` is `radtan` or `kb`. Intrinsics are in pixels; `dist: float64` contains
`[k1, k2, p1, p2, k3]` for radtan or `[k1, k2, k3, k4]` for Kannala–Brandt.
`rig_trajectory` indexes a trajectory; `T_rig_cam` is a rigid `float64 (4, 4)`
camera-to-rig transform. Sensor time is true time plus `clock_offset_ns`.
`pixel_pitch_m` and `f_number` must be positive.

The camera follows a rig trajectory, with a fixed extrinsic transform defining
its position and orientation in the rig. Intrinsics map camera directions to
pixels; pixel pitch and f-number additionally enter physical photometry and
optical blur. The `sensor` string records the source camera's sensor identity;
run configurations still select `sensor.family` explicitly.

(scene-camera-validate)=
#### Camera checks — `Camera.validate`

`camera.validate(where, num_trajectories)` checks dimensions, positive focal
lengths/pitch/f-number, distortion length, rigid extrinsics, and rig index.
Returns `None` or raises `BundleError`. Projection-domain checks are performed
when the shared camera model is constructed; a structurally valid distortion
array is not proof of an invertible image mapping.

(scene-light-group)=
### Temporal illumination — `LightGroup`

```python
LightGroup(name, waveform)
```

Group 0 must be `LightGroup("default", {"type": "constant"})`. Other waveforms
accept `fourier` (`f0_hz`, `a`, `phi`) or `pulse` (`f_hz`, `duty`, `rise_s`,
`phase_s`, `floor`). Fourier amplitudes satisfy `sum(abs(a)) <= 1`; pulse duty is
the full width at half maximum divided by the period.

A group separates spatial shading from shared illumination timing. Each
member light is shaded at its mean intensity, and the common unit-mean
waveform modulates that contribution. Changing the waveform therefore changes
event timing without requiring a separate spatial material for each light.

(scene-light-group-validate)=
#### Waveform checks — `LightGroup.validate`

`group.validate(where)` checks the name and waveform, returning `None` or
raising `BundleError`. Fourier lists must match and preserve non-negative
illumination; pulse ramps, width, and floor must define a valid unit-mean pulse.

(scene-light)=
### Spatial illumination — `Light`

```python
Light(kind, group, color, intensity, body=None, position=...,
      direction=..., spot_inner_rad=0.0, spot_outer_rad=0.0,
      casts_shadows=True, proxy=False, size=None, axis=None)
```

The position default is `[0, 0, 0]`; the direction default is `[0, 0, -1]`.

`kind` is `point`, `spot`, `directional`, `ambient`, `background`, `rect`, or
`disk`. `group` indexes light groups; `body` optionally indexes a carrier body.
`color` is `float32 (3,)`; `position` and unit `direction` are `float64 (3,)` in
the carrier frame, or origin-relative world frame without a carrier. Area lights
require positive `size: float64 (2,)` and an in-plane unit `axis` (default +X).
For disks, `size` gives ellipse diameters. Proxy lights represent static captured
illumination and require scan geometry.

`rect` and `disk` records are valid bundle data; the current renderer refuses
area lights. The Blender converter also rejects them.

(scene-light-validate)=
#### Light checks — `Light.validate`

`light.validate(where, num_groups, num_bodies)` checks references, light
geometry, cone angles, and proxy restrictions; it returns `None` or raises
`BundleError`. Structural acceptance of an area-light record does not imply
support in the current renderer.

(scene-static-trajectory)=
### Constant pose — `static_trajectory`

`ballasim.scene.bundle.static_trajectory(q, p, span_ns, *, K=6, h_ns=None)`
returns a `Trajectory` holding one normalized scalar-first quaternion `(4,)`
and position `(3,)` exactly over the requested interval. `h_ns=None` chooses
a valid covering grid automatically. Use it for a stationary rig when a
camera still needs a trajectory reference; the interval must have positive
duration.

(scene-meshless-carrier)=
### Moving light frame — `meshless_carrier`

`ballasim.scene.bundle.meshless_carrier(id, name, trajectory, semantic=0)`
returns a `Body` with no triangles and an identity fixed transform, carried by
the specified trajectory index. It provides a pose for an attached moving light
without introducing a visible object. The bundle must actually attach a light
to this body for its empty geometry to validate.


## References

- [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).
  Imported geometry and material conventions.
- Burner et al., [EVIMO2](https://arxiv.org/pdf/2205.03467), 2022.
  Released measured-pose and scanned-object data.
- {download}`Scene-bundle implementation <../../../../../python/ballasim/scene/bundle.py>`.
  Record validation and persistence rules.
