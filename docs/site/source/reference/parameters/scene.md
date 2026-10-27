# Scene source and appearance

Scene scope: top-level `scene.*`. See [Scenes & assets](../../guide/scenes.md)
for source formats.


## Parameter details

(scene-path)=
### Scene input — `scene.path`

Selects the geometry and motion source before run parameters are resolved.
Use a scene YAML file, a saved bundle directory, or a Blender exchange directory.
The loader identifies the format from the path contents and obtains camera names,
calibration, and trajectory support from that source. An in-memory `SceneBundle`
or `scene.evimo2.sequence` can supply the scene instead.

(scene-background-radiance)=
### Miss radiance — `scene.background_radiance`

Adds a spatially uniform background light for rays that miss every mesh. This
controls the contrast at silhouettes against otherwise empty space; it does not
add geometry, depth, or an instance label. A value of zero adds no background
light and preserves any background already present in the bundle. Its units
follow the run's photometry, and its modulation follows `background_group`.

(scene-background-group)=
### Background modulation — `scene.background_group`

Names the existing light group assigned to the added background. Use `default`
for constant radiance, or a waveform group when the background should share the
scene's illumination flicker. The group must exist even when the added radiance
is zero. EVIMO2 sources use their separate `background_albedo` control.

(scene-baked-scale)=
### Captured-light scale — `scene.baked_scale`

Multiplies every baked material's radiance scale and every proxy light's
intensity by the same factor. Applying both preserves the relative brightness
of the captured room and objects lit by its proxy lighting. Lit materials and
ordinary lights are unchanged. Under reference photometry, a common scale is
canceled when all illumination comes from proxies and no lit emission supplies
an independent brightness reference.

(scene-allow-missing-reflections)=
### Reflection approximation — `scene.allow_missing_reflections`

Allows specular materials that the default reflection policy rejects: roughness
below `0.2`, or below `0.5` when metallic is positive. The direct GGX response is
still evaluated, but secondary reflection rays are absent. Enabling this option
accepts that omission; it does not add a reflection pass. The run records the
affected materials and reflection bounds for review.

## Quick lookup

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `scene.path` | `path` | required | Scene YAML, bundle directory, or Blender exchange directory; required unless an EVIMO2 source or in-memory bundle is supplied. |
| `scene.background_radiance` | `float` | `0.0` | Radiance on geometry misses for `scene.path` sources; `0` leaves the background unchanged. Range: ≥0. |
| `scene.baked_scale` | `float` | `1.0` | Common multiplier for baked scan radiance and its proxy lights. Range: ≥0. |
| `scene.background_group` | `string` | `"default"` | Existing light group controlling background radiance. |
| `scene.allow_missing_reflections` | `bool` | `false` | Allow materials below the reflection-policy roughness limits; omitted reflections are reported in `run.json`. |

## Source constraints

Supply `scene.path`, `scene.evimo2.sequence`, or an in-memory bundle. The run
resolver identifies YAML, saved bundles, and Blender exchanges from the source.
`background_group` must name an existing light group. Background controls for
`scene.path` sources are inactive for EVIMO2, which uses its own lighting fields.

Without `allow_missing_reflections`, specular roughness below `0.2`, or metallic
roughness below `0.5`, is rejected. Enabling it accepts the omitted-reflection
approximation. A common `baked_scale` is ineffective under reference photometry
when every light is a proxy and no lit emission provides an independent scale.

## References

- Debevec, [Rendering Synthetic Objects into Real Scenes](https://www.pauldebevec.com/Research/IBL/debevec-siggraph98.pdf),
  SIGGRAPH 1998. Background on combining captured illumination with inserted geometry.
- [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).
  Material and asset conventions used by the importer.
