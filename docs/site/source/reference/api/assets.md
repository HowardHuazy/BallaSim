# Asset import and scene authoring

Import geometry with its source units and appearance, then assemble a validated
scene. These methods implement the [Scenes & assets](../../guide/scenes.md)
workflow.

(scene-load-asset)=
## Mesh import — `load_asset`

```python
from ballasim.scene.assets import load_asset

load_asset(path, *, format=None, scale=None, up_axis="+z", node=None,
           uv_origin="bottom", max_texture_size=4096, shading="auto",
           pbr=False, tile_texels="auto", workers=1, use_cache=True)
```

Return `Asset(mesh, materials, info)` from glTF, OBJ, PLY, or a Replica directory.
`scale` converts source units to meters and is required for OBJ/PLY. `format`
overrides extension detection. `shading` accepts `auto`, `lit`, or `baked`;
`pbr=True` is glTF-only. `tile_texels` accepts `auto`, `4`, `8`, or `16` for
Replica; `workers` controls its packing threads. Unsupported inputs raise
`AssetError`; missing files raise `FileNotFoundError`.

Import converts source coordinates and materials to BallaSim conventions,
removes degenerate/duplicate triangles, and records repairs in `info`.
`max_texture_size` downsamples ordinary textures in linear light; it trades
texture detail for memory. `node` selects one glTF subtree. The source asset and
dependent buffers/textures must remain available for content checks, even when
an imported result is cached.

(scene-asset)=
### Import result — `Asset`

`Asset(mesh, materials, info={})` contains one imported `Mesh`, its `Material`
slots, and a diagnostics dictionary. Geometry remains in the asset frame; body
placement is a later operation. `info` includes source/cache information and
any dropped features, repairs, or approximation warnings, so review it before
treating an import as a faithful source conversion.

(scene-cached-key)=
### Existing import lookup — `cached_key`

`cached_key(path, **options)` accepts the same import options except `use_cache`
and returns an existing cache key or `None`. The key includes source content,
dependency content, importer version, and output-relevant options. Missing source
files or a missing cache entry return `None`; invalid options still raise.
This does not import or build the asset, but may read files to hash dependencies.

(scene-is-cached)=
### Cache availability — `is_cached`

`is_cached(path, **options)` returns whether `cached_key()` finds an entry.
It accepts the same options and performs the same source checks. Use it for
availability reporting when the key itself is not needed.

(scene-load-scene)=
## YAML assembly — `load_scene`

```python
from ballasim.scene.authoring import load_scene, asset_specs

load_scene(path, *, files=None)
```

`load_scene()` returns a validated `SceneBundle`. If supplied, `files` is extended
with resolved asset and texture paths. The loader applies body-frame choices,
fits keyframes, assembles rigs/lights, and records source attribution. Relative
paths resolve beside the YAML file; environment variables override declared
scene variables. Malformed scene specifications raise
`SceneSpecError`; accepted approximations may issue `SceneWarning`.

(scene-asset-specs)=
### Declared assets — `asset_specs`

`asset_specs(path)` returns a list of `(pathlib.Path, options_dict)` pairs for
the asset imports requested by a scene YAML. It validates the source's schema
and asset entries without loading meshes or building trajectories. Use it to
inspect dependencies or query import-cache availability before assembly.


## References

- [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).
  Imported geometry and material conventions.
- Burner et al., [EVIMO2](https://arxiv.org/pdf/2205.03467), 2022.
  Released measured-pose and scanned-object data.
- {download}`Scene-bundle implementation <../../../../../python/ballasim/scene/bundle.py>`.
  Record validation and persistence rules.
