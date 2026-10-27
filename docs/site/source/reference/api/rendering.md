# Rendering and geometric queries

`ballasim._render` evaluates a posed scene at requested camera, pixel, and time
coordinates. It integrates a finite pixel response with fixed scrambled
subsamples and returns separate mean-illumination contributions for each light
group. The sensor stage subsequently applies illumination waveforms,
photocurrent conversion, and threshold crossing.

The renderer includes direct diffuse/GGX shading, baked radiance, and supported
shadow/AO approximations. It does not trace general scene reflections or
interreflection. Although scene bundles can store `rect` and `disk` emitters,
this build rejects area lights when constructing a renderer.

(render-backends)=
## Backend availability — `backends`

`_render.backends()` returns the usable backend names in the current process:
`embree`, and `optix` when the optional module, CUDA device, and driver support
are available. Direct renderer construction requires one of these explicit
names; it does not select a CPU fallback for an unavailable OptiX request.

(render-renderer)=
## Scene renderer — `Renderer`

```python
from ballasim import _render

renderer = _render.Renderer.from_values(bundle, resolved.values())
renderer.build()
```

Owns the converted scene, acceleration structures, pixel-response tables, and
pose cache. `bundle` may be a `SceneBundle` or an equivalent nested mapping of
arrays. Query results preserve request order and do not depend on batch size or
thread count. Each pixel uses deterministic subsample positions from its pattern
seed; temporal queries evaluate the geometry's continuous-time poses.

For explicit construction, the main defaults are:

```python
Renderer(bundle, backend="optix", n_subsamples=64,
         sensor_rgb_weights=(0.52, 0.37, 0.11), *, seed=0,
         aperture_px=1.0, blur_sigma_px=0.15, truncation_sigmas=3.0,
         warp_cells=256, lut_samples_per_px=2, num_threads=0,
         max_rays_per_launch=16777216, pixel_integral="phase1",
         unlit_mask=False, texture_shadow="pooled",
         shading_patches="material", mover_ao="off", nir_fraction=[])
```

These are constructor defaults, not the resolved fidelity preset. `n_subsamples`
is in `[1, 1024]`; powers of two preserve nested sampling patterns. Aperture and
blur define the optical integration support in pixels. RGB weights form the
monochrome material response; `nir_fraction` supplies one light-group fraction
row per bundle camera, with empty rows meaning no NIR contribution.

(render-from-values)=
### Resolved settings — `Renderer.from_values`

`Renderer.from_values(bundle, values)` returns an unbuilt renderer using the
`{run: {...}, cameras: {name: {...}}}` mapping returned by
`Resolved.values()`. It applies spatial sampling, pixel response, execution
settings, backend selection, and each camera's NIR fractions by name.
Run-wide camera settings must agree across the selected cameras. This method
does not apply event thresholds or convert renderer values into photocurrent.

(render-build)=
### Preprocessing — `Renderer.build`

`renderer.build()` constructs trace geometry, acceleration structures, linear
monochrome mip textures, camera-ray lookup tables, and sample seeds. It returns
`None` and is also called implicitly by the first query. Build errors identify
invalid scene/options or unavailable backend resources. Repeated queries reuse
these structures and evaluate only the required poses.

(render-render)=
## Pixel-response integration — `Renderer.render`

```python
renderer.render(requests, *, sub_instance=False, centre=False,
                sample_class=False, pv=None)
```

Accepts a structured array or mapping with equal-length one-dimensional fields
`cam: uint16`, `pixel: uint32`, and `t_ns: int64`. Camera values are bundle
indices; pixel ID is `y * width + x`; times use the scene's true clock.
Returns a dictionary with integrated `irradiance`, a visibility `signature`,
and timing `stats`. Invalid camera/pixel indices or unsupported query times raise
`ValueError`.

Each row approximates the pixel-area mean at one instant. Contributions are in
linear renderer units per light group, before the group's time waveform;
baked materials contribute their captured radiance through the same interface.
`signature` encodes sampled surface/shadow state for change detection. It is
not an instance ID or a collision-free segmentation label.

`sub_instance=True` exposes the instance at each optical subsample. `centre=True`
adds a separate center-ray geometric query; its label need not match every
subsample near an edge. `sample_class=True` adds packed geometry, shadow-edge,
texture, and specular classification bits used by adaptive sampling.

`pv="edge"` or `"all"` additionally requests packed visibility predictor codes
and cluster tables. These diagnostics support at most eight shadow lights and
are rejected with specular materials, traced mover AO, or two-channel NIR on
OptiX. `pv_written` marks rows with stored entries; unwritten rows are zeroed.

| Output | Dtype and shape | Meaning |
| --- | --- | --- |
| `irradiance` | `float32 (N, G)` | Pixel integral per light group. |
| `signature` | `uint64 (N,)` | Packed visibility/shadow change signature. |
| `sub_instance` | `uint32 (N, S)` | Optional instance IDs for S optical subsamples. |
| `centre_instance`, `centre_semantic` | `uint32 (N,)`, `uint16 (N,)` | Optional center-ray labels. |
| `centre_depth`, `centre_flow` | `float32 (N,)`, `float32 (N, 2)` | Optional Z-depth in m and raw-image flow in px/s. |
| `sample_class` | `uint32 (N,)` | Optional packed sampling classification. |
| `pv_written` | `uint8 (N,)` | Whether predictor state was written. |
| `pv_codes`, `pv_table` | `uint32 (N, words)` | Packed predictor arrays; width depends on sampling, lights, and groups. |

`stats` contains `trace_ms`, `shade_ms`, and `host_ms`. These are stage timings
for the call, not per-request latencies.

(render-gt)=
## Instantaneous geometry — `Renderer.gt`

```python
renderer.gt(cam, x, y, t_ns, *, states=False, normals=False)
```

Queries rays through distorted image coordinates in one camera. `x` and `y`
are equal-length one-dimensional floating arrays; `t_ns` is an equal-length
`int64` array. Integer image positions are pixel centers. The returned depth
is camera Z in meters; flow is the instantaneous derivative of raw pixel
position in px/s, computed from relative body/camera twist and the projection
Jacobian. It is independent of the time spacing between query rows.

Returns `instance: uint32`, `semantic: uint16`, and `depth`, `flow_x`, `flow_y`
as `float32`, all shape `(N,)`. Misses have instance/semantic zero and NaN depth
and flow. Positions must lie within the camera's checked image support and
all referenced motion must support the requested times.

`states=True` adds `uint32 (N, L, 2)` shadow-state pairs: state code and the
blocker's float32 ray-parameter bits for each shadow light. `normals=True` adds
`normal: float32 (N, 3)`, the hit triangle's geometric unit normal in the camera
frame, oriented toward the camera and NaN on misses. These optional plan-pass
outputs require Embree; the basic geometry query supports both backends.

(render-dense)=
## Image-shaped queries — `Renderer.dense`

```python
renderer.dense(cam, t_ns, channels=["depth", "instance", "flow", "irradiance"])
```

Evaluates every pixel of one camera at one scalar true-clock time and returns
only the requested channels. Geometric channels use center rays; irradiance
uses the same finite-pixel integral as `render()`. This is an instantaneous
query, not an exposure-integrated frame or an event-sensor simulation.
Unknown channel names raise `ValueError`.

| Channel | Dtype and shape |
| --- | --- |
| `depth` | `float32 (H, W)`, camera Z in m |
| `instance`, `semantic` | `uint32 (H, W)`, `uint16 (H, W)` |
| `flow` | `float32 (H, W, 2)`, raw px/s |
| `irradiance` | `float32 (H, W, G)` |
| `shadow_factor` | `float32 (H, W, G)` |

`shadow_factor` is the ratio to irradiance with all baked-proxy shadow factors
set to one. It is one where that denominator is zero, and throughout scenes
without baked proxies. It is not a general shadow-visibility image for ordinary
lit surfaces.

(render-probe)=
## Individual subsamples — `Renderer.probe`

`renderer.probe(requests)` accepts equal-length fields `cam: uint16`,
`pixel: uint32`, `k: uint32`, `t_ns: int64`, and `light: uint32`. `k` selects a
fixed optical subsample in `[0, 1024)`; `light` indexes shadow lights rather than
all bundle lights. The query returns the state of that one sample without
integrating its full pixel response. Probes are unavailable with traced mover AO.

Returns `(N,)` arrays `instance`, `surface`, and `mask` (`uint32`), plus
`f: float32` and `stats` containing `probe_ms`/`host_ms`. `surface` is the material
or crease-patch key, depending on renderer settings. `mask` includes the same
unlit-light filtering as the visibility signature. `f` is the selected light's
Lambert irradiance term, not the complete shaded radiance or a GGX BSDF value;
it is zero for baked receivers. The extra static-occlusion word used by baked
signatures is not included in probe output.

(render-set-window)=
## Time-cache base — `Renderer.set_window`

`renderer.set_window(t_base_ns)` sets the base of internal signed 32-bit time
offsets and clears the lazily filled pose cache. It returns `None`.
Group nearby queries under a suitable base to retain pose reuse; requests too
far from the base automatically rebase. This is a cache hint, not a crop of the
allowed scene interval or a change to the motion clock.

(render-hold)=
## Held-body acceleration — `Renderer.hold`

`renderer.hold(ids, t_ns)` merges the specified moving body IDs into static
trace geometry at their poses at `t_ns`, returning whether geometry was rebuilt.
The caller must guarantee exact stillness throughout every later queried time
until the next `hold()` call. An empty list restores the moving geometries;
unknown/non-moving IDs are ignored.

Held geometry can change coincident-surface tie choices. Proxy mover AO retains
its held-body proxies, whereas traced mover AO rejects holding participating
movers. Scenes with baked proxies keep inserted blockers separate from scan
geometry: `holds_movers` is false and `hold()` returns false without merging.

(render-proxy-visibility)=
## Captured-light diagnostic — `Renderer.proxy_visibility`

`renderer.proxy_visibility(cam, t_ns)` returns a Python list of blocked-ray
fractions, one per proxy shadow light in bundle order. From camera center-ray
hits on baked receivers, it tests proxy rays against the scan alone.
A fraction near one can identify a proxy placed inside captured fixture geometry.
It does not measure the final inserted-object shadow factor; scenes without
baked proxies return an empty list.

(render-split-geometries)=
## Scan/blocker split — `Renderer.split_geometries`

`renderer.split_geometries()` builds the scene if needed and returns the number
of static bodies separated into their own trace geometries for baked-proxy
shadow handling. They remain distinct from scan geometry so inserted blockers
can cast new shadows without reapplying captured scan shadows. The result is
zero without baked proxies.

(render-properties)=
## Renderer state and capabilities

These read-only properties describe configuration or capability; they do not
perform a render.

| Property | Meaning |
| --- | --- |
| `backend` | Active `embree` or `optix` backend. |
| `built` | Whether preprocessing has completed. |
| `window` | Current internal time-offset base in ns. |
| `n_subsamples` | Configured optical sample count. |
| `num_groups`, `light_groups` | Group count and ordered group names. |

(render-holds-movers)=
### Static merging — `Renderer.holds_movers`

Reports whether held movers may be merged with the world. It is false for
baked-proxy scenes, which require inserted geometry to remain separately
addressable for shadow tests. Individual AO constraints still apply to `hold()`.

(render-has-specular)=
### Specular content — `Renderer.has_specular`

Reports whether the scene contains a material with a specular lobe. It describes
material content, not whether secondary reflections are rendered. Predictor
visibility tables use Lambert terms and reject specular scenes.

(render-nir-two-channel)=
### Spectral representation — `Renderer.nir_two_channel`

Reports whether camera/light-group NIR fractions differ and require
separate visible and NIR texture channels. Equal fractions can be folded into
one effective albedo representation. Querying it validates fraction shapes and
bounds even before scene preprocessing.

(render-pv-supported)=
### Predictor kernel availability — `Renderer.pv_supported`

Reports whether the backend has a predictor-output kernel for its spectral
representation. It is false for two-channel NIR on OptiX. A true value does not
waive the separate limits on shadow-light count, specular materials, or traced AO.

(render-shading-patches)=
## Surface grouping — `shading_patches`

```python
_render.shading_patches(positions, indices, normals=None,
                        materials=None, specular=None)
```

Returns `uint32 (T,)` connected shading-patch IDs numbered by first triangle.
Inputs are vertex positions `(V, 3)`, triangle indices `(T, 3)`, optional vertex
normals `(V, 3)`, and material indices `(T,)`. `specular` is an optional per-material
boolean/byte array identifying lobes; absent material indices mean material zero.

Patches split at material changes and disconnected shells. At declared hard
edges they split when the dihedral exceeds 30° or a neighboring material is
specular; without vertex normals they split above 30°. Equal-position UV seams
with equal normals remain connected. This exposes the grouping used by
`shading_patches="crease"`; it does not modify or subdivide the mesh.

(render-texture-stats)=
## Texture diagnostics — `texture_stats`

```python
_render.texture_stats(material, sensor_rgb_weights=(0.52, 0.37, 0.11), theta=0.25)
```

Measures the texture as read by monochrome shading: linearized texels multiplied
by base color and sensor RGB weights. The returned dictionary includes source
(`none`, `srgb8`, or `f16`), image size `[W, H]`, texel count, non-black intensity
percentiles, and black/clipped shares. A byte texture is clipped when any channel
is at least 254.

`step_gt_theta_share` and `step_gt_theta4_share` measure non-black, non-clipped
byte texels whose one-code increase in every channel exceeds the corresponding
log-contrast threshold. They diagnose texture quantization relative to an event
threshold; they are not measured event rates. Clipping/step shares are `None`
without byte textures, and step shares are also `None` when no eligible texels
remain. Positive finite `theta` and non-negative finite weights are required.

(render-reflection-policy)=
## Missing-reflection estimate — `reflection_policy`

```python
_render.reflection_policy(bundle, values, groups=(), nir_fraction=())
```

Returns one report per used specular material, ordered by body/material indices.
`values` uses the resolved renderer mapping. `groups` optionally supplies one
`(minimum, maximum)` waveform multiplier pair per light group; omission assumes
constant illumination. `nir_fraction` contains the simulated camera rows and
extends albedo extrema over their visible/NIR mixtures.

Reports contain material parameters, `refused`, `f_max`, `l_max`, `l_diff_min`,
and `bound = log(1 + f_max * l_max / l_diff_min)`. The latter estimates omitted
log contrast from a conservative radiance envelope and ambient lower bound;
it becomes infinite without a positive lower bound. The estimate excludes
second-order glossy highlights and mover-AO attenuation. It is not an actual
reflection trace.

Roughness below 0.2, or below 0.5 with positive metallic, sets `refused=True`.
The run API enforces this policy unless `scene.allow_missing_reflections` accepts
the approximation. Calling this reporting function neither enables reflections
nor rejects the material on the caller's behalf.

(render-ao-proxy)=
## Ambient-occlusion fit — `ao_proxy`

`_render.ao_proxy(positions, indices)` fits at most eight body-frame spheres to
area-weighted surface samples of a triangle mesh. It returns `centres: float64
(K, 3)`, `radii: float64 (K,)`, `radius_b`, and `d0`. The combined radius is
`sqrt(sum(radii**2))`; `d0` sets the distance window for the renderer's local
proxy-occlusion approximation. Geometry without surface area gives no spheres.

This utility reproduces the deterministic mover-AO fit; it does not return a
per-pixel occlusion image or bake static ambient occlusion. Moving-body AO
attenuates ambient illumination on other receivers, not general indirect light.
Invalid vertex references raise `ValueError`.

(render-ao-directions)=
## Ambient sample directions — `ao_directions`

`_render.ao_directions(pattern_seed, n)` returns `float32 (n, 3)` cosine-distributed
unit directions about local +Z for traced mover AO. The seed is the pixel's
subsample-pattern seed; power-of-two prefixes form nested stratified sets.
Transform them into the surface frame before interpreting them as world rays.
Increasing `n` improves directional sampling; it does not expand the renderer's
local mover-only occlusion model.

## BSDF numerical interfaces

These functions evaluate the same local shading expressions used by rendering.
They are useful for comparison and renderer integration, but do not integrate
scene illumination, trace secondary rays, or modify a `Renderer` instance.

(render-bsdf-eval)=
### Scattering value — `bsdf_eval`

```python
_render.bsdf_eval(n, wv, wl, rho, metallic, alpha2, s, *,
                  dtype="float64", parts=False)
```

Evaluates the local BSDF in sr⁻¹:

```text
f = (1 - metallic) * (1 - E_s(view)) * rho / pi + D * V * F
```

The lobe uses isotropic GGX, height-correlated Smith masking, and the renderer's
Schlick Fresnel approximation. `rho` is monochrome diffuse/metal reflectance;
`s` is dielectric normal-incidence reflectance. `alpha2` means squared GGX width,
which is material roughness to the fourth power, or an already filtered width.
For `n · wl <= 0`, the result is zero. With zero metallic and specular, it reduces
to Lambert `rho / pi` on the lit hemisphere.

| Input / output | Shape and meaning |
| --- | --- |
| `n`, `wv`, `wl` | `float64 (N, 3)` unit vectors; view/light point away from the surface. |
| `rho`, `metallic`, `alpha2`, `s` | `float64 (N,)` or length-one arrays. |
| Return value | `float64 (N,)` BSDF values. |
| `parts=True` | Dictionary of `(N,)` arrays: `f`, `e_s`, `lobe`, `t`. |

`e_s` is the view-dependent specular albedo; `lobe` is `D*V*F`; `t` is the
interpolation coordinate `sqrt(1 - sqrt(floored_view_cosine))`. The layered diffuse
weight depends only on view, so the complete layered BSDF is not reciprocal.
The model is single scattering and has no multiple-scattering compensation.
Supply valid material parameters and normalized directions; the helper is not
a scene-material validator.

`dtype="float32"` rounds inputs and evaluates at executor precision;
`"float64"` evaluates the reference expressions. Both return float64 arrays,
with fp32 results exactly represented after conversion.

(render-bsdf-alpha-eff)=
### Filtered lobe width — `bsdf_alpha_eff`

```python
_render.bsdf_alpha_eff(alpha2, alpha2_add_mean, r_s, *, dtype="float64")
```

Returns effective squared GGX width as a float64 array. Each input has the common
row count or length one. `alpha2_add_mean` is the mean normal-prefilter addition;
`r_s` is the length of the cluster's mean unit normal, with one denoting identical
normals. Before smooth floors/caps, the addition is
`2*(1-r_s**2)/(3*r_s-r_s**3)`.

Normal variation broadens the lobe rather than adding reflected geometry.
The result uses smooth regularization near canceling normals, a width floor
of `1e-8`, and a smooth upper cap below one. Precision selection follows
`bsdf_eval()`; the value is a squared width, not a roughness parameter.

(render-bsdf-tables)=
### Coefficient export — `bsdf_tables`

`_render.bsdf_tables()` returns copies of the float32 `(66, 66)` cubic B-spline
coefficient tables `spec_a`, `spec_b`, `ltc_m00`, `ltc_m02`, `ltc_m20`, and
`ltc_m22`, plus `nodes=64`. Specular albedo is assembled as `F0*A + F90*B`;
LTC entries describe the inverse linear transform used by shading expressions.

Storage is `[roughness_coordinate][view_coordinate]`, with coordinates
`(sqrt(alpha), sqrt(1 - sqrt(view_cosine)))`. Each axis has 64 nodes at `i/63`
and one ghost coefficient at either side. These are spline coefficients, not
a raw sampled image; use the corresponding interpolation convention. Exported
LTC data does not imply that area-light rendering is enabled in this build.

## Static triangle queries

These lower-level functions intersect one static mesh without scene materials,
body trajectories, camera projection, or sensor response. Origins and directions
must be expressed in the same coordinate frame as the vertices. Directions
need not be unit length: the reported `t` is a ray parameter, not necessarily a
metric distance.

(render-embree-intersect)=
### CPU closest hit — `embree_intersect`

`_core.embree_intersect(vertices, indices, origins, directions)` builds an Embree
query for the supplied triangle arrays and returns the closest hit for each ray.
Use it for independent geometric checks in a common static frame. Arrays of
origins and directions must have equal row counts; invalid vertex indices raise
`ValueError` before tracing.

(render-optix-intersect)=
### GPU closest hit — `intersect`

`_optix.intersect(vertices, indices, origins, directions)` provides the same
input/output convention through the optional OptiX module. It requires an
available CUDA/OptiX device and driver. Coincident surfaces and exact shared-edge
hits can select different primitives between acceleration structures; do not
interpret the chosen primitive as a portable tie-breaking rule.

| Array | Dtype and shape | Meaning |
| --- | --- | --- |
| `vertices` | `float32 (V, 3)` | Static mesh coordinates. |
| `indices` | Exact `uint32 (T, 3)` | Triangle vertex references; implicit integer conversion is disabled. |
| `origins`, `directions` | `float32 (N, 3)` | Ray origins and direction vectors. |
| Return `t` | `float32 (N,)` | Hit point is `origin + t * direction`; infinity on misses. |
| Return `prim` | `uint32 (N,)` | Triangle row; `0xffffffff` on misses. |
| Return `u`, `v` | `float32 (N,)` | Weights of triangle vertices 1 and 2; zero on misses. |

The remaining barycentric weight is `1-u-v` for vertex 0. These queries report
geometric hits, not per-event instance/semantic labels.

## Implementation references

- {download}`Renderer interface <../../../../../src/render/renderer.h>`.
  Query semantics, pose-cache behavior, and supported controls.
- {download}`BSDF expressions <../../../../../src/render/shade/bsdf.h>`.
  Scattering conventions and numerical regularization.

```{toctree}
:hidden:
:maxdepth: 1

Reference ray casting <raycast>
```
