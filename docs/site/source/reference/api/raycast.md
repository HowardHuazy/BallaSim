# Reference geometric queries

These CPU helpers support geometric inspection without running the asynchronous
sensor. They use center or explicit offset rays, independently of finite-pixel
radiance integration.

## Reference ray casting

Functions below are in `ballasim.scene.raycast` and use the CPU backend.

(scene-render-dense)=
### Depth and identity — `render_dense`

`render_dense(bundle, cam_index, t_ns, *, offsets=((0.0, 0.0),), bodies=None)`
returns `(depth, instance)` images of shape `(H, W)`. Depth is `float32` camera
Z-depth in meters, not Euclidean ray distance; instance IDs are `uint32` and
misses are zero. `cam_index` and optional `bodies` use bundle list indices.
Several offsets cast multiple rays and retain the closest positive depth per
pixel. This geometric query does not simulate radiance or event thresholds.

(scene-pixel-rays)=
### Image rays — `pixel_rays`

`pixel_rays(cam, offset=(0.0, 0.0))` unprojects image positions
`(x + offset[0], y + offset[1])` in row-major order. It returns camera-frame
`float64 (H*W, 3)` rays and a boolean validity mask. Rays are scaled to Z=1
for depth intersection; invalid projection locations are indicated by the mask.
Use a zero offset for the integer-center camera convention.

(scene-camera-model)=
### Projection adapter — `camera_model`

`camera_model(cam, support_px=1.0)` constructs the shared-core radtan or
Kannala–Brandt camera from a scene `Camera`. `support_px` extends the checked
image domain around pixel centers, so use support large enough for the intended
subpixel offsets. A non-injective or unsupported projection domain raises an
error instead of producing arbitrary inverse rays.

(scene-camera-pose)=
### Camera placement — `camera_pose`

`camera_pose(bundle, cam_index, t_ns)` returns a `float64 (4, 4)` transform
from camera to origin-relative world coordinates. It evaluates the rig and
composes `T_world_rig @ T_rig_cam`. The scalar time uses the scene's true clock;
sensor timestamp offsets do not change the geometric pose query.

(scene-body-poses)=
### Body placement — `body_poses`

`body_poses(bundle, t_ns)` returns one body-to-world `(4, 4)` transform per
bundle body, in list order. Static bodies use their fixed transforms; moving
bodies evaluate their referenced trajectories. Positions retain the bundle
origin convention. A query outside a required trajectory's support raises.

(scene-cell-grid)=
### Subpixel grid — `cell_grid`

`cell_grid(n)` returns `n*n` `(x, y)` offsets at centers of a regular grid over
`[0, 1) × [0, 1)`. Passing them to `render_dense()` gives a nearest-depth
coverage approximation over that cell. It does not integrate the optical pixel
response used by the event renderer.

(scene-merge-in-camera)=
### Posed triangle arrays — `merge_in_camera`

`merge_in_camera(meshes, indices, t_cam_body, ids)` accepts parallel lists of
body-frame `(V, 3)` vertices, `(T, 3)` triangle indices, camera-from-body `(4, 4)`
transforms, and instance IDs. It returns merged `float32` camera-relative
vertices, `uint32` indices, and `uint32` per-triangle IDs. Transform arithmetic
uses float64 before the final vertex conversion.

(scene-cast)=
### Closest-hit query — `cast`

`cast(vertices, indices, prim_id, rays, valid)` casts `(N, 3)` ray directions
from the camera origin against camera-relative triangles. It returns
`(depth, instance)` vectors of length N, with zero for misses or invalid rays.
Depth is the intersection parameter, so Z-depth interpretation requires the
Z=1 directions returned by `pixel_rays()`.

(scene-combine-min-depth)=
### Subpixel depth reduction — `combine_min_depth`

`combine_min_depth(samples)` accepts a non-empty list of `(depth, instance)`
array pairs of matching shape. It returns the smallest positive depth and its
corresponding ID at each array location, with zero when all samples miss.
This is a closest-surface reduction, not an average or antialiased depth.

| Function | Returns |
| --- | --- |
| `render_dense(bundle, cam_index, t_ns, *, offsets=((0.0, 0.0),), bodies=None)` | `(depth, instance)` images `(H, W)`: `float32` camera Z-depth in m and `uint32` IDs; misses are zero. `bodies` selects body indices. Multiple offsets take minimum depth. |
| `pixel_rays(cam, offset=(0.0, 0.0))` | Row-major rays `(H*W, 3)` and validity mask. |
| `camera_model(cam, support_px=1.0)` | Shared-core radtan or Kannala–Brandt model. |
| `camera_pose(bundle, cam_index, t_ns)` | `float64 (4, 4)` camera-to-world transform. |
| `body_poses(bundle, t_ns)` | List of body-to-world transforms. |
| `cell_grid(n)` | `n*n` subpixel offsets over a pixel cell. |
