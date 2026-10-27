# Camera models and pixel sampling

`ballasim._core` provides batched camera geometry and deterministic spatial
sampling. Camera methods describe projection and ray generation; sampling
methods describe the finite-pixel integration pattern. Scene-level rendering
uses the [renderer API](simulation.md).

## Build and array conventions

`_core.build_info()` returns the version, compiler and booleans indicating which
modules were built, without importing them. `_core.WINDOW_NS` is `8_388_608`.

Floating-point inputs are CPU arrays and may be converted to the required dtype
and contiguous layout. RNG input arrays require exact `uint32` and may be
strided; incompatible dtypes raise `TypeError` rather than being cast.

Results are NumPy arrays. Image coordinates use `(x, y)` with integer-valued
pixel centers, x right and y down. Camera points use the camera frame, with z
forward. [Rotation and spline primitives](geometry.md) document motion arrays.

## Camera models

(radtan-camera)=
### `_core.RadtanCamera`

```python
_core.RadtanCamera(fx, fy, cx, cy, k1=0, k2=0, p1=0, p2=0, k3=0,
                   *, width, height, z_min=0, support_px=0.95)
```

Constructs a pinhole camera with OpenCV radial–tangential distortion. The
coefficients are ordered `(k1, k2, p1, p2, k3)`; zero coefficients give the
undistorted pinhole. `fx`, `fy`, `cx` and `cy` are in pixels, and `z_min` excludes
camera-frame points with z at or below that value.

Construction checks a monotone distortion branch covering the image and
`support_px` beyond its pixel centers. This prevents a folded distortion
polynomial from projecting distant off-axis geometry back into the image.
`r_mono`, `r_img` and `r_lim` report the monotonicity radius, required image-support
radius and chosen projection limit in normalized coordinates.

(kannala-brandt-camera)=
### `_core.KannalaBrandtCamera`

```python
_core.KannalaBrandtCamera(fx, fy, cx, cy, k1=0, k2=0, k3=0, k4=0,
                         *, width, height, support_px=0.95)
```

Constructs the equidistant polynomial fisheye model
`theta_d=theta*(1+k1*theta²+k2*theta⁴+k3*theta⁶+k4*theta⁸)`. Unlike pinhole
projection, it can represent directions beyond 90° when the calibrated domain
permits them. `theta_mono`, `theta_img` and `theta_lim` are the corresponding
monotonicity, support and projection-limit angles in radians.

Both constructors require finite positive focal lengths and positive image
dimensions. Their subsample support must fit strictly inside a valid injective
domain. `support_px` describes ray support, rather than optical blur by itself.

### `project(X)`

Maps camera-frame points to distorted image coordinates. It returns
`(pixels, valid)` and uses the calibrated model's domain, including the radtan
depth limit. A valid point may lie outside the image rectangle; apply an image
bound test separately when selecting visible pixels. Invalid rows contain `NaN`.

### `unproject(uv)`

Inverts distorted image coordinates into camera-frame rays on the selected
monotone branch, returning `(rays, valid)`. This finds a ray, not a 3D point:
depth must be supplied separately. Radtan rays are `(x_n, y_n, 1)`, so their ray
parameter equals camera z; Kannala–Brandt rays are unit vectors. Invalid rays
contain `NaN`.

### `project_jacobian(X)`

Returns `(J, valid)`, where `J=d(pixel)/d(X_camera)` is the projection derivative.
It converts small camera-frame displacements into image displacements and is
used in image-speed and solid-angle calculations. It differentiates the full
distorted projection, including division by depth for radtan; it is not a
derivative with respect to calibration parameters.

### `RadtanCamera.distort_jacobian(xn)`

Returns `d(x_distorted)/d(x_normalized)` at normalized coordinates `(X/Z,Y/Z)`.
The result is `(N,2,2)`, without focal-length scaling or perspective division.
Use `project_jacobian` when the input is a 3D displacement; this method isolates
the distortion polynomial's local area and direction change.

### `ray_lut(max_offset_px, samples_per_px=2)`

Builds a `RayLut` of exact unprojections at grid nodes, covering the image and
requested pixel-center offsets. Nodes outside the camera's injective domain
raise `ValueError`. A higher `samples_per_px` improves interpolation on curved
ray fields and increases table memory; it does not change calibration.

### `RayLut.lookup(uv)`

Returns rays by bilinear interpolation of the grid. This avoids an iterative
unprojection for every subsample at the cost of interpolation error. Lookup
returns no validity mask and does not renormalize interpolated fisheye rays.
Positions outside the grid extrapolate from the border cell; stay within the
built support for normal ray generation. `origin` gives the first node's pixel
position, and `shape` gives `(rows, columns)`.

### Camera array reference

Shared operations and model-specific methods are listed below. Inputs and
numerical outputs use `float64`; validity masks use `bool`.

| Method | Input | Returns |
| --- | --- | --- |
| `project(X)` | Camera-frame points `(N, 3)` | `(pixels (N, 2), valid (N,))` |
| `project_jacobian(X)` | Camera-frame points `(N, 3)` | `(d_pixel_d_X (N, 2, 3), valid (N,))` |
| `unproject(uv)` | Distorted pixel positions `(N, 2)` | `(rays (N, 3), valid (N,))` |
| `RadtanCamera.distort_jacobian(xn)` | Normalized coordinates `(N, 2)` | Distortion Jacobians `(N, 2, 2)` |
| `ray_lut(max_offset_px, samples_per_px=2)` | Support radius and grid resolution | `RayLut` |
| `RayLut.lookup(uv)` | Distorted pixel positions `(N, 2)` | Rays `(N, 3)` |

Both camera classes also expose the read-only `support_px` property.

## Random and pixel sampling

### `_core.philox4x32_10(counter, key)`

Evaluates ten rounds of the Philox counter-based generator. Each counter/key
row deterministically produces four `uint32` words. There is no mutable stream
position: the same row gives the same result regardless of evaluation order or
batching. Callers construct counters and keys; this primitive does not return
Gaussian draws or floating-point uniforms.

### `_core.sobol(index, dim)`

Returns one dimension of the unscrambled Sobol sequence at each supplied index,
using Joe–Kuo direction numbers. Dimensions are `0`–`7`; outputs are unsigned
32-bit fixed-point fractions. Converting a word to `float64` and dividing by
`2**32` gives its coordinate in `[0,1)`. Sequence index denotes a sample, rather
than mutable generator state.

### `_core.owen_sobol(index, dim, seed)`

Returns the shuffled, hash-based Owen-scrambled Sobol coordinates used for
spatial sampling. The same seed combines a shared index shuffle with a
dimension-specific scramble. Dimensions 0 and 1 preserve nested two-dimensional
power-of-two prefixes, allowing higher pixel sample counts to extend an existing
pattern instead of redrawing it. Scrambling changes the numerical integration
pattern, not the physical pixel response.

### `_core.subsample_pattern_seed(run_seed, camera, pixel)`

Derives a pattern seed from the run seed, camera index and full-frame pixel ID.
The renderer reuses this pattern over time, so static geometry is not affected
by unrelated spatial redraws at each temporal query. `camera` must be below
256; the returned seed is 32-bit and is not a unique identifier for every pixel.

### `_core.WarpTable`

```python
_core.WarpTable(w=1.0, sigma=0.15, n_sigma=3.0, cells=256)
```

Tabulates the inverse CDF of the box-aperture response convolved with a Gaussian,
truncated and renormalized on `[-x_max,x_max]`, where
`x_max=w/2+n_sigma*sigma`. Width `w`, Gaussian standard deviation `sigma` and
`x_max` use pixels. At least one of `w` and `sigma` must be positive;
`n_sigma>0`, and `cells` must lie in `[2,65535]`.

This warp converts uniform samples into equal-weight pixel-response samples.
Changing `w` or `sigma` changes the represented image formation; changing
`cells` changes the table approximation. Construction retains fp64 and fp32
tables for the corresponding evaluation paths.

### `WarpTable.inverse_cdf(xi)` and `inverse_cdf_f32(xi)`

Map one-dimensional probabilities in `[0,1)` to offsets from the pixel center.
The returned offsets follow the tabulated spatial response; they are not
uniform offsets except for a pure box. `inverse_cdf` uses fp64 and its variant
uses fp32. Out-of-range probabilities raise `ValueError`.

### `WarpTable.subsample_offsets(pattern_seed, count)`

Returns the first `count` two-dimensional scrambled Sobol samples after the
separable warp, as `(count,2)` offsets in pixels. Add them to an integer pixel
center before ray generation. The seed controls the pattern; the table controls
its spatial density. `subsample_offsets_f32` uses the fp32 table and arithmetic.

### Sampling array reference

| Callable | Inputs | Returns |
| --- | --- | --- |
| `philox4x32_10(counter, key)` | `uint32 (N, 4)` counters, `uint32 (N, 2)` keys | `uint32 (N, 4)` |
| `sobol(index, dim)` | `uint32 (N,)` indices, dimension `0`–`7` | `uint32 (N,)` fixed-point samples |
| `owen_sobol(index, dim, seed)` | Sobol indices, dimension and scramble seed | `uint32 (N,)` scrambled samples |
| `subsample_pattern_seed(run_seed, camera, pixel)` | Integer identifiers; `camera < 256` | Pixel-pattern scramble seed |

| Method | Result |
| --- | --- |
| `inverse_cdf(xi)` | `float64 (N,)` offsets for probabilities `xi (N,)` in `[0, 1)` |
| `inverse_cdf_f32(xi)` | The same operation in `float32` |
| `subsample_offsets(pattern_seed, count)` | `float64 (count, 2)` offsets from the pixel center |
| `subsample_offsets_f32(pattern_seed, count)` | The same offsets in `float32` |

## References and implementation

The fisheye model follows Kannala and Brandt,
[A Generic Camera Model and Calibration Method for Conventional, Wide-Angle,
and Fish-Eye Lenses](https://doi.org/10.1109/TPAMI.2006.153).
Counter-based sampling follows Salmon et al.,
[Random123](https://www.deshawresearch.com/resources_random123.html).
The spatial scramble follows Burley,
[Practical Hash-based Owen Scrambling](https://jcgt.org/published/0009/04/01/paper.pdf).
Camera-domain checks and finite-pixel warping are BallaSim's implementation
choices; these references do not prescribe their exact tolerances or defaults.

The callable contracts are defined in `src/bindings/core_camera.cpp` and
`src/bindings/core_rng.cpp`; projection, monotonicity checks and LUT interpolation
are in `src/core/camera/`.
