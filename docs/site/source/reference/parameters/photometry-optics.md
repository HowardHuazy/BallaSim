# Photometry and optics

Camera scope: `cameras.<camera>.photometry.*` and `cameras.<camera>.optics.*`.
See [Configuration](../../guide/configuration.md) for photometry selection.

## Photometry

(photometry-mode)=
### `photometry.mode`

Select how renderer values become signal photocurrent. `relative` uses an
explicit scale; `physical` derives it from radiometric calibration; `reference`
solves it from an image statistic at the start of the run. This choice determines
what a change in light intensity means physically. It does not select the pixel's
contrast thresholds or temporal response.

In physical mode the on-axis scale is proportional to
$\pi T\,\mathrm{QE}\,\kappa d^2 q/(4N^2)$, where $T$ is transmittance,
$d$ is pixel pitch, $N$ is the working f-number, and $q$ is electron charge.
Unit conversion produces fA per renderer unit. The formula assumes calibrated
luminance and the stated illuminant response; selecting this mode cannot
calibrate arbitrary light values. EVIMO2 placeholder lighting is not radiometric
calibration, and the resolver warns about that combination.

(photometry-illuminance-scale)=
### `photometry.illuminance_scale`

Multiply renderer irradiance by this factor to obtain signal photocurrent in fA,
before adding dark current. Relative photometry has no neutral scale, so a
positive value is required. Raising the scale increases signal current and
changes the cascade's bandwidth and signal-to-dark-current ratio. It need not
change ideal log-contrast events when all light levels scale together and dark
current is negligible.

Physical mode derives this value; reference mode solves it at initialization.
An explicit value is inactive in those modes. For reference runs, inspect the
solved camera value in `run.json`, rather than the resolver's initial placeholder.

(photometry-reference-current-fa)=
### `photometry.reference_current_fA`

Set the target median **signal** current of the first frame's reference pixels.
The engine renders the starting scene, applies mean light waveforms and spatial
gain, and solves a common current scale. Dark current is excluded from the
target. This makes the assumed illumination explicit when absolute light units
are unavailable.

A nonzero render statistic is required. Reference normalization absorbs a common
change in scene brightness; it should not be used to measure the effect of such
a change as if illumination were held calibrated.

(photometry-reference-instance)=
### `photometry.reference_instance`

Restrict the reference statistic to center rays hitting the named body. An empty
name uses every geometry hit. A reference body can isolate a known surface from
a bright emitter or an unrepresentative background. Its visible pixels must
provide a usable nonzero signal at initialization; changing the selected surface
can change the solved scale for the entire camera.

## Lens transmission and illumination

(optics-f-number)=
### `optics.f_number`

The working f-number controls image irradiance through the $1/N^2$ pupil-area
factor in physical photometry. It defaults to the scene camera's value. Doubling
it reduces the on-axis signal current by a factor of four when other calibration
values are fixed. It also increases diffraction blur when that effect is enabled.
This parameter does not introduce depth of field or refocus the camera.

(optics-transmittance)=
### `optics.transmittance`

The fraction of light transmitted through the lens and any optical filter, used
in physical photometry. It scales signal current uniformly, independently of
vignetting. This distinguishes light lost in the optical path from photons not
converted by `sensor.qe`. Relative mode already specifies its net scale and does
not apply this physical conversion separately.

(optics-vignetting)=
### `optics.vignetting`

Enable a static spatial photocurrent gain map. The `null` default enables it in
physical and reference photometry and disables it in relative mode. A dimmer
corner changes local sensor bandwidth, dark-current contrast compression, and
noise conditions even when the scene itself is uniform.

The gain is applied in photocurrent conversion; it is not baked into the
renderer’s geometric ground truth. Reference normalization includes the map
when solving its median current.

(optics-vignetting-law)=
### `optics.vignetting_law`

Choose the geometric falloff of the gain map. `cos4` uses $\cos^4\theta$ for the
chief-ray angle. `jacobian` uses the camera model's solid-angle density:

```{math}
g(\mathbf w)=\frac{\cos\theta\,(d\Omega/dA)(\mathbf w)}{(d\Omega/dA)(\mathbf z)}.
```

Both are unity on the optical axis. The Jacobian form reduces to the cosine-fourth
law for an undistorted pinhole, but accounts for the area magnification of
calibrated distortion. It assumes a small fixed pupil, rather than a full lens
system. Fidelity and balanced derive this law when the gain map is active;
frozen EVIMO2 profiles and the other accuracy profiles retain `cos4` unless
explicitly overridden.

(optics-relative-illumination)=
### `optics.relative_illumination`

Multiply the active lens gain by $1+c_2r^2+c_4r^4$, with image radius normalized
to the farthest corner. Supply `[c2, c4]` from a relative-illumination fit, or `[]`
for no additional correction. This compact model captures smooth residual lens
falloff beyond the projection-derived law. The polynomial must remain positive
over the image and has no effect while vignetting is disabled.

(optics-veiling-glare)=
### `optics.veiling_glare`

Add $g$ times each light group's full-frame mean irradiance uniformly across the
image. This approximates a diffuse stray-light veil: bright regions raise the
signal floor in dark regions and reduce contrast. The parameter is the additive
coefficient $g$, related to the veiling-glare index by $g=\mathrm{VGI}/(1-\mathrm{VGI})$;
it is not VGI itself.

This model does not generate spatial flare ghosts or directional streaks.
A cropped sensor still needs full-frame means to reproduce the uncropped veil;
the low-level sensor interface accepts those means explicitly.

## Blur and diffraction

(optics-blur-sigma-px)=
### `optics.blur_sigma_px`

Set the standard deviation of the Gaussian optical blur in pixels. Convolution
with the box aperture forms the spatial pixel response sampled by the renderer.
A larger value broadens moving edges and can reduce their peak log-contrast
change. It changes scene-to-pixel image formation, not the sensor's temporal
filter or timestamp precision.

The resolver records the effective value after any micrometer conversion and
quadrature addition of diffraction. Selected cameras must agree on the resulting
pixel response because the current renderer shares that response across the run.

(optics-blur-sigma-um)=
### `optics.blur_sigma_um`

Express the optical Gaussian standard deviation on the sensor plane in
micrometers. When set, division by the scene camera's pixel pitch replaces the
pixel-valued blur before diffraction is added. Use this when the lens blur is
known in physical sensor units or when comparing sensors with different pitches.
`null` leaves `blur_sigma_px` as the optical core.

(optics-diffraction)=
### `optics.diffraction`

Add the Gaussian approximation to the Airy core in quadrature:

```{math}
\sigma_{\mathrm{effective}}^2=\sigma_{\mathrm{optical}}^2+(0.42\lambda N)^2,
```

with all lengths converted to the same sensor-plane units. The approximation
uses the central lobe's width; it does not reproduce Airy rings or a wavelength
spectrum. Enabling diffraction therefore changes blur in relative photometry as
well as in physical mode, even though only physical mode derives brightness from
f-number.

(optics-wavelength-nm)=
### `optics.wavelength_nm`

Set the effective wavelength used in the diffraction-width approximation.
Longer wavelengths broaden its Gaussian contribution linearly. Choose a value
representative of the band being modeled; this scalar does not turn the RGB/NIR
material mixture into a spectral renderer. It is inactive with diffraction off.

## Sources and model scope

The irradiance and fixed-pupil treatment follows the camera-radiometry model in
[pbrt, Camera Models](https://pbr-book.org/4ed/Cameras_and_Film).
The Airy-core approximation follows Zhang, Zerubia, and Olivo-Marin,
[Gaussian approximations of diffraction-limited point spread functions](https://doi.org/10.1364/AO.46.001819).
The polynomial relative-illumination correction is an empirical calibration
model; its coefficients are not universal lens constants.

BallaSim's mode derivation is implemented in
{download}`resolve.cpp <../../../../../src/config/resolve.cpp>`; geometric gain
is defined in {download}`solid_angle.h <../../../../../src/core/camera/solid_angle.h>`.

## Quick reference

### Photometry

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `photometry.illuminance_scale` | `float · fA` | required | Signal photocurrent per unit rendered irradiance. Required in relative mode. Range: >0. |
| `photometry.mode` | `enum` | `"relative"` | `relative` applies a stated scale; `physical` uses calibrated light and lens units; `reference` solves a scale from image pixels. Choices: `relative`, `physical`, `reference`. |
| `photometry.reference_current_fA` | `float? · fA` | `null` | Target median signal photocurrent in the initial reference pixels. Required in reference mode. Range: >0. |
| `photometry.reference_instance` | `string` | `""` | Body name selecting reference pixels; an empty string uses all geometry hits. Reference mode only. |

### Optics

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `optics.veiling_glare` | `float` | `0.0` | Uniform irradiance added from each light group's frame mean; coefficient `g = VGI / (1 - VGI)`. Range: [0, 1). |
| `optics.vignetting` | `bool?` | `null` | Enable lens gain; `null` disables it in relative mode and enables it in physical/reference modes. |
| `optics.vignetting_law` | `enum` | derived | `cos4` uses the chief-ray cosine; `jacobian` uses the camera model's solid-angle Jacobian. Applies with vignetting enabled. Choices: `cos4`, `jacobian`. |
| `optics.relative_illumination` | `float[]` | `[]` | Additional lens gain coefficients `[c2, c4]` in normalized image radius; `[]` disables the correction. Vignetting only. |
| `optics.blur_sigma_px` | `float · px` | `0.15` | Gaussian optical blur in pixels; selected cameras must agree after unit conversion. Range: ≥0. |
| `optics.f_number` | `float` | derived | Working lens f-number, from the bundle camera. Used by physical photometry and diffraction. Range: >0. |
| `optics.transmittance` | `float` | `0.9` | Lens/filter transmittance. Physical photometry only. Range: (0, 1]. |
| `optics.blur_sigma_um` | `float? · um` | `null` | Gaussian blur on the sensor; overrides `blur_sigma_px` after conversion by pixel pitch. `null` uses the pixel value. Range: ≥0. |
| `optics.diffraction` | `bool` | `false` | Add Gaussian-equivalent diffraction blur in quadrature; selected cameras must agree on the resulting pixel response. |
| `optics.wavelength_nm` | `float · nm` | `550.0` | Effective wavelength for diffraction blur; inactive when diffraction is disabled. Range: >0. |

### Resolution rules

`relative` requires `illuminance_scale`. `reference` requires
`reference_current_fA` and solves the scale at run initialization. `physical`
uses the bundle pixel pitch, lens parameters, `sensor.qe`, and `sensor.kappa`;
physical Blender sources also require `scene.blender.lm_per_w`.

`optics.vignetting_law` derives `jacobian` under fidelity/balanced when vignetting
is enabled, except for frozen EVIMO2 v0/v1 sensor profiles; other cases use
`cos4`. An explicit value overrides the derivation.

`relative_illumination` must be empty or contain `[c2, c4]`; its gain polynomial
must remain positive over the image. Physical photometry expects RGB weights
summing to one and warns otherwise. Pixel aperture, effective blur, and RGB
weights are shared by selected cameras.
