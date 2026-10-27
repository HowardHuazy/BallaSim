# Cameras & optics

A camera combines a projection model, a spatial pixel response, and a conversion
from scene radiance to photocurrent. These describe different parts of image
formation: projection places a surface on the image; the pixel response averages
nearby rays; photometry determines the current supplied to the
[event sensor](sensors.md).

## Projection and calibration

BallaSim supports OpenCV radial–tangential (`radtan`) and Kannala–Brandt (`kb`)
cameras. Intrinsics and distortion coefficients belong to the scene camera.
`T_rig_cam` places the camera in its moving rig; changing sensor thresholds does
not change this calibration.

Image coordinates use distorted pixels with integer-valued pixel centers.
Projection returns both coordinates and validity. A point may project outside
the image rectangle while remaining valid for the mathematical camera model.
The constructor checks an injective domain so that unprojection has a unique
solution over the image and its subsample support.

Use the [camera methods](../reference/api/core.md) to project points, inspect
Jacobians, unproject pixels, or build a ray lookup table. See
[Scenes & assets](scenes.md) for authoring camera records and importing calibration.

## Spatial response

A pixel integrates light over a finite area. The simulator models that response
as a box aperture convolved with a Gaussian optical blur. Rays drawn from this
response smooth geometric, texture, and shadow boundaries before the sensor
compares log intensity. This is spatial integration; it does not introduce a
frame rate or an exposure into the asynchronous event stream.

`sensor.aperture_px` sets the box width. `optics.blur_sigma_px` sets the Gaussian
standard deviation, or it can be derived from sensor-plane micrometers and
diffraction. `accuracy.n_subsamples` controls how accurately the renderer
integrates this response. Increasing that count changes the numerical estimate;
increasing blur changes the modeled camera.

## From radiance to current

Choose a photometry mode according to the information available:

- **Relative:** provide the current scale for a scene with arbitrary light units.
- **Physical:** provide calibrated light units, pixel pitch, lens parameters, and
  sensor responsivity. The resolver computes the current scale.
- **Reference:** specify a target median signal current on selected image pixels.
  The engine solves the scene scale at initialization.

The [photometry and optics settings](../reference/parameters/photometry-optics.md)
explain each control and its units. The spectral response, quantum efficiency,
and aperture entries are documented with the
[sensor settings](../reference/parameters/sensor.md).

## Lens gain

Vignetting changes the mean illumination across the sensor. The `cos4` law models
natural falloff for a pinhole projection; the `jacobian` law also accounts for the
calibrated model's angular-to-image-area mapping. A radial polynomial can supply
additional measured relative illumination. These gain maps do not simulate a
complete lens prescription, pupil aberrations, or mechanical vignetting.

```{toctree}
:hidden:
:maxdepth: 1

Camera models and pixel sampling <../reference/api/core>
Photometry and lens settings <../reference/parameters/photometry-optics>
```
