# Blender exchange and calibration methods

The [Blender settings](../parameters/blender.md) describe conversion policy and
motion sampling. These methods read exchanges, inspect conversion, and import
camera calibration.

(scene-read-exchange)=
## Blender exchange reader — `read_exchange`

```python
from ballasim.blender.exchange import read_exchange
read_exchange(directory)
```

`read_exchange()` returns `Exchange(root, doc, arrays, digest, version, info)`.
It checks the format and version, required and unknown keys, and array archive.
Unreadable exchanges raise `ExchangeError`; missing files raise
`FileNotFoundError`.

`Exchange` stores the root path, parsed document, contiguous NumPy arrays,
content digest, `(major, minor)` version, and informational reader diagnostics.
The digest uses canonical scene content, excluding producer metadata and archive
container bytes. A different major format is rejected; unknown keys in a newer
minor are reported, while unknown keys in a known minor are errors.

(scene-blender-check)=
### Static conversion report — `check`

```python
from ballasim.blender.convert import check

check(directory, options=None, *, config=None, sets=())
```

`check()` returns a report dictionary containing converted static geometry,
cameras, lights, timeline, refusal status, and whether a motion session is
needed. `config` applies a run configuration and collects unresolved required
keys; `sets` contains configuration overrides and requires `config`.

`options` supplies converter values directly when no run configuration is
given. With `config`, the configuration's Blender settings take precedence and
the report includes missing run requirements. This check converts static facts
and reports whether motion sampling is needed; it does not produce a completed
animated scene or run the event simulator.

(scene-calibration-preset)=
### Camera preset — `calibration.preset`

`ballasim.blender.calibration.preset(name)` returns a calibration dictionary
for `evimo2_left`, `evimo2_right`, or `evimo2_samsung`. It includes resolution,
intrinsics, distortion, and source sensor/pixel/lens metadata. An unknown name
raises `CalibrationError`; the record does not itself construct a scene camera.

(scene-calibration-kalibr)=
### Kalibr import — `calibration.kalibr`

`ballasim.blender.calibration.kalibr(text, camera=None)` parses one camera from
a YAML camchain and returns a calibration dictionary. `camera` accepts a name
such as `"cam1"` or index `1`; omission is allowed only for a single-camera
document. Missing or malformed fields raise `CalibrationError`. Supported
camera-model validation remains part of Blender conversion.

(scene-calibration-opencv)=
### OpenCV import — `calibration.opencv`

`ballasim.blender.calibration.opencv(text, camera=None)` parses OpenCV
FileStorage or ROS camera-info YAML. `camera` optionally selects a nested
mapping. It returns resolution, intrinsics, and distortion in a calibration
dictionary. Skewed camera matrices, malformed required fields, and unrecognized
calibration layouts raise `CalibrationError`. The converter separately checks
whether the parsed distortion model is supported.

`ballasim.blender.calibration.preset(name)`, `kalibr(text, camera=None)`, and
`opencv(text, camera=None)` return calibration dictionaries. Invalid calibration
data raises `CalibrationError`. Kalibr selects a camera by name or integer;
OpenCV selects an optional nested mapping.


## References

- [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).
  Imported geometry and material conventions.
- Burner et al., [EVIMO2](https://arxiv.org/pdf/2205.03467), 2022.
  Released measured-pose and scanned-object data.
- {download}`Scene-bundle implementation <../../../../../python/ballasim/scene/bundle.py>`.
  Record validation and persistence rules.
