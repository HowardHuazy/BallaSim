# Python conventions

Python methods are documented within the manual topic that uses them:
[scene objects](../guide/scenes.md), [motion](../guide/trajectories.md),
[cameras](../guide/cameras.md), [sensors](../guide/sensors.md),
[running simulations](../guide/simulations.md), and [outputs](../guide/outputs.md).
This page defines their shared array, time, and error conventions.

## Array conventions

NumPy arrays are used throughout the Python interfaces. Array shapes are written
as `(N, 3)` for `N` three-component vectors. Times passed to native numerical
interfaces use `numpy.int64` nanoseconds. Quaternion components use `(w, x, y, z)`.

Native functions return NumPy arrays owned by Python. Supply the documented
dtype for integer timestamps, mesh indices, and random-number counters; these
inputs reject implicit casts. Floating-point batch interfaces may convert array
layouts and dtypes as required.

## Complete run

```python
import ballasim
from ballasim import config

spec = config.load("run.yaml", sets={"run.device": "cpu"})
summary = ballasim.run(spec)
```

Optional modules must be available in the installed build; see
[Installation](../getting-started/installation.md).

## Time and frames

True time belongs to the scene. Reported event time additionally includes the
camera clock and readout model; the two are not interchangeable. Keep timestamps
as integers when converting file columns or evaluating trajectories. Distances
use meters, angular rates use radians per second, and poses map local coordinates
into their named parent frame.

## Errors and optional modules

`ValueError` reports invalid values or unsupported combinations. Configuration,
scene, frame, and export interfaces add domain-specific subclasses and diagnostic
records. Inspect those records before changing parameters: a missing calibration
value and an unsupported rendering feature require different remedies.

`ballasim._core.build_info()` reports which native modules were built. Their
presence does not imply a usable GPU; device availability is checked separately.
