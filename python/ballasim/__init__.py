"""BallaSim: an asynchronous, physically grounded event-camera simulator.

Extension modules: ``ballasim._core`` (CPU core, always built), ``ballasim._cuda`` (CUDA kernels),
``ballasim._optix`` (OptiX backend, built only when ``OPTIX_ROOT`` is set), ``ballasim._render``
(the renderer, docs/dev/RENDER.md), ``ballasim._sensor`` (the pixel model, docs/dev/SENSOR.md) and
``ballasim._engine`` (the C++/CUDA engine, docs/dev/ENGINE_CPP.md; its device half with CUDA and OptiX),
``ballasim._scene`` (the scene side: the EVIMO2 trajectory fitter, docs/dev/CPP_PORT.md item 5; links OpenBLAS),
``ballasim._imu`` (the IMU, docs/dev/IMU.md; built with every Python build) and ``ballasim._export`` (the format
exporters, docs/dev/EXPORT.md; links liblz4, libzstd and libbz2) and ``ballasim._frames`` (frames on the event clock,
docs/design/phase3/frames.md: waveform means, schedules, the exposure reference).
The optional modules are imported explicitly by their users; ``_core.build_info()`` says which were
built without importing them.

Development hook: ``BALLASIM_BUILD_DIR`` (os.pathsep-separated directories) is appended to the
package search path, so extension modules built by a plain CMake build (e.g.
``build/linux-cuda-optix/src/render``) import as ``ballasim._render`` without reinstalling.
Installed modules take precedence; the hook only adds modules that are not installed.

Running: ``ballasim.run(spec | path, bundle=None, report=None, on_window=None, cancel=None)`` (:mod:`ballasim.engine`)
on a run configuration loaded by ``ballasim.config.load`` (format 2, resolved in C++; docs/dev/P0_PARAMS.md); progress
and cancel (``on_window``, ``cancel``: BL-0b, contract C9; docs/dev/ENGINE.md).
"""

import os as _os

__version__ = "0.1.0.dev0"

for _dir in filter(None, _os.environ.get("BALLASIM_BUILD_DIR", "").split(_os.pathsep)):
    if _os.path.isdir(_dir) and _dir not in __path__:
        __path__.append(_dir)
del _os


def run(spec, *, bundle=None, report=None, on_window=None, cancel=None):
    """Runs a run configuration (``ballasim.config.load``'s RunSpec, or a path); see ballasim.engine.run, which takes
    the same keywords (``on_window`` and ``cancel``: BL-0b)."""
    from ballasim.engine import run as _run

    return _run(spec, bundle=bundle, report=report, on_window=on_window, cancel=cancel)
