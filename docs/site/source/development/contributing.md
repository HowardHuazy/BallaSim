# Contributing

Use the source environment described in [Installation](../getting-started/installation.md).
Keep changes focused on one interface or behavior and update its tests and
documentation together.

## Source layout

| Path | Contents |
| --- | --- |
| `src/core/` | Shared math, camera models, random sampling, time and trajectory evaluation |
| `src/scene/`, `src/config/` | Scene conversion, fitting, parameter schema and resolution |
| `src/render/`, `src/sensor/`, `src/engine/` | Rendering, pixel model and asynchronous engine |
| `src/frames/`, `src/imu/`, `src/export/` | Frame, IMU and external format support |
| `python/ballasim/` | Python API, I/O, CLI, browser interface and visualization |
| `tests/cpp/`, `tests/gpu/`, `tests/render/`, `tests/sensor/` | Native core, device, rendering, and sensor tests |
| `schema/` | Versioned exchange and job contracts |
| `docs/site/source/` | Published documentation sources |

See [Architecture](architecture.md) for subsystem responsibilities.

## Native development

Activate the conda environment before using a preset. For a CPU build:

```bash
cmake --preset linux-cpu
cmake --build --preset linux-cpu
ctest --preset linux-cpu --output-on-failure
```

Available presets are `linux-cpu`, `linux-cpu-debug`, `linux-cuda`,
`linux-cuda-optix` and `linux-cuda-optix-debug`. OptiX presets require
`OPTIX_ROOT`. Presets build native modules and tests with warnings treated as
errors.

Python normally imports the installed modules. After native changes, rerun the
editable install command before using the Python interface.

## Tests

The public snapshot includes native core, device, rendering, and sensor tests.
Run the affected CTest cases after rebuilding:

```bash
ctest --preset linux-cpu --output-on-failure
ctest --preset linux-cpu -LE slow
```

Choose the matching CUDA/OptiX preset for device changes. Tests requiring an
unavailable device or dataset may skip. The Python regression suite and its
preset-runner helper are maintained in the development checkout and are not
included in this snapshot.

## Implementation conventions

Use Python type annotations and `snake_case` names; the configured Python line
length is 120 characters. Follow adjacent C++ code and its C++20 conventions.
Preserve integer-nanosecond timestamps, explicit units and array dtype and shape
checks at interface boundaries.

Shared CPU/CUDA math follows the project's floating-point policy. Do not enable
fast math or implicit contraction; CMake checks the configured flags. Keep random
draws keyed by their defined purpose so execution order does not change results.

Define parameters in the native descriptor tables and use the common resolver.
When changing a serialized interface, update its format or feature contract and
the corresponding reader tests. Keep optional dependencies imported by the
features that use them.

## Documentation

Pages use MyST Markdown, Sphinx and Furo. Build the site in a separate Python
environment:

```bash
python -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/site/requirements.txt
.venv-docs/bin/python -m sphinx -W --keep-going -b html \
  -d build/docs/doctrees docs/site/source build/docs/html
.venv-docs/bin/python -m http.server 8000 --bind 127.0.0.1 --directory build/docs/html
```

Open `http://127.0.0.1:8000/`. The documentation build reads the package version as
text and does not import BallaSim or execute examples. Add new pages to the
appropriate `toctree` in `index.md`. Use short descriptions, explicit defaults and
units, and relative links to related pages.
