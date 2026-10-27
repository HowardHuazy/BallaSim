# Installation

BallaSim is built from source on Linux x86-64. Python 3.12 or later and a CPU
with AVX2 and FMA3 are required. The supplied environment uses Python 3.12.

## Requirements

| Component | Required version or purpose |
| --- | --- |
| C++ compiler | C++20; the supplied environment uses GCC 14 |
| CMake and Ninja | CMake 3.27 or later; Ninja builds the native modules |
| Embree | 4.4 or later; required for CPU ray tracing |
| OpenBLAS and libpng | Required by scene import and trajectory fitting |
| CUDA toolkit | Optional; the supplied environment uses CUDA 12.8 |
| OptiX SDK | Optional; OptiX 9.1 headers and an NVIDIA R590 or newer driver |
| LZ4, Zstandard and bzip2 | Required for the optional export module |

The GPU engine requires both CUDA and OptiX. A CUDA-only build provides CUDA
kernels but uses the CPU engine for full simulations.

## Create the environment

From the repository root, create and activate the pinned conda environment:

```bash
conda create -n ballasim --file tools/env/linux-64.lock
conda activate ballasim
```

Alternatively, resolve a new environment from `tools/env/environment.yml` with
`conda env create -f tools/env/environment.yml`.

## Install the package

For a CPU build:

```bash
BALLASIM_CUDA=OFF BALLASIM_OPTIX=OFF \
  python -m pip install --no-build-isolation --no-deps -e .
```

For a GPU build, install the OptiX SDK separately and set `OPTIX_ROOT` to the
directory containing `include/optix.h`:

```bash
export OPTIX_ROOT=/path/to/optix-9.1.0
BALLASIM_CUDA=ON BALLASIM_OPTIX=ON \
  python -m pip install --no-build-isolation --no-deps -e .
```

`BALLASIM_CUDA`, `BALLASIM_OPTIX` and `BALLASIM_EXPORT` accept `AUTO`, `ON` or
`OFF`. `AUTO` skips unavailable optional components; `ON` requires them. All three
default to `AUTO`.

CUDA builds target compute capability 7.5 by default. To select another target,
add, for example, `-C cmake.define.CMAKE_CUDA_ARCHITECTURES=86` to the install
command.

## Optional dependencies

| Extra | Provides |
| --- | --- |
| `viz` | Matplotlib and OpenCV for event visualization |
| `ui` | Tornado and the visualization dependencies for the browser interface |
| `test` | pytest, Hypothesis, visualization and browser UI dependencies |

To install an extra from this checkout, replace `-e .` with `-e '.[ui]'` in the
chosen install command and omit `--no-deps`. The pinned environment already
includes the visualization and test dependencies.

## Inspect the installation

```bash
python -m ballasim probe --json
```

The report lists the package version, interpreter, compiled modules and supported
protocols. A built GPU module still requires a usable GPU and driver at runtime.

Rerun the install command after editing C++ or CUDA sources; editable installs do
not rebuild native modules automatically. For build presets and checks, see
[Contributing](../development/contributing.md).
