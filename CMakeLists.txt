cmake_minimum_required(VERSION 3.27)

# The version string has a single source, python/ballasim/__init__.py; scikit-build-core passes it in.
if(DEFINED SKBUILD_PROJECT_VERSION)
  set(_balla_version "${SKBUILD_PROJECT_VERSION}")
  set(_balla_version_full "${SKBUILD_PROJECT_VERSION_FULL}")
else()
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/python/ballasim/__init__.py" _balla_version_line
       REGEX "^__version__ = ")
  string(REGEX REPLACE "^__version__ = \"(.*)\"$" "\\1" _balla_version_full "${_balla_version_line}")
  string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _balla_version "${_balla_version_full}")
endif()

project(BallaSim VERSION ${_balla_version} LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(BALLASIM_CUDA AUTO CACHE STRING "Build the CUDA kernels and the _cuda module (AUTO, ON, OFF)")
set(BALLASIM_OPTIX AUTO CACHE STRING
    "Build the OptiX backend and the _optix module (AUTO, ON, OFF); needs OPTIX_ROOT")
set(BALLASIM_EXPORT AUTO CACHE STRING
    "Build the format exporters and the _export module (AUTO, ON, OFF); needs liblz4, libzstd and libbz2")
set_property(CACHE BALLASIM_CUDA PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE BALLASIM_OPTIX PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE BALLASIM_EXPORT PROPERTY STRINGS AUTO ON OFF)
option(BALLASIM_BUILD_TESTS "Build the C++ unit tests" ON)
option(BALLASIM_BUILD_PYTHON "Build the nanobind extension modules" ON)
option(BALLASIM_WERROR "Treat compiler warnings as errors" OFF)
set(BALLASIM_CPU_ARCH "x86-64-v3" CACHE STRING
    "-march level of the CPU core (DESIGN §4.6: AVX2 + FMA3; checked at import)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Installed modules find Embree (and the environment's libstdc++) where they were linked.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

include(Warnings)
include(FloatingPoint)
include(BallaCUDA)

# ---------------------------------------------------------------------------------------------
# Shared BALLA_HD core: header-only. Every target that includes it inherits the FP policy.
# ---------------------------------------------------------------------------------------------
add_library(balla_core INTERFACE)
add_library(balla::core ALIAS balla_core)
target_include_directories(balla_core INTERFACE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(balla_core INTERFACE cxx_std_20)
balla_apply_fp_policy(balla_core)
balla_apply_cpu_arch(balla_core)

# The browser UI's event wire (balla_wire STATIC; docs/design/phase3/ui.md §3.10)
add_subdirectory(src/core/wire)

# ---------------------------------------------------------------------------------------------
# Embree backend (reference and oracle; always built)
# ---------------------------------------------------------------------------------------------
find_package(embree 4.4 CONFIG REQUIRED)
add_library(balla_embree STATIC src/backend/embree/embree_mesh.cpp)
target_link_libraries(balla_embree PUBLIC balla_core PRIVATE embree)
balla_warnings(balla_embree)

# ---------------------------------------------------------------------------------------------
# CUDA kernels and the OptiX backend (optional)
# ---------------------------------------------------------------------------------------------
if(BALLASIM_HAVE_CUDA)
  add_library(balla_kernels STATIC src/kernels/spline_eval.cu)
  target_link_libraries(balla_kernels PUBLIC balla_core CUDA::cudart_static)
  set_target_properties(balla_kernels PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  balla_warnings(balla_kernels)
endif()

if(BALLASIM_HAVE_OPTIX)
  # The single definition of the OptiX function table and its once-per-process initialization;
  # every library that calls OptiX links this one (optix_function_table.h).
  add_library(balla_optix_runtime STATIC src/backend/optix/optix_function_table.cpp)
  target_include_directories(balla_optix_runtime PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(balla_optix_runtime PUBLIC OptiX::Headers)
  balla_warnings(balla_optix_runtime)

  # Device programs are compiled to OptiX-IR at build time and embedded into the library.
  add_library(balla_optix_ir OBJECT src/backend/optix/mesh_programs.cu)
  set_target_properties(balla_optix_ir PROPERTIES CUDA_OPTIX_COMPILATION ON)
  target_link_libraries(balla_optix_ir PRIVATE balla_core OptiX::Headers)
  balla_warnings(balla_optix_ir)
  balla_embed_binary(
    INPUT $<TARGET_OBJECTS:balla_optix_ir>
    DEPENDS balla_optix_ir
    SYMBOL kMeshProgramsIr
    NAMESPACE "balla::optix"
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/mesh_programs_ir.cpp")

  add_library(balla_optix STATIC
    src/backend/optix/optix_mesh.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/generated/mesh_programs_ir.cpp")
  target_link_libraries(balla_optix PUBLIC balla_core PRIVATE balla_optix_runtime CUDA::cudart_static)
  balla_warnings(balla_optix)
endif()

# ---------------------------------------------------------------------------------------------
# Python extension modules
# ---------------------------------------------------------------------------------------------
if(BALLASIM_BUILD_PYTHON)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
               OPTIONAL_COMPONENTS Development.SABIModule)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
                  COMMAND_ERROR_IS_FATAL ANY)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_core STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE
    src/bindings/core_module.cpp
    src/bindings/core_traj.cpp
    src/bindings/core_camera.cpp
    src/bindings/core_rng.cpp
    src/bindings/core_config.cpp
    src/bindings/core_wire.cpp)
  # balla_config (src/config, added below) holds the parameter schema; it is host-only, so _core stays free of CUDA.
  target_link_libraries(_core PRIVATE balla_core balla_embree balla_config balla_wire)
  # build_info() reports the optional modules of this build; the stream modules are defined later
  # (add_subdirectory below), which a generator expression sees.
  target_compile_definitions(_core PRIVATE
    BALLASIM_VERSION="${_balla_version_full}"
    BALLASIM_BUILT_WITH_CUDA=$<BOOL:${BALLASIM_HAVE_CUDA}>
    BALLASIM_BUILT_WITH_OPTIX=$<BOOL:${BALLASIM_HAVE_OPTIX}>
    BALLASIM_BUILT_WITH_RENDER=$<TARGET_EXISTS:_render>
    BALLASIM_BUILT_WITH_SENSOR=$<TARGET_EXISTS:_sensor>
    BALLASIM_BUILT_WITH_ENGINE=$<TARGET_EXISTS:_engine>
    BALLASIM_BUILT_WITH_SCENE=$<TARGET_EXISTS:_scene>
    BALLASIM_BUILT_WITH_IMU=$<TARGET_EXISTS:_imu>
    BALLASIM_BUILT_WITH_EXPORT=$<TARGET_EXISTS:_export>
    BALLASIM_BUILT_WITH_FRAMES=$<TARGET_EXISTS:_frames>)
  balla_warnings(_core)
  install(TARGETS _core LIBRARY DESTINATION ballasim)

  if(BALLASIM_HAVE_CUDA)
    nanobind_add_module(_cuda STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE src/bindings/cuda_module.cpp)
    target_link_libraries(_cuda PRIVATE balla_core balla_kernels)
    balla_warnings(_cuda)
    install(TARGETS _cuda LIBRARY DESTINATION ballasim)
  endif()

  if(BALLASIM_HAVE_OPTIX)
    nanobind_add_module(_optix STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE src/bindings/optix_module.cpp)
    target_link_libraries(_optix PRIVATE balla_core balla_optix)
    balla_warnings(_optix)
    install(TARGETS _optix LIBRARY DESTINATION ballasim)
  endif()
endif()

# ---------------------------------------------------------------------------------------------
# C++ unit tests (doctest)
# ---------------------------------------------------------------------------------------------
if(BALLASIM_BUILD_TESTS)
  enable_testing()
  find_package(doctest CONFIG REQUIRED)
  add_executable(balla_core_tests
    tests/cpp/main.cpp
    tests/cpp/test_time.cpp
    tests/cpp/test_pmath.cpp
    tests/cpp/test_special.cpp
    tests/cpp/test_so3.cpp
    tests/cpp/test_philox.cpp
    tests/cpp/test_sobol.cpp
    tests/cpp/test_warp.cpp
    tests/cpp/test_spline.cpp
    tests/cpp/test_camera.cpp
    tests/cpp/test_backend_api.cpp
    tests/cpp/test_embree.cpp
    tests/cpp/test_wire.cpp)
  target_link_libraries(balla_core_tests PRIVATE balla_core balla_embree balla_wire doctest::doctest)
  balla_warnings(balla_core_tests)
  add_test(NAME balla_core_tests COMMAND balla_core_tests)

  # Linux fault injection interposes pthread_create to test partial pool startup without
  # exhausting the machine's thread limit. Keep it isolated from the other test executables.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Threads REQUIRED)
    add_executable(balla_thread_pool_tests
      tests/cpp/main.cpp
      tests/cpp/test_thread_pool_failure.cpp
      src/render/thread_pool.cpp
      src/sensor/thread_pool.cpp)
    target_link_libraries(balla_thread_pool_tests PRIVATE balla_core doctest::doctest
                          Threads::Threads ${CMAKE_DL_LIBS})
    balla_warnings(balla_thread_pool_tests)
    balla_lint_fp_flags(balla_thread_pool_tests)
    add_test(NAME balla_thread_pool_tests COMMAND balla_thread_pool_tests)
    set_tests_properties(balla_thread_pool_tests PROPERTIES TIMEOUT 20)
  endif()

  # GPU tests: bitwise CPU/GPU parity of the shared maths (DESIGN §12.1 L1) and, with OptiX, ray
  # agreement with Embree (L2). Reported as skipped when no CUDA device is visible.
  if(BALLASIM_HAVE_CUDA)
    add_library(balla_gpu_test_kernels STATIC tests/gpu/parity.cu)
    target_link_libraries(balla_gpu_test_kernels PUBLIC balla_core CUDA::cudart_static)
    set_target_properties(balla_gpu_test_kernels PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    balla_warnings(balla_gpu_test_kernels)

    add_executable(balla_gpu_tests tests/gpu/main.cpp tests/gpu/test_parity.cpp
                   tests/gpu/test_spline_device.cpp)
    target_link_libraries(balla_gpu_tests PRIVATE balla_core balla_kernels balla_gpu_test_kernels
                          doctest::doctest)
    if(BALLASIM_HAVE_OPTIX)
      target_sources(balla_gpu_tests PRIVATE tests/gpu/test_optix.cpp)
      target_link_libraries(balla_gpu_tests PRIVATE balla_optix balla_embree)
    endif()
    balla_warnings(balla_gpu_tests)
    add_test(NAME balla_gpu_tests COMMAND balla_gpu_tests)
    set_tests_properties(balla_gpu_tests PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()

balla_lint_fp_flags(balla_core balla_embree balla_kernels balla_optix_runtime balla_optix balla_optix_ir
                    _core _cuda _optix balla_core_tests balla_gpu_test_kernels balla_gpu_tests)

# ---------------------------------------------------------------------------------------------
# Stream subprojects (docs/dev/CONTRACTS.md): Phase 1's four and Phase 3's IMU (docs/dev/IMU.md). Each
# stream owns its directory: libraries, its nanobind module (installed into ballasim/), its C++ tests,
# and its own balla_lint_fp_flags call. All five are required: _core links balla_config (the parameter
# schema, docs/dev/P0_PARAMS.md), which links balla_sensor_params and includes the render and engine
# headers for their rows. The order matters: imu precedes config, which takes its imu.* rows (P3-A A3).
# ---------------------------------------------------------------------------------------------
foreach(_balla_stream render sensor imu config engine)
  add_subdirectory(src/${_balla_stream})
endforeach()
# The scene side in C++ (docs/dev/CPP_PORT.md item 5): host-only, OpenBLAS; _core does not link it.
add_subdirectory(src/scene)
# Phase 3 outputs (docs/dev/EXPORT.md, IMU.md): the IMU above (always built) and the format exporters (optional: liblz4,
# libzstd and libbz2); post-processes of a finished run that _core does not link.
add_subdirectory(src/export)
# Frames (docs/design/phase3/frames.md): module ballasim._frames; _core does not link it.
add_subdirectory(src/frames)

# Each extension module links private copies of the static libraries; without this they would export balla::
# symbols, the OptiX function table and CUB's GNU-unique caches, which glibc merges across modules (and
# RTLD_GLOBAL would interpose). Only PyInit_<module> needs to be visible.
if(BALLASIM_BUILD_PYTHON AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  foreach(_balla_module _core _cuda _optix _render _sensor _engine _scene _imu _export _frames)
    if(TARGET ${_balla_module})
      target_link_options(${_balla_module} PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
  endforeach()
endif()

message(STATUS "BallaSim ${PROJECT_VERSION}: CUDA=${BALLASIM_HAVE_CUDA} OptiX=${BALLASIM_HAVE_OPTIX} "
               "CPU arch=${BALLASIM_CPU_ARCH}")
