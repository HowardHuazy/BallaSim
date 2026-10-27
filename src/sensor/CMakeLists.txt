# Stream P (pixel/sensor, P1-M3): contract C3 of docs/dev/CONTRACTS.md.
#   balla_sensor_params host-only parameters: the rows and family priors, validation, maps, noise tables
#                       (no CUDA; balla_config and _core link it, docs/dev/P0_PARAMS.md §1.3)
#   balla_sensor        C++20 library (CPU kernel of pixel.h, multithreaded over pixels)
#   balla_sensor_aps    host-only APS readout model (aps.h; Phase 3 frames, docs/design/phase3/frames.md §4.10)
#   ballasim._sensor    nanobind module
#   balla_sensor_tests  doctest unit tests (ctest)
#   balla_resim_tests   doctest unit tests of the re-simulation (ROI model, output order, rederive; the fast tier)
#   balla_sensor_bench  CPU and GPU throughput benchmark (not a test; see docs/dev/SENSOR.md)
find_package(Threads REQUIRED)

add_library(balla_sensor_params STATIC
  params.cpp
  param_fields.cpp
  maps.cpp
  noise_tables.cpp)
target_link_libraries(balla_sensor_params PUBLIC balla_core)
balla_warnings(balla_sensor_params)

# The DAVIS-like APS frame sensor: its own small library, so that _sensor and the event path link none of it.
add_library(balla_sensor_aps STATIC aps.cpp)
target_link_libraries(balla_sensor_aps PUBLIC balla_core)
balla_warnings(balla_sensor_aps)

add_library(balla_sensor STATIC
  thread_pool.cpp
  sensor_model.cpp
  resim.cpp)
target_link_libraries(balla_sensor PUBLIC balla_core balla_sensor_params Threads::Threads)
balla_warnings(balla_sensor)

# The sensor census (census.h, PHASE2 P2-E E0): a study build whose CPU kernels count cell classes and bridge draws.
# KernelStats gains its counters, so the CUDA kernels (which sum into it) cannot share the build: CPU only.
option(BALLASIM_SENSOR_CENSUS "Build the sensor census counters (a CPU-only study build; docs/dev/SENSOR.md)" OFF)
if(BALLASIM_SENSOR_CENSUS)
  if(BALLASIM_HAVE_CUDA)
    message(FATAL_ERROR "BALLASIM_SENSOR_CENSUS needs a CPU-only build (BALLASIM_CUDA=OFF)")
  endif()
  target_compile_definitions(balla_sensor PUBLIC BALLASIM_SENSOR_CENSUS=1)
endif()

# fp32 kernel on the GPU (the same BALLA_HD code; bitwise equal to the CPU fp32 path).
if(BALLASIM_HAVE_CUDA)
  add_library(balla_sensor_cuda STATIC sensor_cuda.cu)
  target_link_libraries(balla_sensor_cuda PUBLIC balla_core CUDA::cudart_static)
  set_target_properties(balla_sensor_cuda PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  balla_warnings(balla_sensor_cuda)
  target_link_libraries(balla_sensor PUBLIC balla_sensor_cuda)
  target_compile_definitions(balla_sensor PUBLIC BALLASIM_SENSOR_CUDA=1)
endif()

# P2-E E5's dry launches (sensor_cuda.cu, slice_order.h): a study build that re-launches every general-pass slice in
# other thread orders on scratch states, timing only (BALLA_SENSOR_DRY=1 at run time). CUDA only; off in every other
# build, which compiles none of the dry launches.
option(BALLASIM_SENSOR_DRY "Build the sensor's dry launches (a CUDA study build; docs/dev/SENSOR.md)" OFF)
if(BALLASIM_SENSOR_DRY)
  if(NOT BALLASIM_HAVE_CUDA)
    message(FATAL_ERROR "BALLASIM_SENSOR_DRY needs a CUDA build (BALLASIM_CUDA=ON)")
  endif()
  target_compile_definitions(balla_sensor_cuda PRIVATE BALLASIM_SENSOR_DRY=1)
endif()

if(BALLASIM_BUILD_PYTHON)
  nanobind_add_module(_sensor STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE sensor_module.cpp)
  target_link_libraries(_sensor PRIVATE balla_sensor)
  balla_warnings(_sensor)
  install(TARGETS _sensor LIBRARY DESTINATION ballasim)
endif()

if(BALLASIM_BUILD_TESTS)
  set(_sensor_tests "${PROJECT_SOURCE_DIR}/tests/sensor")
  add_executable(balla_sensor_tests
    "${_sensor_tests}/main.cpp"
    "${_sensor_tests}/test_stage1.cpp"
    "${_sensor_tests}/test_expcubic.cpp"
    "${_sensor_tests}/test_noise.cpp"
    "${_sensor_tests}/test_pixel.cpp"
    "${_sensor_tests}/test_model.cpp"
    "${_sensor_tests}/test_cuda.cpp"
    "${_sensor_tests}/test_lean.cpp"
    "${_sensor_tests}/test_hpf.cpp"
    "${_sensor_tests}/test_limiter.cpp")
  target_link_libraries(balla_sensor_tests PRIVATE balla_sensor doctest::doctest)
  balla_warnings(balla_sensor_tests)
  # P2-E E5's slice-order cases (slice_order.h, the dry launches' per-slice key).
  target_sources(balla_sensor_tests PRIVATE "${_sensor_tests}/test_slice_order.cpp")
  add_test(NAME balla_sensor_tests COMMAND balla_sensor_tests)
  # 25-40 s: the slow tier (ctest -LE slow leaves it out; docs/dev/BUILD.md §4).
  set_tests_properties(balla_sensor_tests PROPERTIES LABELS slow)

  # The re-simulation's cases (phase3/ui.md §8.1 W8, S1-S5): small synthetic models, outside the slow label.
  add_executable(balla_resim_tests "${_sensor_tests}/main.cpp" "${_sensor_tests}/test_resim.cpp")
  target_link_libraries(balla_resim_tests PRIVATE balla_sensor doctest::doctest)
  balla_warnings(balla_resim_tests)
  add_test(NAME balla_resim_tests COMMAND balla_resim_tests)

  add_executable(balla_sensor_bench "${_sensor_tests}/bench_sensor.cpp")
  target_link_libraries(balla_sensor_bench PRIVATE balla_sensor)
  balla_warnings(balla_sensor_bench)
endif()

balla_lint_fp_flags(balla_sensor_params balla_sensor_aps balla_sensor balla_sensor_cuda _sensor balla_sensor_tests
                    balla_resim_tests balla_sensor_bench)
