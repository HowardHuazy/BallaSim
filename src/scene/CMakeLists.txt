# The scene side in C++ (docs/dev/CPP_PORT.md item 5): module ballasim._scene.
#
# Targets:
#   balla_scene        host-only library: NumPy's order of operations (npo, npla), SciPy's sparse structure, SciPy's
#                      Rotation, the EVIMO2 trajectory fitter (traj/) and the EVIMO2 bundle build (evimo2/, with the
#                      mesh geometry and an exact k-d tree), the EVIMO object catalogue and the PLY import
#                      (assets/, libpng), bitwise as the Python, the Replica import and the scan report's mesh
#                      statistics; BLAS and LAPACK from the environment's OpenBLAS, as NumPy and SciPy call it; the
#                      fits on render's thread pool
#   _scene             nanobind module installed as ballasim._scene: build_evimo2 (the adapter's build), the Blender
#                      exchange's statics (blender/, docs/design/phase3/blender.md §3.8) and the test bindings
#   balla_scene_tests  doctest: sparse structure, OpenBLAS one-thread pinning in pool threads
#   balla_motion_tests doctest: the Blender motion session's clock, decoding, error model, sampler, declared keys,
#                      breakpoint detection, piecewise fit, validation, rigs, modes and the session itself (motion/,
#                      docs/design/phase3/blender.md §4.2, §4.7), against synthetic and recorded oracles

# The OpenBLAS NumPy and SciPy load (conda: libcblas.so / liblapack.so are links to it; its soname is
# libopenblas.so.0, so the process keeps one instance).
find_library(BALLA_OPENBLAS NAMES openblas cblas REQUIRED)
# The environment's libpng (textures of the PLY import, S5).
find_library(BALLA_PNG NAMES png16 png REQUIRED)
find_path(BALLA_PNG_INCLUDE png.h REQUIRED)

add_library(balla_scene STATIC
  blas.cpp
  npo.cpp
  npla.cpp
  pyfmt.cpp
  sparse.cpp
  rotation.cpp
  geometry.cpp
  kdtree.cpp
  traj/basis.cpp
  traj/banded.cpp
  traj/retime.cpp
  traj/gating.cpp
  traj/holds.cpp
  traj/pin.cpp
  traj/fit_smooth.cpp
  evimo2/build.cpp
  evimo2/catalogue.cpp
  blender/convert.cpp
  blender/assemble.cpp
  blender/mesh.cpp
  blender/crop.cpp
  blender/materials.cpp
  blender/camera.cpp
  blender/lights.cpp
  motion/time_map.cpp
  motion/decode.cpp
  motion/static_distance.cpp
  motion/error_model.cpp
  motion/sampler.cpp
  motion/pipeline.cpp
  motion/driver.cpp
  motion/keys.cpp
  motion/detect.cpp
  motion/fit_piecewise.cpp
  motion/checks.cpp
  motion/session.cpp
  assets/png.cpp
  assets/ply.cpp
  assets/replica.cpp
  assets/mesh_stats.cpp
  "${PROJECT_SOURCE_DIR}/src/render/thread_pool.cpp")
target_link_libraries(balla_scene PUBLIC balla_core "${BALLA_OPENBLAS}" "${BALLA_PNG}")
target_include_directories(balla_scene SYSTEM PRIVATE "${BALLA_PNG_INCLUDE}")
balla_warnings(balla_scene)
set(_scene_targets balla_scene)

if(BALLASIM_BUILD_PYTHON)
  nanobind_add_module(_scene STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE bindings/scene_module.cpp
                      bindings/blender_bindings.cpp)
  target_link_libraries(_scene PRIVATE balla_scene)
  balla_warnings(_scene)
  install(TARGETS _scene LIBRARY DESTINATION ballasim)
  list(APPEND _scene_targets _scene)
endif()

if(BALLASIM_BUILD_TESTS)
  find_package(Threads REQUIRED)
  add_executable(balla_scene_tests "${PROJECT_SOURCE_DIR}/tests/cpp/main.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/scene/test_scene.cpp")
  target_link_libraries(balla_scene_tests PRIVATE balla_scene doctest::doctest Threads::Threads)
  balla_warnings(balla_scene_tests)
  add_test(NAME balla_scene_tests COMMAND balla_scene_tests)
  list(APPEND _scene_targets balla_scene_tests)

  add_executable(balla_motion_tests "${PROJECT_SOURCE_DIR}/tests/cpp/main.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/scene/test_motion.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/scene/test_motion_oracles.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/scene/test_motion_keys.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/scene/test_motion_detect.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/scene/test_motion_fit.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/scene/test_motion_session.cpp")
  target_link_libraries(balla_motion_tests PRIVATE balla_scene doctest::doctest Threads::Threads)
  balla_warnings(balla_motion_tests)
  add_test(NAME balla_motion_tests COMMAND balla_motion_tests)
  list(APPEND _scene_targets balla_motion_tests)
endif()

balla_lint_fp_flags(${_scene_targets})
