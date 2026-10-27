# The IMU (docs/dev/IMU.md; docs/design/phase3/outputs.md, P3-A). A post-process of a finished run: host only, fp64.
#   balla_imu        the IMU model: parameters, truth, filter and processes (header-only), the simulator
#                    (simulator.cpp)
#   ballasim._imu    nanobind module, always built with the Python bindings (it needs no compressor);
#                    _core.build_info()['imu'] reports it; resolve_imu (config::resolve_section) links balla_config,
#                    whose schema takes the rows of param_fields.h (src/config follows this directory)
#   balla_imu_tests  doctest: truth (U1-U3), filter (U4), processes (U5, U6, U9), device (U7, U8, U10, U11),
#                    parameters (U13, with the schema rows); links balla_sensor for U10's SensorModel::readout() and
#                    balla_config for the schema
add_library(balla_imu STATIC simulator.cpp)
target_link_libraries(balla_imu PUBLIC balla_core)
balla_warnings(balla_imu)
set(_imu_targets balla_imu)

if(BALLASIM_BUILD_PYTHON)
  nanobind_add_module(_imu STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE imu_module.cpp)
  target_link_libraries(_imu PRIVATE balla_imu balla_config)
  balla_warnings(_imu)
  install(TARGETS _imu LIBRARY DESTINATION ballasim)
  list(APPEND _imu_targets _imu)
endif()

if(BALLASIM_BUILD_TESTS)
  set(_imu_tests "${PROJECT_SOURCE_DIR}/tests/cpp/imu")
  add_executable(balla_imu_tests "${PROJECT_SOURCE_DIR}/tests/cpp/main.cpp" "${_imu_tests}/test_imu_params.cpp"
                 "${_imu_tests}/test_imu_truth.cpp" "${_imu_tests}/test_imu_filter.cpp"
                 "${_imu_tests}/test_imu_processes.cpp" "${_imu_tests}/test_imu_device.cpp"
                 "${_imu_tests}/test_imu_joins.cpp")
  target_link_libraries(balla_imu_tests PRIVATE balla_imu balla_sensor balla_config doctest::doctest)
  target_include_directories(balla_imu_tests PRIVATE "${PROJECT_SOURCE_DIR}/tests/cpp")
  balla_warnings(balla_imu_tests)
  add_test(NAME balla_imu_tests COMMAND balla_imu_tests)
  list(APPEND _imu_targets balla_imu_tests)
endif()

balla_lint_fp_flags(${_imu_targets})
