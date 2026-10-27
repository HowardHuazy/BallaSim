# The parameter framework (docs/dev/P0_PARAMS.md): the schema of every tunable and the resolver.
#   balla_config        host-only library: the rows without a target, the concatenated schema and the accuracy
#                       profiles (schema.cpp); layers, resolution, validation, derivations and digests (resolve.cpp);
#                       the frames' values of a run's cameras (resolve_frames.cpp, frames.md §3.2).
#                       The targets' rows are header-only next to their structs (render/, engine/; engine/param_fields.h
#                       also holds apply()) or in balla_sensor_params
#   balla_config_tests  doctest: rows, field counts, custom-row round trips, defaults, family tables, the sensor
#                       maps at the defaults, D11 statistics, the label cut and the single PSF source; the
#                       resolver's diagnostics, derived_from by perturbation, explain and relevant (test_diagnostics);
#                       the IMU's rows in the resolver: the digest skip, validate and resolve_section (test_imu_section);
#                       the frames' rows, resolve_section by name and resolve_frames (test_frames_section)
add_library(balla_config STATIC schema.cpp resolve.cpp resolve_frames.cpp tier.cpp)
target_link_libraries(balla_config PUBLIC balla_core balla_sensor_params)
balla_warnings(balla_config)
set(_config_targets balla_config)

if(BALLASIM_BUILD_TESTS)
  add_executable(balla_config_tests "${PROJECT_SOURCE_DIR}/tests/cpp/main.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/test_params.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/test_diagnostics.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/test_imu_section.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/test_frames_section.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/test_tier.cpp")
  target_link_libraries(balla_config_tests PRIVATE balla_config balla_sensor doctest::doctest)
  balla_warnings(balla_config_tests)
  add_test(NAME balla_config_tests COMMAND balla_config_tests)
  list(APPEND _config_targets balla_config_tests)
endif()

balla_lint_fp_flags(${_config_targets})
