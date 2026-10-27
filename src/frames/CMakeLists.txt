# Frames (docs/design/phase3/frames.md): reference, matched and full RGB frames on the event clock.
#
# Targets:
#   balla_frames        host library: frame schedules and sensor-clock stamps (timing), the exposure reference
#                       (reference: BallaSim's own exposure integral, the edge-mask probe, visibility), APS frames
#                       (aps_frames: photometry inputs and the frame loop over sensor/aps.h; frames_digest), route S's
#                       resampler (resample: overscan source, quadrature and cells operators), the Cycles timing
#                       (cycles_timing: each sub-render's frame, shutter and rolling duration, the realized times and
#                       snap, node times, motion steps, the sub-render and row-band rules; on the scene library's
#                       TimeMap); later increments add the plan; links balla_config for resolve_frames, which _frames binds
#                       (the frames' rows are header-only: param_fields.h, registered by config/schema.cpp)
#   _frames             nanobind module installed as ballasim._frames (bindings/frames_module.cpp)
#   balla_frames_tests  doctest: waveform interval means (U1), schedules (U2), Cycles timing (U3), the exposure
#                       reference (U4), the resampler (U5, I-M3 synthetic), APS (U9)
add_library(balla_frames STATIC timing.cpp reference.cpp aps_frames.cpp resample.cpp cycles_timing.cpp)
target_link_libraries(balla_frames PUBLIC balla_core balla_render balla_sensor_params balla_sensor_aps balla_scene balla_config)
balla_warnings(balla_frames)
set(_frames_targets balla_frames)

if(BALLASIM_BUILD_PYTHON)
  nanobind_add_module(_frames STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE bindings/frames_module.cpp)
  target_link_libraries(_frames PRIVATE balla_frames)
  balla_warnings(_frames)
  install(TARGETS _frames LIBRARY DESTINATION ballasim)
  list(APPEND _frames_targets _frames)
endif()

if(BALLASIM_BUILD_TESTS)
  add_executable(balla_frames_tests "${PROJECT_SOURCE_DIR}/tests/cpp/main.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/frames/test_waveform_mean.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/frames/test_schedule.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/frames/test_cycles_timing.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/frames/test_reference.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/frames/test_aps.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/frames/test_resample.cpp")
  target_include_directories(balla_frames_tests PRIVATE "${PROJECT_SOURCE_DIR}/tests/render")
  target_link_libraries(balla_frames_tests PRIVATE balla_frames doctest::doctest)
  balla_warnings(balla_frames_tests)
  add_test(NAME balla_frames_tests COMMAND balla_frames_tests)
  list(APPEND _frames_targets balla_frames_tests)
endif()

balla_lint_fp_flags(${_frames_targets})
