# The format exporters (docs/dev/EXPORT.md; docs/design/phase3/outputs.md, P3-A): C++ writers that read a finished run.
#   balla_export        host-only library: block codecs over liblz4 (frame API), libzstd and libbz2, the CRC-32, the
#                       output sink with its SHA-256 at close and the compression pool (header-only parts); the time
#                       rules (time_map.h), the RAW and DAT writers (raw_writer.cpp over evt.h and dat.h) and the IMU
#                       CSV sidecar (imu_csv.h, A5); the ROS1 message definitions (msgdefs.h) and the rosbag1 writer
#                       (ros1.cpp, A6); the ROS 2 schemas, CDR (cdr.h) and the MCAP writer (mcap.cpp, A7), sharing the
#                       ROS writers' streams (ros_streams.h); the FlatBuffers builder (flatbuf_lite.h) and the AEDAT4
#                       writer (aedat4.cpp, A8); the frame streams of the AEDAT4, rosbag1 and MCAP writers with their
#                       8-bit map (frame_map.h) and the Kalibr files (kalibr.cpp, A9)
#   ballasim._export    nanobind module; _core.build_info()['export'] reports it
#   balla_export_tests  doctest: known answers (F9) and the pool's thread-count independence (F7); the RAW, DAT and
#                       CSV writers (F1 vectors, F4 edges, F7, F8 streaming); the ROS1 serializers and bag (F1 golden
#                       message and bag, F7, F8, the refusals); CDR and MCAP (F1 golden messages and file, the layout,
#                       F7, F8, the refusals); the FlatBuffers builder and AEDAT4 (F1 golden tables and file,
#                       dv-processing's own tables, the file re-read, F7, F8, the refusals); frames and Kalibr (F10
#                       maps, dv-processing's own FRME packets, the ROS1 and CDR images, the Kalibr texts, A9)
# Optional, as BALLASIM_EXPORT says: AUTO (the default) builds it when the three libraries and their headers are found
# (the ballasim environment has them, tools/env/environment.yml) and otherwise skips it with a status line, like the
# other optional modules; ON fails without them (release builds, PHASE3.md D-P3-7); OFF skips it.
if(BALLASIM_EXPORT STREQUAL "OFF")
  message(STATUS "Export module disabled: BALLASIM_EXPORT=OFF")
  return()
endif()
find_path(BALLA_LZ4_INCLUDE lz4frame.h)
find_library(BALLA_LZ4 NAMES lz4)
find_path(BALLA_ZSTD_INCLUDE zstd.h)
find_library(BALLA_ZSTD NAMES zstd)
find_path(BALLA_BZ2_INCLUDE bzlib.h)
find_library(BALLA_BZ2 NAMES bz2)
set(_export_missing "")
foreach(_v BALLA_LZ4_INCLUDE BALLA_LZ4 BALLA_ZSTD_INCLUDE BALLA_ZSTD BALLA_BZ2_INCLUDE BALLA_BZ2)
  if(NOT ${_v})
    list(APPEND _export_missing ${_v})
  endif()
endforeach()
if(_export_missing)
  set(_export_why "not found: ${_export_missing} (liblz4, libzstd and libbz2 with headers)")
  if(BALLASIM_EXPORT STREQUAL "ON")
    message(FATAL_ERROR "BALLASIM_EXPORT=ON: ${_export_why}")
  endif()
  message(STATUS "Export module disabled: ${_export_why}")
  return()
endif()

find_package(Threads REQUIRED)
add_library(balla_export STATIC compress.cpp raw_writer.cpp ros1.cpp mcap.cpp aedat4.cpp kalibr.cpp)
target_link_libraries(balla_export PUBLIC balla_core Threads::Threads
                      PRIVATE "${BALLA_LZ4}" "${BALLA_ZSTD}" "${BALLA_BZ2}")
target_include_directories(balla_export SYSTEM PRIVATE "${BALLA_LZ4_INCLUDE}" "${BALLA_ZSTD_INCLUDE}"
                           "${BALLA_BZ2_INCLUDE}")
balla_warnings(balla_export)
set(_export_targets balla_export)

if(BALLASIM_BUILD_PYTHON)
  nanobind_add_module(_export STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE export_module.cpp)
  target_link_libraries(_export PRIVATE balla_export)
  balla_warnings(_export)
  install(TARGETS _export LIBRARY DESTINATION ballasim)
  list(APPEND _export_targets _export)
endif()

if(BALLASIM_BUILD_TESTS)
  add_executable(balla_export_tests "${PROJECT_SOURCE_DIR}/tests/cpp/main.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/export/test_export_infra.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/export/test_export_raw.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/export/test_export_ros1.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/export/test_export_mcap.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/export/test_export_aedat4.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/cpp/export/test_export_frames.cpp")
  target_link_libraries(balla_export_tests PRIVATE balla_export doctest::doctest)
  balla_warnings(balla_export_tests)
  add_test(NAME balla_export_tests COMMAND balla_export_tests)
  list(APPEND _export_targets balla_export_tests)
endif()

balla_lint_fp_flags(${_export_targets})
