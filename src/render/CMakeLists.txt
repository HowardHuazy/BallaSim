# Stream R (P1-M2): the renderer of contract C2 (docs/dev/CONTRACTS.md, docs/dev/RENDER.md).
#
# Targets:
#   balla_render_core     bundle, preprocessed scene, pose table, Embree tracer, host executor
#   balla_render_device   CUDA executor kernels + OptiX tracer (BALLASIM_HAVE_OPTIX)
#   balla_render          the Renderer, on top of both
#   _render               nanobind module installed as ballasim._render
#   balla_render_tests    doctest unit and oracle tests (CPU; OptiX cases skip without a device)
#   balla_render_bench    throughput benchmark (tests/render/bench.cpp)

set(_render_tests "${PROJECT_SOURCE_DIR}/tests/render")

add_library(balla_render_core STATIC
  ao_proxy.cpp
  bundle.cpp
  patches.cpp
  reflection_policy.cpp
  scene_data.cpp
  texture.cpp
  thread_pool.cpp
  pose_table.cpp
  host_executor.cpp
  embree/embree_tracer.cpp)
target_link_libraries(balla_render_core PUBLIC balla_core PRIVATE embree)
balla_warnings(balla_render_core)

add_library(balla_render STATIC renderer.cpp)
target_link_libraries(balla_render PUBLIC balla_render_core)
balla_warnings(balla_render)
set(_render_targets balla_render_core balla_render)

if(BALLASIM_HAVE_OPTIX)
  # Device programs: OptiX-IR, embedded in the library (no files are loaded at runtime).
  add_library(balla_render_optix_ir OBJECT optix/programs.cu)
  set_target_properties(balla_render_optix_ir PROPERTIES CUDA_OPTIX_COMPILATION ON)
  target_link_libraries(balla_render_optix_ir PRIVATE balla_core OptiX::Headers)
  balla_warnings(balla_render_optix_ir)
  balla_embed_binary(
    INPUT $<TARGET_OBJECTS:balla_render_optix_ir>
    DEPENDS balla_render_optix_ir
    SYMBOL kRenderProgramsIr
    NAMESPACE "balla::render"
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/render_programs_ir.cpp")

  add_library(balla_render_device STATIC
    cuda/device_executor.cu
    optix/optix_tracer.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/generated/render_programs_ir.cpp")
  target_link_libraries(balla_render_device PUBLIC balla_render_core PRIVATE balla_optix_runtime
                        CUDA::cudart_static)
  set_target_properties(balla_render_device PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  balla_warnings(balla_render_device)
  target_link_libraries(balla_render PRIVATE balla_render_device)
  target_compile_definitions(balla_render PUBLIC BALLA_RENDER_HAVE_OPTIX=1)
  list(APPEND _render_targets balla_render_optix_ir balla_render_device)
else()
  target_compile_definitions(balla_render PUBLIC BALLA_RENDER_HAVE_OPTIX=0)
endif()

# BSDF tables (p2f.md §4.3): the fp64 generator of shade/bsdf_tables.inc. The `bsdf_tables` target rewrites the
# checked-in file; the ctest bsdf_tables_regen regenerates it into the build tree and requires bitwise equality.
find_package(Threads REQUIRED)
add_executable(balla_bsdf_gen_tables "${PROJECT_SOURCE_DIR}/tools/bsdf_tables/gen_tables.cpp")
target_link_libraries(balla_bsdf_gen_tables PRIVATE balla_core Threads::Threads)
balla_warnings(balla_bsdf_gen_tables)
if(NOT BALLASIM_BUILD_TESTS)
  set_target_properties(balla_bsdf_gen_tables PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
add_custom_target(bsdf_tables
  COMMAND balla_bsdf_gen_tables --out "${CMAKE_CURRENT_SOURCE_DIR}/shade/bsdf_tables.inc"
  COMMENT "Regenerating src/render/shade/bsdf_tables.inc"
  VERBATIM)
list(APPEND _render_targets balla_bsdf_gen_tables)

if(BALLASIM_BUILD_PYTHON)
  nanobind_add_module(_render STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE bindings/render_module.cpp)
  target_link_libraries(_render PRIVATE balla_render)
  balla_warnings(_render)
  install(TARGETS _render LIBRARY DESTINATION ballasim)
  list(APPEND _render_targets _render)
endif()

if(BALLASIM_BUILD_TESTS)
  add_executable(balla_render_tests
    "${_render_tests}/main.cpp"
    "${_render_tests}/test_ewa.cpp"
    "${_render_tests}/test_shading.cpp"
    "${_render_tests}/test_area_lights.cpp"
    "${_render_tests}/test_bsdf.cpp"
    "${_render_tests}/test_pose.cpp"
    "${_render_tests}/test_renderer.cpp"
    "${_render_tests}/test_patches.cpp"
    "${_render_tests}/test_lobe.cpp"
    "${_render_tests}/test_ao.cpp"
    "${_render_tests}/test_codes.cpp"
    "${_render_tests}/test_bundle.cpp"
    "${_render_tests}/test_baked.cpp"
    "${_render_tests}/test_pieces.cpp"
    "${_render_tests}/test_nir.cpp"
    "${_render_tests}/test_tile.cpp")
  target_include_directories(balla_render_tests PRIVATE "${_render_tests}")
  target_link_libraries(balla_render_tests PRIVATE balla_render doctest::doctest)
  if(BALLASIM_HAVE_OPTIX)
    # Bitwise CPU/GPU parity of the device executor; skips itself without a CUDA device.
    target_sources(balla_render_tests PRIVATE "${_render_tests}/test_device.cpp")
    target_link_libraries(balla_render_tests PRIVATE balla_render_device CUDA::cudart_static)
  endif()
  balla_warnings(balla_render_tests)
  add_test(NAME balla_render_tests COMMAND balla_render_tests)
  set_tests_properties(balla_render_tests PROPERTIES SKIP_RETURN_CODE 77)  # OptiX build without a device

  # test_bsdf.cpp shares the generator's fp64 references (tables_ref.h).
  target_include_directories(balla_render_tests PRIVATE "${PROJECT_SOURCE_DIR}/tools/bsdf_tables")
  add_test(NAME bsdf_tables_regen
           COMMAND balla_bsdf_gen_tables --out "${CMAKE_CURRENT_BINARY_DIR}/bsdf_tables_regen.inc"
                   --check "${CMAKE_CURRENT_SOURCE_DIR}/shade/bsdf_tables.inc")

  add_executable(balla_render_bench "${_render_tests}/bench.cpp")
  target_include_directories(balla_render_bench PRIVATE "${_render_tests}")
  target_link_libraries(balla_render_bench PRIVATE balla_render)
  balla_warnings(balla_render_bench)
  # Smoke run of the benchmark on a reduced scene; the full measurement is run by hand.
  add_test(NAME balla_render_bench_smoke COMMAND balla_render_bench --smoke)
  set_tests_properties(balla_render_bench_smoke PROPERTIES LABELS perf)
  list(APPEND _render_targets balla_render_tests balla_render_bench)
endif()

balla_lint_fp_flags(${_render_targets})
