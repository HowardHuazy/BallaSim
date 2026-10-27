# The C++/CUDA engine (docs/dev/ENGINE_CPP.md): module ballasim._engine.
#
# Targets:
#   balla_engine        host instantiation of the WindowEngine and the Engine (Embree, the CPU sensor;
#                       always); with CUDA and OptiX also the device instantiation (BALLA_ENGINE_HAVE_DEVICE)
#   _engine             nanobind module installed as ballasim._engine
#   balla_engine_tests  doctest: primitives host = device, assembly, eta_L goldens, batching, the host engine

add_library(balla_engine STATIC par_host.cpp window.cpp motion.cpp engine_host.cpp)
target_link_libraries(balla_engine PUBLIC balla_core balla_render balla_sensor)
set(_engine_device OFF)
if(BALLASIM_HAVE_OPTIX AND TARGET balla_sensor_cuda)
  set(_engine_device ON)
  target_sources(balla_engine PRIVATE par_cuda.cu engine.cu)
  target_link_libraries(balla_engine PRIVATE CUDA::cudart_static)
  set_target_properties(balla_engine PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  target_compile_definitions(balla_engine PUBLIC BALLA_ENGINE_HAVE_DEVICE=1)
else()
  target_compile_definitions(balla_engine PUBLIC BALLA_ENGINE_HAVE_DEVICE=0)
endif()
balla_warnings(balla_engine)
set(_engine_targets balla_engine)

if(BALLASIM_BUILD_PYTHON)
  nanobind_add_module(_engine STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE bindings/engine_module.cpp)
  target_link_libraries(_engine PRIVATE balla_engine balla_sensor)
  balla_warnings(_engine)
  install(TARGETS _engine LIBRARY DESTINATION ballasim)
  list(APPEND _engine_targets _engine)
endif()

if(BALLASIM_BUILD_TESTS)
  set(_engine_tests "${PROJECT_SOURCE_DIR}/tests/cpp/engine")
  add_executable(balla_engine_tests "${_engine_tests}/main.cpp" "${_engine_tests}/test_window.cpp"
                 "${_engine_tests}/test_engine.cpp" "${_engine_tests}/test_plan.cpp"
                 "${_engine_tests}/test_localize.cpp" "${_engine_tests}/test_motion.cpp")
  target_include_directories(balla_engine_tests PRIVATE "${PROJECT_SOURCE_DIR}/tests/render")
  target_link_libraries(balla_engine_tests PRIVATE balla_engine doctest::doctest)
  if(_engine_device)
    target_sources(balla_engine_tests PRIVATE "${_engine_tests}/device_helpers.cu" "${_engine_tests}/test_device.cpp")
    target_link_libraries(balla_engine_tests PRIVATE CUDA::cudart_static)
    set_target_properties(balla_engine_tests PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  balla_warnings(balla_engine_tests)
  add_test(NAME balla_engine_tests COMMAND balla_engine_tests)
  list(APPEND _engine_targets balla_engine_tests)
endif()

balla_lint_fp_flags(${_engine_targets})
