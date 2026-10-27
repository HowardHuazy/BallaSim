# Finds user-installed OptiX SDK headers through OPTIX_ROOT only (cache variable or environment).
#
# OptiX headers are proprietary: they are never vendored, copied or downloaded. The runtime
# (libnvoptix.so.1) ships with the NVIDIA driver and is loaded by optixInit() through dlopen.
#
# Result: OptiXUser_FOUND, OptiXUser_VERSION, OptiXUser_INCLUDE_DIR, imported target OptiX::Headers.
if(NOT OPTIX_ROOT AND DEFINED ENV{OPTIX_ROOT})
  set(OPTIX_ROOT "$ENV{OPTIX_ROOT}")
endif()
set(OPTIX_ROOT "${OPTIX_ROOT}" CACHE PATH "Root of the OptiX SDK (contains include/optix.h)")

# find_path() never searches again once it has a cached hit, so a reconfigure with another (or no)
# OPTIX_ROOT would keep the old SDK. Forget the result whenever the root changes.
if(NOT "${OPTIX_ROOT}" STREQUAL "${_OPTIXUSER_SEARCHED_ROOT}")
  unset(OptiXUser_INCLUDE_DIR CACHE)
  set(_OPTIXUSER_SEARCHED_ROOT "${OPTIX_ROOT}" CACHE INTERNAL "OPTIX_ROOT of the last OptiX search")
endif()

if(OPTIX_ROOT)
  find_path(OptiXUser_INCLUDE_DIR NAMES optix.h PATHS "${OPTIX_ROOT}/include"
            NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
endif()

if(OptiXUser_INCLUDE_DIR)
  file(STRINGS "${OptiXUser_INCLUDE_DIR}/optix.h" _optix_version_line
       REGEX "^#define OPTIX_VERSION [0-9]+")
  string(REGEX REPLACE "^#define OPTIX_VERSION ([0-9]+).*" "\\1" _optix_v "${_optix_version_line}")
  math(EXPR _optix_major "${_optix_v} / 10000")
  math(EXPR _optix_minor "(${_optix_v} % 10000) / 100")
  math(EXPR _optix_micro "${_optix_v} % 100")
  set(OptiXUser_VERSION "${_optix_major}.${_optix_minor}.${_optix_micro}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiXUser
  REQUIRED_VARS OptiXUser_INCLUDE_DIR
  VERSION_VAR OptiXUser_VERSION)

if(OptiXUser_FOUND AND NOT TARGET OptiX::Headers)
  add_library(OptiX::Headers INTERFACE IMPORTED)
  target_include_directories(OptiX::Headers SYSTEM INTERFACE "${OptiXUser_INCLUDE_DIR}")
  target_link_libraries(OptiX::Headers INTERFACE ${CMAKE_DL_LIBS})
endif()
mark_as_advanced(OptiXUser_INCLUDE_DIR)
