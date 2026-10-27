# CUDA toolchain and OptiX discovery.
#
# Outputs: BALLASIM_HAVE_CUDA, BALLASIM_HAVE_OPTIX, targets CUDA::cudart_static and OptiX::Headers.
# nvcc 12.8 rejects gcc 15, so the CUDA host compiler defaults to the C++ compiler of the project,
# which the presets and the conda environment set to the conda gcc 14 toolchain (DESIGN §4.4). It
# is passed explicitly (-ccbin) rather than left to conda's NVCC_PREPEND_FLAGS.
include(CheckLanguage)

set(BALLASIM_HAVE_CUDA OFF)
set(BALLASIM_HAVE_OPTIX OFF)

if(NOT BALLASIM_CUDA STREQUAL "OFF")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures (75 = RTX 2080 Ti)")
  endif()
  if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "nvcc host compiler" FORCE)
  endif()
  # The probe project of check_language() sees only the environment, not this cache entry.
  set(ENV{CUDAHOSTCXX} "${CMAKE_CUDA_HOST_COMPILER}")
  check_language(CUDA)
  # check_language() also leaves the probe's CMAKE_CUDA_HOST_COMPILER behind as a normal variable;
  # drop it so the cache entry above stays authoritative.
  unset(CMAKE_CUDA_HOST_COMPILER)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    find_package(CUDAToolkit REQUIRED)
    set(BALLASIM_HAVE_CUDA ON)
  elseif(BALLASIM_CUDA STREQUAL "ON")
    message(FATAL_ERROR "BALLASIM_CUDA=ON but no CUDA compiler was found")
  endif()
endif()

if(NOT BALLASIM_OPTIX STREQUAL "OFF")
  if(NOT BALLASIM_HAVE_CUDA)
    if(BALLASIM_OPTIX STREQUAL "ON")
      message(FATAL_ERROR "BALLASIM_OPTIX=ON requires CUDA")
    endif()
  else()
    if(BALLASIM_OPTIX STREQUAL "ON")
      find_package(OptiXUser 9.1 REQUIRED)
    else()
      find_package(OptiXUser 9.1 QUIET)
    endif()
    if(OptiXUser_FOUND)
      set(BALLASIM_HAVE_OPTIX ON)
    else()
      message(STATUS "OptiX backend disabled: set OPTIX_ROOT to an OptiX 9.1 SDK whose licence you "
                     "have accepted (headers only; they are never copied or downloaded).")
    endif()
  endif()
endif()

include(EmbedBinary)
