# Configure-time lint: fail if any flag that relaxes IEEE semantics reaches BallaSim code.
set(_BALLA_BANNED_FP_FLAGS
  "-ffast-math" "-Ofast" "--use_fast_math" "-use_fast_math" "-funsafe-math-optimizations"
  "-fassociative-math" "-freciprocal-math" "-ffinite-math-only" "-ffp-contract=fast"
  "-ffp-contract=on" "--fmad=true" "-fmad=true" "-ftz=true" "--ftz=true")

# The first banned flag in `flags`, or "". Generator expressions ($<...:-ffast-math>), -Xcompiler=a,b lists and
# quoting are split into plain tokens first, so wrapped flags are found too.
function(_balla_banned_fp_flag out flags)
  string(REGEX REPLACE "[:>,;\"]" " " _tokens "${flags}")
  string(REPLACE "=-" " -" _tokens "${_tokens}")
  set(_hit "")
  foreach(_flag IN LISTS _BALLA_BANNED_FP_FLAGS)
    string(FIND " ${_tokens} " " ${_flag} " _pos)
    if(NOT _pos EQUAL -1)
      set(_hit "${_flag}")
      break()
    endif()
  endforeach()
  set(${out} "${_hit}" PARENT_SCOPE)
endfunction()

function(_balla_check_flags origin flags)
  _balla_banned_fp_flag(_flag "${flags}")
  if(_flag)
    message(FATAL_ERROR "Floating-point policy violation: '${_flag}' in ${origin}. "
                        "BallaSim bans fast-math style flags (DESIGN §4.6).")
  endif()
endfunction()

# Checks the global flags and the compile options of each existing target in the argument list.
function(balla_lint_fp_flags)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _cfg)
  foreach(_lang C CXX CUDA)
    _balla_check_flags("CMAKE_${_lang}_FLAGS" "${CMAKE_${_lang}_FLAGS} ${CMAKE_${_lang}_FLAGS_${_cfg}}")
  endforeach()
  # nvcc reads these at build time, after configure: a banned flag there would bypass every check above.
  foreach(_var NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS)
    _balla_check_flags("environment variable ${_var}" "$ENV{${_var}}")
  endforeach()
  foreach(_target IN LISTS ARGN)
    if(NOT TARGET ${_target})
      continue()
    endif()
    foreach(_prop COMPILE_OPTIONS INTERFACE_COMPILE_OPTIONS)
      get_target_property(_opts ${_target} ${_prop})
      if(_opts)
        list(JOIN _opts " " _opts)
        _balla_check_flags("${_prop} of ${_target}" "${_opts}")
      endif()
    endforeach()
  endforeach()
endfunction()
