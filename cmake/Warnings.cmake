# Compiler warnings for BallaSim's own targets (never applied to third-party code).
function(balla_warnings target)
  set(_cxx_warnings
    -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual -Wcast-qual
    -Wformat=2 -Wimplicit-fallthrough
    # Accidental float<->double conversions break the fp32/fp64 contracts of the shared maths.
    -Wdouble-promotion -Wfloat-conversion)
  if(BALLASIM_WERROR)
    list(APPEND _cxx_warnings -Werror)
  endif()
  list(JOIN _cxx_warnings "," _host_warnings)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${_cxx_warnings}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wdouble-promotion,-Wfloat-conversion>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${BALLASIM_WERROR}>>:--Werror=all-warnings>)
endfunction()
