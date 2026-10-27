# Floating-point policy (DESIGN §4.6, §12.1).
#
# The shared BALLA_HD maths must give bitwise-identical results on the CPU and the GPU (L1), so
# neither compiler may contract a*b+c into an fma, reassociate, flush denormals or use approximate
# division and square roots. Explicit fma calls remain allowed where an algorithm wants them.
include(FlagLint)

function(balla_apply_fp_policy target)
  target_compile_options(${target} INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off -fno-fast-math>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -ftz=false -prec-div=true -prec-sqrt=true
                               -Xcompiler=-ffp-contract=off,-fno-fast-math>)
endfunction()

# x86-64-v3 (AVX2 + FMA3) is the CPU baseline: on baseline x86-64 every explicit fma becomes an
# out-of-line libm call. The extension checks the CPU at import time.
function(balla_apply_cpu_arch target)
  if(BALLASIM_CPU_ARCH AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(${target} INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:-march=${BALLASIM_CPU_ARCH}>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=${BALLASIM_CPU_ARCH}>)
  endif()
endfunction()
