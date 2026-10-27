// BALLA_HD marks functions shared by host and device code (DESIGN §4.4). Everything under
// src/core/ is header-only and must compile both as plain C++20 and as CUDA device code, so it
// uses no exceptions, no allocation and no std:: functions that lack device overloads.
#pragma once

#if defined(__CUDACC__)
#define BALLA_HD __host__ __device__
#else
#define BALLA_HD
#endif

// Full unrolling of loops with compile-time trip counts in device code. Host compilers unroll
// these small loops on their own, and GCC would warn about an unknown pragma.
#if defined(__CUDA_ARCH__)
#define BALLA_UNROLL _Pragma("unroll")
#else
#define BALLA_UNROLL
#endif

// One out-of-line copy per kernel of a large device function that its callers rarely execute (the
// special functions of pmath/special.h). Inlined at every call site, such functions made up most of
// the sensor kernels' code and spills (docs/dev/SENSOR.md "GPU execution"). Host code is unaffected.
#if defined(__CUDA_ARCH__)
#define BALLA_DEVICE_NOINLINE __noinline__
#else
#define BALLA_DEVICE_NOINLINE
#endif
