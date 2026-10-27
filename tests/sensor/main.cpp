#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "sensor/sensor_model.h"

// Without a CUDA sensor kernel or a device the GPU cases (test suite "cuda") are filtered out, so
// doctest counts them as skipped rather than passed.
int main(int argc, char** argv) {
  doctest::Context ctx;
  if (!balla::sensor::sensor_cuda_available()) ctx.addFilter("test-suite-exclude", "cuda");
  ctx.applyCommandLine(argc, argv);
  return ctx.run();
}
