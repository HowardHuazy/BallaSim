// Renderer test runner. OptiX test cases run only when render::optix_available() reports a CUDA
// device with an OptiX driver; the runner prints which backends it exercises. In a build with OptiX
// but no such device it exits with 77, which ctest reports as skipped, after the Embree cases pass.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cstdio>

#include "render/renderer.h"

int main(int argc, char** argv) {
  const bool optix = balla::render::optix_available();
  std::printf("render tests: backends embree%s\n", optix ? " + optix" : " only");
  doctest::Context context(argc, argv);
  const int res = context.run();
#if BALLA_RENDER_HAVE_OPTIX
  if (res == 0 && !optix && !context.shouldExit()) {
    std::puts("no OptiX device: the OptiX cases were skipped");
    return 77;
  }
#endif
  return res;
}
