// GPU test runner: exits with 77, which ctest reports as skipped, when no CUDA device is visible or no
// driver is installed; a driver that is present but unusable is a failure, not a skip.
#define DOCTEST_CONFIG_IMPLEMENT
#include <cuda_runtime.h>
#include <doctest/doctest.h>

#include <cstdio>

int main(int argc, char** argv) {
  int n = 0, driver = 0;
  const cudaError_t err = cudaGetDeviceCount(&n);
  cudaDriverGetVersion(&driver);  // 0 when no driver is installed
  if (err == cudaErrorNoDevice || (err == cudaErrorInsufficientDriver && driver == 0) || (err == cudaSuccess && n == 0)) {
    std::puts("no CUDA device or driver: skipping the GPU tests");
    return 77;
  }
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA unusable (driver version %d): %s\n", driver, cudaGetErrorString(err));
    return 1;
  }
  doctest::Context context(argc, argv);
  return context.run();
}
