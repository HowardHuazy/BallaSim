// Device versions of the engine's primitives (par.h): CUB on the caller's stream, with the caller's
// temporary storage.
#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <algorithm>
#include <string>

#include "engine/par.h"
#include "kernels/device_array.h"

namespace balla::engine {
namespace {

using kernels::check_cuda;

cudaStream_t st(Stream s) { return static_cast<cudaStream_t>(s); }

}  // namespace

template <>
void* allocate<Space::kDevice>(size_t bytes) {
  void* p = nullptr;
  check_cuda(cudaMalloc(&p, std::max<size_t>(bytes, 1)), "engine cudaMalloc");
  return p;
}

template <>
void release<Space::kDevice>(void* p) {
  cudaFree(p);
}

template <>
void* allocate_host<Space::kDevice>(size_t bytes) {
  void* p = nullptr;
  check_cuda(cudaMallocHost(&p, std::max<size_t>(bytes, 1)), "engine cudaMallocHost");
  return p;
}

template <>
void release_host<Space::kDevice>(void* p) {
  cudaFreeHost(p);
}

void check_launch(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) check_cuda(err, (std::string("engine ") + what).c_str());
}

using Device = Par<Space::kDevice>;

size_t Device::scan_temp_bytes(size_t n) {
  size_t a = 0, b = 0;
  check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, a, static_cast<const uint32_t*>(nullptr),
                                           static_cast<uint32_t*>(nullptr), n),
             "engine scan");
  check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, b, static_cast<const uint64_t*>(nullptr),
                                           static_cast<uint64_t*>(nullptr), n),
             "engine scan");
  return std::max(a, b);
}

size_t Device::sort_temp_bytes(size_t n, int end_bit) {
  cub::DoubleBuffer<uint64_t> k(nullptr, nullptr);
  cub::DoubleBuffer<uint32_t> v(nullptr, nullptr);
  size_t bytes = 0;
  check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, bytes, k, v, n, 0, end_bit), "engine sort");
  return bytes;
}

size_t Device::select_temp_bytes(size_t n) {
  const cub::CountingInputIterator<uint32_t> index(0);
  size_t bytes = 0;
  check_cuda(cub::DeviceSelect::Flagged(nullptr, bytes, index, static_cast<const uint8_t*>(nullptr),
                                        static_cast<uint32_t*>(nullptr), static_cast<unsigned long long*>(nullptr), n),
             "engine select");
  return bytes;
}

void Device::exclusive_scan(const uint32_t* in, uint32_t* out, size_t n, T& temp, Stream s) {
  if (n == 0) return;
  size_t bytes = 0;
  check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n, st(s)), "engine scan");
  check_cuda(cub::DeviceScan::ExclusiveSum(temp.ensure(bytes), bytes, in, out, n, st(s)), "engine scan");
}

void Device::exclusive_scan(const uint64_t* in, uint64_t* out, size_t n, T& temp, Stream s) {
  if (n == 0) return;
  size_t bytes = 0;
  check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n, st(s)), "engine scan");
  check_cuda(cub::DeviceScan::ExclusiveSum(temp.ensure(bytes), bytes, in, out, n, st(s)), "engine scan");
}

bool Device::sort_pairs(uint64_t* keys, uint64_t* keys_alt, uint32_t* vals, uint32_t* vals_alt, size_t n, int end_bit,
                        T& temp, Stream s) {
  if (n == 0) return false;
  // The double-buffer form: the passes alternate between the caller's arrays, so the temporary
  // storage is only the histograms (the plain form would allocate a second copy of the pairs).
  cub::DoubleBuffer<uint64_t> k(keys, keys_alt);
  cub::DoubleBuffer<uint32_t> v(vals, vals_alt);
  size_t bytes = 0;
  check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, bytes, k, v, n, 0, end_bit, st(s)), "engine sort");
  check_cuda(cub::DeviceRadixSort::SortPairs(temp.ensure(bytes), bytes, k, v, n, 0, end_bit, st(s)), "engine sort");
  return k.selector == 1;
}

void Device::select_flagged(const uint8_t* flags, uint32_t* out, size_t n, unsigned long long* count, T& temp,
                            Stream s) {
  if (n == 0) {
    *count = 0;
    return;
  }
  const cub::CountingInputIterator<uint32_t> index(0);
  size_t bytes = 0;
  check_cuda(cub::DeviceSelect::Flagged(nullptr, bytes, index, flags, out, count, n, st(s)), "engine select");
  check_cuda(cub::DeviceSelect::Flagged(temp.ensure(bytes), bytes, index, flags, out, count, n, st(s)),
             "engine select");
}

void Device::copy(void* dst, const void* src, size_t bytes, Stream s) {
  if (bytes > 0) check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, st(s)), "engine copy");
}

void Device::zero(void* dst, size_t bytes, Stream s) {
  if (bytes > 0) check_cuda(cudaMemsetAsync(dst, 0, bytes, st(s)), "engine memset");
}

void Device::upload(void* dst, const void* host, size_t bytes, Stream s) {
  if (bytes > 0) check_cuda(cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice, st(s)), "engine upload");
}

void Device::download(void* host, const void* src, size_t bytes, Stream s) {
  if (bytes > 0) check_cuda(cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost, st(s)), "engine download");
  check_cuda(cudaStreamSynchronize(st(s)), "engine download");
}

void Device::download_async(void* host, const void* src, size_t bytes, Stream s) {
  if (bytes > 0) check_cuda(cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost, st(s)), "engine download");
}

void Device::sync(Stream s) { check_cuda(cudaStreamSynchronize(st(s)), "engine sync"); }

Stream Device::create_stream() {
  cudaStream_t s = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "engine cudaStreamCreate");
  return s;
}

void Device::destroy_stream(Stream s) {
  if (s != nullptr) cudaStreamDestroy(st(s));
}

size_t Device::free_bytes() {
  size_t free = 0, total = 0;
  check_cuda(cudaMemGetInfo(&free, &total), "engine cudaMemGetInfo");
  return free;
}

}  // namespace balla::engine
