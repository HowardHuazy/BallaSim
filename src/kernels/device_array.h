// CUDA error checks and RAII device buffers shared by the host launchers of the kernels.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace balla::kernels {

inline void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    cudaGetLastError();  // clear a non-sticky error, so a caught failure does not resurface at the next check
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
  }
}

// Device allocation of n trivially copyable elements, freed on destruction.
template <class T>
class DeviceArray {
 public:
  explicit DeviceArray(size_t n) : n_(n) {
    if (n_ > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::runtime_error("CUDA error in cudaMalloc: size overflows size_t");
    }
    if (n_ > 0) check_cuda(cudaMalloc(&ptr_, n_ * sizeof(T)), "cudaMalloc");
  }
  explicit DeviceArray(std::span<const T> host) : DeviceArray(host.size()) { upload(host.data()); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  T* get() const { return ptr_; }
  size_t size() const { return n_; }
  void upload(const T* host) {
    if (n_ > 0) {
      check_cuda(cudaMemcpy(ptr_, host, n_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
  }
  void download(T* host) const {
    if (n_ > 0) {
      check_cuda(cudaMemcpy(host, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
    }
  }
  std::vector<T> download() const {
    std::vector<T> host(n_);
    download(host.data());
    return host;
  }

 private:
  T* ptr_ = nullptr;
  size_t n_;
};

}  // namespace balla::kernels
