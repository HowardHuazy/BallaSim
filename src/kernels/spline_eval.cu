#include "kernels/spline_eval.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include "kernels/device_array.h"

namespace balla::kernels {
namespace {

template <int K, class T>
__global__ void spline_evaluate_kernel(traj::SplineView<T> spline, int32_t base_segment,
                                       const int32_t* tau_ns, uint32_t n, traj::Kinematics<T>* out,
                                       uint8_t* valid) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  traj::SegmentTime st;
  const bool ok =
      traj::locate_offset(base_segment, tau_ns[i], spline.h_ns, spline.num_segments, &st);
  valid[i] = ok ? 1 : 0;
  out[i] = ok ? traj::evaluate<K>(spline, st) : traj::Kinematics<T>{};
}

}  // namespace

int cuda_device_count() {
  // Any error (no device, no driver or a broken one) is no usable device; the launches keep their checks.
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();  // clear the error, so that later checks do not see it
    return 0;
  }
  return count;
}

template <class T>
void spline_evaluate_device(int order, std::span<const traj::SplineNode<T>> nodes, int32_t h_ns,
                            T inv_h_s, int32_t base_segment, std::span<const int32_t> tau_ns,
                            std::span<traj::Kinematics<T>> out, std::span<uint8_t> valid) {
  if (order != 4 && order != 6) throw std::invalid_argument("spline order must be 4 or 6");
  if (out.size() != tau_ns.size() || valid.size() != tau_ns.size()) {
    throw std::invalid_argument("spline_evaluate_device: output sizes differ from query count");
  }
  if (cuda_device_count() == 0) throw std::runtime_error("no CUDA device available");
  const size_t n = tau_ns.size();
  if (n == 0) return;

  const DeviceArray<traj::SplineNode<T>> d_nodes(nodes);
  const DeviceArray<int32_t> d_tau(tau_ns);
  DeviceArray<traj::Kinematics<T>> d_out(n);
  DeviceArray<uint8_t> d_valid(n);

  const traj::SplineView<T> view{d_nodes.get(), static_cast<int32_t>(nodes.size()) - order + 1,
                                 h_ns, inv_h_s};
  const unsigned int block = 256;
  const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
  const uint32_t count = static_cast<uint32_t>(n);
  if (order == 4) {
    spline_evaluate_kernel<4, T><<<grid, block>>>(view, base_segment, d_tau.get(), count,
                                                  d_out.get(), d_valid.get());
  } else {
    spline_evaluate_kernel<6, T><<<grid, block>>>(view, base_segment, d_tau.get(), count,
                                                  d_out.get(), d_valid.get());
  }
  check_cuda(cudaGetLastError(), "spline_evaluate_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "spline_evaluate_kernel");
  d_out.download(out.data());
  d_valid.download(valid.data());
}

template void spline_evaluate_device<float>(int, std::span<const traj::SplineNode<float>>, int32_t,
                                            float, int32_t, std::span<const int32_t>,
                                            std::span<traj::Kinematics<float>>, std::span<uint8_t>);
template void spline_evaluate_device<double>(int, std::span<const traj::SplineNode<double>>,
                                             int32_t, double, int32_t, std::span<const int32_t>,
                                             std::span<traj::Kinematics<double>>,
                                             std::span<uint8_t>);

}  // namespace balla::kernels
