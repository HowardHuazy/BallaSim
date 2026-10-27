#include "parity.h"

#include <cuda_runtime.h>

#include <utility>

#include "kernels/device_array.h"

namespace balla::gpu_test {
namespace {

using kernels::check_cuda;
using kernels::DeviceArray;

struct NoContext {};

template <class Case>
__device__ auto evaluate_with(const NoContext&, const Case& c) {
  return evaluate(c);
}
template <class Context, class Case>
__device__ auto evaluate_with(const Context& ctx, const Case& c) {
  return evaluate(ctx, c);
}

template <class Context, class Case, class Result>
__global__ void evaluate_kernel(Context ctx, const Case* cases, Result* out, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = evaluate_with(ctx, cases[i]);
}

// Runs evaluate(ctx, case) for every case; ctx is passed by value and its pointers must address
// device memory.
template <class Result, class Context, class Case>
std::vector<Result> run(const Context& ctx, std::span<const Case> cases) {
  if (cases.empty()) return {};
  const DeviceArray<Case> d_cases(cases);
  DeviceArray<Result> d_out(cases.size());
  const unsigned int block = 128;
  const unsigned int grid = static_cast<unsigned int>((cases.size() + block - 1) / block);
  evaluate_kernel<<<grid, block>>>(ctx, d_cases.get(), d_out.get(),
                                   static_cast<uint32_t>(cases.size()));
  check_cuda(cudaGetLastError(), "evaluate_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "evaluate_kernel");
  return d_out.download();
}

}  // namespace

std::vector<RngResult> evaluate_on_device(std::span<const RngCase> cases) {
  return run<RngResult>(NoContext{}, cases);
}

std::vector<Vec2<float>> evaluate_on_device(const rng::WarpTable<float>& table,
                                            std::span<const OffsetCase> cases) {
  const DeviceArray<float> cdf{std::span<const float>(table.cdf())};
  const DeviceArray<float> slope{std::span<const float>(table.slope())};
  const DeviceArray<uint16_t> guide{std::span<const uint16_t>(table.guide())};
  rng::WarpTableView<float> view = table.view();
  view.cdf = cdf.get();
  view.slope = slope.get();
  view.guide = guide.get();
  return run<Vec2<float>>(view, cases);
}

template <class T>
std::vector<TrigResult<T>> evaluate_on_device(std::span<const TrigCase<T>> cases) {
  return run<TrigResult<T>>(NoContext{}, cases);
}

template <class T>
std::vector<ExpLogResult<T>> evaluate_on_device(std::span<const ExpLogCase<T>> cases) {
  return run<ExpLogResult<T>>(NoContext{}, cases);
}

template <class T>
std::vector<CameraResult<T>> evaluate_on_device(const camera::RadtanCamera<T>& cam,
                                                std::span<const CameraCase<T>> cases) {
  return run<CameraResult<T>>(cam, cases);
}

template <class T>
std::vector<CameraResult<T>> evaluate_on_device(const camera::KannalaBrandtCamera<T>& cam,
                                                std::span<const CameraCase<T>> cases) {
  return run<CameraResult<T>>(cam, cases);
}

std::vector<Vec3<float>> evaluate_on_device(const camera::RayLut<float>& lut,
                                            std::span<const Vec2<float>> pixels) {
  const DeviceArray<Vec3<float>> rays{std::span<const Vec3<float>>(lut.rays())};
  camera::RayLutView<float> view = lut.view();
  view.rays = rays.get();
  return run<Vec3<float>>(view, pixels);
}

template <class T>
std::vector<BsdfResult<T>> evaluate_on_device(const render::bsdf::BsdfTables& host_tables,
                                              std::span<const BsdfCase<T>> cases) {
  const auto upload = [](const float* table) {
    return DeviceArray<float>{std::span<const float>(table, render::kTableSize)};
  };
  const DeviceArray<float> a = upload(host_tables.spec_a), b = upload(host_tables.spec_b);
  const DeviceArray<float> m00 = upload(host_tables.ltc_m00), m02 = upload(host_tables.ltc_m02);
  const DeviceArray<float> m20 = upload(host_tables.ltc_m20), m22 = upload(host_tables.ltc_m22);
  const render::bsdf::BsdfTables device_tables{a.get(), b.get(), m00.get(), m02.get(), m20.get(), m22.get()};
  return run<BsdfResult<T>>(device_tables, cases);
}

template std::vector<TrigResult<float>> evaluate_on_device(std::span<const TrigCase<float>>);
template std::vector<TrigResult<double>> evaluate_on_device(std::span<const TrigCase<double>>);
template std::vector<ExpLogResult<float>> evaluate_on_device(std::span<const ExpLogCase<float>>);
template std::vector<ExpLogResult<double>> evaluate_on_device(std::span<const ExpLogCase<double>>);
template std::vector<CameraResult<float>> evaluate_on_device(const camera::RadtanCamera<float>&,
                                                             std::span<const CameraCase<float>>);
template std::vector<CameraResult<double>> evaluate_on_device(const camera::RadtanCamera<double>&,
                                                              std::span<const CameraCase<double>>);
template std::vector<CameraResult<float>> evaluate_on_device(
    const camera::KannalaBrandtCamera<float>&, std::span<const CameraCase<float>>);
template std::vector<CameraResult<double>> evaluate_on_device(
    const camera::KannalaBrandtCamera<double>&, std::span<const CameraCase<double>>);
template std::vector<BsdfResult<float>> evaluate_on_device(const render::bsdf::BsdfTables&,
                                                           std::span<const BsdfCase<float>>);
template std::vector<BsdfResult<double>> evaluate_on_device(const render::bsdf::BsdfTables&,
                                                            std::span<const BsdfCase<double>>);

}  // namespace balla::gpu_test
