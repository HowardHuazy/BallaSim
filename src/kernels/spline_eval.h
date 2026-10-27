// Device evaluation of a cumulative spline at int32 offsets from a knot-aligned base (M2 A10).
// The kernel runs the same BALLA_HD code as the host, so results are bitwise equal to
// traj::evaluate<K, T> on the host (DESIGN §12.1 L1).
#pragma once

#include <cstdint>
#include <span>

#include "core/traj/spline.h"

namespace balla::kernels {

// Number of visible CUDA devices (0 without a device or a usable driver).
int cuda_device_count();

// Evaluates every query on the default CUDA device, T = float (fp32 path) or double (the fp64
// pose-table path, DESIGN §6.9). valid[i] = 0 marks a time outside the span (out[i] is then
// zero). Throws std::runtime_error on CUDA errors or if there is no device.
template <class T>
void spline_evaluate_device(int order, std::span<const traj::SplineNode<T>> nodes, int32_t h_ns,
                            T inv_h_s, int32_t base_segment, std::span<const int32_t> tau_ns,
                            std::span<traj::Kinematics<T>> out, std::span<uint8_t> valid);

extern template void spline_evaluate_device<float>(int, std::span<const traj::SplineNode<float>>,
                                                   int32_t, float, int32_t, std::span<const int32_t>,
                                                   std::span<traj::Kinematics<float>>,
                                                   std::span<uint8_t>);
extern template void spline_evaluate_device<double>(int, std::span<const traj::SplineNode<double>>,
                                                    int32_t, double, int32_t,
                                                    std::span<const int32_t>,
                                                    std::span<traj::Kinematics<double>>,
                                                    std::span<uint8_t>);

}  // namespace balla::kernels
