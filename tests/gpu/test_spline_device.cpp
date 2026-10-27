#include <doctest/doctest.h>

#include <cstring>
#include <random>
#include <vector>

#include "core/math/so3.h"
#include "core/traj/spline.h"
#include "core/traj/spline_build.h"
#include "kernels/spline_eval.h"

using namespace balla;
using namespace balla::traj;

namespace {

constexpr int32_t kH = 5'000'000;

std::vector<SplineNode<double>> random_nodes(int n) {
  std::mt19937_64 rng(21);
  std::normal_distribution<double> g;
  std::vector<Quat<double>> q;
  std::vector<Vec3<double>> p;
  Quat<double> r = quat_identity<double>();
  Vec3<double> x{1.0, -2.0, 0.5}, w{0, 0, 0};
  for (int i = 0; i < n; ++i) {
    q.push_back(r);
    p.push_back(x);
    w += 0.3 * Vec3<double>{g(rng), g(rng), g(rng)};
    r = r * so3::exp(0.005 * w);
    x += 0.01 * Vec3<double>{g(rng), g(rng), g(rng)};
  }
  return build_spline_nodes(q, p, Vec3<double>{0.5, -1.0, 0.5});
}

template <int K, class T>
void check_device_matches_host(const std::vector<SplineNode<double>>& nodes64) {
  const std::vector<SplineNode<T>> nodes = convert_nodes<T>(nodes64);
  const int32_t num_segments = static_cast<int32_t>(nodes.size()) - K + 1;
  const T inv_h = static_cast<T>(1e9 / kH);
  const int32_t base = 3;
  std::mt19937_64 rng(22);
  std::uniform_int_distribution<int32_t> tau(-4 * kH, (num_segments - base + 1) * kH);
  std::vector<int32_t> taus(50000);
  for (int32_t& t : taus) t = tau(rng);
  taus[0] = (num_segments - base) * kH;  // closed right end
  std::vector<Kinematics<T>> device(taus.size());
  std::vector<uint8_t> valid(taus.size());
  kernels::spline_evaluate_device<T>(K, nodes, kH, inv_h, base, taus, device, valid);

  const SplineView<T> view{nodes.data(), num_segments, kH, inv_h};
  const Kinematics<T> zero{};
  size_t mismatches = 0, inside = 0, outside_nonzero = 0;
  for (size_t i = 0; i < taus.size(); ++i) {
    SegmentTime st{};
    const bool ok = locate_offset(base, taus[i], kH, num_segments, &st);
    CHECK(valid[i] == (ok ? 1 : 0));
    if (!ok) {  // out of the span: the documented zero value
      outside_nonzero += std::memcmp(&zero, &device[i], sizeof(zero)) != 0;
      continue;
    }
    ++inside;
    const Kinematics<T> host = evaluate<K>(view, st);
    mismatches += std::memcmp(&host, &device[i], sizeof(host)) != 0;
  }
  CHECK(inside > taus.size() / 2);
  CHECK(inside < taus.size());  // the draws reach both sides of the span
  CHECK(mismatches == 0);
  CHECK(outside_nonzero == 0);
}

}  // namespace

TEST_CASE("device spline evaluation is bitwise equal to the host (fp32 path and fp64 pose table)") {
  const std::vector<SplineNode<double>> nodes = random_nodes(300);
  check_device_matches_host<4, float>(nodes);
  check_device_matches_host<6, float>(nodes);
  check_device_matches_host<4, double>(nodes);
  check_device_matches_host<6, double>(nodes);
}
