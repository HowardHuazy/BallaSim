#include "scene/traj/retime.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scene/blas.h"
#include "scene/npo.h"

namespace balla::scene::traj {

namespace {

// design @ coef for the (n, 2) design [1, x]: NumPy's matrix @ vector (cblas_dgemv, column-major transposed).
std::vector<double> design_times(const std::vector<double>& design, size_t n, const std::vector<double>& coef) {
  std::vector<double> out(n);
  cblas_dgemv(blas::kColMajor, blas::kTrans, 2, static_cast<int>(n), 1.0, design.data(), 2, coef.data(), 1, 0.0,
              out.data(), 1);
  return out;
}

}  // namespace

Retiming retime_stamps(std::span<const int64_t> t_raw_ns, int64_t nominal_period_ns, double huber_k, int iterations) {
  const size_t n = t_raw_ns.size();
  if (n < 2) throw std::invalid_argument("re-timing needs at least two stamps");
  Retiming out;
  out.t_raw_ns.assign(t_raw_ns.begin(), t_raw_ns.end());
  const auto period = static_cast<double>(nominal_period_ns);
  std::vector<int64_t> steps(n - 1);
  for (size_t i = 0; i + 1 < n; ++i) {
    steps[i] = static_cast<int64_t>(std::nearbyint(static_cast<double>(t_raw_ns[i + 1] - t_raw_ns[i]) / period));
    if (steps[i] < 1) throw std::invalid_argument("stamps must be strictly increasing by at least half the nominal period");
  }
  out.frame_index.assign(n, 0);
  for (size_t i = 1; i < n; ++i) out.frame_index[i] = out.frame_index[i - 1] + steps[i - 1];
  std::vector<double> y(n), x(n), design(2 * n), w(n, 1.0), coef(2, 0.0);
  for (size_t i = 0; i < n; ++i) {
    y[i] = static_cast<double>(t_raw_ns[i] - t_raw_ns[0]);
    x[i] = static_cast<double>(out.frame_index[i]);
    design[2 * i] = 1.0;
    design[2 * i + 1] = x[i];
  }
  const double rcond = std::numeric_limits<double>::epsilon() * static_cast<double>(std::max<size_t>(n, 2));
  std::vector<double> a(2 * n), b(n), r(n), dev(n), w_new(n);
  for (int it = 0; it < iterations; ++it) {
    for (size_t i = 0; i < n; ++i) {
      const double sw = std::sqrt(w[i]);
      a[2 * i] = design[2 * i] * sw;
      a[2 * i + 1] = design[2 * i + 1] * sw;
      b[i] = y[i] * sw;
    }
    coef = npo::lstsq(a, static_cast<int>(n), 2, b, rcond);
    const std::vector<double> fit = design_times(design, n, coef);
    for (size_t i = 0; i < n; ++i) r[i] = y[i] - fit[i];
    const double med = npo::median(r);
    for (size_t i = 0; i < n; ++i) dev[i] = std::abs(r[i] - med);
    const double scale = 1.4826 * npo::median(dev) + 1e-3;
    const double num = huber_k * scale;
    bool close = true;
    for (size_t i = 0; i < n; ++i) {
      w_new[i] = std::min(1.0, num / std::max(std::abs(r[i]), 1e-12));
      const bool c = (std::abs(w_new[i] - w[i]) <= 1e-12 + 0.0 * std::abs(w[i]) && std::isfinite(w[i])) ||
                     w_new[i] == w[i];
      close = close && c;
    }
    if (close) break;
    w = w_new;
  }
  const std::vector<double> fit = design_times(design, n, coef);
  out.t_ns.resize(n);
  std::vector<double> res(n);
  for (size_t i = 0; i < n; ++i) {
    out.t_ns[i] = t_raw_ns[0] + static_cast<int64_t>(std::nearbyint(fit[i]));
    res[i] = y[i] - fit[i];
  }
  out.a_ns = static_cast<double>(t_raw_ns[0]) + coef[0];
  out.b_ns = coef[1];
  out.residual_std_ns = npo::std0(res);
  out.num_gaps = std::count_if(steps.begin(), steps.end(), [](int64_t s) { return s > 1; });
  return out;
}

}  // namespace balla::scene::traj
