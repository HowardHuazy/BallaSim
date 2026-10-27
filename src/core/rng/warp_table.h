// Host-side construction of the warp table (see warp.h), in fp64 from the exact CDF of the
// truncated box(w) ⊛ Gaussian(sigma) density, and of its search guide.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/rng/warp.h"

namespace balla::rng {

class PixelResponse {
 public:
  // w: aperture width (px), sigma: optical blur (px), truncation at w/2 + n_sigma * sigma.
  PixelResponse(double w, double sigma, double n_sigma)
      : w_(w), sigma_(sigma), x_max_(0.5 * w + n_sigma * sigma) {
    if (!(w >= 0.0) || !(sigma >= 0.0) || !(n_sigma > 0.0) || (w == 0.0 && sigma == 0.0)) {
      throw std::invalid_argument("PixelResponse needs w >= 0, sigma >= 0, n_sigma > 0, w + sigma > 0");
    }
    mass_ = untruncated_cdf(x_max_) - untruncated_cdf(-x_max_);
  }

  double x_max() const { return x_max_; }

  // CDF and density of the truncated distribution on [-x_max, x_max].
  double cdf(double x) const {
    return (untruncated_cdf(x) - untruncated_cdf(-x_max_)) / mass_;
  }
  double pdf(double x) const { return untruncated_pdf(x) / mass_; }

 private:
  static double phi(double z) { return std::exp(-0.5 * z * z) / std::sqrt(2.0 * std::numbers::pi); }
  static double Phi(double z) { return 0.5 * std::erfc(-z / std::numbers::sqrt2); }

  // With G1(y) = int Phi(y/sigma) dy = y Phi(y/sigma) + sigma phi(y/sigma), the CDF of
  // box(w) ⊛ N(0, sigma^2) is (G1(x + w/2) - G1(x - w/2)) / w.
  double untruncated_cdf(double x) const {
    if (sigma_ == 0.0) return std::clamp((x + 0.5 * w_) / w_, 0.0, 1.0);
    if (w_ == 0.0) return Phi(x / sigma_);
    return (G1(x + 0.5 * w_) - G1(x - 0.5 * w_)) / w_;
  }
  double untruncated_pdf(double x) const {
    if (sigma_ == 0.0) return std::abs(x) <= 0.5 * w_ ? 1.0 / w_ : 0.0;
    if (w_ == 0.0) return phi(x / sigma_) / sigma_;
    return (Phi((x + 0.5 * w_) / sigma_) - Phi((x - 0.5 * w_) / sigma_)) / w_;
  }
  double G1(double y) const {
    const double z = y / sigma_;
    return y * Phi(z) + sigma_ * phi(z);
  }

  double w_, sigma_, x_max_, mass_;
};

// Cells of a warp table: the search guide holds start cells as uint16 (warp.h).
inline constexpr int32_t kMinWarpCells = 2, kMaxWarpCells = 65535;

template <class T>
class WarpTable {
 public:
  WarpTable(const PixelResponse& h, int32_t cells) : n_(cells), x_max_(h.x_max()) {
    if (cells < kMinWarpCells || cells > kMaxWarpCells) {
      throw std::invalid_argument("WarpTable needs " + std::to_string(kMinWarpCells) + " to " +
                                  std::to_string(kMaxWarpCells) + " cells");
    }
    const double dx = h.x_max() / cells;
    cdf_.resize(static_cast<size_t>(cells) + 1);
    slope_.resize(static_cast<size_t>(cells) + 1);
    for (int32_t i = 0; i <= cells; ++i) {
      const double x = (i - cells) * dx;
      cdf_[static_cast<size_t>(i)] = static_cast<T>(i == cells ? 0.5 : (i == 0 ? 0.0 : h.cdf(x)));
      slope_[static_cast<size_t>(i)] = static_cast<T>(1.0 / h.pdf(x));
    }
    dx_ = static_cast<T>(dx);
    // From the stored (type T) CDF values, exactly as warp_inverse_cdf compares them.
    guide_.resize(kWarpGuideCells);
    int32_t lo = 0;
    for (int32_t g = 0; g < kWarpGuideCells; ++g) {
      const T p = static_cast<T>(g) / static_cast<T>(2 * kWarpGuideCells);
      while (lo + 1 < cells && cdf_[static_cast<size_t>(lo) + 1] <= p) ++lo;
      guide_[static_cast<size_t>(g)] = static_cast<uint16_t>(lo);
    }
  }

  WarpTableView<T> view() const {
    return {cdf_.data(), slope_.data(), guide_.data(), n_, dx_};
  }
  const std::vector<T>& cdf() const { return cdf_; }
  const std::vector<T>& slope() const { return slope_; }
  const std::vector<uint16_t>& guide() const { return guide_; }
  int32_t cells() const { return n_; }
  double x_max() const { return x_max_; }

 private:
  int32_t n_;
  double x_max_;
  T dx_{};
  std::vector<T> cdf_, slope_;
  std::vector<uint16_t> guide_;
};

}  // namespace balla::rng
