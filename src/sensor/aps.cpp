// DAVIS-like APS readout (docs/design/phase3/frames.md §4.10); the model and its draws are described in aps.h.
#include "sensor/aps.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "core/pmath/explog.h"
#include "core/pmath/trig.h"
#include "core/rng/purpose.h"

namespace balla::sensor {

namespace {

constexpr double kTwoPi = 0x1.921fb54442d18p+2;
constexpr double kHalfLogTwoPi = 0.9189385332046728;  // ln(2 pi) / 2
// ln k! for k = 0 .. 9 (correctly rounded).
constexpr double kLogFactorial[10] = {0.0,
                                      0.0,
                                      0.6931471805599453,
                                      1.791759469228055,
                                      3.1780538303479458,
                                      4.787491742782046,
                                      6.579251212010101,
                                      8.525161361065415,
                                      10.60460290274525,
                                      12.801827480081469};

// Philox words -> uniforms in the open interval (0, 1): 32 bits (as the sensor maps) or 52 bits from two words, so
// that x + 1/2 is exact (53 bits would round 2^53 - 1/2 up to 2^53, i.e. u = 1).
double open_u32(uint32_t u) { return (static_cast<double>(u) + 0.5) * 0x1p-32; }
double open_u52(uint32_t hi, uint32_t lo) {
  const uint64_t x = ((static_cast<uint64_t>(hi) << 32) | lo) >> 12;
  return (static_cast<double>(x) + 0.5) * 0x1p-52;
}

// Both Box-Muller normals of words 0 and 1 of a block: r cos(2 pi u1), r sin(2 pi u1).
void box_muller(const rng::U32x4& r, double* z0, double* z1) {
  const double rad = std::sqrt(-2.0 * pmath::log(open_u32(r.v[0])));
  double s = 0.0, c = 0.0;
  pmath::sincos(kTwoPi * open_u32(r.v[1]), &s, &c);
  *z0 = rad * c;
  *z1 = rad * s;
}

// ln P(X = k) of a Poisson mean mu >= 10 at an integer k >= 0. Below 10 from the table; from 10 the saddle-point
// form (k - mu) - k log1p((k - mu)/mu) - ln(2 pi k)/2 - stirlerr(k), with stirlerr(k) = ln k! - (k ln k - k +
// ln(2 pi k)/2) from Stirling's series (error < 2e-14 at k = 10), which keeps the cancellation of
// -mu + k ln mu - ln k! out at any mu.
double log_poisson(double k, double mu) {
  if (k < 10.0) return -mu + k * pmath::log(mu) - kLogFactorial[static_cast<int>(k)];
  const double r = 1.0 / k, r2 = r * r;
  const double stirlerr = r * (1.0 / 12 - r2 * (1.0 / 360 - r2 * (1.0 / 1260 - r2 * (1.0 / 1680 - r2 / 1188))));
  const double d = k - mu;
  return d - k * pmath::log1p(d / mu) - (kHalfLogTwoPi + 0.5 * pmath::log(k)) - stirlerr;
}

}  // namespace

void ApsParams::validate() const {
  const auto check = [](bool ok, const char* field, const std::string& rule) {
    if (!ok) throw std::invalid_argument(std::string("APS: ") + field + " must be " + rule);
  };
  check(bits >= 8 && bits <= 16, "bits", "in [8, 16]");
  check(std::isfinite(full_well_e) && full_well_e > 0, "full_well_e", "finite and > 0");
  check(std::isfinite(read_noise_e) && read_noise_e >= 0, "read_noise_e", "finite and >= 0");
  check(std::isfinite(dark_e_s) && dark_e_s >= 0, "dark_e_s", "finite and >= 0");
  check(std::isfinite(prnu) && prnu >= 0, "prnu", "finite and >= 0");
  check(std::isfinite(dsnu_e) && dsnu_e >= 0, "dsnu_e", "finite and >= 0");
  check(std::isfinite(gain_dn_e) && gain_dn_e > 0, "gain_dn_e", "finite and > 0");
  check(std::isfinite(offset_dn) && offset_dn >= 0, "offset_dn", "finite and >= 0");
}

void check_layout(uint32_t camera, int64_t full_width, int64_t full_rows) {
  if (camera > 255) throw std::invalid_argument("APS: camera index must be < 256 (purpose.h layout)");
  if (full_width > kApsMaxSide || full_rows > kApsMaxSide) {
    throw std::invalid_argument("APS: the full image's width and rows must be <= 65535 (purpose.h layout: pixel "
                                "y W + x < 2^32), got width " + std::to_string(full_width) + ", rows " +
                                std::to_string(full_rows));
  }
}

uint64_t aps_index(int64_t t_soe_ns, uint32_t j) {
  if (j >= static_cast<uint32_t>(kApsBlocks)) throw std::invalid_argument("APS: draw index j must be < 64");
  if (t_soe_ns < -kApsTimeBias || t_soe_ns >= 3 * kApsTimeBias) {
    throw std::invalid_argument("APS: a row's start of exposure " + std::to_string(t_soe_ns) +
                                " ns is outside [-2^56, 3 2^56) ns, the APS counter range");
  }
  return static_cast<uint64_t>(t_soe_ns + kApsTimeBias) * static_cast<uint64_t>(kApsBlocks) + j;
}

ApsFixedPattern aps_fixed_pattern(const ApsParams& a, uint32_t camera, uint32_t pixel) {
  double zp = 0.0, zd = 0.0, unused = 0.0;
  box_muller(rng::draw(a.seed, rng::Purpose::kApsMap, camera, pixel, 0), &zp, &unused);
  box_muller(rng::draw(a.seed, rng::Purpose::kApsMap, camera, pixel, 1), &zd, &unused);
  return {a.prnu * zp, a.dsnu_e * zd};
}

double aps_poisson(double mu, const ApsParams& a, uint32_t camera, uint32_t pixel, int64_t t_soe_ns, bool* fallback,
                   int32_t max_blocks) {
  *fallback = false;
  if (!(mu > 0.0)) return 0.0;
  if (mu < 10.0) {
    // Inversion: the first k with u <= F(k). The summed F can stall a few ulps below 1 while u < 1 lies above it;
    // the loop then stops at the first term that no longer moves the sum. Up to the mode the terms grow, so each is
    // at least F/k >= F/10 and cannot be absorbed: the stop only happens in the far tail (u within a few ulps of 1).
    const rng::U32x4 r = rng::draw(a.seed, rng::Purpose::kApsShot, camera, pixel, aps_index(t_soe_ns, 0));
    const double u = open_u52(r.v[0], r.v[1]);
    double p = pmath::exp(-mu), f = p, k = 0.0;
    while (u > f) {
      k += 1.0;
      p *= mu / k;
      const double next = f + p;
      if (next == f) break;
      f = next;
    }
    return k;
  }
  // PTRS (Hoermann 1993, algorithm PTRS), constants for mu >= 10.
  const double b = 0.931 + 2.53 * std::sqrt(mu);
  const double ac = -0.059 + 0.02483 * b;
  const double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
  const double v_r = 0.9277 - 3.6224 / (b - 2.0);
  for (int32_t j = 0; j < max_blocks && j < kApsBlocks; ++j) {
    const rng::U32x4 r =
        rng::draw(a.seed, rng::Purpose::kApsShot, camera, pixel, aps_index(t_soe_ns, static_cast<uint32_t>(j)));
    const double u = open_u52(r.v[0], r.v[1]) - 0.5, v = open_u52(r.v[2], r.v[3]);
    const double us = 0.5 - std::fabs(u);
    const double k = std::floor((2.0 * ac / us + b) * u + mu + 0.43);
    if (us >= 0.07 && v <= v_r) return k;
    if (k < 0.0 || (us < 0.013 && v > us)) continue;
    if (pmath::log(v) + pmath::log(inv_alpha) - pmath::log(ac / (us * us) + b) <= log_poisson(k, mu)) return k;
  }
  *fallback = true;
  double z0 = 0.0, z1 = 0.0;
  box_muller(rng::draw(a.seed, rng::Purpose::kApsRead, camera, pixel, aps_index(t_soe_ns, 0)), &z0, &z1);
  return std::fmax(0.0, std::nearbyint(mu + std::sqrt(mu) * z1));
}

int64_t aps_frame(const ApsParams& a, uint32_t camera, const ApsRegion& g, const double* mean_signal_e,
                  const int64_t* row_soe_ns, double exposure_s, uint16_t* dn) {
  a.validate();
  check_layout(camera, g.full_width, int64_t{g.y0} + g.height);
  if (g.width < 0 || g.height < 0 || g.x0 < 0 || g.y0 < 0 || int64_t{g.x0} + g.width > g.full_width) {
    throw std::invalid_argument("APS: the region lies outside the full image");
  }
  if (!(std::isfinite(exposure_s) && exposure_s >= 0)) throw std::invalid_argument("APS: exposure_s must be >= 0");
  const double dark = a.dark_e_s * exposure_s;
  const double dn_max = std::ldexp(1.0, a.bits) - 1.0;
  int64_t fallbacks = 0;
  for (int32_t i = 0; i < g.height; ++i) {
    const int64_t t_soe = row_soe_ns[i];
    aps_index(t_soe, 0);  // the row's counters exist
    const auto full_row = static_cast<uint64_t>(g.y0 + i) * static_cast<uint64_t>(g.full_width);
    for (int32_t x = 0; x < g.width; ++x) {
      const size_t at = static_cast<size_t>(i) * static_cast<size_t>(g.width) + static_cast<size_t>(x);
      const double signal = mean_signal_e[at];
      if (std::isnan(signal)) {  // no preimage in the lens domain
        dn[at] = 0;
        continue;
      }
      const auto pixel = static_cast<uint32_t>(full_row + static_cast<uint64_t>(g.x0 + x));
      const ApsFixedPattern fp = aps_fixed_pattern(a, camera, pixel);
      const double mu = std::fmax(0.0, (1.0 + fp.p) * signal) + dark;
      bool fell_back = false;
      const double shot = std::isfinite(mu) ? aps_poisson(mu, a, camera, pixel, t_soe, &fell_back) : mu;
      fallbacks += fell_back ? 1 : 0;
      double z_read = 0.0, z_spare = 0.0;
      box_muller(rng::draw(a.seed, rng::Purpose::kApsRead, camera, pixel, aps_index(t_soe, 0)), &z_read, &z_spare);
      const double n = std::fmin(shot + fp.d, a.full_well_e) + a.read_noise_e * z_read;
      const double y = std::nearbyint(a.gain_dn_e * n + a.offset_dn);
      dn[at] = static_cast<uint16_t>(std::fmin(std::fmax(y, 0.0), dn_max));
    }
  }
  return fallbacks;
}

}  // namespace balla::sensor
