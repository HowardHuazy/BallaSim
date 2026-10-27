// U9 (docs/design/phase3/frames.md §8.1): the DAVIS-like APS (§4.10; sensor/aps.h, frames/aps_frames.h). The E6
// port: EMVA 1288 photon transfer on simulated flat fields recovers K, sigma_d, DSNU and PRNU within 5 % (CT
// e6/e6_aps_model.py); the dark temporal variance is K^2 (sigma_r^2 + I_dark T) + 1/12 within 3 %; DN is bitwise a
// function of (seed, camera, pixel, t_soe), independent of order and threads; rows 1 ns apart share no counter.
// Also: the Poisson draws against the exact law, the photometry inputs (scale, gain map, veil, NaN, saturation), the
// values' bounds (FRM-VALUE), FRM-APS-SCALE, FRM-APS-EXPOSURE, the engine's cos^4 rule and the digests.
#include <doctest/doctest.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "core/rng/purpose.h"
#include "frames/aps_frames.h"
#include "sensor/aps.h"

namespace {

using namespace balla;
using frames::Roi;
using frames::RowTiming;
using frames::Shutter;
using frames::Readout;

constexpr int32_t kW = 346, kH = 260;  // DAVIS346

// One flat-field frame of the whole sensor through aps_frame (global shutter: every row at t_soe).
std::vector<double> flat(const sensor::ApsParams& a, double signal_e, int64_t t_soe, double exposure_s) {
  const std::vector<double> s(static_cast<size_t>(kW) * kH, signal_e);
  const std::vector<int64_t> rows(kH, t_soe);
  std::vector<uint16_t> dn(s.size());
  CHECK(sensor::aps_frame(a, 0, {kW, 0, 0, kW, kH}, s.data(), rows.data(), exposure_s, dn.data()) == 0);
  return {dn.begin(), dn.end()};
}

double mean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}
double var(const std::vector<double>& v) {
  const double m = mean(v);
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return s / static_cast<double>(v.size());
}
// EMVA two-image temporal variance (E6): mean((y0 - y1)^2)/2 - (mean(y0) - mean(y1))^2/2.
double temporal_var(const std::vector<double>& y0, const std::vector<double>& y1) {
  double s = 0;
  for (size_t i = 0; i < y0.size(); ++i) s += (y0[i] - y1[i]) * (y0[i] - y1[i]);
  const double d = mean(y0) - mean(y1);
  return s / static_cast<double>(y0.size()) / 2 - 0.5 * d * d;
}

}  // namespace

TEST_CASE("APS: the E6 port recovers K, sigma_d, DSNU and PRNU within 5 % (EMVA 1288 photon transfer)") {
  sensor::ApsParams a;  // E6's parameters: full well = 55 e- at 55 dB, K = (2^10 - 1) / FW
  a.full_well_e = 55.0 * std::pow(10.0, 55.0 / 20.0);
  a.gain_dn_e = 1023.0 / a.full_well_e;
  a.seed = 1;
  const double t_exp = 0.01;
  struct Level {
    double mu, var;
    std::vector<double> y0, y1;
  };
  std::vector<Level> rows;
  for (int i = 0; i < 19; ++i) {  // photo-electrons up to 90 % of the full well, two images each
    const double signal = 0.9 * i / 18.0 * a.full_well_e;
    const int64_t t = 1'000'000'000LL * i;
    Level l{0, 0, flat(a, signal, t, t_exp), flat(a, signal, t + 50'000'000, t_exp)};
    l.mu = 0.5 * (mean(l.y0) + mean(l.y1));
    l.var = temporal_var(l.y0, l.y1);
    rows.push_back(std::move(l));
  }
  const double mu_dark = rows[0].mu, var_dark = rows[0].var, sat = 1023.0 - mu_dark;
  double sx = 0, sy = 0, sxx = 0, sxy = 0, m = 0;  // least squares var = K x + offset over the linear range
  for (size_t i = 1; i < rows.size(); ++i) {
    const double x = rows[i].mu - mu_dark;
    if (!(x > 0 && x < 0.7 * sat)) continue;
    sx += x, sy += rows[i].var, sxx += x * x, sxy += x * rows[i].var, m += 1;
  }
  const double k_hat = (m * sxy - sx * sy) / (m * sxx - sx * sx), offset = (sy - k_hat * sx) / m;
  const double sigma_d_hat = std::sqrt(std::max(offset, 0.0) - 1.0 / 12) / k_hat;
  const auto average = [](const Level& l) {
    std::vector<double> v(l.y0.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = 0.5 * (l.y0[i] + l.y1[i]);
    return v;
  };
  const double s2_dark = var(average(rows[0])) - var_dark / 2;
  size_t mid = 1;
  for (size_t i = 1; i < rows.size(); ++i) {
    if (std::fabs(rows[i].mu - mu_dark - 0.5 * sat) < std::fabs(rows[mid].mu - mu_dark - 0.5 * sat)) mid = i;
  }
  const double s2_mid = var(average(rows[mid])) - rows[mid].var / 2;
  const double dsnu_hat = std::sqrt(std::max(s2_dark, 0.0)) / k_hat;
  const double prnu_hat = std::sqrt(std::max(s2_mid - s2_dark, 0.0)) / (rows[mid].mu - mu_dark);
  const double sigma_d = std::hypot(a.read_noise_e, std::sqrt(a.dark_e_s * t_exp));
  MESSAGE("K ", k_hat / a.gain_dn_e - 1, ", sigma_d ", sigma_d_hat / sigma_d - 1, ", DSNU ", dsnu_hat / a.dsnu_e - 1,
          ", PRNU ", prnu_hat / a.prnu - 1);
  CHECK(std::fabs(k_hat / a.gain_dn_e - 1) < 0.05);
  CHECK(std::fabs(sigma_d_hat / sigma_d - 1) < 0.05);
  CHECK(std::fabs(dsnu_hat / a.dsnu_e - 1) < 0.05);
  CHECK(std::fabs(prnu_hat / a.prnu - 1) < 0.05);
}

TEST_CASE("APS: the dark temporal variance is K^2 (sigma_r^2 + I_dark T) + 1/12 within 3 %") {
  const sensor::ApsParams a;  // the frames.aps defaults
  for (double t_exp : {0.01, 0.1}) {
    const auto y0 = flat(a, 0.0, 5'000'000'000LL, t_exp), y1 = flat(a, 0.0, 5'000'000'001LL, t_exp);
    const double want = a.gain_dn_e * a.gain_dn_e * (a.read_noise_e * a.read_noise_e + a.dark_e_s * t_exp) + 1.0 / 12;
    const double got = temporal_var(y0, y1);
    MESSAGE("T ", t_exp, " s: variance ", got, " DN^2, model ", want);
    CHECK(std::fabs(got / want - 1) < 0.03);
  }
}

TEST_CASE("APS: Poisson draws follow the law; the fallback after the block cap") {
  const sensor::ApsParams a;
  for (double mu : {0.3, 3.0, 9.99, 10.0, 12.0, 250.0, 1e6}) {
    const int n = 200000;
    std::vector<double> ks(n);
    double s = 0, s2 = 0;
    int fallbacks = 0;
    for (int i = 0; i < n; ++i) {
      bool fb = true;
      ks[static_cast<size_t>(i)] = sensor::aps_poisson(mu, a, 3, static_cast<uint32_t>(i), 777, &fb);
      fallbacks += fb ? 1 : 0;
      s += ks[static_cast<size_t>(i)];
    }
    CHECK(fallbacks == 0);
    const double m = s / n;
    for (double k : ks) s2 += (k - m) * (k - m);
    const double v = s2 / (n - 1);
    CHECK(std::fabs(m - mu) < 5 * std::sqrt(mu / n));
    CHECK(std::fabs(v / mu - 1) < 0.02);
    if (mu == 3.0 || mu == 12.0 || mu == 250.0) {  // frequencies against the exact pmf within 5 sigma
      for (int k = std::max(0, static_cast<int>(mu - 3 * std::sqrt(mu))); k <= mu + 3 * std::sqrt(mu); ++k) {
        const double pmf = std::exp(-mu + k * std::log(mu) - std::lgamma(k + 1.0));
        const auto obs = static_cast<double>(std::count(ks.begin(), ks.end(), static_cast<double>(k)));
        CHECK(std::fabs(obs - n * pmf) < 5 * std::sqrt(n * pmf) + 1);
      }
    }
  }
  bool fb = false;
  CHECK(sensor::aps_poisson(0.0, a, 0, 0, 0, &fb) == 0.0);
  const double k = sensor::aps_poisson(400.0, a, 0, 9, 1234, &fb, 0);
  CHECK(fb);
  CHECK(k >= 0);
  CHECK(k == std::nearbyint(k));
  CHECK(std::fabs(k - 400.0) < 120.0);
}

TEST_CASE("APS: rows 1 ns apart share no counter; the counter range is refused outside") {
  using sensor::aps_index;
  for (int64_t t : {int64_t{-(1LL << 56)}, int64_t{-1}, int64_t{0}, int64_t{123456789}, int64_t{3 * (1LL << 56) - 2}}) {
    CHECK(aps_index(t, 63) + 1 == aps_index(t + 1, 0));  // [index(t, 0), index(t, 63)] tile without overlap
    CHECK(aps_index(t, 0) < aps_index(t, 63));
  }
  CHECK_THROWS_AS(aps_index(-(1LL << 56) - 1, 0), std::invalid_argument);
  CHECK_THROWS_AS(aps_index(3 * (1LL << 56), 0), std::invalid_argument);
  CHECK_THROWS_AS(aps_index(0, 64), std::invalid_argument);
  // Temporal noise of rows 1 ns apart is uncorrelated; the fixed pattern is shared.
  const sensor::ApsParams a;
  const int n = 50000;
  double sab = 0, saa = 0, sbb = 0;
  for (int i = 0; i < n; ++i) {
    bool fb = false;
    const double ka = sensor::aps_poisson(500.0, a, 1, static_cast<uint32_t>(i), 42, &fb) - 500.0;
    const double kb = sensor::aps_poisson(500.0, a, 1, static_cast<uint32_t>(i), 43, &fb) - 500.0;
    sab += ka * kb, saa += ka * ka, sbb += kb * kb;
  }
  CHECK(std::fabs(sab / std::sqrt(saa * sbb)) < 0.02);
  const auto p1 = sensor::aps_fixed_pattern(a, 1, 77), p2 = sensor::aps_fixed_pattern(a, 1, 77);
  CHECK(p1.p == p2.p);
  CHECK(p1.d == p2.d);
}

TEST_CASE("APS frames: bitwise per seed, independent of order and threads") {
  const sensor::ApsParams a = frames::aps_params({}, 99);
  const Roi roi{10, 7, 50, 37};
  const RowTiming timing{4'000'000, 20'000, 48, Shutter::kRolling, Readout::kBottomUp};
  const size_t npix = 40 * 30;
  std::mt19937_64 gen(5);
  std::uniform_real_distribution<float> u(0.0f, 2.0f);
  std::vector<float> mono(3 * npix);
  for (float& v : mono) v = u(gen);
  mono[17] = std::nanf("");
  const std::vector<int64_t> soe{1'000'000'000, 1'033'333'333, 1'066'666'667};
  frames::ApsPhotometry ph{20.0, 0.05, {}};  // up to 2000 e- of signal
  const auto one = frames::aps_frames(a, ph, 2, 64, roi, timing, soe, mono, 1);
  const auto many = frames::aps_frames(a, ph, 2, 64, roi, timing, soe, mono, 7);
  CHECK(one.dn == many.dn);
  CHECK(one.veil_mean == many.veil_mean);
  CHECK(one.dn[17] == 0);  // no preimage
  // Reversed order, and frame 1 alone, give the same frames.
  std::vector<float> rev(mono.size());
  for (size_t k = 0; k < 3; ++k) std::copy_n(mono.begin() + (2 - k) * npix, npix, rev.begin() + k * npix);
  const std::vector<int64_t> soe_rev(soe.rbegin(), soe.rend());
  const auto back = frames::aps_frames(a, ph, 2, 64, roi, timing, soe_rev, rev, 3);
  for (size_t k = 0; k < 3; ++k) {
    CHECK(std::equal(one.dn.begin() + k * npix, one.dn.begin() + (k + 1) * npix, back.dn.begin() + (2 - k) * npix));
  }
  const auto alone = frames::aps_frames(a, ph, 2, 64, roi, timing, std::span(soe).subspan(1, 1),
                                        std::span(mono).subspan(npix, npix), 2);
  CHECK(std::equal(alone.dn.begin(), alone.dn.end(), one.dn.begin() + npix));
  // Another seed or camera changes almost every pixel.
  const auto other = frames::aps_frames(frames::aps_params({}, 100), ph, 2, 64, roi, timing, soe, mono, 4);
  const auto cam3 = frames::aps_frames(a, ph, 3, 64, roi, timing, soe, mono, 4);
  size_t same_seed = 0, same_cam = 0;
  for (size_t i = 0; i < one.dn.size(); ++i) same_seed += one.dn[i] == other.dn[i], same_cam += one.dn[i] == cam3.dn[i];
  CHECK(same_seed < one.dn.size() / 4);
  CHECK(same_cam < one.dn.size() / 4);
  // The ROI keys pixels of the full image: the full frame's ROI rows are the ROI's frame.
  std::vector<float> full(64 * 48, 0.7f);
  std::vector<float> part(npix, 0.7f);
  const auto f = frames::aps_frames(a, ph, 2, 64, {0, 0, 64, 48}, timing, std::span(soe).first(1), full, 2);
  const auto r = frames::aps_frames(a, {20.0, 0.0, {}}, 2, 64, roi, timing, std::span(soe).first(1), part, 2);
  const auto fv = frames::aps_frames(a, {20.0, 0.0, {}}, 2, 64, {0, 0, 64, 48}, timing, std::span(soe).first(1),
                                     full, 2);
  for (int32_t y = 0; y < 30; ++y) {
    CHECK(std::equal(r.dn.begin() + y * 40, r.dn.begin() + (y + 1) * 40, fv.dn.begin() + (y + 7) * 64 + 10));
  }
  CHECK(f.dn != fv.dn);  // the veil moved the signal
}

TEST_CASE("APS frames: each row's draws are keyed by its own start of exposure t_soe(k, r)") {
  // §3.8, §4.10: row r of frame k starts at frame_soe[k] + rho(r) Delta, rho(r) = r top-down and H - 1 - r
  // bottom-up; a one-row region read under a rolling shutter equals a global-shutter frame starting then.
  const sensor::ApsParams a = frames::aps_params({}, 21);
  constexpr int32_t w = 64, h = 48;
  constexpr int64_t soe = 2'000'000'000, dt = 30'000;
  std::mt19937_64 gen(9);
  std::uniform_real_distribution<float> u(0.0f, 2.0f);
  std::vector<float> row(56);
  for (float& v : row) v = u(gen);
  const auto frame = [&](Roi roi, Shutter s, Readout r, int64_t t, double veil, std::span<const float> mono) {
    const RowTiming timing{4'000'000, dt, h, s, r};
    return frames::aps_frames(a, {20.0, veil, {}}, 1, w, roi, timing, std::vector<int64_t>{t}, mono, 2).dn;
  };
  for (int32_t y : {0, 23, h - 1}) {
    const Roi roi{4, y, 60, y + 1};
    const auto global_at = [&](int64_t t) { return frame(roi, Shutter::kGlobal, Readout::kTopDown, t, 0.05, row); };
    const auto down = frame(roi, Shutter::kRolling, Readout::kTopDown, soe, 0.05, row);
    const auto up = frame(roi, Shutter::kRolling, Readout::kBottomUp, soe, 0.05, row);
    CHECK(down == global_at(soe + y * dt));
    CHECK(up == global_at(soe + (h - 1 - y) * dt));
    CHECK(down != up);  // rows read at other times draw otherwise
    CHECK(frame(roi, Shutter::kGlobal, Readout::kBottomUp, soe, 0.05, row) == global_at(soe));  // no delay
  }
  // In a region of several rows, each row is still the one-row frame at its own start (no veil: the ROI mean moves).
  std::vector<float> four;
  for (int i = 0; i < 4; ++i) four.insert(four.end(), row.begin(), row.end());
  const auto block = frame({4, 10, 60, 14}, Shutter::kRolling, Readout::kBottomUp, soe, 0.0, four);
  for (int32_t y = 10; y < 14; ++y) {
    const auto alone = frame({4, y, 60, y + 1}, Shutter::kGlobal, Readout::kTopDown, soe + (h - 1 - y) * dt, 0.0, row);
    CHECK(std::equal(alone.begin(), alone.end(), block.begin() + (y - 10) * 56));
  }
}

TEST_CASE("APS: golden DN of a region (the draw layout: kApsMap, kApsShot, kApsRead; inversion, PTRS, full well)") {
  sensor::ApsParams a;
  a.dark_e_s = 0, a.read_noise_e = 2, a.gain_dn_e = 1, a.bits = 16, a.offset_dn = 100, a.seed = 2026;
  const std::vector<double> signal{0, 0.5, 3, 9.5, 10, 250, 5000, 40000};  // 2 rows x 4 of the full image's 64
  const std::vector<int64_t> rows{1'000'000'000, 1'000'020'000};
  std::vector<uint16_t> dn(signal.size());
  CHECK(sensor::aps_frame(a, 1, {64, 5, 3, 4, 2}, signal.data(), rows.data(), 0.01, dn.data()) == 0);
  const std::vector<uint16_t> golden{88, 93, 78, 135, 106, 322, 5139, 31002};
  for (size_t i = 0; i < dn.size(); ++i) CHECK(dn[i] == golden[i]);
}

TEST_CASE("APS frames: row starts outside int64 or the counter range are refused before any arithmetic wraps") {
  const sensor::ApsParams a = frames::aps_params({}, 1);
  const std::vector<float> mono(4 * 48, 1.0f);
  const auto run = [&](int64_t row_dt, int64_t soe) {
    const RowTiming t{1'000'000, row_dt, 48, Shutter::kRolling, Readout::kBottomUp};
    return frames::aps_frames(a, {1.0, 0, {}}, 0, 4, {0, 0, 4, 48}, t, std::vector<int64_t>{soe}, mono, 1);
  };
  CHECK_THROWS_WITH_AS(run(int64_t{1} << 58, 0), doctest::Contains("exceed 2^63 ns"), std::invalid_argument);
  const int64_t top = 3 * (int64_t{1} << 56);
  CHECK_THROWS_WITH_AS(run(int64_t{1} << 52, top - (int64_t{10} << 52)), doctest::Contains("counter range"),
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(run(0, -(int64_t{1} << 56) - 1), doctest::Contains("counter range"), std::invalid_argument);
  CHECK_NOTHROW(run(int64_t{1} << 52, top - (int64_t{48} << 52)));  // the last row starts at 3 2^56 - 2^52
}

TEST_CASE("APS: the purpose.h layout (camera < 256, sides <= 65535) is refused beyond, as SensorModel does") {
  const sensor::ApsParams a;
  std::vector<double> signal(4, 100.0);
  std::vector<uint16_t> dn(4);
  const std::vector<int64_t> rows{0};
  const auto one = [&](uint32_t camera, sensor::ApsRegion g) {
    return sensor::aps_frame(a, camera, g, signal.data(), rows.data(), 0.01, dn.data());
  };
  CHECK_NOTHROW(one(255, {65535, 65531, 65534, 4, 1}));  // the last pixel y W + x = 65535^2 - 1 < 2^32
  CHECK_THROWS_WITH_AS(one(256, {64, 0, 0, 4, 1}), doctest::Contains("camera index must be < 256"),
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(one(0, {65536, 0, 0, 4, 1}), doctest::Contains("<= 65535"), std::invalid_argument);
  CHECK_THROWS_WITH_AS(one(0, {64, 0, 65535, 4, 1}), doctest::Contains("<= 65535"), std::invalid_argument);
  CHECK_THROWS_AS(one(0, {64, 2147483647, 0, 4, 1}), std::invalid_argument);  // x0 + width in int64, no wrap
  // aps_frames refuses before its workers start.
  const std::vector<float> mono(4, 1.0f);
  const RowTiming t{1'000'000, 0, 1, Shutter::kGlobal, Readout::kTopDown};
  const auto frames_of = [&](uint32_t camera, int32_t full_width, int32_t height) {
    RowTiming tt = t;
    tt.height = height;
    return frames::aps_frames(a, {1.0, 0, {}}, camera, full_width, {0, 0, 4, 1}, tt, std::vector<int64_t>{0}, mono,
                              1);
  };
  CHECK_NOTHROW(frames_of(255, 65535, 65535));
  CHECK_THROWS_WITH_AS(frames_of(256, 4, 1), doctest::Contains("(purpose.h layout)"), std::invalid_argument);
  CHECK_THROWS_WITH_AS(frames_of(0, 65536, 1), doctest::Contains("<= 65535"), std::invalid_argument);
  CHECK_THROWS_WITH_AS(frames_of(0, 4, 65536), doctest::Contains("<= 65535"), std::invalid_argument);
}

TEST_CASE("APS frames: scale, gain map, veil, saturation and clipping") {
  frames::ApsSettings s;
  s.read_noise_e = 0, s.dark_current_e_s = 0, s.prnu = 0, s.dsnu_e = 0, s.full_well_e = 1e9, s.gain_dn_per_e = 1.0;
  s.offset_dn = 0, s.bits = 16;
  const sensor::ApsParams a = frames::aps_params(s, 0);
  const int32_t w = 200, h = 100;
  const RowTiming timing{10'000'000, 0, h, Shutter::kGlobal, Readout::kTopDown};
  std::vector<float> mono(static_cast<size_t>(w) * h, 0.5f), gain(mono.size(), 1.0f);
  for (size_t i = 0; i < mono.size(); i += 2) gain[i] = 0.25f;
  mono[1] = std::nanf("");  // excluded from the veil mean, DN 0
  const double s_fA = 30.0, g = 0.2;
  const auto out = frames::aps_frames(a, {s_fA, g, gain}, 0, w, {0, 0, w, h}, timing, std::vector<int64_t>{0}, mono);
  const double e_bar = (0.5 * 0.25 * 10000 + 0.5 * 1.0 * 9999) / 19999;
  CHECK(out.veil_mean[0] == doctest::Approx(e_bar).epsilon(1e-14));
  const double scale = s_fA * 1e-15 * 0.01 / 1.602176634e-19;
  double sum_dim = 0, sum_bright = 0;
  for (size_t i = 0; i < mono.size(); ++i) (i % 2 == 0 ? sum_dim : sum_bright) += out.dn[i];
  const double want_dim = scale * (0.25 * 0.5 + g * e_bar), want_bright = scale * (0.5 + g * e_bar);
  CHECK(out.dn[1] == 0);
  CHECK(std::fabs(sum_dim / 10000 - want_dim) < 5 * std::sqrt(want_dim / 10000));
  CHECK(std::fabs(sum_bright / 9999 - want_bright) < 5 * std::sqrt(want_bright / 9999));
  // Saturation at the full well, then the ADC's range.
  s.full_well_e = 500;
  auto sat = frames::aps_frames(frames::aps_params(s, 0), {s_fA, 0, {}}, 0, w, {0, 0, w, h}, timing,
                                std::vector<int64_t>{0}, mono);
  CHECK(sat.dn[0] == 500);
  s.bits = 8;
  sat = frames::aps_frames(frames::aps_params(s, 0), {s_fA, 0, {}}, 0, w, {0, 0, w, h}, timing,
                           std::vector<int64_t>{0}, mono);
  CHECK(sat.dn[0] == 255);
  CHECK(sat.warnings.empty());
}

TEST_CASE("APS values, refusals and warnings") {
  const sensor::ApsParams d = frames::aps_params({}, 1234);
  CHECK(d.gain_dn_e == 1023.0 / 30900.0);
  CHECK(d.seed == 1234);
  CHECK(d.bits == 10);
  frames::ApsSettings s;
  s.gain_dn_per_e = 0.5;
  s.seed = 7;
  CHECK(frames::aps_params(s, 1234).gain_dn_e == 0.5);
  CHECK(frames::aps_params(s, 1234).seed == 7);
  const auto refused = [](frames::ApsSettings bad) {
    try {
      frames::aps_params(bad, 0);
    } catch (const frames::FramesError& e) {
      return e.code == "FRM-VALUE";
    }
    return false;
  };
  frames::ApsSettings b;
  b.bits = 7;
  CHECK(refused(b));
  b.bits = 17;
  CHECK(refused(b));
  b = {};
  b.full_well_e = 0;
  CHECK(refused(b));
  b = {};
  b.read_noise_e = -1;
  CHECK(refused(b));
  b = {};
  b.gain_dn_per_e = 0.0;
  CHECK(refused(b));
  b = {};
  b.seed = -1;
  CHECK(refused(b));
  b = {};
  b.prnu = std::nan("");
  CHECK(refused(b));

  const std::vector<float> mono(16, 1.0f);
  const RowTiming t{10'000'000, 0, 4, Shutter::kGlobal, Readout::kTopDown};
  const std::vector<int64_t> soe{0};
  try {
    frames::aps_frames(d, {0.0, 0, {}}, 0, 4, {0, 0, 4, 4}, t, soe, mono);
    FAIL("no FRM-APS-SCALE");
  } catch (const frames::FramesError& e) {
    CHECK(e.code == "FRM-APS-SCALE");
  }
  CHECK(frames::aps_frames(d, {1.0, 0, {}}, 0, 4, {0, 0, 4, 4}, t, soe, mono).warnings.empty());
  for (int64_t bad : {int64_t{1'500}, int64_t{5'000'000'000}}) {
    const RowTiming tb{bad, 0, 4, Shutter::kGlobal, Readout::kTopDown};
    const auto w = frames::aps_frames(d, {1.0, 0, {}}, 0, 4, {0, 0, 4, 4}, tb, soe, mono).warnings;
    REQUIRE(w.size() == 1);
    CHECK(w[0].code == "FRM-APS-EXPOSURE");
  }
  const RowTiming edge{frames::kApsMaxExposureNs, 0, 4, Shutter::kGlobal, Readout::kTopDown};
  CHECK(frames::aps_frames(d, {1.0, 0, {}}, 0, 4, {0, 0, 4, 4}, edge, soe, mono).warnings.empty());
  CHECK_THROWS_AS(frames::aps_frames(d, {1.0, 0, {}}, 0, 4, {0, 0, 5, 4}, t, soe, mono), frames::FramesError);
  CHECK_THROWS_AS(frames::aps_frames(d, {1.0, 0, {}}, 0, 4, {0, 0, 4, 4}, t, soe, std::vector<float>(15)),
                  std::invalid_argument);
}

TEST_CASE("APS photometry inputs: the engine's cos^4 rule and the vignetting switch") {
  CHECK(frames::vignetting_on(std::nullopt, "physical"));
  CHECK(frames::vignetting_on(std::nullopt, "reference"));
  CHECK_FALSE(frames::vignetting_on(std::nullopt, "relative"));
  CHECK(frames::vignetting_on(true, "relative"));
  CHECK_FALSE(frames::vignetting_on(false, "physical"));
  render::Camera cam;
  cam.width = 64, cam.height = 48, cam.fx = 50, cam.fy = 50, cam.cx = 31.5, cam.cy = 23.5;
  const auto g = frames::vignetting_gain(cam, 0.95, {0, 0, 64, 48});
  REQUIRE(g.size() == 64u * 48u);
  for (int32_t y : {0, 23, 47}) {
    for (int32_t x : {0, 31, 63}) {
      const double r2 = ((x - 31.5) * (x - 31.5) + (y - 23.5) * (y - 23.5)) / 2500.0;
      CHECK(g[static_cast<size_t>(y * 64 + x)] == doctest::Approx(1.0 / ((1 + r2) * (1 + r2))).epsilon(1e-6));
    }
  }
  const auto roi = frames::vignetting_gain(cam, 0.95, {5, 6, 20, 30});
  for (int32_t y = 0; y < 24; ++y) {
    CHECK(std::equal(roi.begin() + y * 15, roi.begin() + (y + 1) * 15, g.begin() + (y + 6) * 64 + 5));
  }
  CHECK_THROWS_AS(frames::vignetting_gain(cam, 0.95, {0, 0, 65, 48}), frames::FramesError);
}

TEST_CASE("APS gain map under optics.vignetting_law jacobian and relative illumination: the analytic lens") {
  // P2-F F13-min: the Jacobian law of the engine's gain map (Camera::init_photometry) from the frames side, against the
  // closed forms at several field angles: pinhole cos^4, radial radtan cos^4 / det J_d with det J_d = s (1 + 3 k1 r^2 +
  // 5 k2 r^4) (s = 1 + k1 r^2 + k2 r^4, r the undistorted radius), Kannala-Brandt cos t sin t / (t_d t_d').
  const auto fill = [](render::Camera& cam, render::CameraModelKind model, std::vector<double> dist) {
    cam.width = 64, cam.height = 48, cam.fx = 40, cam.fy = 40, cam.cx = 31.5, cam.cy = 23.5;
    cam.model = model, cam.dist = std::move(dist);
  };
  const auto g_at = [](const std::vector<float>& g, int32_t x, int32_t y) {
    return static_cast<double>(g[y * 64 + x]);
  };
  const Roi full{0, 0, 64, 48};
  {  // the pinhole: the law is cos^4 (to float rounding)
    render::Camera cam;
    fill(cam, render::CameraModelKind::kRadtan, {});
    const auto law = frames::vignetting_gain(cam, 0.95, full, true);
    const auto cos4 = frames::vignetting_gain(cam, 0.95, full, false);
    for (size_t i = 0; i < law.size(); ++i) CHECK(law[i] == doctest::Approx(cos4[i]).epsilon(2e-7));
  }
  {  // radial radtan along the principal row and the diagonal: undistort by Newton, then the closed form
    render::Camera cam;
    const double k1 = -0.2, k2 = 0.05;
    fill(cam, render::CameraModelKind::kRadtan, {k1, k2, 0.0, 0.0, 0.0});
    const auto law = frames::vignetting_gain(cam, 0.95, full, true);
    double worst = 0.0, spread = 0.0;
    for (const auto& [x, y] : {std::pair{32, 24}, {40, 24}, {50, 24}, {63, 24}, {63, 47}, {0, 0}, {10, 30}}) {
      const double xd = (x - cam.cx) / cam.fx, yd = (y - cam.cy) / cam.fy;
      const double rd = std::hypot(xd, yd);
      double r = rd;
      for (int i = 0; i < 60; ++i) {  // r (1 + k1 r^2 + k2 r^4) = rd
        const double r2 = r * r;
        r -= (r * (1 + k1 * r2 + k2 * r2 * r2) - rd) / (1 + 3 * k1 * r2 + 5 * k2 * r2 * r2);
      }
      const double r2 = r * r, s = 1 + k1 * r2 + k2 * r2 * r2, det = s * (1 + 3 * k1 * r2 + 5 * k2 * r2 * r2);
      const double want = 1.0 / ((1 + r2) * (1 + r2)) / det;
      worst = std::max(worst, std::abs(g_at(law, x, y) / want - 1));
      spread = std::max(spread, std::abs(det - 1));
    }
    CHECK(worst < 3e-7);   // float32 map
    CHECK(spread > 0.05);  // the lens moves the law well beyond that, at the corner
  }
  {  // Kannala-Brandt: theta_d = t (1 + k1 t^2 + k2 t^4), law = cos t sin t / (t_d t_d'),
     // t_d' = 1 + 3 k1 t^2 + 5 k2 t^4
    render::Camera cam;
    const double k1 = 0.05, k2 = -0.01;
    fill(cam, render::CameraModelKind::kKannalaBrandt, {k1, k2, 0.0, 0.0});
    cam.fx = cam.fy = 30.0;
    const auto law = frames::vignetting_gain(cam, 0.95, full, true);
    double worst = 0.0;
    for (const auto& [x, y] : {std::pair{40, 24}, {50, 24}, {63, 24}, {63, 47}, {0, 0}}) {
      const double rd = std::hypot(x - cam.cx, y - cam.cy) / cam.fx;
      double t = rd;
      for (int i = 0; i < 60; ++i) {
        const double t2 = t * t;
        t -= (t * (1 + k1 * t2 + k2 * t2 * t2) - rd) / (1 + 3 * k1 * t2 + 5 * k2 * t2 * t2);
      }
      const double t2 = t * t, td = t * (1 + k1 * t2 + k2 * t2 * t2), tdp = 1 + 3 * k1 * t2 + 5 * k2 * t2 * t2;
      worst = std::max(worst, std::abs(g_at(law, x, y) / (std::cos(t) * std::sin(t) / (td * tdp)) - 1));
    }
    CHECK(worst < 3e-7);
  }
  {  // relative illumination: G(r~) on the law's gain, r~ the full-frame radius over the farthest corner's
    render::Camera cam;
    fill(cam, render::CameraModelKind::kRadtan, {-0.1, 0.02, 0.0, 0.0, 0.0});
    const auto law = frames::vignetting_gain(cam, 0.95, full, true);
    const auto ri = frames::vignetting_gain(cam, 0.95, full, true, {-0.15, 0.03});
    const auto roi = frames::vignetting_gain(cam, 0.95, {5, 6, 20, 30}, true, {-0.15, 0.03});
    const double r2c = (63 - 31.5) * (63 - 31.5) + 23.5 * 23.5;
    for (const auto& [x, y] : {std::pair{0, 0}, {63, 47}, {40, 24}, {31, 23}}) {
      const double u = ((x - 31.5) * (x - 31.5) + (y - 23.5) * (y - 23.5)) / r2c;
      CHECK(g_at(ri, x, y) == doctest::Approx(g_at(law, x, y) * (1 - 0.15 * u + 0.03 * u * u)).epsilon(2e-7));
    }
    for (int32_t y = 0; y < 24; ++y) {
      CHECK(std::equal(roi.begin() + y * 15, roi.begin() + (y + 1) * 15, ri.begin() + (y + 6) * 64 + 5));
    }
    CHECK_THROWS_AS(frames::vignetting_gain(cam, 0.95, full, true, {-3.0, 0.0}), std::invalid_argument);
  }
}

TEST_CASE("frames_digest: two levels; the APS parameters and every base part move it") {
  const frames::DigestParts parts{"ballasim.frames/1", "b", "c", "frames.mode='reference'\n", "", ""};
  const std::string base = frames::frames_digest_base(parts);
  CHECK(base.size() == 64);
  CHECK(base == frames::frames_digest_base(parts));
  for (int i = 0; i < 6; ++i) {
    frames::DigestParts p = parts;
    std::string* f[] = {&p.format, &p.bundle_digest, &p.camera_digest, &p.values, &p.plan_digest, &p.manifest_sha256};
    *f[i] += "x";
    CHECK(frames::frames_digest_base(p) != base);
  }
  const sensor::ApsParams a = frames::aps_params({}, 3);
  sensor::ApsParams b = a;
  b.read_noise_e = 56;
  CHECK(frames::frames_digest(base, &a) == frames::frames_digest(base, &a));
  CHECK(frames::frames_digest(base, &a) != frames::frames_digest(base, &b));
  CHECK(frames::frames_digest(base, &a) != frames::frames_digest(base, nullptr));
  CHECK(frames::frames_digest(base, &a) != frames::frames_digest(base + "0", &a));
  char gain[32];
  const auto end = std::to_chars(gain, gain + sizeof(gain), 1023.0 / 30900.0).ptr;  // shortest round trip
  CHECK(frames::aps_canonical(a) == "bits=10\ndark_current_e_s=18000\ndsnu_e=20\nfull_well_e=30900\ngain_dn_per_e=" +
                                        std::string(gain, end) +
                                        "\noffset_dn=16\nprnu=0.042\nread_noise_e=55\nseed=3\n");
}
