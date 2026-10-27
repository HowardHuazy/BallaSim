// EWA texture prefilter (render/shade/ewa.h) against the exact Gaussian integral of the
// piecewise-constant texel image, plus its C2 behaviour across mip levels and texel boundaries; and the texture
// diagnostics of the scan report (render::texture_stats, scans.md §4.6, T7).
#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

#include "render/bundle.h"
#include "render/shade/ewa.h"
#include "render/texture.h"

using namespace balla;
using namespace balla::render;

namespace {

struct TestTexture {
  int32_t width, height;
  std::vector<double> level0;
  std::vector<float> texels;
  std::vector<TextureLevel> levels;
  TextureView view() const { return {levels.data(), static_cast<int32_t>(levels.size()), 1, 0}; }
};

template <class Fn>
TestTexture make_texture(int32_t w, int32_t h, Fn value) {
  TestTexture t{w, h, {}, {}, {}};
  t.level0.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
  for (int32_t j = 0; j < h; ++j) {
    for (int32_t i = 0; i < w; ++i) t.level0[static_cast<size_t>(j) * w + i] = value(i, j);
  }
  t.levels = build_mip_pyramid(t.level0, w, h, t.texels);
  return t;
}

float filter(const TextureView& tex, const float* texels, Vec2<float> uv, Sym2f cov) {
  return ewa_lookup(tex, texels, uv, cov);
}

double phi(double z) { return 0.5 * std::erfc(-z / std::numbers::sqrt2); }

// Exact integral of a texture that varies only along x (texel i spans [i, i + 1) in texel units,
// repeat wrap) against a Gaussian of variance var_x centred at x.
double exact_x(const TestTexture& t, double x, double var_x) {
  const double s = std::sqrt(var_x);
  double sum = 0.0;
  const int32_t lo = static_cast<int32_t>(std::floor(x - 12 * s)) - 1, hi = static_cast<int32_t>(std::ceil(x + 12 * s)) + 1;
  for (int32_t i = lo; i <= hi; ++i) {
    const int32_t ii = ((i % t.width) + t.width) % t.width;
    sum += t.level0[static_cast<size_t>(ii)] * (phi((i + 1 - x) / s) - phi((i - x) / s));
  }
  return sum;
}

float lookup(const TestTexture& t, double x_tex, double y_tex, double sxx, double sxy, double syy) {
  // Covariance given in level-0 texel units; ewa_lookup takes uv units.
  const double w = t.width, h = t.height;
  const Sym2f cov{static_cast<float>(sxx / (w * w)), static_cast<float>(sxy / (w * h)),
                  static_cast<float>(syy / (h * h))};
  return filter(t.view(), t.texels.data(), {static_cast<float>(x_tex / w), static_cast<float>(y_tex / h)}, cov);
}

}  // namespace

TEST_CASE("mip pyramid: box levels down to 1x1 preserve the mean, odd sizes included") {
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (auto [w, h] : {std::pair{64, 32}, std::pair{37, 23}, std::pair{1, 5}}) {
    double mean = 0.0;
    const TestTexture t = make_texture(w, h, [&](int, int) {
      const double v = u(rng);
      mean += v;
      return v;
    });
    mean /= w * h;
    REQUIRE(t.levels.back().width == 1);
    REQUIRE(t.levels.back().height == 1);
    CHECK(t.texels[t.levels.back().offset] == doctest::Approx(mean).epsilon(1e-6));
    for (size_t l = 1; l < t.levels.size(); ++l) {
      CHECK(t.levels[l].width == std::max(1, t.levels[l - 1].width / 2));
      CHECK(t.levels[l].scale_x == doctest::Approx(static_cast<double>(w) / t.levels[l].width));
    }
  }
}

TEST_CASE("EWA of a constant texture is that constant for any footprint") {
  const TestTexture t = make_texture(256, 128, [](int, int) { return 0.37; });
  std::mt19937 rng(4);
  std::uniform_real_distribution<double> pos(-300.0, 600.0), logs(-3.0, 9.0), ang(0.0, 3.14159);
  for (int i = 0; i < 2000; ++i) {
    const double a = std::exp2(logs(rng)), b = std::exp2(logs(rng)), th = ang(rng);
    const double c = std::cos(th), s = std::sin(th);
    const float v = lookup(t, pos(rng), pos(rng), a * c * c + b * s * s, (a - b) * c * s, a * s * s + b * c * c);
    CHECK(std::abs(v - 0.37f) <= 1e-6f);
  }
}

TEST_CASE("EWA matches the exact Gaussian integral of smooth textures on every mip level") {
  // Sinusoids along x with wavelength 16 sigma: the filter's fourth-order shape error is ~1e-4.
  for (double sigma : {0.7, 1.3, 2.9, 5.5, 11.0, 23.0}) {
    const double lambda = 16.0 * sigma;
    const int32_t w = static_cast<int32_t>(std::ceil(lambda * 8.0 / 64.0)) * 64;  // whole periods
    const double period = static_cast<double>(w) / std::round(w / lambda);
    const TestTexture t = make_texture(w, 64, [&](int i, int) {
      return 0.5 + 0.3 * std::cos(2.0 * std::numbers::pi * (i + 0.5) / period);
    });
    double worst = 0.0;
    for (int k = 0; k < 200; ++k) {
      const double x = 0.137 + k * period / 97.0, y = 31.3;
      // Rotated anisotropic footprint whose x-variance is sigma^2 (the texture varies along x only).
      const double sxx = sigma * sigma, syy = 4.0 * sigma * sigma, sxy = 0.5 * sigma * sigma;
      const double got = lookup(t, x, y, sxx, sxy, syy);
      worst = std::max(worst, std::abs(got - exact_x(t, x, sxx)));
    }
    INFO("sigma = " << sigma);
    CHECK(worst <= 1e-3 * 0.3);  // relative to the modulation amplitude
  }
}

TEST_CASE("EWA is C2 across mip-level switches and texel boundaries") {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const TestTexture t = make_texture(512, 512, [&](int, int) { return u(rng); });  // white noise

  // Footprint size sweep through the level switches, where the minor variance plus the texel box
  // variance 1/12 reaches 4^k (kLevelSigma = 1).
  for (double level : {1.0, 2.0, 3.0}) {
    const double s_switch = std::sqrt(std::pow(4.0, level) - 1.0 / 12.0);
    const double h = 1e-3 * s_switch;
    auto f = [&](double s) { return static_cast<double>(lookup(t, 200.3, 170.7, s * s, 0.0, s * s)); };
    const double left = (f(s_switch - h) - f(s_switch - 2 * h)) / h;
    const double right = (f(s_switch + 2 * h) - f(s_switch + h)) / h;
    const double d2_left = (f(s_switch - h) - 2 * f(s_switch - 2 * h) + f(s_switch - 3 * h)) / (h * h);
    const double d2_right = (f(s_switch + 3 * h) - 2 * f(s_switch + 2 * h) + f(s_switch + h)) / (h * h);
    INFO("switch at sigma " << s_switch << ": slopes " << left << " " << right);
    CHECK(std::abs(f(s_switch + 1e-4) - f(s_switch - 1e-4)) <= 2e-5);  // C0
    CHECK(std::abs(right - left) <= 0.05 * std::abs(left) + 3 * h * (std::abs(d2_left) + std::abs(d2_right)));  // C1
  }

  // Centre sweep across many texel boundaries at a footprint near the magnification floor: the
  // second differences stay bounded by the smooth scale (no steps or kinks at boundaries).
  {
    const double h = 1.0 / 256.0;
    std::vector<double> v;
    for (int k = 0; k < 2048; ++k) v.push_back(lookup(t, 100.0 + k * h, 50.25, 0.5, 0.1, 0.4));
    double max_d1 = 0.0, max_d2 = 0.0;
    for (size_t k = 1; k + 1 < v.size(); ++k) {
      max_d1 = std::max(max_d1, std::abs(v[k + 1] - v[k - 1]) / (2 * h));
      max_d2 = std::max(max_d2, std::abs(v[k + 1] - 2 * v[k] + v[k - 1]) / (h * h));
    }
    INFO("max |f'| = " << max_d1 << ", max |f''| = " << max_d2);
    CHECK(max_d1 < 2.0);   // texel-scale slopes of a unit-range noise texture
    CHECK(max_d2 < 10.0);  // a step (C0 break) would give ~1/h = 256 per texel boundary
  }
}

TEST_CASE("EWA clamps extreme footprints and stays bounded") {
  const TestTexture t = make_texture(128, 128, [](int i, int j) { return ((i / 4 + j / 4) % 2) ? 1.0 : 0.0; });
  // Degenerate (rank-1) and non-finite footprints filter to values inside the texel range.
  for (const Sym2f cov : {Sym2f{1e-2f, 1e-2f, 1e-2f}, Sym2f{1e30f, 0.0f, 1e30f}, Sym2f{NAN, 0.0f, 1.0f},
                          Sym2f{0.0f, 0.0f, 0.0f}}) {
    const float v = filter(t.view(), t.texels.data(), {0.3f, 0.6f}, cov);
    CHECK(std::isfinite(v));
    CHECK(v >= -1e-6f);
    CHECK(v <= 1.0f + 1e-6f);
  }
  // A footprint wider than the texture gives the texture mean.
  const float mean = filter(t.view(), t.texels.data(), {0.3f, 0.6f}, {100.0f, 0.0f, 100.0f});
  CHECK(mean == doctest::Approx(0.5f).epsilon(1e-4));
}

TEST_CASE("EWA is C2 across the footprint variance cap") {
  // A texture varying along y only and a footprint stretched along x past the cap (4 max(W, H))^2 =
  // 1024 texel^2: the lookup leaves the band average for the texture mean without a step or a kink,
  // at the cap and where the blend ends (2 x cap), and far above the cap it is the mean (no
  // under-filtering).
  const TestTexture t = make_texture(8, 8, [](int, int j) { return (j + 0.5) / 8.0; });
  // Major variance m including the texel box variance 1/12 that ewa_lookup adds.
  auto f = [&](double m) { return static_cast<double>(lookup(t, 4.0, 1.5, m - 1.0 / 12.0, 0.0, 0.0)); };
  for (double knot : {1024.0, 2048.0}) {
    const double h = 1e-3 * knot;
    const double left = (f(knot - h) - f(knot - 2 * h)) / h;
    const double right = (f(knot + 2 * h) - f(knot + h)) / h;
    const double d2_left = (f(knot - h) - 2 * f(knot - 2 * h) + f(knot - 3 * h)) / (h * h);
    const double d2_right = (f(knot + 3 * h) - 2 * f(knot + 2 * h) + f(knot + h)) / (h * h);
    INFO("knot " << knot << ": slopes " << left << " " << right);
    CHECK(std::abs(f(knot + 1e-2) - f(knot - 1e-2)) <= 1e-5);  // C0
    // C1; the last term is the fp32 rounding floor of the difference quotients.
    CHECK(std::abs(right - left) <=
          0.05 * std::abs(left) + 3 * h * (std::abs(d2_left) + std::abs(d2_right)) + 2.4e-7 / h);
  }
  CHECK(f(1e6) == doctest::Approx(0.5).epsilon(1e-6));
}

TEST_CASE("texture_stats: black, clipped and one-code steps of an 8-bit texture; binary16 and untextured materials") {
  // Grey texels of codes 0 (black), 1, 2, 3, 10, 50, 128, 254 (clipped) and 255 (clipped), weights summing to 1.
  const std::vector<uint8_t> codes{0, 1, 2, 3, 10, 50, 128, 254, 255};
  Material m;
  m.texture_width = 3;
  m.texture_height = 3;
  m.texture_channels = 3;
  for (const uint8_t c : codes) m.texture.insert(m.texture.end(), {c, c, c});
  const std::array<double, 3> w{0.52, 0.37, 0.11};
  const auto step = [](int c) {
    return std::log(srgb_to_linear(static_cast<uint8_t>(c + 1)) / srgb_to_linear(static_cast<uint8_t>(c)));
  };
  const double theta = 0.25;
  TextureStats s = texture_stats(m, w, theta);
  CHECK(s.source == TextureSource::kSrgb8);
  CHECK(s.texels == 9);
  CHECK(s.black == 1);
  CHECK(s.clipped == 2);
  // Grey steps (the weights sum to one only up to rounding, so count with the computed step): codes 1, 2, 3 exceed
  // 0.25 (0.69, 0.41, 0.29), and 10 does not (0.095), but it exceeds theta / 4.
  int64_t gt = 0, gt4 = 0;
  for (const int c : {1, 2, 3, 10, 50, 128}) {
    gt += step(c) > theta;
    gt4 += step(c) > theta / 4;
  }
  CHECK(gt == 3);
  CHECK(gt4 == 4);
  CHECK(s.step_gt_theta == gt);
  CHECK(s.step_gt_theta4 == gt4);
  // Percentiles over the 8 texels that are not black, NumPy's linear rule: p50 halfway between codes 10 and 50.
  const auto mono = [&](int c) {
    const double x = srgb_to_linear(static_cast<uint8_t>(c));
    return (w[0] * x + w[1] * x) + w[2] * x;
  };
  const double p50 = mono(50) - (mono(50) - mono(10)) * 0.5;  // _lerp's form for t >= 0.5
  CHECK(s.mono_p1_p50_p99[1] == p50);
  CHECK(s.mono_p1_p50_p99[0] == doctest::Approx(mono(1) + (mono(2) - mono(1)) * 0.07).epsilon(1e-15));
  // The base colour folds into the weights. With base colour (0, 0, 1) only blue counts: (0, 0, 7) steps by
  // ln(8 / 7) = 0.13 (above theta / 4 only), and (9, 9, 0) has T(code) = 0 < T(code + 1), an infinite step.
  m.base_color = {0.0f, 0.0f, 1.0f};
  m.texture = {0, 0, 0, 0, 0, 7, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s = texture_stats(m, w, theta);
  CHECK(s.black == 7);
  CHECK(s.clipped == 0);
  CHECK(s.step_gt_theta == 1);
  CHECK(s.step_gt_theta4 == 2);
  // T over the two texels that are not black: 0 and 0.11 lin(7); p1 lies 1 % of the way between them.
  CHECK(s.mono_p1_p50_p99[0] == (0.11 * srgb_to_linear(7)) * 0.01);
  CHECK_THROWS_AS(texture_stats(m, w, 0.0), std::invalid_argument);
  CHECK_THROWS_AS(texture_stats(m, w, std::nan("")), std::invalid_argument);
  CHECK_THROWS_AS(texture_stats(m, {-0.52, 0.37, 0.11}, 0.25), std::invalid_argument);  // schema: at_least(0)
  CHECK_THROWS_AS(texture_stats(m, {0.52, std::nan(""), 0.11}, 0.25), std::invalid_argument);
  // binary16: 1.0, 0.5, a subnormal, and +-0 (black); no clipping or steps.
  Material h;
  h.texture_width = 2;
  h.texture_height = 2;
  h.texture_channels = 3;
  h.texture_linear = {0x3C00, 0x3C00, 0x3C00, 0x3800, 0x3800, 0x3800, 0x0001, 0x0000, 0x0000, 0x8000, 0x0000, 0x8000};
  h.shading = Shading::kBaked;
  s = texture_stats(h, {1.0, 0.0, 0.0}, theta);
  CHECK(s.source == TextureSource::kBinary16);
  CHECK(s.black == 1);
  CHECK(s.clipped == 0);
  CHECK(s.step_gt_theta == 0);
  CHECK(s.mono_p1_p50_p99[1] == 0.5);
  CHECK(s.mono_p1_p50_p99[0] == std::ldexp(1.0, -24) + (0.5 - std::ldexp(1.0, -24)) * 0.02);
  // No texture: the weighted base colour.
  Material plain;
  plain.base_color = {0.5f, 0.25f, 1.0f};
  s = texture_stats(plain, w, theta);
  CHECK(s.source == TextureSource::kNone);
  CHECK(s.texels == 0);
  CHECK(s.mono_p1_p50_p99[2] == (0.52 * 0.5 + 0.37 * 0.25) + 0.11);
}
