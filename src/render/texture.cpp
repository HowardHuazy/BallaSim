#include "render/texture.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "render/bundle.h"

namespace balla::render {
namespace {

// Box resampling along one axis: n_src samples -> n_dst samples, where destination sample i
// averages the source interval [i r, (i + 1) r), r = n_src / n_dst, weighting each source sample
// by its overlap. stride/count describe the other axis.
void resample_axis(const std::vector<double>& src, int32_t n_src, int32_t n_dst, int32_t other,
                   bool along_x, std::vector<double>& dst) {
  const double r = static_cast<double>(n_src) / n_dst;
  dst.assign(static_cast<size_t>(n_dst) * static_cast<size_t>(other), 0.0);
  for (int32_t i = 0; i < n_dst; ++i) {
    const double a = i * r, b = (i + 1) * r;
    const int32_t s0 = static_cast<int32_t>(std::floor(a));
    const int32_t s1 = std::min(n_src - 1, static_cast<int32_t>(std::ceil(b)) - 1);
    for (int32_t s = s0; s <= s1; ++s) {
      const double w = (std::min(b, s + 1.0) - std::max(a, static_cast<double>(s))) / r;
      if (w <= 0.0) continue;
      for (int32_t o = 0; o < other; ++o) {
        const size_t si = along_x ? static_cast<size_t>(o) * n_src + s : static_cast<size_t>(s) * other + o;
        const size_t di = along_x ? static_cast<size_t>(o) * n_dst + i : static_cast<size_t>(i) * other + o;
        dst[di] += w * src[si];
      }
    }
  }
}

// Box resampling of a width x height image to nw x nh (x first, then y: the pyramid's rule).
std::vector<double> resample_image(const std::vector<double>& src, int32_t width, int32_t height, int32_t nw,
                                   int32_t nh) {
  if (width == nw && height == nh) return src;
  std::vector<double> tmp, out;
  resample_axis(src, width, nw, height, true, tmp);
  resample_axis(tmp, height, nh, nw, false, out);
  return out;
}

double clamp01(double v) { return std::min(1.0, std::max(0.0, v)); }

// The NIR albedo of nir_form assumed, scale, affine or scalar from r = b_R R_lin.
double nir_of_red(const Material& m, double r) {
  const double p0 = m.nir_params.empty() ? 1.0 : m.nir_params[0], p1 = m.nir_params.empty() ? 0.0 : m.nir_params[1];
  switch (m.nir_form) {
    case NirForm::kScale: return p0 * r;
    case NirForm::kAffine: return clamp01(p0 + p1 * r);
    case NirForm::kScalar: return p0;
    case NirForm::kAssumed:
    case NirForm::kTexture: break;
  }
  return r;
}

}  // namespace

double srgb_to_linear(uint8_t c) {
  const double v = c / 255.0;
  return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
}

std::vector<TextureLevel> build_mono_mip_pyramid(const uint8_t* texture, int32_t width,
                                                 int32_t height, int32_t channels,
                                                 const std::array<double, 3>& weights,
                                                 std::vector<float>& texels, int32_t max_levels) {
  double lut[256];
  for (int i = 0; i < 256; ++i) lut[i] = srgb_to_linear(static_cast<uint8_t>(i));
  std::vector<double> level(static_cast<size_t>(width) * static_cast<size_t>(height));
  for (size_t i = 0; i < level.size(); ++i) {
    const uint8_t* px = texture + i * static_cast<size_t>(channels);
    level[i] = (weights[0] * lut[px[0]] + weights[1] * lut[px[1]]) + weights[2] * lut[px[2]];
  }
  return build_mip_pyramid(std::move(level), width, height, texels, max_levels);
}

double binary16_to_double(uint16_t bits) {
  const int exponent = (bits >> 10) & 0x1F;
  const int mantissa = bits & 0x3FF;
  double v = 0.0;
  if (exponent == 0x1F) {
    v = mantissa == 0 ? std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
  } else if (exponent == 0) {
    if (mantissa == 0) return 0.0;  // +0 and -0
    v = std::ldexp(static_cast<double>(mantissa), -24);  // subnormal: m 2^-24
  } else {
    v = std::ldexp(static_cast<double>(mantissa | 0x400), exponent - 25);  // (1024 + m) 2^(e - 25)
  }
  return (bits & 0x8000) != 0 ? -v : v;
}

std::vector<TextureLevel> build_linear_mip_pyramid(const uint16_t* texture, int32_t width, int32_t height,
                                                   const std::array<double, 3>& weights, std::vector<float>& texels,
                                                   int32_t max_levels) {
  std::vector<double> level(static_cast<size_t>(width) * static_cast<size_t>(height));
  for (size_t i = 0; i < level.size(); ++i) {
    const uint16_t* px = texture + 3 * i;
    level[i] = (weights[0] * binary16_to_double(px[0]) + weights[1] * binary16_to_double(px[1])) +
               weights[2] * binary16_to_double(px[2]);
  }
  return build_mip_pyramid(std::move(level), width, height, texels, max_levels);
}

std::vector<TextureLevel> build_mip_pyramid(std::vector<double> level, int32_t width, int32_t height,
                                            std::vector<float>& texels, int32_t max_levels) {
  if (max_levels < 0) throw std::invalid_argument("render: max_levels must be >= 0");
  std::vector<TextureLevel> levels;
  int32_t w = width, h = height;
  std::vector<double> tmp;
  for (;;) {
    if (texels.size() + level.size() > std::numeric_limits<uint32_t>::max()) {
      throw std::length_error("render: textures exceed 2^32 texels");
    }
    levels.push_back({static_cast<uint32_t>(texels.size()), w, h,
                      static_cast<float>(static_cast<double>(width) / w),
                      static_cast<float>(static_cast<double>(height) / h)});
    for (double v : level) texels.push_back(static_cast<float>(v));
    if ((w == 1 && h == 1) || static_cast<int32_t>(levels.size()) == max_levels) break;
    const int32_t nw = std::max(1, w / 2), nh = std::max(1, h / 2);
    resample_axis(level, w, nw, h, true, tmp);
    resample_axis(tmp, h, nh, nw, false, level);
    w = nw;
    h = nh;
  }
  return levels;
}

BandImages band_images(const Material& m, const std::array<double, 3>& wb) {
  BandImages im;
  im.vis_const = (wb[0] + wb[1]) + wb[2];
  const double b_red = static_cast<double>(m.base_color[0]);
  const bool nir_tex = m.nir_form == NirForm::kTexture;
  im.nir_const = nir_tex ? 0.0 : nir_of_red(m, b_red);
  const bool vis_tex = !m.texture.empty();
  if (!vis_tex && !nir_tex) return im;
  // The grid: the larger size per axis of the textures present.
  const int32_t vw = vis_tex ? m.texture_width : 0, vh = vis_tex ? m.texture_height : 0;
  const int32_t nw = nir_tex ? m.nir_texture_width : 0, nh = nir_tex ? m.nir_texture_height : 0;
  im.width = std::max(vw, nw);
  im.height = std::max(vh, nh);
  const size_t texels = static_cast<size_t>(im.width) * static_cast<size_t>(im.height);
  if (vis_tex) {
    double lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = srgb_to_linear(static_cast<uint8_t>(i));
    std::vector<double> vis(static_cast<size_t>(vw) * static_cast<size_t>(vh)), red;
    if (!nir_tex) red.resize(vis.size());
    for (size_t i = 0; i < vis.size(); ++i) {
      const uint8_t* px = m.texture.data() + i * static_cast<size_t>(m.texture_channels);
      vis[i] = (wb[0] * lut[px[0]] + wb[1] * lut[px[1]]) + wb[2] * lut[px[2]];
      if (!nir_tex) red[i] = nir_of_red(m, b_red * lut[px[0]]);
    }
    im.vis = resample_image(vis, vw, vh, im.width, im.height);
    if (!nir_tex) im.nir = resample_image(red, vw, vh, im.width, im.height);
  } else {
    im.vis.assign(texels, im.vis_const);
  }
  if (nir_tex) {
    std::vector<double> nir(static_cast<size_t>(nw) * static_cast<size_t>(nh));
    for (size_t i = 0; i < nir.size(); ++i) nir[i] = static_cast<double>(m.nir_texture[i]) / 255.0;
    im.nir = resample_image(nir, nw, nh, im.width, im.height);
  }
  return im;
}

std::vector<TextureLevel> build_mip_pyramid2(std::vector<double> a, std::vector<double> b, int32_t width,
                                             int32_t height, std::vector<float>& texels) {
  std::vector<float> ta, tb;
  std::vector<TextureLevel> levels = build_mip_pyramid(std::move(a), width, height, ta);
  build_mip_pyramid(std::move(b), width, height, tb);  // the same levels
  for (TextureLevel& lv : levels) {
    const size_t n = static_cast<size_t>(lv.width) * static_cast<size_t>(lv.height), from = lv.offset;
    if (texels.size() + 2 * n > std::numeric_limits<uint32_t>::max()) {
      throw std::length_error("render: textures exceed 2^32 texels");
    }
    lv.offset = static_cast<uint32_t>(texels.size());
    for (size_t i = 0; i < n; ++i) {
      texels.push_back(ta[from + i]);
      texels.push_back(tb[from + i]);
    }
  }
  return levels;
}

// --- texture diagnostics (scans.md §4.6, S2c) ------------------------------------------------------------------------

namespace {

// np.percentile(a, q) (method "linear"): the order statistics at floor and ceil of (n - 1) q / 100, blended by
// NumPy's _lerp. Reorders `a`.
double percentile(std::vector<double>& a, double q) {
  if (a.empty()) return std::numeric_limits<double>::quiet_NaN();
  const double index = (q / 100.0) * static_cast<double>(a.size() - 1);
  const auto lo = static_cast<size_t>(std::floor(index));
  const size_t hi = std::min(lo + 1, a.size() - 1);
  const double t = index - static_cast<double>(lo);
  std::nth_element(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(lo), a.end());
  const double x = a[lo];
  const double y = hi == lo ? x : *std::min_element(a.begin() + static_cast<std::ptrdiff_t>(lo) + 1, a.end());
  const double d = y - x;
  return t >= 0.5 ? y - d * (1.0 - t) : x + d * t;
}

}  // namespace

TextureStats texture_stats(const Material& m, const std::array<double, 3>& w, double theta) {
  if (!std::isfinite(theta) || theta <= 0.0) throw std::invalid_argument("texture_stats: theta must be finite and > 0");
  for (double x : w) {
    // The schema's hard bound on sensor.rgb_weights (at_least(0), param_fields.h): no run accepts a negative weight.
    if (!std::isfinite(x) || x < 0.0) {
      throw std::invalid_argument("texture_stats: the sensor RGB weights must be finite and >= 0");
    }
  }
  const std::array<double, 3> wb{w[0] * static_cast<double>(m.base_color[0]),
                                 w[1] * static_cast<double>(m.base_color[1]),
                                 w[2] * static_cast<double>(m.base_color[2])};
  TextureStats s;
  if (m.texture.empty() && m.texture_linear.empty()) {
    const double v = (wb[0] + wb[1]) + wb[2];
    s.mono_p1_p50_p99 = {v, v, v};
    return s;
  }
  s.width = m.texture_width;
  s.height = m.texture_height;
  s.texels = static_cast<int64_t>(m.texture_width) * m.texture_height;
  std::vector<double> mono;
  mono.reserve(static_cast<size_t>(s.texels));
  if (!m.texture.empty()) {
    s.source = TextureSource::kSrgb8;
    const auto c = static_cast<size_t>(m.texture_channels);
    if ((c != 3 && c != 4) || m.texture.size() != static_cast<size_t>(s.texels) * c) {
      throw std::invalid_argument("texture_stats: the texture is not H x W x 3 or 4");
    }
    double lut[257];
    for (int i = 0; i < 256; ++i) lut[i] = srgb_to_linear(static_cast<uint8_t>(i));
    lut[256] = 0.0;  // never read: codes >= 254 are clipped and have no step
    const double step4 = theta / 4.0;
    for (size_t i = 0; i < static_cast<size_t>(s.texels); ++i) {
      const uint8_t* px = m.texture.data() + i * c;
      if (px[0] == 0 && px[1] == 0 && px[2] == 0) {
        ++s.black;
        continue;
      }
      const double t0 = (wb[0] * lut[px[0]] + wb[1] * lut[px[1]]) + wb[2] * lut[px[2]];
      mono.push_back(t0);
      if (px[0] >= 254 || px[1] >= 254 || px[2] >= 254) {
        ++s.clipped;
        continue;
      }
      const double t1 = (wb[0] * lut[px[0] + 1] + wb[1] * lut[px[1] + 1]) + wb[2] * lut[px[2] + 1];
      const double step = t0 > 0.0 ? std::log(t1 / t0) : (t1 > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
      s.step_gt_theta += step > theta ? 1 : 0;
      s.step_gt_theta4 += step > step4 ? 1 : 0;
    }
  } else {
    s.source = TextureSource::kBinary16;
    if (m.texture_linear.size() != static_cast<size_t>(s.texels) * 3) {
      throw std::invalid_argument("texture_stats: texture_linear is not H x W x 3");
    }
    for (size_t i = 0; i < static_cast<size_t>(s.texels); ++i) {
      const uint16_t* px = m.texture_linear.data() + 3 * i;
      // Both zeros are black (bits 0x0000 and 0x8000).
      if (((px[0] | px[1] | px[2]) & 0x7FFF) == 0) {
        ++s.black;
        continue;
      }
      mono.push_back((wb[0] * binary16_to_double(px[0]) + wb[1] * binary16_to_double(px[1])) +
                     wb[2] * binary16_to_double(px[2]));
    }
  }
  s.mono_p1_p50_p99 = {percentile(mono, 1.0), percentile(mono, 50.0), percentile(mono, 99.0)};
  return s;
}

}  // namespace balla::render
