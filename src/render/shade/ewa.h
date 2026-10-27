// Single-Gaussian EWA prefilter of a mono texture (DESIGN §7.3, Phase 1 scope; S2 MAT-09-12). The
// cluster shading (render/shade/shade.h) calls it once per sub-sample, at the sub-sample's own UV
// with that sub-sample's share of the pixel footprint.
//
// The filter is a C2 function of the footprint centre and covariance, so a sub-sample's texture
// value is C2 in time while it stays on one triangle (DESIGN §7.10):
//   - texels are point-fetched and weighted in fp32 software (hardware filtering is banned: its
//     8-bit fractional weights step a slowly moving texture, S2 MAT-11);
//   - the Gaussian weight is truncated at q = d^T S^-1 d <= 4 with the C2 taper (1 - q/4)^3, whose
//     value and first two derivatives vanish on the boundary; the taper narrows the kernel, so the
//     Gaussian parameter is widened by 1/kTaperVariance to keep the requested variance;
//   - the mip level comes from the minor axis and the two bracketing levels are blended with the
//     C2 smootherstep of frac(level) (a linear blend would put derivative kinks into the signal);
//   - each level is a box average of level 0, which adds the box variance (s^2 - 1)/12 (s = level-0
//     texels per level texel, per axis); that variance is subtracted from the kernel, so every level
//     estimates the same Gaussian-filtered texture and the blend changes only fourth-order terms;
//   - texels are boxes: the footprint gets the box variance 1/12 texel^2 added, so the filter
//     estimates the integral of the piecewise-constant texture image (what the test oracle and a
//     nearest-texel magnification show), not of a texel-centre point sampling;
//   - anisotropy is clamped and the kernel floored smoothly (C2 polynomial blends instead of max);
//   - footprints past the variance cap blend into the texture mean with the C2 smootherstep.
//
// P2-F F11a (plan p2f §4.9): a TextureView of two channels (TextureView::channels, the visible and the near-infrared
// albedo interleaved per texel) is looked up by ewa_lookup_n<2>, which returns both from one weight set: the weights, the
// level and the blends do not depend on the channel, so each channel is what the mono lookup of its own image gives
// (to the rounding of the shared weights' sum). The one-channel lookup is the same code with kCh = 1.
//
// A tile atlas (TextureView::tile_pitch P > 0; scans.md §4.5.6, S5a) is a grid of P x P cells, one per face, each
// holding its face's tile and gutters. A lookup stays in the cell its uv lies in: texel indices clamp to the cell's
// edges instead of wrapping, the pyramid stops at one texel per cell, and the variance cap and the mean are the
// cell's (sigma_cap 4P, the cell's top-level texel). Within a cell the value stays C2 in uv; the cells of a face's
// neighbours are never read. Textures without tiles take the code path above unchanged. The code is one template over
// the channels (kCh) and the tiling (kTile); a two-channel texture is never tiled (SceneData refuses the combination:
// a tile atlas is a scan's, baked and mono), so ewa_lookup_n<2> instantiates the untiled form only.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/pmath/explog.h"
#include "core/pmath/smootherstep.h"
#include "render/scene_view.h"

namespace balla::render {

struct Sym2f {
  float xx, xy, yy;
};

namespace ewa {

// Per-axis variance of the 2-D radial kernel exp(-q/2) (1 - q/4)^3 relative to the Gaussian's:
// int_0^4 x e^(-x/2) (1 - x/4)^3 dx / (2 int_0^4 e^(-x/2) (1 - x/4)^3 dx), with x = r^2 / s^2.
inline constexpr float kTaperVariance = 0.310123246f;
// Target kernel sigma at the finer of the two blended levels, in level texels. At sigma >= 1 the
// discrete kernel's centroid ripple is <= 2e-4 texel; at the coarser level sigma lies in [0.5, 1)
// and its blend weight vanishes as sigma -> 0.5 (ripple <= 3e-3 texel there).
inline constexpr float kLevelSigma = 1.0f;
// log2(e): the Gaussian weights are e^(-q/2) = 2^(-q/2 log2(e)), evaluated by the cheaper exp2.
inline constexpr float kLog2e = 1.44269504088896341f;
// Anisotropy is clamped at sigma_major / sigma_minor ~ 16 (eigenvalue ratio 1/256), BallaSim's own choice;
// pbrt-v4's image textures default to maxanisotropy 8 (src/pbrt/textures.cpp @b4ce968).
inline constexpr float kMinEigenRatio = 1.0f / 256.0f;
// Floor of the kernel variance at the evaluated level (texel^2): keeps at least one texel centre
// inside the support for any centre (magnification).
inline constexpr float kMinLevelVariance = 0.25f;

struct Eigen2 {
  float max, min;
};

BALLA_HD inline Eigen2 eigenvalues(Sym2f s) {
  const float m = 0.5f * (s.xx + s.yy);
  const float d = 0.5f * (s.xx - s.yy);
  const float r = sqrt(d * d + s.xy * s.xy);
  return {m + r, m - r};
}

BALLA_HD inline int32_t wrap(int32_t i, int32_t n) {
  const int32_t r = i % n;
  return r < 0 ? r + n : r;
}

// C2 lower floor: x for x >= f, f (x/f + (1 - x/f)^3) below; >= 0.615 f; equal to f at x = 0.
BALLA_HD inline float smooth_floor(float x, float f) {
  if (x >= f) return x;
  const float a = 1.0f - x / f;
  return x + f * (a * a * a);
}

// The tile cell holding uv (TextureView::tile_pitch P > 0): its index floor(uv cells) per axis (cells = W / P, the same
// integer at every level of a tile pyramid, so every level picks the same cell), in the unwrapped indices of the
// kernel's centre, and wrapped into the texture, where it is read (cells repeat like the texels of an untiled
// texture). At level l the cell's first texel has index x (P >> l), and uv W_l lies inside the cell, since
// fl(uv W_l) = fl(uv cells) (P >> l) for a power-of-two P.
struct TileCell {
  int32_t x, y;    // unwrapped cell index
  int32_t wx, wy;  // wrapped into [0, cells)
};
BALLA_HD inline TileCell tile_cell(const TextureLevel& l0, int32_t pitch, Vec2<float> uv) {
  const int32_t nx = l0.width / pitch, ny = l0.height / pitch;
  const int32_t x = static_cast<int32_t>(floor(uv.x * static_cast<float>(nx)));
  const int32_t y = static_cast<int32_t>(floor(uv.y * static_cast<float>(ny)));
  return {x, y, wrap(x, nx), wrap(y, ny)};
}

// Kernel value at one level: centre (u, v) in uv units, covariance in level-0 texel units. The
// footprint's texels are summed row by row in increasing column order on one thread, so the CPU and
// the GPU add in the same order (DESIGN §7.9). kCh channels are interleaved per texel (out[0..kCh)). kTile: the texel
// indices clamp to the tile cell of side `pitch` (this level's) that holds uv (tile_cell); a cell of one texel is that
// texel (the normalized weights would only round).
template <int32_t kCh, bool kTile>
BALLA_HD inline void evaluate_level(const TextureLevel& lv, const float* texels, Vec2<float> uv, Sym2f cov0,
                                    float* out, TileCell cell, int32_t pitch) {
  // kTile: the cell's first texel in the centre's indices (ox, oy) and in the texture's (tx, ty).
  const int32_t ox = cell.x * pitch, oy = cell.y * pitch, tx = cell.wx * pitch, ty = cell.wy * pitch;
  if (kTile && pitch == 1) {
    const float* texel = texels + lv.offset +
                         (static_cast<uint64_t>(ty) * static_cast<uint64_t>(lv.width) + static_cast<uint64_t>(tx)) * kCh;
    for (int32_t ch = 0; ch < kCh; ++ch) out[ch] = texel[ch];
    return;
  }
  // Remove the variance the level's box averaging already applied, then go to level units.
  const float sx = lv.scale_x, sy = lv.scale_y;
  Sym2f c{(cov0.xx - (sx * sx - 1.0f) * (1.0f / 12.0f)) / (sx * sx), cov0.xy / (sx * sy),
          (cov0.yy - (sy * sy - 1.0f) * (1.0f / 12.0f)) / (sy * sy)};
  const float lmin = eigenvalues(c).min;
  const float add = smooth_floor(lmin, kMinLevelVariance) - lmin;
  c.xx += add;
  c.yy += add;
  // Gaussian parameter of the tapered kernel, and its inverse (quadratic form of q).
  const float gxx = c.xx * (1.0f / kTaperVariance), gxy = c.xy * (1.0f / kTaperVariance),
              gyy = c.yy * (1.0f / kTaperVariance);
  // Two divisions per level and none per row: IEEE division is a multi-instruction sequence on the
  // GPU, and the renderer evaluates a lookup per textured sub-sample.
  const float inv_det = 1.0f / (gxx * gyy - gxy * gxy);
  const float axx = gyy * inv_det, axy = -gxy * inv_det, ayy = gxx * inv_det;
  const float inv_axx = 1.0f / axx;
  const float cx = uv.x * static_cast<float>(lv.width) - 0.5f;
  const float cy = uv.y * static_cast<float>(lv.height) - 0.5f;
  // Rows inside the ellipse q <= 4 satisfy |dy| <= 2 sqrt(g_yy); each row's span solves q(dx) = 4.
  const float ry = 2.0f * sqrt(gyy);
  const int32_t j0 = static_cast<int32_t>(floor(cy - ry)) + 1;
  const int32_t j1 = static_cast<int32_t>(floor(cy + ry));
  float sum[kCh];
  for (int32_t ch = 0; ch < kCh; ++ch) sum[ch] = 0.0f;
  float sum_w = 0.0f;
  for (int32_t j = j0; j <= j1; ++j) {
    const float dy = static_cast<float>(j) - cy;
    const float b = axy * dy;
    const float disc = b * b - axx * (ayy * dy * dy - 4.0f);
    if (!(disc > 0.0f)) continue;
    const float s = sqrt(disc);
    const int32_t i0 = static_cast<int32_t>(floor(cx + (-b - s) * inv_axx)) + 1;
    const int32_t i1 = static_cast<int32_t>(floor(cx + (-b + s) * inv_axx));
    if (i1 < i0) continue;
    const int32_t jj = kTile ? ty + clamp(j - oy, 0, pitch - 1) : wrap(j, lv.height);
    const float* row = texels + lv.offset + static_cast<uint64_t>(jj) * static_cast<uint64_t>(lv.width) * kCh;
    const float qy = ayy * dy * dy;
    [[maybe_unused]] const bool inside = i0 >= 0 && i1 < lv.width;  // else repeat wrap (the glTF default sampler)
    for (int32_t i = i0; i <= i1; ++i) {
      const float dx = static_cast<float>(i) - cx;
      const float q = (axx * dx * dx + 2.0f * b * dx) + qy;
      const float taper = max(0.0f, 1.0f - 0.25f * q);
      const float w = pmath::exp2(-0.5f * q * kLog2e) * (taper * taper * taper);
      if constexpr (kTile) {
        const float* texel = row + static_cast<int64_t>(tx + clamp(i - ox, 0, pitch - 1)) * kCh;
        for (int32_t ch = 0; ch < kCh; ++ch) sum[ch] += w * texel[ch];
      } else {
        const float* texel = row + static_cast<int64_t>(inside ? i : wrap(i, lv.width)) * kCh;
        for (int32_t ch = 0; ch < kCh; ++ch) sum[ch] += w * texel[ch];
      }
      sum_w += w;
    }
  }
  if (sum_w > 0.0f) {
    for (int32_t ch = 0; ch < kCh; ++ch) out[ch] = sum[ch] / sum_w;
    return;
  }
  // Unreachable while the variance floor holds; nearest texel as a safe answer.
  const int32_t ni = static_cast<int32_t>(floor(cx + 0.5f)), nj = static_cast<int32_t>(floor(cy + 0.5f));
  const int32_t ii = kTile ? tx + clamp(ni - ox, 0, pitch - 1) : wrap(ni, lv.width);
  const int32_t jn = kTile ? ty + clamp(nj - oy, 0, pitch - 1) : wrap(nj, lv.height);
  for (int32_t ch = 0; ch < kCh; ++ch) {
    out[ch] = texels[lv.offset + (static_cast<uint64_t>(jn) * static_cast<uint64_t>(lv.width) +
                                  static_cast<uint64_t>(ii)) * kCh + static_cast<uint64_t>(ch)];
  }
}

// The filtered value (ewa_lookup_n); kTile: TextureView::tile_pitch > 0.
template <int32_t kCh, bool kTile>
BALLA_HD inline void lookup_n(const TextureView& tex, const float* texels, Vec2<float> uv, Sym2f cov_uv, float* out) {
  const TextureLevel& l0 = tex.levels[0];
  const float w0 = static_cast<float>(l0.width), h0 = static_cast<float>(l0.height);
  Sym2f c{cov_uv.xx * (w0 * w0) + (1.0f / 12.0f), cov_uv.xy * (w0 * h0), cov_uv.yy * (h0 * h0) + (1.0f / 12.0f)};
  // Footprints wider than the texture (grazing views, degenerate Jacobians) filter to the texture
  // mean, which the 1 x 1 top level holds. Only the major axis may be that wide, and a grazing
  // footprint still sees a band average there, so from var_cap to 2 var_cap the footprint's own
  // value blends into the mean with the C2 smootherstep. From 2 var_cap on, and for NaN, the mean
  // is returned directly, so the loops stay bounded. A tile cell is the texture here: its side P sets the cap, and
  // its top-level texel (the cell mean) is the mean.
  const int32_t top = tex.num_levels - 1;
  const float sigma_cap = kTile ? 4.0f * static_cast<float>(tex.tile_pitch) : 4.0f * max(w0, h0);
  const float var_cap = sigma_cap * sigma_cap;
  TileCell cell{0, 0, 0, 0};
  if constexpr (kTile) cell = tile_cell(l0, tex.tile_pitch, uv);
  Eigen2 e = eigenvalues(c);
  float to_mean = 0.0f, mean[kCh];
  for (int32_t ch = 0; ch < kCh; ++ch) mean[ch] = 0.0f;
  if (!(e.max <= var_cap)) {
    const TextureLevel& tl = tex.levels[top];
    // A tile pyramid's top level holds one texel per cell (pitch 1 there): the cell's mean.
    const float* top_texel =
        kTile ? texels + tl.offset + (static_cast<uint64_t>(cell.wy) * static_cast<uint64_t>(tl.width) +
                                      static_cast<uint64_t>(cell.wx)) * kCh
              : texels + tl.offset;
    for (int32_t ch = 0; ch < kCh; ++ch) mean[ch] = top_texel[ch];
    const float t = e.max / var_cap - 1.0f;
    if (!(t < 1.0f)) {  // also catches NaN from a degenerate footprint
      for (int32_t ch = 0; ch < kCh; ++ch) out[ch] = mean[ch];
      return;
    }
    to_mean = pmath::smootherstep(t);
  }
  // Anisotropy clamp: raise the minor eigenvalue towards max/256 along its own eigenvector.
  const float ratio = e.max > 0.0f ? e.min / e.max : 1.0f;
  if (ratio < kMinEigenRatio) {
    const float raised = e.max * smooth_floor(max(ratio, 0.0f), kMinEigenRatio);
    const float add = raised - e.min;
    // Projector onto the minor eigenvector: (max I - C) / (max - min); max - min >= 0.99 max.
    const float k = add / (e.max - e.min);
    c = {c.xx + k * (e.max - c.xx), c.xy - k * c.xy, c.yy + k * (e.max - c.yy)};
    e.min = raised;
  }
  // Level from the minor axis: sigma at the finer blended level is in [kLevelSigma, 2 kLevelSigma).
  const float level = 0.5f * pmath::log2(max(e.min, 1e-30f)) - pmath::log2(kLevelSigma);
  int32_t lo = 0;
  float blend = 0.0f;
  if (level > 0.0f) {
    const float fl = floor(level);
    if (fl >= static_cast<float>(top)) {
      lo = top;
    } else {
      lo = static_cast<int32_t>(fl);
      blend = pmath::smootherstep(level - fl);
    }
  }
  // The tile pitch at level l is tile_pitch >> l (0 without tiles, unused).
  float value[kCh], at[kCh];
  for (int32_t ch = 0; ch < kCh; ++ch) value[ch] = 0.0f;
  if (blend < 1.0f) {
    evaluate_level<kCh, kTile>(tex.levels[lo], texels, uv, c, at, cell, tex.tile_pitch >> lo);
    for (int32_t ch = 0; ch < kCh; ++ch) value[ch] = (1.0f - blend) * at[ch];
  }
  if (blend > 0.0f) {
    evaluate_level<kCh, kTile>(tex.levels[lo + 1], texels, uv, c, at, cell, tex.tile_pitch >> (lo + 1));
    for (int32_t ch = 0; ch < kCh; ++ch) value[ch] += blend * at[ch];
  }
  for (int32_t ch = 0; ch < kCh; ++ch) {
    out[ch] = to_mean > 0.0f ? (1.0f - to_mean) * value[ch] + to_mean * mean[ch] : value[ch];
  }
}

}  // namespace ewa

// Filtered texture value at uv for a footprint of covariance cov_uv (uv units squared); kCh = tex.channels values
// into out[0..kCh). Precondition (render::validate): (|uv| + 32) max(W, H) <= 2^30, so every texel index fits int32. A
// tile atlas (tile_pitch > 0; mono only, see the header) needs its pyramid capped at log2(tile_pitch) + 1 levels
// (render::SceneData).
template <int32_t kCh>
BALLA_HD inline void ewa_lookup_n(const TextureView& tex, const float* texels, Vec2<float> uv, Sym2f cov_uv,
                                  float* out) {
  if constexpr (kCh == 1) {
    if (tex.tile_pitch > 0) {
      ewa::lookup_n<1, true>(tex, texels, uv, cov_uv, out);
      return;
    }
  }
  ewa::lookup_n<kCh, false>(tex, texels, uv, cov_uv, out);
}

// The one-channel (mono) lookup.
BALLA_HD inline float ewa_lookup(const TextureView& tex, const float* texels, Vec2<float> uv, Sym2f cov_uv) {
  float v;
  ewa_lookup_n<1>(tex, texels, uv, cov_uv, &v);
  return v;
}

}  // namespace balla::render
