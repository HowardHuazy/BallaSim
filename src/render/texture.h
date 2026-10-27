// Host construction of the renderer's mono textures (DESIGN §7.2-7.3, S2 MAT-03-06).
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "render/bundle.h"
#include "render/scene_view.h"

namespace balla::render {

// sRGB EOTF (IEC 61966-2-1), applied before any filtering: the log of gamma-encoded values would
// scale log-contrast by ~1/2.2 (DESIGN §7.1).
double srgb_to_linear(uint8_t c);

struct Material;

// The texture diagnostics of `ballasim scan report` (docs/design/phase3/scans.md §3.6, §4.6; S2c), per material. T is
// the texel value the renderer reads, sum_c w_c b_c x_c (w: the sensor RGB weights, b: the base colour, x: the IEC
// decode of an 8-bit code or a binary16 value; the order of build_mono_mip_pyramid). A texel is black when every
// channel is 0; an 8-bit texel is clipped when some channel is >= 254. Of the 8-bit texels neither black nor clipped,
// step_gt_theta (step_gt_theta4) counts those whose one-code step ln(T(code + 1 in every channel) / T(code)) exceeds
// theta (theta / 4); T(code) = 0 < T(code + 1) counts as an infinite step. The percentiles are NumPy's (linear rule)
// of T over the texels that are not black (the atlas background; NaN when every texel is). A binary16 texture has no
// clipping or code steps (counts 0); a material without texture reports its weighted base colour as every percentile.
enum class TextureSource : uint8_t { kNone = 0, kSrgb8 = 1, kBinary16 = 2 };
struct TextureStats {
  TextureSource source = TextureSource::kNone;
  int32_t width = 0, height = 0;
  int64_t texels = 0, black = 0, clipped = 0, step_gt_theta = 0, step_gt_theta4 = 0;
  std::array<double, 3> mono_p1_p50_p99{};
};
// Throws std::invalid_argument for a theta that is not finite and positive or for weights that are not finite and
// >= 0 (the schema's hard bound on sensor.rgb_weights).
TextureStats texture_stats(const Material& material, const std::array<double, 3>& sensor_rgb_weights, double theta);

// Appends the box mip pyramid of a texture to `texels` and returns its levels. Level 0 is
// sum_c weights[c] * linear(texture_c), with `weights` the sensor RGB weights times the material's
// base colour; filtering and weighting commute, so this equals weighting the filtered RGB.
// Each level halves the size (floor, at least 1) down to 1 x 1, and every texel is the exact area
// average of the previous level over its footprint in uv, so odd sizes keep texel centres aligned.
// Computed in fp64, stored in fp32. Throws std::length_error if the texel array would exceed
// 2^32 entries. max_levels > 0 stops the pyramid after that many levels (a tile atlas of pitch P takes log2(P) + 1,
// scans.md §4.5.6: its sides are multiples of P, so every level up to the last halves exactly and keeps the cells
// apart); 0 builds every level; negative throws std::invalid_argument.
std::vector<TextureLevel> build_mono_mip_pyramid(const uint8_t* texture, int32_t width,
                                                 int32_t height, int32_t channels,
                                                 const std::array<double, 3>& weights,
                                                 std::vector<float>& texels, int32_t max_levels = 0);

// The pyramid of a linear mono level 0 (row-major, width x height), as above.
std::vector<TextureLevel> build_mip_pyramid(std::vector<double> level0, int32_t width, int32_t height,
                                            std::vector<float>& texels, int32_t max_levels = 0);

// IEEE binary16 bits as their exact fp64 value (normals and subnormals; both zeros give +0). Infinities and NaN, which
// C1's validate() refuses in texture_linear, give +-inf and NaN.
double binary16_to_double(uint16_t bits);

// The pyramid of a linear binary16 RGB texture (C1 Material.texture_linear, H x W x 3; scans.md §4.1): level 0 is
// sum_c weights[c] * h_c with h_c the exact binary16 values, in the order of build_mono_mip_pyramid.
std::vector<TextureLevel> build_linear_mip_pyramid(const uint16_t* texture, int32_t width, int32_t height,
                                                   const std::array<double, 3>& weights, std::vector<float>& texels,
                                                   int32_t max_levels = 0);

// P2-F F11a (p2f.md §4.9): a material's albedo in the visible band and in the near infrared, at level 0.
//   visible  w . (b x rgb_lin)  (the sensor weights w times the base colour b: `wb`, as build_mono_mip_pyramid)
//   NIR      per Material::nir_form, with r = b_R R_lin (the red channel; 1 where there is no texture):
//            assumed r;  scale s r;  affine clamp(a + b r, 0, 1);  scalar c;  texture nir_texture / 255 (linear)
//            (nir_params = [s, -], [a, b], [c, -], the default [1, 0]).
// The images share one grid (the larger texture size in each axis; the smaller texture box-resampled to it, exactly
// where the sizes divide); width = height = 0 without a texture on the material (visible or NIR), then only the
// constants hold. vis_const and nir_const are always the untextured values (base colour terms; the NIR texture's form
// has no constant and leaves nir_const 0).
struct BandImages {
  int32_t width = 0, height = 0;
  std::vector<double> vis, nir;
  double vis_const = 0.0, nir_const = 0.0;
};
BandImages band_images(const Material& m, const std::array<double, 3>& wb);

// The pyramid of two level-0 images on one grid (row-major, width x height), as build_mip_pyramid, with the two
// channels interleaved per texel (TextureView::channels 2): a level's offset counts floats and its texel (i, j) holds
// (a, b) at offset + 2 (j width + i).
std::vector<TextureLevel> build_mip_pyramid2(std::vector<double> a, std::vector<double> b, int32_t width,
                                             int32_t height, std::vector<float>& texels);

}  // namespace balla::render
