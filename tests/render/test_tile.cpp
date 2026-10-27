// Tile-mode textures (docs/design/phase3/scans.md §4.5.6, §8 T8; increment S5a): a per-face tile atlas of pitch P
// has its pyramid capped at log2(P) + 1 levels, whose top level holds each cell's mean, and render/shade/ewa.h reads
// only the cell its uv lies in. The gutter case of T8 (gutter texels equal their neighbours' interior texels) belongs
// to the Replica importer that writes them (S5b).
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "render/shade/ewa.h"
#include "render/texture.h"

using namespace balla;
using namespace balla::render;

namespace {

int32_t tile_levels(int32_t pitch) { return std::countr_zero(static_cast<uint32_t>(pitch)) + 1; }

struct Atlas {
  int32_t pitch, width, height;
  std::vector<double> level0;
  std::vector<float> texels;
  std::vector<TextureLevel> levels;
  TextureView view() const { return {levels.data(), static_cast<int32_t>(levels.size()), 1, pitch}; }
  float lookup(double x_tex, double y_tex, double sxx, double sxy, double syy) const {
    // Centre in level-0 texel units, covariance in level-0 texel units squared; ewa_lookup takes uv units.
    const double w = width, h = height;
    const Sym2f cov{static_cast<float>(sxx / (w * w)), static_cast<float>(sxy / (w * h)),
                    static_cast<float>(syy / (h * h))};
    return ewa_lookup(view(), texels.data(), {static_cast<float>(x_tex / w), static_cast<float>(y_tex / h)}, cov);
  }
};

// cells_x x cells_y cells of side `pitch` (pitch 0: one untiled texture of cells_x x cells_y texels).
template <class Fn>
Atlas make_atlas(int32_t pitch, int32_t cells_x, int32_t cells_y, Fn value) {
  const int32_t side = pitch > 0 ? pitch : 1;
  Atlas a{pitch, side * cells_x, side * cells_y, {}, {}, {}};
  a.level0.resize(static_cast<size_t>(a.width) * static_cast<size_t>(a.height));
  for (int32_t j = 0; j < a.height; ++j) {
    for (int32_t i = 0; i < a.width; ++i) a.level0[static_cast<size_t>(j) * a.width + i] = value(i, j);
  }
  a.levels = build_mip_pyramid(a.level0, a.width, a.height, a.texels, pitch > 0 ? tile_levels(pitch) : 0);
  return a;
}

uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

// A footprint covariance (level-0 texels^2) with eigenvalues 2^[-4, 15] (up to sigma 181, past every variance cap
// here, 4P <= 64) at a random angle, or one of the degenerate footprints the renderer can produce.
struct Footprint {
  double xx, xy, yy;
};
Footprint footprint(std::mt19937& rng, int k) {
  switch (k % 16) {
    case 0: return {std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0};
    case 1: return {1e30, 0.0, 1e30};
    case 2: return {0.0, 0.0, 0.0};
    case 3: return {4.0, 2.0, 1.0};  // rank 1
    default: break;
  }
  std::uniform_real_distribution<double> logs(-4.0, 15.0), ang(0.0, 3.14159265);
  const double a = std::exp2(logs(rng)), b = std::exp2(logs(rng)), th = ang(rng);
  const double c = std::cos(th), s = std::sin(th);
  return {a * c * c + b * s * s, (a - b) * c * s, a * s * s + b * c * c};
}

// Cell (cx, cy) of an atlas as a texture of its own: pitch P, one cell, the same texels.
Atlas cell_tile(const Atlas& a, int32_t cx, int32_t cy) {
  const int32_t p = a.pitch;
  return make_atlas(p, 1, 1, [&](int i, int j) {
    return a.level0[static_cast<size_t>(cy * p + j) * a.width + static_cast<size_t>(cx * p + i)];
  });
}

// The atlases of the lookup tests are 3 x 2 cells: not square, so a lookup that swaps the axes of the cell index or of
// the cell's first texel reads another cell (or past the texture) at every cell off the diagonal, and every cell
// (cx, cy) is tested, the corner and edge ones whose clamped kernels wrap to the opposite side of the atlas included.
constexpr int32_t kNx = 3, kNy = 2;

// The footprint's eigenvalue ratio min / max (0 for NaN, infinite and zero footprints).
double anisotropy(const Footprint& f) {
  const double m = 0.5 * (f.xx + f.yy), r = std::hypot(0.5 * (f.xx - f.yy), f.xy);
  return m + r > 0.0 && std::isfinite(m + r) ? (m - r) / (m + r) : 0.0;
}

}  // namespace

TEST_CASE("tile pyramid: log2(P) + 1 exact halvings, the top level one mean per cell") {
  for (int32_t p : {4, 8, 16}) {
    std::mt19937 rng(static_cast<uint32_t>(p));
    std::uniform_real_distribution<double> u(0.0, 1.0);
    // Dyadic texels (multiples of 2^-8): every cell mean is exact in fp64 and in fp32.
    const Atlas dyadic = make_atlas(p, 3, 2, [&](int, int) { return std::floor(u(rng) * 256.0) / 256.0; });
    const Atlas noise = make_atlas(p, 3, 2, [&](int, int) { return u(rng); });
    for (const Atlas* a : {&dyadic, &noise}) {
      INFO("P = " << p << (a == &dyadic ? ", dyadic texels" : ", noise"));
      REQUIRE(static_cast<int32_t>(a->levels.size()) == tile_levels(p));
      for (size_t l = 0; l < a->levels.size(); ++l) {
        CHECK(a->levels[l].width == a->width >> l);
        CHECK(a->levels[l].height == a->height >> l);
        CHECK(a->levels[l].scale_x == static_cast<float>(1 << l));
        CHECK(a->levels[l].scale_y == static_cast<float>(1 << l));
      }
      const TextureLevel& top = a->levels.back();
      REQUIRE((top.width == 3 && top.height == 2));
      for (int32_t cy = 0; cy < 2; ++cy) {
        for (int32_t cx = 0; cx < 3; ++cx) {
          double mean = 0.0;
          for (int32_t j = 0; j < p; ++j) {
            for (int32_t i = 0; i < p; ++i) mean += a->level0[static_cast<size_t>(cy * p + j) * a->width + cx * p + i];
          }
          mean /= p * p;
          const float got = a->texels[top.offset + static_cast<uint32_t>(cy * 3 + cx)];
          if (a == &dyadic) {
            CHECK(got == static_cast<float>(mean));
          } else {
            CHECK(std::abs(got - static_cast<float>(mean)) <= std::abs(static_cast<float>(mean)) * 0x1p-23f);
          }
        }
      }
    }
  }
  // Untiled textures keep every level down to 1 x 1; a negative cap is refused.
  std::vector<float> texels;
  CHECK(build_mip_pyramid(std::vector<double>(24 * 16, 0.5), 24, 16, texels).size() == 5);  // ... 3 x 2, 1 x 1
  CHECK_THROWS_AS(build_mip_pyramid(std::vector<double>(16, 0.5), 4, 4, texels, -1), std::invalid_argument);
  // The 8-bit and binary16 builders pass the cap through and keep level 0.
  std::vector<uint8_t> rgb8(16 * 8 * 3);
  std::vector<uint16_t> rgb16(16 * 8 * 3);
  for (size_t i = 0; i < rgb8.size(); ++i) {
    rgb8[i] = static_cast<uint8_t>((i * 37) % 251);
    rgb16[i] = static_cast<uint16_t>((i * 997) % 0x5000);
  }
  const std::array<double, 3> w{0.52, 0.37, 0.11};
  std::vector<float> full8, capped8, full16, capped16;
  const auto f8 = build_mono_mip_pyramid(rgb8.data(), 16, 8, 3, w, full8);
  const auto c8 = build_mono_mip_pyramid(rgb8.data(), 16, 8, 3, w, capped8, 3);
  const auto f16 = build_linear_mip_pyramid(rgb16.data(), 16, 8, w, full16);
  const auto c16 = build_linear_mip_pyramid(rgb16.data(), 16, 8, w, capped16, 3);
  CHECK((f8.size() == 5 && c8.size() == 3 && f16.size() == 5 && c16.size() == 3));
  CHECK(std::equal(capped8.begin(), capped8.end(), full8.begin()));  // the capped levels are the leading ones
  CHECK(std::equal(capped16.begin(), capped16.end(), full16.begin()));
}

TEST_CASE("tile lookups read only their own cell: bitwise unchanged when the other cells change") {
  // T8: every cell (cx, cy) of a 3 x 2 atlas, lookups anywhere inside the cell (up to 1e-3 texel from its edges) for
  // footprints from below the magnification floor to past the cell's variance cap, and degenerate ones. Changing every
  // other cell leaves each lookup bit for bit; filling them with NaN (the read instrument: one read of a NaN texel at
  // any weight, zero included, makes the result NaN) leaves it too, so no read leaves the cell at any level.
  for (int32_t p : {4, 8, 16}) {
    int32_t n_total = 0, differ_total = 0;
    for (int32_t cy = 0; cy < kNy; ++cy) {
      for (int32_t cx = 0; cx < kNx; ++cx) {
        std::mt19937 rng(100 + 16 * static_cast<uint32_t>(p) + static_cast<uint32_t>(cy * kNx + cx));
        std::uniform_real_distribution<double> u(0.2, 1.0);
        std::vector<double> own(static_cast<size_t>(p) * p);
        for (double& c : own) c = u(rng);
        auto in_cell = [&](int i, int j) { return i / p == cx && j / p == cy; };
        auto with = [&](auto other) {
          return make_atlas(p, kNx, kNy, [&](int i, int j) {
            if (!in_cell(i, j)) return other(i, j);
            return own[static_cast<size_t>(j - cy * p) * p + static_cast<size_t>(i - cx * p)];
          });
        };
        const Atlas a = with([&](int, int) { return u(rng); });
        const Atlas b = with([&](int i, int j) { return (i + 2 * j) % 3 == 0 ? 0.0 : 50.0; });
        const Atlas poisoned = with([](int, int) { return std::numeric_limits<double>::quiet_NaN(); });
        const float cell_mean = a.texels[a.levels.back().offset + static_cast<uint32_t>(cy * kNx + cx)];
        std::uniform_real_distribution<double> px(cx * p + 1e-3, (cx + 1) * p - 1e-3),
            py(cy * p + 1e-3, (cy + 1) * p - 1e-3);
        for (int k = 0; k < 1500; ++k) {
          const Footprint f = footprint(rng, k);
          const double x = px(rng), y = py(rng);
          const float va = a.lookup(x, y, f.xx, f.xy, f.yy);
          CAPTURE(p);
          CAPTURE(cx);
          CAPTURE(cy);
          CAPTURE(k);
          REQUIRE(std::isfinite(va));
          CHECK(bits(b.lookup(x, y, f.xx, f.xy, f.yy)) == bits(va));
          CHECK(bits(poisoned.lookup(x, y, f.xx, f.xy, f.yy)) == bits(va));
          ++n_total;
          differ_total += va != cell_mean ? 1 : 0;
        }
      }
    }
    // Many footprints filter inside the cell rather than taking its mean (those past 2 (4P)^2 take it).
    INFO("P = " << p << ": " << differ_total << " of " << n_total << " lookups differ from their cell's mean");
    CHECK(differ_total > 3 * n_total / 8);
  }
}

TEST_CASE("a tile cell filters like the same tile alone, and its far footprints give the cell mean") {
  // T8, against a texture holding only the cell's tile (pitch P, one cell), at the same physical point and footprint,
  // on every cell of
  //  - a 4 x 2 atlas, whose sides are powers of two: the covariance in uv units and the centres on the 2^-8 grid are
  //    exact at the atlas and at the tile, so the two lookups run the same arithmetic on the same values and agree bit
  //    for bit, whatever the footprint (the clamp at the cell's edges, the cell's first texel, the level choice and the
  //    cell mean are the tile's);
  //  - a 3 x 2 atlas at arbitrary positions: within 2e-5 relative (scans.md §8 T8; the centre's fp32 rounding at
  //    absolute texel indices differs, §2.4 TF: 8.9e-6). The covariance is rounded to fp32 in uv units, which differ
  //    between the two, and the anisotropy clamp (min eigenvalue >= max / 256) turns a rounding of its entries into
  //    up to 256 times that relative change of the minor axis (2.07e-5 at a footprint of ratio 1/2000 in a measured
  //    run), so this comparison keeps footprints of ratio >= 1/16; the 4 x 2 atlas above has every footprint.
  // Past twice the cap (4P)^2 the value is the cell's top-level texel, the mean of the cell's own P x P texels
  // (computed here from level 0, not read back from the pyramid), whichever cell the uv lies in.
  for (int32_t p : {4, 8, 16}) {
    for (const int32_t nx : {4, 3}) {
      std::mt19937 rng(200 + static_cast<uint32_t>(p) + 100 * static_cast<uint32_t>(nx));
      std::uniform_real_distribution<double> u(0.2, 1.0);
      const Atlas atlas = make_atlas(p, nx, kNy, [&](int, int) { return u(rng); });
      const TextureLevel& top = atlas.levels.back();
      REQUIRE((top.width == nx && top.height == kNy));
      std::uniform_real_distribution<double> pos(1e-3, p - 1e-3);
      double worst = 0.0;
      int32_t compared = 0;
      for (int32_t cy = 0; cy < kNy; ++cy) {
        for (int32_t cx = 0; cx < nx; ++cx) {
          const Atlas tile = cell_tile(atlas, cx, cy);
          double sum = 0.0;
          for (int32_t j = 0; j < p; ++j) {
            const size_t row = static_cast<size_t>(cy * p + j) * atlas.width + static_cast<size_t>(cx * p);
            for (int32_t i = 0; i < p; ++i) sum += atlas.level0[row + static_cast<size_t>(i)];
          }
          const double mean = sum / (p * p);
          for (int k = 0; k < 2000; ++k) {
            const Footprint f = footprint(rng, k);
            CAPTURE(p);
            CAPTURE(nx);
            CAPTURE(cx);
            CAPTURE(cy);
            CAPTURE(k);
            if (nx == 4) {
              const double x = std::max(std::floor(pos(rng) * 256.0), 1.0) / 256.0;
              const double y = std::max(std::floor(pos(rng) * 256.0), 1.0) / 256.0;
              const float vt = tile.lookup(x, y, f.xx, f.xy, f.yy);
              REQUIRE(std::isfinite(vt));
              CHECK(bits(atlas.lookup(x + cx * p, y + cy * p, f.xx, f.xy, f.yy)) == bits(vt));
            } else if (anisotropy(f) >= 1.0 / 16.0 || k % 16 < 4) {  // degenerate footprints (k % 16 < 4) all
              const double x = pos(rng), y = pos(rng);
              const double vt = tile.lookup(x, y, f.xx, f.xy, f.yy);
              const double va = atlas.lookup(x + cx * p, y + cy * p, f.xx, f.xy, f.yy);
              REQUIRE(std::isfinite(vt));
              worst = std::max(worst, std::abs(va - vt) / vt);
              ++compared;
            }
          }
          const double far = 2.5 * 16.0 * p * p;
          const double x = (cx + 0.5) * p, y = (cy + 0.5) * p;
          const float far_mean = atlas.lookup(x, y, far, 0.0, far);
          CHECK(far_mean == atlas.texels[top.offset + static_cast<uint32_t>(cy * nx + cx)]);
          CHECK(std::abs(static_cast<double>(far_mean) - mean) <= mean * 0x1p-22);
          CHECK(atlas.lookup(x, y, far, 0.0, 0.0) == far_mean);  // a grazing footprint, along one axis only
          CHECK(atlas.lookup(x, y, 0.0, 0.0, far) == far_mean);
          CHECK(tile.lookup(0.5 * p, 0.5 * p, far, 0.0, far) == tile.texels[tile.levels.back().offset]);
          CHECK(tile.lookup(0.5 * p, 0.5 * p, far, 0.0, far) == far_mean);  // the same texel, from the tile alone
        }
      }
      if (nx == 3) {
        INFO("P = " << p << ", 3 x 2 atlas: " << compared << " lookups, worst relative difference " << worst);
        CHECK(compared > 3000);
        CHECK(worst <= 2e-5);
      }
    }
  }
}

TEST_CASE("tile lookups are C2 in uv inside a cell, where the kernel support is clamped at the cell's edges") {
  // Centre sweeps across the whole of cell (2, 1) of an atlas of unit-range noise, along x and along y, at footprints
  // whose support crosses the cell's edges near them (sigma 0.7-3 texels, level 0 and level 1): the second differences
  // stay at the smooth scale (a step at a clamped index would give ~1/h per texel boundary), and the ends keep reading
  // the cell.
  const int32_t p = 16, cx = 2, cy = 1;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const Atlas a = make_atlas(p, kNx, kNy, [&](int, int) { return u(rng); });
  for (const bool along_x : {true, false}) {
    for (const double s2 : {0.5, 2.0, 9.0}) {
      const double h = 1.0 / 256.0;
      std::vector<double> v;
      for (double t = h; t < p - h; t += h) {
        const double x = cx * p + (along_x ? t : 0.37), y = cy * p + (along_x ? 0.37 : t);
        v.push_back(a.lookup(x, y, s2, 0.1 * s2, 0.8 * s2));
      }
      double max_d1 = 0.0, max_d2 = 0.0;
      for (size_t k = 1; k + 1 < v.size(); ++k) {
        max_d1 = std::max(max_d1, std::abs(v[k + 1] - v[k - 1]) / (2 * h));
        max_d2 = std::max(max_d2, std::abs(v[k + 1] - 2 * v[k] + v[k - 1]) / (h * h));
      }
      INFO((along_x ? "x" : "y") << " sweep, variance " << s2 << ": max |f'| = " << max_d1 << ", max |f''| = "
                                 << max_d2);
      CHECK(max_d1 < 2.0);
      CHECK(max_d2 < 10.0);
    }
  }
}

TEST_CASE("tile lookups wrap the cell index outside [0, 1)") {
  // uv outside the atlas picks the cell its wrapped position lies in, as untiled textures repeat (ewa::tile_cell): the
  // lookup at a position shifted by whole atlases equals, bit for bit, the lookup of the cell that position wraps to as
  // a texture of its own (cell_tile), at every cell of a 2 x 4 atlas (the other orientation of a non-square grid; sides
  // that are powers of two and positions on the 2^-8 grid keep the centres and the covariance exact, as in the
  // single-tile test above), for footprints of every kind.
  const int32_t p = 8, nx = 2, ny = 4;
  std::mt19937 rng(9);
  std::uniform_real_distribution<double> u(0.2, 1.0);
  const Atlas a = make_atlas(p, nx, ny, [&](int, int) { return u(rng); });
  std::vector<Atlas> tiles;
  for (int32_t cy = 0; cy < ny; ++cy) {
    for (int32_t cx = 0; cx < nx; ++cx) tiles.push_back(cell_tile(a, cx, cy));
  }
  const int32_t shifts[][2] = {{0, 0}, {1, -2}, {-1, 3}, {2, 1}, {-2, -1}};
  std::uniform_real_distribution<double> local(0.0, p - 1.0 / 128.0);
  for (int k = 0; k < 4000; ++k) {
    const int32_t cx = k % nx, cy = (k / nx) % ny, *s = shifts[(k / (nx * ny)) % 5];
    const Footprint f = footprint(rng, k);
    const double lx = std::max(std::floor(local(rng) * 256.0), 1.0) / 256.0;
    const double ly = std::max(std::floor(local(rng) * 256.0), 1.0) / 256.0;
    CAPTURE(k);
    CAPTURE(cx);
    CAPTURE(cy);
    const float ref = tiles[static_cast<size_t>(cy * nx + cx)].lookup(lx, ly, f.xx, f.xy, f.yy);
    REQUIRE(std::isfinite(ref));
    CHECK(bits(a.lookup(lx + cx * p + s[0] * a.width, ly + cy * p + s[1] * a.height, f.xx, f.xy, f.yy)) == bits(ref));
  }
}
