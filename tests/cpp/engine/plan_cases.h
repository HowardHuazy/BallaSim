// P2-D D1: a random pixel stage under the obliquity factors and P&V's plan bits (plan.h PixelPlan, PvPlanPass), and
// its run on one memory space, for test_plan.cpp (host) and test_device.cpp (host = device). P2-F F9b: optionally with
// two moving bodies' AO receivers (plan.h AoMover).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "engine/motion.h"
#include "engine/par.h"

namespace balla::engine::test {

struct PixelStageInput {
  PixelStage st;
  const CameraGeometry* geo;
  std::vector<float> depth, normal;
  std::vector<uint32_t> instance, state;
  std::vector<int8_t> level;
  std::vector<float> v_hat;
};
struct PixelStageRun {
  TileTotals tiles;
  std::vector<double> k, k_base;
  std::vector<int32_t> reach;
  std::vector<uint8_t> flags, pv_plan;
  std::vector<float> width;
  std::vector<unsigned long long> counts;
};

// Random receivers on geo's pixels (misses, NaN normals, two moving receivers' instances) under two point lights, a
// directional one and a moving point light, a moving camera; with `band`, I8's band and the seeds with random shadow
// states; random tile levels and bounds for PvPlanPass; with `ao` (P2-F F9b), two moving bodies' AO reach.
inline PixelStageInput pixel_stage_input(const CameraGeometry& geo, bool band, uint64_t seed, bool ao = false) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const size_t n = geo.n_pixels, tiles = static_cast<size_t>(geo.tiles_x) * static_cast<size_t>(geo.tiles_y);
  const auto w = static_cast<size_t>(geo.width);
  PixelStageInput in{};
  in.geo = &geo;
  PlanParams& p = in.st.params;
  p.omega = 0.4;
  p.v = 0.3;
  p.jmax = geo.speed_factor_max;
  p.jmin = geo.speed_factor_min;
  p.support = geo.support_px;
  p.big = std::max(geo.tiles_x, geo.tiles_y);
  p.lights = 1;
  p.n_shadow = 3;
  p.n_moving = 1;
  p.n_bits = 4;
  p.r_wc = Mat3d{{{0.8, 0.0, 0.6}, {0.0, 1.0, 0.0}, {-0.6, 0.0, 0.8}}};
  p.c_c = {0.1, -0.2, 0.05};
  p.width = geo.width;
  p.height = geo.height;
  p.band = band ? 1 : 0;
  p.seeds = band ? 1 : 0;
  p.n_movers = 2;
  p.obliquity = 1;
  p.bits = 1;
  p.pv = 1;
  p.pitch = 1.0 / 16;
  p.crop_x0 = 8;
  p.crop_y0 = 4;
  p.crop_x1 = geo.width - 6;
  p.crop_y1 = geo.height - 2;
  const Vec3d lights[3] = {{0.3, -1.5, 1.0}, {-0.8, -1.2, 0.4}, {-0.3, -0.9, -0.3}};  // the last: directional
  for (uint32_t l = 0; l < 3; ++l) {
    const Vec3d pos = l == 2 ? (1.0 / norm(lights[l])) * lights[l] : lights[l];
    in.st.shadow_lights.push_back({l == 2 ? 1u : 0u, 2 * l, 2, pos, l});
    for (uint32_t b = 0; b < 2; ++b) {
      ShadowBody sb{};
      sb.mode = l == 2 ? ShadowBody::kCylinder : (b == 0 ? ShadowBody::kBeyond : ShadowBody::kCone);
      sb.near = 0.3 + 0.2 * b;
      sb.cos_half = 0.2;
      sb.axis = Vec3d{0.0, 0.8, 0.6};
      sb.centre = Vec3d{0.0, 0.0, 1.5};
      sb.radius = 0.4;
      sb.c_dot_l = 0.2;
      sb.c_dot_c = 2.25;
      sb.speed = 0.5 + b;
      sb.keep = b == 0 ? ShadowBody::kBand : ShadowBody::kKeep;
      sb.extent = b == 0 ? 0.05 : 0.004;  // the band body near region (d)'s threshold, the kept one thin (R4)
      sb.far = 1.5;
      in.st.shadow_bodies.push_back(sb);
    }
  }
  in.st.moving.push_back({MovingLight::kPoint, 0, 0, Vec3d{0.5, -1.0, 0.8}, 0.4, 0.9, 0.0});
  in.st.movers.push_back({5, Vec3d{0.0, 0.1, 1.6}, 0.3, 1.2, 0.004});
  in.st.movers.push_back({9, Vec3d{0.3, 0.0, 1.9}, 0.6, 0.0, 0.004});
  if (ao) {  // receivers about 1.3-2.2 m from the camera; reaches covering part of them, one body much faster
    in.st.ao.push_back({Vec3d{0.5, 0.1, 1.5}, 0.45, 0.4});
    in.st.ao.push_back({Vec3d{1.4, -0.3, 1.0}, 0.6, 40.0});
    p.n_ao = 2;
  }
  in.depth.resize(n);
  in.instance.resize(n);
  in.normal.resize(3 * n);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (size_t i = 0; i < n; ++i) {
    const bool miss = u(rng) < 0.05;
    in.depth[i] = miss ? nan : static_cast<float>(1.2 + 0.01 * static_cast<double>(i % w) + 0.3 * u(rng));
    const double r = u(rng);
    in.instance[i] = miss ? 0 : (r < 0.15 ? 5 : (r < 0.25 ? 9 : 1 + static_cast<uint32_t>(i / w / 24)));
    Vec3d nn{u(rng) - 0.5, u(rng) - 0.5, -1.0 - u(rng)};
    nn = (1.0 / norm(nn)) * nn;
    const bool bad = !miss && u(rng) < 0.01;
    in.normal[3 * i] = miss || bad ? nan : static_cast<float>(nn.x);
    in.normal[3 * i + 1] = miss || bad ? nan : static_cast<float>(nn.y);
    in.normal[3 * i + 2] = miss || bad ? nan : static_cast<float>(nn.z);
  }
  if (band) {
    in.state.resize(2 * n * 4);
    for (size_t i = 0; i < n * 4; ++i) {
      const double r = u(rng);
      in.state[2 * i] = r < 0.5 ? 0u : (r < 0.7 ? 1u : (r < 0.85 ? kStateMover + 5 : kStateMover + 9));
      if (r > 0.98) in.state[2 * i] |= kStateMulti;
      const float t = static_cast<float>(0.2 + 0.6 * u(rng));
      uint32_t bits = 0;
      std::memcpy(&bits, &t, sizeof bits);
      in.state[2 * i + 1] = bits;
    }
  }
  in.level.resize(tiles);
  in.v_hat.resize(tiles);
  for (size_t t = 0; t < tiles; ++t) {
    in.level[t] = static_cast<int8_t>(u(rng) < 0.1 ? -1 : static_cast<int>(1 + rng() % 9));
    in.v_hat[t] = static_cast<float>(200.0 * u(rng));
  }
  return in;
}

// The pixel stage (enqueue_pixel_stage, collect_pixel_stage) and, with the plan bits, PvPlanPass at the input's tile
// levels (capped at 6) and bounds: the per-pixel rates, reaches, flags, plan bits, R4's widths, k_base, the tile totals
// and the counts (the last four empty where their key is off).
template <Space S>
PixelStageRun pixel_stage_run(const PixelStageInput& in) {
  using P = Par<S>;
  const CameraGeometry& geo = *in.geo;
  const size_t n = geo.n_pixels;
  const PlanParams& p = in.st.params;
  PlanBuffers<S> b;
  Buffer<float, S> depth, normal, v_hat;
  Buffer<uint32_t, S> instance, state, tile;
  Buffer<int8_t, S> level;
  Buffer<unsigned long long, S> counts;
  const uint32_t* st = in.state.empty() ? nullptr : upload_vector(state, in.state, nullptr);
  enqueue_pixel_stage<S>(in.st, geo, upload_vector(depth, in.depth, nullptr), 1, b, nullptr,
                         upload_vector(instance, in.instance, nullptr), st, nullptr,
                         upload_vector(normal, in.normal, nullptr));
  PixelStageRun out;
  out.tiles = collect_pixel_stage<S>(geo, b, nullptr, p.band != 0, p.seeds != 0, p.obliquity != 0, p.n_ao != 0);
  const auto get = [n](auto* dst, const auto* src) {
    dst->resize(n);
    P::download(dst->data(), src, n * sizeof((*dst)[0]), nullptr);
  };
  get(&out.k, b.k.data());
  get(&out.reach, b.reach.data());
  get(&out.flags, b.flags.data());
  if (p.bits == 0) return out;
  unsigned long long* c = counts.ensure(4);
  P::zero(c, 4 * sizeof(unsigned long long), nullptr);
  P::for_each(n,
              PvPlanPass{b.pv_plan.data(), p.pv != 0 ? b.pv_width.data() : nullptr,
                         upload_vector(tile, geo.pixel_tile, nullptr), upload_vector(level, in.level, nullptr),
                         upload_vector(v_hat, in.v_hat, nullptr), 6, p.pitch, geo.width, p.crop_x0, p.crop_y0,
                         p.crop_x1, p.crop_y1, c, p.obliquity},
              nullptr);
  get(&out.pv_plan, b.pv_plan.data());
  if (p.obliquity != 0) get(&out.k_base, b.k_base.data());
  if (p.pv != 0) get(&out.width, b.pv_width.data());
  out.counts.resize(4);
  P::download(out.counts.data(), c, 4 * sizeof(unsigned long long), nullptr);
  return out;
}

}  // namespace balla::engine::test
