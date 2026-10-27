// End-to-end oracles of the renderer (contract C2) on every available backend:
//   (1) a plane at a known pose: centre depth and instantaneous flow exact for a translating
//       camera, a rotating camera and a moving plane (M2 D3, D8);
//   (2) a textured plane under a static camera: irradiance against the per-sub-sample estimator
//       evaluated exactly, and against the exact integral of the texel image over
//       box(1 px) * Gaussian(0.15 px);
//   (3) the hard shadow of a moving occluder: edge position against geometry, and the static gate
//       (unaffected pixels are bitwise identical over time, DESIGN §7.9 B1);
//   (4) OptiX against Embree on a mixed scene (DESIGN §12.1 L2);
//   (5) determinism across batch sizes, thread counts and request order;
//   (6) a UV seam (two atlas charts with gutters) against one continuous chart, static and while
//       the seam crosses a moving pixel (DESIGN §13.1 O6 seam variant);
//   (7) per-face colour cells (Stream S's vertex-colour bake, a fully fragmented atlas) against
//       per-face materials, and OptiX against Embree on it.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numbers>
#include <numeric>
#include <random>
#include <vector>

#include "core/rng/warp_table.h"
#include "core/time/time.h"
#include "core/traj/spline_build.h"
#include "render/census.h"
#include "render/embree/embree_tracer.h"
#include "render/executor.h"
#include "render/pose_table.h"
#include "render/ray.h"
#include "render/renderer.h"
#include "render/scene_data.h"
#include "render/shade/lights.h"
#include "render/shade/shade.h"
#include "render/texture.h"
#include "render/thread_pool.h"
#include "scenes.h"

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;

namespace {

constexpr int64_t kT0 = 1'000'000'000, kT1 = 1'250'000'000;

struct Pose {
  Mat3<double> r;
  Vec3<double> p;
};

Pose eval_pose(const Trajectory& tr, int64_t t) {
  std::vector<Quat<double>> qs;
  std::vector<Vec3<double>> ps;
  for (size_t c = 0; c < tr.pos.size() / 3; ++c) {
    qs.push_back({tr.rot[4 * c], tr.rot[4 * c + 1], tr.rot[4 * c + 2], tr.rot[4 * c + 3]});
    ps.push_back({tr.pos[3 * c], tr.pos[3 * c + 1], tr.pos[3 * c + 2]});
  }
  const auto nodes = traj::build_spline_nodes(qs, ps, {0, 0, 0});
  const traj::SplineView<double> v{nodes.data(), static_cast<int32_t>(nodes.size()) - tr.order + 1,
                                   static_cast<int32_t>(tr.h_ns), 1e9 / static_cast<double>(tr.h_ns)};
  traj::SegmentTime st{};
  REQUIRE(traj::locate(t, tr.t0_ns, v.h_ns, v.num_segments, &st));
  const auto k = traj::evaluate<6>(v, st);
  return {to_matrix(k.q), k.p};
}

Vec3<double> mul_t(const Mat3<double>& a, Vec3<double> v) {
  return {a.m[0][0] * v.x + a.m[1][0] * v.y + a.m[2][0] * v.z, a.m[0][1] * v.x + a.m[1][1] * v.y + a.m[2][1] * v.z,
          a.m[0][2] * v.x + a.m[1][2] * v.y + a.m[2][2] * v.z};
}

// Pinhole flow of a camera-frame point moving with velocity xdot.
std::pair<double, double> pinhole_flow(double f, Vec3<double> x, Vec3<double> xdot) {
  return {f * (xdot.x * x.z - x.x * xdot.z) / (x.z * x.z), f * (xdot.y * x.z - x.y * xdot.z) / (x.z * x.z)};
}

std::array<double, 16> pose_matrix(const Mat3<double>& r, Vec3<double> p) {
  return {r.m[0][0], r.m[0][1], r.m[0][2], p.x, r.m[1][0], r.m[1][1], r.m[1][2], p.y,
          r.m[2][0], r.m[2][1], r.m[2][2], p.z, 0, 0, 0, 1};
}

Mat3<double> rot_y(double a) {
  return {{{std::cos(a), 0, std::sin(a)}, {0, 1, 0}, {-std::sin(a), 0, std::cos(a)}}};
}

double phi(double z) { return 0.5 * std::erfc(-z / std::numbers::sqrt2); }

// Exact integral of a texel row (texel i spans [i, i + 1) in texel units, repeat wrap) against a
// Gaussian of variance var centred at x.
double texel_image_filtered(const std::vector<double>& row, double x, double var) {
  const double s = std::sqrt(var);
  const auto n = static_cast<int32_t>(row.size());
  double sum = 0.0;
  const auto lo = static_cast<int32_t>(std::floor(x - 12 * s)) - 1, hi = static_cast<int32_t>(std::ceil(x + 12 * s)) + 1;
  for (int32_t i = lo; i <= hi; ++i) {
    sum += row[static_cast<size_t>(((i % n) + n) % n)] * (phi((i + 1 - x) / s) - phi((i - x) / s));
  }
  return sum;
}

// std::max(worst, NaN) keeps worst: a NaN error becomes +inf here, so NaN output fails the bound it is held to.
double worse(double worst, double err) {
  return std::isnan(err) ? std::numeric_limits<double>::infinity() : std::max(worst, err);
}

struct Requests {
  std::vector<uint16_t> cam;
  std::vector<uint32_t> pixel;
  std::vector<int64_t> t;
};

}  // namespace

TEST_CASE("oracle 1: depth and flow of a plane, exact for camera and plane motion") {
  constexpr double kF = 60.0;
  struct Case {
    const char* name;
    Trajectory rig, plane;
    bool plane_moves;
  };
  const Quat<double> id = quat_identity<double>();
  const std::vector<Case> cases = {
      {"translating camera", moving_trajectory(kT0, kT1, id, {0.1, -0.05, 0.0}, {0.3, -0.2, 0.5}, {0, 0, 0}),
       static_trajectory(kT0, kT1, id, {0, 0, 2.0}), false},
      {"rotating camera", moving_trajectory(kT0, kT1, id, {0, 0, 0}, {0, 0, 0}, {0.0, 0.8, 0.0}),
       static_trajectory(kT0, kT1, id, {0, 0, 2.0}), false},
      {"moving plane", static_trajectory(kT0, kT1, id, {0, 0, 0}),
       moving_trajectory(kT0, kT1, id, {-0.1, 0.05, 2.2}, {0.2, 0.1, -0.3}, {0.0, 0.0, 0.5}), true},
  };
  for (BackendKind backend : backends()) {
    for (const Case& c : cases) {
      SceneBundle s;
      s.light_groups = {"default"};
      s.trajectories = {c.rig, c.plane};
      s.cameras.push_back(pinhole(64, 48, kF, 0));
      Body plane = make_body(7, plane_mesh(3.0, 4));
      plane.trajectory = 1;  // a trajectory for the static case too: it then is a mover with zero motion
      if (!c.plane_moves) {
        plane.trajectory.reset();
        plane.T_world_body = translation(0, 0, 2.0);
      }
      s.bodies.push_back(plane);
      Renderer r(s, options(backend));
      std::vector<uint16_t> cams;
      std::vector<float> xs, ys;
      std::vector<int64_t> ts;
      std::mt19937 rng(11);
      std::uniform_real_distribution<float> ux(0.0f, 63.0f), uy(0.0f, 47.0f);
      std::uniform_int_distribution<int64_t> ut(kT0, kT1);
      for (int i = 0; i < 400; ++i) {
        cams.push_back(0);
        xs.push_back(ux(rng));
        ys.push_back(uy(rng));
        ts.push_back(ut(rng));
      }
      const std::vector<GtSample> gt = r.gt(cams, xs, ys, ts);
      double worst_depth = 0.0, worst_flow = 0.0;
      for (size_t i = 0; i < gt.size(); ++i) {
        const Pose cam = eval_pose(c.rig, ts[i]);
        const Pose pl = c.plane_moves ? eval_pose(c.plane, ts[i]) : Pose{{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 2.0}};
        // Ray through the pixel, intersected with the plane (normal = plane body z axis) in fp64.
        const Vec3<double> dc{(static_cast<double>(xs[i]) - 31.5) / kF, (static_cast<double>(ys[i]) - 23.5) / kF, 1.0};
        const Vec3<double> dw = cam.r * dc;
        const Vec3<double> nw{pl.r.m[0][2], pl.r.m[1][2], pl.r.m[2][2]};
        const double t = dot(nw, pl.p - cam.p) / dot(nw, dw);
        const Vec3<double> xc = t * dc;
        // Velocity of the seen point in the camera frame (M2 D1), from the known twists.
        Vec3<double> xdot;
        if (std::string(c.name) == "translating camera") {
          xdot = -1.0 * Vec3<double>{0.3, -0.2, 0.5};
        } else if (std::string(c.name) == "rotating camera") {
          xdot = -1.0 * cross(mul_t(cam.r, Vec3<double>{0, 0.8, 0}), xc);
        } else {
          const Vec3<double> xw = cam.p + t * dw;
          xdot = Vec3<double>{0.2, 0.1, -0.3} + cross(Vec3<double>{0, 0, 0.5}, xw - pl.p);
        }
        const auto [fx, fy] = pinhole_flow(kF, xc, xdot);
        REQUIRE(gt[i].instance == 7u);
        CHECK(gt[i].semantic == 107u);
        worst_depth = worse(worst_depth, std::abs(static_cast<double>(gt[i].depth) - xc.z) / xc.z);
        worst_flow = worse(worst_flow, std::hypot(static_cast<double>(gt[i].flow_x) - fx, static_cast<double>(gt[i].flow_y) - fy) / (1.0 + std::hypot(fx, fy)));
      }
      MESSAGE(backend_name(backend) << ", " << std::string(c.name) << ": depth max rel error " << worst_depth
                                    << ", flow max rel error " << worst_flow);
      CHECK(worst_depth <= 2e-6);
      CHECK(worst_flow <= 2e-5);
    }
  }
}

TEST_CASE("oracle 1b: misses give instance 0 and NaN ground truth") {
  SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories = {static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0})};
  s.cameras.push_back(pinhole(32, 24, 30.0, 0));
  Body b = make_body(2, plane_mesh(0.1));
  b.T_world_body = translation(0, 0, 2.0);
  s.bodies.push_back(b);
  for (BackendKind backend : backends()) {
    Renderer r(s, options(backend));
    const DenseFrame f = r.dense(0, kT0 + 5, false, true);
    REQUIRE(f.gt.size() == 32u * 24u);
    const GtSample corner = f.gt[0];
    CHECK(corner.instance == 0u);
    CHECK(corner.semantic == 0u);
    CHECK(std::isnan(corner.depth));
    CHECK(std::isnan(corner.flow_x));
    const GtSample centre = f.gt[12 * 32 + 16];
    CHECK(centre.instance == 2u);
    CHECK(centre.depth == doctest::Approx(2.0f));
    CHECK(centre.flow_x == 0.0f);
  }
}

TEST_CASE("oracle 2: textured plane: per-sub-sample texture lookups and the pixel-response integral") {
  // Texture varying along u only, 2048 texels across a 2 m plane at 1 m: ~8.5 texels per pixel
  // (EVIMO2 minifies 4.6-16.5x). Period 128 texels (~15 px facing the camera, down to ~5 px on the
  // far side of the tilted plane).
  constexpr int kTexW = 2048, kTexH = 4;
  constexpr double kHalf = 1.0, kZ = 1.0, kF = 120.0;
  constexpr int kW = 96, kH = 48;
  std::vector<uint8_t> tex(static_cast<size_t>(kTexW) * kTexH * 3);
  std::vector<double> lin(kTexW);
  for (int i = 0; i < kTexW; ++i) {
    const double v = 0.45 + 0.3 * std::cos(2.0 * std::numbers::pi * (i + 0.5) / 128.0);
    const double enc = v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
    const auto byte = static_cast<uint8_t>(std::lround(enc * 255.0));
    lin[static_cast<size_t>(i)] = srgb_to_linear(byte);
    for (int j = 0; j < kTexH; ++j) {
      for (int c = 0; c < 3; ++c) tex[(static_cast<size_t>(j) * kTexW + i) * 3 + c] = byte;
    }
  }
  const rng::PixelResponse response(1.0, 0.15, 3.0);
  for (BackendKind backend : backends()) {
    for (double theta : {0.0, 0.6}) {
      SceneBundle s;
      s.light_groups = {"default"};
      s.trajectories = {static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0})};
      s.cameras.push_back(pinhole(kW, kH, kF, 0));
      Body plane = make_body(1, plane_mesh(kHalf, 8));
      plane.materials[0].texture = tex;
      plane.materials[0].texture_width = kTexW;
      plane.materials[0].texture_height = kTexH;
      plane.materials[0].texture_channels = 3;
      plane.T_world_body = pose_matrix(rot_y(theta), {0, 0, kZ});
      s.bodies.push_back(plane);
      s.lights.push_back(ambient_light(1.0f));
      const RendererOptions opts = options(backend);
      Renderer r(s, opts);
      // The renderer's own sub-sample rays, for the estimator oracle.
      const SceneData data(s, opts.scene);
      const SceneView view = data.host_view();
      const uint32_t n_sub = opts.scene.n_subsamples;
      std::vector<uint16_t> cams;
      std::vector<uint32_t> pixels;
      std::vector<int64_t> ts;
      for (int y = 20; y < 28; ++y) {
        for (int x = 4; x < kW - 4; ++x) {
          cams.push_back(0);
          pixels.push_back(static_cast<uint32_t>(y * kW + x));
          ts.push_back(kT0 + 1000);
        }
      }
      const RenderResult res = r.render(cams, pixels, ts);
      // Pixel x-position of the texel boundary at plane coordinate s: x_n = s cos / (Z - s sin).
      auto boundary_px = [&](int i) {
        const double sb = -kHalf + 2.0 * kHalf * i / kTexW;
        return 0.5 * (kW - 1) + kF * sb * std::cos(theta) / (kZ - sb * std::sin(theta));
      };
      double worst_estimator = 0.0, worst_integral = 0.0, sq_integral = 0.0;
      for (size_t k = 0; k < pixels.size(); ++k) {
        const double got = res.irradiance[k];
        // (a) The estimator: the mean over the pixel's sub-samples of the texel image filtered at
        // each sub-sample's own texture position with its footprint, (0.5 / sqrt(N))^2 px^2 per
        // axis mapped through the plane's projection (texels are boxes: exact integral, fp64).
        double estimate = 0.0;
        for (uint32_t j = 0; j < n_sub; ++j) {
          const Vec3f d = subsample_direction(view, view.cameras[0], pixels[k], j);
          const double xn = d.x / d.z, c = std::cos(theta), sn = std::sin(theta);
          const double sp = xn * kZ / (c + xn * sn);                       // plane coordinate hit
          const double x_tex = (sp + kHalf) / (2.0 * kHalf) * kTexW;       // texel i spans [i, i + 1)
          const double dtex_dpx = kTexW / (2.0 * kHalf) * kZ * c / ((c + xn * sn) * (c + xn * sn)) / kF;
          estimate += texel_image_filtered(lin, x_tex, 0.25 / n_sub * dtex_dpx * dtex_dpx);
        }
        estimate /= n_sub;
        worst_estimator = worse(worst_estimator, std::abs(got - estimate) / estimate);
        // (b) What it estimates: the exact integral of the texel image over the pixel response.
        const double px = pixels[k] % kW;
        double exact = 0.0;
        for (int i = 0; i < kTexW; ++i) {
          const double a = boundary_px(i) - px, b = boundary_px(i + 1) - px;
          if (b < -response.x_max() || a > response.x_max()) continue;
          exact += lin[static_cast<size_t>(i)] * (response.cdf(std::min(b, response.x_max())) -
                                                  response.cdf(std::max(a, -response.x_max())));
        }
        const double e = (got - exact) / exact;
        worst_integral = worse(worst_integral, std::abs(e));
        sq_integral += e * e;
      }
      const double rms_integral = std::sqrt(sq_integral / static_cast<double>(pixels.size()));
      MESSAGE(backend_name(backend) << ", plane tilt " << theta << " rad: max relative error against the estimator "
                                    << worst_estimator << "; against the pixel integral max " << worst_integral
                                    << ", RMS " << rms_integral);
      // fp32 arithmetic and the EWA's discretization (test_ewa.cpp: <= 1e-3 of the modulation).
      CHECK(worst_estimator <= 3e-4);
      // The quadrature error of 64 Owen-scrambled Sobol points on a 5-15 px period sinusoid.
      CHECK(worst_integral <= 0.01);
      CHECK(rms_integral <= 0.003);
    }
  }
}

namespace {

// Plane strip at z = 0 of its body frame, s in [s0, s1] (x) by t in [-h, h] (y), as one quad whose
// u runs from u0 to u1 and v from 0 to 1.
void add_quad(Mesh& m, double s0, double s1, double h, float u0, float u1) {
  const auto base = static_cast<uint32_t>(m.positions.size() / 3);
  const double xs[4] = {s0, s1, s1, s0}, ys[4] = {-h, -h, h, h};
  const float us[4] = {u0, u1, u1, u0}, vs[4] = {0.0f, 0.0f, 1.0f, 1.0f};
  for (int i = 0; i < 4; ++i) {
    m.positions.insert(m.positions.end(), {static_cast<float>(xs[i]), static_cast<float>(ys[i]), 0.0f});
    m.normals.insert(m.normals.end(), {0.0f, 0.0f, 1.0f});
    m.uvs.insert(m.uvs.end(), {us[i], vs[i]});
  }
  m.indices.insert(m.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
}

void set_texture(Material& m, std::vector<uint8_t> rgb, int w, int h) {
  m.texture = std::move(rgb);
  m.texture_width = w;
  m.texture_height = h;
  m.texture_channels = 3;
}

}  // namespace

TEST_CASE("oracle 6: a UV seam with gutters is invisible, static and as it crosses a moving pixel") {
  // A plane of two quads meeting at s = 0, rendered twice: (a) with one continuous chart, (b) with
  // the quads in two charts at different places of a wider atlas with a black gap between them;
  // each chart carries a gutter of G texels continuing the texture across the seam (the importer's
  // job, DESIGN §7.3). Both atlases hold the same texels at integer offsets, so every sub-sample
  // reads the same values in either; a UV mean across the seam would land in the black gap.
  constexpr int kTexels = 256, kGutter = 8, kGap = 128, kRows = 4;  // continuous atlas: 256 texels per m
  constexpr double kHalf = 0.5, kF = 64.0;                           // 4 texels per pixel at 1 m
  constexpr int kW = 64, kH = 16;
  std::vector<uint8_t> row(kTexels);
  for (int i = 0; i < kTexels; ++i) {
    const double sc = -kHalf + (i + 0.5) / kTexels;
    row[static_cast<size_t>(i)] = static_cast<uint8_t>(std::lround(120.0 + 90.0 * std::cos(2.0 * std::numbers::pi * sc / 0.13)));
  }
  auto atlas = [&](int width, const std::vector<std::pair<int, int>>& spans) {  // (atlas start, source start..)
    std::vector<uint8_t> t(static_cast<size_t>(width) * kRows * 3, 0);
    for (auto [dst, src_begin] : spans) {
      for (int i = 0; i < kTexels / 2 + kGutter; ++i) {
        for (int y = 0; y < kRows; ++y) {
          for (int c = 0; c < 3; ++c) t[(static_cast<size_t>(y) * width + dst + i) * 3 + c] = row[static_cast<size_t>(src_begin + i)];
        }
      }
    }
    return t;
  };
  // (a) continuous: u in [0, 1] over s in [-0.5, 0.5].
  Mesh continuous;
  add_quad(continuous, -kHalf, 0.0, 0.25, 0.0f, 0.5f);
  add_quad(continuous, 0.0, kHalf, 0.25, 0.5f, 1.0f);
  std::vector<uint8_t> tex_a(static_cast<size_t>(kTexels) * kRows * 3);
  for (int y = 0; y < kRows; ++y) {
    for (int i = 0; i < kTexels; ++i) {
      for (int c = 0; c < 3; ++c) tex_a[(static_cast<size_t>(y) * kTexels + i) * 3 + c] = row[static_cast<size_t>(i)];
    }
  }
  // (b) charts: left half (+ gutter on its right) at texel 0, right half (+ gutter on its left)
  // after a black gap; the right quad's u is the continuous u shifted by a whole number of texels.
  constexpr int kWide = kTexels / 2 + kGutter + kGap + kTexels / 2 + kGutter;
  constexpr int kRight = kTexels / 2 + kGutter + kGap + kGutter;  // atlas texel of s = 0 in chart B
  Mesh charts;
  add_quad(charts, -kHalf, 0.0, 0.25, 0.0f, static_cast<float>(kTexels / 2) / kWide);
  add_quad(charts, 0.0, kHalf, 0.25, static_cast<float>(kRight) / kWide, static_cast<float>(kRight + kTexels / 2) / kWide);
  const std::vector<uint8_t> tex_b = atlas(kWide, {{0, 0}, {kRight - kGutter, kTexels / 2 - kGutter}});

  // Camera translating along x at 0.125 m/s (8 px/s): the seam crosses pixels 30-33 within 250 ms.
  const Trajectory rig = moving_trajectory(kT0, kT1, quat_identity<double>(), {-0.016, 0, 0}, {0.125, 0, 0}, {0, 0, 0});
  auto bundle = [&](const Mesh& mesh, const std::vector<uint8_t>& tex, int width) {
    SceneBundle s;
    s.light_groups = {"default"};
    s.trajectories = {rig};
    s.cameras.push_back(pinhole(kW, kH, kF, 0));
    Body plane = make_body(1, mesh);
    set_texture(plane.materials[0], tex, width, kRows);
    plane.T_world_body = translation(0, 0, 1.0);
    s.bodies.push_back(plane);
    s.lights.push_back(ambient_light(1.0f));
    return s;
  };
  const SceneBundle a = bundle(continuous, tex_a, kTexels), b = bundle(charts, tex_b, kWide);
  // Static check: rows across the seam at the first time; moving check: 4 pixels next to the seam
  // at 250 times 1 ms apart.
  std::vector<uint16_t> cams;
  std::vector<uint32_t> pixels;
  std::vector<int64_t> ts;
  for (int y = 6; y < 10; ++y) {
    for (int x = 20; x < 44; ++x) {
      cams.push_back(0);
      pixels.push_back(static_cast<uint32_t>(y * kW + x));
      ts.push_back(kT0);
    }
  }
  for (int k = 0; k < 250; ++k) {
    for (int x = 30; x < 34; ++x) {
      cams.push_back(0);
      pixels.push_back(static_cast<uint32_t>(8 * kW + x));
      ts.push_back(kT0 + 1'000'000 * static_cast<int64_t>(k));
    }
  }
  for (BackendKind backend : backends()) {
    Renderer ra(a, options(backend)), rb(b, options(backend));
    const RenderResult ea = ra.render(cams, pixels, ts), eb = rb.render(cams, pixels, ts);
    double worst = 0.0, lo = 1e30, hi = 0.0;
    for (size_t k = 0; k < ea.count; ++k) {
      worst = worse(worst, std::abs(std::log(static_cast<double>(eb.irradiance[k]) / static_cast<double>(ea.irradiance[k]))));
      lo = std::min(lo, static_cast<double>(eb.irradiance[k]));
      hi = std::max(hi, static_cast<double>(eb.irradiance[k]));
    }
    // One surface and one material throughout: the signature never changes, so any irradiance step
    // the seam caused would be invisible to the integrator. The moving pixels' log irradiance must
    // be smooth: second differences over the 0.008 px steps stay at the curvature scale of the
    // texture (~2e-5), where a step of the size of the integrator's tolerance would show as 2e-3.
    bool signature_constant = true;
    for (size_t k = 96; k + 4 < ea.count; ++k) signature_constant &= eb.signature[k] == eb.signature[k + 4];
    double worst_d2 = 0.0;
    for (size_t k = 96 + 8; k < eb.count; ++k) {
      const double l0 = std::log(static_cast<double>(eb.irradiance[k - 8])),
                   l1 = std::log(static_cast<double>(eb.irradiance[k - 4])),
                   l2 = std::log(static_cast<double>(eb.irradiance[k]));
      worst_d2 = worse(worst_d2, std::abs(l2 - 2.0 * l1 + l0));
    }
    MESSAGE(backend_name(backend) << ": seam vs continuous chart, max |log ratio| " << worst << " (irradiance "
                                  << lo << " .. " << hi << "); moving pixels' max |second difference| of log "
                                  << "irradiance " << worst_d2);
    CHECK(worst <= 1e-4);  // fp32 UV rounding; the integrator's tolerance is 2e-3
    CHECK(worst_d2 <= 2e-4);
    CHECK(lo >= 0.99 * srgb_to_linear(*std::min_element(row.begin(), row.end())));  // nothing reads the gap
    CHECK(signature_constant);
  }
}

TEST_CASE("oracle 7: per-face colour cells (vertex-colour bakes) equal per-face materials") {
  // A 1 m plane of 40 x 40 quads (~1.6 px each) with one colour per face, in scan-like (shuffled)
  // face order, rendered (a) with Stream S's vertex-colour bake: every face its own 4 x 4 texel cell,
  // laid out in face order, all three corners at the cell centre (zero UV gradient), and (b)
  // untextured with one material per face. Per-sub-sample lookups read each sub-sample's own face
  // colour, so (a) equals (b) up to summation order; a single lookup per cluster would read the cell
  // at the mean UV of unrelated faces. This fragmented atlas also checks OptiX against Embree.
  constexpr int kQuads = 40, kCell = 4, kW = 64, kH = 64;
  const Mesh grid = plane_mesh(0.5, kQuads);
  const size_t faces = grid.indices.size() / 3;
  std::vector<size_t> order(faces);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937(17));
  int side = 1;
  while (static_cast<size_t>(side) * side < faces) side *= 2;
  const int tw = side * kCell;
  std::vector<uint8_t> cells(static_cast<size_t>(tw) * tw * 3, 0);
  Mesh baked, split;
  std::vector<Material> per_face;
  double lo = 1e30, hi = 0.0;
  for (size_t f = 0; f < faces; ++f) {
    const size_t src = order[f];
    double cx = 0.0, cy = 0.0;
    for (int c = 0; c < 3; ++c) {
      const uint32_t v = grid.indices[3 * src + static_cast<size_t>(c)];
      cx += static_cast<double>(grid.positions[3 * v]) / 3.0;
      cy += static_cast<double>(grid.positions[3 * v + 1]) / 3.0;
    }
    // A colour ramp along x with a ripple along y.
    const std::array<uint8_t, 3> rgb{static_cast<uint8_t>(std::lround(40 + 180 * (cx + 0.5))),
                                     static_cast<uint8_t>(std::lround(120 + 60 * std::sin(9.0 * cy))),
                                     static_cast<uint8_t>(std::lround(200 - 150 * (cx + 0.5)))};
    const int ci = static_cast<int>(f) % side, cj = static_cast<int>(f) / side;
    for (int y = 0; y < kCell; ++y) {
      for (int x = 0; x < kCell; ++x) {
        for (int c = 0; c < 3; ++c) cells[(static_cast<size_t>(cj * kCell + y) * tw + ci * kCell + x) * 3 + c] = rgb[static_cast<size_t>(c)];
      }
    }
    const float u = (static_cast<float>(ci) + 0.5f) / static_cast<float>(side), v = (static_cast<float>(cj) + 0.5f) / static_cast<float>(side);
    for (int c = 0; c < 3; ++c) {
      const uint32_t vi = grid.indices[3 * src + static_cast<size_t>(c)];
      for (Mesh* m : {&baked, &split}) {
        m->positions.insert(m->positions.end(), grid.positions.begin() + 3 * vi, grid.positions.begin() + 3 * vi + 3);
        m->normals.insert(m->normals.end(), {0.0f, 0.0f, 1.0f});
        m->indices.push_back(static_cast<uint32_t>(3 * f + static_cast<size_t>(c)));
      }
      baked.uvs.insert(baked.uvs.end(), {u, v});
    }
    split.material_ids.push_back(static_cast<uint16_t>(f));
    Material m;
    double mono = 0.0;
    for (int c = 0; c < 3; ++c) {
      m.base_color[static_cast<size_t>(c)] = static_cast<float>(srgb_to_linear(rgb[static_cast<size_t>(c)]));
      mono += static_cast<double>(m.base_color[static_cast<size_t>(c)]) / 3.0;
    }
    lo = std::min(lo, mono);
    hi = std::max(hi, mono);
    per_face.push_back(m);
  }
  auto bundle = [&](Body body) {
    SceneBundle s;
    s.light_groups = {"default"};
    s.trajectories = {static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0})};
    s.cameras.push_back(pinhole(kW, kH, 64.0, 0));
    body.T_world_body = pose_matrix(rot_y(0.3), {0, 0, 1.0});
    s.bodies.push_back(std::move(body));
    s.lights.push_back(ambient_light(1.0f));
    return s;
  };
  Body textured = make_body(1, baked);
  set_texture(textured.materials[0], cells, tw, tw);
  Body materials = make_body(1, split);
  materials.materials = per_face;
  const SceneBundle a = bundle(textured), b = bundle(materials);
  std::vector<uint16_t> cams;
  std::vector<uint32_t> pixels;
  std::vector<int64_t> ts;
  for (int y = 8; y < kH - 8; ++y) {
    for (int x = 12; x < kW - 12; ++x) {
      cams.push_back(0);
      pixels.push_back(static_cast<uint32_t>(y * kW + x));
      ts.push_back(kT0 + 1000);
    }
  }
  std::vector<RenderResult> baked_results;
  for (BackendKind backend : backends()) {
    Renderer ra(a, options(backend)), rb(b, options(backend));
    const RenderResult ea = ra.render(cams, pixels, ts), eb = rb.render(cams, pixels, ts);
    double worst = 0.0;
    for (size_t k = 0; k < ea.count; ++k) {
      const double x = ea.irradiance[k], y = eb.irradiance[k];
      worst = worse(worst, std::abs(x - y) / y);
    }
    MESSAGE(backend_name(backend) << ": face cells vs per-face materials, max relative difference " << worst);
    CHECK(worst <= 1e-5);
    baked_results.push_back(ea);
  }
  if (baked_results.size() == 2) {
    // Backends may break a shared-edge tie towards different faces; each such sub-sample moves the
    // value by at most the colour contrast / N, and only those pixel-samples differ.
    const RenderResult& e = baked_results[0];
    const RenderResult& o = baked_results[1];
    size_t equal = 0;
    double worst = 0.0;
    for (size_t k = 0; k < e.count; ++k) {
      equal += e.irradiance[k] == o.irradiance[k];
      worst = worse(worst, std::abs(static_cast<double>(e.irradiance[k]) - static_cast<double>(o.irradiance[k])));
    }
    MESSAGE("fragmented atlas: OptiX and Embree bitwise equal on " << equal << " of " << e.count
                                                                   << " pixel-samples, max difference " << worst);
    CHECK(equal >= e.count * 99 / 100);
    CHECK(worst <= 2.0 * (hi - lo) / 64.0);
  }
}

TEST_CASE("oracle 3: hard shadow edge of a moving occluder") {
  constexpr double kF = 100.0, kFloorZ = 3.0, kHx = 0.1, kHz = 0.005;
  constexpr int kW = 160, kH = 40;
  const Vec3<double> light{-1.0, 0.0, 0.5};
  const Quat<double> id = quat_identity<double>();
  SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories = {static_trajectory(kT0, kT1, id, {0, 0, 0}),
                    moving_trajectory(kT0, kT1, id, {-0.35, 0.0, 1.5}, {3.0, 0.0, 0.0}, {0, 0, 0})};
  s.cameras.push_back(pinhole(kW, kH, kF, 0));
  Body floor = make_body(1, plane_mesh(4.0, 2));
  floor.T_world_body = translation(0, 0, kFloorZ);
  s.bodies.push_back(floor);
  Body blade = make_body(2, box_mesh(static_cast<float>(kHx), 3.0f, static_cast<float>(kHz)));
  blade.trajectory = 1;
  s.bodies.push_back(blade);
  s.lights.push_back(point_light({light.x, light.y, light.z}, 9.0f));
  s.lights.push_back(ambient_light(0.1f));
  const rng::PixelResponse response(1.0, 0.15, 3.0);
  for (BackendKind backend : backends()) {
    Renderer r(s, options(backend));
    const int y = kH / 2;
    std::vector<double> edges;
    RenderResult previous;
    for (int64_t t : {kT0 + 40'000'000, kT0 + 90'000'000}) {
      // Analytic shadow edge: the blade's right edge nearest the light, projected from the light
      // onto the floor, then into the image.
      const Pose b = eval_pose(s.trajectories[1], t);
      const double xe = b.p.x + kHx, ze = b.p.z - kHz;
      const double xs = light.x + (xe - light.x) * (kFloorZ - light.z) / (ze - light.z);
      const double edge_px = 0.5 * (kW - 1) + kF * xs / kFloorZ;
      std::vector<uint16_t> cams(kW, 0);
      std::vector<uint32_t> pixels(kW);
      std::vector<int64_t> ts(kW, t);
      for (int x = 0; x < kW; ++x) pixels[static_cast<size_t>(x)] = static_cast<uint32_t>(y * kW + x);
      const RenderResult res = r.render(cams, pixels, ts);
      // Normalized lit fraction: (L - rho A) / (rho E(x)/pi), E from the analytic point light at the
      // pixel centre's floor point (the cluster shades at its centroid).
      auto lit_fraction = [&](int x) {
        const Vec3<double> p{kFloorZ * (x - 0.5 * (kW - 1)) / kF, kFloorZ * (y - 0.5 * (kH - 1)) / kF, kFloorZ};
        const Vec3<double> to = light - p;
        const double e = 9.0 * (-to.z / norm(to)) / dot(to, to);  // floor normal faces -z
        return (static_cast<double>(res.irradiance[static_cast<size_t>(x)]) - 0.1) / (e / std::numbers::pi);
      };
      // A straight edge at E lights the fraction H(x - E) of pixel x's footprint (H: CDF of the pixel
      // response). Fit E to the rendered profile; with N = 64 the staircase limits each pixel to
      // ~1/64 of coverage.
      const int x0 = static_cast<int>(std::floor(edge_px)) - 4;
      double worst_profile = 0.0, best_e = 0.0, best_err = 1e30;
      for (int x = x0; x < x0 + 10; ++x) {
        worst_profile = worse(worst_profile, std::abs(lit_fraction(x) - response.cdf(std::clamp(x - edge_px, -0.95, 0.95))));
      }
      for (double e = edge_px - 1.0; e <= edge_px + 1.0; e += 1e-3) {
        double err = 0.0;
        for (int x = x0; x < x0 + 10; ++x) err += std::pow(lit_fraction(x) - response.cdf(std::clamp(x - e, -0.95, 0.95)), 2);
        if (err < best_err) {
          best_err = err;
          best_e = e;
        }
      }
      MESSAGE(backend_name(backend) << ": shadow edge " << edge_px << " px, fitted " << best_e << " px, profile error "
                                    << worst_profile);
      CHECK(std::abs(best_e - edge_px) <= 0.02);
      CHECK(worst_profile <= 0.03);
      CHECK(lit_fraction(x0 - 6) == doctest::Approx(0.0).scale(1.0).epsilon(1e-6));  // umbra: ambient only
      CHECK(lit_fraction(x0 + 16) == doctest::Approx(1.0).epsilon(2e-3));              // fully lit
      edges.push_back(edge_px);
      if (!previous.signature.empty()) {
        // Static gate: pixels the occluder and its shadow never reached are bitwise identical.
        for (int x = kW - 8; x < kW; ++x) {
          CHECK(res.irradiance[static_cast<size_t>(x)] == previous.irradiance[static_cast<size_t>(x)]);
          CHECK(res.signature[static_cast<size_t>(x)] == previous.signature[static_cast<size_t>(x)]);
        }
      }
      previous = res;
    }
    CHECK(edges[1] - edges[0] > 10.0);  // the shadow moved with the occluder
  }
}

namespace {

// Floor and wall (static, textured), four movers on independent paths, point, spot and
// directional lights with shadows plus ambient, two light groups.
SceneBundle mixed_scene() {
  SceneBundle s;
  s.light_groups = {"default", "mains"};
  s.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.05, 0.02, 0.0}), {0.0, -0.3, 0.0},
                                             {0.1, 0.05, 0.2}, {0.05, 0.3, -0.1}));
  const std::vector<std::pair<Vec3<double>, Vec3<double>>> paths = {
      {{-0.4, 0.1, 1.6}, {0.6, -0.1, 0.0}}, {{0.3, -0.2, 2.0}, {-0.4, 0.3, 0.2}},
      {{0.0, 0.3, 1.2}, {0.1, -0.5, 0.3}}, {{0.5, 0.2, 2.4}, {-0.8, 0.0, -0.2}}};
  for (size_t i = 0; i < paths.size(); ++i) {
    s.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.3 * i, -0.2, 0.1 * i}),
                                               paths[i].first, paths[i].second,
                                               {1.5 - 0.7 * i, 2.0 + 0.5 * i, -1.0 + 0.3 * i}));
  }
  s.cameras.push_back(pinhole(120, 90, 90.0, 0));
  // Static world: textured floor (checker with fine detail) and a wall.
  std::vector<uint8_t> tex(256 * 256 * 3);
  for (int j = 0; j < 256; ++j) {
    for (int i = 0; i < 256; ++i) {
      const uint8_t v = static_cast<uint8_t>(((i / 16 + j / 16) % 2) ? 220 : 40 + (i * 7 + j * 3) % 50);
      for (int c = 0; c < 3; ++c) tex[(static_cast<size_t>(j) * 256 + i) * 3 + c] = static_cast<uint8_t>(v - 10 * c);
    }
  }
  Body floor = make_body(1, plane_mesh(3.0, 16), {0.9f, 0.8f, 0.7f});
  floor.materials[0].texture = tex;
  floor.materials[0].texture_width = floor.materials[0].texture_height = 256;
  floor.materials[0].texture_channels = 3;
  floor.T_world_body = pose_matrix(Mat3<double>{{{1, 0, 0}, {0, 0, -1}, {0, 1, 0}}}, {0, 0.6, 1.8});
  s.bodies.push_back(floor);
  Body wall = make_body(2, plane_mesh(3.0, 3), {0.5f, 0.6f, 0.7f});
  wall.T_world_body = translation(0, 0, 3.2);
  s.bodies.push_back(wall);
  for (uint32_t i = 0; i < 4; ++i) {
    Body b = make_body(10 + i, box_mesh(0.08f + 0.02f * static_cast<float>(i), 0.06f, 0.1f),
                       {0.3f + 0.15f * static_cast<float>(i), 0.5f, 0.4f});
    b.materials[0].emissive = {i == 2 ? 0.2f : 0.0f, 0.0f, 0.0f};
    b.trajectory = static_cast<int32_t>(i) + 1;
    s.bodies.push_back(b);
  }
  // A static body whose id sorts after the movers' (the static world and the movers interleave).
  Body pillar = make_body(60, box_mesh(0.05f, 0.6f, 0.05f), {0.9f, 0.9f, 0.9f});
  pillar.T_world_body = translation(0.45, 0.0, 2.6);
  s.bodies.push_back(pillar);
  s.lights.push_back(point_light({-0.6, -0.8, 0.6}, 3.0f));
  Light spot;
  spot.kind = BundleLightKind::kSpot;
  spot.group = 1;
  spot.position = {0.8, -0.9, 0.4};
  spot.direction = {-0.3, 0.8, 0.6};
  spot.spot_inner_rad = 0.3f;
  spot.spot_outer_rad = 0.6f;
  spot.intensity = 4.0f;
  s.lights.push_back(spot);
  Light sun;
  sun.kind = BundleLightKind::kDirectional;
  sun.direction = {0.3, 0.9, 0.3};
  sun.intensity = 0.5f;
  sun.casts_shadows = true;
  s.lights.push_back(sun);
  Light fill = point_light({0.0, -0.5, 0.0}, 0.5f, false, 1);
  fill.body = 4;  // carried by the mover bodies[4] (id 12), no shadows
  s.lights.push_back(fill);
  s.lights.push_back(ambient_light(0.05f));
  s.lights.push_back(ambient_light(0.02f, 1));
  return s;
}

Requests random_requests(size_t n, uint32_t seed) {
  Requests q;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> pix(0, 120 * 90 - 1);
  std::uniform_int_distribution<int64_t> t(kT0, kT1);
  // Times on a coarse lattice so that pose slots are shared, as in the integrator's rounds.
  for (size_t i = 0; i < n; ++i) {
    q.cam.push_back(0);
    q.pixel.push_back(pix(rng));
    q.t.push_back(kT0 + (t(rng) - kT0) / 65536 * 65536);
  }
  return q;
}

}  // namespace

TEST_CASE("oracle 4: OptiX agrees with Embree on a mixed scene (DESIGN §12.1 L2)") {
  if (!optix_available()) {
    MESSAGE("OptiX unavailable: parity test skipped");
    return;
  }
  const SceneBundle s = mixed_scene();
  const Requests q = random_requests(20000, 21);
  Renderer em(s, options(BackendKind::kEmbree, 6));
  Renderer ox(s, options(BackendKind::kOptix));
  const RenderFlags flags{true, true};
  const RenderResult a = em.render(q.cam, q.pixel, q.t, flags);
  const RenderResult b = ox.render(q.cam, q.pixel, q.t, flags);
  size_t same_sub = 0, compared = 0, centre_same = 0;
  double worst_irr = 0.0, worst_depth = 0.0;
  for (size_t i = 0; i < a.sub_instance.size(); ++i) same_sub += a.sub_instance[i] == b.sub_instance[i];
  for (size_t r = 0; r < a.count; ++r) {
    if (a.signature[r] == b.signature[r]) {
      ++compared;
      for (int g = 0; g < a.num_groups; ++g) {
        const double x = a.irradiance[r * 2 + static_cast<size_t>(g)], y = b.irradiance[r * 2 + static_cast<size_t>(g)];
        worst_irr = worse(worst_irr, std::abs(x - y) / std::max(std::abs(x), 1e-3));
      }
    }
    if (a.centre[r].instance == b.centre[r].instance) {
      ++centre_same;
      if (a.centre[r].instance != 0) {
        worst_depth = worse(worst_depth, static_cast<double>(std::abs(a.centre[r].depth - b.centre[r].depth) / a.centre[r].depth));
      }
    }
  }
  const double agreement = static_cast<double>(same_sub) / static_cast<double>(a.sub_instance.size());
  MESSAGE("sub-sample instance agreement " << agreement << ", irradiance max rel diff " << worst_irr << " over "
                                           << compared << " pixel-samples without flips");
  CHECK(agreement >= 0.9999);
  CHECK(compared >= a.count * 99 / 100);
  CHECK(worst_irr <= 1e-4);
  CHECK(centre_same >= a.count * 9999 / 10000);
  CHECK(worst_depth <= 1e-5);
}

TEST_CASE("oracle 5: results do not depend on batch size, thread count or request order") {
  const SceneBundle s = mixed_scene();
  const Requests q = random_requests(3000, 31);
  std::vector<size_t> perm(q.cam.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(5));
  Requests p;
  for (size_t i : perm) {
    p.cam.push_back(q.cam[i]);
    p.pixel.push_back(q.pixel[i]);
    p.t.push_back(q.t[i]);
  }
  const RenderFlags flags{true, true};
  for (BackendKind backend : backends()) {
    RendererOptions base = options(backend, 6);
    Renderer reference(s, base);
    const RenderResult ref = reference.render(q.cam, q.pixel, q.t, flags);
    // Every body is rendered, including the static one whose id sorts after the movers'.
    for (uint32_t id : {1u, 2u, 10u, 11u, 12u, 13u, 60u}) {
      CHECK(std::count(ref.sub_instance.begin(), ref.sub_instance.end(), id) > 0);
    }
    const RenderResult again = reference.render(q.cam, q.pixel, q.t, flags);  // warm pose cache
    CHECK(again.irradiance == ref.irradiance);
    CHECK(again.signature == ref.signature);
    for (uint64_t rays : {uint64_t{64} * 13, uint64_t{1000}, uint64_t{1} << 18}) {
      RendererOptions o = options(backend, backend == BackendKind::kEmbree ? 1 : 6);
      o.max_rays_per_launch = rays;
      if (rays == 1000) o.pose_cache_bytes = 1;  // the pose cache holds one launch: render()'s groups reset it
      Renderer r(s, o);
      r.set_window(kT0 + 123);
      const RenderResult res = r.render(p.cam, p.pixel, p.t, flags);
      size_t mismatches = 0;
      for (size_t k = 0; k < perm.size(); ++k) {
        const size_t i = perm[k];
        bool same = res.signature[k] == ref.signature[i] && res.centre[k].instance == ref.centre[i].instance &&
                    std::memcmp(&res.centre[k], &ref.centre[i], sizeof(GtSample)) == 0;
        for (int g = 0; g < res.num_groups; ++g) same &= res.irradiance[k * 2 + static_cast<size_t>(g)] == ref.irradiance[i * 2 + static_cast<size_t>(g)];
        for (uint32_t j = 0; j < res.subsamples; ++j) same &= res.sub_instance[k * res.subsamples + j] == ref.sub_instance[i * res.subsamples + j];
        mismatches += !same;
      }
      INFO(backend_name(backend) << ", " << rays << " rays per launch");
      CHECK(mismatches == 0);
    }
  }
}

TEST_CASE("plan-pass shadow states (ENGINE_CPP I8): the movers blocking each centre ray's shadow ray, on Embree") {
  const SceneBundle s = mixed_scene();
  Renderer r(s, options(BackendKind::kEmbree, 6));
  r.build();
  const int32_t bits = r.num_shadow_lights();
  REQUIRE(bits == 3);
  const size_t n = 120 * 90;
  std::vector<GtQuery> q(n);
  for (size_t i = 0; i < n; ++i) q[i] = {kT0 + 50'000'000, static_cast<float>(i % 120), static_cast<float>(i / 120)};
  std::vector<GtSample> plain(n), with(n);
  std::vector<uint32_t> states(2 * n * static_cast<size_t>(bits), 99);
  r.gt_into(0, q.data(), n, plain.data(), nullptr);
  r.gt_into(0, q.data(), n, with.data(), nullptr, states.data());
  CHECK(std::memcmp(plain.data(), with.data(), n * sizeof(GtSample)) == 0);  // the states change no ground truth
  size_t movers = 0, lit = 0, multi = 0;
  for (size_t i = 0; i < n; ++i) {
    for (int32_t b = 0; b < bits; ++b) {
      const uint32_t word = states[2 * (i * static_cast<size_t>(bits) + static_cast<size_t>(b))];
      const uint32_t st = word & ~kShadowMulti;  // the I8 review: another mover blocks the ray too
      float t;
      std::memcpy(&t, &states[2 * (i * static_cast<size_t>(bits) + static_cast<size_t>(b)) + 1], sizeof t);
      CHECK((st == kShadowLit || st == kShadowStatic || (st >= kShadowMover + 10 && st <= kShadowMover + 13)));
      CHECK((word == st || st >= kShadowMover));  // only a mover's state carries the flag
      if (with[i].instance == kInstanceNone) CHECK(st == kShadowStatic);  // no receiver
      if (st >= kShadowMover) CHECK((t > 0.0f && (b == 2 || t < 1.0f)));  // point and spot lights: towards the light
      movers += st >= kShadowMover;
      lit += st == kShadowLit;
      multi += word != st;
    }
  }
  MESSAGE("states: " << movers << " blocked by movers (" << multi << " by more than one), " << lit << " lit of "
                     << n * static_cast<size_t>(bits));
  CHECK(movers > 100);
  CHECK(multi > 0);  // 20: two of the scene's movers in line with a light (OptiX: test_device.cpp's equal states)
  CHECK(lit > n);
}

TEST_CASE("plan-pass normals (P2-D D1): each hit triangle's unit geometric normal in the camera frame, on Embree") {
  // The mixed scene's camera moves and turns; its floor and wall are planes at right angles: every wall pixel has one
  // normal, every floor pixel another, orthogonal to it; every normal is unit, NaN exactly on a miss, and the ground
  // truth does not change with the output. Facing the camera: the hit point x = depth * d / d.z lies on the camera's
  // side of its plane, n . x < 0.
  const SceneBundle s = mixed_scene();
  Renderer r(s, options(BackendKind::kEmbree, 6));
  r.build();
  const size_t w = 120, h = 90, n = w * h;
  for (const int64_t t : {kT0 + 20'000'000, kT0 + 50'000'000}) {
    std::vector<GtQuery> q(n);
    for (size_t i = 0; i < n; ++i) q[i] = {t, static_cast<float>(i % w), static_cast<float>(i / w)};
    std::vector<GtSample> plain(n), with(n);
    std::vector<float> normal(3 * n, 7.0f);
    r.gt_into(0, q.data(), n, plain.data(), nullptr);
    r.gt_into(0, q.data(), n, with.data(), nullptr, nullptr, normal.data());
    CHECK(std::memcmp(plain.data(), with.data(), n * sizeof(GtSample)) == 0);
    std::vector<float> wall, floor_n;
    size_t misses = 0, hits = 0, away = 0;
    for (size_t i = 0; i < n; ++i) {
      const float* v = &normal[3 * i];
      if (with[i].instance == kInstanceNone) {
        CHECK((std::isnan(v[0]) && std::isnan(v[1]) && std::isnan(v[2])));
        ++misses;
        continue;
      }
      ++hits;
      CHECK(std::fabs(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - 1.0f) < 1e-5f);
      // The pinhole's ray through pixel (x, y) is ((x - cx) / f, (y - cy) / f, 1): the hit at depth * ray.
      const float cx = 59.5f, cy = 44.5f, f = 90.0f;
      const float px = (static_cast<float>(i % w) - cx) / f, py = (static_cast<float>(i / w) - cy) / f;
      away += v[0] * px + v[1] * py + v[2] > 1e-4f ? 1 : 0;
      if (with[i].instance == 2 && wall.empty()) wall.assign(v, v + 3);
      if (with[i].instance == 1 && floor_n.empty()) floor_n.assign(v, v + 3);
      const std::vector<float>* ref = with[i].instance == 2 ? &wall : (with[i].instance == 1 ? &floor_n : nullptr);
      if (ref != nullptr) {
        CHECK(std::fabs(v[0] - (*ref)[0]) + std::fabs(v[1] - (*ref)[1]) + std::fabs(v[2] - (*ref)[2]) < 1e-5f);
      }
    }
    REQUIRE((wall.size() == 3 && floor_n.size() == 3));
    CHECK(std::fabs(wall[0] * floor_n[0] + wall[1] * floor_n[1] + wall[2] * floor_n[2]) < 1e-5f);
    CHECK(away == 0);
    CHECK(hits > n / 2);
  }
  // Without the floor and the wall most centre rays miss: NaN there, unit normals on the boxes.
  SceneBundle sparse = mixed_scene();
  sparse.bodies.erase(sparse.bodies.begin(), sparse.bodies.begin() + 2);
  Renderer rs(sparse, options(BackendKind::kEmbree, 6));
  std::vector<GtQuery> q(n);
  for (size_t i = 0; i < n; ++i) q[i] = {kT0 + 50'000'000, static_cast<float>(i % w), static_cast<float>(i / w)};
  std::vector<GtSample> g(n);
  std::vector<float> normal(3 * n, 7.0f);
  rs.gt_into(0, q.data(), n, g.data(), nullptr, nullptr, normal.data());
  size_t misses = 0;
  for (size_t i = 0; i < n; ++i) {
    const float* v = &normal[3 * i];
    const bool miss = g[i].instance == kInstanceNone;
    misses += miss ? 1 : 0;
    CHECK((miss ? std::isnan(v[0]) && std::isnan(v[1]) && std::isnan(v[2])
                : std::fabs(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - 1.0f) < 1e-5f));
  }
  CHECK((misses > n / 2 && misses < n));
}

TEST_CASE("two cameras: each request uses its own camera") {
  // Camera 1 has its own rig (static, valid for the first 100 ms only), size, focal length and mounting. Requests
  // of both cameras interleave in one render(): pose-cache keys, the centre flag's per-camera grouping and width,
  // pixel and time checks must all follow each request's own camera.
  SceneBundle s = mixed_scene();
  constexpr int64_t kSpan1 = 100'000'000;
  constexpr int kW1 = 64, kH1 = 48;
  s.trajectories.push_back(
      static_trajectory(kT0, kT0 + kSpan1, so3::exp(Vec3<double>{0.0, 0.05, 0.02}), {0.1, -0.1, 0.0}));
  Camera c1 = pinhole(kW1, kH1, 70.0, static_cast<int>(s.trajectories.size()) - 1);
  c1.name = "cam1";
  c1.T_rig_cam = translation(0.15, -0.1, 0.2);
  s.cameras.push_back(c1);
  // The same camera 1 next to another camera 0: camera 1's results depend only on itself and its index.
  SceneBundle other = s;
  other.cameras[0] = pinhole(32, 24, 40.0, 0);
  other.cameras[0].T_rig_cam = translation(-0.2, 0.1, 0.0);
  std::mt19937 rng(41);
  Requests q, q0, q1, same0;  // interleaved, camera 0's, camera 1's, camera 1's pixels and times on camera 0
  for (int i = 0; i < 2000; ++i) {
    const auto cam = static_cast<uint16_t>(rng() % 2);
    const uint32_t pixel = static_cast<uint32_t>(rng() % (cam == 0 ? 120 * 90 : kW1 * kH1));
    const int64_t t = kT0 + static_cast<int64_t>(rng() % static_cast<uint64_t>(cam == 0 ? kT1 - kT0 : kSpan1)) / 65536 * 65536;
    for (Requests* r : {&q, cam == 0 ? &q0 : &q1}) {
      r->cam.push_back(cam);
      r->pixel.push_back(pixel);
      r->t.push_back(t);
    }
    if (cam == 1) {
      same0.cam.push_back(0);
      same0.pixel.push_back(pixel);
      same0.t.push_back(t);
    }
  }
  const RenderFlags flags{true, true};
  // Request k of `a` equals request i of `b`, bit for bit.
  auto same = [](const RenderResult& a, size_t k, const RenderResult& b, size_t i) {
    const auto g = static_cast<size_t>(a.num_groups), n = static_cast<size_t>(a.subsamples);
    return std::memcmp(&a.irradiance[k * g], &b.irradiance[i * g], g * sizeof(float)) == 0 && a.signature[k] == b.signature[i] &&
           std::memcmp(&a.sub_instance[k * n], &b.sub_instance[i * n], n * sizeof(uint32_t)) == 0 &&
           std::memcmp(&a.centre[k], &b.centre[i], sizeof(GtSample)) == 0;
  };
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    Renderer r(s, options(backend, 6));
    const RenderResult mixed = r.render(q.cam, q.pixel, q.t, flags);
    const RenderResult only0 = r.render(q0.cam, q0.pixel, q0.t, flags), only1 = r.render(q1.cam, q1.pixel, q1.t, flags);
    const RenderResult beside = Renderer(other, options(backend, 6)).render(q1.cam, q1.pixel, q1.t, flags);
    const RenderResult on0 = r.render(same0.cam, same0.pixel, same0.t, flags);
    std::vector<float> x, y;
    for (uint32_t p : q1.pixel) {
      x.push_back(static_cast<float>(p % kW1));
      y.push_back(static_cast<float>(p / kW1));
    }
    const std::vector<GtSample> gt1 = r.gt(q1.cam, x, y, q1.t);
    size_t mismatches = 0, k0 = 0, k1 = 0, differ = 0, centre_mismatches = 0;
    for (size_t k = 0; k < q.cam.size(); ++k) mismatches += !same(mixed, k, q.cam[k] == 0 ? only0 : only1, q.cam[k] == 0 ? k0++ : k1++);
    for (size_t i = 0; i < q1.cam.size(); ++i) {
      mismatches += !same(only1, i, beside, i);
      centre_mismatches += std::memcmp(&only1.centre[i], &gt1[i], sizeof(GtSample)) != 0;
      differ += !same(only1, i, on0, i);
    }
    CHECK(mismatches == 0);
    CHECK(centre_mismatches == 0);
    const auto movers1 = std::count_if(only1.sub_instance.begin(), only1.sub_instance.end(),
                                       [](uint32_t id) { return id >= 10 && id <= 13; });
    MESSAGE(backend_name(backend) << ": " << q1.cam.size() << " camera-1 requests, " << differ
                                  << " differ from camera 0's at the same pixel and time; " << movers1 << " mover sub-samples");
    CHECK(differ > 0);
    CHECK(movers1 > 0);  // camera 1 sees movers: their rows are per camera
    const std::vector<uint16_t> cam1{1};
    CHECK_THROWS_AS(r.render(cam1, std::vector<uint32_t>{kW1 * kH1}, std::vector<int64_t>{kT0}), std::invalid_argument);
    CHECK(r.render(std::vector<uint16_t>{0}, std::vector<uint32_t>{kW1 * kH1}, std::vector<int64_t>{kT0}).count == 1);
    const std::vector<int64_t> late{kT0 + 2 * kSpan1};
    CHECK(r.render(std::vector<uint16_t>{0}, std::vector<uint32_t>{5}, late).count == 1);
    CHECK_THROWS_AS(r.render(cam1, std::vector<uint32_t>{5}, late), std::invalid_argument);
  }
}

TEST_CASE("render_into and gt_into on host pointers (Embree) equal render and gt; hold ends the window") {
  // The engine's entry points on the host instantiation (ENGINE_CPP §1.7): window pose rows, out_index rows
  // over several chunks, the stage timers, the census's class routing (P2-B B0), gt_into's rows per run of equal
  // times. OptiX: test_device.cpp.
  RendererOptions o = options(BackendKind::kEmbree, 6);
  o.max_rays_per_launch = uint64_t{1} << 14;
  o.census = true;
  Renderer r(mixed_scene(), o);
  const int64_t t_w = time::window_start(time::window_index(kT0) + 3);
  const WindowClock clock{t_w, 10, 0, 0};
  std::mt19937 rng(23);
  const size_t n = 3000, groups = static_cast<size_t>(r.num_groups());
  std::vector<uint16_t> cam(n, 0);
  std::vector<uint32_t> pixel(n), out_index(n);
  std::vector<int64_t> t(n);
  std::vector<SampleRequest> requests(n);
  for (size_t i = 0; i < n; ++i) {
    pixel[i] = static_cast<uint32_t>(rng() % (120 * 90));
    const auto slot = static_cast<uint32_t>(rng() % 8193);
    t[i] = slot_time(clock, slot);
    requests[i] = {0, 0, pixel[i], 0, r.options().scene.n_subsamples, 0, slot};
    out_index[i] = static_cast<uint32_t>(i);
  }
  std::shuffle(out_index.begin(), out_index.end(), rng);
  std::vector<SampleRequest> own = requests;  // the census's expectation below: slots, before the renderer's rows
  const std::vector<SampleRequest> slots = requests;  // the clock's slots (the census's executor assigns own's rows)
  const RenderResult ref = r.render(cam, pixel, t);
  std::vector<float> irr(n * groups);
  std::vector<uint64_t> sig(n);
  std::vector<uint32_t> klass(n, 0);
  CHECK_THROWS_AS(r.render_into(requests.data(), n, nullptr, irr.data(), sig.data(), nullptr), std::logic_error);
  r.begin_window(0, clock, nullptr);
  // A refused call (a slot out of range after new ones) changes neither its requests nor the window table: the
  // rows assigned below are exact.
  std::vector<SampleRequest> refused(requests.begin(), requests.begin() + 100);
  refused.push_back({0, 0, 0, 0, r.options().scene.n_subsamples, 0, num_slots(clock)});
  CHECK_THROWS_AS(r.assign_pose_rows(refused.data(), refused.size(), nullptr), std::invalid_argument);
  for (size_t i = 0; i < 100; ++i) REQUIRE(refused[i].pose_slot == requests[i].pose_slot);
  r.assign_pose_rows(requests.data(), n, nullptr);
  r.render_into(requests.data(), n, out_index.data(), irr.data(), sig.data(), nullptr, klass.data());
  const RenderStats stats = r.take_render_stats();
  CHECK(stats.trace_ms > 0.0);
  CHECK(stats.shade_ms > 0.0);
  for (size_t i = 0; i < n; ++i) {
    REQUIRE(std::memcmp(&irr[out_index[i] * groups], &ref.irradiance[i * groups], groups * sizeof(float)) == 0);
    REQUIRE(sig[out_index[i]] == ref.signature[i]);
  }
  {
    // The census over chunks with out_index and class words (the engine's path): request i's counts go to the class
    // of its word at row out_index[i]. The expectation: the same requests traced in one batch by an executor of its
    // own, routed by the words render_into wrote.
    const SceneData data(mixed_scene(), o.scene);
    ThreadPool pool(4);
    HostExecutor host(pool);
    host.upload_scene(data);
    EmbreeTracer tracer(pool);
    tracer.build(data, host.scene_view());
    host.begin_window(0, clock, nullptr);
    host.assign_pose_rows(own.data(), static_cast<uint32_t>(n), nullptr);
    const uint32_t n_sub = o.scene.n_subsamples;
    const TraceBatch batch{own.data(), static_cast<uint32_t>(n), n_sub, host.window_poses(), nullptr, 0};
    std::vector<HitRecord> hits(n * n_sub);
    tracer.sample(batch, hits.data(), nullptr);
    uint64_t want[kCensusWords] = {};
    size_t band = 0;
    for (size_t i = 0; i < n; ++i) {
      const uint32_t* word = &klass[out_index[i]];
      census_add(want, own[i], census_sample(host.scene_view(), batch.poses, own[i], hits.data() + i * n_sub), word);
      band += (*word & (kClassMulti | kClassShadowEdge)) != 0;
    }
    MESSAGE("census: " << band << " of " << n << " band words; reusable band " << want[kCensusBand + kCensusReusable]
                       << ", rest " << want[kCensusRest + kCensusReusable] << "; mismatched band "
                       << want[kCensusBand + kCensusMismatched] << ", rest " << want[kCensusRest + kCensusMismatched]);
    CHECK(stats.census_on);
    for (int w = 0; w < kCensusWords; ++w) CHECK(stats.census[static_cast<size_t>(w)] == want[w]);
    CHECK(band > 0);
    CHECK(band < n);
    CHECK(want[kCensusBand + kCensusReusable] > 0);
    CHECK(want[kCensusRest + kCensusReusable] > 0);
    // render() writes no class words: the same counts, all in the rest (its own take: the call's totals only).
    CHECK(ref.stats.census_on);
    for (int w = 0; w < kCensusRest - kCensusBand; ++w) {  // the words of a class
      CHECK(ref.stats.census[static_cast<size_t>(kCensusRest + w)] == want[kCensusBand + w] + want[kCensusRest + w]);
      CHECK(ref.stats.census[static_cast<size_t>(kCensusBand + w)] == 0);
    }
  }
  {
    // One call renders one N (ENGINE_CPP P2-B B2): the requests at 4 N through render_into(subsamples 4 N) equal
    // render() of a renderer at 4 N (launches of a quarter the requests); on the host a request of another N, or an N
    // above kMaxSubsamples, is refused.
    const uint32_t big = 4 * o.scene.n_subsamples;
    RendererOptions o4 = o;
    o4.scene.n_subsamples = big;
    o4.census = false;
    Renderer r4(mixed_scene(), o4);
    const RenderResult ref4 = r4.render(cam, pixel, t);
    std::vector<SampleRequest> at4 = slots;
    for (SampleRequest& q : at4) q.k_count = big;
    r.assign_pose_rows(at4.data(), n, nullptr);
    std::vector<float> irr4(n * groups);
    std::vector<uint64_t> sig4(n);
    r.render_into(at4.data(), n, out_index.data(), irr4.data(), sig4.data(), nullptr, nullptr, big);
    r.take_render_stats();
    for (size_t i = 0; i < n; ++i) {
      REQUIRE(std::memcmp(&irr4[out_index[i] * groups], &ref4.irradiance[i * groups], groups * sizeof(float)) == 0);
      REQUIRE(sig4[out_index[i]] == ref4.signature[i]);
    }
    CHECK(std::memcmp(irr4.data(), irr.data(), irr.size() * sizeof(float)) != 0);  // another N, other values
    at4[17].k_count = o.scene.n_subsamples;
    CHECK_THROWS_AS(r.render_into(at4.data(), n, nullptr, irr4.data(), sig4.data(), nullptr, nullptr, big),
                    std::invalid_argument);
    CHECK_THROWS_AS(r.render_into(at4.data(), n, nullptr, irr4.data(), sig4.data(), nullptr, nullptr,
                                  static_cast<uint32_t>(kMaxSubsamples) + 1),
                    std::invalid_argument);
  }
  std::vector<float> x(n), y(n);
  std::vector<GtQuery> queries(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>(rng() % 12000) * 0.01f - 0.4f;
    y[i] = static_cast<float>(rng() % 9000) * 0.01f - 0.4f;
    t[i] = i % 7 == 0 ? kT0 + static_cast<int64_t>(rng() % 200'000'000) : t[i - 1];
    queries[i] = {t[i], x[i], y[i]};
  }
  const std::vector<GtSample> gref = r.gt(cam, x, y, t);
  std::vector<GtSample> g(n);
  r.gt_into(0, queries.data(), n, g.data(), nullptr);
  CHECK(std::memcmp(g.data(), gref.data(), n * sizeof(GtSample)) == 0);
  // hold() ends the window: its rows have the old stride.
  REQUIRE(r.hold(std::vector<uint32_t>{11}, t_w));
  CHECK_THROWS_AS(r.render_into(requests.data(), n, nullptr, irr.data(), sig.data(), nullptr), std::logic_error);
  CHECK_THROWS_AS(r.assign_pose_rows(requests.data(), n, nullptr), std::logic_error);
}

TEST_CASE("held movers trace as part of the static world (ENGINE_CPP I3b)") {
  // Two of the mixed scene's movers stand still at rotated poses, with ids after every static body's (61, 63).
  // hold() must give a bundle with them static at the hold pose bit for bit (the same world arrays), keep the
  // other movers, rebuild only when the set changes, and an empty set must restore the movers.
  SceneBundle s = mixed_scene();
  s.trajectories[2] = static_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.3, -0.2, 0.1}), {0.3, -0.2, 2.0});
  s.trajectories[4] = static_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.9, -0.2, 0.3}), {0.25, 0.15, 1.9});
  s.bodies[3].id = 61;  // trajectory 2
  s.bodies[5].id = 63;  // trajectory 4
  SceneBundle st = s;
  for (const size_t b : {size_t{3}, size_t{5}}) {
    const Pose p = eval_pose(s.trajectories[static_cast<size_t>(*s.bodies[b].trajectory)], kT0);
    st.bodies[b].T_world_body = pose_matrix(p.r, p.p);
    st.bodies[b].trajectory.reset();
  }
  Requests q;
  for (const int64_t t : {kT0 + 1'000'000, kT0 + 120'000'000}) {
    for (uint32_t p = 0; p < 120 * 90; ++p) {
      q.cam.push_back(0);
      q.pixel.push_back(p);
      q.t.push_back(t);
    }
  }
  const RenderFlags flags{true, true};
  auto same = [](const RenderResult& a, const RenderResult& b) {
    return a.irradiance == b.irradiance && a.signature == b.signature && a.sub_instance == b.sub_instance &&
           std::memcmp(a.centre.data(), b.centre.data(), a.centre.size() * sizeof(GtSample)) == 0;
  };
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    Renderer r(s, options(backend, 6));
    const RenderResult movers = r.render(q.cam, q.pixel, q.t, flags);
    const std::vector<uint32_t> held{63, 61, 999, 1};  // unknown and static ids are ignored
    CHECK(r.hold(held, kT0));
    CHECK_FALSE(r.hold(std::vector<uint32_t>{61, 63}, kT0 + 7'000'000));  // the same set and poses
    const RenderResult merged = r.render(q.cam, q.pixel, q.t, flags);
    Renderer ref(st, options(backend, 6));
    const RenderResult stat = ref.render(q.cam, q.pixel, q.t, flags);
    CHECK(same(merged, stat));
    for (uint32_t id : {10u, 12u, 60u, 61u, 63u}) {
      CHECK(std::count(merged.sub_instance.begin(), merged.sub_instance.end(), id) > 0);
    }
    CHECK(r.hold({}, 0));
    CHECK(same(r.render(q.cam, q.pixel, q.t, flags), movers));
  }
}

TEST_CASE("held sets that shrink, grow and add normals and UVs (ENGINE_CPP I3b)") {
  // The textured floor (normals and UVs) and the wall become still movers with ids after every static body's,
  // which leaves the pillar (neither) as the static world. Holding the floor adds normals and UVs (zero for the
  // pillar), holding a box adds neither; the merged world keeps the world's arrays and replaces the tail, so
  // each set must equal its static bundle bit for bit whatever set came before, and the movers come back after
  // the world's arrays grew.
  SceneBundle s = mixed_scene();
  const auto n = static_cast<int32_t>(s.trajectories.size());
  s.trajectories.push_back(static_trajectory(kT0, kT1, so3::exp(Vec3<double>{std::numbers::pi / 2, 0.0, 0.0}), {0.0, 0.6, 1.8}));
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0.0, 0.0, 3.2}));
  s.trajectories[2] = static_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.3, -0.2, 0.1}), {0.3, -0.2, 2.0});
  s.bodies[0].id = 70;  // floor
  s.bodies[0].trajectory = n;
  s.bodies[1].id = 71;  // wall
  s.bodies[1].trajectory = n + 1;
  s.bodies[3].id = 61;  // a box (trajectory 2)
  auto with_static = [&](std::initializer_list<size_t> bodies) {
    SceneBundle st = s;
    for (const size_t b : bodies) {
      const Pose p = eval_pose(s.trajectories[static_cast<size_t>(*s.bodies[b].trajectory)], kT0);
      st.bodies[b].T_world_body = pose_matrix(p.r, p.p);
      st.bodies[b].trajectory.reset();
    }
    return st;
  };
  Requests q;
  for (uint32_t p = 0; p < 120 * 90; ++p) {
    q.cam.push_back(0);
    q.pixel.push_back(p);
    q.t.push_back(kT0 + 9'000'000);
  }
  const RenderFlags flags{true, true};
  auto same = [](const RenderResult& a, const RenderResult& b) {
    return a.irradiance == b.irradiance && a.signature == b.signature && a.sub_instance == b.sub_instance &&
           std::memcmp(a.centre.data(), b.centre.data(), a.centre.size() * sizeof(GtSample)) == 0;
  };
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    Renderer r(s, options(backend, 6));
    const RenderResult movers = r.render(q.cam, q.pixel, q.t, flags);
    const RenderResult floor = Renderer(with_static({0}), options(backend, 6)).render(q.cam, q.pixel, q.t, flags);
    const RenderResult box = Renderer(with_static({3}), options(backend, 6)).render(q.cam, q.pixel, q.t, flags);
    const RenderResult both = Renderer(with_static({0, 3}), options(backend, 6)).render(q.cam, q.pixel, q.t, flags);
    const std::pair<std::vector<uint32_t>, const RenderResult*> sets[] = {
        {{70}, &floor}, {{61}, &box}, {{61, 70}, &both}, {{70}, &floor}, {{61}, &box}};
    for (const auto& [ids, expected] : sets) {
      INFO("held " << ids.size() << " bodies, first " << ids[0]);
      CHECK(r.hold(ids, kT0));
      CHECK(same(r.render(q.cam, q.pixel, q.t, flags), *expected));
    }
    CHECK((backend == BackendKind::kOptix) == (r.held_device_bytes() > 0));
    CHECK(r.hold({}, 0));
    CHECK(same(r.render(q.cam, q.pixel, q.t, flags), movers));
  }
}

TEST_CASE("renderer input validation") {
  SceneBundle s = mixed_scene();
  Renderer r(s, options(BackendKind::kEmbree, 2));
  const std::vector<uint16_t> cam{0};
  const std::vector<int64_t> t{kT0};
  CHECK_THROWS_AS(r.render(cam, std::vector<uint32_t>{120 * 90}, t), std::invalid_argument);
  CHECK_THROWS_AS(r.render(std::vector<uint16_t>{1}, std::vector<uint32_t>{0}, t), std::invalid_argument);
  CHECK_THROWS_AS(r.render(cam, std::vector<uint32_t>{0}, std::vector<int64_t>{kT1 + 1}), std::invalid_argument);
  // The failed request leaves the renderer usable.
  CHECK(r.render(cam, std::vector<uint32_t>{5}, t).count == 1);
  SceneBundle bad_semantic = s;
  bad_semantic.bodies[0].semantic = 70000;  // would wrap to 4464 in the u16 ground truth
  CHECK_THROWS_AS(Renderer(bad_semantic, options(BackendKind::kEmbree)), std::invalid_argument);
  // Texel indices beyond int32 (the GPU loop over them did not end), and a spot cone whose fp32 cosines coincide
  // (its falloff divided 0 by 0).
  SceneBundle huge_uv = mixed_scene();
  huge_uv.bodies[0].mesh.uvs[0] = 1e9f;
  CHECK_THROWS_AS(validate(huge_uv), std::invalid_argument);
  SceneBundle pencil = mixed_scene();
  pencil.lights[1].spot_inner_rad = 1e-4f;
  pencil.lights[1].spot_outer_rad = 2e-4f;
  CHECK_THROWS_AS(validate(pencil), std::invalid_argument);
  s.bodies[0].id = s.bodies[1].id;
  CHECK_THROWS_AS(Renderer(s, options(BackendKind::kEmbree)), std::invalid_argument);
}

TEST_CASE("a failed build leaves the renderer unbuilt; the next call builds again") {
  // A pose cache too large to allocate fails the build after the scene, executor and tracer are built. Each
  // call must fail cleanly and leave nothing half built (the next one would dereference a missing part).
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    RendererOptions o = options(backend, 2);
    o.pose_cache_bytes = uint64_t{1} << 62;
    Renderer r(mixed_scene(), o);
    for (int k = 0; k < 2; ++k) {
      CHECK_THROWS_AS(r.render(std::vector<uint16_t>{0}, std::vector<uint32_t>{5}, std::vector<int64_t>{kT0}),
                      std::bad_alloc);
      REQUIRE_FALSE(r.built());
    }
    CHECK_THROWS_AS(r.hold(std::vector<uint32_t>{11}, kT0), std::bad_alloc);
    CHECK_FALSE(r.built());
  }
}

TEST_CASE("the P2-B B0 census follows the probe reuse rule (render/census.h)") {
  // A static camera before a static two-triangle wall, a thin static rod (0.1 px at 1 m, narrower than the probes'
  // spacing: the rule's first gap) and a mover; N = 64, every pixel, on the host.
  SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
  s.trajectories.push_back(moving_trajectory(kT0, kT1, quat_identity<double>(), {-0.3, 0.0, 1.0}, {0.1, 0.0, 0.0},
                                             {0.0, 0.0, 0.0}));
  s.cameras.push_back(pinhole(64, 48, 60.0, 0));
  Body wall = make_body(1, plane_mesh(2.0, 1));
  wall.T_world_body = translation(0, 0, 1.5);
  s.bodies.push_back(wall);
  Body rod = make_body(2, box_mesh(0.0008f, 1.0f, 0.0008f));
  rod.T_world_body = translation(0.05, 0, 1.0);
  s.bodies.push_back(rod);
  Body mover = make_body(3, box_mesh(0.05f, 0.05f, 0.05f));
  mover.trajectory = 1;
  s.bodies.push_back(mover);
  s.lights.push_back(ambient_light(1.0f));
  constexpr uint32_t kN = 64;
  SceneSettings settings;
  settings.n_subsamples = kN;
  const SceneData data(s, settings);
  ThreadPool pool(4);
  HostExecutor host(pool);
  host.upload_scene(data);
  EmbreeTracer tracer(pool);
  tracer.build(data, host.scene_view());
  PoseTable table(static_cast<int32_t>(data.geometries().size()), static_cast<int32_t>(data.lights().size()), 64);
  table.reset(kT0);
  const int64_t t = kT0 + 100'000'000;
  uint32_t slot = 0;
  REQUIRE(table.slot(0, t, &slot));
  const PoseTableView poses = host.sync_poses(table, 0);
  std::vector<SampleRequest> requests;
  for (uint32_t p = 0; p < 64 * 48; ++p) requests.push_back({0, 0, p, 0, kN, 0, slot});
  const auto count = static_cast<uint32_t>(requests.size());
  const TraceBatch batch{requests.data(), count, kN, poses, nullptr, 0};
  std::vector<HitRecord> hits(static_cast<size_t>(count) * kN);
  tracer.sample(batch, hits.data(), nullptr);

  uint64_t reusable = 0, mismatched = 0, coherent = 0, ring = 0;
  for (uint32_t r = 0; r < count; ++r) {
    const HitRecord* h = hits.data() + static_cast<size_t>(r) * kN;
    const CensusCounts c = census_sample(host.scene_view(), poses, requests[r], h);
    REQUIRE(c.mismatched <= c.reusable);
    REQUIRE(c.reusable <= kN - kCensusProbes);
    bool probes = h[0].instance == 1, mover_hit = false, rod_hit = false;
    for (uint32_t k = 0; k < kN; ++k) {
      if (k < kCensusProbes) probes = probes && h[k].instance == h[0].instance && h[k].prim == h[0].prim;
      mover_hit = mover_hit || h[k].instance == 3;
      rod_hit = rod_hit || (k >= kCensusProbes && h[k].instance == 2);
    }
    if (!probes) REQUIRE(c.reusable == 0);  // the probes on the rod, the mover or two wall triangles
    if (c.mismatched > 0) REQUIRE(rod_hit);  // the mover is culled by its sphere: only the rod is missed
    reusable += c.reusable;
    mismatched += c.mismatched;
    coherent += probes;
    // The mover's bounding sphere (radius 0.087 m) is wider than the box (0.05 m): wall rays beside it are traced.
    ring += probes && !mover_hit && c.reusable == 0;
  }
  CHECK(mismatched > 0);
  CHECK(ring > 0);
  CHECK(reusable > (kN - kCensusProbes) * coherent * 9 / 10);
  // A probe on another triangle, or N <= 16: nothing is reused.
  const uint32_t r0 = 5 * 64 + 50;  // a wall pixel away from the rod, the mover and the diagonal
  std::vector<HitRecord> h(hits.begin() + r0 * kN, hits.begin() + (r0 + 1) * kN);
  REQUIRE(census_sample(host.scene_view(), poses, requests[r0], h.data()).reusable == kN - kCensusProbes);
  h[5].prim ^= 1u;
  CHECK(census_sample(host.scene_view(), poses, requests[r0], h.data()).reusable == 0);
  SampleRequest r16 = requests[r0];
  r16.k_count = kCensusProbes;
  CHECK(census_sample(host.scene_view(), poses, r16, hits.data() + r0 * kN).reusable == 0);
  // The class word's band bits (shadow edge, two clusters) route a sample to the band words, else (or without a
  // word) to the rest.
  uint64_t v[kCensusWords] = {};
  const uint32_t edge = kClassShadowEdge, multi = kClassMulti, textured = kClassTextured;
  census_add(v, requests[r0], {3, 1}, &edge);
  census_add(v, requests[r0], {2, 0}, &multi);
  census_add(v, requests[r0], {1, 0}, &textured);
  census_add(v, requests[r0], {4, 2}, nullptr);
  CHECK(v[kCensusBand + kCensusSubsamples] == 2 * (kN - kCensusProbes));
  CHECK(v[kCensusBand + kCensusReusable] == 5);
  CHECK(v[kCensusBand + kCensusMismatched] == 1);
  CHECK(v[kCensusRest + kCensusSubsamples] == 2 * (kN - kCensusProbes));
  CHECK(v[kCensusRest + kCensusReusable] == 5);
  CHECK(v[kCensusRest + kCensusMismatched] == 2);

  // Through the renderer (execution.census): the totals of each render call, the same hits (Embree), no class words.
  RendererOptions o = options(BackendKind::kEmbree, 4);
  o.scene.n_subsamples = kN;
  o.census = true;
  Renderer rd(s, o);
  std::vector<uint16_t> cams(count, 0);
  std::vector<uint32_t> pixels(count);
  std::iota(pixels.begin(), pixels.end(), 0u);
  const std::vector<int64_t> times(count, t);
  for (int call = 0; call < 2; ++call) {
    const RenderResult res = rd.render(cams, pixels, times);
    CHECK(res.stats.census_on);
    CHECK(res.stats.census[kCensusRest + kCensusSubsamples] == uint64_t{count} * (kN - kCensusProbes));
    CHECK(res.stats.census[kCensusRest + kCensusReusable] == reusable);
    CHECK(res.stats.census[kCensusRest + kCensusMismatched] == mismatched);
    CHECK(res.stats.census[kCensusBand + kCensusSubsamples] == 0);
  }
  o.census = false;
  CHECK_FALSE(Renderer(s, o).render(cams, pixels, times).stats.census_on);
}

TEST_CASE("fixed64 (PHASE2 D-P2-3): phase1 at N <= 64, T1 above it, prefix terms, all-N ratios, no-prefix clusters") {
  // A static camera over a textured floor with a box (a mover held still) and its shadow, a wall whose visible side
  // faces away from the second light, and misses beside it with a background radiance. Traced once at N = 1024 and
  // shaded at k_count 16 ... 1024 under both integrals on the host (device = host: test_device.cpp).
  SceneBundle s;
  s.light_groups = {"default", "mains"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0.3, 0.4, 1.3}));
  s.cameras.push_back(pinhole(40, 30, 32.0, 0));
  std::vector<uint8_t> tex(64 * 64 * 3);
  for (size_t i = 0; i < tex.size(); ++i) tex[i] = static_cast<uint8_t>((i * 37) % 251);
  Body floor = make_body(1, plane_mesh(3.0, 12), {0.8f, 0.7f, 0.6f});
  floor.materials[0].texture = tex;
  floor.materials[0].texture_width = floor.materials[0].texture_height = 64;
  floor.materials[0].texture_channels = 3;
  floor.T_world_body = {1, 0, 0, 0, 0, 0, -1, 0.5, 0, 1, 0, 1.8, 0, 0, 0, 1};
  s.bodies.push_back(floor);
  Body wall = make_body(2, plane_mesh(0.8, 2));
  wall.T_world_body = translation(-0.7, -0.3, 3.0);
  s.bodies.push_back(wall);
  Body box = make_body(3, box_mesh(0.2f, 0.1f, 0.15f), {0.3f, 0.6f, 0.4f});
  box.trajectory = 1;
  s.bodies.push_back(box);
  s.lights.push_back(point_light({-0.5, -0.9, 0.8}, 1.5f, true, 0));
  s.lights.push_back(point_light({-0.6, -0.3, 4.0}, 1.0f, true, 1));  // behind the wall
  s.lights.push_back(background_light(0.07f, 1));
  s.lights.push_back(ambient_light(0.1f));
  constexpr uint32_t kStride = 1024;
  SceneSettings settings;
  settings.n_subsamples = kStride;
  const SceneData phase1(s, settings);
  settings.pixel_integral = PixelIntegral::kFixed64;
  const SceneData fixed(s, settings);
  ThreadPool pool(4);
  HostExecutor h1(pool), h64(pool);
  h1.upload_scene(phase1);
  h64.upload_scene(fixed);
  EmbreeTracer tracer(pool);
  tracer.build(phase1, h1.scene_view());
  PoseTable table(static_cast<int32_t>(phase1.geometries().size()), static_cast<int32_t>(phase1.lights().size()), 64);
  table.reset(kT0);
  uint32_t slot = 0;
  REQUIRE(table.slot(0, kT0 + 100'000'000, &slot));
  const PoseTableView poses = h1.sync_poses(table, 0);
  REQUIRE(h64.sync_poses(table, 0).rows == poses.rows);
  std::vector<SampleRequest> traced;
  for (uint32_t p = 0; p < 40 * 30; ++p) traced.push_back({0, 0, p, 0, kStride, 0, slot});
  const auto count = static_cast<uint32_t>(traced.size());
  std::vector<HitRecord> hits(static_cast<size_t>(count) * kStride);
  std::vector<uint32_t> occ(hits.size());
  tracer.sample({traced.data(), count, kStride, poses, nullptr, 0}, hits.data(), occ.data());
  struct Shaded {
    std::vector<float> irr;
    std::vector<uint64_t> sig;
    std::vector<uint32_t> klass;
  };
  // Pixel-samples r0 .. r0 + n of the traced records (or of `h`, `o`: one request's) at k_count k.
  const auto shade = [&](HostExecutor& ex, uint32_t k, uint32_t r0, uint32_t n, const HitRecord* h, const uint32_t* o) {
    std::vector<SampleRequest> req(traced.begin() + r0, traced.begin() + r0 + n);
    for (SampleRequest& q : req) q.k_count = k;
    Shaded out{std::vector<float>(2 * n), std::vector<uint64_t>(n), std::vector<uint32_t>(n)};
    ex.shade_into(0, {req.data(), n, kStride, poses, nullptr, 0}, h, o, out.irr.data(), out.sig.data(), nullptr,
                  nullptr, out.klass.data());
    return out;
  };
  const auto all = [&](HostExecutor& ex, uint32_t k) { return shade(ex, k, 0, count, hits.data(), occ.data()); };
  // N <= 64: the same values, signatures and class words (every k_count up to 64 is phase1's).
  for (const uint32_t k : {16u, 33u, 64u}) {
    const Shaded a = all(h1, k), b = all(h64, k);
    CHECK(std::memcmp(a.irr.data(), b.irr.data(), a.irr.size() * sizeof(float)) == 0);
    CHECK((a.sig == b.sig && a.klass == b.klass));
  }
  // T1: a pixel-sample with one cluster and no shadow edge keeps its N = 64 value at power-of-two N under fixed64
  // (textured, all misses, a light behind the surface); phase1 does not.
  const Shaded at64 = all(h64, 64);
  const auto interior = [](uint32_t c) { return (c & (kClassMulti | kClassShadowEdge)) == 0; };
  for (const uint32_t k : {128u, 256u, 1024u}) {
    INFO("N = " << k);
    const Shaded f = all(h64, k), p = all(h1, k);
    size_t n_interior = 0, textured = 0, missed = 0, away = 0, moved = 0;
    for (uint32_t r = 0; r < count; ++r) {
      if (!interior(f.klass[r])) continue;
      ++n_interior;
      CHECK(std::memcmp(&f.irr[2 * r], &at64.irr[2 * r], 2 * sizeof(float)) == 0);
      const HitRecord& h0 = hits[static_cast<size_t>(r) * kStride];
      textured += (f.klass[r] & kClassTextured) != 0;
      missed += h0.instance == kInstanceNone;
      away += h0.instance == 2 && occ[static_cast<size_t>(r) * kStride] == 2u;  // the wall: light 1 behind it
      moved += std::memcmp(&p.irr[2 * r], &at64.irr[2 * r], 2 * sizeof(float)) != 0;
    }
    CHECK(textured > 20);
    CHECK(missed > 20);
    CHECK(away > 20);
    CHECK(moved > textured / 2);
  }
  // T1 holds at powers of two only: at N = 97, 97 * fl(1/97) = 1 - 2^-24, so a full cluster's coverage weight moves
  // interior values by an ulp (the band's N_base and N_band must be powers of two; ENGINE_CPP "P2-B B3 notes").
  {
    const Shaded f = all(h64, 97);
    size_t n_interior = 0, off = 0;
    for (uint32_t r = 0; r < count; ++r) {
      if (!interior(f.klass[r])) continue;
      ++n_interior;
      off += std::memcmp(&f.irr[2 * r], &at64.irr[2 * r], 2 * sizeof(float)) != 0;
      for (uint32_t g = 0; g < 2; ++g) CHECK(f.irr[2 * r + g] == doctest::Approx(at64.irr[2 * r + g]).epsilon(3e-7));
    }
    CHECK(n_interior > 100);
    CHECK(off > n_interior / 2);
  }
  // One interior box sample lit by light 0, at N = 256 (records copied, then edited).
  uint32_t rb = count;
  for (uint32_t r = 0; r < count && rb == count; ++r) {
    const size_t i = static_cast<size_t>(r) * kStride;
    if (interior(at64.klass[r]) && hits[i].instance == 3 && (occ[i] & 1u) == 0) rb = r;
  }
  REQUIRE(rb < count);
  const auto one = [&](HostExecutor& ex, const std::vector<HitRecord>& h, const std::vector<uint32_t>& o) {
    return shade(ex, 256, rb, 1, h.data(), o.data()).irr;
  };
  const std::vector<HitRecord> h0(hits.begin() + rb * kStride, hits.begin() + (rb + 1) * kStride);
  const std::vector<uint32_t> o0(occ.begin() + rb * kStride, occ.begin() + (rb + 1) * kStride);
  const std::vector<float> lit = one(h64, h0, o0);
  // Per-member terms from the prefix: other faces of the box at k >= 64 (same cluster, recorded shadow bits) move
  // phase1's means, not fixed64's.
  std::vector<HitRecord> h = h0;
  for (uint32_t k = 64; k < 256; ++k) h[k].prim = (h[k].prim + 4) % 12;
  CHECK(one(h64, h, o0) == lit);
  CHECK(one(h1, h, o0) != one(h1, h0, o0));
  // Shadow ratios from all N: light 0 blocked at k >= 64 gives about a quarter of its term (its f is near uniform).
  std::vector<uint32_t> o = o0;
  for (uint32_t k = 0; k < 256; ++k) o[k] |= 1u;
  const std::vector<float> dark = one(h64, h0, o);
  for (uint32_t k = 0; k < 64; ++k) o[k] = o0[k];
  const float quarter = one(h64, h0, o)[0];
  REQUIRE(dark[0] < lit[0]);
  CHECK((quarter - dark[0]) / (lit[0] - dark[0]) == doctest::Approx(0.25).epsilon(0.02));
  // A cluster with no member below 64 takes all its members: wall records at k >= 64 shade as phase1 (both clusters
  // untextured; coverage over all N); one wall member below 64 makes fixed64 take that member's terms alone.
  uint32_t rw = count;
  for (uint32_t r = 0; r < count && rw == count; ++r) {
    if (interior(at64.klass[r]) && hits[static_cast<size_t>(r) * kStride].instance == 2) rw = r;
  }
  REQUIRE(rw < count);
  h = h0;
  o = o0;
  for (uint32_t k = 64; k < 256; ++k) {
    h[k] = hits[static_cast<size_t>(rw) * kStride + k];
    o[k] = occ[static_cast<size_t>(rw) * kStride + k];
  }
  CHECK(one(h64, h, o) == one(h1, h, o));
  h[5] = hits[static_cast<size_t>(rw) * kStride + 5];
  o[5] = occ[static_cast<size_t>(rw) * kStride + 5];
  CHECK(one(h64, h, o) != one(h1, h, o));
}

TEST_CASE("texture_shadow weighted (P2-F F1): the per-sub-sample sum under uniform f, lit and umbra as pooled, T1") {
  // A static camera facing a textured plane at z = 2 m that fills the view (every pixel-sample one textured cluster),
  // a box outside the view casting the shadows of a point light (bit 0, group 0, with the ambient) and of a
  // directional light (bit 1, group 1 alone, so its f is uniform over the plane) into it. Traced once at N = 1024,
  // shaded on the host under {pooled, weighted} x {phase1, fixed64} (device = host: test_device.cpp).
  SceneBundle s;
  s.light_groups = {"default", "sun"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
  s.cameras.push_back(pinhole(40, 30, 32.0, 0));
  std::vector<uint8_t> tex(64 * 64 * 3);
  for (size_t i = 0; i < tex.size(); ++i) tex[i] = static_cast<uint8_t>(40 + (i / 3 * 37) % 211);
  Body plane = make_body(1, plane_mesh(1.5, 4));
  plane.materials[0].texture = tex;
  plane.materials[0].texture_width = plane.materials[0].texture_height = 64;
  plane.materials[0].texture_channels = 3;
  plane.T_world_body = {-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 2.0, 0, 0, 0, 1};  // normal -z: facing the camera
  s.bodies.push_back(plane);
  Body box = make_body(2, box_mesh(0.15f, 0.3f, 0.1f), {0.5f, 0.5f, 0.5f});
  box.T_world_body = translation(1.2, 0.0, 1.0);  // outside the view (half-width 0.625 m at z = 1 m)
  s.bodies.push_back(box);
  s.lights.push_back(point_light({2.0, 0.0, 0.0}, 2.0f, true, 0));
  Light sun;
  sun.kind = BundleLightKind::kDirectional;
  sun.group = 1;
  sun.direction = {-0.7071067811865476, 0.0, 0.7071067811865476};
  sun.intensity = 1.0f;
  sun.casts_shadows = true;
  s.lights.push_back(sun);
  s.lights.push_back(ambient_light(0.1f));
  constexpr uint32_t kStride = 1024, kPixels = 40 * 30;
  ThreadPool pool(4);
  struct Arm {
    std::unique_ptr<SceneData> data;
    std::unique_ptr<HostExecutor> ex;
  };
  // arms[weighted][fixed64]
  Arm arms[2][2];
  for (int w = 0; w < 2; ++w) {
    for (int f = 0; f < 2; ++f) {
      SceneSettings settings;
      settings.n_subsamples = kStride;
      settings.pixel_integral = f != 0 ? PixelIntegral::kFixed64 : PixelIntegral::kPhase1;
      settings.texture_shadow = w != 0 ? TextureShadow::kWeighted : TextureShadow::kPooled;
      arms[w][f].data = std::make_unique<SceneData>(s, settings);
      arms[w][f].ex = std::make_unique<HostExecutor>(pool);
      arms[w][f].ex->upload_scene(*arms[w][f].data);
    }
  }
  HostExecutor& h0 = *arms[0][0].ex;
  REQUIRE(arms[1][1].ex->scene_view().texture_shadow == 1);
  REQUIRE(h0.scene_view().texture_shadow == 0);
  EmbreeTracer tracer(pool);
  tracer.build(*arms[0][0].data, h0.scene_view());
  PoseTable table(static_cast<int32_t>(arms[0][0].data->geometries().size()),
                  static_cast<int32_t>(arms[0][0].data->lights().size()), 64);
  table.reset(kT0);
  uint32_t slot = 0;
  REQUIRE(table.slot(0, kT0 + 100'000'000, &slot));
  const PoseTableView poses = h0.sync_poses(table, 0);
  for (auto& row : arms) {
    for (Arm& a : row) REQUIRE(a.ex->sync_poses(table, 0).rows == poses.rows);
  }
  std::vector<SampleRequest> traced;
  for (uint32_t p = 0; p < kPixels; ++p) traced.push_back({0, 0, p, 0, kStride, 0, slot});
  std::vector<HitRecord> hits(static_cast<size_t>(kPixels) * kStride);
  std::vector<uint32_t> occ(hits.size());
  tracer.sample({traced.data(), kPixels, kStride, poses, nullptr, 0}, hits.data(), occ.data());
  REQUIRE(std::all_of(hits.begin(), hits.end(), [](const HitRecord& h) { return h.instance == 1u; }));  // one cluster
  struct Shaded {
    std::vector<float> irr;
    std::vector<uint32_t> klass;
  };
  const auto shade = [&](int w, int f, uint32_t k, const uint32_t* o) {
    std::vector<SampleRequest> req(traced);
    for (SampleRequest& q : req) q.k_count = k;
    Shaded out{std::vector<float>(2 * kPixels), std::vector<uint32_t>(kPixels)};
    std::vector<uint64_t> sig(kPixels);
    arms[w][f].ex->shade_into(0, {req.data(), kPixels, kStride, poses, nullptr, 0}, hits.data(), o, out.irr.data(),
                              sig.data(), nullptr, nullptr, out.klass.data());
    return out;
  };
  const auto same = [](const float* a, const float* b) { return std::memcmp(a, b, 2 * sizeof(float)) == 0; };

  // Traced shadows: a pixel-sample lit or shadowed for each light on every sub-sample (no kClassShadowEdge) has the
  // same value under both ratios, bit for bit; at a shadow edge most move. T1 under weighted: the edge-free values at
  // N = 128, 256 and 1024 under fixed64 equal N = 64's.
  const Shaded p64 = shade(0, 0, 64, occ.data()), w64 = shade(1, 0, 64, occ.data());
  size_t lit = 0, umbra = 0, edges = 0, moved = 0;
  for (uint32_t r = 0; r < kPixels; ++r) {
    REQUIRE(p64.klass[r] == w64.klass[r]);
    if ((w64.klass[r] & kClassShadowEdge) == 0) {
      CHECK(same(&w64.irr[2 * r], &p64.irr[2 * r]));
      const uint32_t o = occ[static_cast<size_t>(r) * kStride];
      lit += o == 0;
      umbra += o == 3;
    } else {
      ++edges;
      moved += !same(&w64.irr[2 * r], &p64.irr[2 * r]);
    }
  }
  CHECK(lit > 200);
  CHECK(umbra > 10);
  CHECK(edges > 20);
  CHECK(moved > edges / 2);
  for (const uint32_t k : {128u, 256u, 1024u}) {
    INFO("N = " << k);
    const Shaded w = shade(1, 1, k, occ.data()), p = shade(0, 1, k, occ.data());
    size_t interior = 0;
    for (uint32_t r = 0; r < kPixels; ++r) {
      if ((w.klass[r] & kClassShadowEdge) != 0) continue;
      ++interior;
      CHECK(same(&w.irr[2 * r], &w64.irr[2 * r]));
      CHECK(same(&w.irr[2 * r], &p.irr[2 * r]));
    }
    CHECK(interior > 200);
  }

  // Shadow bits set by hand (the shading reads them as given): light 0 lit, the directional light (bit 1) blocked on a
  // random half of the sub-samples. Its group's value against the per-sub-sample sum (1/N) sum_k rho_k f_k V_k / pi,
  // with rho_k each member's lookup at the term footprint 0.25 / member_count (fixed64: 64 above N = 64) and the term
  // members' mean albedo in front under fixed64.
  std::mt19937 rng(20261013);
  std::vector<uint32_t> occ_r(occ.size());
  for (uint32_t& o : occ_r) o = (rng() & 1u) << 1;
  const SceneView& view = h0.scene_view();
  const CameraView& cam = view.cameras[0];
  const int32_t geo = find_body(view, 1)->geometry;  // the static world (the plane and the box)
  const MaterialView& mat = view.materials[hit_material(view.geometries[geo], hits[0].prim)];
  REQUIRE(mat.texture >= 0);
  const LightView& sun_view = view.lights[view.shadow_light[1]];
  REQUIRE(sun_view.kind == LightKind::kDirectional);
  const auto expected = [&](uint32_t r, uint32_t n, bool fixed, bool weighted) {
    const uint32_t terms = fixed ? member_count(1, n) : n;
    const float var = subsample_footprint_variance(terms);
    double rho_terms = 0, rho_all = 0, rho_lit = 0, f_lit = 0, f_all = 0, sum_lit = 0;
    for (uint32_t k = 0; k < n; ++k) {
      const size_t i = static_cast<size_t>(r) * kStride + k;
      const SurfaceSample ss = surface_sample(view.geometries[geo], pose_row(poses, slot, geo), hits[i].prim,
                                              subsample_direction(view, cam, r, k), true);
      const double rho = subsample_albedo(cam, view.textures[mat.texture], view.texels, ss, var);
      const double f = lambert_irradiance(sun_view, light_row(poses, slot, view.shadow_light[1]), ss.p, ss.n_shade);
      const bool v = (occ_r[i] & 2u) == 0;
      if (k < terms) rho_terms += rho;
      rho_all += rho * f;
      f_all += f;
      if (v) {
        rho_lit += rho * f;
        f_lit += f;
        sum_lit += rho * f;
      }
    }
    const double rho_bar = rho_terms / terms;
    if (!fixed) {  // phase1: the per-sub-sample sum (weighted) or rho_bar times the f-weighted lit share (pooled)
      return weighted ? sum_lit / n / std::numbers::pi : rho_bar * f_lit / n / std::numbers::pi;
    }
    return rho_bar * (f_all / n) * (weighted ? rho_lit / rho_all : f_lit / f_all) / std::numbers::pi;
  };
  for (const auto& [n, fixed] : {std::pair{64u, false}, std::pair{256u, false}, std::pair{256u, true},
                                 std::pair{1024u, true}}) {
    INFO("N = " << n << (fixed ? " (fixed64)" : ""));
    const Shaded w = shade(1, fixed, n, occ_r.data()), p = shade(0, fixed, n, occ_r.data());
    size_t biased = 0, checked = 0;
    for (uint32_t r = 0; r < kPixels; r += 7) {
      const double ew = expected(r, n, fixed, true), ep = expected(r, n, fixed, false);
      CHECK(w.irr[2 * r + 1] == doctest::Approx(ew).epsilon(2e-5).scale(0));
      CHECK(p.irr[2 * r + 1] == doctest::Approx(ep).epsilon(2e-5).scale(0));
      CHECK(w.irr[2 * r] == p.irr[2 * r]);  // light 0 lit everywhere: its group is the same in both
      if (!fixed) biased += std::abs(static_cast<double>(p.irr[2 * r + 1]) / ew - 1) > 1e-3;  // ew: per sub-sample
      ++checked;
    }
    CHECK(checked > 150);
    if (!fixed) CHECK(biased > checked / 10);  // the pooled ratio's bias over texture, which weighted removes
  }
  // All lit and all blocked by hand: both ratios are exactly 1 and 0, so the values agree bit for bit.
  for (const uint32_t bits : {0u, 3u}) {
    std::vector<uint32_t> o(occ.size(), bits);
    for (const auto& [n, fixed] : {std::pair{64u, 0}, std::pair{1024u, 1}}) {
      const Shaded w = shade(1, fixed, n, o.data()), p = shade(0, fixed, n, o.data());
      CHECK(std::memcmp(w.irr.data(), p.irr.data(), w.irr.size() * sizeof(float)) == 0);
    }
  }
}
