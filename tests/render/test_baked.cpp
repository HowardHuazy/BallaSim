// Baked radiance, U0 (docs/design/phase3/scans.md §4.1, increment S1b), on every available backend:
//   T2: a baked material renders L_g = T S_g, bit for bit the ambient-only emulation of today's lit path with
//       A_g = S_g = fl32(double(K) double(a_g)) (textured plane and sphere, an untextured box; K 1 and 75; one group and
//       shares 0.3 / 0.7; N = 64 and fixed64 at N = 256), signatures included;
//   lights: no light reaches a baked receiver; lights other than proxies trace no shadow ray there (signatures and the
//       plan pass's states), proxies do (their factor is 1 where no inserted object blocks them, U0 bit for bit, and U2's
//       elsewhere: T5, below); under accuracy.texture_shadow weighted too (P2-F F1's switch: with F = 1 the same T S_g,
//       §4.2 (i)); P2-D's P&V rows and probes give baked clusters no light;
//   T4: the binary16 decode, and level 0 of a binary16 ramp's pyramid is fl32(sum_c w_c b_c h_c) bit for bit; the scale
//       table.
// The geometry split of scenes with baked proxies (scans.md §4.2, increment S3a-1): the scan alone in the world, the
// other static bodies trace geometries of their own (world frame, dynamic 0, their pose rows' spheres); lit receivers
// render bit for bit as in the merged world (irradiance, signatures, shadow masks; the plan pass maps split blockers
// to static); Renderer::hold keeps the movers apart.
// U2 (increment S3a-2): a baked receiver's proxies follow the inserted / scan rule (occlusion and static words), the
// static word enters the signature, and a baked cluster's radiance takes the shadow factor F (T5): against analytic
// per-sub-sample values (directional and point proxies, partial coverage, a static scan edge crossing an inserted one,
// two proxies in one group, two light groups, movers, baked props, fixed64 at N = 256), and the arithmetic of
// accumulate_baked and cluster_radiance on hand-made sums (pooled and texture-weighted); a probe's mask on a baked
// receiver is render_into's occlusion word.
// S3b: the plan pass's states of a baked receiver's proxies (lit when no inserted object blocks, static under the scan's
// own shadow or a split static body, mover + id), the start diagnostic Renderer::proxy_visibility, and the dense
// channel shadow_factor (the ratio to the irradiance with every proxy factor 1).
#include <doctest/doctest.h>
#if BALLA_RENDER_HAVE_OPTIX
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "core/math/so3.h"
#include "core/time/time.h"
#include "render/embree/embree_tracer.h"
#include "render/executor.h"
#include "render/pose.h"
#include "render/ray.h"
#include "render/renderer.h"
#include "render/scene_data.h"
#include "render/shade/codes.h"
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
constexpr int kW = 64, kH = 48;

// UV sphere about the body origin: n_lat x n_lon quads, outward normals, u = longitude / 2 pi, v = latitude / pi.
Mesh sphere_mesh(float radius, int n_lat, int n_lon) {
  Mesh m;
  for (int i = 0; i <= n_lat; ++i) {
    const double th = std::numbers::pi * i / n_lat;
    for (int j = 0; j <= n_lon; ++j) {
      const double ph = 2.0 * std::numbers::pi * j / n_lon;
      const double n[3] = {std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};
      for (double c : n) {
        m.positions.push_back(static_cast<float>(static_cast<double>(radius) * c));
        m.normals.push_back(static_cast<float>(c));
      }
      m.uvs.insert(m.uvs.end(), {static_cast<float>(j) / static_cast<float>(n_lon),
                                 static_cast<float>(i) / static_cast<float>(n_lat)});
    }
  }
  for (int i = 0; i < n_lat; ++i) {
    for (int j = 0; j < n_lon; ++j) {
      const auto a = static_cast<uint32_t>(i * (n_lon + 1) + j), b = a + 1, c = a + static_cast<uint32_t>(n_lon) + 1,
                 d = c + 1;
      m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
    }
  }
  return m;
}

std::vector<uint8_t> pattern(int w, int h, int seed) {
  std::vector<uint8_t> t(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
  for (size_t i = 0; i < t.size(); ++i) t[i] = static_cast<uint8_t>((i * (37 + 2 * static_cast<size_t>(seed)) + 11) % 253);
  return t;
}

void set_texture(Material& m, std::vector<uint8_t> rgb, int w, int h) {
  m.texture = std::move(rgb);
  m.texture_width = w;
  m.texture_height = h;
  m.texture_channels = 3;
}

void bake(Material& m, float k, std::vector<float> shares) {
  m.shading = Shading::kBaked;
  m.baked_scale = k;
  m.baked_groups = std::move(shares);
}

// A tilted textured plane (id 1) behind a textured sphere (id 2) and an untextured box (id 3), all static, under a
// static camera; `groups` light groups, no lights.
SceneBundle room(int groups) {
  SceneBundle s;
  s.light_groups = groups == 1 ? std::vector<std::string>{"default"} : std::vector<std::string>{"default", "mains"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
  s.cameras.push_back(pinhole(kW, kH, 60.0, 0));
  Body plane = make_body(1, plane_mesh(1.2, 6), {0.9f, 0.8f, 1.0f});
  set_texture(plane.materials[0], pattern(64, 64, 0), 64, 64);
  plane.T_world_body = {std::cos(0.4), 0, std::sin(0.4), 0, 0, 1, 0, 0, -std::sin(0.4), 0, std::cos(0.4), 2.2, 0, 0, 0, 1};
  s.bodies.push_back(plane);
  Body sphere = make_body(2, sphere_mesh(0.35f, 12, 24));
  set_texture(sphere.materials[0], pattern(32, 16, 1), 32, 16);
  sphere.T_world_body = translation(0.3, 0.1, 1.5);
  s.bodies.push_back(sphere);
  Body box = make_body(3, box_mesh(0.15f, 0.2f, 0.1f), {0.2f, 0.5f, 0.9f});
  box.T_world_body = translation(-0.45, -0.15, 1.3);
  s.bodies.push_back(box);
  return s;
}

std::vector<float> scales(float k, const std::vector<float>& shares, size_t groups) {
  std::vector<float> s(groups);
  for (size_t g = 0; g < groups; ++g) {
    const double a = shares.empty() ? (g == 0 ? 1.0 : 0.0) : static_cast<double>(shares[g]);
    s[g] = static_cast<float>(static_cast<double>(k) * a);
  }
  return s;
}

RenderResult render_all(const SceneBundle& s, const RendererOptions& o, int64_t t, RenderFlags flags = {}) {
  Renderer r(s, o);
  std::vector<uint16_t> cam(kW * kH, 0);
  std::vector<uint32_t> pixel(kW * kH);
  std::vector<int64_t> ts(kW * kH, t);
  for (uint32_t p = 0; p < pixel.size(); ++p) pixel[p] = p;
  return r.render(cam, pixel, ts, flags);
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

TEST_CASE("T2: baked U0 equals the ambient-only emulation bit for bit (scans.md §4.1)") {
  for (BackendKind backend : backends()) {
    for (const auto& [n_sub, integral] : {std::pair{64u, PixelIntegral::kPhase1}, std::pair{256u, PixelIntegral::kFixed64}}) {
      for (int groups : {1, 2}) {
        for (float k : {1.0f, 75.0f}) {
          const std::vector<float> shares = groups == 1 ? std::vector<float>{} : std::vector<float>{0.3f, 0.7f};
          INFO(backend_name(backend) << ", N = " << n_sub << ", groups " << groups << ", K = " << k);
          SceneBundle baked = room(groups);
          for (Body& b : baked.bodies) bake(b.materials[0], k, shares);
          SceneBundle emulation = room(groups);
          const std::vector<float> s_g = scales(k, shares, static_cast<size_t>(groups));
          for (int g = 0; g < groups; ++g) emulation.lights.push_back(ambient_light(s_g[static_cast<size_t>(g)], g));
          RendererOptions o = options(backend);
          o.scene.n_subsamples = n_sub;
          o.scene.pixel_integral = integral;
          o.scene.sensor_rgb_weights = {0.52, 0.37, 0.11};
          const RenderResult a = render_all(baked, o, kT0 + 1000, {true, false});
          const RenderResult e = render_all(emulation, o, kT0 + 1000, {true, false});
          CHECK(same_bits(a.irradiance, e.irradiance));
          CHECK(a.signature == e.signature);
          size_t hit[4] = {};
          for (const uint32_t id : a.sub_instance) ++hit[id < 4 ? id : 0];
          CHECK((hit[1] > 0 && hit[2] > 0 && hit[3] > 0 && hit[0] > 0));  // every body seen, and misses
          double mean = 0.0;
          for (size_t i = 0; i < a.count; ++i) mean += static_cast<double>(a.irradiance[i * static_cast<size_t>(groups)]);
          CHECK(mean > 0.0);
          if (groups == 2) {  // the shares split each pixel: group 1 / group 0 = fl32(0.7 K) / fl32(0.3 K)
            for (size_t i = 0; i < a.count; ++i) {
              const float g0 = a.irradiance[2 * i], g1 = a.irradiance[2 * i + 1];
              if (g0 > 0.0f) CHECK(std::abs(g1 / g0 - s_g[1] / s_g[0]) <= 1e-5f * s_g[1] / s_g[0]);
            }
          }
        }
      }
    }
  }
}

namespace {

// room(2) baked at K 75 and shares 0.3 / 0.7 (the plane a scan), with a lit box (id 9) crossing between the plane and a
// shadow-casting point light in group 1 (`proxy`: the scan's proxy), plus ambient, a spot and a directional light that
// cast no shadows. `lit_plane`: the plane lit instead (the control: the box's shadow on it moves the signatures).
SceneBundle lit_scene(bool lights, bool proxy, bool lit_plane = false) {
  SceneBundle s = room(2);
  for (Body& b : s.bodies) bake(b.materials[0], 75.0f, {0.3f, 0.7f});
  if (lit_plane) {
    Material& m = s.bodies[0].materials[0];
    m.shading = Shading::kLit;
    m.baked_scale = 1.0f;
    m.baked_groups.clear();
  } else {
    s.bodies[0].scan = true;
  }
  s.trajectories.push_back(moving_trajectory(kT0, kT1, quat_identity<double>(), {-0.6, -0.55, 1.9}, {4.0, 0.0, 0.0},
                                             {0.0, 0.0, 0.0}));
  Body mover = make_body(9, box_mesh(0.12f, 0.12f, 0.12f), {0.6f, 0.6f, 0.6f});
  mover.trajectory = 1;
  s.bodies.push_back(mover);
  if (!lights) return s;
  Light lamp = point_light({0.0, -1.2, 1.6}, 2.0f, true, 1);
  lamp.proxy = proxy;
  s.lights.push_back(lamp);
  s.lights.push_back(ambient_light(0.3f, 0));
  Light spot;
  spot.kind = BundleLightKind::kSpot;
  spot.position = {0.2, -0.5, 0.5};
  spot.direction = {0.0, 0.5, 1.0};
  spot.spot_inner_rad = 0.4f;
  spot.spot_outer_rad = 0.8f;
  spot.intensity = 3.0f;
  spot.casts_shadows = false;
  s.lights.push_back(spot);
  Light sun;
  sun.kind = BundleLightKind::kDirectional;
  sun.direction = {0.1, 0.8, 0.6};
  sun.intensity = 0.4f;
  sun.group = 1;
  sun.casts_shadows = false;
  s.lights.push_back(sun);
  return s;
}

}  // namespace

TEST_CASE("lights do not reach baked receivers; only proxies trace shadow rays there (scans.md §4.1)") {
  constexpr int64_t ta = kT0 + 100'000'000, tb = kT0 + 160'000'000;  // the box at x -0.2 and 0.04 (camera frame)
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    const RendererOptions o = options(backend);
    const RenderFlags flags{true, false};
    const RenderResult dark_a = render_all(lit_scene(false, false), o, ta, flags);
    const RenderResult dark_b = render_all(lit_scene(false, false), o, tb, flags);
    // Plane-only pixel-samples (every sub-sample on the plane) at both times.
    std::vector<size_t> plane;
    for (size_t i = 0; i < dark_a.count; ++i) {
      bool only = true;
      for (uint32_t k = 0; k < dark_a.subsamples; ++k) {
        only = only && dark_a.sub_instance[i * dark_a.subsamples + k] == 1 &&
               dark_b.sub_instance[i * dark_b.subsamples + k] == 1;
      }
      if (only) plane.push_back(i);
    }
    REQUIRE(plane.size() > 1000);
    auto moved = [&](const RenderResult& a, const RenderResult& b) {
      size_t n = 0;
      for (size_t i : plane) n += a.signature[i] != b.signature[i];
      return n;
    };
    auto plane_irradiance = [&](const RenderResult& a, const RenderResult& b) {
      for (size_t i : plane) {
        for (size_t g = 0; g < 2; ++g) {
          if (std::memcmp(&a.irradiance[2 * i + g], &b.irradiance[2 * i + g], sizeof(float)) != 0) return false;
        }
      }
      return true;
    };
    // The control: on a lit plane the box's shadow moves signatures between the two times.
    const RenderResult lit_a = render_all(lit_scene(true, false, true), o, ta);
    const RenderResult lit_b = render_all(lit_scene(true, false, true), o, tb);
    CHECK(moved(lit_a, lit_b) > 20);
    for (const TextureShadow ts : {TextureShadow::kPooled, TextureShadow::kWeighted}) {
      INFO("accuracy.texture_shadow " << (ts == TextureShadow::kWeighted ? "weighted" : "pooled"));
      RendererOptions ow = o;
      ow.scene.texture_shadow = ts;
      // A light other than a proxy: the plane's radiance and signatures are those of the scene without lights.
      for (int64_t t : {ta, tb}) {
        const RenderResult on = render_all(lit_scene(true, false), ow, t);
        const RenderResult& off = t == ta ? dark_a : dark_b;
        CHECK(plane_irradiance(on, off));
        for (size_t i : plane) CHECK(on.signature[i] == off.signature[i]);
      }
      // A proxy: its shadow bits on the plane are traced as on the lit plane (equal signatures). Its factor (U2, §4.2)
      // darkens the lamp's group (1) where the box blocks it (no fill here: a black umbra), and leaves the default
      // group (0), whose share no proxy touches, and every pixel without a blocked bit U0 bit for bit.
      for (int64_t t : {ta, tb}) {
        const RenderResult px = render_all(lit_scene(true, true), ow, t);
        const RenderResult& lit = t == ta ? lit_a : lit_b;
        const RenderResult& dark = t == ta ? dark_a : dark_b;
        size_t shadowed = 0;
        for (size_t i : plane) {
          CHECK(px.signature[i] == lit.signature[i]);
          CHECK(std::memcmp(&px.irradiance[2 * i], &dark.irradiance[2 * i], sizeof(float)) == 0);
          if (px.signature[i] == dark.signature[i]) {
            CHECK(std::memcmp(&px.irradiance[2 * i + 1], &dark.irradiance[2 * i + 1], sizeof(float)) == 0);
          } else {
            ++shadowed;
            CHECK(px.irradiance[2 * i + 1] < dark.irradiance[2 * i + 1]);
          }
        }
        CHECK(shadowed > 20);
      }
    }
  }
}

TEST_CASE("the plan pass traces no light but proxies from baked receivers (lights.h shadow_states), on Embree") {
  const int64_t t = kT0 + 100'000'000;
  std::vector<GtQuery> q(kW * kH);
  for (size_t i = 0; i < q.size(); ++i) q[i] = {t, static_cast<float>(i % kW), static_cast<float>(i / kW)};
  auto states = [&](const SceneBundle& s, std::vector<GtSample>& gt) {
    Renderer r(s, options(BackendKind::kEmbree, 4));
    REQUIRE(r.num_shadow_lights() == 1);  // the lamp
    std::vector<uint32_t> out(q.size() * 2, 99);
    gt.resize(q.size());
    r.gt_into(0, q.data(), q.size(), gt.data(), nullptr, out.data());
    return out;
  };
  std::vector<GtSample> gt_lit, gt_other, gt_proxy;
  const auto lit = states(lit_scene(true, false, true), gt_lit);
  const auto other = states(lit_scene(true, false), gt_other);
  const auto proxy = states(lit_scene(true, true), gt_proxy);
  size_t plane = 0, lit_movers = 0;
  for (size_t i = 0; i < q.size(); ++i) {
    if (gt_lit[i].instance != 1) continue;
    REQUIRE((gt_other[i].instance == 1 && gt_proxy[i].instance == 1));
    ++plane;
    lit_movers += lit[2 * i] >= kShadowMover;
    CHECK(other[2 * i] == kShadowStatic);  // a baked plane: the lamp traces no ray
    // A proxy traces on the plane, by S3b's rule (§4.2): no inserted blocker is lit (the lit plane's state, t 0); a split
    // static body (sphere, box) blocks as the lit plane's merged world did (static); a mover is the lit plane's mover,
    // unless a split static body that comes first in geometry order blocks too (then static; multi counts movers only).
    const uint32_t s = proxy[2 * i], m = lit[2 * i];
    if (m == kShadowLit) {
      CHECK((s == kShadowLit && proxy[2 * i + 1] == 0u));
    } else if (m == kShadowStatic) {
      CHECK(s == kShadowStatic);
    } else {
      CHECK(((s & ~kShadowMulti) == (m & ~kShadowMulti) || s == kShadowStatic));
    }
  }
  CHECK(plane > 1000);
  CHECK(lit_movers > 20);  // the lamp's shadow of the box on the plane
}

TEST_CASE("P2-D's P&V rows and probes give a baked cluster no light (codes.h pv_table_row, probe.h)") {
  const int64_t t_w = time::window_start(time::window_index(kT0 + 100'000'000));
  const WindowClock clock{t_w, 10, 0, 0};
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    Renderer r(lit_scene(true, true), options(backend));  // the lamp a proxy: its bits are traced on the plane
    const int32_t groups = r.num_groups(), lights = r.num_shadow_lights();
    REQUIRE(lights == 1);
    const uint32_t n_sub = r.options().scene.n_subsamples, words = pv_table_words(groups, lights);
    std::vector<SampleRequest> req(kW * kH), probes(kW * kH);
    for (uint32_t p = 0; p < req.size(); ++p) {
      req[p] = {0, 0, p, 0, n_sub, 0, 0};
      probes[p] = {0, static_cast<uint16_t>(RequestKind::kProbe), p, 0, 1, 0, 0};  // sub-sample 0, the lamp
    }
    std::vector<float> irr(req.size() * static_cast<size_t>(groups));
    std::vector<uint64_t> sig(req.size());
    std::vector<uint32_t> slot(req.size()), table(req.size() * words);
    std::vector<uint32_t> codes(req.size() * pv_code_words(n_sub, lights));
    uint32_t count = 0;
    const PvOut pv{slot.data(), codes.data(), table.data(), &count, static_cast<uint32_t>(req.size()), 1};
    r.begin_window(0, clock, nullptr);
    r.assign_pose_rows(req.data(), req.size(), nullptr);
    r.assign_pose_rows(probes.data(), probes.size(), nullptr);
    r.render_into(req.data(), req.size(), nullptr, irr.data(), sig.data(), nullptr, nullptr, 0, &pv);
    std::vector<ProbeOut> got(probes.size());
    r.probe_into(probes.data(), probes.size(), got.data(), nullptr);
    r.take_render_stats();
    size_t baked_rows = 0, lit_rows = 0, baked_probes = 0, lit_probes = 0;
    for (size_t i = 0; i < req.size(); ++i) {
      REQUIRE(slot[i] < count);
      const uint32_t* t = table.data() + static_cast<size_t>(slot[i]) * words;
      for (uint32_t row = 0; row < (t[0] & 7u); ++row) {
        const uint32_t* w = t + 1 + row * pv_row_words(groups, lights) + 3 + groups;  // a, F, R of the lamp
        const uint32_t inst = t[1 + row * pv_row_words(groups, lights)];
        if (inst == 1) {
          ++baked_rows;
          CHECK((w[0] == 0u && w[1] == 0u && w[2] == 0u));
        } else if (inst == 9) {
          lit_rows += bits_to_float(w[0]) > 0.0f;
        }
      }
      if (got[i].instance == 1) {
        ++baked_probes;
        CHECK(got[i].f == 0.0f);
      } else if (got[i].instance == 9) {
        lit_probes += got[i].f > 0.0f;
      }
    }
    CHECK((baked_rows > 1000 && lit_rows > 10 && baked_probes > 1000 && lit_probes > 10));
  }
}

TEST_CASE("T4: binary16 values, the linear pyramid's level 0 and the scale table (scans.md §4.1, §3.2)") {
  CHECK(binary16_to_double(0x3C00) == 1.0);
  CHECK(binary16_to_double(0x0001) == std::ldexp(1.0, -24));
  CHECK(binary16_to_double(0x03FF) == std::ldexp(1023.0, -24));
  CHECK(binary16_to_double(0x0400) == std::ldexp(1.0, -14));
  CHECK(binary16_to_double(0x7BFF) == 65504.0);
  CHECK(binary16_to_double(0x3555) == 0.333251953125);
  CHECK(binary16_to_double(0xC000) == -2.0);
  CHECK((binary16_to_double(0x8000) == 0.0 && !std::signbit(binary16_to_double(0x8000))));
  CHECK(std::isinf(binary16_to_double(0x7C00)));
  CHECK(std::isnan(binary16_to_double(0x7E00)));

  // Every finite non-negative binary16 value (0x0000-0x7BFF) once, in a 128 x 83 x 3 ramp, -0 at the end.
  constexpr int tw = 128, th = 83;
  std::vector<uint16_t> ramp(static_cast<size_t>(tw) * th * 3);
  for (size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<uint16_t>(i % 0x7C00);
  ramp.back() = 0x8000;
  auto oracle = [](uint16_t h) {  // (1 + m / 1024) 2^(e - 15), m 2^-24: another route to the same exact values
    const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
    return e == 0 ? m * std::pow(2.0, -24) : (1.0 + m / 1024.0) * std::pow(2.0, e - 15);
  };
  SceneBundle s = room(2);
  s.bodies.resize(2);
  Material& m = s.bodies[0].materials[0];
  m.texture.clear();
  m.texture_linear = ramp;
  m.texture_width = tw;
  m.texture_height = th;
  m.texture_channels = 3;
  bake(m, 75.0f, {0.3f, 0.7f});
  bake(s.bodies[1].materials[0], 2.5f, {});  // all in group 0
  SceneSettings settings;
  settings.sensor_rgb_weights = {0.52, 0.37, 0.11};
  const SceneData data(s, settings);
  const std::array<double, 3> wb{0.52 * static_cast<double>(0.9f), 0.37 * static_cast<double>(0.8f), 0.11};
  REQUIRE(data.materials()[0].texture == 0);
  const TextureLevel& level0 = data.texture_levels()[0][0];
  REQUIRE((level0.width == tw && level0.height == th));
  size_t mismatched = 0;
  for (size_t i = 0; i < static_cast<size_t>(tw) * th; ++i) {
    const auto want = static_cast<float>((wb[0] * oracle(ramp[3 * i]) + wb[1] * oracle(ramp[3 * i + 1])) +
                                         wb[2] * oracle(ramp[3 * i + 2]));
    mismatched += std::memcmp(&want, &data.texels()[level0.offset + i], sizeof(float)) != 0;
  }
  CHECK(mismatched == 0);
  // The scale table: S_g = fl32(double(K) double(a_g)) per baked material, in material order; lit materials -1.
  const std::vector<float> want{static_cast<float>(75.0 * static_cast<double>(0.3f)),
                                static_cast<float>(75.0 * static_cast<double>(0.7f)), 2.5f, 0.0f};
  CHECK(same_bits(data.baked_scale(), want));
  CHECK((data.materials()[0].baked == 0 && data.materials()[1].baked == 2));
  const SceneView v = data.host_view();
  CHECK((v.has_baked == 1 && v.baked_scale == data.baked_scale().data()));
  const SceneData lit(room(2), settings);
  CHECK((lit.host_view().has_baked == 0 && lit.host_view().baked_scale == nullptr && lit.materials()[0].baked == -1));
}

namespace {

// scans.md §4.2's geometry split (S3a-1): room(2)'s plane a baked scan (K 75, shares 0.3 / 0.7); the sphere, the box
// and a card (id 4) lit inserted static objects; a lit mover (id 9) between the lamp and the card; the lamp (group 1)
// and a sun (group 0) cast shadows: the box shades the sphere from the sun, the mover the card from the lamp. `proxy`:
// the lamp is the scan's proxy, which splits the sphere, the box and the card from the world (then the plane alone).
// `turned`: the camera turned and moved off the world origin, so that camera-frame shadow-ray origins are rounded.
SceneBundle split_scene(bool proxy, bool turned = false) {
  SceneBundle s = room(2);
  if (turned) {
    s.trajectories[0] = static_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.03, -0.05, 0.02}), {0.04, -0.03, -0.05});
  }
  bake(s.bodies[0].materials[0], 75.0f, {0.3f, 0.7f});
  s.bodies[0].scan = true;
  Body card = make_body(4, plane_mesh(0.3, 3), {0.7f, 0.6f, 0.5f});
  card.T_world_body = translation(-0.1, -0.12, 2.05);
  s.bodies.push_back(card);
  s.trajectories.push_back(moving_trajectory(kT0, kT1, quat_identity<double>(), {-0.6, -0.55, 1.9}, {4.0, 0.0, 0.0},
                                             {0.0, 0.0, 0.0}));
  Body mover = make_body(9, box_mesh(0.12f, 0.12f, 0.12f), {0.6f, 0.6f, 0.6f});
  mover.trajectory = 1;
  s.bodies.push_back(mover);
  Light lamp = point_light({0.0, -1.2, 1.6}, 2.0f, true, 1);
  lamp.proxy = proxy;
  s.lights.push_back(lamp);  // shadow bit 0
  Light sun;
  sun.kind = BundleLightKind::kDirectional;
  sun.direction = {0.75, 0.25, 0.2};  // from the box's centre through the sphere's
  sun.intensity = 0.8f;
  s.lights.push_back(sun);  // shadow bit 1
  s.lights.push_back(ambient_light(0.2f, 0));
  return s;
}

constexpr uint32_t kPlane = 1, kSphere = 2, kBox = 3, kCard = 4, kMover = 9;

bool lit_receiver(uint32_t id) { return id == kSphere || id == kBox || id == kCard || id == kMover; }

}  // namespace

TEST_CASE("the scan / non-scan split: trace geometries, views and pose rows (scans.md §4.2)") {
  SceneSettings settings;
  const SceneData split(split_scene(true), settings), merged(split_scene(false), settings);
  REQUIRE(split.baked_proxies());
  REQUIRE_FALSE(merged.baked_proxies());
  const SceneView sv = split.host_view(), mv = merged.host_view();
  CHECK((sv.baked_proxies == 1 && mv.baked_proxies == 0));
  // Without proxies: the world (bodies 1-4) and the mover. With them: the world holds the scan; the sphere, the box and
  // the card follow among the movers by id, static (trajectory -1, dynamic 0) with finite spheres.
  REQUIRE(merged.geometries().size() == 2);
  REQUIRE(split.geometries().size() == 5);
  const std::vector<int32_t> want_split{0, 1, 2, 3, 4}, want_merged{0, 0, 0, 0, 1};
  for (size_t i = 0; i < 5; ++i) {
    CHECK(split.bodies()[i].geometry == want_split[i]);
    CHECK(merged.bodies()[i].geometry == want_merged[i]);
  }
  CHECK((mv.geometries[0].dynamic == 0 && mv.geometries[1].dynamic == 1));
  CHECK((sv.geometries[0].dynamic == 0 && sv.geometries[4].dynamic == 1 && std::isinf(sv.geometries[0].sphere_radius)));
  const std::vector<uint32_t> ids{kPlane, kSphere, kBox, kCard, kMover};
  for (int32_t g = 1; g <= 3; ++g) {
    INFO("split geometry " << g);
    const GeometryData& d = split.geometries()[static_cast<size_t>(g)];
    CHECK((d.trajectory == -1 && d.instance == ids[static_cast<size_t>(g)] && d.instances.empty()));
    CHECK((sv.geometries[g].dynamic == 0 && sv.geometries[g].instances == nullptr &&
           std::isfinite(sv.geometries[g].sphere_radius) && sv.geometries[g].sphere_radius > 0.0f));
  }
  for (uint32_t t : split.geometries()[0].instances) CHECK(t == kPlane);
  // Each split body's triangles are the merged world's of that body: the same vertices bit for bit and materials.
  const GeometryData& world = merged.geometries()[0];
  for (int32_t g = 0; g <= 3; ++g) {
    INFO("body " << ids[static_cast<size_t>(g)]);
    const GeometryData& d = split.geometries()[static_cast<size_t>(g)];
    size_t k = 0;
    for (size_t t = 0; t < world.num_triangles(); ++t) {
      if (world.instances[t] != ids[static_cast<size_t>(g)]) continue;
      REQUIRE(k < d.num_triangles());
      CHECK(world.materials[t] == d.materials[k]);
      for (size_t c = 0; c < 3; ++c) {
        CHECK(std::memcmp(&world.positions[3 * world.indices[3 * t + c]], &d.positions[3 * d.indices[3 * k + c]],
                          3 * sizeof(float)) == 0);
      }
      ++k;
    }
    CHECK(k == d.num_triangles());
  }
  // Pose rows: a split body's row is the world's (same frame, twist), its sphere centred at R_CW (c_W - p_WC) and
  // holding every vertex; the world keeps centre 0 and radius +inf. The camera is the world frame here.
  const int64_t t = kT0 + 100'000'000;
  std::vector<PoseRow> rows(5), rows_merged(2);
  std::vector<LightRow> lights(2), lights_merged(2);
  REQUIRE(compute_slot(split.pose_view(), 0, t, rows.data(), lights.data()));
  REQUIRE(compute_slot(merged.pose_view(), 0, t, rows_merged.data(), lights_merged.data()));
  CHECK(std::memcmp(&rows[0], &rows_merged[0], sizeof(PoseRow)) == 0);
  CHECK(std::memcmp(&rows[4], &rows_merged[1], sizeof(PoseRow)) == 0);
  CHECK((rows[0].sphere.x == 0.0f && rows[0].sphere.y == 0.0f && rows[0].sphere.z == 0.0f &&
         std::isinf(rows[0].sphere.w)));
  for (int32_t g = 1; g <= 3; ++g) {
    INFO("split geometry " << g);
    const PoseRow& r = rows[static_cast<size_t>(g)];
    CHECK(std::memcmp(r.m, rows[0].m, sizeof(r.m)) == 0);
    CHECK(std::memcmp(&r.omega, &rows[0].omega, sizeof(Float4)) == 0);
    CHECK(std::memcmp(&r.v, &rows[0].v, sizeof(Float4)) == 0);
    const GeometryData& d = split.geometries()[static_cast<size_t>(g)];
    CHECK(r.sphere.w == d.radius);
    const double c[3] = {static_cast<double>(r.sphere.x), static_cast<double>(r.sphere.y),
                         static_cast<double>(r.sphere.z)};
    const double want[3] = {d.centre.x, d.centre.y, d.centre.z};
    for (int i = 0; i < 3; ++i) CHECK(std::abs(c[i] - want[i]) <= 1e-6);
    // The largest |X_cam - centre| over the vertices (X_cam = R^T (X - p)), against sphere_overlaps' inflated radius.
    double worst = 0.0;
    for (size_t v = 0; v < d.num_vertices(); ++v) {
      double x_cam[3] = {0.0, 0.0, 0.0};
      for (int i = 0; i < 3; ++i) {
        const float m[4] = {r.m[i].x, r.m[i].y, r.m[i].z, r.m[i].w};
        const double dx = static_cast<double>(d.positions[3 * v + static_cast<size_t>(i)]) - static_cast<double>(m[3]);
        for (int j = 0; j < 3; ++j) x_cam[j] += static_cast<double>(m[j]) * dx;
      }
      const double e[3] = {x_cam[0] - c[0], x_cam[1] - c[1], x_cam[2] - c[2]};
      worst = std::max(worst, std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]));
    }
    CHECK(worst <= static_cast<double>(r.sphere.w * 1.0001f + 1e-5f));
    CHECK(worst > 0.9 * static_cast<double>(r.sphere.w));  // a tight bound (about the AABB centre)
  }
}

TEST_CASE("lit receivers render bit for bit with and without the split, on every backend (scans.md §4.2)") {
  constexpr int64_t ta = kT0 + 100'000'000, tb = kT0 + 160'000'000;
  for (BackendKind backend : backends()) {
    for (const auto& [t, turned] :
         {std::pair{ta, false}, std::pair{tb, false}, std::pair{ta, true}, std::pair{tb, true}}) {
      INFO(backend_name(backend) << ", t " << t << std::string(turned ? ", turned camera" : ""));
      const RenderFlags flags{true, false, true};
      const RenderResult split = render_all(split_scene(true, turned), options(backend), t, flags);
      const RenderResult merged = render_all(split_scene(false, turned), options(backend), t, flags);
      // Pixel-samples off the plane (its proxy bits are traced, and its radiance takes U2's factor, only with the
      // proxy): irradiance, signatures, sub-sample instances and class words.
      size_t off_plane = 0, on[10] = {};
      for (size_t i = 0; i < split.count; ++i) {
        bool plane = false;
        for (uint32_t k = 0; k < split.subsamples; ++k) {
          const uint32_t id = merged.sub_instance[i * split.subsamples + k];
          plane = plane || id == kPlane;
          if (id < 10) ++on[id];
        }
        if (plane) continue;
        ++off_plane;
        CHECK(std::memcmp(&split.irradiance[2 * i], &merged.irradiance[2 * i], 2 * sizeof(float)) == 0);
        CHECK(split.signature[i] == merged.signature[i]);
        CHECK(split.sample_class[i] == merged.sample_class[i]);
        CHECK(std::equal(split.sub_instance.begin() + static_cast<ptrdiff_t>(i * split.subsamples),
                         split.sub_instance.begin() + static_cast<ptrdiff_t>((i + 1) * split.subsamples),
                         merged.sub_instance.begin() + static_cast<ptrdiff_t>(i * split.subsamples)));
      }
      CHECK(off_plane > 500);
      CHECK((on[kSphere] > 1000 && on[kBox] > 1000 && on[kCard] > 1000 && on[kMover] > 500));
    }
  }
}

TEST_CASE("the split's shadow masks of lit receivers equal the merged world's; their static words are 0 (Embree)") {
  // Tracer::sample on both scenes: equal hits, and on lit receivers equal masks, among them the box's sun shadow on
  // the sphere (a split blocker) and the mover's lamp shadow on the card. The static words of the split scene are 0
  // on lit receivers and misses (and here on the plane: nothing static blocks the lamp, so no group has a word); shading
  // with them equals shading without wherever no baked sub-sample is shadowed, and the plane's shadowed pixels take U2's
  // factor (a lower irradiance in the lamp's group).
  const int64_t t_w = time::window_start(time::window_index(kT0 + 100'000'000));
  const WindowClock clock{t_w, 10, 0, 0};
  ThreadPool pool(4);
  RendererOptions o = options(BackendKind::kEmbree);
  const uint32_t n_sub = o.scene.n_subsamples;
  struct Traced {
    std::vector<HitRecord> hits;
    std::vector<uint32_t> occluded, statics;
    std::vector<float> irr;
    std::vector<uint64_t> sig;
  };
  auto trace = [&](const SceneData& data, bool statics) {
    HostExecutor host(pool);
    host.upload_scene(data);
    EmbreeTracer tracer(pool);
    tracer.build(data, host.scene_view());
    std::vector<SampleRequest> req(kW * kH);
    for (uint32_t p = 0; p < req.size(); ++p) req[p] = {0, 0, p, 0, n_sub, 0, 0};
    host.begin_window(0, clock, nullptr);
    host.assign_pose_rows(req.data(), static_cast<uint32_t>(req.size()), nullptr);
    const TraceBatch batch{req.data(), static_cast<uint32_t>(req.size()), n_sub, host.window_poses(), nullptr, 0};
    Traced out;
    const size_t subs = req.size() * n_sub;
    out.hits.resize(subs);
    out.occluded.resize(subs);
    uint32_t* st = statics ? host.static_occlusion_buffer(0, subs) : nullptr;
    if (st != nullptr) std::fill(st, st + subs, 0xFFFFFFFFu);
    tracer.sample(batch, out.hits.data(), out.occluded.data(), st);
    if (st != nullptr) out.statics.assign(st, st + subs);
    const auto groups = static_cast<size_t>(data.num_groups());
    out.irr.resize(req.size() * groups);
    out.sig.resize(req.size());
    host.shade_into(0, batch, out.hits.data(), out.occluded.data(), out.irr.data(), out.sig.data(), nullptr, nullptr,
                    nullptr, nullptr, st);
    if (statics) {
      std::vector<float> irr(out.irr.size());
      std::vector<uint64_t> sig(out.sig.size());
      host.shade_into(0, batch, out.hits.data(), out.occluded.data(), irr.data(), sig.data(), nullptr, nullptr,
                      nullptr);
      size_t baked_shadowed = 0, equal = 0, off_plane = 0;
      for (size_t i = 0; i < req.size(); ++i) {
        bool plane = false, shadowed = false;
        for (uint32_t k = 0; k < n_sub; ++k) {
          const size_t j = i * n_sub + k;
          plane = plane || out.hits[j].instance == kPlane;
          shadowed = shadowed || (out.hits[j].instance == kPlane && out.occluded[j] != 0);
        }
        if (!plane) {
          ++off_plane;
          equal += std::memcmp(&irr[i * groups], &out.irr[i * groups], groups * sizeof(float)) == 0 &&
                   sig[i] == out.sig[i];
        } else if (shadowed) {
          ++baked_shadowed;
          CHECK(out.irr[i * groups + 1] < irr[i * groups + 1]);  // the lamp's group: its proxy shadow
        }
      }
      CHECK(equal == off_plane);
      CHECK(baked_shadowed > 100);
    }
    return out;
  };
  for (const bool turned : {false, true}) {
    const std::string pose = turned ? "turned camera" : "camera at the origin";
    INFO(pose);
    const SceneData split(split_scene(true, turned), o.scene), merged(split_scene(false, turned), o.scene);
    const Traced a = trace(split, true), b = trace(merged, false);
    REQUIRE(a.statics.size() == a.hits.size());
    for (size_t i = 0; i < a.hits.size(); ++i) {
      if (lit_receiver(a.hits[i].instance) || a.hits[i].instance == kInstanceNone) CHECK(a.statics[i] == 0u);
    }
    size_t sun_on_sphere = 0, lamp_on_card = 0, lit = 0;
    for (size_t i = 0; i < a.hits.size(); ++i) {
      REQUIRE(a.hits[i].instance == b.hits[i].instance);
      CHECK(a.hits[i].t_hit == b.hits[i].t_hit);
      if (!lit_receiver(a.hits[i].instance)) continue;
      ++lit;
      CHECK(a.occluded[i] == b.occluded[i]);
      sun_on_sphere += a.hits[i].instance == kSphere && (a.occluded[i] & 2u) != 0;
      lamp_on_card += a.hits[i].instance == kCard && (a.occluded[i] & 1u) != 0;
    }
    MESSAGE(pose << ": lit sub-samples " << lit << ", the box's sun shadow on the sphere " << sun_on_sphere
            << ", the mover's lamp shadow on the card " << lamp_on_card);
    CHECK(sun_on_sphere > 200);
    CHECK(lamp_on_card > 200);
  }
}

TEST_CASE("the plan pass: a split static blocker is static, never a mover (lights.h shadow_states), on every backend") {
  const int64_t t = kT0 + 100'000'000;
  std::vector<GtQuery> q(kW * kH);
  // Off the pixel centres: row 26's centre rays run exactly along the box's lower edge (y = 0.05 at z = 1.2; f = 60), where
  // the hit ties between the front face and the back-facing +y one and the offset origin of the shadow ray is rounded to
  // either side of the front face (a self hit at ~1e-9 m on 8 pixels of OptiX's split scene; DESIGN 12.1 L2 excludes edges).
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = {t, static_cast<float>(i % kW) + 0.29f, static_cast<float>(i / kW) + 0.37f};
  }
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    // On OptiX the centre-shadow raygen maps a blocker by the device executor's GeometryView::dynamic: device buffers.
    auto states = [&](const SceneBundle& s, std::vector<GtSample>& gt) {
      Renderer r(s, options(backend, 4));
      REQUIRE(r.num_shadow_lights() == 2);
      std::vector<uint32_t> out(q.size() * 4, 99);
      gt.resize(q.size());
      if (backend == BackendKind::kEmbree) {
        r.gt_into(0, q.data(), q.size(), gt.data(), nullptr, out.data());
        return out;
      }
#if BALLA_RENDER_HAVE_OPTIX
      cudaStream_t stream = nullptr;
      GtQuery* dq = nullptr;
      GtSample* dg = nullptr;
      uint32_t* ds = nullptr;
      REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
      REQUIRE(cudaMalloc(&dq, q.size() * sizeof(GtQuery)) == cudaSuccess);
      REQUIRE(cudaMalloc(&dg, q.size() * sizeof(GtSample)) == cudaSuccess);
      REQUIRE(cudaMalloc(&ds, out.size() * sizeof(uint32_t)) == cudaSuccess);
      REQUIRE(cudaMemcpy(dq, q.data(), q.size() * sizeof(GtQuery), cudaMemcpyHostToDevice) == cudaSuccess);
      REQUIRE(cudaMemcpy(ds, out.data(), out.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess);
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);  // pageable copies may land after cudaMemcpy returns
      r.gt_into(0, dq, q.size(), dg, stream, ds);
      REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
      REQUIRE(cudaMemcpy(gt.data(), dg, q.size() * sizeof(GtSample), cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(out.data(), ds, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
      for (void* p : {static_cast<void*>(dq), static_cast<void*>(dg), static_cast<void*>(ds)}) cudaFree(p);
      cudaStreamDestroy(stream);
#endif
      return out;
    };
    std::vector<GtSample> gt_split, gt_merged;
    const auto split = states(split_scene(true), gt_split), merged = states(split_scene(false), gt_merged);
    size_t lit = 0, card_mover = 0, sphere_static = 0;
    for (size_t i = 0; i < q.size(); ++i) {
      REQUIRE(gt_split[i].instance == gt_merged[i].instance);
      for (uint32_t b = 0; b < 2; ++b) {
        const uint32_t s = split[4 * i + 2 * b], m = merged[4 * i + 2 * b];
        for (uint32_t id : {kSphere, kBox, kCard}) CHECK((s & ~kShadowMulti) != kShadowMover + id);
        if (!lit_receiver(gt_split[i].instance)) continue;
        ++lit;
        if (m == kShadowLit || m == kShadowStatic) {
          CHECK(s == m);
        } else {  // a mover blocks: in the split the same, or a split static body blocks too (static then)
          CHECK(((s & ~kShadowMulti) == (m & ~kShadowMulti) || s == kShadowStatic));
        }
        card_mover += gt_split[i].instance == kCard && b == 0 && s == kShadowMover + kMover;
        sphere_static += gt_split[i].instance == kSphere && b == 1 && s == kShadowStatic;
      }
    }
    CHECK(lit > 1000);
    CHECK(card_mover > 5);
    CHECK(sphere_static > 5);
  }
}

TEST_CASE("no mover is held in a scene with baked proxies (Renderer::hold, scans.md §4.2)") {
  const int64_t t = kT0 + 100'000'000;
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    Renderer split(split_scene(true), options(backend)), merged(split_scene(false), options(backend));
    CHECK_FALSE(split.holds_movers());  // known before the build
    CHECK(merged.holds_movers());
    std::vector<uint16_t> cam(kW * kH, 0);
    std::vector<uint32_t> pixel(kW * kH);
    std::vector<int64_t> ts(kW * kH, t);
    for (uint32_t p = 0; p < pixel.size(); ++p) pixel[p] = p;
    const RenderResult before = split.render(cam, pixel, ts);
    CHECK_FALSE(split.hold(std::vector<uint32_t>{kMover, kSphere}, t));  // the guard: the movers stay apart
    const RenderResult after = split.render(cam, pixel, ts);
    CHECK(same_bits(before.irradiance, after.irradiance));
    CHECK(before.signature == after.signature);
    CHECK(split.held_device_bytes() == 0);
    CHECK(merged.hold(std::vector<uint32_t>{kMover}, t));  // without proxies the mover joins the world (I3b)
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// U2 (scans.md §4.2, increment S3a-2).

namespace {

// A baked scan plane (id 1, K 2, one group) at z = 3 under two proxy point lights (group 0, shadow bits 0 and 1) and a
// proxy fill of `fill` (none for 0), plus a non-proxy ambient that is not part of the fill.
SceneBundle micro_scene(float fill) {
  SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
  s.cameras.push_back(pinhole(kW, kH, 60.0, 0));
  Body plane = make_body(1, plane_mesh(2.5, 1), {0.9f, 0.7f, 0.4f});
  bake(plane.materials[0], 2.0f, {});
  plane.scan = true;
  plane.T_world_body = translation(0, 0, 3.0);
  s.bodies.push_back(plane);
  Light a = point_light({-0.4, -0.3, 0.0}, 3.0f);
  a.proxy = true;
  s.lights.push_back(a);
  Light b = point_light({0.5, 0.2, 0.5}, 1.5f);
  b.proxy = true;
  s.lights.push_back(b);
  s.lights.push_back(ambient_light(0.1f));
  if (fill > 0.0f) {
    Light f = ambient_light(fill);
    f.proxy = true;
    s.lights.push_back(f);
  }
  return s;
}

}  // namespace

TEST_CASE("U2's arithmetic: accumulate_baked and cluster_radiance on hand-made sums (scans.md §4.2)") {
  constexpr double kInvPi = 0.318309886183790672;
  for (const float fill : {0.3f, 0.0f}) {
    INFO("fill " << fill);
    SceneSettings settings;
    settings.sensor_rgb_weights = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    const SceneData data(micro_scene(fill), settings);
    const SceneView v = data.host_view();
    // The tables of scans.md §3.2: the fill is the proxy ambient only, M_g the two proxies' bits, b_g the first.
    CHECK(v.baked_proxies == 1);
    CHECK(v.fill[0] == fill);
    CHECK(v.proxy_mask[0] == 3u);
    CHECK(v.proxy_slot[0] == 0);
    CHECK((v.proxy_slot[1] == -1 && v.proxy_mask[1] == 0u && v.fill[1] == 0.0f));
    std::vector<PoseRow> rows(data.geometries().size());
    std::vector<LightRow> lrows(data.lights().size());
    REQUIRE(compute_slot(data.pose_view(), 0, kT0 + 1000, rows.data(), lrows.data()));
    const PoseTableView poses{rows.data(), lrows.data(), static_cast<int32_t>(rows.size()),
                              static_cast<int32_t>(lrows.size())};
    const Vec3f p{0.2f, 0.1f, 3.0f}, n_front{0.0f, 0.0f, -1.0f}, n_back{0.0f, 0.0f, 1.0f};
    const double a_fill = static_cast<double>(fill);
    // f_l = I / d^2 max(0, n . omega) for the camera-facing normal (0, 0, -1), in fp64 from the float positions.
    double f_l[2];
    for (int l = 0; l < 2; ++l) {
      const double dx = static_cast<double>(lrows[static_cast<size_t>(l)].position[0]) - static_cast<double>(p.x);
      const double dy = static_cast<double>(lrows[static_cast<size_t>(l)].position[1]) - static_cast<double>(p.y);
      const double dz = static_cast<double>(lrows[static_cast<size_t>(l)].position[2]) - static_cast<double>(p.z);
      const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
      f_l[l] = static_cast<double>(data.lights()[static_cast<size_t>(l)].intensity) / (d * d) * (-dz / d);
    }
    constexpr int32_t kSlot = ClusterSums<2>::kTerms;
    const auto add = [&](ClusterSums<2>& c, Vec3f n, float rho, uint32_t occ, uint32_t st, bool weighted) {
      accumulate_baked(c, v, poses, 0, p, n, rho, occ, st, weighted);
    };
    const auto factor = [&](Vec3f n, uint32_t occ, uint32_t st) {
      ClusterSums<2> c;
      c.clear();
      add(c, n, 1.0f, occ, st, false);
      CHECK(c.t[kSlot + 1] == 0.0f);  // the pair's second slot stays 0 in the pooled form
      CHECK((c.t[kSlot + 2] == 0.0f && c.t[kSlot + 3] == 0.0f));  // and the other pair is not written
      return static_cast<double>(c.t[kSlot]);
    };
    const auto F = [&](double blocked_f, double static_f) {  // n = (f - blocked - static)/pi + A over d = (f - static)/pi + A
      const double all = f_l[0] + f_l[1];
      const double nn = (all - blocked_f - static_f) * kInvPi + a_fill, dd = (all - static_f) * kInvPi + a_fill;
      return dd > 0.0 ? nn / dd : 1.0;
    };
    const auto close = [](double got, double want) { return std::abs(got - want) <= 2e-6 * std::max(1.0, want); };
    CHECK(factor(n_front, 0u, 0u) == 1.0);                // no blocked proxy: 1 exactly, whatever the static word
    CHECK(factor(n_front, 0u, 3u) == 1.0);
    CHECK(close(factor(n_front, 1u, 0u), F(f_l[0], 0.0)));              // proxy 1 blocked, proxy 2 lights
    CHECK(close(factor(n_front, 2u, 0u), F(f_l[1], 0.0)));
    CHECK(close(factor(n_front, 3u, 0u), F(f_l[0] + f_l[1], 0.0)));     // both blocked: the fill's share only
    CHECK(close(factor(n_front, 1u, 2u), F(f_l[0], f_l[1])));           // R4: proxy 2 static (the scan), proxy 1 inserted
    CHECK(close(factor(n_front, 2u, 1u), F(f_l[1], f_l[0])));
    if (fill > 0.0f) {
      CHECK(close(factor(n_front, 1u, 2u), a_fill / (f_l[0] * kInvPi + a_fill)));  // = A / (e1 / pi + A)
    } else {
      CHECK(factor(n_front, 3u, 0u) == 0.0);              // no fill, every proxy blocked: a black umbra
      CHECK(factor(n_front, 1u, 2u) == 0.0);              // (proxy 1 blocked, proxy 2 static: nothing lights it)
    }
    // A normal facing away from both lamps: f = 0, so d = fill: a factor of 1 (or the guard's 1 without a fill).
    CHECK(factor(n_back, 1u, 0u) == 1.0);
    CHECK(factor(n_back, 3u, 0u) == 1.0);

    // A cluster of four members with texel values T_k and states: lit; proxy 1 blocked; both blocked; proxy 1 blocked
    // with proxy 2 static. cluster_radiance: pooled (sum F / count) and, under texture_shadow weighted, the T-weighted
    // mean (sum T F / sum T); both exactly U0 when nothing is blocked.
    const float T[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const uint32_t occ[4] = {0u, 1u, 3u, 1u}, st[4] = {0u, 0u, 0u, 2u};
    const double Fk[4] = {1.0, F(f_l[0], 0.0), F(f_l[0] + f_l[1], 0.0), F(f_l[0], f_l[1])};
    const MaterialView& m = data.materials()[0];
    REQUIRE(m.baked == 0);
    const float scale = data.baked_scale()[0];
    const SampleRequest req{0, 0, 0, 0, 4, 0, 0};
    const auto cluster = [&](const SceneView& view, const uint32_t* o, const uint32_t* s, bool weighted, bool factor_on,
                             float* rad) {
      ClusterSums<2> c;
      c.clear();
      for (int k = 0; k < 4; ++k) {
        accumulate(c, view, poses, 0, p, n_front, T[k], 0u, true, false, false, false);  // the member terms: T in t[6]
        if (factor_on) add(c, n_front, T[k], o[k], s[k], weighted);
      }
      cluster_radiance(view, poses, req, m, true, 4, 4, c, rad, 1.0f, factor_on);
    };
    SceneView weighted_view = v;
    weighted_view.texture_shadow = 1;
    float u0[1], pooled[1], weighted[1], pooled_w[1];
    cluster(v, occ, st, false, false, u0);
    cluster(v, occ, st, false, true, pooled);
    cluster(weighted_view, occ, st, true, true, weighted);
    CHECK(u0[0] == (1.0f / 4.0f * 10.0f) * scale);
    const double t_bar = 2.5, base = t_bar * static_cast<double>(scale);
    CHECK(std::abs(static_cast<double>(pooled[0]) - base * (Fk[0] + Fk[1] + Fk[2] + Fk[3]) / 4.0) <= 1e-6 * base);
    const double tf = 1.0 * Fk[0] + 2.0 * Fk[1] + 3.0 * Fk[2] + 4.0 * Fk[3];
    CHECK(std::abs(static_cast<double>(weighted[0]) - base * tf / 10.0) <= 1e-6 * base);
    CHECK(std::abs(weighted[0] - pooled[0]) > 1e-4 * base);  // the covariance of T and F: the two forms differ
    // Nothing blocked: bit for bit U0, in both forms (x / x of equal sums).
    const uint32_t none[4] = {0u, 0u, 0u, 0u}, stat_only[4] = {3u, 0u, 1u, 2u};
    for (const uint32_t* s : {none, stat_only}) {
      cluster(v, none, s, false, true, pooled_w);
      CHECK(pooled_w[0] == u0[0]);
      cluster(weighted_view, none, s, true, true, pooled_w);
      CHECK(pooled_w[0] == u0[0]);
    }
  }
}

namespace {

constexpr double kInvPiD = 0.318309886183790672;

// An axis-aligned rectangle at z = const, facing the camera (a card), in the world frame at the rendered time.
struct Rect {
  double z, x0, x1, y0, y1;
  bool empty() const { return !(x1 > x0 && y1 > y0); }
};

// A proxy light: directional (v: propagation direction) or point (v: position).
struct Proxy {
  bool directional;
  std::array<double, 3> v;
  float intensity;
  int group;
};

// The T5 scene (scans.md §8): a baked scan plane (id 1; untextured, K 1.5) at z = 3 behind an inserted lit card (id 2) at
// z = `inserted.z` and optionally a baked scan card (id 3) between them, under proxy lights and a proxy fill per group.
// The camera is the world frame (static, identity), 64 x 48 pixels at f = 60.
struct U2Options {
  std::vector<Proxy> proxies;
  std::vector<float> fill{0.3f};          // proxy ambient per light group; 0: none
  std::vector<float> shares;              // the plane's a_g; empty: one group
  Rect inserted{2.0, -0.5, 0.1, -0.35, 0.15};
  Rect scan_card{2.5, 0.0, 0.0, 0.0, 0.0};  // empty: none
  bool mover = false;                     // the inserted card moves (1 m/s along x), at `inserted` at kRenderT
  bool baked_card = false;                // the inserted card a baked prop (K 1, untextured), not a scan
  bool textured_plane = false;            // the plane carries an 8-bit pattern (64 x 64 over 5 m)
  TextureShadow texture_shadow = TextureShadow::kPooled;
};

constexpr int64_t kRenderT = kT0 + 100'000'000;

Mesh rect_mesh(double w, double h) {
  Mesh m = plane_mesh(0.5, 1);
  for (size_t v = 0; v < m.positions.size() / 3; ++v) {
    m.positions[3 * v] = static_cast<float>(static_cast<double>(m.positions[3 * v]) * w);
    m.positions[3 * v + 1] = static_cast<float>(static_cast<double>(m.positions[3 * v + 1]) * h);
  }
  return m;
}

SceneBundle u2_bundle(const U2Options& op) {
  SceneBundle s;
  const size_t groups = std::max<size_t>(op.fill.size(), op.shares.size());
  s.light_groups = groups <= 1 ? std::vector<std::string>{"default"} : std::vector<std::string>{"default", "mains"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
  s.cameras.push_back(pinhole(kW, kH, 60.0, 0));
  Body plane = make_body(1, plane_mesh(2.5, 1), {0.9f, 0.7f, 0.4f});
  if (op.textured_plane) set_texture(plane.materials[0], pattern(64, 64, 2), 64, 64);
  bake(plane.materials[0], 1.5f, op.shares);
  plane.scan = true;
  plane.T_world_body = translation(0, 0, 3.0);
  s.bodies.push_back(plane);
  if (!op.inserted.empty()) {
    const Rect& r = op.inserted;
    Body card = make_body(2, rect_mesh(r.x1 - r.x0, r.y1 - r.y0), {0.7f, 0.6f, 0.5f});
    if (op.baked_card) bake(card.materials[0], 1.0f, op.shares);
    const double cx = 0.5 * (r.x0 + r.x1), cy = 0.5 * (r.y0 + r.y1);
    if (op.mover) {
      // 1 m/s along x, at (cx, cy, z) at kRenderT: p0 is corrected by the position the pose table measures (the spline
      // is linear in p0, so one step is exact).
      double p0x = cx - 1.0 * static_cast<double>(kRenderT - kT0) * 1e-9;
      for (int pass = 0; pass < 2; ++pass) {
        SceneBundle trial = s;
        trial.trajectories.push_back(
            moving_trajectory(kT0, kT1, quat_identity<double>(), {p0x, cy, r.z}, {1.0, 0.0, 0.0}, {0, 0, 0}));
        Body probe = card;
        probe.trajectory = 1;
        trial.bodies.push_back(probe);
        const SceneData data(trial, SceneSettings{});
        std::vector<PoseRow> rows(data.geometries().size());
        std::vector<LightRow> lrows(data.lights().size());
        compute_slot(data.pose_view(), 0, kRenderT, rows.data(), lrows.data());
        p0x += cx + static_cast<double>(rows[1].m[0].w);  // the body is at -p in the camera (= world) frame
      }
      s.trajectories.push_back(
          moving_trajectory(kT0, kT1, quat_identity<double>(), {p0x, cy, r.z}, {1.0, 0.0, 0.0}, {0, 0, 0}));
      card.trajectory = 1;
    } else {
      card.T_world_body = translation(cx, cy, r.z);
    }
    s.bodies.push_back(card);
  }
  if (!op.scan_card.empty()) {
    const Rect& r = op.scan_card;
    Body card = make_body(3, rect_mesh(r.x1 - r.x0, r.y1 - r.y0), {0.5f, 0.5f, 0.5f});
    bake(card.materials[0], 1.0f, op.shares);
    card.scan = true;
    card.T_world_body = translation(0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1), r.z);
    s.bodies.push_back(card);
  }
  for (const Proxy& p : op.proxies) {
    Light l;
    if (p.directional) {
      l.kind = BundleLightKind::kDirectional;
      l.direction = p.v;
    } else {
      l.kind = BundleLightKind::kPoint;
      l.position = p.v;
    }
    l.intensity = p.intensity;
    l.group = p.group;
    l.proxy = true;
    s.lights.push_back(l);
  }
  for (size_t g = 0; g < op.fill.size(); ++g) {
    if (op.fill[g] <= 0.0f) continue;
    Light a = ambient_light(op.fill[g], static_cast<int>(g));
    a.proxy = true;
    s.lights.push_back(a);
  }
  return s;
}

// Per sub-sample oracle (fp64): the proxies' f, I (an inserted card blocks), W (a scan card blocks, or the proxy lies
// behind the plane) at the plane point P, and F of the light group `group` by the rule of scans.md §4.2.
struct Oracle {
  struct Light {
    bool directional;
    std::array<double, 3> v;  // propagation direction (unit) or position, as the renderer's float light rows hold them
    double intensity;
    int group;
  };
  std::vector<Light> lights;
  std::vector<Rect> inserted, scan;
  std::vector<double> fill;

  // Whether P + t d (t in (0, t_max]) crosses the rectangle; near: within `margin` of a boundary or of the end.
  static bool crosses(const Rect& r, const std::array<double, 3>& p, const std::array<double, 3>& d, double t_max,
                      bool* near) {
    if (r.empty() || d[2] == 0.0) return false;
    const double t = (r.z - p[2]) / d[2];
    if (!(t > 0.0)) return false;
    const double x = p[0] + t * d[0], y = p[1] + t * d[1], margin = 3e-5;
    if (t > t_max + margin / std::abs(d[2])) return false;
    const double edge = std::min(std::min(std::abs(x - r.x0), std::abs(x - r.x1)),
                                 std::min(std::abs(y - r.y0), std::abs(y - r.y1)));
    const bool inside = x > r.x0 && x < r.x1 && y > r.y0 && y < r.y1;
    if (edge < margin && x > r.x0 - margin && x < r.x1 + margin && y > r.y0 - margin && y < r.y1 + margin) *near = true;
    return inside && t <= t_max;
  }

  // F of group g at P (normal toward the camera). near_edge is set when a blocking test is within the margin of an edge.
  struct LightState {
    double f;
    bool inserted, world;  // I and W of the proxy at the point
  };
  double factor(int g, const std::array<double, 3>& p, bool* near_edge, bool* any_blocked = nullptr,
                std::vector<LightState>* states = nullptr) const {
    using L = LightState;
    std::vector<L> ls;
    bool any_inserted = false;
    for (const Light& l : lights) {
      if (l.group != g) continue;
      // The shadow ray starts at the hit point offset by 256 ulps along the normal (ray.h offset_origin): towards the
      // camera for this plane, 256 * 2^-22 m at z = 3. f is that of the hit point itself.
      const std::array<double, 3> origin{p[0], p[1], p[2] - 256.0 * std::ldexp(1.0, -22)};
      std::array<double, 3> to{}, to_ray{};  // unit directions to the light from the hit point and from the origin
      double e = l.intensity, t_max = 1e300;
      if (l.directional) {
        to = to_ray = {-l.v[0], -l.v[1], -l.v[2]};
      } else {
        const std::array<double, 3> d{l.v[0] - p[0], l.v[1] - p[1], l.v[2] - p[2]};
        const std::array<double, 3> dr{l.v[0] - origin[0], l.v[1] - origin[1], l.v[2] - origin[2]};
        const double d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2], d2r = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
        e = l.intensity / d2;
        for (size_t c = 0; c < 3; ++c) {
          to[c] = d[c] / std::sqrt(d2);
          to_ray[c] = dr[c] / std::sqrt(d2r);
        }
        t_max = std::sqrt(d2r);
      }
      const double cos_i = -to[2];  // n = (0, 0, -1)
      L x{e * std::max(0.0, cos_i), false, false};
      if (!(cos_i > 0.0)) {
        x.world = true;  // self-occluded: W without a ray
      } else {
        for (const Rect& r : inserted) x.inserted = x.inserted || crosses(r, origin, to_ray, t_max, near_edge);
        for (const Rect& r : scan) x.world = x.world || crosses(r, origin, to_ray, t_max, near_edge);
      }
      any_inserted = any_inserted || x.inserted;
      ls.push_back(x);
    }
    if (any_blocked != nullptr) *any_blocked = any_inserted;
    if (states != nullptr) *states = ls;
    if (!any_inserted) return 1.0;
    double num = 0.0, den = 0.0;
    const double a = static_cast<size_t>(g) < fill.size() ? fill[static_cast<size_t>(g)] : 0.0;
    for (const L& x : ls) {
      if (x.world) continue;
      den += x.f;
      if (!x.inserted) num += x.f;
    }
    const double n = num * kInvPiD + a, d = den * kInvPiD + a;
    return d > 0.0 ? n / d : 1.0;
  }
};

struct U2Stats {
  size_t checked = 0;       // plane-only pixel-samples compared
  size_t lit = 0;           // ... in which no proxy of any group is blocked by an inserted object
  size_t umbra = 0;         // ... in which every sub-sample has one blocked
  size_t partial = 0;       // ... the rest
  size_t masked = 0;        // ... every sub-sample blocked by an inserted object and by the scan alike (F = 1)
  size_t exact = 0;         // ... with the result bit for bit U0 (no factor below 1)
  size_t skipped_edge = 0;  // plane-only pixel-samples skipped: a sub-sample within 3e-5 m of a blocker's edge
  size_t bad = 0;           // comparisons out of bound
  size_t card_pixels = 0, card_bad = 0;  // pixel-samples of the inserted card alone, baked: U0 bit for bit
  size_t r4 = 0, r4_bad = 0;  // two proxies in a group, proxy 0 static-blocked, proxy 1 inserted-blocked on every
                              // sub-sample (R4): the pixel is t S A / (f_1 / pi + A)
  double worst = 0.0;       // the largest error against the oracle, relative to the pixel's U0 value
  double total_min = 1e300, total_max = -1e300;  // umbra pixels: sum over groups of irradiance / sum of U0 values
};

// Renders `op` on `backend` and compares every plane-only pixel-sample with the oracle: the irradiance of each light
// group is (t S_g) times the mean of F over the pixel's N sub-samples, within the summation bound; pixels where no
// factor is below 1 are U0 bit for bit. A pixel with a sub-sample within 3e-5 m of a blocker's edge (the oracle and the
// tracer may differ there by a sub-sample) is skipped. A baked card (a prop or a mover) that nothing blocks renders U0.
U2Stats check_u2(const U2Options& op, BackendKind backend, uint32_t n_sub, PixelIntegral integral) {
  const SceneBundle b = u2_bundle(op);
  RendererOptions o = options(backend);
  o.scene.n_subsamples = n_sub;
  o.scene.pixel_integral = integral;
  o.scene.texture_shadow = op.texture_shadow;
  const RenderResult r = render_all(b, o, kRenderT, {true, false});
  const SceneData data(b, o.scene);
  const SceneView v = data.host_view();
  std::vector<PoseRow> rows(data.geometries().size());
  std::vector<LightRow> lrows(data.lights().size());
  REQUIRE(compute_slot(data.pose_view(), 0, kRenderT, rows.data(), lrows.data()));
  REQUIRE(lrows.size() == op.proxies.size());

  Oracle oracle;
  oracle.inserted = {op.inserted};  // a mover is at `inserted` at kRenderT by its trajectory
  oracle.scan = {op.scan_card};
  const auto groups = static_cast<size_t>(data.num_groups());
  for (size_t g = 0; g < groups; ++g) oracle.fill.push_back(static_cast<double>(v.fill[g]));
  for (size_t j = 0; j < lrows.size(); ++j) {
    const Proxy& p = op.proxies[j];
    Oracle::Light l;
    l.directional = p.directional;
    for (size_t c = 0; c < 3; ++c) {
      l.v[c] = static_cast<double>(p.directional ? lrows[j].direction[c] : lrows[j].position[c]);
    }
    l.intensity = static_cast<double>(data.lights()[j].intensity);
    l.group = p.group;
    oracle.lights.push_back(l);
  }

  const float albedo = data.materials()[0].albedo;
  const double tol = (2.0 * n_sub + 6.0) * std::ldexp(1.0, -24);
  U2Stats st;
  const float* const irr = r.irradiance.data();
  for (size_t i = 0; i < r.count; ++i) {
    bool plane_only = true, card_only = op.baked_card && !op.inserted.empty();
    for (uint32_t k = 0; k < n_sub; ++k) {
      plane_only = plane_only && r.sub_instance[i * n_sub + k] == 1;
      card_only = card_only && r.sub_instance[i * n_sub + k] == 2;
    }
    if (card_only) {  // the card's own sub-samples: nothing blocks its proxies (its own geometry is skipped)
      ++st.card_pixels;
      for (size_t g = 0; g < groups; ++g) {
        const float u0 = data.materials()[1].albedo * data.baked_scale()[static_cast<size_t>(data.materials()[1].baked) + g];
        st.card_bad += std::memcmp(&irr[i * groups + g], &u0, sizeof(float)) != 0;
      }
    }
    if (!plane_only) continue;
    std::vector<double> mean(groups, 0.0);
    bool near = false, all_lit = true, all_blocked = true, all_masked = true, r4_all = true;
    double r4_f = 0.0;
    for (uint32_t k = 0; k < n_sub; ++k) {
      const Vec3f d = subsample_direction(v, v.cameras[0], static_cast<uint32_t>(i), k);
      const double t = 3.0 / static_cast<double>(d.z);
      const std::array<double, 3> pt{static_cast<double>(d.x) * t, static_cast<double>(d.y) * t, 3.0};
      bool any_group = false, all_one = true;
      for (size_t g = 0; g < groups; ++g) {
        bool any = false;
        std::vector<Oracle::LightState> ls;
        const double f = oracle.factor(static_cast<int>(g), pt, &near, &any, &ls);
        if (g == 0 && ls.size() == 2) {
          const bool state = ls[0].world && ls[1].inserted && !ls[1].world;
          r4_all = r4_all && state;
          r4_f = ls[1].f;
        }
        mean[g] += f;
        any_group = any_group || any;
        all_one = all_one && f == 1.0;
      }
      all_lit = all_lit && !any_group;
      all_blocked = all_blocked && any_group;
      all_masked = all_masked && any_group && all_one;
    }
    if (near) {
      ++st.skipped_edge;
      continue;
    }
    ++st.checked;
    bool exact = true;
    double got_total = 0.0, base_total = 0.0;
    for (size_t g = 0; g < groups; ++g) {
      const double base = static_cast<double>(albedo * data.baked_scale()[g]);
      const double want = base * (mean[g] / static_cast<double>(n_sub));
      const double got = static_cast<double>(irr[i * groups + g]);
      got_total += got;
      base_total += base;
      const double err = base > 0.0 ? std::abs(got - want) / base : std::abs(got - want);
      st.worst = std::max(st.worst, err);
      st.bad += err > tol;
      if (mean[g] == static_cast<double>(n_sub)) {  // nothing blocked in group g: U0, bit for bit
        const float u0 = albedo * data.baked_scale()[g];
        st.bad += std::memcmp(&irr[i * groups + g], &u0, sizeof(float)) != 0;
      } else {
        exact = false;
      }
    }
    if (r4_all && oracle.lights.size() == 2) {  // the closed form of R4, apart from the oracle's own rule
      ++st.r4;
      const double a = oracle.fill[0];
      const double want = static_cast<double>(albedo * data.baked_scale()[0]) * (a / (r4_f * kInvPiD + a));
      st.r4_bad += std::abs(static_cast<double>(irr[i * groups]) - want) > 4e-7 * want;
    }
    st.exact += exact;
    st.lit += all_lit;
    st.umbra += all_blocked;
    st.partial += !all_lit && !all_blocked;
    st.masked += all_masked;
    if (all_blocked && !all_masked) {
      st.total_min = std::min(st.total_min, got_total / base_total);
      st.total_max = std::max(st.total_max, got_total / base_total);
    }
  }
  return st;
}

}  // namespace

TEST_CASE("T5: U2's shadow factor against analytic per-sub-sample values (scans.md §4.2)") {
  // Directions are propagation directions. (1, 0.5, 1) shifts a card's shadow on the plane by (z_plane - z_card) (1, 0.5).
  const Proxy sun{true, {1.0, 0.5, 1.0}, 2.0f, 0}, sun2{true, {-1.0, 0.5, 1.0}, 1.5f, 0};
  const Proxy lamp{false, {-1.0, -0.8, -2.0}, 20.0f, 0};  // 5 m behind the camera plane, off to the side
  const Rect scan_card{2.5, -0.9, -0.4, -0.2, 0.2};  // its sun shadow: x [-0.4, 0.1], y [0.05, 0.45]
  struct Case {
    std::string name;
    U2Options op;
    size_t min_umbra, min_partial, min_masked, min_lit;
  };
  std::vector<Case> cases;
  {  // A: one directional proxy, a static inserted card: partial coverage along its shadow's edges, an umbra inside
    U2Options op;
    op.proxies = {sun};
    op.inserted = {2.0, -1.4, -0.7, -0.5, 0.0};
    cases.push_back({"directional, static card", op, 20, 30, 0, 100});
  }
  {  // B: a point proxy 5 m from the plane (the expected mean within 4 N 2^-24: the tolerance below is that bound's)
    U2Options op;
    op.proxies = {lamp};
    op.inserted = {2.0, -0.5, -0.2, -0.3, 0.0};
    cases.push_back({"point proxy at 5 m", op, 20, 20, 0, 100});
  }
  {  // C: the scan card's static shadow edge crosses the inserted card's umbra and its edge (q1b case 1): the
     // inserted shadow lies in the static one but for a 0.02 m strip; inside it F = 1 exactly (a mover in a static shadow)
    U2Options op;
    op.proxies = {sun};
    op.inserted = {2.0, -1.3, -0.88, -0.4, -0.1};
    op.scan_card = scan_card;
    cases.push_back({"static edge x inserted edge", op, 0, 3, 20, 100});
  }
  {  // D: two proxies in one group (R4): the scan card's shadow of proxy 1 over the inserted card's shadow of proxy 2
    U2Options op;
    op.proxies = {sun, sun2};
    op.inserted = {2.0, 0.6, 1.1, -0.45, -0.05};
    op.scan_card = scan_card;
    cases.push_back({"two proxies in one group", op, 20, 10, 0, 100});
  }
  {  // E: the mover removed: no inserted object, so every pixel-sample is U0 bit for bit
    U2Options op;
    op.proxies = {sun};
    op.inserted = {2.0, 0.0, 0.0, 0.0, 0.0};
    op.scan_card = scan_card;
    cases.push_back({"no inserted object", op, 0, 0, 0, 2000});
  }
  {  // F, G, H: the inserted card a mover, a baked prop, a baked mover (it does not shadow itself)
    U2Options op;
    op.proxies = {sun};
    op.inserted = {2.0, -1.4, -0.7, -0.5, 0.0};
    op.mover = true;
    cases.push_back({"mover", op, 20, 30, 0, 100});
    op.mover = false;
    op.baked_card = true;
    cases.push_back({"baked prop", op, 20, 30, 0, 100});
    op.mover = true;
    cases.push_back({"baked mover", op, 20, 30, 0, 100});
  }
  for (BackendKind backend : backends()) {
    for (const auto& [n_sub, integral] : {std::pair{64u, PixelIntegral::kPhase1}, std::pair{256u, PixelIntegral::kFixed64}}) {
      for (const Case& c : cases) {
        INFO(backend_name(backend) << ", " << c.name << ", N = " << n_sub << (n_sub == 256 ? " fixed64" : ""));
        const U2Stats st = check_u2(c.op, backend, n_sub, integral);
        MESSAGE(backend_name(backend) << ", " << c.name << ", N " << n_sub << ": checked " << st.checked << " (lit " << st.lit
                                      << ", umbra " << st.umbra << ", partial " << st.partial << ", masked " << st.masked
                                      << ", U0 " << st.exact << "), skipped at an edge " << st.skipped_edge
                                      << ", worst error " << st.worst << ", card pixels " << st.card_pixels);
        CHECK(st.bad == 0);
        CHECK(st.card_bad == 0);
        CHECK(st.checked > 300);
        CHECK(st.umbra >= c.min_umbra);
        CHECK(st.partial >= c.min_partial / (n_sub == 256 ? 2 : 1));
        CHECK(st.masked >= c.min_masked);
        CHECK(st.lit >= c.min_lit);
        if (c.op.baked_card) CHECK(st.card_pixels > 50);
        CHECK(st.r4_bad == 0);
        if (c.name == "two proxies in one group") CHECK(st.r4 >= 10);  // A / (e2 / pi + A), proxy 1 static
      }
    }
  }
}

TEST_CASE("T5: two light groups, shares 0.3 / 0.7 and the example of scans.md §4.3") {
  // A proxy in 'mains' (group 1) darkens the mains share only: the default group is U0 bit for bit, and the total
  // ratio of a shadowed to a lit point is 1 - a (1 - F) with F = A / (E / pi + A): at A = 0.1841, E / pi = 0.4819 and
  // a = 0.7 it is 0.4935 (the design's example, whose helper is S4's).
  const double e_over_pi = 0.4819, fill = 0.1841;
  const double cos_i = 1.0 / 1.5;  // direction (1, 0.5, 1): n . omega
  U2Options op;
  op.shares = {0.3f, 0.7f};
  op.fill = {0.0f, static_cast<float>(fill)};
  op.proxies = {{true, {1.0, 0.5, 1.0}, static_cast<float>(std::numbers::pi * e_over_pi / cos_i), 1}};
  op.inserted = {2.0, -1.4, -0.7, -0.5, 0.0};
  const double f_umbra = fill / (e_over_pi + fill), want = 1.0 - 0.7 * (1.0 - f_umbra);
  CHECK(std::abs(want - 0.4935) < 1e-4);
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    const U2Stats st = check_u2(op, backend, 64, PixelIntegral::kPhase1);
    CHECK(st.bad == 0);
    CHECK(st.umbra >= 50);
    CHECK(st.partial >= 30);
    CHECK(std::abs(st.total_min - want) < 2e-6);
    CHECK(std::abs(st.total_max - want) < 2e-6);
  }
}

TEST_CASE("T5: accuracy.texture_shadow weighted on a textured baked plane (scans.md §4.2 (i))") {
  // The plane carries a pattern: inside a pixel T varies with F at a shadow edge. Pooled and weighted are both exactly
  // U0 where no proxy is blocked, and equal (to rounding) where F is the same on every sub-sample; across an edge
  // the weighted form follows the covariance of T and F. The pooled form is the oracle's mean of F times the mean T.
  U2Options op;
  op.proxies = {{true, {1.0, 0.5, 1.0}, 2.0f, 0}};
  op.inserted = {2.0, -1.4, -0.7, -0.5, 0.0};
  op.textured_plane = true;
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    U2Options lit = op;
    lit.proxies.clear();
    lit.fill = {};
    RendererOptions o = options(backend);
    const RenderResult u0 = render_all(u2_bundle(lit), o, kRenderT, {true, false});
    RenderResult by_form[2];
    for (const TextureShadow ts : {TextureShadow::kPooled, TextureShadow::kWeighted}) {
      op.texture_shadow = ts;
      RendererOptions ot = o;
      ot.scene.texture_shadow = ts;
      by_form[ts == TextureShadow::kWeighted ? 1 : 0] = render_all(u2_bundle(op), ot, kRenderT, {true, false});
    }
    size_t both_u0 = 0, edge = 0, differ = 0, equal_in_umbra = 0, umbra = 0;
    for (size_t i = 0; i < u0.count; ++i) {
      bool plane_only = true;
      for (uint32_t k = 0; k < u0.subsamples; ++k) plane_only = plane_only && u0.sub_instance[i * u0.subsamples + k] == 1;
      if (!plane_only) continue;
      const float a = u0.irradiance[i], p = by_form[0].irradiance[i], w = by_form[1].irradiance[i];
      CHECK(w <= a * (1.0f + 1e-6f));  // a shadow never brightens
      CHECK(p <= a * (1.0f + 1e-6f));
      if (p == a || w == a) {
        CHECK(p == a);  // no blocked proxy: both forms are U0 bit for bit
        CHECK(w == a);
        ++both_u0;
        continue;
      }
      ++edge;
      const double rel = std::abs(static_cast<double>(w) - static_cast<double>(p)) / static_cast<double>(a);
      differ += rel > 1e-3;
      // F the same on every sub-sample (an umbra pixel): the two forms agree to rounding
      const double f_umbra = 0.3 / (2.0 * (1.0 / 1.5) * 0.318309886183790672 + 0.3);
      if (std::abs(static_cast<double>(p) / static_cast<double>(a) - f_umbra) < 1e-5) {
        ++umbra;
        equal_in_umbra += rel < 2e-6;
      }
    }
    MESSAGE(backend_name(backend) << ": U0 " << both_u0 << ", shadowed " << edge << " (umbra " << umbra
                                  << "), weighted differs from pooled by > 1e-3 on " << differ);
    CHECK(both_u0 > 2000);
    CHECK(edge > 100);
    CHECK(differ > 5);
    CHECK(umbra > 30);
    CHECK(equal_in_umbra == umbra);
  }
}

TEST_CASE("the static word enters the signature through static_term; a zero word adds nothing (Embree)") {
  // The scan card's shadow holds the inserted card's (case C of T5): there the proxy is blocked by both, the occlusion
  // bit is 0 and the static bit 1. Shading the same hits with the static words, with them zeroed and without a buffer:
  // the signature differs exactly in pixel-samples with a non-zero word, and zeroed words equal no buffer.
  U2Options op;
  op.proxies = {{true, {1.0, 0.5, 1.0}, 2.0f, 0}};
  op.inserted = {2.0, -1.3, -0.88, -0.4, -0.1};
  op.scan_card = {2.5, -0.9, -0.4, -0.2, 0.2};
  const SceneBundle b = u2_bundle(op);
  ThreadPool pool(4);
  RendererOptions o = options(BackendKind::kEmbree);
  const uint32_t n_sub = o.scene.n_subsamples;
  const SceneData data(b, o.scene);
  HostExecutor host(pool);
  host.upload_scene(data);
  EmbreeTracer tracer(pool);
  tracer.build(data, host.scene_view());
  std::vector<SampleRequest> req(kW * kH);
  for (uint32_t p = 0; p < req.size(); ++p) req[p] = {0, 0, p, 0, n_sub, 0, 0};
  host.begin_window(0, WindowClock{time::window_start(time::window_index(kRenderT)), 10, 0, 0}, nullptr);
  host.assign_pose_rows(req.data(), static_cast<uint32_t>(req.size()), nullptr);
  const TraceBatch batch{req.data(), static_cast<uint32_t>(req.size()), n_sub, host.window_poses(), nullptr, 0};
  const size_t subs = req.size() * n_sub;
  std::vector<HitRecord> hits(subs);
  std::vector<uint32_t> occluded(subs);
  uint32_t* st = host.static_occlusion_buffer(0, subs);
  tracer.sample(batch, hits.data(), occluded.data(), st);
  std::vector<uint32_t> words(st, st + subs), zeros(subs, 0u);
  std::vector<float> irr(req.size() * static_cast<size_t>(data.num_groups()));
  auto shade = [&](const uint32_t* statics) {
    std::vector<uint64_t> sig(req.size());
    host.shade_into(0, batch, hits.data(), occluded.data(), irr.data(), sig.data(), nullptr, nullptr, nullptr, nullptr,
                    statics);
    return sig;
  };
  const std::vector<uint64_t> with = shade(words.data()), without = shade(nullptr), zeroed = shade(zeros.data());
  CHECK(zeroed == without);  // a zero word adds nothing to any signature
  size_t nonzero = 0, differ = 0;
  for (size_t i = 0; i < req.size(); ++i) {
    bool any = false;
    for (uint32_t k = 0; k < n_sub; ++k) {
      any = any || words[i * n_sub + k] != 0u;
    }
    nonzero += any;
    differ += with[i] != zeroed[i];
    CHECK((with[i] != zeroed[i]) == any);
  }
  MESSAGE("pixel-samples with a static word: " << nonzero << ", signatures that differ: " << differ);
  CHECK(nonzero > 20);
}

TEST_CASE("a probe's mask on a baked receiver is render_into's occlusion word (renderer.cpp probe_chunks, lights.h)") {
  // A scene with baked proxies traces a baked receiver's occlusion bits by U2's rule (the inserted objects, minus the
  // scan's own shadow), and probes are the renderer's own sub-sample states: their masks are those words, not the lit
  // rule's (any geometry). Here a scan card shadows the scan plane from the sun proxy; in the second run an inserted
  // card shadows another part of it (the sun's shadow of the card at z 2.0 lies at x [-0.4, 0.3], y [0, 0.5], the scan
  // card's at y [-0.45, -0.05]). The oracle is the Embree tracer's sample with static words on the same one-sub-sample
  // requests, and the lit rule (no buffer) must differ from it where the scan card's shadow lies.
  const Proxy sun{true, {1.0, 0.5, 1.0}, 2.0f, 0};
  constexpr uint32_t kScanCard = 3;  // u2_bundle's ids: the plane 1, the inserted card 2, the scan card 3
  const std::array<uint32_t, 4> ks{0, 21, 42, 63};
  for (const bool inserted : {false, true}) {
    U2Options op;
    op.proxies = {sun};
    op.scan_card = inserted ? Rect{2.5, -0.9, -0.4, -0.7, -0.3} : Rect{2.5, -0.9, -0.4, -0.2, 0.2};
    op.inserted = inserted ? Rect{2.0, -1.4, -0.7, -0.5, 0.0} : Rect{2.0, 0.0, 0.0, 0.0, 0.0};
    const SceneBundle b = u2_bundle(op);
    const int64_t t_w = time::window_start(time::window_index(kRenderT));
    const WindowClock clock{t_w, 10, 0, 0};
    std::vector<SampleRequest> req, probes;
    for (uint32_t p = 0; p < static_cast<uint32_t>(kW * kH); ++p) {
      for (const uint32_t k : ks) {
        req.push_back({0, 0, p, k, 1, 0, 0});
        probes.push_back({0, static_cast<uint16_t>(RequestKind::kProbe), p, k, 1, 0, 0});  // the sun
      }
    }
    // The oracle: the Embree tracer's words on the same requests, with the static buffer (U2) and without (the lit rule).
    ThreadPool pool(4);
    const RendererOptions oe = options(BackendKind::kEmbree);
    const SceneData data(b, oe.scene);
    REQUIRE(data.baked_proxies());
    HostExecutor host(pool);
    host.upload_scene(data);
    EmbreeTracer tracer(pool);
    tracer.build(data, host.scene_view());
    host.begin_window(0, clock, nullptr);
    host.assign_pose_rows(req.data(), static_cast<uint32_t>(req.size()), nullptr);
    const TraceBatch batch{req.data(), static_cast<uint32_t>(req.size()), 1, host.window_poses(), nullptr, 0};
    std::vector<HitRecord> hits(req.size());
    std::vector<uint32_t> u2(req.size()), lit(req.size());
    uint32_t* st = host.static_occlusion_buffer(0, req.size());
    tracer.sample(batch, hits.data(), u2.data(), st);
    tracer.sample(batch, hits.data(), lit.data(), nullptr);
    size_t differing = 0, shadowed = 0;
    for (size_t i = 0; i < req.size(); ++i) {
      if (hits[i].instance != kPlane) continue;
      differing += u2[i] != lit[i];
      shadowed += u2[i] != 0;
    }
    MESSAGE(std::string(inserted ? "inserted card" : "no inserted object")
            << ": plane probes whose lit-rule mask differs from U2's " << differing << ", with a non-zero U2 word "
            << shadowed);
    CHECK(differing > 100);  // the scan card's own shadow: the lit rule sets the bit, the U2 word leaves it out
    if (inserted) {
      CHECK(shadowed > 20);
    } else {
      CHECK(shadowed == 0);  // nothing inserted: no ray's W is ever needed, every word is 0
    }
    for (BackendKind backend : backends()) {
      INFO(backend_name(backend) << ", " << (inserted ? "inserted card" : "no inserted object"));
      Renderer r(b, options(backend));
      r.begin_window(0, clock, nullptr);
      r.assign_pose_rows(probes.data(), probes.size(), nullptr);
      std::vector<ProbeOut> got(probes.size());
      r.probe_into(probes.data(), probes.size(), got.data(), nullptr);
      r.take_render_stats();
      size_t bad = 0, plane = 0;
      for (size_t i = 0; i < probes.size(); ++i) {
        const bool baked = hits[i].instance == kPlane || hits[i].instance == kScanCard;
        plane += baked;
        bad += got[i].mask != u2[i];
        if (!inserted && baked) CHECK(got[i].mask == 0u);
      }
      CHECK(plane > 2000);
      // Embree: the same rays and the same rule, bit for bit; OptiX differs from it at triangle-edge ties only.
      if (backend == BackendKind::kEmbree) {
        CHECK(bad == 0);
      } else {
        CHECK(bad * 1000 <= probes.size());
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// S3b: the plan pass's rule for baked receivers, the start diagnostic and the dense channel shadow_factor.

namespace {

// gt_into with the plan pass's shadow words on a backend (OptiX through device buffers).
std::vector<uint32_t> plan_states(const SceneBundle& s, BackendKind backend, int64_t t, std::vector<GtSample>& gt) {
  std::vector<GtQuery> q(static_cast<size_t>(kW) * kH);
  for (size_t i = 0; i < q.size(); ++i) q[i] = {t, static_cast<float>(i % kW), static_cast<float>(i / kW)};
  Renderer r(s, options(backend, 4));
  std::vector<uint32_t> out(q.size() * 2 * static_cast<size_t>(r.num_shadow_lights()), 99);
  gt.resize(q.size());
  if (backend == BackendKind::kEmbree) {
    r.gt_into(0, q.data(), q.size(), gt.data(), nullptr, out.data());
    return out;
  }
#if BALLA_RENDER_HAVE_OPTIX
  cudaStream_t stream = nullptr;
  GtQuery* dq = nullptr;
  GtSample* dg = nullptr;
  uint32_t* ds = nullptr;
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  REQUIRE(cudaMalloc(&dq, q.size() * sizeof(GtQuery)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dg, q.size() * sizeof(GtSample)) == cudaSuccess);
  REQUIRE(cudaMalloc(&ds, out.size() * sizeof(uint32_t)) == cudaSuccess);
  REQUIRE(cudaMemcpy(dq, q.data(), q.size() * sizeof(GtQuery), cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(ds, out.data(), out.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);  // pageable copies may land after cudaMemcpy returns
  r.gt_into(0, dq, q.size(), dg, stream, ds);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(cudaMemcpy(gt.data(), dg, q.size() * sizeof(GtSample), cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(out.data(), ds, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
  for (void* p : {static_cast<void*>(dq), static_cast<void*>(dg), static_cast<void*>(ds)}) cudaFree(p);
  cudaStreamDestroy(stream);
#endif
  return out;
}

// The S3b scene: the T5 scene's scan plane (id 1, z 3) lit by one proxy point lamp off to the side (left of and level
// with the camera), with a mover card (id 2, z 2.0, along x at 1 m/s: at `mover` at kRenderT), a scan card (id 3,
// z 2.5; geometry 0 with the plane: W), and a static lit card (id 4, z 1.5; split from the scan as an inserted object).
// The rectangles were chosen (by a search over the 64 x 48 image) so that the cards do not hide one another from the
// camera and their shadows of the lamp on the plane give every combination: open, W alone, the mover, the static card,
// the mover over the static card (the mover is first in geometry order), and W over the static card.
struct S3bScene {
  SceneBundle bundle;
  Rect mover{2.0, -0.4, 0.15, 0.2, 0.45}, statics{1.5, -0.6, -0.02, 0.0, 0.44}, scan{2.5, -0.6, 0.05, 0.19, 0.93};
  std::array<double, 3> lamp{-1.4, 0.3, 0.05};
};

S3bScene s3b_scene(bool shade = false) {
  S3bScene sc;
  U2Options op;
  op.proxies = {Proxy{false, sc.lamp, 3.0f, 0}};
  op.inserted = sc.mover;
  op.mover = true;
  op.scan_card = sc.scan;
  sc.bundle = u2_bundle(op);
  Body card = make_body(4, rect_mesh(sc.statics.x1 - sc.statics.x0, sc.statics.y1 - sc.statics.y0), {0.6f, 0.5f, 0.4f});
  card.T_world_body = translation(0.5 * (sc.statics.x0 + sc.statics.x1), 0.5 * (sc.statics.y0 + sc.statics.y1),
                                  sc.statics.z);
  sc.bundle.bodies.push_back(card);
  if (shade) {  // a lampshade: a small scan card just ahead of the lamp, out of the camera's view
    Body cap = make_body(5, rect_mesh(0.08, 0.08), {0.5f, 0.5f, 0.5f});
    bake(cap.materials[0], 1.0f, {});
    cap.scan = true;
    cap.T_world_body = translation(sc.lamp[0], sc.lamp[1], sc.lamp[2] + 0.01);
    sc.bundle.bodies.push_back(cap);
  }
  return sc;
}

struct Expected {
  uint32_t state = kShadowLit;
  double t = 0.0;
  bool near = false;       // a blocking test within 3e-5 m of a rectangle's edge: not compared
  bool im = false, is = false, w = false;  // the mover, the static card, the scan card block the lamp's ray
};

// The plan-pass state of a baked receiver hit at `hit` (instance `id`): the rule of scans.md §4.2 on rectangles.
Expected s3b_expect(const S3bScene& sc, const std::array<double, 3>& hit, uint32_t id) {
  const auto& l = sc.lamp;
  const std::array<double, 3> origin{hit[0], hit[1], hit[2] - 256.0 * std::ldexp(1.0, -22)};
  const std::array<double, 3> dr{l[0] - origin[0], l[1] - origin[1], l[2] - origin[2]};
  const double len = std::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
  const std::array<double, 3> to{dr[0] / len, dr[1] / len, dr[2] / len};
  Expected e;
  e.im = Oracle::crosses(sc.mover, origin, to, len, &e.near);
  e.is = Oracle::crosses(sc.statics, origin, to, len, &e.near);
  e.w = id == 1 && Oracle::crosses(sc.scan, origin, to, len, &e.near);  // (the scan card's own surface is no W)
  if (!e.im && !e.is) return e;
  if (e.w) {
    e.state = kShadowStatic;
  } else if (e.im) {
    e.state = kShadowMover + 2;  // the mover (id 2) comes before the static card (id 4) in geometry order
    e.t = (origin[2] - sc.mover.z) / (origin[2] - l[2]);
  } else {
    e.state = kShadowStatic;
  }
  return e;
}

}  // namespace

TEST_CASE("S3b: the plan pass's states of baked receivers (lights.h shadow_states<true>, scans.md §4.2), on every backend") {
  const S3bScene sc = s3b_scene();
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    std::vector<GtSample> gt;
    const auto states = plan_states(sc.bundle, backend, kRenderT, gt);
    REQUIRE(states.size() == static_cast<size_t>(kW * kH) * 2);
    // Pixel centre rays through the pinhole at f = 60 (the camera is the world frame).
    size_t checked = 0, skipped = 0, bad = 0, lit = 0, mover = 0, statics_only = 0, w_over_static = 0, both = 0,
           multi = 0, scan_card = 0, w_only = 0, w_over_mover = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      if (gt[i].instance != 1 && gt[i].instance != 3) continue;
      const double x = (static_cast<double>(i % kW) - 0.5 * (kW - 1)) / 60.0,
                   y = (static_cast<double>(i / kW) - 0.5 * (kH - 1)) / 60.0, z = static_cast<double>(gt[i].depth);
      const Expected e = s3b_expect(sc, {x * z, y * z, z}, gt[i].instance);
      if (e.near) {
        ++skipped;
        continue;
      }
      ++checked;
      scan_card += gt[i].instance == 3;
      const uint32_t s = states[2 * i];
      multi += (s & kShadowMulti) != 0;
      bool ok = s == e.state;
      if (e.state == kShadowMover + 2) {
        ok = ok && std::abs(static_cast<double>(bits_to_float(states[2 * i + 1])) - e.t) < 1e-4;  // the mover's t
      }
      if (e.state == kShadowLit) ok = ok && states[2 * i + 1] == 0u;
      bad += !ok;
      lit += e.state == kShadowLit;
      mover += e.state == kShadowMover + 2;
      both += e.im && e.is && !e.w;
      statics_only += e.is && !e.im && !e.w;
      w_over_static += e.is && e.w;
      w_only += e.w && !e.im && !e.is;
      w_over_mover += e.w && e.im && !e.is;
    }
    MESSAGE(backend_name(backend) << ": checked " << checked << " (skipped near an edge " << skipped << "): lit " << lit
                                  << " (the scan alone blocks " << w_only << "), mover " << mover << " (over the static card "
                                  << both << "), static card " << statics_only << ", scan over the static card "
                                  << w_over_static << ", scan over the mover " << w_over_mover
                                  << ", receivers on the scan card " << scan_card);
    CHECK(checked > 2000);
    CHECK(bad == 0);
    CHECK(multi == 0);  // one mover only; a split static body is no second moving shadow
    CHECK((lit > 500 && w_only > 20 && mover > 20 && statics_only > 20 && w_over_static > 5 && w_over_mover > 5 && both >= 2 &&
           scan_card > 10));
  }
}

TEST_CASE("S3b: a lit receiver's plan pass and a non-proxy light are as before in a baked-proxy scene (Embree)") {
  // A lit card behind the others, lit by the same lamp as a *non-proxy* shadow light too: states of lit receivers keep
  // the any-geometry rule (kAll) with a split static body mapped to kShadowStatic.
  S3bScene sc = s3b_scene();
  Light lamp2 = point_light({-0.3, 0.4, 0.1}, 2.0f, true, 0);  // not a proxy: no ray from the scan plane
  sc.bundle.lights.push_back(lamp2);
  std::vector<GtSample> gt;
  const auto states = plan_states(sc.bundle, BackendKind::kEmbree, kRenderT, gt);
  const size_t bits = states.size() / gt.size() / 2;
  REQUIRE(bits == 2);
  size_t plane = 0, card = 0;
  for (size_t i = 0; i < gt.size(); ++i) {
    if (gt[i].instance == 1 || gt[i].instance == 3) {  // baked receivers: the non-proxy lamp (bit 1) traces nothing
      CHECK(states[2 * (i * bits + 1)] == kShadowStatic);
      ++plane;
    } else if (gt[i].instance == 4) {  // the static card, lit: both lamps follow the any-geometry rule
      const uint32_t s = states[2 * (i * bits)];
      CHECK((s == kShadowLit || s == kShadowStatic || (s & ~kShadowMulti) == kShadowMover + 2));
      ++card;
    }
  }
  CHECK(plane > 1000);
  CHECK(card > 100);
}

TEST_CASE("S3b: Renderer::proxy_visibility, the share of a proxy's rays that the scan blocks, on every backend") {
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    {  // The scan card (W) blocks part of the lamp's rays from the plane: the oracle's share over the baked hits.
      const S3bScene sc = s3b_scene();
      Renderer r(sc.bundle, options(backend, 4));
      const std::vector<float> share = r.proxy_visibility(0, kRenderT);
      REQUIRE(share.size() == 1);
      std::vector<GtSample> gt;
      // The receivers are the backend's own centre hits: the mover's right edge (x = 0.15 at z = 2) lies on column 36's
      // centre rays, where OptiX and Embree break the tie differently (8 pixels; DESIGN 12.1 L2 excludes edges).
      plan_states(sc.bundle, backend, kRenderT, gt);
      double open = 0, blocked = 0, near_open = 0, near_blocked = 0;
      for (size_t i = 0; i < gt.size(); ++i) {
        if (gt[i].instance != 1 && gt[i].instance != 3) continue;
        const double x = (static_cast<double>(i % kW) - 0.5 * (kW - 1)) / 60.0, y = (static_cast<double>(i / kW) - 0.5 * (kH - 1)) / 60.0,
                     z = static_cast<double>(gt[i].depth);
        const std::array<double, 3> origin{x * z, y * z, z - 256.0 * std::ldexp(1.0, -22)};
        const auto& l = sc.lamp;
        const std::array<double, 3> dr{l[0] - origin[0], l[1] - origin[1], l[2] - origin[2]};
        const double len = std::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
        const std::array<double, 3> to{dr[0] / len, dr[1] / len, dr[2] / len};
        bool near = false;
        const bool w = gt[i].instance == 1 && Oracle::crosses(sc.scan, origin, to, len, &near);
        (w ? (near ? near_blocked : blocked) : (near ? near_open : open)) += 1.0;
      }
      const double lo = blocked / (blocked + open + near_open + near_blocked),
                   hi = (blocked + near_blocked) / (blocked + open + near_open + near_blocked);
      MESSAGE(backend_name(backend) << ": share " << share[0] << ", the oracle's [" << lo << ", " << hi << "]");
      CHECK(share[0] >= lo - 1e-3);
      CHECK(share[0] <= hi + 1e-3);
      CHECK(share[0] > 0.02);
      CHECK(share[0] < 0.5);
      CHECK(r.split_geometries() == 1);  // the static card; the mover is a mover, the scan cards the world
    }
    {  // A lampshade between the lamp and the room: every ray is blocked
      const S3bScene sc = s3b_scene(true);
      Renderer r(sc.bundle, options(backend, 4));
      const std::vector<float> share = r.proxy_visibility(0, kRenderT);
      REQUIRE(share.size() == 1);
      CHECK(share[0] > 0.999f);
    }
    {  // No scan card: nothing blocks; a scene without baked proxies has no diagnostic and no split
      S3bScene sc = s3b_scene();
      sc.bundle.bodies.erase(std::remove_if(sc.bundle.bodies.begin(), sc.bundle.bodies.end(),
                                            [](const Body& b) { return b.id == 3; }),
                             sc.bundle.bodies.end());
      Renderer r(sc.bundle, options(backend, 4));
      const std::vector<float> share = r.proxy_visibility(0, kRenderT);
      REQUIRE(share.size() == 1);
      CHECK(share[0] == 0.0f);
      SceneBundle lit = room(1);
      Renderer rl(lit, options(backend, 4));
      CHECK(rl.proxy_visibility(0, kRenderT).empty());
      CHECK(rl.split_geometries() == 0);
    }
  }
}

TEST_CASE("S3b: the dense channel shadow_factor is the ratio to the unshadowed irradiance (scans.md §3.7), on every backend") {
  // The T5 scene with a directional proxy and a static inserted card: its umbra has one F on every sub-sample of a
  // pixel (a directional light has one f), so F-bar is that F exactly and the channel reproduces it within 2^-23.
  const Proxy sun{true, {1.0, 0.5, 1.0}, 2.0f, 1};  // in light group 1 (the scan's share 0.7)
  U2Options op;
  op.proxies = {sun};
  op.inserted = {2.0, -1.4, -0.7, -0.5, 0.0};
  op.shares = {0.3f, 0.7f};
  op.fill = {0.1f, 0.3f};  // group 1's proxy fill A^f_1 = 0.3
  const SceneBundle b = u2_bundle(op);
  const SceneData data(b, SceneSettings{});
  const SceneView v = data.host_view();
  // F of the umbra, from the renderer's own arithmetic (shade.h baked_shadow_factor): the sun blocked, nothing static.
  std::vector<PoseRow> rows(data.geometries().size());
  std::vector<LightRow> lrows(data.lights().size());
  REQUIRE(compute_slot(data.pose_view(), 0, kRenderT, rows.data(), lrows.data()));
  const PoseTableView poses{rows.data(), lrows.data(), static_cast<int32_t>(rows.size()),
                            static_cast<int32_t>(lrows.size())};
  const float f_umbra = baked_shadow_factor(v, poses, 0, Vec3f{0.0f, 0.0f, 3.0f}, Vec3f{0.0f, 0.0f, -1.0f}, 1, 1u, 0u);
  REQUIRE((f_umbra > 0.0f && f_umbra < 0.5f));
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    RendererOptions o = options(backend);
    for (const uint32_t n_sub : {16u, 64u}) {
      INFO("N " << n_sub);
      o.scene.n_subsamples = n_sub;
      Renderer r(b, o);
      const DenseFrame with = r.dense(0, kRenderT, true, false, true);
      const DenseFrame without = r.dense(0, kRenderT, true, false, false);
      REQUIRE(with.shadow_factor.size() == static_cast<size_t>(kW * kH) * 2);
      CHECK(without.shadow_factor.empty());
      CHECK(same_bits(with.irradiance, without.irradiance));  // the channel changes nothing it is read beside
      const RenderResult sub = render_all(b, o, kRenderT, {true, false});
      size_t umbra = 0, lit = 0, partial = 0, other_group = 0, bad = 0;
      for (size_t i = 0; i < static_cast<size_t>(kW * kH); ++i) {
        bool plane_only = true;
        for (uint32_t k = 0; k < n_sub; ++k) plane_only = plane_only && sub.sub_instance[i * n_sub + k] == 1;
        const float s0 = with.shadow_factor[2 * i], s1 = with.shadow_factor[2 * i + 1];
        CHECK(s0 == 1.0f);  // no proxy in group 0: its factor is 1 everywhere
        other_group += s0 == 1.0f;
        CHECK((s1 > 0.0f && s1 <= 1.0f));
        if (!plane_only) continue;
        if (s1 == 1.0f) {
          ++lit;
        } else if (std::abs(s1 - f_umbra) <= std::ldexp(1.0f, -23) * f_umbra) {
          ++umbra;
        } else {
          ++partial;  // some sub-sample is lit: F-bar is at least (1 - F) / N above the umbra's, far beyond 2^-23
          bad += !(s1 > f_umbra * 1.001f && s1 <= 1.0f);
        }
      }
      MESSAGE(backend_name(backend) << ", N " << n_sub << ": lit " << lit << ", umbra (within 2^-23) " << umbra
                                    << ", partial " << partial << ", group 0 equal to 1 on " << other_group);
      CHECK(umbra > 30);
      CHECK(partial > 10);
      CHECK(lit > 500);
      CHECK(bad == 0);
    }
  }
}

TEST_CASE("S3b: shadow_factor is 1 everywhere without baked proxies, and the irradiance is unchanged (every backend)") {
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    for (const bool lit_lamp : {false, true}) {  // baked U0 (no light), and a lit scene with a shadow-casting lamp
      SceneBundle s = lit_scene(lit_lamp, false, lit_lamp);
      Renderer r(s, options(backend));
      const DenseFrame with = r.dense(0, kT0 + 100'000'000, true, false, true);
      const DenseFrame without = r.dense(0, kT0 + 100'000'000, true, false, false);
      CHECK(same_bits(with.irradiance, without.irradiance));
      REQUIRE(with.shadow_factor.size() == with.irradiance.size());
      CHECK(std::all_of(with.shadow_factor.begin(), with.shadow_factor.end(), [](float f) { return f == 1.0f; }));
    }
  }
}

// Batch 5's merge of S3a-2 with P2-F F6, F9 and F11a: the device's baked-proxy kernels are non-V1 kernels (no kV1 x kBaked
// twin), so a scene with baked proxies and a lobe, mover AO (proxy or traced) or two-channel NIR is refused at build on
// every backend (Renderer::build); without proxies the same scenes render, and so do baked proxies with none of them.
TEST_CASE("baked proxies with a specular lobe, mover AO or two-channel NIR are refused at build (every backend)") {
  constexpr int64_t t = kT0 + 100'000'000;
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    const RendererOptions o = options(backend);
    CHECK_NOTHROW(render_all(lit_scene(true, true), o, t));
    SceneBundle lobe = lit_scene(true, true);
    lobe.bodies.back().materials[0].specular = 0.5f;
    lobe.bodies.back().materials[0].roughness = 0.4f;
    CHECK_THROWS_AS(render_all(lobe, o, t), std::invalid_argument);
    SceneBundle plain = lobe;  // the control: the lobe without proxies renders
    plain.lights[0].proxy = false;
    plain.bodies[0].scan = false;
    CHECK_NOTHROW(render_all(plain, o, t));
    for (const MoverAo ao : {MoverAo::kProxy, MoverAo::kTraced}) {
      RendererOptions oa = o;
      oa.scene.mover_ao = ao;
      CHECK_THROWS_AS(render_all(lit_scene(true, true), oa, t), std::invalid_argument);
      CHECK_NOTHROW(render_all(lit_scene(true, false), oa, t));
    }
    RendererOptions on = o;  // two-channel NIR: the cameras' f differ between the two light groups
    on.nir_fraction = {{0.0, 1.0}};
    CHECK_THROWS_AS(render_all(lit_scene(true, true), on, t), std::invalid_argument);
  }
}
