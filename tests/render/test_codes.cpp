// P2-D D2 (plan ballasim_out/p2_plans/p2d.md §3.1 Step A, §3.6, §6 row D2): render_into's P&V outputs and the probes
// on the host backend (render/shade/codes.h, render/shade/probe.h). The outputs change no irradiance, signature or
// class word; an entry exactly for the edge-class pixel-samples; the codes and table keys rebuild the signature, the
// rows the irradiance, the counts the codes; F, R and the probes' f agree with each other and, in a scene of shadow
// lights alone, a R with the radiance; probes reproduce the codes' states at lattice times in any order and chunking.
// Host = device and Embree against OptiX: test_device.cpp (GPU).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/time/time.h"
#include "render/renderer.h"
#include "render/scene_data.h"
#include "render/shade/codes.h"
#include "render/shade/shade.h"
#include "scenes.h"

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;

namespace {

constexpr int64_t kT0 = 1'000'000'000, kT1 = 1'250'000'000;
constexpr int kW = 80, kH = 60;

// A textured floor, a wall that leaves the view's corners empty (misses: the environment, with a background radiance),
// three movers in front, `shadow_lights` point lights casting shadows (the fourth a directional one), and, unless
// `pure`, ambient light, a spot without shadows and an emissive mover: with `pure` every radiance term is a shadow
// light's, so a cluster's w L_g is the sum of its a R over group g's lights. With `stripes`, a plate of 1 cm quads
// whose triangles cycle through six materials: its pixels hold five or more clusters (the table's overflow).
SceneBundle codes_scene(int shadow_lights, bool pure = false, bool stripes = false) {
  SceneBundle s;
  s.light_groups = {"default", "mains"};
  s.trajectories.push_back(moving_trajectory(kT0, kT1, quat_identity<double>(), {0, -0.1, 0}, {0.1, 0.0, 0.05},
                                             {0.05, 0.1, 0.0}));
  for (int i = 0; i < 3; ++i) {
    s.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.3 * i, 0.2, -0.1}),
                                               {-0.35 + 0.35 * i, 0.05 * i, 1.3 + 0.25 * i}, {0.4, -0.1 * i, 0.0},
                                               {1.0, 1.5 - i, 0.4}));
  }
  s.cameras.push_back(pinhole(kW, kH, 70.0, 0));
  std::vector<uint8_t> tex(64 * 64 * 3);
  for (size_t i = 0; i < tex.size(); ++i) tex[i] = static_cast<uint8_t>((i * 37) % 251);
  Body floor = make_body(1, plane_mesh(2.0, 8), {0.8f, 0.7f, 0.6f});
  floor.materials[0].texture = tex;
  floor.materials[0].texture_width = floor.materials[0].texture_height = 64;
  floor.materials[0].texture_channels = 3;
  floor.T_world_body = {1, 0, 0, 0, 0, 0, -1, 0.45, 0, 1, 0, 1.9, 0, 0, 0, 1};
  s.bodies.push_back(floor);
  Body wall = make_body(2, plane_mesh(1.1, 2), {0.5f, 0.6f, 0.7f});
  wall.T_world_body = translation(0.1, -0.2, 2.6);
  s.bodies.push_back(wall);
  if (stripes) {
    Body plate = make_body(3, plane_mesh(0.3, 60), {0.9f, 0.2f, 0.2f});
    for (int m = 1; m < 6; ++m) {
      Material mat;
      mat.base_color = {0.15f * static_cast<float>(m), 0.9f - 0.1f * static_cast<float>(m), 0.4f};
      plate.materials.push_back(mat);
    }
    for (size_t t = 0; t < plate.mesh.indices.size() / 3; ++t) plate.mesh.material_ids.push_back(static_cast<uint16_t>(t % 6));
    plate.T_world_body = translation(-0.6, -0.5, 2.4);
    s.bodies.push_back(plate);
  }
  for (uint32_t i = 0; i < 3; ++i) {
    Body b = make_body(20 + i, box_mesh(0.09f, 0.07f, 0.11f), {0.3f, 0.5f + 0.1f * static_cast<float>(i), 0.4f});
    if (!pure && i == 1) b.materials[0].emissive = {0.1f, 0.0f, 0.0f};
    b.trajectory = static_cast<int32_t>(i) + 1;
    s.bodies.push_back(b);
  }
  for (int l = 0; l < shadow_lights; ++l) {
    if (l == 3) {
      Light sun;
      sun.kind = BundleLightKind::kDirectional;
      sun.direction = {0.25, 0.9, 0.35};
      sun.intensity = 0.4f;
      sun.group = 1;
      s.lights.push_back(sun);
      continue;
    }
    const int p = l % 5;
    s.lights.push_back(point_light({-0.7 + 0.35 * p, -0.9, 0.4 + 0.15 * p + 0.02 * (l / 5)}, 1.6f, true, l % 2));
  }
  s.lights.push_back(background_light(0.07f, 1));
  if (!pure) {
    s.lights.push_back(ambient_light(0.08f));
    Light spot;
    spot.kind = BundleLightKind::kSpot;
    spot.position = {0.5, -0.8, 0.2};
    spot.direction = {-0.2, 0.7, 0.7};
    spot.spot_inner_rad = 0.3f;
    spot.spot_outer_rad = 0.7f;
    spot.intensity = 2.0f;
    spot.casts_shadows = false;
    s.lights.push_back(spot);
  }
  return s;
}

// n requests at random pixels and window-clock slots (the clock's slots, before assign_pose_rows).
std::vector<SampleRequest> random_requests(size_t n, uint32_t n_sub, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<SampleRequest> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = {0, 0, static_cast<uint32_t>(rng() % (kW * kH)), 0, n_sub, 0, static_cast<uint32_t>(rng() % 8193)};
  }
  return out;
}

// One render_into's outputs on the host (out_index: a permutation), with or without P&V outputs.
struct Rendered {
  std::vector<float> irr;
  std::vector<uint64_t> sig;
  std::vector<uint32_t> klass, slot, codes, table;
  uint32_t count = 0, code_words = 0, table_words = 0;
};

Rendered render_rows(Renderer& r, std::vector<SampleRequest> requests, const std::vector<uint32_t>& out_index,
                     uint32_t n_sub, const WindowClock& clock, int pv_mode, uint32_t capacity = 0) {
  const size_t n = requests.size();
  const int32_t lights = r.num_shadow_lights(), groups = r.num_groups();
  Rendered o;
  o.irr.resize(n * static_cast<size_t>(groups));
  o.sig.resize(n);
  o.klass.resize(n);
  o.code_words = pv_code_words(n_sub, lights);
  o.table_words = pv_table_words(groups, lights);
  PvOut pv;
  if (pv_mode != 0) {
    if (capacity == 0) capacity = static_cast<uint32_t>(n);
    o.slot.assign(n, 0xABCDu);
    o.codes.assign(static_cast<size_t>(capacity) * o.code_words, 0);
    o.table.assign(static_cast<size_t>(capacity) * o.table_words, 0);
    pv = {o.slot.data(), o.codes.data(), o.table.data(), &o.count, capacity, pv_mode == 2 ? 1u : 0u};
  }
  r.begin_window(0, clock, nullptr);
  r.assign_pose_rows(requests.data(), n, nullptr);
  r.render_into(requests.data(), n, out_index.data(), o.irr.data(), o.sig.data(), nullptr, o.klass.data(), n_sub,
                pv_mode != 0 ? &pv : nullptr);
  r.take_render_stats();
  return o;
}

bool edge(uint32_t word) { return (word & (kClassMulti | kClassShadowEdge)) != 0; }

// The decoded state of one entry: per sub-sample its row field and mask; per table row its key and count.
struct Entry {
  std::vector<uint32_t> row, mask;
  uint32_t rows = 0;
  bool overflow = false;
};

Entry decode(const Rendered& o, uint32_t e, uint32_t n_sub, int32_t lights) {
  Entry d;
  const uint32_t bits = pv_code_bits(lights);
  const uint32_t* w = o.codes.data() + static_cast<size_t>(e) * o.code_words;
  for (uint32_t k = 0; k < n_sub; ++k) {
    const uint32_t c = pv_get_code(w, k, bits);
    d.row.push_back(c & 7u);
    d.mask.push_back(c >> 3);
  }
  const uint32_t header = o.table[static_cast<size_t>(e) * o.table_words];
  d.rows = header & 7u;
  d.overflow = (header & kPvTableOverflow) != 0;
  return d;
}

const uint32_t* table_row(const Rendered& o, uint32_t e, uint32_t row, int32_t groups, int32_t lights) {
  return o.table.data() + static_cast<size_t>(e) * o.table_words + 1 + row * pv_row_words(groups, lights);
}

float f32(uint32_t bits) { return bits_to_float(bits); }

}  // namespace

TEST_CASE("P2-D D2: render_into's P&V outputs on the host (codes, tables; outputs unchanged)") {
  const int64_t t_w = time::window_start(time::window_index(kT0) + 3);
  const WindowClock clock{t_w, 10, 0, 0};
  for (const auto& [n_sub, integral] : {std::pair{64u, PixelIntegral::kPhase1}, std::pair{256u, PixelIntegral::kPhase1},
                                        std::pair{256u, PixelIntegral::kFixed64}}) {
    for (const int shadows : {0, 1, 3, 8}) {
      for (const bool unlit : {false, true}) {
        for (const bool pure : {false, true}) {
          if ((unlit || pure) && shadows == 0) continue;
          const std::string what = "N " + std::to_string(n_sub) +
                                   (integral == PixelIntegral::kFixed64 ? " fixed64" : "") + ", shadow lights " +
                                   std::to_string(shadows) + (unlit ? ", unlit mask" : "") +
                                   (pure ? ", shadow lights only" : ", stripes");
          INFO(what);
          RendererOptions o = options(BackendKind::kEmbree);
          o.scene.n_subsamples = n_sub;
          o.scene.pixel_integral = integral;
          o.scene.unlit_mask = unlit;
          o.max_rays_per_launch = uint64_t{1} << 16;  // several chunks
          const SceneBundle scene = codes_scene(shadows, pure, !pure);
          Renderer r(scene, o);
          const size_t n = 1500;
          const std::vector<SampleRequest> req = random_requests(n, n_sub, n_sub + 7u * shadows + unlit);
          std::vector<uint32_t> out_index(n);
          for (size_t i = 0; i < n; ++i) out_index[i] = static_cast<uint32_t>(i);
          std::shuffle(out_index.begin(), out_index.end(), std::mt19937(5));
          const Rendered plain = render_rows(r, req, out_index, n_sub, clock, 0);
          const Rendered edge_out = render_rows(r, req, out_index, n_sub, clock, 1);
          const Rendered all = render_rows(r, req, out_index, n_sub, clock, 2);
          // Outputs unchanged, bit for bit.
          for (const Rendered* x : {&edge_out, &all}) {
            CHECK(std::memcmp(x->irr.data(), plain.irr.data(), plain.irr.size() * sizeof(float)) == 0);
            CHECK(x->sig == plain.sig);
            CHECK(x->klass == plain.klass);
          }
          // An entry exactly for the edge-class rows; every row under all.
          uint32_t edges = 0;
          for (size_t row = 0; row < n; ++row) {
            edges += edge(plain.klass[row]) ? 1u : 0u;
            REQUIRE((edge_out.slot[row] < n) == edge(plain.klass[row]));
            if (!edge(plain.klass[row])) REQUIRE(edge_out.slot[row] == kPvNone);
            REQUIRE(all.slot[row] < n);
          }
          CHECK(edge_out.count == edges);
          CHECK(all.count == n);
          CHECK(edges > 0);
          CHECK(edges < n);
          const int32_t groups = r.num_groups(), lights = r.num_shadow_lights();
          const SceneData data(scene, o.scene);
          const SceneView& view = data.host_view();
          size_t multi = 0, shadow_edges = 0, five = 0;
          for (size_t row = 0; row < n; ++row) {
            const uint32_t e = all.slot[row];
            const Entry d = decode(all, e, n_sub, lights);
            // The edge mode's entry of an edge row holds the same words.
            if (edge(plain.klass[row])) {
              const uint32_t f = edge_out.slot[row];
              REQUIRE(std::equal(all.codes.begin() + e * all.code_words, all.codes.begin() + (e + 1) * all.code_words,
                                 edge_out.codes.begin() + f * all.code_words));
              const uint32_t used = 1 + d.rows * pv_row_words(groups, lights);
              REQUIRE(std::equal(all.table.begin() + e * all.table_words, all.table.begin() + e * all.table_words + used,
                                 edge_out.table.begin() + f * all.table_words));
            }
            // Rows in order of first sub-sample: row c first appears after rows < c; counts and the header agree.
            std::vector<uint32_t> count(8, 0);
            uint32_t next = 0;
            for (uint32_t k = 0; k < n_sub; ++k) {
              const uint32_t c = d.row[k];
              REQUIRE(c != 4u);
              REQUIRE(c != 5u);
              if (c < kPvRows && count[c] == 0) REQUIRE(c == next++);
              ++count[c];
              if (lights < 32) REQUIRE((d.mask[k] >> lights) == 0u);
            }
            REQUIRE(d.rows == next);
            REQUIRE(d.overflow == (count[kPvCodeOverflow] > 0));
            five += d.overflow ? 1u : 0u;
            multi += d.rows + (count[kPvCodeEnv] > 0 ? 1u : 0u) + (d.overflow ? 1u : 0u) > 1 ? 1u : 0u;
            for (uint32_t k = 1; k < n_sub; ++k) shadow_edges += d.mask[k] != d.mask[0] ? 1u : 0u;
            if (d.overflow) continue;  // a fifth cluster has no key: the rest needs every row
            for (uint32_t c = 0; c < d.rows; ++c) REQUIRE(table_row(all, e, c, groups, lights)[2] == count[c]);
            // The keys and masks rebuild the signature (the environment: misses here, key (0, 0)).
            uint64_t sig = 0;
            for (uint32_t k = 0; k < n_sub; ++k) {
              const uint32_t c = d.row[k];
              const uint32_t* t = c < kPvRows ? table_row(all, e, c, groups, lights) : nullptr;
              sig += signature_term(k, t != nullptr ? t[0] : kInstanceNone, t != nullptr ? t[1] : 0u, d.mask[k]);
            }
            REQUIRE(sig == plain.sig[row]);
            // The rows rebuild the irradiance: sum over the clusters in order of first sub-sample of w L_g, the
            // environment's w B_g, as the shading adds them.
            std::vector<float> irr(static_cast<size_t>(groups), 0.0f);
            const float inv_n = 1.0f / static_cast<float>(n_sub);
            std::vector<bool> seen(8, false);
            for (uint32_t k = 0; k < n_sub; ++k) {
              const uint32_t c = d.row[k];
              if (seen[c]) continue;
              seen[c] = true;
              const float w = static_cast<float>(count[c]) * inv_n;
              for (int32_t g = 0; g < groups; ++g) {
                if (c == kPvCodeEnv) {
                  if (view.has_background != 0) irr[static_cast<size_t>(g)] += w * view.background[g];
                } else {
                  irr[static_cast<size_t>(g)] += w * f32(table_row(all, e, c, groups, lights)[3 + g]);
                }
              }
            }
            REQUIRE(std::memcmp(irr.data(), &plain.irr[row * static_cast<size_t>(groups)], irr.size() * sizeof(float)) ==
                    0);
            for (uint32_t c = 0; c < d.rows; ++c) {
              const uint32_t* t = table_row(all, e, c, groups, lights);
              const float w = static_cast<float>(t[2]) * inv_n;
              std::vector<double> share(static_cast<size_t>(groups), 0.0);
              for (int32_t b = 0; b < lights; ++b) {
                const float a = f32(t[3 + groups + 3 * b]), f = f32(t[4 + groups + 3 * b]), ratio = f32(t[5 + groups + 3 * b]);
                REQUIRE(a >= 0.0f);
                REQUIRE(f >= 0.0f);
                REQUIRE(ratio >= 0.0f);
                REQUIRE(ratio <= 1.0f + 1e-6f);
                if (f == 0.0f) REQUIRE(ratio == 0.0f);
                // A light whose bit is set on every member: R 0; clear on every member: R 1 (where f > 0).
                bool any = false, every = true;
                for (uint32_t k = 0; k < n_sub; ++k) {
                  if (d.row[k] != c) continue;
                  any = any || ((d.mask[k] >> b) & 1u) != 0;
                  every = every && ((d.mask[k] >> b) & 1u) != 0;
                }
                if (f > 0.0f && !any) REQUIRE(ratio == 1.0f);
                if (every) REQUIRE(ratio == 0.0f);
                share[static_cast<size_t>(view.lights[view.shadow_light[b]].group)] += static_cast<double>(a) * static_cast<double>(ratio);
              }
              if (pure) {  // w L_g = sum of a R over the group's shadow lights
                for (int32_t g = 0; g < groups; ++g) {
                  const double want = static_cast<double>(w) * static_cast<double>(f32(t[3 + g]));
                  REQUIRE(std::fabs(share[static_cast<size_t>(g)] - want) <= 1e-5 * want + 1e-12);
                }
              }
            }
          }
          MESSAGE(what << ": " << edges << " edge rows of " << n << "; " << multi << " entries with clusters, "
                       << shadow_edges << " sub-samples off sub-sample 0's mask, " << five << " with a fifth cluster");
          CHECK(multi > 0);
          if (!pure) CHECK(five > 0);
          if (shadows > 0) CHECK(shadow_edges > 0);
        }
      }
    }
  }
}

TEST_CASE("P2-D D2: probes reproduce render_into's states at lattice times, in any order and chunking") {
  const int64_t t_w = time::window_start(time::window_index(kT0) + 3);
  const WindowClock clock{t_w, 10, 0, 0};
  for (const auto& [unlit, patches] :
       {std::pair{false, ShadingPatches::kMaterial}, std::pair{true, ShadingPatches::kMaterial},
        std::pair{false, ShadingPatches::kCrease}, std::pair{true, ShadingPatches::kCrease}}) {
    INFO(std::string(unlit ? "unlit mask" : "as traced") << (patches == ShadingPatches::kCrease ? ", crease" : ""));
    const uint32_t n_sub = 256;
    RendererOptions o = options(BackendKind::kEmbree);
    o.scene.n_subsamples = n_sub;
    o.scene.unlit_mask = unlit;
    o.scene.shading_patches = patches;  // P2-F F5: probes report the cluster key (the patch under crease)
    Renderer r(codes_scene(3), o);
    const int32_t groups = r.num_groups(), lights = r.num_shadow_lights();
    const size_t n = 300;
    const std::vector<SampleRequest> req = random_requests(n, n_sub, 31 + unlit);
    std::vector<uint32_t> rows(n);
    for (size_t i = 0; i < n; ++i) rows[i] = static_cast<uint32_t>(i);
    const Rendered all = render_rows(r, req, rows, n_sub, clock, 2);
    // Every sub-sample of 40 requests in a shuffled order, request i's probes with shadow light i mod L.
    std::vector<SampleRequest> probes;
    std::vector<std::pair<size_t, uint32_t>> who;
    for (size_t i = 0; i < 40; ++i) {
      for (uint32_t k = 0; k < n_sub; ++k) {
        SampleRequest q = req[i];
        q.k_begin = k;
        q.k_count = 1;
        q.flags = static_cast<uint16_t>(static_cast<uint32_t>(RequestKind::kProbe) |
                                        static_cast<uint32_t>(i % static_cast<size_t>(lights)) << kProbeLightShift);
        probes.push_back(q);
        who.emplace_back(i, k);
      }
    }
    std::vector<size_t> order(probes.size());
    for (size_t j = 0; j < order.size(); ++j) order[j] = j;
    std::shuffle(order.begin(), order.end(), std::mt19937(3));
    std::vector<SampleRequest> mapped(probes.size());
    for (size_t j = 0; j < order.size(); ++j) mapped[j] = probes[order[j]];
    r.begin_window(0, clock, nullptr);
    r.assign_pose_rows(mapped.data(), mapped.size(), nullptr);
    std::vector<ProbeOut> got(probes.size());
    r.probe_into(mapped.data(), mapped.size(), got.data(), nullptr);
    const RenderStats st = r.take_render_stats();
    CHECK(st.probe_ms > 0.0);
    CHECK(st.trace_ms == 0.0);
    CHECK(st.shade_ms == 0.0);
    std::vector<ProbeOut> state(probes.size());
    for (size_t j = 0; j < order.size(); ++j) state[order[j]] = got[j];
    // The codes' state of every probed sub-sample; per cluster the probes' f sum to F_s,l of request i's light.
    std::map<std::pair<size_t, uint32_t>, double> f_sum;
    for (size_t j = 0; j < probes.size(); ++j) {
      const auto [i, k] = who[j];
      const uint32_t e = all.slot[i];
      const Entry d = decode(all, e, n_sub, lights);
      const ProbeOut& p = state[j];
      REQUIRE(p.mask == d.mask[k]);
      if (d.row[k] == kPvCodeEnv) {
        REQUIRE(p.instance == kInstanceNone);
        REQUIRE(p.surface == 0u);
        REQUIRE(p.f == 0.0f);
        continue;
      }
      if (patches == ShadingPatches::kCrease && d.row[k] == kPvCodeOverflow) continue;  // a fifth cluster (box faces)
      REQUIRE(d.row[k] < kPvRows);
      const uint32_t* t = table_row(all, e, d.row[k], groups, lights);
      REQUIRE(p.instance == t[0]);
      REQUIRE(p.surface == t[1]);  // under crease the patch (P2-F F5), not the material
      REQUIRE(p.f >= 0.0f);
      f_sum[{i, d.row[k]}] += static_cast<double>(p.f);
    }
    size_t compared = 0;
    for (size_t i = 0; i < 40; ++i) {
      const uint32_t e = all.slot[i];
      const Entry d = decode(all, e, n_sub, lights);
      const auto b = static_cast<int32_t>(i % static_cast<size_t>(lights));
      for (uint32_t c = 0; c < d.rows; ++c) {
        const double want = static_cast<double>(f32(table_row(all, e, c, groups, lights)[4 + groups + 3 * b]));
        REQUIRE(std::fabs(f_sum[{i, c}] - want) <= 1e-5 * want + 1e-12);
        compared += want > 0.0 ? 1u : 0u;
      }
    }
    MESSAGE("probes: " << probes.size() << ", clusters with F > 0 compared: " << compared);
    CHECK(compared > 20);
    // Another order and chunking (2^12 rays per launch), and the host API through the pose cache: equal states.
    RendererOptions o2 = o;
    o2.max_rays_per_launch = uint64_t{1} << 12;
    Renderer r2(codes_scene(3), o2);
    std::vector<SampleRequest> rev(probes.rbegin(), probes.rend());
    r2.begin_window(0, clock, nullptr);
    r2.assign_pose_rows(rev.data(), rev.size(), nullptr);
    std::vector<ProbeOut> got2(rev.size());
    r2.probe_into(rev.data(), rev.size(), got2.data(), nullptr);
    r2.take_render_stats();
    for (size_t j = 0; j < probes.size(); ++j) {
      REQUIRE(std::memcmp(&got2[probes.size() - 1 - j], &state[j], sizeof(ProbeOut)) == 0);
    }
    std::vector<uint16_t> cam(probes.size(), 0);
    std::vector<uint32_t> pixel, kk, light;
    std::vector<int64_t> t;
    for (const SampleRequest& q : probes) {
      pixel.push_back(q.pixel);
      kk.push_back(q.k_begin);
      light.push_back(static_cast<uint32_t>(q.flags) >> kProbeLightShift);
      t.push_back(slot_time(clock, q.pose_slot));
    }
    const ProbeResult host = r2.probe(cam, pixel, kk, t, light);
    CHECK(host.stats.probe_ms > 0.0);
    REQUIRE(std::memcmp(host.out.data(), state.data(), state.size() * sizeof(ProbeOut)) == 0);
    // Refused on the host: a probe of two sub-samples, a light out of range.
    std::vector<SampleRequest> bad(mapped.begin(), mapped.begin() + 4);
    bad[2].k_count = 2;
    CHECK_THROWS_AS(r.probe_into(bad.data(), bad.size(), got.data(), nullptr), std::invalid_argument);
    bad[2].k_count = 1;
    bad[3].flags = static_cast<uint16_t>(static_cast<uint32_t>(RequestKind::kProbe) | uint32_t{3} << kProbeLightShift);
    CHECK_THROWS_AS(r.probe_into(bad.data(), bad.size(), got.data(), nullptr), std::invalid_argument);
    light[0] = 7;
    CHECK_THROWS_AS(r2.probe(cam, pixel, kk, t, light), std::invalid_argument);
  }
}

TEST_CASE("P2-D D2: capacity, refusals and execution.pv_outputs (the cost measurement's scratch)") {
  const int64_t t_w = time::window_start(time::window_index(kT0) + 3);
  const WindowClock clock{t_w, 10, 0, 0};
  const size_t n = 1200;
  for (const uint32_t n_sub : {64u, 256u}) {
    INFO("N " << n_sub);
    RendererOptions o = options(BackendKind::kEmbree);
    o.scene.n_subsamples = n_sub;
    Renderer r(codes_scene(3), o);
    const std::vector<SampleRequest> req = random_requests(n, n_sub, 77);
    std::vector<uint32_t> rows(n);
    for (size_t i = 0; i < n; ++i) rows[i] = static_cast<uint32_t>(n - 1 - i);
    const Rendered plain = render_rows(r, req, rows, n_sub, clock, 0);
    uint32_t edges = 0;
    for (const uint32_t w : plain.klass) edges += edge(w) ? 1u : 0u;
    REQUIRE(edges > 2);
    // A capacity of half the edge rows: that many entries, the other edge rows kPvOverflow, the count all of them.
    const Rendered half = render_rows(r, req, rows, n_sub, clock, 1, edges / 2);
    CHECK(half.count == edges);
    uint32_t kept = 0, over = 0;
    std::vector<bool> used(edges / 2, false);
    for (size_t row = 0; row < n; ++row) {
      if (!edge(plain.klass[row])) {
        REQUIRE(half.slot[row] == kPvNone);
      } else if (half.slot[row] == kPvOverflow) {
        ++over;
      } else {
        REQUIRE(half.slot[row] < edges / 2);
        REQUIRE(!used[half.slot[row]]);
        used[half.slot[row]] = true;
        ++kept;
      }
    }
    CHECK(kept == edges / 2);
    CHECK(over == edges - edges / 2);
    CHECK(std::memcmp(half.irr.data(), plain.irr.data(), plain.irr.size() * sizeof(float)) == 0);
    // execution.pv_outputs: the calls at N >= 256 write the edge rows to scratch and count them; outputs unchanged.
    RendererOptions om = o;
    om.pv_outputs = true;
    Renderer m(codes_scene(3), om);
    std::vector<SampleRequest> q = req;
    std::vector<float> irr(plain.irr.size());
    std::vector<uint64_t> sig(n);
    std::vector<uint32_t> klass(n);
    for (int call = 0; call < 2; ++call) {  // two calls: their totals add up
      m.begin_window(0, clock, nullptr);
      q = req;
      m.assign_pose_rows(q.data(), n, nullptr);
      m.render_into(q.data(), n, rows.data(), irr.data(), sig.data(), nullptr, klass.data(), n_sub);
    }
    const RenderStats st = m.take_render_stats();
    CHECK(st.pv_on);
    const uint64_t want = n_sub >= kPvMeasureN ? 2u * edges : 0u;
    CHECK(st.pv_entries == want);
    CHECK(st.pv_bytes ==
          want * sizeof(uint32_t) * (pv_code_words(n_sub, 3) + pv_table_words(m.num_groups(), 3)));
    CHECK(std::memcmp(irr.data(), plain.irr.data(), irr.size() * sizeof(float)) == 0);
    CHECK(sig == plain.sig);
    CHECK(klass == plain.klass);
    CHECK(!r.take_render_stats().pv_on);
  }
  // More than kPvMaxLights shadow lights: refused.
  RendererOptions o = options(BackendKind::kEmbree);
  Renderer r(codes_scene(9), o);
  std::vector<SampleRequest> req = random_requests(16, o.scene.n_subsamples, 1);
  std::vector<float> irr(16 * 2);
  std::vector<uint64_t> sig(16);
  std::vector<uint32_t> slot(16), codes(16 * 64), table(16 * 200);
  uint32_t count = 0;
  PvOut pv{slot.data(), codes.data(), table.data(), &count, 16, 0};
  r.begin_window(0, clock, nullptr);
  r.assign_pose_rows(req.data(), req.size(), nullptr);
  CHECK_THROWS_AS(r.render_into(req.data(), req.size(), nullptr, irr.data(), sig.data(), nullptr, nullptr, 0, &pv),
                  std::invalid_argument);
  CHECK_THROWS_AS(r.render(std::vector<uint16_t>{0}, std::vector<uint32_t>{0}, std::vector<int64_t>{kT0},
                           RenderFlags{false, false, false, 1}),
                  std::invalid_argument);
}

// P2-F F9b x P2-D D2 (the lane's merge recipe, ballasim_out/p2_batch/p2f-shade/F9b/README.md "Merge with P2-D D2"): a
// code's mask field holds the shadow lights' bits only, so accuracy.mover_ao traced's AO bit (render/shade/ao_traced.h
// kAoTracedBit), which the signature takes, has none, and a probe's mask would carry it only with shadow lights. P&V is
// therefore not eligible under traced: render and render_into refuse P&V outputs, probe_into and probe refuse probes
// (the same calls pass under off), and execution.pv_outputs writes no entry and changes no output.
TEST_CASE("P2-F F9b x P2-D D2: P&V outputs and probes refused under mover_ao traced, pv_outputs writes none") {
  const int64_t t_w = time::window_start(time::window_index(kT0) + 3);
  const WindowClock clock{t_w, 10, 0, 0};
  const uint32_t n_sub = kPvMeasureN;
  const size_t n = 256;
  for (const int shadows : {0, 3}) {
    for (const MoverAo arm : {MoverAo::kOff, MoverAo::kTraced}) {
      const bool traced = arm == MoverAo::kTraced;
      INFO("shadow lights = " << shadows << (traced ? ", mover_ao traced" : ", mover_ao off"));
      RendererOptions o = options(BackendKind::kEmbree);
      o.scene.n_subsamples = n_sub;
      o.scene.mover_ao = arm;
      Renderer r(codes_scene(shadows), o);
      const int32_t groups = r.num_groups();
      const std::vector<SampleRequest> req = random_requests(n, n_sub, 5);
      std::vector<SampleRequest> q = req;
      std::vector<float> irr(n * static_cast<size_t>(groups));
      std::vector<uint64_t> sig(n);
      std::vector<uint32_t> slot(n), codes(n * pv_code_words(n_sub, shadows)), table(n * pv_table_words(groups, shadows));
      uint32_t count = 0;
      PvOut pv{slot.data(), codes.data(), table.data(), &count, static_cast<uint32_t>(n), 1};  // all rows
      r.begin_window(0, clock, nullptr);
      r.assign_pose_rows(q.data(), n, nullptr);
      const auto into = [&] { r.render_into(q.data(), n, nullptr, irr.data(), sig.data(), nullptr, nullptr, n_sub, &pv); };
      const auto whole = [&] {
        r.render(std::vector<uint16_t>{0}, std::vector<uint32_t>{q[0].pixel}, std::vector<int64_t>{t_w + 1000},
                 RenderFlags{false, false, false, 1});
      };
      SampleRequest probe = req[0];
      probe.k_count = 1;
      probe.flags = static_cast<uint16_t>(RequestKind::kProbe);
      ProbeOut state{};
      const auto probe_into = [&] {
        r.begin_window(0, clock, nullptr);
        r.assign_pose_rows(&probe, 1, nullptr);
        r.probe_into(&probe, 1, &state, nullptr);
      };
      const auto probe_host = [&] {
        r.probe(std::vector<uint16_t>{0}, std::vector<uint32_t>{probe.pixel}, std::vector<uint32_t>{0},
                std::vector<int64_t>{t_w + 1000}, std::vector<uint32_t>{0});
      };
      if (traced) {
        CHECK_THROWS_AS(into(), std::invalid_argument);
        CHECK_THROWS_AS(whole(), std::invalid_argument);
        CHECK_THROWS_AS(probe_into(), std::invalid_argument);
        CHECK_THROWS_AS(probe_host(), std::invalid_argument);
      } else {
        CHECK_NOTHROW(into());
        CHECK(count == n);
        CHECK_NOTHROW(whole());
        CHECK_NOTHROW(probe_into());
        CHECK_NOTHROW(probe_host());
      }
      r.take_render_stats();
      // execution.pv_outputs at N = kPvMeasureN: under traced no entry; the outputs are the plain renderer's.
      RendererOptions om = o;
      om.pv_outputs = true;
      Renderer m(codes_scene(shadows), om);
      std::vector<float> irr_p(irr.size()), irr_m(irr.size());
      std::vector<uint64_t> sig_p(n), sig_m(n);
      std::vector<uint32_t> class_p(n), class_m(n);
      for (int arm_m = 0; arm_m < 2; ++arm_m) {
        Renderer& x = arm_m == 0 ? r : m;
        q = req;
        x.begin_window(0, clock, nullptr);
        x.assign_pose_rows(q.data(), n, nullptr);
        x.render_into(q.data(), n, nullptr, (arm_m == 0 ? irr_p : irr_m).data(), (arm_m == 0 ? sig_p : sig_m).data(),
                      nullptr, (arm_m == 0 ? class_p : class_m).data(), n_sub);
      }
      r.take_render_stats();
      const RenderStats st = m.take_render_stats();
      uint64_t edges = 0;
      for (const uint32_t w : class_p) edges += (w & (kClassMulti | kClassShadowEdge)) != 0 ? 1u : 0u;
      CHECK(st.pv_on);
      if (shadows > 0) CHECK(edges > 0);  // entries the off arm writes and traced does not
      CHECK(st.pv_entries == (traced ? 0u : edges));
      if (traced) CHECK(st.pv_bytes == 0u);
      CHECK(std::memcmp(irr_m.data(), irr_p.data(), irr_p.size() * sizeof(float)) == 0);
      CHECK(sig_m == sig_p);
      CHECK(class_m == class_p);
    }
  }
}
