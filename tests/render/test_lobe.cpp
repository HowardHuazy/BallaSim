// P2-F F6, specular lobes in the cluster shading (render/shade/shade.h lobe_radiance; plan p2f §4.3, §4.8, §7 F6):
// - a single-member cluster equals bsdf.h<double> through the plugin's expressions within 5e-5 (the plan's
//   end-to-end gate), direct light and uniform ambient;
// - s = m = 0 through the lobe path is the Lambert path bit for bit (every lobe term is exactly 0);
// - the pooled and albedo-weighted shadow pairs: R^s of a textured cluster under texture_shadow weighted, per
//   dielectric / metal / mixed material, and the capacity rule (more than kUnweightedLights shadow lights: R^s = R);
// - a mean normal that cancels (r_s -> 0) leaves the radiance finite and continuous;
// - the class word's specular bit and its 12-bit shadowed counter;
// - the scene's material views, tables, patches on glossy hard edges, and the renderer on a glossy plane against a
//   per-sub-sample fp64 integral and against the Lambert scene it must leave bit for bit.
// Host = device parity under lobes is test_device.cpp's (GPU).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <random>
#include <vector>

#include "render/embree/embree_tracer.h"
#include "render/executor.h"
#include "render/pose_table.h"
#include "render/scene_data.h"
#include "render/shade/bsdf.h"
#include "render/shade/bsdf_tables.h"
#include "render/shade/shade.h"
#include "render/thread_pool.h"
#include "scenes.h"

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;

namespace {

constexpr double kPi = std::numbers::pi;

// One camera-frame fixture: lights (point lights, the first `shadow` of them casting shadows, then the others), one
// ambient group, the BSDF tables, and the pose table that carries the lights' rows.
struct Fixture {
  std::vector<LightView> lights;
  std::vector<LightRow> rows;
  SceneView s{};
  CameraView cam{};  // cluster_radiance of the kV1 sums reads the camera's nir_fraction (P2-F F11a: zeros, a mono scene)
  PoseTableView poses{};

  void add_light(Vec3f position, float intensity, bool shadow, int32_t group = 0) {
    LightView l{};
    l.kind = LightKind::kPoint;
    l.group = group;
    l.intensity = intensity;
    l.shadow_bit = shadow ? s.num_shadow_lights : -1;
    if (shadow) s.shadow_light[s.num_shadow_lights++] = static_cast<int32_t>(lights.size());
    lights.push_back(l);
    rows.push_back({{position.x, position.y, position.z}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}});
  }
  void finish(float ambient, int32_t groups = 1) {
    s.cameras = &cam;
    s.num_cameras = 1;
    s.lights = lights.data();
    s.num_lights = static_cast<int32_t>(lights.size());
    s.num_groups = groups;
    for (int32_t g = 0; g < groups; ++g) s.ambient[g] = ambient;
    s.specular = 1;
    s.bsdf = bsdf::host_tables();
    poses = PoseTableView{nullptr, rows.data(), 0, s.num_lights};
  }
};

MaterialView lobe_material(float albedo, float roughness, float metallic, float specular, float emissive = 0.0f) {
  MaterialView m{albedo, emissive, -1, -1};
  const double r2 = static_cast<double>(roughness) * static_cast<double>(roughness);
  m.alpha2 = static_cast<float>(r2 * r2);
  m.metallic = metallic;
  m.specular = specular;
  m.lobe = (metallic > 0.0f || specular > 0.0f) ? 1 : 0;
  return m;
}

// The fp64 radiance of a single-member cluster at x with unit normal n: the plugin's expressions (bsdf.h<double>),
// each light as e max(0, n . w_l) f(w_v, w_l) when unshadowed, and the uniform ambient A (rho (1 - m)(1 - E_s) + E_s).
double reference_single(const SceneView& s, const PoseTableView& poses, const MaterialView& m, Vec3f x, Vec3f n,
                        const std::vector<bool>& blocked_shadow_bits) {
  const auto d = [](float f) { return static_cast<double>(f); };
  const Vec3<double> xd{d(x.x), d(x.y), d(x.z)}, nd{d(n.x), d(n.y), d(n.z)};
  const double xl = std::sqrt(dot(xd, xd));
  const Vec3<double> wv = (-1.0 / xl) * xd;
  const bsdf::BsdfParams<double> p{d(m.albedo), d(m.metallic), d(m.alpha2), d(m.specular)};
  double total = 0.0;
  for (int32_t j = 0; j < s.num_lights; ++j) {
    const LightView& l = s.lights[j];
    if (l.shadow_bit >= 0 && blocked_shadow_bits[static_cast<size_t>(l.shadow_bit)]) continue;
    const LightRow& row = light_row(poses, 0, j);
    const Vec3<double> to = Vec3<double>{d(row.position[0]), d(row.position[1]), d(row.position[2])} - xd;
    const double dist2 = dot(to, to);
    const Vec3<double> wl = (1.0 / std::sqrt(dist2)) * to;
    const double mu_l = dot(nd, wl);
    if (mu_l <= 0.0) continue;
    total += d(l.intensity) / dist2 * mu_l * bsdf::bsdf_eval(s.bsdf, p, nd, wv, wl);
  }
  const bsdf::ViewTerms<double> vt = bsdf::view_terms(nd, wv);
  const bsdf::Fresnel<double> fr = bsdf::fresnel_params(p.rho, p.metallic, p.s);
  const double e_s = bsdf::spec_albedo(fr, bsdf::spec_ab(s.bsdf, vt.t, bsdf::table_r(p.alpha2)));
  total += d(s.ambient[0]) * (p.rho * bsdf::diffuse_weight(p.metallic, e_s) + e_s);
  return total + d(m.emissive);
}

// The cluster's sums for members (p_k, n_k, rho_k, occlusion_k), through accumulate as the executors call it.
template <int32_t kShadow, bool kV1, int32_t kUnwLights = unweighted_lights(kShadow, kV1)>
ClusterSums<kShadow, kV1, kUnwLights> sums_of(const SceneView& s, const PoseTableView& poses, const MaterialView& m,
                                              bool textured, const std::vector<Vec3f>& p, const std::vector<Vec3f>& n,
                                              const std::vector<float>& rho, const std::vector<uint32_t>& occ) {
  ClusterSums<kShadow, kV1, kUnwLights> c;
  c.clear();
  const bool weighted = textured && s.texture_shadow != 0;
  const bool unweighted = wants_unweighted(s, m, textured);
  for (size_t k = 0; k < p.size(); ++k) {
    accumulate(c, s, poses, 0, p[k], n[k], rho[k], occ[k], true, weighted, unweighted, true, 0.0f, s.cameras[0].nir_fraction);
  }
  return c;
}

SampleRequest request() { return {0, 0, 0, 0, 1, 0, 0}; }

}  // namespace

TEST_CASE("lobe cluster: a single member equals bsdf.h<double> within 5e-5 (direct light and ambient)") {
  std::mt19937 rng(61);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f), u01(0.0f, 1.0f);
  const float ms[3] = {0.0f, 0.3f, 1.0f}, ss[4] = {0.0f, 0.04f, 0.5f, 1.0f};
  double worst = 0.0;
  int checked = 0;
  for (int i = 0; i < 4000; ++i) {
    Fixture f;
    const float intensity = 1.0f + 9.0f * u01(rng);
    f.add_light({0.8f * u(rng), 0.8f * u(rng), 0.3f * u(rng)}, intensity, true);  // shadow bit 0
    f.add_light({0.8f * u(rng), 0.8f * u(rng), 0.3f * u(rng)}, 0.5f * intensity, false);
    f.finish(0.5f * u01(rng));
    const float rho = 0.05f + 0.95f * u01(rng), r = 0.2f + 0.75f * u01(rng);  // r <= 0.95: alpha^2 <= 0.815, below alpha_eff's cap
    const MaterialView m = lobe_material(rho, r, ms[i % 3], ss[(i / 3) % 4]);
    if (m.lobe == 0) continue;
    const Vec3f x{0.3f * u(rng), 0.3f * u(rng), 0.8f + 0.5f * u01(rng)};
    // A shading normal facing the camera, within 70 degrees of the view.
    Vec3f n;
    do {
      n = normalize(Vec3f{0.9f * u(rng), 0.9f * u(rng), -1.0f});
    } while (dot(n, normalize(-1.0f * x)) < 0.34f);
    const bool blocked = i % 5 == 0;
    const auto sums = sums_of<1, true>(f.s, f.poses, m, false, {x}, {n}, {0.0f}, {blocked ? 1u : 0u});
    float rad[kMaxLightGroups];
    cluster_radiance(f.s, f.poses, request(), m, false, 1, 1, sums, rad, 1.0f);
    const double ref = reference_single(f.s, f.poses, m, x, n, {blocked});
    // Relative to the radiance, with a floor of 0.02 (fp32 tables and the cancellation of 1 - E_s near a mirror leave
    // absolute errors of 1e-7, which only matter where the radiance itself is that small).
    const double rel = std::abs(static_cast<double>(rad[0]) - ref) / std::max(ref, 0.02);
    if (rel > worst) {
      worst = rel;
      MESSAGE("new worst " << rel << ": m " << m.metallic << " s " << m.specular << " alpha2 " << m.alpha2 << " rho "
                           << m.albedo << " blocked " << blocked << " got " << rad[0] << " ref " << ref << " mu_v "
                           << dot(n, normalize(-1.0f * x)));
    }
    ++checked;
    CHECK(rel <= 5e-5);
  }
  MESSAGE("single-member cluster against bsdf.h<double>: " << checked << " configurations, worst relative error "
                                                           << worst << " (gate 5e-5)");
  CHECK(checked > 3000);
}

TEST_CASE("lobe cluster: s = m = 0 through lobe_radiance is the Lambert path bit for bit") {
  std::mt19937 rng(62);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f), u01(0.0f, 1.0f);
  for (int i = 0; i < 400; ++i) {
    Fixture f;
    f.add_light({0.8f * u(rng), 0.8f * u(rng), 0.3f * u(rng)}, 4.0f, true);
    f.add_light({0.8f * u(rng), 0.8f * u(rng), 0.3f * u(rng)}, 2.0f, true, 1);
    f.add_light({0.8f * u(rng), 0.8f * u(rng), 0.3f * u(rng)}, 3.0f, false);
    f.finish(0.3f, 2);
    f.s.specular = 0;  // the Lambert scene's own sums; the lobe path is called directly below
    const bool textured = i % 2 == 0;
    f.s.texture_shadow = i % 4 < 2 ? 1 : 0;
    MaterialView lambert{0.2f + 0.7f * u01(rng), 0.05f * u01(rng), -1, -1};
    MaterialView lobe = lambert;
    lobe.lobe = 1;  // s = m = 0: no lobe term, but the lobe path is taken
    lobe.alpha2 = static_cast<float>(std::pow(0.2 + 0.8 * static_cast<double>(u01(rng)), 4));
    f.s.bsdf = bsdf::host_tables();
    std::vector<Vec3f> p, n;
    std::vector<float> rho;
    std::vector<uint32_t> occ;
    const int members = 1 + i % 7;
    for (int k = 0; k < members; ++k) {
      p.push_back({0.3f * u(rng), 0.3f * u(rng), 1.0f + 0.2f * u(rng)});
      n.push_back(normalize(Vec3f{0.4f * u(rng), 0.4f * u(rng), -1.0f}));
      rho.push_back(0.1f + 0.8f * u01(rng));
      occ.push_back(static_cast<uint32_t>(rng() & 3u));
    }
    const float ao = i % 3 == 0 ? 0.7f : 1.0f;
    const auto sums_v1 = sums_of<4, true>(f.s, f.poses, lobe, textured, p, n, rho, occ);
    const auto sums_l = sums_of<4, false>(f.s, f.poses, lambert, textured, p, n, rho, occ);
    float a[kMaxLightGroups], b[kMaxLightGroups];
    cluster_radiance(f.s, f.poses, request(), lobe, textured, static_cast<uint32_t>(members), static_cast<uint32_t>(members), sums_v1, a, ao);
    cluster_radiance(f.s, f.poses, request(), lambert, textured, static_cast<uint32_t>(members), static_cast<uint32_t>(members), sums_l, b, ao);
    for (int g = 0; g < 2; ++g) CHECK(std::memcmp(&a[g], &b[g], sizeof(float)) == 0);
  }
}

TEST_CASE("lobe cluster: R^s of a textured cluster under texture_shadow weighted, and the capacity rule") {
  // Two members of one cluster, albedos 0.2 (lit) and 0.8 (shadowed), the same shading normal: the pooled ratio
  // sum f V / sum f is 1/2, the albedo-weighted one sum rho f V / sum rho f is 0.2 (f equal).
  const auto build = [&](int shadow_lights, bool weighted_on) {
    Fixture f;
    f.add_light({0.5f, -0.4f, 0.2f}, 6.0f, true);
    for (int j = 1; j < shadow_lights; ++j) f.add_light({-0.6f, 0.3f * static_cast<float>(j), 0.1f}, 0.0f, true);
    f.finish(0.0f);
    f.s.texture_shadow = weighted_on ? 1 : 0;
    return f;
  };
  const Vec3f x{0.0f, 0.0f, 1.0f}, nrm{0.0f, 0.0f, -1.0f};
  struct Case {
    double m, s;
  };
  for (const Case c : {Case{0.0, 0.5}, Case{1.0, 0.0}, Case{0.5, 0.5}}) {
    for (int lights : {1, 4, 5}) {
      for (bool weighted_on : {true, false}) {
        INFO("m = " << c.m << ", s = " << c.s << ", shadow lights " << lights << ", weighted " << weighted_on);
        Fixture f = build(lights, weighted_on);
        const MaterialView mat = lobe_material(0.5f, 0.4f, static_cast<float>(c.m), static_cast<float>(c.s));
        const std::vector<Vec3f> p{x, x}, n{nrm, nrm};
        const std::vector<float> rho{0.2f, 0.8f};
        const std::vector<uint32_t> occ{0u, 1u};
        const auto sums = sums_of<32, true>(f.s, f.poses, mat, true, p, n, rho, occ);
        float rad[kMaxLightGroups];
        cluster_radiance(f.s, f.poses, request(), mat, true, 2, 2, sums, rad, 1.0f);
        // fp64 expectation from the members' own terms: f_k from light 0, V = (1, 0), rho the members' albedos.
        const LightRow& row = light_row(f.poses, 0, 0);
        const double dx = static_cast<double>(row.position[0]), dy = static_cast<double>(row.position[1]), dz = static_cast<double>(row.position[2]) - 1.0;
        const double dist2 = dx * dx + dy * dy + dz * dz, dist = std::sqrt(dist2);
        const Vec3<double> wl{dx / dist, dy / dist, dz / dist}, wv{0.0, 0.0, -1.0}, nd{0.0, 0.0, -1.0};
        const double e = 6.0 / dist2, fk = e * std::max(0.0, dot(nd, wl));
        const double rho_bar = 0.5 * (0.2 + 0.8);
        double ratio_rho = 0.5, ratio_s = 0.5;  // pooled
        if (weighted_on) {
          ratio_rho = (0.2 * fk) / (0.2 * fk + 0.8 * fk);
          const bool unweighted_pair = lights <= kUnweightedLights;
          const double w_d = (1.0 - c.m) * c.s;
          ratio_s = unweighted_pair ? (w_d * fk + c.m * 0.2 * fk) / (w_d * 2.0 * fk + c.m * 1.0 * fk) : ratio_rho;
        }
        const bsdf::BsdfParams<double> pr{rho_bar, c.m, static_cast<double>(mat.alpha2), c.s};
        const double mu_l = std::max(0.0, dot(nd, wl));
        const bsdf::ViewTerms<double> vt = bsdf::view_terms(nd, wv);
        const bsdf::Fresnel<double> fr = bsdf::fresnel_params(pr.rho, pr.metallic, pr.s);
        const double e_s = bsdf::spec_albedo(fr, bsdf::spec_ab(f.s.bsdf, vt.t, bsdf::table_r(pr.alpha2)));
        const double lobe = bsdf::lobe(pr.alpha2, fr, nd, wv, vt, wl);
        const double expected = rho_bar * bsdf::diffuse_weight(pr.metallic, e_s) * e * mu_l * ratio_rho / kPi +
                                e * mu_l * lobe * ratio_s;
        CHECK(static_cast<double>(rad[0]) == doctest::Approx(expected).epsilon(2e-5));
        // The ratios differ materially, so the check discriminates between R^s and R^rho where they should.
        if (weighted_on && lights <= kUnweightedLights && c.m < 1.0 && c.s > 0.0) CHECK(ratio_s - ratio_rho > 0.05);
      }
    }
  }
}

TEST_CASE("lobe cluster: the unweighted pair's capacity per kernel (<2, 32> and <8, 32> carry none)") {
  // device_executor.cu's kUnwLightsOf: the 2- and 8-slot kernels of kMaxShadowLights shadow lights are launched only above
  // kUnweightedLights shadow lights, where R^s = R and the pair is never read, so they hold none (plan p2f.md §4.10:
  // "kShadow <= 4 kernels only"); the 32-slot one keeps it (it takes every scene above N = 256).
  static_assert(unweighted_lights(0, true) == 0 && unweighted_lights(1, true) == 1 && unweighted_lights(4, true) == 4);
  static_assert(unweighted_lights(kMaxShadowLights, true) == kUnweightedLights);
  static_assert(unweighted_lights(kMaxShadowLights, true, false) == 0 && unweighted_lights(4, false) == 0);
  using Full = ClusterSums<kMaxShadowLights, true>;
  using None = ClusterSums<kMaxShadowLights, true, 0>;
  static_assert(None::kUnw == 0 && None::kSize == None::kUnwBase);
  static_assert(Full::kSize == None::kSize + 2 * kUnweightedLights);
  static_assert(ClusterSums<kMaxShadowLights, false>::kUnw == 0 && ClusterSums<4, true>::kSize == 8 + 2 * 4 + 2 * 4);
  // In a scene above the capacity rule the pair-less sums shade exactly as the full ones.
  for (int lights : {5, 8, 32}) {
    Fixture f;
    f.add_light({0.5f, -0.4f, 0.2f}, 6.0f, true);
    for (int j = 1; j < lights; ++j) f.add_light({-0.6f, 0.3f * static_cast<float>(j), 0.1f}, 1.0f, true);
    f.finish(0.1f);
    f.s.texture_shadow = 1;
    const std::vector<Vec3f> p{{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}}, n{{0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, -1.0f}};
    const std::vector<float> rho{0.2f, 0.8f};
    const std::vector<uint32_t> occ{0u, 1u};
    for (const MaterialView& mat : {lobe_material(0.5f, 0.4f, 0.0f, 0.5f), lobe_material(0.5f, 0.4f, 1.0f, 0.0f),
                                    lobe_material(0.5f, 0.4f, 0.5f, 0.5f)}) {
      INFO("shadow lights " << lights << ", metallic " << mat.metallic);
      CHECK_FALSE(wants_unweighted(f.s, mat, true));
      const auto full = sums_of<kMaxShadowLights, true>(f.s, f.poses, mat, true, p, n, rho, occ);
      const auto none = sums_of<kMaxShadowLights, true, 0>(f.s, f.poses, mat, true, p, n, rho, occ);
      float a[kMaxLightGroups], b[kMaxLightGroups];
      cluster_radiance(f.s, f.poses, request(), mat, true, 2, 2, full, a, 1.0f);
      cluster_radiance(f.s, f.poses, request(), mat, true, 2, 2, none, b, 1.0f);
      CHECK(std::memcmp(&a[0], &b[0], sizeof(float)) == 0);
      for (int e = 0; e < None::kSize; ++e) CHECK(std::memcmp(&full.t[e], &none.t[e], sizeof(float)) == 0);
    }
  }
}

TEST_CASE("lobe cluster: a mean normal that cancels leaves the radiance finite and continuous") {
  // |n_u| from 1 down to 1e-7 along one direction, then exactly 0. The widening alpha_eff^2(r_s) is steep near r_s = 1
  // (the model: normals spread by 14 degrees already flatten a highlight), so the continuity check is below r_s = 0.1,
  // where alpha_eff^2 has reached its cap and only Lambert's share (proportional to r_s) is left to move: geometric
  // steps of 0.9 stay within 12 % of the radiance at 0.1, also across r_s = delta_r = 1e-3 (the floor of the width).
  Fixture f;
  f.add_light({0.4f, 0.2f, 0.1f}, 5.0f, false);
  f.finish(0.2f);
  for (const MaterialView& m : {lobe_material(0.5f, 0.3f, 0.0f, 0.5f), lobe_material(0.5f, 0.3f, 1.0f, 0.0f)}) {
    const Vec3f x{0.0f, 0.0f, 1.0f}, n_dir = normalize(Vec3f{0.3f, 0.1f, -1.0f});
    const auto shade_at = [&](double r) {
      const Vec3f nu = static_cast<float>(r) * n_dir;
      const auto sums = sums_of<1, true>(f.s, f.poses, m, false, {x}, {nu}, {0.0f}, {0u});
      float rad[kMaxLightGroups];
      cluster_radiance(f.s, f.poses, request(), m, false, 1, 1, sums, rad, 1.0f);
      REQUIRE(std::isfinite(rad[0]));
      REQUIRE(rad[0] >= 0.0f);
      return static_cast<double>(rad[0]);
    };
    for (const double r : {1.0, 0.5, 1e-2, 1e-3, 1e-5, 0.0}) shade_at(r);
    const double at_01 = shade_at(0.1);
    double previous = at_01, largest_step = 0.0;
    for (double r = 0.09; r > 1e-7; r *= 0.9) {
      const double v = shade_at(r);
      largest_step = std::max(largest_step, std::abs(v - previous));
      previous = v;
    }
    CHECK(largest_step <= 0.12 * at_01);
    MESSAGE("largest step " << largest_step << " of " << at_01 << " (r_s = 0.1); at r_s = 0: " << shade_at(0.0));
  }
}

TEST_CASE("class word: the specular bit (alpha <= 1/2, a lobe) and the 12-bit shadowed counter") {
  CHECK_FALSE(class_specular(lobe_material(0.5f, 0.5f, 0.0f, 0.0f)));             // no lobe
  CHECK(class_specular(lobe_material(0.5f, std::sqrt(0.5f), 0.0f, 0.04f)));       // alpha = r^2 = 0.5 - 1e-ish
  CHECK_FALSE(class_specular(lobe_material(0.5f, 0.75f, 0.0f, 0.04f)));           // alpha = 0.5625
  CHECK(class_specular(lobe_material(0.5f, 0.2f, 1.0f, 0.0f)));
  CHECK((kClassSpecular & (kClassMulti | kClassShadowEdge | kClassTextured | kClassShadowed)) == 0);
  CHECK(kClassShadowed == 4095u);
  CHECK(sample_class(0u, false, 0u, false, true) == kClassSpecular);
  CHECK(sample_class(0u, false, 0u, false, false) == 0u);
  CHECK((sample_class(4095u, false, 0u, false, false) & kClassShadowed) == 4095u);
  CHECK((sample_class(4096u, false, 0u, false, false) & (kClassShadowed | kClassSpecular)) == 0u);
}

TEST_CASE("scene data: material views, tables, glossy patches and Lambert scenes untouched") {
  // Two bodies on a plane: materials r = 0.3 / m = 0.5 / s = 0.04 and a plain one.
  const auto bundle = [&](bool glossy) {
    SceneBundle s;
    s.light_groups = {"default"};
    s.trajectories.push_back(static_trajectory(1'000'000'000, 1'200'000'000, quat_identity<double>(), {0, 0, 0}));
    s.cameras.push_back(pinhole(32, 24, 32.0, 0));
    Body a = make_body(5, plane_mesh(0.5), {0.4f, 0.4f, 0.4f});
    a.T_world_body = translation(0, 0, 2.0);
    if (glossy) {
      a.materials[0].roughness = 0.3f;
      a.materials[0].metallic = 0.5f;
      a.materials[0].specular = 0.04f;
    }
    Body b = make_body(6, plane_mesh(0.2), {0.9f, 0.9f, 0.9f});
    b.T_world_body = translation(0.7, 0, 2.0);
    s.bodies = {a, b};
    s.lights = {ambient_light(0.2f)};
    return s;
  };
  SceneSettings st;
  const SceneData lam(bundle(false), st), glo(bundle(true), st);
  const SceneView vl = lam.host_view(), vg = glo.host_view();
  CHECK(vl.specular == 0);
  CHECK(vl.bsdf.spec_a == nullptr);
  CHECK_FALSE(lam.has_specular());
  REQUIRE(glo.has_specular());
  CHECK(vg.specular == 1);
  CHECK(vg.bsdf.spec_a == bsdf::host_tables().spec_a);
  const MaterialView& m = glo.materials()[0];
  CHECK(m.lobe == 1);
  CHECK(m.metallic == 0.5f);
  CHECK(m.specular == 0.04f);
  CHECK(m.alpha2 == doctest::Approx(0.0081).epsilon(1e-6));  // alpha = r^2: alpha^2 = r^4 (r = 0.3f)
  CHECK(glo.materials()[1].lobe == 0);
  CHECK(glo.materials()[1].alpha2 == 0.0f);
  CHECK(glo.materials()[1].albedo == lam.materials()[1].albedo);
}

TEST_CASE("scene data: a hard edge of 10 degrees cuts a glossy body's patch (rule (b)) and not a Lambert one's") {
  // Two flat-shaded quads sharing an edge, turned by 10 degrees (below the 30 degree crease): one patch for a Lambert
  // material, two for a material with a lobe (a highlight is discontinuous at a hard edge at any angle).
  const double a = 10.0 * std::numbers::pi / 180.0;
  const float c = static_cast<float>(std::cos(a)), s = static_cast<float>(std::sin(a));
  const auto bundle = [&](bool glossy) {
    SceneBundle b;
    b.light_groups = {"default"};
    b.trajectories.push_back(static_trajectory(1'000'000'000, 1'200'000'000, quat_identity<double>(), {0, 0, 0}));
    b.cameras.push_back(pinhole(32, 24, 32.0, 0));
    Mesh m;
    m.positions = {-1, 0, 0, 0, 0, 0, 0, 1, 0, -1, 1, 0, 0, 0, 0, c, 0, s, c, 1, s, 0, 1, 0};
    m.indices = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};
    m.normals = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, -s, 0, c, -s, 0, c, -s, 0, c, -s, 0, c};
    Body body = make_body(5, m, {0.5f, 0.5f, 0.5f});
    body.T_world_body = translation(0, 0, 2.0);
    if (glossy) body.materials[0].specular = 0.04f;
    b.bodies = {body};
    b.lights = {ambient_light(0.2f)};
    return b;
  };
  SceneSettings crease;
  crease.shading_patches = ShadingPatches::kCrease;
  const SceneData lam(bundle(false), crease), glo(bundle(true), crease);
  CHECK(lam.patch_material().size() == 1);
  CHECK(glo.patch_material().size() == 2);
}

TEST_CASE("renderer: a glossy plane against the per-sub-sample fp64 integral; a Lambert plane next to a lobe is Lambert") {
  constexpr uint32_t kW = 24, kH = 18, kN = 64, kPixels = kW * kH;
  constexpr int64_t kT0 = 1'000'000'000, kT1 = 1'200'000'000;
  const auto bundle = [&](bool glossy_far_body) {
    SceneBundle s;
    s.light_groups = {"default"};
    s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
    s.cameras.push_back(pinhole(kW, kH, 24.0, 0));
    Body plane = make_body(5, plane_mesh(1.5), {0.6f, 0.6f, 0.6f});
    plane.T_world_body = translation(0, 0, 1.5);
    plane.materials[0].roughness = 0.5f;
    plane.materials[0].specular = glossy_far_body ? 0.0f : 0.5f;
    s.bodies.push_back(plane);
    if (glossy_far_body) {  // a glossy body out of the view: the scene has a lobe, the plane does not
      Body far = make_body(6, box_mesh(0.1f, 0.1f, 0.1f), {0.5f, 0.5f, 0.5f});
      far.T_world_body = translation(0, 0, -5.0);
      far.materials[0].roughness = 0.4f;
      far.materials[0].specular = 0.5f;
      s.bodies.push_back(far);
    }
    s.lights = {point_light({0.5, -0.3, 0.2}, 3.0f, true), point_light({-0.4, 0.2, 0.3}, 1.0f, false),
                ambient_light(0.15f)};
    return s;
  };
  RendererOptions o = options(BackendKind::kEmbree);
  ThreadPool pool(4);
  SceneSettings st = o.scene;
  st.n_subsamples = kN;
  struct Shot {
    std::vector<float> irr, ref;  // ref: the per-sub-sample fp64 integral (with_ref)
    std::vector<uint32_t> klass;
  };
  const auto shoot = [&](const SceneBundle& b, bool with_ref) {
    const SceneData data(b, st);
    HostExecutor host(pool);
    host.upload_scene(data);
    EmbreeTracer tracer(pool);
    tracer.build(data, host.scene_view());
    PoseTable table(static_cast<int32_t>(data.geometries().size()), static_cast<int32_t>(data.lights().size()), 64);
    table.reset(kT0);
    uint32_t slot = 0;
    REQUIRE(table.slot(0, kT0 + 100'000'000, &slot));
    const PoseTableView poses = host.sync_poses(table, 0);
    std::vector<SampleRequest> req;
    for (uint32_t p = 0; p < kPixels; ++p) req.push_back({0, 0, p, 0, kN, 0, slot});
    const TraceBatch batch{req.data(), kPixels, kN, poses, nullptr, 0};
    std::vector<HitRecord> hits(static_cast<size_t>(kPixels) * kN);
    std::vector<uint32_t> occluded(static_cast<size_t>(kPixels) * kN);
    tracer.sample(batch, hits.data(), occluded.data());
    Shot shot;
    shot.irr.resize(kPixels);
    shot.klass.resize(kPixels);
    std::vector<uint64_t> sig(kPixels);
    host.shade_into(0, batch, hits.data(), occluded.data(), shot.irr.data(), sig.data(), nullptr, nullptr,
                    shot.klass.data());
    if (with_ref) {
      // The per-sub-sample fp64 integral: each sub-sample's own point, normal and view through bsdf.h<double>.
      const SceneView view = host.scene_view();
      const GeometryView& geo = view.geometries[0];  // the static world
      const MaterialView& m = view.materials[0];
      shot.ref.assign(kPixels, 0.0f);
      for (uint32_t p = 0; p < kPixels; ++p) {
        double sum = 0.0;
        for (uint32_t k = 0; k < kN; ++k) {
          const HitRecord& h = hits[static_cast<size_t>(p) * kN + k];
          REQUIRE(h.instance == 5u);
          const Vec3f dir = subsample_direction(view, view.cameras[0], p, k);
          const SurfaceSample ss = surface_sample(geo, pose_row(poses, slot, 0), h.prim, dir, false);
          sum += reference_single(view, poses, m, ss.p, ss.n_shade,
                                  {((occluded[static_cast<size_t>(p) * kN + k] >> 0) & 1u) != 0});
        }
        shot.ref[p] = static_cast<float>(sum / kN);
      }
    }
    return shot;
  };
  // (1) The glossy plane (s = 0.5, r = 0.5) against the fp64 integral of the per-sub-sample BSDF.
  const Shot glossy = shoot(bundle(false), true);
  REQUIRE(glossy.ref.size() == kPixels);
  int outside = 0;
  double worst = 0.0;
  for (uint32_t p = 0; p < kPixels; ++p) {
    const double got = glossy.irr[p], ref = glossy.ref[p];
    const double rel = std::abs(got - ref) / std::max(ref, 1e-3);
    worst = std::max(worst, rel);
    outside += rel > 0.01;
    CHECK((glossy.klass[p] & kClassSpecular) != 0);  // alpha = 0.25 <= 1/2
  }
  MESSAGE("glossy plane: worst relative difference to the per-sub-sample fp64 integral " << worst << ", " << outside
                                                                                           << " of " << kPixels
                                                                                           << " pixels above 1 %");
  CHECK(outside <= static_cast<int>(kPixels / 100));  // DESIGN §7.11: 1 % on 99 % of single-cluster pixels
  CHECK(worst <= 0.02);
  // (2) The Lambert plane, with and without a glossy body elsewhere in the scene: bit for bit, no specular bit.
  const Shot alone = shoot(bundle(true), false);
  SceneBundle only = bundle(true);
  only.bodies.pop_back();
  const Shot lone = shoot(only, false);
  REQUIRE(alone.irr.size() == kPixels);
  CHECK(std::memcmp(alone.irr.data(), lone.irr.data(), kPixels * sizeof(float)) == 0);
  for (uint32_t p = 0; p < kPixels; ++p) CHECK((alone.klass[p] & kClassSpecular) == 0);
}
