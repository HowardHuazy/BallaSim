// C1 feature "baked" in render::validate (docs/design/phase3/scans.md §3.1, §8 T7): the same rules as
// bundle.py's validate(), on bundles built as an exporter would write them, and the renderer accepting them, tile
// atlases included (S5a); the renderer's refusal of what it does not render yet (rect and disk lights, P2-F F8a-2), and
// the reflection policy (render/reflection_policy.h, DESIGN §7.7).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "render/bundle.h"
#include "render/reflection_policy.h"
#include "render/renderer.h"
#include "render/scene_data.h"
#include "scenes.h"

namespace balla::render::test {
namespace {

constexpr int64_t kT0 = 0, kT1 = 100'000'000;

// A static scan room (baked, 4 x 4 binary16 texture, shares over two groups), a lit mover, a fill and a point proxy.
SceneBundle baked_scene() {
  SceneBundle s;
  s.light_groups = {"default", "mains"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, Quat<double>{1, 0, 0, 0}, {0, 0, 0}));
  s.trajectories.push_back(moving_trajectory(kT0, kT1, Quat<double>{1, 0, 0, 0}, {0, 0, 1.0}, {0.1, 0, 0}, {0, 0, 0}));
  s.cameras.push_back(pinhole(16, 12, 20.0, 0));
  Body room = make_body(1, plane_mesh(2.0, 2));
  room.T_world_body = translation(0, 0, 2.0);
  room.scan = true;
  Material& m = room.materials[0];
  m.shading = Shading::kBaked;
  m.baked_scale = 75.0f;
  m.baked_groups = {0.3f, 0.7f};
  m.texture_linear.assign(4 * 4 * 3, 0x3C00);  // 1.0
  m.texture_linear[0] = 0x8000;                // -0 passes
  m.texture_linear[1] = 0x0001;                // the smallest subnormal
  m.texture_linear[2] = 0x7BFF;                // 65504, the largest finite value
  m.texture_width = m.texture_height = 4;
  m.texture_channels = 3;
  m.texture_tile = 4;
  s.bodies.push_back(room);
  Body box = make_body(2, box_mesh(0.1f, 0.1f, 0.1f));
  box.trajectory = 1;
  s.bodies.push_back(box);
  Light fill = ambient_light(0.2f, 1);
  fill.proxy = true;
  s.lights.push_back(fill);
  Light lamp = point_light({0.0, 0.5, 0.5}, 3.0f, true, 1);
  lamp.proxy = true;
  s.lights.push_back(lamp);
  return s;
}

template <class Edit>
std::string refusal(Edit edit) {
  SceneBundle s = baked_scene();
  edit(s);
  try {
    validate(s);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

bool contains(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

}  // namespace

TEST_CASE("C1 feature baked: a valid scan bundle passes; the defaults are Phase 1's") {
  CHECK_NOTHROW(validate(baked_scene()));
  const Material m;
  const Body b;
  const Light l;
  CHECK(m.shading == Shading::kLit);
  CHECK(m.baked_scale == 1.0f);
  CHECK((m.baked_groups.empty() && m.texture_linear.empty() && m.texture_tile == 0));
  CHECK((!b.scan && !l.proxy));
  // A fill proxy may cast no shadows (ambient and background are constants); a lit material in a scan body is
  // captured geometry too.
  CHECK(refusal([](SceneBundle& s) { s.lights[0].casts_shadows = false; }).empty());
  CHECK(refusal([](SceneBundle& s) {
          s.bodies[0].materials.push_back(Material{});
          s.bodies[0].mesh.material_ids.assign(s.bodies[0].mesh.indices.size() / 3, 0);
          s.bodies[0].mesh.material_ids[1] = 1;
        }).empty());
}

TEST_CASE("C1 feature baked: every rule of scans.md §3.1 refuses, naming the item") {
  struct Case {
    const char* name;
    void (*edit)(SceneBundle&);
    const char* message;
  };
  const Case cases[] = {
      {"negative K", [](SceneBundle& s) { s.bodies[0].materials[0].baked_scale = -1.0f; }, "baked_scale"},
      {"infinite K",
       [](SceneBundle& s) { s.bodies[0].materials[0].baked_scale = std::numeric_limits<float>::infinity(); },
       "baked_scale"},
      {"shares of a lit material",
       [](SceneBundle& s) {
         s.bodies[1].materials[0].baked_groups = {1.0f, 0.0f};
       },
       "baked_groups is set on a lit material"},
      {"one share per group", [](SceneBundle& s) { s.bodies[0].materials[0].baked_groups = {1.0f}; },
       "one share per light group"},
      {"negative share", [](SceneBundle& s) { s.bodies[0].materials[0].baked_groups = {1.5f, -0.5f}; }, ">= 0"},
      {"shares sum", [](SceneBundle& s) { s.bodies[0].materials[0].baked_groups = {0.3f, 0.71f}; }, "sum to 1"},
      {"NaN texel", [](SceneBundle& s) { s.bodies[0].materials[0].texture_linear[5] = 0x7E00; }, "NaN or Inf"},
      {"Inf texel", [](SceneBundle& s) { s.bodies[0].materials[0].texture_linear[5] = 0x7C00; }, "NaN or Inf"},
      {"negative texel", [](SceneBundle& s) { s.bodies[0].materials[0].texture_linear[5] = 0x8001; }, "negative"},
      {"both textures",
       [](SceneBundle& s) {
         Material& m = s.bodies[0].materials[0];
         m.texture.assign(4 * 4 * 3, 128);
       },
       "both texture and texture_linear"},
      {"linear texture on a lit material",
       [](SceneBundle& s) {
         Material& m = s.bodies[0].materials[0];
         m.shading = Shading::kLit;
         m.baked_groups.clear();
       },
       "texture_linear is set on a lit material"},
      {"texture size", [](SceneBundle& s) { s.bodies[0].materials[0].texture_linear.pop_back(); }, "H x W x 3"},
      {"emission on baked", [](SceneBundle& s) { s.bodies[0].materials[0].emissive = {0.0f, 0.1f, 0.0f}; },
       "already contains emission"},
      {"tile not a power of two", [](SceneBundle& s) { s.bodies[0].materials[0].texture_tile = 6; }, "power of two"},
      {"tile below 4", [](SceneBundle& s) { s.bodies[0].materials[0].texture_tile = 2; }, "power of two"},
      {"tile not dividing", [](SceneBundle& s) { s.bodies[0].materials[0].texture_tile = 8; }, "divide"},
      {"tile without texture", [](SceneBundle& s) { s.bodies[1].materials[0].texture_tile = 4; }, "divide"},
      {"proxy point without shadows", [](SceneBundle& s) { s.lights[1].casts_shadows = false; }, "must cast shadows"},
      {"proxy on a moving carrier", [](SceneBundle& s) { s.lights[1].body = 1; }, "captured, static lighting"},
      {"proxies without a scan body", [](SceneBundle& s) { s.bodies[0].scan = false; }, "need a scan body"},
      {"scan on a moving body",
       [](SceneBundle& s) {
         s.bodies[1].scan = true;
         s.bodies[1].materials[0] = s.bodies[0].materials[0];
         s.bodies[1].materials[0].texture_tile = 0;
         s.bodies[1].mesh.uvs.assign(s.bodies[1].mesh.positions.size() / 3 * 2, 0.5f);
       },
       "a scan body is static"},
      {"scan on a lit-only body",
       [](SceneBundle& s) {
         s.bodies[1].scan = true;
         s.bodies[1].trajectory.reset();
       },
       "none of the body's used materials is baked"},
  };
  for (const Case& c : cases) {
    CAPTURE(c.name);
    const std::string what = refusal(c.edit);
    CHECK(contains(what, c.message));
  }
}

TEST_CASE("the renderer renders baked materials (scans S1b) and tile atlases (S5a)") {
  SceneBundle baked = baked_scene();
  CHECK_NOTHROW(Renderer(baked, options(BackendKind::kEmbree, 1)));
  baked.bodies[0].materials[0].texture_tile = 0;
  CHECK_NOTHROW(Renderer(baked, options(BackendKind::kEmbree, 1)));
  // A lit tile atlas, with the scan and proxies gone.
  SceneBundle tiled = baked_scene();
  tiled.lights.clear();
  tiled.bodies[0].scan = false;
  Material& m = tiled.bodies[0].materials[0];
  m.shading = Shading::kLit;
  m.baked_groups.clear();
  m.texture_linear.clear();
  m.texture.assign(4 * 4 * 3, 128);
  CHECK_NOTHROW(validate(tiled));
  CHECK_NOTHROW(Renderer(tiled, options(BackendKind::kEmbree, 1)));
  m.texture_tile = 0;
  CHECK_NOTHROW(Renderer(tiled, options(BackendKind::kEmbree, 1)));
}

// ---------------------------------------------------------------------------------------------------------------
// P2-F F4: the C1 features "pbr", "area_light" and "nir" (p2f.md §4.3, §4.6, §4.9, §4.11) and the reflection policy.

namespace {

// A static floor (a lit plane of albedo 0.5 at z = 0, half-size 1) under a white point light of intensity 4 at
// (0, 0, 2) and an ambient 0.2; the camera rig is static. The floor's material is the one under test.
SceneBundle pbr_scene() {
  SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, Quat<double>{1, 0, 0, 0}, {0, 0, -1.0}));
  s.cameras.push_back(pinhole(16, 12, 20.0, 0));
  Body floor = make_body(1, plane_mesh(1.0, 2), {0.5f, 0.5f, 0.5f});
  floor.materials[0].roughness = 0.3f;
  floor.materials[0].specular = 0.04f;
  s.bodies.push_back(floor);
  s.lights.push_back(point_light({0.0, 0.0, 2.0}, 4.0f));
  s.lights.push_back(ambient_light(0.2f));
  return s;
}

Light rect_light(double s_u, double s_v) {
  Light l;
  l.kind = BundleLightKind::kRect;
  l.intensity = 2.0f;
  l.position = {0.0, 0.0, 1.5};
  l.direction = {0.0, 0.0, -1.0};
  l.size = {s_u, s_v};
  return l;
}

template <class Edit>
std::string pbr_refusal(Edit edit) {
  SceneBundle s = pbr_scene();
  edit(s);
  try {
    validate(s);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

constexpr std::array<double, 3> kGrey{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

}  // namespace

TEST_CASE("C1 features pbr, area_light, nir: the defaults are Lambert and point lights; valid values pass") {
  const Material m;
  const Body b;
  const Light l;
  CHECK((m.roughness == 1.0f && m.metallic == 0.0f && m.specular == 0.0f && !m.has_lobe()));
  CHECK((m.nir_form == NirForm::kAssumed && m.nir_params.empty() && m.nir_texture.empty()));
  CHECK(b.normal_prefilter_m == 0.0);
  CHECK((l.size == std::array<double, 2>{0.0, 0.0} && l.axis == std::array<double, 3>{1.0, 0.0, 0.0}));
  CHECK(pbr_refusal([](SceneBundle&) {}).empty());
  CHECK(pbr_refusal([](SceneBundle& s) {
          Material& f = s.bodies[0].materials[0];
          f.roughness = 0.0f;
          f.metallic = 1.0f;
          f.specular = 1.0f;
          f.nir_form = NirForm::kTexture;
          f.nir_texture.assign(6, 200);
          f.nir_texture_width = 3;
          f.nir_texture_height = 2;
          s.bodies[0].normal_prefilter_m = 5e-4;
          s.lights.push_back(rect_light(0.3, 0.2));
          Light disk = rect_light(0.1, 0.1);
          disk.kind = BundleLightKind::kDisk;
          disk.direction = {0.0, 1.0, 0.0};
          disk.axis = {0.0, 0.0, 1.0};
          s.lights.push_back(disk);
        }).empty());
}

TEST_CASE("C1 features pbr, area_light, nir: every rule refuses, naming the item") {
  struct Case {
    const char* name;
    std::function<void(SceneBundle&)> edit;
    const char* message;
  };
  const auto mat = [](SceneBundle& s) -> Material& { return s.bodies[0].materials[0]; };
  const std::vector<Case> cases{
      {"roughness > 1", [&](SceneBundle& s) { mat(s).roughness = 1.5f; }, "roughness, metallic and specular"},
      {"metallic < 0", [&](SceneBundle& s) { mat(s).metallic = -0.1f; }, "roughness, metallic and specular"},
      {"specular NaN", [&](SceneBundle& s) { mat(s).specular = std::numeric_limits<float>::quiet_NaN(); },
       "roughness, metallic and specular"},
      {"baked with a lobe",
       [&](SceneBundle& s) {
         mat(s).shading = Shading::kBaked;
         s.bodies[0].scan = true;
       },
       "a baked material is radiance: roughness"},
      {"baked with nir",
       [&](SceneBundle& s) {
         mat(s) = Material{};
         mat(s).shading = Shading::kBaked;
         mat(s).nir_form = NirForm::kScale;
         s.bodies[0].scan = true;
       },
       "it has no NIR albedo"},
      {"nir_params of 1", [&](SceneBundle& s) { mat(s).nir_params = {1.0}; }, "nir_params must be 2 finite values"},
      {"nir_params NaN", [&](SceneBundle& s) { mat(s).nir_params = {1.0, std::nan("")}; },
       "nir_params must be 2 finite values"},
      {"scale below 0",
       [&](SceneBundle& s) {
         mat(s).nir_form = NirForm::kScale;
         mat(s).nir_params = {-0.1, 0.0};
       },
       "must be >= 0"},
      {"scalar below 0",
       [&](SceneBundle& s) {
         mat(s).nir_form = NirForm::kScalar;
         mat(s).nir_params = {-0.1, 0.0};
       },
       "must be >= 0"},
      {"texture form without a texture", [&](SceneBundle& s) { mat(s).nir_form = NirForm::kTexture; },
       "nir_form texture needs nir_texture"},
      {"nir texture on another form",
       [&](SceneBundle& s) {
         mat(s).nir_texture.assign(4, 1);
         mat(s).nir_texture_width = mat(s).nir_texture_height = 2;
       },
       "nir_texture is set on a form other than texture"},
      {"nir texture size",
       [&](SceneBundle& s) {
         mat(s).nir_form = NirForm::kTexture;
         mat(s).nir_texture.assign(5, 1);
         mat(s).nir_texture_width = mat(s).nir_texture_height = 2;
       },
       "nir_texture must be H x W"},
      {"prefilter < 0", [](SceneBundle& s) { s.bodies[0].normal_prefilter_m = -1e-3; }, "normal_prefilter_m"},
      {"prefilter inf", [](SceneBundle& s) { s.bodies[0].normal_prefilter_m = INFINITY; }, "normal_prefilter_m"},
      {"rect size 0", [](SceneBundle& s) { s.lights.push_back(rect_light(0.3, 0.0)); }, "needs size [s_u, s_v] > 0"},
      {"rect direction not unit",
       [](SceneBundle& s) {
         Light l = rect_light(0.3, 0.3);
         l.direction = {0.0, 0.0, -2.0};
         s.lights.push_back(l);
       },
       "unit direction and axis"},
      {"rect axis out of plane",
       [](SceneBundle& s) {
         Light l = rect_light(0.3, 0.3);
         l.axis = {0.0, 0.6, 0.8};
         s.lights.push_back(l);
       },
       "axis must lie in the emitter's plane"},
      {"size on a point light", [](SceneBundle& s) { s.lights[0].size = {0.1, 0.1}; },
       "size and axis belong to rect and disk lights"},
      {"axis on a point light", [](SceneBundle& s) { s.lights[0].axis = {0.0, 1.0, 0.0}; },
       "size and axis belong to rect and disk lights"},
  };
  for (const Case& c : cases) {
    CAPTURE(c.name);
    CHECK(contains(pbr_refusal(c.edit), c.message));
  }
}

TEST_CASE("the renderer renders specular lobes (F6) and refuses area lights until P2-F renders them; Lambert pbr and nir render") {
  SceneBundle glossy = pbr_scene();
  CHECK_NOTHROW(Renderer(glossy, options(BackendKind::kEmbree, 1)));  // s > 0 (F6 shades it; test_lobe.cpp)
  glossy.bodies[0].materials[0].specular = 0.0f;
  glossy.bodies[0].materials[0].metallic = 0.5f;
  CHECK_NOTHROW(Renderer(glossy, options(BackendKind::kEmbree, 1)));
  SceneBundle area = pbr_scene();
  area.bodies[0].materials[0] = make_body(1, Mesh{}, {0.5f, 0.5f, 0.5f}).materials[0];
  area.lights.push_back(rect_light(0.3, 0.3));
  CHECK_THROWS_WITH_AS(Renderer(area, options(BackendKind::kEmbree, 1)), doctest::Contains("rect and disk lights"),
                       std::invalid_argument);
  // SceneData itself carries the area light's data since F8a-1 (the shadow rays; tests/render/test_area_lights.cpp), as
  // its own kind: never a directional light by fall-through.
  CHECK(SceneData(area, options(BackendKind::kEmbree, 1).scene).lights().back().kind == LightKind::kRect);
  area.lights.back().kind = BundleLightKind::kDisk;
  CHECK(SceneData(area, options(BackendKind::kEmbree, 1).scene).lights().back().kind == LightKind::kDisk);
  // s = m = 0 with any roughness, NIR fields and a prefilter radius: the Lambert floor, bit for bit (f_NIR = 0).
  SceneBundle plain = pbr_scene(), lambert = pbr_scene();
  plain.bodies[0].materials[0] = make_body(1, Mesh{}, {0.5f, 0.5f, 0.5f}).materials[0];
  Material& m = lambert.bodies[0].materials[0];
  m.specular = 0.0f;
  m.roughness = 0.05f;
  m.nir_form = NirForm::kAffine;
  m.nir_params = {0.132, 0.745};
  lambert.bodies[0].normal_prefilter_m = 5e-4;
  Renderer ra(plain, options(BackendKind::kEmbree, 1)), rb(lambert, options(BackendKind::kEmbree, 1));
  std::vector<uint16_t> cams(16 * 12, 0);
  std::vector<uint32_t> pixels(16 * 12);
  std::vector<int64_t> ts(16 * 12, kT0 + 1000);
  for (uint32_t p = 0; p < pixels.size(); ++p) pixels[p] = p;
  const RenderResult a = ra.render(cams, pixels, ts), b = rb.render(cams, pixels, ts);
  CHECK(a.irradiance == b.irradiance);
  CHECK(a.signature == b.signature);
  CHECK(*std::max_element(a.irradiance.begin(), a.irradiance.end()) > 0.0f);
}

TEST_CASE("reflection policy: DESIGN §7.7's thresholds; no report without a lobe") {
  const auto report = [](float r, float m, float s) {
    SceneBundle b = pbr_scene();
    Material& f = b.bodies[0].materials[0];
    f.roughness = r;
    f.metallic = m;
    f.specular = s;
    return reflection_policy(b, kGrey);
  };
  CHECK(report(0.1f, 0.0f, 0.0f).empty());   // s = m = 0: Lambert, nothing to miss
  CHECK(report(0.19f, 0.0f, 0.04f).at(0).refused);
  CHECK_FALSE(report(0.2f, 0.0f, 0.04f).at(0).refused);
  CHECK(report(0.49f, 0.5f, 0.04f).at(0).refused);
  CHECK(report(0.49f, 0.5f, 0.0f).at(0).refused);
  CHECK_FALSE(report(0.5f, 1.0f, 0.0f).at(0).refused);
  CHECK_FALSE(report(0.3f, 0.0f, 0.04f).at(0).refused);
  // An unused material (no triangle names it) is not reported.
  SceneBundle b = pbr_scene();
  b.bodies[0].materials.push_back(b.bodies[0].materials[0]);
  b.bodies[0].materials[0].specular = 0.0f;
  b.bodies[0].mesh.material_ids.assign(b.bodies[0].mesh.indices.size() / 3, 0);
  CHECK(reflection_policy(b, kGrey).empty());
}

TEST_CASE("reflection policy: the bound against a hand computation") {
  const double pi = std::numbers::pi;
  // rho = 0.5, E_max = 4 / 2^2 (the floor straight below the light), A = 0.2:
  // L_max = 0.5 (1/pi + 0.2), L_diff,min = 0.5 * 0.2, bound = ln(1 + L_max / L_diff,min).
  SceneBundle s = pbr_scene();
  const auto r0 = reflection_policy(s, kGrey);
  REQUIRE(r0.size() == 1);
  CHECK((r0[0].body == 0 && r0[0].material == 0));
  CHECK(r0[0].roughness == 0.3f);
  CHECK(r0[0].l_max == doctest::Approx(0.5 * (1.0 / pi + 0.2)).epsilon(1e-6));
  CHECK(r0[0].l_diff_min == doctest::Approx(0.1).epsilon(1e-6));
  CHECK(r0[0].bound == doctest::Approx(std::log1p(0.5 * (1.0 / pi + 0.2) / 0.1)).epsilon(1e-6));
  // A white static box whose face x = 0.9 is the light's nearest geometry (exact point-to-triangle distance).
  Body box = make_body(2, box_mesh(0.1f, 0.1f, 0.1f));
  box.T_world_body = translation(1.0, 0.0, 2.0);
  s.bodies.push_back(box);
  const double e = 4.0 / (0.9 * 0.9);
  CHECK(reflection_policy(s, kGrey).at(0).l_max == doctest::Approx(1.0 * (e / pi + 0.2)).epsilon(1e-6));
  // A rect light of radiance 2 adds E <= pi L.
  s.lights.push_back(rect_light(0.3, 0.3));
  CHECK(reflection_policy(s, kGrey).at(0).l_max == doctest::Approx(1.0 * ((e + pi * 2.0) / pi + 0.2)).epsilon(1e-6));
  // A metal, or no ambient: nothing bounds the missing reflection.
  SceneBundle metal = pbr_scene();
  metal.bodies[0].materials[0].metallic = 1.0f;
  metal.bodies[0].materials[0].roughness = 0.6f;
  CHECK(reflection_policy(metal, kGrey).at(0).l_diff_min == 0.0);
  CHECK(std::isinf(reflection_policy(metal, kGrey).at(0).bound));
  SceneBundle dark = pbr_scene();
  dark.lights.pop_back();
  CHECK(std::isinf(reflection_policy(dark, kGrey).at(0).bound));
  // A textured floor: L_diff,min takes the darkest texel and L_max the brightest (sRGB 0 and 255 here); without UVs
  // the texture renders as its mean, so both take the mean.
  SceneBundle tex = pbr_scene();
  Material& t = tex.bodies[0].materials[0];
  t.base_color = {1.0f, 1.0f, 1.0f};
  t.texture = {0, 0, 0, 255, 255, 255};
  t.texture_width = 2;
  t.texture_height = 1;
  t.texture_channels = 3;
  const auto rt = reflection_policy(tex, kGrey);
  CHECK(rt.at(0).l_diff_min == 0.0);
  CHECK(rt.at(0).l_max == doctest::Approx(1.0 / pi + 0.2).epsilon(1e-6));
  tex.bodies[0].mesh.uvs.clear();
  const auto rm = reflection_policy(tex, kGrey);
  CHECK(rm.at(0).l_diff_min == doctest::Approx(0.5 * 0.2).epsilon(1e-6));
  CHECK(rm.at(0).l_max == doctest::Approx(0.5 * (1.0 / pi + 0.2)).epsilon(1e-6));
}

TEST_CASE("reflection policy: rho_max and rho_min take the cameras' NIR mix (1 - f) rho_vis + f rho_NIR") {
  const double pi = std::numbers::pi;
  const auto lmax = [&](double rho) { return rho * (1.0 / pi + 0.2); };  // pbr_scene: E_max = 4 / 2^2, A = 0.2
  SceneBundle s = pbr_scene();                                           // rho_vis 0.5
  Material& m = s.bodies[0].materials[0];
  m.nir_form = NirForm::kScalar;
  m.nir_params = {0.8, 0.0};  // rho_NIR 0.8
  const auto report = [&](const std::vector<std::vector<double>>& f) {
    return reflection_policy(s, kGrey, {}, f).at(0);
  };
  // No row, an empty row and f = 0: the visible albedo, bit for bit what the policy gave without nir_fraction.
  const ReflectionReport base = reflection_policy(s, kGrey).at(0);
  for (const auto& f : {std::vector<std::vector<double>>{}, std::vector<std::vector<double>>{{}},
                        std::vector<std::vector<double>>{{0.0}}}) {
    const ReflectionReport r = report(f);
    CHECK((r.l_max == base.l_max && r.l_diff_min == base.l_diff_min && r.bound == base.bound));
  }
  CHECK(base.l_max == doctest::Approx(lmax(0.5)).epsilon(1e-6));
  // One camera at f = 1 (a strobe): rho_NIR alone; f = 0.5: the mix 0.65.
  CHECK(report({{1.0}}).l_max == doctest::Approx(lmax(0.8)).epsilon(1e-6));
  CHECK(report({{1.0}}).l_diff_min == doctest::Approx(0.8 * 0.2).epsilon(1e-6));
  CHECK(report({{0.5}}).l_max == doctest::Approx(lmax(0.65)).epsilon(1e-6));
  // Cameras at f = 0 and f = 1: rho_hi = 0.8 and rho_lo = 0.5.
  const ReflectionReport two = report({{0.0}, {1.0}});
  CHECK(two.l_max == doctest::Approx(lmax(0.8)).epsilon(1e-6));
  CHECK(two.l_diff_min == doctest::Approx(0.5 * 0.2).epsilon(1e-6));
  CHECK(two.bound == doctest::Approx(std::log1p(lmax(0.8) / (0.5 * 0.2))).epsilon(1e-6));
  // An NIR albedo above 1 (the scale form allows it) raises L_max; the cap min(1, (1 - m) rho_lo) is on rho_lo.
  m.nir_form = NirForm::kScale;
  m.nir_params = {2.4, 0.0};  // 2.4 b_R R_lin = 1.2
  CHECK(report({{1.0}}).l_max == doctest::Approx(lmax(1.2)).epsilon(1e-6));
  CHECK(report({{1.0}}).l_diff_min == doctest::Approx(0.2).epsilon(1e-6));
  // The dye print on a textured floor (sRGB 0 and 255 texels, base colour 1): visible 0 / 1, NIR 0.745 + 0.132 R =
  // 0.745 / 0.877. f = 1 ranges over the NIR texels, f = 0.5 over their mixes; with f in {0, 1} the extremes are the
  // visible 0 and the NIR 0.877; without UVs the texture renders as its mean (0.5 visible, 0.811 NIR).
  SceneBundle d = pbr_scene();
  Material& t = d.bodies[0].materials[0];
  t.base_color = {1.0f, 1.0f, 1.0f};
  t.texture = {0, 0, 0, 255, 255, 255};
  t.texture_width = 2;
  t.texture_height = 1;
  t.texture_channels = 3;
  t.nir_form = NirForm::kAffine;
  t.nir_params = {0.745, 0.132};
  const auto dye = [&](const SceneBundle& b, const std::vector<std::vector<double>>& f) {
    return reflection_policy(b, kGrey, {}, f).at(0);
  };
  CHECK(dye(d, {{1.0}}).l_max == doctest::Approx(lmax(0.877)).epsilon(1e-6));
  CHECK(dye(d, {{1.0}}).l_diff_min == doctest::Approx(0.745 * 0.2).epsilon(1e-6));
  CHECK(dye(d, {{0.5}}).l_max == doctest::Approx(lmax(0.5 * (1.0 + 0.877))).epsilon(1e-6));
  CHECK(dye(d, {{0.5}}).l_diff_min == doctest::Approx(0.5 * 0.745 * 0.2).epsilon(1e-6));
  CHECK(dye(d, {{0.0}, {1.0}}).l_max == doctest::Approx(lmax(1.0)).epsilon(1e-6));  // the visible 1 beats the NIR 0.877
  CHECK(dye(d, {{0.0}, {1.0}}).l_diff_min == doctest::Approx(0.0).epsilon(1e-6));   // the visible texel 0
  d.bodies[0].mesh.uvs.clear();
  CHECK(dye(d, {{1.0}}).l_max == doctest::Approx(lmax(0.811)).epsilon(1e-6));
  CHECK(dye(d, {{0.5}}).l_max == doctest::Approx(lmax(0.5 * (0.5 + 0.811))).epsilon(1e-6));
  CHECK_THROWS_WITH_AS(reflection_policy(s, kGrey, {}, {{1.5}}), doctest::Contains("nir_fraction must lie in [0, 1]"),
                       std::invalid_argument);
}

TEST_CASE("reflection policy: every light kind, a fractional metallic, albedo above 1 and colour weights") {
  const double pi = std::numbers::pi;
  // The floor of pbr_scene (rho 0.5, ambient 0.2, grey weights): L_max = rho (E_max / pi + A), L_diff,min =
  // min(1, (1 - m) rho) A. Each light kind alone supplies E_max.
  const auto with_light = [](Light l) {
    SceneBundle s = pbr_scene();
    s.lights[0] = l;
    return s;
  };
  const auto lmax = [&](double e, double rho = 0.5, double a = 0.2) { return rho * (e / pi + a); };
  // Directional: E = I (a plane normal to the beam, no falloff).
  Light dir;
  dir.kind = BundleLightKind::kDirectional;
  dir.intensity = 3.0f;
  dir.direction = {0.0, 0.0, -1.0};
  CHECK(reflection_policy(with_light(dir), kGrey).at(0).l_max == doctest::Approx(lmax(3.0)).epsilon(1e-6));
  // Spot: E <= I / d^2 at the nearest geometry (d = 2 straight below), the cone aside.
  Light spot;
  spot.kind = BundleLightKind::kSpot;
  spot.intensity = 4.0f;
  spot.position = {0.0, 0.0, 2.0};
  spot.direction = {0.0, 0.0, -1.0};
  spot.spot_inner_rad = 0.2f;
  spot.spot_outer_rad = 0.4f;
  CHECK(reflection_policy(with_light(spot), kGrey).at(0).l_max == doctest::Approx(lmax(1.0)).epsilon(1e-6));
  // Metallic 0.5 (roughness 0.6, so not refused): L_diff,min = min(1, 0.5 * 0.5) * 0.2, not (1 - m^2) rho A.
  SceneBundle half = pbr_scene();
  half.bodies[0].materials[0].metallic = 0.5f;
  half.bodies[0].materials[0].roughness = 0.6f;
  const ReflectionReport rh = reflection_policy(half, kGrey).at(0);
  CHECK_FALSE(rh.refused);
  CHECK(rh.l_diff_min == doctest::Approx(0.25 * 0.2).epsilon(1e-6));
  CHECK(rh.bound == doctest::Approx(std::log1p(lmax(1.0) / (0.25 * 0.2))).epsilon(1e-6));
  // Albedo above 1: the cap min(1, .) on the rendered radiance A [(1 - m)(1 - E_s) rho + E_s], so L_diff,min = A.
  SceneBundle bright = pbr_scene();
  bright.bodies[0].materials[0].base_color = {3.0f, 3.0f, 3.0f};
  const ReflectionReport rb = reflection_policy(bright, kGrey).at(0);
  CHECK(rb.l_diff_min == doctest::Approx(0.2).epsilon(1e-6));
  CHECK(rb.l_max == doctest::Approx(lmax(1.0, 3.0)).epsilon(1e-6));
  // Colour weights w = (0.4, 0.6, 1.0) (sum 2): albedo is w . base_color = 0.4 + 0.3 + 0.25 = 0.95; a light is
  // I w . colour / sum(w): the point light (0.5, 1, 0) of intensity 4 gives 4 * 0.8 / 2 = 1.6 (E_max = 1.6 / 4), a
  // white ambient 0.2 gives 0.2 * 2 / 2.
  SceneBundle col = pbr_scene();
  col.bodies[0].materials[0].base_color = {1.0f, 0.5f, 0.25f};
  col.lights[0].color = {0.5f, 1.0f, 0.0f};
  const ReflectionReport rc = reflection_policy(col, {0.4, 0.6, 1.0}).at(0);
  CHECK(rc.l_max == doctest::Approx(0.95 * (0.4 / pi + 0.2)).epsilon(1e-6));
  CHECK(rc.l_diff_min == doctest::Approx(0.95 * 0.2).epsilon(1e-6));
}

TEST_CASE("reflection policy: the nearest point of a triangle by region (face, edge, vertex)") {
  const double pi = std::numbers::pi;
  // One triangle (0,0,0) (1,0,0) (0,1,0) lit by a point light of intensity 4 at p: E_max = 4 / |p - nearest|^2.
  struct Case {
    const char* region;
    std::array<double, 3> p;
    double d2;
  };
  const Case cases[] = {{"face", {0.25, 0.25, 1.0}, 1.0},   {"edge ab", {0.5, -1.0, 1.0}, 2.0},
                        {"edge ac", {-1.0, 0.5, 1.0}, 2.0}, {"edge bc (hypotenuse)", {2.0, 2.0, 1.0}, 5.5},
                        {"vertex a", {-1.0, -1.0, 1.0}, 3.0}, {"vertex b", {2.0, -1.0, 1.0}, 3.0},
                        {"vertex c", {-1.0, 2.0, 1.0}, 3.0}};
  for (const Case& c : cases) {
    SceneBundle s = pbr_scene();
    s.bodies[0].mesh = Mesh{};
    s.bodies[0].mesh.positions = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    s.bodies[0].mesh.indices = {0, 1, 2};
    s.lights[0].position = c.p;
    INFO(c.region);
    CHECK(reflection_policy(s, kGrey).at(0).l_max == doctest::Approx(0.5 * (4.0 / c.d2 / pi + 0.2)).epsilon(1e-6));
  }
}

TEST_CASE("reflection policy: a light group's waveform peak raises the bound, its trough lowers the denominator") {
  const double pi = std::numbers::pi;
  // Two groups: the point light (E = 4 / 2^2 = 1) and the ambient 0.2 on the floor of rho 0.5.
  const auto scene = [](int point_group, int ambient_group) {
    SceneBundle s = pbr_scene();
    s.light_groups = {"default", "pulse"};
    s.lights[0].group = point_group;
    s.lights[1].group = ambient_group;
    return s;
  };
  const auto run = [&](const SceneBundle& s, std::vector<GroupRange> g) { return reflection_policy(s, kGrey, g).at(0); };
  // The lamp in a pulse group peaking at 10: E_max = 10, the ambient (constant) unchanged.
  const ReflectionReport peak = run(scene(1, 0), {{1.0, 1.0}, {0.2, 10.0}});
  CHECK(peak.l_max == doctest::Approx(0.5 * (10.0 / pi + 0.2)).epsilon(1e-6));
  CHECK(peak.l_diff_min == doctest::Approx(0.1).epsilon(1e-6));
  CHECK(peak.bound == doctest::Approx(std::log1p(peak.l_max / 0.1)).epsilon(1e-6));
  // The ambient in the pulse group (range [0.5, 2]): L_max sees 0.4, L_diff,min only 0.5 * 0.1 = 0.05.
  const ReflectionReport amb = run(scene(0, 1), {{1.0, 1.0}, {0.5, 2.0}});
  CHECK(amb.l_max == doctest::Approx(0.5 * (1.0 / pi + 0.4)).epsilon(1e-6));
  CHECK(amb.l_diff_min == doctest::Approx(0.5 * 0.5 * 0.2).epsilon(1e-6));
  // An off phase at 0 leaves no ambient to compare with: unbounded.
  const ReflectionReport off = run(scene(0, 1), {{1.0, 1.0}, {0.0, 2.0}});
  CHECK(off.l_diff_min == 0.0);
  CHECK(std::isinf(off.bound));
  // A background of radiance 3 in the pulse group (peak 2) outshines the floor: L_max = 6.
  SceneBundle bg = scene(0, 0);
  bg.lights.push_back(background_light(3.0f, 1));
  CHECK(run(bg, {{1.0, 1.0}, {0.5, 2.0}}).l_max == doctest::Approx(6.0).epsilon(1e-6));
  // A baked body (K = 2, white) splits its radiance over the groups by shares a_g, each following its waveform: its
  // peak is K sum_g a_g m_g,max = 2 (0.5 * 1 + 0.5 * 3) = 4; with no shares all of it is in group 0 (peak 2 * 1).
  SceneBundle baked = scene(0, 0);
  Body bb = make_body(2, box_mesh(0.1f, 0.1f, 0.1f));
  bb.T_world_body = translation(3.0, 0.0, 0.5);
  bb.materials[0].shading = Shading::kBaked;
  bb.materials[0].baked_scale = 2.0f;
  baked.bodies.push_back(bb);
  CHECK(run(baked, {{1.0, 1.0}, {0.5, 3.0}}).l_max == doctest::Approx(2.0).epsilon(1e-6));
  baked.bodies[1].materials[0].baked_groups = {0.5f, 0.5f};
  CHECK(run(baked, {{1.0, 1.0}, {0.5, 3.0}}).l_max == doctest::Approx(4.0).epsilon(1e-6));
  CHECK(run(baked, {{2.0, 2.0}, {0.5, 3.0}}).l_max == doctest::Approx(2.0 * (0.5 * 2.0 + 0.5 * 3.0)).epsilon(1e-6));
  // No ranges: every group constant, as before. A wrong count or an invalid range is refused.
  CHECK(reflection_policy(scene(1, 0), kGrey).at(0).l_max == doctest::Approx(0.5 * (1.0 / pi + 0.2)).epsilon(1e-6));
  CHECK_THROWS_AS(reflection_policy(scene(1, 0), kGrey, std::vector<GroupRange>{{1.0, 1.0}}), std::invalid_argument);
  CHECK_THROWS_AS(run(scene(1, 0), {{1.0, 1.0}, {2.0, 1.0}}), std::invalid_argument);
  CHECK_THROWS_AS(run(scene(1, 0), {{1.0, 1.0}, {-0.1, 1.0}}), std::invalid_argument);
}

TEST_CASE("reflection policy: a moving body or light is bounded by its control positions' ball") {
  const double pi = std::numbers::pi;
  // A white box (half 0.1) moving along x at z = 1.5 under the light at z = 2, passing x = 0 at t = 0.05 s: the bound
  // must reach at least the static box's at that closest approach (E = 4 / 0.4^2).
  SceneBundle s = pbr_scene();
  s.trajectories.push_back(
      moving_trajectory(kT0, kT1, Quat<double>{1, 0, 0, 0}, {-0.05, 0.0, 1.5}, {1.0, 0.0, 0.0}, {0, 0, 0}));
  Body box = make_body(2, box_mesh(0.1f, 0.1f, 0.1f));
  box.trajectory = 1;
  box.T_world_body = translation(0.0, 0.0, 0.0);
  s.bodies.push_back(box);
  const double closest = 1.0 * (4.0 / (0.4 * 0.4) / pi + 0.2);
  CHECK(reflection_policy(s, kGrey).at(0).l_max >= closest);
  // A light carried by the moving box: its ball is the box's widened by the offset, so it may touch the box: +inf.
  s.lights[0].body = 1;
  s.lights[0].position = {0.0, 0.0, 0.3};
  CHECK(std::isinf(reflection_policy(s, kGrey).at(0).l_max));
  CHECK(std::isinf(reflection_policy(s, kGrey).at(0).bound));
}

}  // namespace balla::render::test
