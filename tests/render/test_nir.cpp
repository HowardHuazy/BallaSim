// P2-F F11a (plan p2f §4.9): the near-infrared albedo. The forms at level 0 (band_images); SceneData's three modes (every
// f_NIR zero: Phase 1's textures bit for bit; every f equal: the mix folded in at level 0; f differing between (camera,
// light group) pairs: two-channel textures whose visible channel is Phase 1's); the two-channel EWA lookup; and the Embree
// renderer's per-(camera, group) mix against its closed form.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "render/renderer.h"
#include "render/scene_data.h"
#include "render/shade/ewa.h"
#include "render/shade/nir.h"
#include "render/texture.h"
#include "scenes.h"

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;

namespace {

constexpr int64_t kT0 = 1'000'000'000, kT1 = 1'250'000'000;

double lin(uint8_t c) { return srgb_to_linear(c); }
double dbl(float x) { return static_cast<double>(x); }

// A 2 x 1 RGB texture (red, then a dark teal) on a material with base colour (0.5, 0.25, 1).
Material tex_material() {
  Material m;
  m.base_color = {0.5f, 0.25f, 1.0f};
  m.texture = {255, 128, 0, 64, 160, 192};
  m.texture_width = 2;
  m.texture_height = 1;
  m.texture_channels = 3;
  return m;
}

std::array<double, 3> weighted(const Material& m, std::array<double, 3> w) {
  return {w[0] * dbl(m.base_color[0]), w[1] * dbl(m.base_color[1]), w[2] * dbl(m.base_color[2])};
}

// A textured 4 x 4 plane (colour texture, the NIR form `form`), seen by `cameras` pinhole cameras at the same pose, lit by
// an ambient light in each of two groups.
SceneBundle plane_scene(Material m, int cameras = 1) {
  SceneBundle s;
  s.light_groups = {"default", "mains"};
  const Quat<double> id = quat_identity<double>();
  s.trajectories.push_back(static_trajectory(kT0, kT1, id, {0, 0, 0}));
  for (int c = 0; c < cameras; ++c) {
    Camera cam = pinhole(16, 12, 20.0, 0);
    cam.name = "cam" + std::to_string(c);
    s.cameras.push_back(cam);
  }
  Body plane = make_body(1, plane_mesh(2.0, 2));
  plane.materials[0] = std::move(m);
  plane.T_world_body = translation(0, 0, 1.0);
  s.bodies.push_back(std::move(plane));
  s.lights = {ambient_light(1.0f, 0), ambient_light(0.5f, 1)};
  return s;
}

Material checker_material(NirForm form, std::vector<double> params) {
  Material m;
  m.base_color = {0.9f, 0.6f, 0.3f};
  std::mt19937 rng(11);
  m.texture.resize(8 * 8 * 3);
  for (uint8_t& c : m.texture) c = static_cast<uint8_t>(30 + rng() % 200);
  m.texture_width = m.texture_height = 8;
  m.texture_channels = 3;
  m.nir_form = form;
  m.nir_params = std::move(params);
  return m;
}

RendererOptions embree_options(std::vector<std::vector<double>> nir) {
  RendererOptions o = options(BackendKind::kEmbree);
  o.scene.n_subsamples = 16;
  o.nir_fraction = std::move(nir);
  return o;
}

}  // namespace

TEST_CASE("band_images: the NIR forms at level 0") {
  const std::array<double, 3> w{0.52, 0.37, 0.11};
  Material m = tex_material();
  const std::array<double, 3> wb = weighted(m, w);
  const double r[2] = {dbl(m.base_color[0]) * lin(255), dbl(m.base_color[0]) * lin(64)};  // b_R R_lin
  const double vis[2] = {(wb[0] * lin(255) + wb[1] * lin(128)) + wb[2] * lin(0),
                         (wb[0] * lin(64) + wb[1] * lin(160)) + wb[2] * lin(192)};
  {
    const BandImages im = band_images(m, wb);  // assumed: the red channel
    REQUIRE((im.width == 2 && im.height == 1));
    for (int i = 0; i < 2; ++i) {
      CHECK(im.vis[i] == vis[i]);
      CHECK(im.nir[i] == r[i]);
    }
    CHECK(im.vis_const == (wb[0] + wb[1]) + wb[2]);
    CHECK(im.nir_const == doctest::Approx(m.base_color[0]).epsilon(1e-15));
  }
  m.nir_form = NirForm::kScale;
  m.nir_params = {1.5, 0.0};
  {
    const BandImages im = band_images(m, wb);
    for (int i = 0; i < 2; ++i) CHECK(im.nir[i] == doctest::Approx(1.5 * r[i]).epsilon(1e-15));
    CHECK(im.nir_const == doctest::Approx(1.5 * dbl(m.base_color[0])).epsilon(1e-15));
  }
  m.nir_form = NirForm::kAffine;  // the ledger's dye print
  m.nir_params = {0.745, 0.132};
  {
    const BandImages im = band_images(m, wb);
    for (int i = 0; i < 2; ++i) CHECK(im.nir[i] == doctest::Approx(0.745 + 0.132 * r[i]).epsilon(1e-15));
  }
  m.nir_params = {0.9, 0.5};  // clamped into [0, 1] at level 0
  CHECK(band_images(m, wb).nir[0] == 1.0);
  m.nir_params = {-0.5, 0.2};
  CHECK(band_images(m, wb).nir[1] == 0.0);
  m.nir_form = NirForm::kScalar;
  m.nir_params = {0.3, 0.0};
  {
    const BandImages im = band_images(m, wb);
    CHECK((im.nir[0] == 0.3 && im.nir[1] == 0.3 && im.nir_const == 0.3));
  }
  // The form texture: a 4 x 2 linear image on a 2 x 1 visible one, on the larger grid (exact for integer ratios).
  m.nir_form = NirForm::kTexture;
  m.nir_params.clear();
  m.nir_texture = {0, 51, 102, 153, 204, 255, 51, 0};
  m.nir_texture_width = 4;
  m.nir_texture_height = 2;
  {
    const BandImages im = band_images(m, wb);
    REQUIRE((im.width == 4 && im.height == 2));
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 4; ++i) {
        CHECK(im.nir[static_cast<size_t>(j * 4 + i)] == doctest::Approx(m.nir_texture[static_cast<size_t>(j * 4 + i)] / 255.0));
        CHECK(im.vis[static_cast<size_t>(j * 4 + i)] == doctest::Approx(vis[i / 2]).epsilon(1e-14));  // replicated
      }
    }
  }
  // A material with no visible texture and a NIR one: constant visible image on the NIR grid.
  m.texture.clear();
  m.texture_width = m.texture_height = m.texture_channels = 0;
  {
    const BandImages im = band_images(m, wb);
    REQUIRE((im.width == 4 && im.height == 2));
    CHECK(im.vis[5] == im.vis_const);
  }
  // Untextured: the constants alone.
  Material plain;
  plain.base_color = {0.8f, 0.4f, 0.2f};
  plain.nir_form = NirForm::kScale;
  plain.nir_params = {0.5, 0.0};
  const BandImages im = band_images(plain, {w[0] * dbl(0.8f), w[1] * dbl(0.4f), w[2] * dbl(0.2f)});
  CHECK((im.width == 0 && im.vis.empty() && im.nir.empty()));
  CHECK(im.nir_const == doctest::Approx(0.5 * dbl(0.8f)).epsilon(1e-7));
}

TEST_CASE("two-channel EWA returns each channel as the mono lookup of its own image, bit for bit") {
  std::mt19937 rng(21);
  std::uniform_real_distribution<double> u(0.0, 1.0), pos(-2.0, 3.0), logs(-4.0, 9.0), ang(0.0, 3.14159);
  for (auto [w, h] : {std::pair{64, 32}, std::pair{37, 23}}) {
    std::vector<double> a(static_cast<size_t>(w) * h), b(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = u(rng);
      b[i] = 2.0 * u(rng);
    }
    std::vector<float> ta, tb, t2;
    const auto la = build_mip_pyramid(a, w, h, ta);
    const auto lb = build_mip_pyramid(b, w, h, tb);
    const auto l2 = build_mip_pyramid2(a, b, w, h, t2);
    REQUIRE(la.size() == l2.size());
    for (size_t l = 0; l < la.size(); ++l) {  // the interleaved levels hold the two mono ones
      REQUIRE((la[l].width == l2[l].width && la[l].height == l2[l].height));
      const size_t n = static_cast<size_t>(la[l].width) * la[l].height;
      for (size_t i = 0; i < n; ++i) {
        REQUIRE(t2[l2[l].offset + 2 * i] == ta[la[l].offset + i]);
        REQUIRE(t2[l2[l].offset + 2 * i + 1] == tb[lb[l].offset + i]);
      }
    }
    const TextureView va{la.data(), static_cast<int32_t>(la.size()), 1}, vb{lb.data(), static_cast<int32_t>(lb.size()), 1},
        v2{l2.data(), static_cast<int32_t>(l2.size()), 2};
    for (int k = 0; k < 600; ++k) {
      const double s1 = std::exp2(logs(rng)), s2 = std::exp2(logs(rng)), th = ang(rng);
      const double c = std::cos(th), s = std::sin(th);
      // Footprints from sub-texel to far past the variance cap, uv units.
      const Sym2f cov{static_cast<float>((s1 * c * c + s2 * s * s) / (w * w)),
                      static_cast<float>((s1 - s2) * c * s / (w * h)), static_cast<float>((s1 * s * s + s2 * c * c) / (h * h))};
      const Vec2<float> uv{static_cast<float>(pos(rng)), static_cast<float>(pos(rng))};
      float got[2];
      ewa_lookup_n<2>(v2, t2.data(), uv, cov, got);
      CHECK(got[0] == ewa_lookup(va, ta.data(), uv, cov));
      CHECK(got[1] == ewa_lookup(vb, tb.data(), uv, cov));
    }
  }
}

TEST_CASE("SceneData: f = 0 is Phase 1's textures, an equal f folds, a differing f is two-channel") {
  const SceneSettings settings = options(BackendKind::kEmbree).scene;
  const std::array<double, 3>& w = settings.sensor_rgb_weights;
  const Material m = checker_material(NirForm::kAffine, {0.745, 0.132});
  const SceneBundle s = plane_scene(m);
  const SceneData base(s, settings);
  REQUIRE(base.texture_levels().size() == 1);
  const auto fill = [&](const SceneData& d) { return d.host_view(); };

  // Every f zero, however stated: Phase 1's pyramid, bit for bit, and no NIR in the views.
  for (const auto& nir : {std::vector<std::vector<double>>{}, std::vector<std::vector<double>>{{}},
                          std::vector<std::vector<double>>{{0.0, 0.0}}}) {
    const SceneData d(s, settings, nir);
    CHECK(d.texels() == base.texels());
    CHECK(d.texture_channels() == std::vector<int32_t>{1});
    const SceneView v = fill(d);
    CHECK(v.nir_two_channel == 0);
    CHECK(v.cameras[0].nir_fraction[1] == 0.0f);
    CHECK(d.materials()[0].nir_albedo == 0.0f);
  }

  // Equal f: one channel, the level 0 the mix of the two images (the 8 x 8 texture's mean at the top level).
  const double f = 0.4;
  const SceneData folded(s, settings, {{f, f}});
  CHECK(folded.texture_channels() == std::vector<int32_t>{1});
  CHECK(fill(folded).nir_two_channel == 0);
  CHECK(fill(folded).cameras[0].nir_fraction[0] == 0.0f);  // the f is in the albedos
  const BandImages im = band_images(m, weighted(m, w));
  std::vector<double> mixed(im.vis.size());
  for (size_t i = 0; i < mixed.size(); ++i) mixed[i] = (1.0 - f) * im.vis[i] + f * im.nir[i];
  std::vector<float> expect;
  build_mip_pyramid(mixed, im.width, im.height, expect);
  CHECK(folded.texels() == expect);
  CHECK(folded.texels() != base.texels());

  // Differing f: the visible channel is Phase 1's pyramid bit for bit, the other the NIR one; the cameras carry the f.
  const SceneData two(s, settings, {{0.2, 0.9}});
  CHECK(two.texture_channels() == std::vector<int32_t>{2});
  const SceneView v = fill(two);
  CHECK(v.nir_two_channel == 1);
  CHECK(v.textures[0].channels == 2);
  CHECK((v.cameras[0].nir_fraction[0] == 0.2f && v.cameras[0].nir_fraction[1] == 0.9f));
  std::vector<float> nir_only;
  const auto nl = build_mip_pyramid(im.nir, im.width, im.height, nir_only);
  const auto& levels = two.texture_levels()[0];
  const auto& base_levels = base.texture_levels()[0];
  REQUIRE(levels.size() == base_levels.size());
  for (size_t l = 0; l < levels.size(); ++l) {
    const size_t n = static_cast<size_t>(levels[l].width) * levels[l].height;
    for (size_t i = 0; i < n; ++i) {
      REQUIRE(two.texels()[levels[l].offset + 2 * i] == base.texels()[base_levels[l].offset + i]);
      REQUIRE(two.texels()[levels[l].offset + 2 * i + 1] == nir_only[nl[l].offset + i]);
    }
  }

  // The rows: one per camera at most, empty or one value per group, each in [0, 1].
  CHECK_THROWS_WITH_AS(SceneData(s, settings, {{0.1, 0.2}, {0.3, 0.4}}), doctest::Contains("nir_fraction has 2 cameras"), std::invalid_argument);
  CHECK_THROWS_WITH_AS(SceneData(s, settings, {{0.1}}), doctest::Contains("one value per light group (2)"), std::invalid_argument);
  CHECK_THROWS_WITH_AS(SceneData(s, settings, {{0.1, 1.5}}), doctest::Contains("[0, 1]"), std::invalid_argument);
  CHECK_THROWS_AS(SceneData(s, settings, {{std::nan(""), 0.5}}), std::invalid_argument);
}

TEST_CASE("SceneData: materials without a texture, with a NIR texture only, and textures without UVs") {
  const SceneSettings settings = options(BackendKind::kEmbree).scene;
  const std::array<double, 3>& w = settings.sensor_rgb_weights;
  Material m;
  m.base_color = {0.8f, 0.4f, 0.2f};
  m.nir_form = NirForm::kScale;
  m.nir_params = {0.5, 0.0};
  const SceneBundle s = plane_scene(m);
  const double vis = (w[0] * dbl(0.8f) + w[1] * dbl(0.4f)) + w[2] * dbl(0.2f);
  {
    const SceneData d(s, settings, {{0.0, 1.0}});  // two-channel, no texture: the constants
    CHECK(d.texture_levels().empty());
    CHECK(d.materials()[0].albedo == static_cast<float>(vis));
    CHECK(d.materials()[0].nir_albedo == doctest::Approx(0.5 * dbl(0.8f)).epsilon(1e-7));
  }
  {
    const SceneData d(s, settings, {{0.25, 0.25}});  // folded
    CHECK(d.materials()[0].albedo == doctest::Approx(0.75 * vis + 0.25 * 0.5 * dbl(0.8f)).epsilon(1e-6));
    CHECK(d.materials()[0].nir_albedo == 0.0f);
  }
  // Only a NIR texture: the material is textured by it, its visible channel the constant.
  Material t = m;
  t.nir_form = NirForm::kTexture;
  t.nir_params.clear();
  t.nir_texture = {0, 64, 128, 255};
  t.nir_texture_width = t.nir_texture_height = 2;
  const SceneBundle st = plane_scene(t);
  {
    const SceneData d(st, settings, {{0.0, 1.0}});
    REQUIRE(d.texture_levels().size() == 1);
    CHECK(d.materials()[0].texture == 0);
    const auto& lv = d.texture_levels()[0][0];
    CHECK(d.texels()[lv.offset + 2 * 3] == static_cast<float>(vis));       // texel 3: visible constant
    CHECK(d.texels()[lv.offset + 2 * 3 + 1] == 1.0f);                     // NIR 255 / 255
  }
  // Without UVs the texture cannot be placed: its mean is the albedo, in both bands.
  SceneBundle nouv = st;
  nouv.bodies[0].mesh.uvs.clear();
  {
    const SceneData d(nouv, settings, {{0.0, 1.0}});
    CHECK(d.materials()[0].texture == -1);
    CHECK(d.materials()[0].albedo == static_cast<float>(vis));
    CHECK(d.materials()[0].nir_albedo == doctest::Approx((0 + 64 + 128 + 255) / 4.0 / 255.0).epsilon(1e-6));
  }
}

TEST_CASE("Embree: each camera and light group sees its own mix of the visible and NIR albedo") {
  const std::vector<double> params{0.5, 0.0};
  Material m;
  m.base_color = {1.0f, 1.0f, 1.0f};
  m.texture = std::vector<uint8_t>(4 * 4 * 3, 128);  // a flat grey: the lookups are exact, any footprint
  m.texture_width = m.texture_height = 4;
  m.texture_channels = 3;
  m.nir_form = NirForm::kScalar;
  m.nir_params = params;
  const double rho_vis = lin(128), rho_nir = 0.5;
  const SceneBundle s = plane_scene(m, 2);
  const std::vector<uint16_t> cam{0, 0, 1, 1};
  const std::vector<uint32_t> pixel{6 * 16 + 8, 5 * 16 + 7, 6 * 16 + 8, 5 * 16 + 7};
  const std::vector<int64_t> t(4, kT0 + 100'000'000);
  const std::vector<std::vector<double>> nir{{0.0, 1.0}, {0.75, 0.25}};
  Renderer r(s, embree_options(nir));
  const RenderResult res = r.render(cam, pixel, t);
  const double ambient[2] = {1.0, 0.5};
  for (size_t k = 0; k < 4; ++k) {
    for (size_t g = 0; g < 2; ++g) {
      const double f = nir[cam[k]][g];
      const double expect = ((1.0 - f) * rho_vis + f * rho_nir) * ambient[g];
      INFO("camera " << cam[k] << " group " << g);
      CHECK(res.irradiance[k * 2 + g] == doctest::Approx(expect).epsilon(2e-6));
    }
  }
  // Equal f for every pair: the same closed form from the folded albedo (one channel).
  Renderer folded(s, embree_options({{0.3, 0.3}, {0.3, 0.3}}));
  const RenderResult rf = folded.render(cam, pixel, t);
  for (size_t k = 0; k < 4; ++k) {
    for (size_t g = 0; g < 2; ++g) {
      CHECK(rf.irradiance[k * 2 + g] == doctest::Approx((0.7 * rho_vis + 0.3 * rho_nir) * ambient[g]).epsilon(2e-6));
    }
  }
  // f = 0 is the renderer without the option, bit for bit; the options reach only the albedo.
  Renderer none(s, embree_options({}));
  Renderer zero(s, embree_options({{0.0, 0.0}, {}}));
  const RenderResult r0 = none.render(cam, pixel, t), rz = zero.render(cam, pixel, t);
  CHECK(std::memcmp(r0.irradiance.data(), rz.irradiance.data(), r0.irradiance.size() * sizeof(float)) == 0);
  CHECK(r0.signature == res.signature);
  for (size_t k = 0; k < 4; ++k) CHECK(r0.irradiance[k * 2] == doctest::Approx(rho_vis).epsilon(2e-6));
}

TEST_CASE("nir_mix is exact at f = 0 and f = 1") {
  for (const float v : {0.0f, 0.03125f, 0.7f, 123.5f}) {
    for (const float n : {0.0f, 0.4f, 1.0f, 9.0f}) {
      CHECK(nir_mix(v, n, 0.0f) == v);
      CHECK(nir_mix(v, n, 1.0f) == n);
    }
  }
}

// Batch 5's merge of F6 and F11a: lobe_radiance shades a cluster with one albedo (its Fresnel F0 and diffuse weight), not
// one per light group, so a lobe in a scene whose f differ between (camera, group) pairs would ignore the NIR channel
// silently: the build refuses it. An equal f (folded at level 0) and f = 0 render with the lobe.
TEST_CASE("a specular lobe with two-channel near-infrared albedo is refused; folded and zero f render") {
  Material m = checker_material(NirForm::kScalar, {0.5, 0.0});
  m.specular = 0.5f;
  m.roughness = 0.4f;
  const SceneBundle s = plane_scene(m, 1);
  const std::vector<uint16_t> cam{0};
  const std::vector<uint32_t> pixel{6 * 16 + 8};
  const std::vector<int64_t> t(1, kT0 + 100'000'000);
  Renderer two(s, embree_options({{0.0, 1.0}}));
  CHECK_THROWS_AS(two.render(cam, pixel, t), std::invalid_argument);
  Renderer folded(s, embree_options({{0.3, 0.3}}));
  CHECK_NOTHROW(folded.render(cam, pixel, t));
  Renderer zero(s, embree_options({}));
  CHECK_NOTHROW(zero.render(cam, pixel, t));
}

// Batch 5's merge of F11a with S5a: a tile atlas (texture_tile) is read per cell and mono (ewa.h), and a two-channel texture
// has no tiled lookup, so a tiled lit material in a scene whose f differ between light groups is refused at build. A
// folded (equal f) or zero f keeps the tile: the mono pyramid stops at log2(P) + 1 levels and the view carries the pitch.
TEST_CASE("a tile atlas with two-channel near-infrared albedo is refused; folded and zero f keep the tile") {
  Material m = checker_material(NirForm::kScalar, {0.5, 0.0});
  m.texture_tile = 4;  // the 8 x 8 texture holds 2 x 2 cells
  const SceneBundle s = plane_scene(m, 1);
  const std::vector<uint16_t> cam{0};
  const std::vector<uint32_t> pixel{6 * 16 + 8};
  const std::vector<int64_t> t(1, kT0 + 100'000'000);
  Renderer two(s, embree_options({{0.0, 1.0}}));
  CHECK_THROWS_AS(two.render(cam, pixel, t), std::invalid_argument);
  for (const std::vector<std::vector<double>>& f : {std::vector<std::vector<double>>{{0.3, 0.3}}, {}}) {
    Renderer r(s, embree_options(f));
    CHECK_NOTHROW(r.render(cam, pixel, t));
    const SceneData data(s, embree_options(f).scene, f);
    REQUIRE(data.texture_tiles().size() == 1);
    CHECK(data.texture_tiles()[0] == 4);
    CHECK(data.texture_channels()[0] == 1);
    CHECK(data.texture_levels()[0].size() == 3);  // log2(4) + 1
    CHECK(data.host_view().textures[0].tile_pitch == 4);
  }
}
