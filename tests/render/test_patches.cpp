// P2-F F5, shading patches (accuracy.shading_patches; render/patches.h, render/shade/shade.h hit_key; plan p2f §4.4):
// the patch rule on hand-built meshes (seams, splits, shells, flat facets, specular hard edges, welding, non-manifold
// edges, the flat-shaded 32 x 16 UV sphere), the scene's global ids and views, and the renderer under crease against
// material from identical hit records: a crease splits clusters (kClassMulti) and changes the signature where a
// sub-sample crosses it, a one-face pixel-sample keeps material's value bit for bit, a two-face one equals the
// per-face sum, and held sets equal their static bundles. Host = device under crease is test_device.cpp's (GPU).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numbers>
#include <set>
#include <vector>

#include "core/math/quat.h"
#include "core/math/so3.h"
#include "core/traj/spline_build.h"
#include "render/embree/embree_tracer.h"
#include "render/executor.h"
#include "render/patches.h"
#include "render/pose_table.h"
#include "render/renderer.h"
#include "render/scene_data.h"
#include "render/shade/shade.h"
#include "render/thread_pool.h"
#include "scenes.h"

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;

namespace {

constexpr int64_t kT0 = 1'000'000'000, kT1 = 1'250'000'000;
constexpr double kDeg = std::numbers::pi / 180;

// A trajectory's pose at t as the renderer evaluates it (test_renderer.cpp's eval_pose), as a T_world_body matrix.
std::array<double, 16> pose_at(const Trajectory& tr, int64_t t) {
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
  const Mat3<double> r = to_matrix(k.q);
  return {r.m[0][0], r.m[0][1], r.m[0][2], k.p.x, r.m[1][0], r.m[1][1], r.m[1][2], k.p.y,
          r.m[2][0], r.m[2][1], r.m[2][2], k.p.z, 0, 0, 0, 1};
}

// Every triangle its own three vertices with its winding normal (flat shading: hard edges wherever faces turn).
Mesh unweld(const Mesh& m) {
  Mesh out;
  for (size_t t = 0; t < m.indices.size() / 3; ++t) {
    float p[3][3];
    for (int c = 0; c < 3; ++c) {
      for (int a = 0; a < 3; ++a) p[c][a] = m.positions[3 * m.indices[3 * t + c] + a];
    }
    const double e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
    const double e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
    double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int c = 0; c < 3; ++c) {
      out.positions.insert(out.positions.end(), {p[c][0], p[c][1], p[c][2]});
      out.normals.insert(out.normals.end(), {static_cast<float>(n[0] / len), static_cast<float>(n[1] / len),
                                             static_cast<float>(n[2] / len)});
      out.indices.push_back(static_cast<uint32_t>(3 * t + c));
    }
  }
  if (!m.material_ids.empty()) out.material_ids = m.material_ids;
  return out;
}

// A UV sphere of n_lon x n_lat (rings at latitude steps of 180 / n_lat degrees), radial vertex normals, outward
// winding: smooth-shaded; unweld() makes it flat-shaded (quads planar: one normal per quad).
Mesh uv_sphere(float r, int n_lon, int n_lat) {
  Mesh m;
  for (int i = 1; i < n_lat; ++i) {
    for (int j = 0; j < n_lon; ++j) {
      const double th = std::numbers::pi * i / n_lat, ph = 2 * std::numbers::pi * j / n_lon;
      const double d[3] = {std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph)};
      for (int a = 0; a < 3; ++a) m.positions.push_back(static_cast<float>(static_cast<double>(r) * d[a]));
      for (int a = 0; a < 3; ++a) m.normals.push_back(static_cast<float>(d[a]));
    }
  }
  const auto top = static_cast<uint32_t>(m.positions.size() / 3), bot = top + 1;
  m.positions.insert(m.positions.end(), {0.0f, r, 0.0f, 0.0f, -r, 0.0f});
  m.normals.insert(m.normals.end(), {0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f});
  const auto at = [&](int i, int j) { return static_cast<uint32_t>(i * n_lon + (j % n_lon)); };
  for (int i = 0; i + 1 < n_lat - 1; ++i) {
    for (int j = 0; j < n_lon; ++j) {
      m.indices.insert(m.indices.end(), {at(i, j), at(i + 1, j + 1), at(i, j + 1), at(i, j), at(i + 1, j),
                                         at(i + 1, j + 1)});
    }
  }
  for (int j = 0; j < n_lon; ++j) {
    m.indices.insert(m.indices.end(), {top, at(0, j), at(0, j + 1), bot, at(n_lat - 2, j + 1), at(n_lat - 2, j)});
  }
  return m;
}

// Two unit quads sharing the edge x = 0 (along y), the second turned by `angle` about it; flat-shaded (each quad its
// own four vertices and normal) or without normals.
Mesh folded_strip(double angle, bool normals) {
  Mesh m;
  const float c = static_cast<float>(std::cos(angle)), s = static_cast<float>(std::sin(angle));
  m.positions = {-1, 0, 0, 0, 0, 0, 0, 1, 0, -1, 1, 0,   // quad A in z = 0, x <= 0
                 0, 0, 0, c, 0, s, c, 1, s, 0, 1, 0};    // quad B from x = 0, turned
  m.indices = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};
  if (normals) m.normals = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, -s, 0, c, -s, 0, c, -s, 0, c, -s, 0, c};
  return m;
}

Patches patches_of(const Mesh& m, std::vector<uint32_t> mats = {}, std::vector<uint8_t> spec = {}) {
  if (mats.empty()) mats.assign(m.indices.size() / 3, 0u);
  return build_patches({m.positions, m.normals, m.indices}, mats, spec);
}

std::vector<uint32_t> ids(std::initializer_list<uint32_t> v) { return v; }

}  // namespace

TEST_CASE("shading patches (P2-F F5): the rule on hand-built meshes") {
  const std::vector<uint8_t> glossy{1}, matte{0};
  // A box without normals: rule (c), every 90 degree edge cuts, each face's two triangles stay together, numbered by
  // first triangle (box_mesh lists the faces z-, z+, y-, y+, x+, x-).
  Patches p = patches_of(box_mesh(0.1f, 0.2f, 0.3f));
  CHECK(p.count == 6);
  CHECK(p.of_triangle == ids({0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}));
  // Smooth shading (shared vertices, one normal each): no hard edge, so one patch at any angle, glossy or not.
  Mesh smooth = box_mesh(0.1f, 0.2f, 0.3f);
  for (size_t v = 0; v < 8; ++v) {
    const float* x = &smooth.positions[3 * v];
    const float l = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    smooth.normals.insert(smooth.normals.end(), {x[0] / l, x[1] / l, x[2] / l});
  }
  CHECK(patches_of(smooth).count == 1);
  CHECK(patches_of(smooth, {}, glossy).count == 1);
  // Flat shading: hard 90 degree edges cut (rule (b)), the faces' diagonals (equal normals) do not.
  const Mesh flat_box = unweld(box_mesh(0.1f, 0.2f, 0.3f));
  CHECK(patches_of(flat_box).of_triangle == ids({0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}));
  CHECK(patches_of(flat_box, {}, glossy).count == 6);

  // A flat-shaded fold below 30 degrees is one patch on a Lambert material and two on a specular one; above 30, two;
  // without normals, the angle alone decides. Normals within 1e-3 rad of each other declare no hard edge.
  for (const auto& [deg, lambert, specular, bare] : {std::tuple{20.0, 1u, 2u, 1u}, std::tuple{29.0, 1u, 2u, 1u},
                                                     std::tuple{31.0, 2u, 2u, 2u}, std::tuple{90.0, 2u, 2u, 2u}}) {
    INFO("fold " << deg << " deg");
    CHECK(patches_of(folded_strip(deg * kDeg, true), {}, matte).count == lambert);
    CHECK(patches_of(folded_strip(deg * kDeg, true), {}, glossy).count == specular);
    CHECK(patches_of(folded_strip(deg * kDeg, false), {}, glossy).count == bare);
  }
  CHECK(patches_of(folded_strip(5e-4, true), {}, glossy).count == 1);
  CHECK(patches_of(folded_strip(2e-3, true), {}, glossy).count == 2);
  CHECK(patches_of(folded_strip(2e-3, true), {}, glossy).of_triangle == ids({0, 0, 1, 1}));

  // A UV seam: the plane's middle column of vertices duplicated (equal positions and normals) joins; welding is by
  // value, so -0 joins +0.
  Mesh seam = folded_strip(0.0, true);
  CHECK(patches_of(seam, {}, glossy).count == 1);
  seam.positions[3 * 4] = -0.0f;  // quad B's copy of (0, 0, 0)
  CHECK(patches_of(seam, {}, glossy).count == 1);
  // Material changes always cut (rule (a)), whatever the angle and normals, also on the smooth box.
  CHECK(patches_of(seam, {0, 0, 1, 1}).of_triangle == ids({0, 0, 1, 1}));
  CHECK(patches_of(seam, {1, 1, 0, 0}).of_triangle == ids({0, 0, 1, 1}));
  CHECK(patches_of(smooth, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}).count == 2);
  // Triangles 0 and 3 meet at the seam (same material: joined), the diagonals cut, triangles 1 and 2 do not touch.
  CHECK(patches_of(seam, {0, 1, 1, 0}).of_triangle == ids({0, 1, 2, 0}));
  // Disconnected shells are separate patches, also when they touch at one vertex only.
  Mesh shells = folded_strip(0.0, true);
  for (size_t v = 4; v < 8; ++v) shells.positions[3 * v + 1] += 1.0f;  // quad B touches quad A at (0, 1, 0) only
  CHECK(patches_of(shells).count == 2);
  for (size_t v = 4; v < 8; ++v) shells.positions[3 * v + 2] += 5.0f;  // apart
  CHECK(patches_of(shells).count == 2);

  // A non-manifold edge (three fins on the y axis at 0, 10 and 120 degrees, no normals): every pair is tested, so the
  // two fins 10 degrees apart join and the third stands alone.
  Mesh fins;
  for (const double deg : {0.0, 10.0, 120.0}) {
    const auto base = static_cast<uint32_t>(fins.positions.size() / 3);
    fins.positions.insert(fins.positions.end(), {0, 0, 0, 0, 1, 0, static_cast<float>(std::cos(deg * kDeg)), 0,
                                                 static_cast<float>(std::sin(deg * kDeg))});
    fins.indices.insert(fins.indices.end(), {base, base + 1, base + 2});
  }
  CHECK(patches_of(fins).of_triangle == ids({0, 0, 1}));

  // The flat-shaded 32 x 16 UV sphere (11.25 degree facets): one patch for Lambert, one per quad or cap triangle
  // (14 x 32 + 2 x 32 = 512) for a specular material; smooth-shaded, one patch either way.
  const Mesh sphere = uv_sphere(0.2f, 32, 16);
  REQUIRE(sphere.indices.size() / 3 == 960);
  const Mesh faceted = unweld(sphere);
  CHECK(patches_of(faceted, {}, matte).count == 1);
  CHECK(patches_of(faceted, {}, glossy).count == 512);
  CHECK(patches_of(sphere, {}, glossy).count == 1);
  // Numbering by first triangle: each patch's first triangle comes after the previous patch's.
  const Patches g = patches_of(faceted, {}, glossy);
  uint32_t next = 0;
  for (const uint32_t q : g.of_triangle) {
    CHECK(q <= next);
    if (q == next) ++next;
  }
  CHECK_THROWS_AS(build_patches({sphere.positions, sphere.normals, sphere.indices}, std::vector<uint32_t>(3), {}),
                  std::invalid_argument);
}

TEST_CASE("shading patches: global ids, patch materials, views and held tails") {
  // Bodies by id: 1 a static flat-shaded box (6 patches), 2 a static two-material plane (2), 3 a moving box without
  // normals (6). Global ids follow body order; each patch's material is its triangles'.
  SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
  s.trajectories.push_back(moving_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 2}, {0.1, 0, 0}, {0, 1, 0}));
  s.cameras.push_back(pinhole(32, 24, 30.0, 0));
  Body b1 = make_body(1, unweld(box_mesh(0.1f, 0.1f, 0.1f)));
  b1.T_world_body = translation(0.3, 0, 2);
  Body b2 = make_body(2, plane_mesh(1.0, 1));
  b2.mesh.material_ids = {0, 1};
  b2.materials.push_back(b2.materials[0]);
  b2.materials[1].base_color = {0.5f, 0.5f, 0.5f};
  b2.T_world_body = translation(0, 0, 3);
  Body b3 = make_body(3, box_mesh(0.1f, 0.1f, 0.1f));
  b3.trajectory = 1;
  s.bodies = {b3, b1, b2};  // any order: sorted by id
  s.lights.push_back(ambient_light(1.0f));
  SceneSettings off, on;
  on.shading_patches = ShadingPatches::kCrease;
  const SceneData d_off(s, off), d_on(s, on);
  const SceneView v_off = d_off.host_view(), v_on = d_on.host_view();
  CHECK(v_off.patch_material == nullptr);
  for (int32_t g = 0; g < v_off.num_geometries; ++g) CHECK(v_off.geometries[g].patches == nullptr);
  REQUIRE(d_on.patch_material().size() == 14);
  CHECK(d_on.patch_material() == ids({0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3}));
  const GeometryData& world = d_on.geometries()[0];
  CHECK(world.patches == ids({0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7}));  // body 1's box, then body 2's plane
  CHECK(d_on.geometries()[1].patches == ids({8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13}));
  REQUIRE(v_on.patch_material != nullptr);
  for (int32_t g = 0; g < v_on.num_geometries; ++g) {
    const GeometryView& gv = v_on.geometries[g];
    REQUIRE(gv.patches != nullptr);
    for (uint32_t t = 0; t < gv.num_triangles; ++t) {
      CHECK(key_material(v_on, hit_key(gv, t)) == hit_material(gv, t));
      CHECK(hit_key(v_off.geometries[g], t) == hit_material(gv, t));  // material: the key is the material
      CHECK(key_material(v_off, hit_key(v_off.geometries[g], t)) == hit_material(gv, t));
    }
  }
  // A held mover's triangles follow the world's with their own global ids.
  const RigidPose pose{};
  const std::vector<int32_t> held{1};
  const auto t_on = d_on.trace_set(held, std::span<const RigidPose>(&pose, 1));
  CHECK(t_on->tail.has_patches);
  CHECK(t_on->tail.patches == d_on.geometries()[1].patches);
  const auto t_off = d_off.trace_set(held, std::span<const RigidPose>(&pose, 1));
  CHECK_FALSE(t_off->tail.has_patches);
  CHECK(t_off->tail.patches.empty());
}

TEST_CASE("shading patches: crease splits a box's crease into two clusters, a localized signature change") {
  // A flat-shaded box turned 45 degrees about y, its front vertical edge (the crease) near the image's centre column,
  // translating along x; two shadowless directional lights in their own groups: group 0 with the ambient lights both
  // visible faces 2:1, group 1 lights one face only. Traced once at two times 0.4 px apart, shaded under material and
  // crease from the same hits.
  constexpr uint32_t kW = 40, kH = 30, kN = 64, kPixels = kW * kH;
  constexpr double kRho = 0.6, kAmbient = 0.1;
  SceneBundle s;
  s.light_groups = {"default", "side"};
  const double v = 0.4 / 32.0 * 1.6 / 0.01;  // 0.4 px at the crease's depth (1.6 m) per 10 ms
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
  s.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{0, std::numbers::pi / 4, 0}),
                                             {0.013, 0.0, 1.6 + 0.3 * std::numbers::sqrt2}, {v, 0, 0}, {0, 0, 0}));
  s.cameras.push_back(pinhole(kW, kH, 32.0, 0));
  Body box = make_body(5, unweld(box_mesh(0.3f, 0.65f, 0.3f)),
                       {static_cast<float>(kRho), static_cast<float>(kRho), static_cast<float>(kRho)});
  box.trajectory = 1;
  s.bodies.push_back(box);
  const double tan_a = 1.0 / 3.0, ca = 1 / std::sqrt(1 + tan_a * tan_a), sa = tan_a * ca;
  Light a;  // light direction l = -d = (sin a, 0, -cos a): the faces' n . l = (cos a +- sin a) / sqrt 2, 2:1
  a.kind = BundleLightKind::kDirectional;
  a.direction = {-sa, 0.0, ca};
  a.intensity = 1.0f;
  a.casts_shadows = false;
  Light b = a;  // l = (sin 60, 0, -cos 60): the left face turned away
  b.group = 1;
  b.direction = {-std::sin(60 * kDeg), 0.0, std::cos(60 * kDeg)};
  s.lights = {a, b, ambient_light(static_cast<float>(kAmbient))};
  RendererOptions o = options(BackendKind::kEmbree);
  ThreadPool pool(4);
  SceneSettings mat_settings = o.scene, crease_settings = o.scene;
  mat_settings.n_subsamples = crease_settings.n_subsamples = kN;
  crease_settings.shading_patches = ShadingPatches::kCrease;
  const SceneData d_mat(s, mat_settings), d_crease(s, crease_settings);
  HostExecutor h_mat(pool), h_crease(pool);
  h_mat.upload_scene(d_mat);
  h_crease.upload_scene(d_crease);
  EmbreeTracer tracer(pool);
  tracer.build(d_mat, h_mat.scene_view());
  PoseTable table(static_cast<int32_t>(d_mat.geometries().size()), static_cast<int32_t>(d_mat.lights().size()), 64);
  table.reset(kT0);
  const int64_t times[2] = {kT0 + 100'000'000, kT0 + 110'000'000};
  uint32_t slots[2];
  for (int i = 0; i < 2; ++i) REQUIRE(table.slot(0, times[i], &slots[i]));
  const PoseTableView poses = h_mat.sync_poses(table, 0);
  REQUIRE(h_crease.sync_poses(table, 0).rows == poses.rows);
  // Each face's patch and radiance per group (the box's patches: z-, z+, y-, y+, x+, x-; z- faces left, x+ right).
  const GeometryData& geo = d_crease.geometries()[1];
  const double n_left[3] = {-std::numbers::sqrt2 / 2, 0, -std::numbers::sqrt2 / 2};
  const double n_right[3] = {std::numbers::sqrt2 / 2, 0, -std::numbers::sqrt2 / 2};
  const auto lambert = [&](const double* n, const Light& l) {
    return std::max(0.0, -(n[0] * l.direction[0] + n[1] * l.direction[1] + n[2] * l.direction[2]));
  };
  const auto radiance = [&](uint32_t patch, int group) {
    const double* n = patch == 0 ? n_left : n_right;
    REQUIRE((patch == 0 || patch == 4));
    return group == 0 ? kRho * (kAmbient + lambert(n, a) / std::numbers::pi) : kRho * lambert(n, b) / std::numbers::pi;
  };
  CHECK(lambert(n_right, a) / lambert(n_left, a) == doctest::Approx(2.0));
  CHECK(radiance(0, 1) == 0.0);

  struct Shot {
    std::vector<HitRecord> hits;
    std::vector<float> irr[2];
    std::vector<uint64_t> sig[2];
    std::vector<uint32_t> klass[2];
  } shot[2];
  for (int i = 0; i < 2; ++i) {
    std::vector<SampleRequest> req;
    for (uint32_t p = 0; p < kPixels; ++p) req.push_back({0, 0, p, 0, kN, 0, slots[i]});
    const TraceBatch batch{req.data(), kPixels, kN, poses, nullptr, 0};
    shot[i].hits.resize(static_cast<size_t>(kPixels) * kN);
    tracer.sample(batch, shot[i].hits.data(), nullptr);
    for (int m = 0; m < 2; ++m) {
      shot[i].irr[m].resize(2 * kPixels);
      shot[i].sig[m].resize(kPixels);
      shot[i].klass[m].resize(kPixels);
      (m == 0 ? h_mat : h_crease)
          .shade_into(0, batch, shot[i].hits.data(), nullptr, shot[i].irr[m].data(), shot[i].sig[m].data(), nullptr,
                      nullptr, shot[i].klass[m].data());
    }
  }
  // Per pixel-sample: all sub-samples on the box, and which faces they hit.
  const auto faces = [&](const Shot& sh, uint32_t r, uint32_t* left, uint32_t* right) {
    *left = *right = 0;
    for (uint32_t k = 0; k < kN; ++k) {
      const HitRecord& h = sh.hits[static_cast<size_t>(r) * kN + k];
      if (h.instance != 5) return false;
      const uint32_t patch = geo.patches[h.prim];
      REQUIRE((patch == 0 || patch == 4));
      (patch == 0 ? *left : *right) += 1;
    }
    return true;
  };
  size_t one_face = 0, two_faces = 0, below = 0, crossed = 0;
  for (uint32_t r = 0; r < kPixels; ++r) {
    uint32_t left = 0, right = 0;
    if (!faces(shot[0], r, &left, &right)) continue;
    const float* im = &shot[0].irr[0][2 * r];
    const float* ic = &shot[0].irr[1][2 * r];
    CHECK((shot[0].klass[0][r] & kClassMulti) == 0);  // material: one cluster
    if (left == 0 || right == 0) {
      ++one_face;
      CHECK((shot[0].klass[1][r] & kClassMulti) == 0);
      CHECK(std::memcmp(im, ic, 2 * sizeof(float)) == 0);  // the same single cluster: bit for bit
      continue;
    }
    ++two_faces;
    CHECK((shot[0].klass[1][r] & kClassMulti) != 0);  // crease: two clusters
    for (int g = 0; g < 2; ++g) {
      const double expected = (left * radiance(0, g) + right * radiance(4, g)) / kN;
      CHECK(static_cast<double>(ic[g]) == doctest::Approx(expected).epsilon(2e-6).scale(0));
    }
    // Both faces lit (group 0): Lambert is linear in n, so material's mean normal gives the same value; group 1's
    // left face is turned away, where material's clamp of the mean normal falls short of the faces' sum.
    CHECK(static_cast<double>(im[0]) == doctest::Approx(static_cast<double>(ic[0])).epsilon(2e-6).scale(0));
    below += static_cast<double>(im[1]) < static_cast<double>(ic[1]) * (1 - 1e-3);
  }
  CHECK(one_face > 150);
  CHECK(two_faces > 20);
  CHECK(below > two_faces / 2);
  // Over 0.4 px of motion: material's signature does not see the crease; crease's changes exactly where a sub-sample
  // moved to the other face (the box everywhere at both times).
  for (uint32_t r = 0; r < kPixels; ++r) {
    uint32_t l0, r0, l1, r1;
    if (!faces(shot[0], r, &l0, &r0) || !faces(shot[1], r, &l1, &r1)) continue;
    CHECK(shot[0].sig[0][r] == shot[1].sig[0][r]);
    bool moved = false;
    for (uint32_t k = 0; k < kN; ++k) {
      const size_t i = static_cast<size_t>(r) * kN + k;
      moved = moved || geo.patches[shot[0].hits[i].prim] != geo.patches[shot[1].hits[i].prim];
    }
    CHECK((shot[0].sig[1][r] != shot[1].sig[1][r]) == moved);
    crossed += moved;
  }
  CHECK(crossed > 10);
}

TEST_CASE("shading patches: held sets under crease equal their static bundles (ENGINE_CPP I3b)") {
  // A flat-shaded box mover stands still; holding it merges its patches into the world's tail, which must render as
  // the bundle with it static (the same global ids: bodies keep their order) on every backend.
  SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0, 0, 0}));
  s.trajectories.push_back(static_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.3, 0.7, 0.1}), {0.1, 0.0, 1.5}));
  s.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.0, 0.4, 0.0}), {-0.3, 0.1, 1.8},
                                             {0.2, 0, 0}, {0.5, 1.0, 0.0}));
  s.cameras.push_back(pinhole(48, 36, 40.0, 0));
  Body wall = make_body(1, plane_mesh(2.0, 2));
  wall.T_world_body = translation(0, 0, 3);
  Body held = make_body(7, unweld(box_mesh(0.2f, 0.15f, 0.1f)), {0.4f, 0.6f, 0.5f});
  held.trajectory = 1;
  Body mover = make_body(9, unweld(box_mesh(0.1f, 0.1f, 0.1f)), {0.7f, 0.3f, 0.3f});
  mover.trajectory = 2;
  s.bodies = {wall, held, mover};
  s.lights = {point_light({1.0, -1.0, 0.5}, 2.0f), ambient_light(0.1f)};
  SceneBundle st = s;
  st.bodies[1].trajectory.reset();
  st.bodies[1].T_world_body = pose_at(s.trajectories[1], kT0);
  std::vector<uint16_t> cam;
  std::vector<uint32_t> pixel;
  std::vector<int64_t> t;
  for (uint32_t p = 0; p < 48 * 36; ++p) {
    cam.push_back(0);
    pixel.push_back(p);
    t.push_back(kT0 + 20'000'000);
  }
  const RenderFlags flags{true, false, true};
  for (BackendKind backend : backends()) {
    INFO(backend_name(backend));
    RendererOptions o = options(backend);
    o.scene.shading_patches = ShadingPatches::kCrease;
    Renderer r(s, o);
    const RenderResult movers = r.render(cam, pixel, t, flags);
    CHECK(r.hold(std::vector<uint32_t>{7}, kT0));
    const RenderResult merged = r.render(cam, pixel, t, flags);
    const RenderResult stat = Renderer(st, o).render(cam, pixel, t, flags);
    CHECK(merged.irradiance == stat.irradiance);
    CHECK(merged.signature == stat.signature);
    CHECK(merged.sample_class == stat.sample_class);
    CHECK(std::count(merged.sub_instance.begin(), merged.sub_instance.end(), 7u) > 100);
    size_t multi = 0;
    for (const uint32_t c : merged.sample_class) multi += (c & kClassMulti) != 0;
    RendererOptions om = o;
    om.scene.shading_patches = ShadingPatches::kMaterial;
    const RenderResult plain = Renderer(st, om).render(cam, pixel, t, flags);
    size_t multi_plain = 0;
    for (const uint32_t c : plain.sample_class) multi_plain += (c & kClassMulti) != 0;
    CHECK(multi > multi_plain);  // the boxes' creases
    CHECK(r.hold({}, 0));
    const RenderResult back = r.render(cam, pixel, t, flags);
    CHECK(back.irradiance == movers.irradiance);
    CHECK(back.signature == movers.signature);
  }
}
