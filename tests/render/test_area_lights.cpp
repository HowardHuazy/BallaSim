// Area lights, data and rays (P2-F F8a-1, plan p2f §4.6): the fixed point sets Y_l, the sub-sample assignment, the
// pose rows, the shadow rays and masks of rect and disk lights on the Embree tracer, and the plan pass's centre ray.
// The shading of area lights is F8a-2's (the Renderer refuses them until then).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <set>
#include <vector>

#include "core/rng/purpose.h"
#include "core/rng/sobol.h"
#include "render/embree/embree_tracer.h"
#include "render/executor.h"
#include "render/pose_table.h"
#include "render/scene_data.h"
#include "render/shade/lights.h"
#include "render/thread_pool.h"
#include "scenes.h"

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;

namespace {

constexpr int64_t kT0 = 1'000'000'000, kT1 = 1'200'000'000;
constexpr int kW = 64, kH = 48;

struct Emitter {
  BundleLightKind kind = BundleLightKind::kRect;
  std::array<double, 3> centre{0.0, -0.5, 0.8}, normal{0.0, 0.0, 1.0}, axis{1.0, 0.0, 0.0};
  std::array<double, 2> size{0.5, 0.5};
};

// The plate that blocks the emitter above the floor, a mover (a trace geometry of its own): a square of half size
// 0.2 in the plane z = 1.5 about (0, -0.2).
constexpr double kPlateZ = 1.5, kPlateX = 0.0, kPlateY = -0.2, kPlateHalf = 0.2, kFloorZ = 2.0;

// A static camera at the origin looking along +z, the floor (body 1) in the plane z = 2, optionally the plate (body 3),
// ambient light (bundle light 0) and the emitter (bundle light 1, shadow bit 0).
SceneBundle scene(const Emitter& e, bool plate, bool shadows = true) {
  SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {0.0, 0.0, 0.0}));
  s.cameras.push_back(pinhole(kW, kH, 60.0, 0));
  Body floor = make_body(1, plane_mesh(3.0, 4), {0.8f, 0.7f, 0.6f});
  floor.T_world_body = translation(0.0, 0.0, kFloorZ);
  s.bodies.push_back(floor);
  if (plate) {
    s.trajectories.push_back(static_trajectory(kT0, kT1, quat_identity<double>(), {kPlateX, kPlateY, kPlateZ}));
    Body p = make_body(3, plane_mesh(kPlateHalf, 1));
    p.trajectory = 1;
    s.bodies.push_back(p);
  }
  s.lights.push_back(ambient_light(0.1f));
  Light l;
  l.kind = e.kind;
  l.position = e.centre;
  l.direction = e.normal;
  l.axis = e.axis;
  l.size = e.size;
  l.intensity = 2.0f;
  l.casts_shadows = shadows;
  s.lights.push_back(l);
  return s;
}

SceneSettings settings(uint32_t n = 64, uint64_t seed = 0) {
  SceneSettings st;
  st.n_subsamples = n;
  st.seed = seed;
  return st;
}

// Everything a mask test needs on the host: scene data, executor, tracer and the pose rows of one slot.
struct Host {
  Host(const SceneBundle& b, const SceneSettings& st)
      : data(b, st),
        pool(4),
        exec(pool),
        tracer(pool),
        table(static_cast<int32_t>(data.geometries().size()), static_cast<int32_t>(data.lights().size()), 8) {
    exec.upload_scene(data);
    tracer.build(data, exec.scene_view());
    table.reset(kT0);
    REQUIRE(table.slot(0, kT0 + 1'000'000, &slot));
    poses = exec.sync_poses(table, 0);
  }
  // Hits and masks of sub-samples k_begin .. k_begin + n - 1 of every pixel.
  void trace(uint32_t k_begin, uint32_t n) {
    std::vector<SampleRequest> req;
    for (uint32_t p = 0; p < static_cast<uint32_t>(kW * kH); ++p) req.push_back({0, 0, p, k_begin, n, 0, slot});
    hits.assign(req.size() * n, HitRecord{});
    occ.assign(req.size() * n, 0xDEADBEEFu);
    tracer.sample({req.data(), static_cast<uint32_t>(req.size()), n, poses, nullptr, 0}, hits.data(), occ.data());
  }
  SceneData data;
  ThreadPool pool;
  HostExecutor exec;
  EmbreeTracer tracer;
  PoseTable table;
  uint32_t slot = 0;
  PoseTableView poses{};
  std::vector<HitRecord> hits;
  std::vector<uint32_t> occ;
};

Vec3<double> to_d(Vec3f v) {
  return {static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)};
}

double dot_d(Vec3<double> a, Vec3<double> b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

}  // namespace

TEST_CASE("area lights: the views, the sets Y_l (one point per cell) and the lights' pose rows") {
  Emitter rect;
  rect.size = {0.4, 0.2};
  Emitter disk;
  disk.kind = BundleLightKind::kDisk;
  disk.centre = {0.3, -0.2, 0.9};
  disk.size = {0.3, 0.5};
  SceneBundle b = scene(rect, false);
  b.lights.push_back(b.lights.back());  // a second area light: the disk, bundle light 2
  b.lights.back().kind = disk.kind;
  b.lights.back().position = disk.centre;
  b.lights.back().size = disk.size;
  b.lights.insert(b.lights.begin() + 1, point_light({0.1, -0.7, 0.4}, 3.0f));  // bundle light 1: a point light
  const SceneSettings st = settings(64, 0x0123456789abcdefull);
  const SceneData data(b, st);

  // The views: ambient is folded; the point light, the rect and the disk follow in bundle order.
  REQUIRE(data.lights().size() == 3);
  REQUIRE(data.num_area_lights() == 2);
  const LightView &lp = data.lights()[0], &lr = data.lights()[1], &ld = data.lights()[2];
  CHECK(lp.kind == LightKind::kPoint);
  CHECK((lp.half_u == 0.0f && lp.half_v == 0.0f && lp.area_gain == 0.0f && lp.point_offset == 0 && lp.area_index == 0));
  CHECK(lr.kind == LightKind::kRect);
  CHECK(lr.half_u == 0.2f);
  CHECK(lr.half_v == 0.1f);
  CHECK(lr.area_gain == 1.0f);
  CHECK(lr.area_index == 0);
  CHECK(lr.point_offset == 0);
  CHECK(ld.kind == LightKind::kDisk);
  CHECK(ld.half_u == 0.15f);
  CHECK(ld.half_v == 0.25f);
  CHECK(ld.area_gain == doctest::Approx(1.00646).epsilon(1e-4));  // pi / (16 sin(pi / 16))
  CHECK(ld.area_index == 1);
  CHECK(ld.point_offset == kLightPoints);
  CHECK((lr.shadow_bit >= 0 && ld.shadow_bit >= 0 && lr.shadow_bit != ld.shadow_bit));
  REQUIRE(data.light_points().size() == 2u * 2u * kLightPoints);
  const SceneView v = data.host_view();
  CHECK(v.light_points == data.light_points().data());
  CHECK(v.num_area_lights == 2);

  // Y_l: the point in cell c lies in cell c, for the rect and (through the concentric map's inverse square) the disk;
  // a rect's coordinates are exact, a disk's lie in the unit disk, and the disk's points have the uniform disk's moments.
  for (int a = 0; a < 2; ++a) {
    const bool is_disk = a == 1;
    const uint32_t light_index = a == 0 ? 2u : 3u;  // bundle indices after the insertion: ambient 0, point 1, rect 2, disk 3
    const uint32_t seed = rng::draw(st.seed, rng::Purpose::kLightSampleSet, 0, light_index, 0).v[0];
    std::vector<bool> seen(kLightPoints, false);
    double mean_a = 0.0, mean_b = 0.0, mean_r2 = 0.0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(kLightPoints); ++i) {
      uint32_t zx, zy;
      rng::owen_sobol_2d(i, seed, &zx, &zy);
      const uint32_t cell = light_cell(zx, zy);
      REQUIRE(!seen[cell]);
      seen[cell] = true;
      const float* ab = data.light_points().data() + 2 * (static_cast<size_t>(a) * kLightPoints + cell);
      const float u = rng::to_unit_interval<float>(zx), w = rng::to_unit_interval<float>(zy);
      if (!is_disk) {
        REQUIRE(ab[0] == 2.0f * u - 1.0f);
        REQUIRE(ab[1] == 2.0f * w - 1.0f);
      } else {
        // The Shirley-Chiu map in double from the same square.
        const double x = 2.0 * static_cast<double>(u) - 1.0, y = 2.0 * static_cast<double>(w) - 1.0;
        double ex = 0.0, ey = 0.0;
        if (x != 0.0 || y != 0.0) {
          const bool x_major = std::abs(x) > std::abs(y);
          const double r = x_major ? x : y;
          const double th = x_major ? (std::numbers::pi / 4.0) * (y / x) : std::numbers::pi / 2.0 - (std::numbers::pi / 4.0) * (x / y);
          ex = r * std::cos(th);
          ey = r * std::sin(th);
        }
        REQUIRE(std::abs(static_cast<double>(ab[0]) - ex) < 1e-6);
        REQUIRE(std::abs(static_cast<double>(ab[1]) - ey) < 1e-6);
        REQUIRE(static_cast<double>(ab[0]) * static_cast<double>(ab[0]) + static_cast<double>(ab[1]) * static_cast<double>(ab[1]) <= 1.0 + 1e-6);
      }
      mean_a += static_cast<double>(ab[0]);
      mean_b += static_cast<double>(ab[1]);
      mean_r2 += static_cast<double>(ab[0]) * static_cast<double>(ab[0]) + static_cast<double>(ab[1]) * static_cast<double>(ab[1]);
    }
    CHECK(std::count(seen.begin(), seen.end(), true) == kLightPoints);
    mean_a /= kLightPoints;
    mean_b /= kLightPoints;
    mean_r2 /= kLightPoints;
    CHECK(std::abs(mean_a) < 2e-3);
    CHECK(std::abs(mean_b) < 2e-3);
    CHECK(mean_r2 == doctest::Approx(is_disk ? 0.5 : 2.0 / 3.0).epsilon(5e-3));  // E[r^2]: unit disk 1/2, [-1, 1]^2 2/3
  }
  // A new seed gives new sets; the same one the same (the set is a function of the run seed and the light's index).
  const SceneData same(b, st), other(b, settings(64, 7));
  const bool same_sets = same.light_points() == data.light_points(), other_sets = other.light_points() != data.light_points();
  CHECK(same_sets);
  CHECK(other_sets);

  // Pose rows (36 B): the point light keeps zero axis_u, the area lights' row has the unit axis; centre, normal and axis
  // of the static camera frame come through unchanged.
  static_assert(sizeof(LightRow) == 36);
  PoseTable table(static_cast<int32_t>(data.geometries().size()), 3, 4);
  table.reset(kT0);
  uint32_t slot = 0;
  REQUIRE(table.slot(0, kT0 + 5'000'000, &slot));
  ThreadPool pool(1);
  table.evaluate(data.pose_view(), pool);
  const PoseTableView pv = table.view();
  const LightRow &rp = light_row(pv, slot, 0), &rr = light_row(pv, slot, 1), &rd = light_row(pv, slot, 2);
  CHECK((rp.axis_u[0] == 0.0f && rp.axis_u[1] == 0.0f && rp.axis_u[2] == 0.0f));
  CHECK(rp.position[0] == doctest::Approx(0.1f));
  CHECK(rp.position[1] == doctest::Approx(-0.7f));
  CHECK(rp.position[2] == doctest::Approx(0.4f));
  CHECK(rr.position[1] == doctest::Approx(-0.5f));
  CHECK(rr.position[2] == doctest::Approx(0.8f));
  CHECK(rr.direction[2] == doctest::Approx(1.0f));
  CHECK(rr.axis_u[0] == doctest::Approx(1.0f));
  CHECK(std::abs(rr.axis_u[1]) + std::abs(rr.axis_u[2]) < 1e-6f);
  CHECK(rd.position[0] == doctest::Approx(0.3f));
  CHECK(rd.axis_u[0] == doctest::Approx(1.0f));
}

TEST_CASE("area lights: a moving, turning light carries its axis like its direction") {
  // An area light and a directional light carried by one spinning mover, the directional light's direction being the
  // area light's axis: the rows' axis_u and direction must agree bit for bit (the same expression, compute_slot).
  SceneBundle b;
  b.light_groups = {"default"};
  b.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.1, -0.2, 0.05}), {0.3, -0.1, 0.2},
                                             {0.4, -0.2, 0.3}, {0.3, 1.1, -0.4}));
  b.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{-0.3, 0.2, 0.7}), {0.1, 0.2, 1.5},
                                             {-0.5, 0.25, 0.1}, {4.0, -9.0, 2.0}));
  b.cameras.push_back(pinhole(kW, kH, 60.0, 0));
  Body carrier = make_body(3, box_mesh(0.1f, 0.05f, 0.2f));
  carrier.trajectory = 1;
  b.bodies.push_back(carrier);
  Body floor = make_body(1, plane_mesh(2.0));
  floor.T_world_body = translation(0.0, 0.0, 3.0);
  b.bodies.push_back(floor);
  Light area;
  area.kind = BundleLightKind::kRect;
  area.position = {0.02, 0.03, -0.1};
  area.direction = {0.0, 0.0, -1.0};
  area.axis = {1.0, 0.0, 0.0};
  area.size = {0.1, 0.1};
  area.intensity = 1.0f;
  area.body = 0;
  Light sun_u;  // the axis as a direction
  sun_u.kind = BundleLightKind::kDirectional;
  sun_u.direction = {1.0, 0.0, 0.0};
  sun_u.intensity = 1.0f;
  sun_u.body = 0;
  Light sun_n = sun_u;  // the normal as a direction
  sun_n.direction = {0.0, 0.0, -1.0};
  b.lights = {area, sun_u, sun_n};
  const SceneData data(b, settings());
  ThreadPool pool(1);
  PoseTable table(static_cast<int32_t>(data.geometries().size()), 3, 16);
  table.reset(kT0);
  for (const int64_t t : {kT0 + 1'000'000, kT0 + 73'456'789, kT1 - 1'000}) {
    uint32_t slot = 0;
    REQUIRE(table.slot(0, t, &slot));
  }
  table.evaluate(data.pose_view(), pool);
  const PoseTableView pv = table.view();
  for (uint32_t slot = 0; slot < table.num_slots(); ++slot) {
    const LightRow &ra = light_row(pv, slot, 0), &ru = light_row(pv, slot, 1), &rn = light_row(pv, slot, 2);
    CHECK(std::memcmp(ra.axis_u, ru.direction, sizeof(ra.axis_u)) == 0);
    CHECK(std::memcmp(ra.direction, rn.direction, sizeof(ra.direction)) == 0);
    CHECK(ra.axis_u[0] * ra.axis_u[0] + ra.axis_u[1] * ra.axis_u[1] + ra.axis_u[2] * ra.axis_u[2] == doctest::Approx(1.0f));
    CHECK(std::abs(ra.axis_u[0] * ra.direction[0] + ra.axis_u[1] * ra.direction[1] + ra.axis_u[2] * ra.direction[2]) < 1e-6f);
    // The directional rows keep a zero axis_u.
    CHECK((ru.axis_u[0] == 0.0f && ru.axis_u[1] == 0.0f && ru.axis_u[2] == 0.0f));
  }
}

TEST_CASE("area lights: sub-samples take distinct points, the seeds are the tagged Philox draws") {
  Emitter rect;
  SceneBundle b = scene(rect, false);
  b.lights.push_back(b.lights.back());  // a second area light, bundle light 2
  b.lights.back().position = {0.5, -0.5, 0.8};
  b.cameras.push_back(b.cameras[0]);  // a second camera
  const SceneSettings st = settings(64, 99);
  const SceneData data(b, st);
  REQUIRE(data.num_area_lights() == 2);
  REQUIRE(data.cameras().size() == 2);
  std::set<uint32_t> distinct;
  for (size_t cam = 0; cam < 2; ++cam) {
    const CameraData& c = *data.cameras()[cam];
    REQUIRE(c.light_seeds.size() == static_cast<size_t>(kW) * kH * 2);
    REQUIRE(c.view.light_seeds == c.light_seeds.data());
    for (const uint32_t pixel : {0u, 1u, 77u, 1500u, static_cast<uint32_t>(kW * kH - 1)}) {
      for (uint32_t a = 0; a < 2; ++a) {
        const uint32_t seed = c.light_seeds[pixel * 2u + a];
        CHECK(seed == rng::draw(st.seed, rng::Purpose::kLightPadding, static_cast<uint32_t>(cam), pixel, a + 1).v[0]);
        distinct.insert(seed);
        // The first N padded pairs fall into N distinct cells of the 32 x 32 grid, for every N up to 1024.
        std::vector<bool> used(kLightPoints, false);
        for (uint32_t k = 0; k < static_cast<uint32_t>(kLightPoints); ++k) {
          uint32_t zx, zy;
          rng::owen_sobol_2d(k, seed, &zx, &zy);
          const uint32_t cell = light_cell(zx, zy);
          REQUIRE(!used[cell]);
          used[cell] = true;
        }
      }
    }
  }
  CHECK(distinct.size() == 20);  // 2 cameras x 5 pixels x 2 lights: no two seeds equal
  // The seeds do not depend on the sub-sample pattern's: a different tag and index.
  CHECK(data.cameras()[0]->light_seeds[0] != data.cameras()[0]->seeds[0]);
  // No area light: no seeds, no sets.
  SceneBundle none = scene(rect, false);
  none.lights.pop_back();
  none.lights.push_back(point_light({0.0, -0.5, 0.8}, 1.0f));
  const SceneData lone(none, st);
  CHECK(lone.num_area_lights() == 0);
  CHECK(lone.light_points().empty());
  CHECK(lone.cameras()[0]->light_seeds.empty());
  CHECK(lone.host_view().light_points == nullptr);
  CHECK(lone.cameras()[0]->view.light_seeds == nullptr);
}

TEST_CASE("area lights: the sample points lie on the emitter (rect, ellipse) in the right frame") {
  for (const BundleLightKind kind : {BundleLightKind::kRect, BundleLightKind::kDisk}) {
    Emitter e;
    e.kind = kind;
    e.normal = {0.0, 0.6, 0.8};
    e.axis = {0.0, 0.8, -0.6};  // perpendicular to the normal
    e.size = {0.4, 0.2};
    Host h(scene(e, false), settings(1024));
    const SceneView& v = h.exec.scene_view();
    const LightView& l = v.lights[0];
    const LightRow& row = light_row(h.poses, h.slot, 0);
    const Vec3<double> c{0.0, -0.5, 0.8}, n{0.0, 0.6, 0.8}, u{0.0, 0.8, -0.6};
    // Blender's local frame of a light that emits along its -Z: x = u, z = -n, y = z x x = (-n) x u.
    const Vec3<double> vv{-n.y * u.z + n.z * u.y, -n.z * u.x + n.x * u.z, -n.x * u.y + n.y * u.x};
    const uint32_t pixel = 1234;
    std::set<std::pair<float, float>> distinct;
    double sum_u = 0.0, sum_v = 0.0;
    for (uint32_t k = 0; k < 1024; ++k) {
      const Vec3<double> y = to_d(area_light_point(v, l, row, 0, pixel, k));
      const Vec3<double> d{y.x - c.x, y.y - c.y, y.z - c.z};
      const double pu = dot_d(d, u), pv = dot_d(d, vv);
      REQUIRE(std::abs(dot_d(d, n)) < 1e-6);  // in the emitter's plane
      if (kind == BundleLightKind::kRect) {
        REQUIRE(std::abs(pu) <= 0.2 + 1e-6);
        REQUIRE(std::abs(pv) <= 0.1 + 1e-6);
      } else {
        REQUIRE((pu / 0.2) * (pu / 0.2) + (pv / 0.1) * (pv / 0.1) <= 1.0 + 1e-5);
      }
      distinct.insert({static_cast<float>(pu), static_cast<float>(pv)});
      sum_u += pu;
      sum_v += pv;
    }
    CHECK(distinct.size() == 1024);
    CHECK(std::abs(sum_u / 1024) < 1e-3);
    CHECK(std::abs(sum_v / 1024) < 1e-3);
    // The set follows the frame: v = u x n, so (u, v, -n) is right-handed. A point in the +u/+v quadrant of the local
    // set maps to the +u and +v side: the point of the cell nearest the (+1, +1) corner of the unit square (rect).
    if (kind == BundleLightKind::kRect) {
      double best = -1e30;
      Vec3<double> far{};
      for (uint32_t k = 0; k < 1024; ++k) {
        const Vec3<double> y = to_d(area_light_point(v, l, row, 0, pixel, k));
        const Vec3<double> d{y.x - c.x, y.y - c.y, y.z - c.z};
        if (dot_d(d, u) + dot_d(d, vv) > best) {
          best = dot_d(d, u) + dot_d(d, vv);
          far = d;
        }
      }
      CHECK(dot_d(far, u) > 0.15);
      CHECK(dot_d(far, vv) > 0.07);
    }
  }
}

TEST_CASE("area shadow rays: one-sided emitter, no self-occlusion bit, a ray to the point") {
  // Camera at the origin looking at the plane z = 1 (triangle normal +z, facing away; n_g flips to the camera side).
  const Triangle tri{{-1.0f, -1.0f, 1.0f}, {1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, 0, 1, 2};
  PoseRow row{};
  row.m[0].x = row.m[1].y = row.m[2].z = 1.0f;
  row.sphere.w = 1.0f;
  const Vec3f d{0.1f, 0.05f, 1.0f};
  const Ray ray = primary_to_geometry(row, d);
  const SurfaceHit sh = reintersect(tri, ray);
  REQUIRE(sh.t == doctest::Approx(1.0f));
  // An emitter at z = 0.2 emitting towards +z (the receiver): LightRow {position, direction, axis}.
  const LightRow front{{0.5f, 0.5f, 0.2f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}};
  const Vec3f y{0.55f, 0.45f, 0.2f};
  ShadowRay sr = area_shadow_ray(front, row, d, ray, sh, y);
  REQUIRE(sr.state == ShadowState::kTrace);
  CHECK(sr.origin_cam.z < 1.0f);
  CHECK(sr.origin_cam.z > 0.9999f);
  CHECK(sr.tmax == kAreaShadowTmax);
  CHECK(sr.tmax < 1.0f);
  const Vec3f end = sr.origin_cam + sr.dir_cam;  // the ray ends at the point (t = 1), the segment a hair short of it
  CHECK(end.x == doctest::Approx(0.55f));
  CHECK(end.z == doctest::Approx(0.2f));

  // The same emitter facing away: the point sees the back of the one-sided emitter.
  const LightRow back{{0.5f, 0.5f, 0.2f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f}};
  CHECK(area_shadow_ray(back, row, d, ray, sh, y).state == ShadowState::kNoContribution);
  // A point behind the hit's geometric plane: no contribution, not a self-occlusion bit (a point light there is one).
  const LightRow behind{{0.5f, 0.5f, 3.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f}};
  CHECK(area_shadow_ray(behind, row, d, ray, sh, {0.5f, 0.5f, 3.0f}).state == ShadowState::kNoContribution);
  const LightView point{LightKind::kPoint, 0, 1.0f, 0.0f, 0.0f, 0, 0};
  CHECK(shadow_ray(point, LightRow{{0.5f, 0.5f, 3.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}}, row, d, ray, sh).state ==
        ShadowState::kSelfOccluded);
  // The receiver beside the emitter's plane (the point at the plane's height, receiver off to the side): grazing, zero.
  CHECK(area_shadow_ray(front, row, d, ray, sh, {0.5f, 0.5f, 1.0f}).state == ShadowState::kNoContribution);
}

namespace {

// The mask bit of the reference for the floor receivers of `scene(e, plate)`: the sample at y sees the emitter's front
// side, lies above the floor's visible side, and the segment to y misses the plate. Computed in double from the hit
// point x. margin: the distance of the plate crossing to the plate's edge, negative inside (in metres).
struct Reference {
  bool counted;  // false: no contribution (back of the emitter, below the floor)
  bool blocked;
  double margin;
};

Reference reference_bit(const Emitter& e, bool plate, Vec3<double> x, Vec3<double> y) {
  const Vec3<double> w{y.x - x.x, y.y - x.y, y.z - x.z};
  const Vec3<double> n{e.normal[0], e.normal[1], e.normal[2]};
  if (!(dot_d(n, w) < 0.0) || !(w.z < 0.0)) return {false, false, 0.0};
  if (!plate) return {true, false, 0.0};
  const double s = (kPlateZ - x.z) / w.z;
  if (!(s > 0.0 && s < static_cast<double>(kAreaShadowTmax))) return {true, false, 1.0};
  const double px = x.x + s * w.x - kPlateX, py = x.y + s * w.y - kPlateY;
  const double edge = std::max(std::abs(px), std::abs(py)) - kPlateHalf;  // negative inside
  return {true, edge < 0.0, edge};
}

struct Tally {
  size_t floor_hits = 0, counted = 0, blocked = 0, mixed_pixels = 0, lit_pixels = 0, dark_pixels = 0, mismatches = 0,
         ambiguous = 0;
};

// Compares the traced masks of sub-samples k_begin .. k_begin + n - 1 with the reference, at every floor hit.
Tally compare(Host& h, const Emitter& e, bool plate, uint32_t k_begin, uint32_t n) {
  h.trace(k_begin, n);
  const SceneView& v = h.exec.scene_view();
  const LightRow& row = light_row(h.poses, h.slot, 0);
  Tally t;
  for (uint32_t p = 0; p < static_cast<uint32_t>(kW * kH); ++p) {
    size_t on = 0, off = 0;
    for (uint32_t k = 0; k < n; ++k) {
      const size_t i = static_cast<size_t>(p) * n + k;
      if (h.hits[i].instance != 1) continue;
      ++t.floor_hits;
      const Vec3f d = subsample_direction(v, v.cameras[0], p, k_begin + k);
      const Vec3<double> x = to_d(h.hits[i].t_hit * d);
      const Vec3<double> y = to_d(area_light_point(v, v.lights[0], row, 0, p, k_begin + k));
      const Reference r = reference_bit(e, plate, x, y);
      const bool bit = (h.occ[i] & 1u) != 0;
      REQUIRE((h.occ[i] & ~1u) == 0);  // only the emitter's bit
      t.counted += r.counted;
      t.blocked += bit;
      (bit ? on : off)++;
      if (r.counted && bit == r.blocked) continue;
      if (!r.counted && !bit) continue;
      if (std::abs(r.margin) < 3e-4) {
        ++t.ambiguous;  // the crossing is within fp32 of the plate's edge
      } else {
        ++t.mismatches;
      }
    }
    if (on > 0 && off > 0) ++t.mixed_pixels;
    t.lit_pixels += on == 0 && off > 0;
    t.dark_pixels += off == 0 && on > 0;
  }
  return t;
}

}  // namespace

TEST_CASE("area lights: Embree shadow masks match the reference, a penumbra under a plate, rect and disk") {
  for (const BundleLightKind kind : {BundleLightKind::kRect, BundleLightKind::kDisk}) {
    Emitter e;
    e.kind = kind;
    Host h(scene(e, true), settings(64, 5));
    INFO("kind " << (kind == BundleLightKind::kRect ? "rect" : "disk"));
    // N = 64 from k 0, and a later batch (k_begin 64): the points follow the sub-sample's index, not its place.
    for (const uint32_t k_begin : {0u, 64u}) {
      const Tally t = compare(h, e, true, k_begin, 64);
      MESSAGE("tally " << (kind == BundleLightKind::kRect ? "rect" : "disk") << " k_begin " << k_begin << ": floor hits "
                       << t.floor_hits << ", counted " << t.counted << ", blocked " << t.blocked << ", mixed pixels "
                       << t.mixed_pixels << ", lit pixels " << t.lit_pixels << ", dark pixels " << t.dark_pixels
                       << ", ambiguous " << t.ambiguous << ", mismatches " << t.mismatches);
      CHECK(t.floor_hits > 100000);
      CHECK(t.mismatches == 0);
      CHECK(t.ambiguous * 1000 < t.floor_hits);
      CHECK(t.blocked > 1000);
      CHECK(t.blocked * 3 < t.counted);  // the plate shadows a part of the floor, not most of it
      CHECK(t.mixed_pixels > 100);       // penumbra pixels: some sub-samples blocked, some not
      CHECK(t.dark_pixels > 20);         // the umbra: every sub-sample blocked
      CHECK(t.lit_pixels > 1000);
    }
    // The N points of one pixel are N distinct points of the emitter: the penumbra fraction of a mixed pixel is a
    // multiple of 1/N and is stable (the same masks again).
    h.trace(0, 64);
    const std::vector<uint32_t> first = h.occ;
    h.trace(64, 64);
    const bool other_points = h.occ != first;  // another batch of sub-samples: other points
    CHECK(other_points);
    h.trace(0, 64);
    const bool same_masks = h.occ == first;  // the same sub-samples: the same masks (fixed sets, no per-call state)
    CHECK(same_masks);
  }
}

TEST_CASE("area lights: no blocker, no bits; samples behind the receiver's plane are no contribution") {
  // (a) A front-facing emitter and nothing in the way: every mask is 0.
  {
    Emitter e;
    Host h(scene(e, false), settings());
    const Tally t = compare(h, e, false, 0, 64);
    CHECK(t.floor_hits > 100000);
    CHECK(t.blocked == 0);
    CHECK(t.mismatches == 0);
  }
  // (b) A vertical emitter in the plane y = -0.3 spanning z = 1.3 .. 2.5 across the floor (z = 2): about half of its
  // points lie below the floor's visible side. A point light there is self-occluded; the area light's bit stays 0.
  {
    Emitter e;
    e.centre = {0.0, -0.3, 1.9};
    e.normal = {0.0, 1.0, 0.0};
    e.axis = {0.0, 0.0, 1.0};
    e.size = {1.2, 1.2};
    Host h(scene(e, false), settings());
    const Tally t = compare(h, e, false, 0, 64);
    CHECK(t.floor_hits > 100000);
    CHECK(t.blocked == 0);
    CHECK(t.mismatches == 0);
    CHECK(t.counted > t.floor_hits / 10);  // a good part of the points count ...
    CHECK(t.counted < t.floor_hits * 9 / 10);  // ... and a good part does not (behind the plane, or the emitter's back)
    // The same emitter as a point light at its centre is self-occluded for the receivers below that height.
    SceneBundle b = scene(e, false);
    b.lights.back() = point_light({0.0, -0.3, 2.5}, 2.0f);  // behind the floor's plane (z > 2)
    Host hp(b, settings());
    hp.trace(0, 64);
    size_t floor_hits = 0, bits = 0;
    for (size_t i = 0; i < hp.hits.size(); ++i) {
      if (hp.hits[i].instance != 1) continue;
      ++floor_hits;
      bits += (hp.occ[i] & 1u) != 0;
    }
    CHECK(bits == floor_hits);
  }
  // (c) Behind the emitter: the receivers behind a one-sided emitter's plane get nothing.
  {
    Emitter e;
    e.normal = {0.0, 0.0, -1.0};  // emits away from the floor
    e.axis = {1.0, 0.0, 0.0};
    Host h(scene(e, true), settings());
    h.trace(0, 64);
    size_t bits = 0;
    for (const uint32_t m : h.occ) bits += (m & 1u) != 0;
    CHECK(bits == 0);
  }
}

TEST_CASE("area lights: other lights' bits are those of the scene without them; casts_shadows off traces nothing") {
  Emitter e;
  const std::array<double, 3> lamp{-0.4, -0.7, 0.2};
  // A point light and an area light: the point light's bit equals the bit in a scene with the point light alone.
  SceneBundle with = scene(e, true);
  with.lights.insert(with.lights.begin() + 1, point_light(lamp, 1.5f));  // bit 0 is the point light, bit 1 the area light
  SceneBundle only = scene(e, true);
  only.lights.back() = point_light(lamp, 1.5f);
  Host hw(with, settings()), ho(only, settings());
  hw.trace(0, 64);
  ho.trace(0, 64);
  REQUIRE(hw.occ.size() == ho.occ.size());
  size_t shadowed = 0, area_bits = 0;
  for (size_t i = 0; i < hw.occ.size(); ++i) {
    CHECK(std::memcmp(&hw.hits[i], &ho.hits[i], sizeof(HitRecord)) == 0);
    REQUIRE((hw.occ[i] & 1u) == (ho.occ[i] & 1u));
    shadowed += (ho.occ[i] & 1u) != 0;
    area_bits += (hw.occ[i] & 2u) != 0;
  }
  CHECK(shadowed > 1000);  // the plate shadows the point light on part of the floor
  CHECK(area_bits > 1000);
  // An area light that casts no shadows has no bit and no ray.
  Host hn(scene(e, true, false), settings());
  hn.trace(0, 64);
  size_t nonzero = 0;
  for (const uint32_t m : hn.occ) nonzero += m != 0;
  CHECK(nonzero == 0);
  CHECK(hn.data.num_shadow_lights() == 0);
}

TEST_CASE("area lights: the plan pass traces the emitter's centre like a point light there") {
  // The same scene with a point light at the area light's centre: the centre rays' shadow states and blocker parameters
  // are bit for bit those of the point light (the emitter faces every receiver, so no rule of the area light applies).
  for (const BundleLightKind kind : {BundleLightKind::kRect, BundleLightKind::kDisk}) {
    Emitter e;
    e.kind = kind;
    Host ha(scene(e, true), settings());
    SceneBundle bp = scene(e, true);
    bp.lights.back() = point_light(e.centre, 2.0f);
    Host hp(bp, settings());
    std::vector<GtRequest> req;
    for (int y = 0; y < kH; ++y) {
      for (int x = 0; x < kW; ++x) req.push_back({0, 0, static_cast<float>(x), static_cast<float>(y), ha.slot});
    }
    const auto count = static_cast<uint32_t>(req.size());
    std::vector<GtHit> gh_a(count), gh_p(count);
    std::vector<uint32_t> sh_a(2u * count, 0xFFFFFFFFu), sh_p(2u * count, 0xFFFFFFFFu);
    ha.tracer.centre({req.data(), count, ha.poses, nullptr, 0}, gh_a.data(), sh_a.data());
    hp.tracer.centre({req.data(), count, hp.poses, nullptr, 0}, gh_p.data(), sh_p.data());
    const bool same_states = sh_a == sh_p;
    CHECK(same_states);
    size_t mover = 0, lit = 0, stat = 0;
    for (uint32_t i = 0; i < count; ++i) {
      mover += sh_a[2 * i] >= kShadowMover && sh_a[2 * i] != kShadowUnknown;
      lit += sh_a[2 * i] == kShadowLit;
      stat += sh_a[2 * i] == kShadowStatic;
    }
    CHECK(mover > 50);  // the plate's shadow of the centre
    CHECK(lit > 500);
    CHECK(mover + lit + stat == count);
  }
}
