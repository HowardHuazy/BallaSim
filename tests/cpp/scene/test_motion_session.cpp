// The motion session (src/scene/motion/session.h; docs/design/phase3/blender.md §3.5, §3.8, §5.1; BL-9) against the
// synthetic oracles of motion_oracles.h: the time settings and the key scene read from an exchange, the whole session on
// a hand-built exchange (columns, survey, presence, Simplify, rigs, statics, fits, outputs), its replay from the log and
// the refusals that need no Blender.
#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "motion_oracles.h"
#include "scene/blender/exchange_in.h"
#include "scene/motion/session.h"

namespace mo = balla::scene::motion;
namespace bl = balla::scene::blender;
namespace mt = balla::scene::motion::testing;
using balla::scene::Tree;

namespace {

// A tetrahedron (4 vertices, 4 loop triangles on 12 corners) of circumradius ~ r about the origin.
bl::MeshIn tetra(const std::string& id, float r) {
  bl::MeshIn m;
  m.id = id;
  m.domain = bl::MeshIn::Normals::kCorner;
  const float v[4][3] = {{r, r, r}, {r, -r, -r}, {-r, r, -r}, {-r, -r, r}};
  for (const auto& p : v) m.co.insert(m.co.end(), p, p + 3);
  const int tri[4][3] = {{0, 1, 2}, {0, 3, 1}, {0, 2, 3}, {1, 3, 2}};
  for (const auto& t : tri) {
    for (const int k : t) m.corner_vert.push_back(k);
  }
  for (int c = 0; c < 12; c += 3) m.tri_corners.insert(m.tri_corners.end(), {c, c + 1, c + 2});
  m.tri_mat.assign(4, 0);
  m.normals.assign(12 * 3, 0.0f);
  for (int c = 0; c < 12; ++c) {
    const float* p = v[m.corner_vert[static_cast<size_t>(c)]];
    const float n = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    for (int k = 0; k < 3; ++k) m.normals[static_cast<size_t>(3 * c + k)] = p[k] / n;
  }
  return m;
}

bl::ObjectIn object(const std::string& name, int mesh, std::array<double, 3> t, int id) {
  bl::ObjectIn o;
  o.key = "[\"" + name + "\"]";
  o.name_full = name;
  o.type = mesh >= 0 ? "MESH" : "CAMERA";
  o.mesh = mesh;
  o.instance_id = id;
  o.matrix_world = {1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2], 0, 0, 0, 1};
  return o;
}

Tree fcurve(const std::string& key, const std::string& path, int index) {
  Tree f = Tree::dict();
  f["key"] = Tree::string(key);
  f["data_path"] = Tree::string(path);
  f["index"] = Tree::integer(index);
  f["extrapolation"] = Tree::integer(0);
  f["keyframes"] = Tree::string("anim." + key + ".keys");
  f["modifiers"] = Tree::list();
  return f;
}

// camera (0) and a tetrahedron (1) whose location is keyed (LINEAR, every 6 frames from frame 1 to 73: the synthetic
// `linear` motion of motion_oracles.h).
bl::ExchangeIn exchange() {
  bl::ExchangeIn ex;
  ex.time = Tree::dict();
  ex.time["fps"] = Tree::integer(24);
  ex.time["fps_base_bits"] = Tree::string("0x3f800000");
  ex.time["frame_start"] = Tree::integer(1);
  ex.time["frame_end"] = Tree::integer(73);
  ex.time["frame_map_old"] = Tree::integer(100);
  ex.time["frame_map_new"] = Tree::integer(100);
  ex.span = Tree::dict();
  ex.span["frame_start"] = Tree::integer(1);
  ex.span["frame_end"] = Tree::integer(73);
  ex.render = Tree::dict();
  ex.render["resolution_x"] = Tree::integer(64);
  ex.render["resolution_y"] = Tree::integer(48);
  ex.render["resolution_percentage"] = Tree::integer(100);
  ex.meshes.push_back(tetra("tet", 0.05f));
  ex.objects.push_back(object("Camera", -1, {0.0, -1.0, 0.2}, 0));
  ex.objects.push_back(object("Tet", 0, {mt::linear_key_value(0, 0), mt::linear_key_value(0, 1),
                                         mt::linear_key_value(0, 2)}, 1));
  ex.objects[1].animation = 0;
  bl::CameraIn cam;
  cam.object = 0;
  cam.props = Tree::dict();
  cam.props["type"] = Tree::string("PERSP");
  cam.props["lens"] = Tree::real(50.0);
  ex.cameras.push_back(cam);
  Tree anim = Tree::dict();
  anim["action"] = Tree::string("act");
  Tree fcs = Tree::list();
  for (int axis = 0; axis < 3; ++axis) {
    const std::string key = "0_" + std::to_string(axis);
    fcs.push_back(fcurve(key, "location", axis));
    std::vector<double>& k = ex.anim_keys["anim." + key + ".keys"];
    for (int i = 0; i <= 12; ++i) {
      k.insert(k.end(), {1.0 + 6.0 * i, mt::linear_key_value(i, axis), 1, 0, 0, 0, 0});
    }
  }
  anim["fcurves"] = fcs;
  ex.animations = Tree::list();
  ex.animations.push_back(anim);
  return ex;
}

// Answers as Blender would: the tetrahedron follows mt::linear (its location is the keyed motion; its spin is not keyed
// and stays out of the test's way), the camera is still. `absent_first`: the tetrahedron is missing at the export instant.
struct Fake {
  std::shared_ptr<std::vector<int32_t>> columns = std::make_shared<std::vector<int32_t>>();
  std::shared_ptr<std::vector<bool>> simplify = std::make_shared<std::vector<bool>>();
  std::shared_ptr<int> shift_with_simplify = std::make_shared<int>(0);
  bool absent_first = false;
  mo::TimeMap time = mt::time24(73);

  mo::SessionIO io() const {
    mo::SessionIO io;
    auto cols = columns;
    auto simp = simplify;
    auto shift = shift_with_simplify;
    const mo::TimeMap tm = time;
    const bool absent = absent_first;
    io.bodies = [cols](std::span<const int32_t> b) { cols->assign(b.begin(), b.end()); };
    io.simplify = [simp](bool on) { simp->push_back(on); };
    io.oracle = [cols, simp, shift, tm, absent](std::span<const int32_t> F, std::span<const float> s) {
      mo::RawPoses r;
      r.n = static_cast<int32_t>(F.size());
      r.bodies = static_cast<int32_t>(cols->size());
      r.m.resize(F.size() * cols->size() * 12);
      r.present.assign(F.size() * cols->size(), 1);
      r.unknown.assign(F.size(), 0);
      for (size_t i = 0; i < F.size(); ++i) {
        const float c = mo::TimeMap::realized(F[i], s[i]);
        r.c_final.push_back(static_cast<double>(tm.reported(c)));
        for (size_t b = 0; b < cols->size(); ++b) {
          mt::Pose p;
          if ((*cols)[b] == 1) {
            p = mt::linear(static_cast<double>(c));
            p.q = {1, 0, 0, 0};
            if (!simp->empty() && simp->back() && *shift != 0) p.p.x += 1e-3;
          } else {
            p = mt::camera(0.0);
          }
          mt::to_matrix34(p, 1.0, &r.m[(i * cols->size() + b) * 12]);
          if (absent && i == 0 && F[i] == 1 && (*cols)[b] == 1) r.present[i * cols->size() + b] = 0;
        }
      }
      return r;
    };
    return io;
  }
};

}  // namespace

TEST_CASE("time_settings and key_scene read the exchange") {
  bl::ExchangeIn ex = exchange();
  ex.time["fps_base_bits"] = Tree::string("0x3f7ff7cf");   // 1000 / 1001 as float32
  ex.render["use_motion_blur"] = Tree::boolean(true);
  ex.render["motion_blur_shutter"] = Tree::real(0.25);
  ex.render["motion_blur_position"] = Tree::string("START");
  const mo::TimeSettings s = mo::time_settings(ex, true);
  CHECK(s.fps == 24);
  CHECK(s.fps_base_bits == 0x3f7ff7cfu);
  CHECK(s.frame_start == 1);
  CHECK(s.frame_end == 73);
  CHECK(s.motion_blur);
  CHECK(s.motion_blur_shutter == 0.25f);
  CHECK(s.motion_blur_position == "START");
  CHECK(s.fps_base_snap);
  ex.time["fps"] = Tree::integer(0);
  CHECK_THROWS_WITH_AS(mo::time_settings(ex, false), doctest::Contains("BRG-PROTO"), std::invalid_argument);

  ex = exchange();
  Tree c = Tree::dict();
  c["type"] = Tree::string("COPY_LOCATION");
  c["target"] = Tree::string("Tet");
  c["mute"] = Tree::boolean(false);
  ex.objects[0].constraints = Tree::list();
  ex.objects[0].constraints.push_back(c);
  c["mute"] = Tree::boolean(true);
  ex.objects[0].constraints.push_back(c);
  ex.objects[0].parent = 1;
  const mo::KeyScene ks = mo::key_scene(ex);
  REQUIRE(ks.objects.size() == 2);
  CHECK(ks.objects[0].parent == 1);
  CHECK(ks.objects[0].constraint_targets == std::vector<int32_t>{1});
  CHECK(ks.objects[1].animations == std::vector<int32_t>{0});
  REQUIRE(ks.animations.size() == 1);
  REQUIRE(ks.animations[0].fcurves.size() == 3);
  CHECK(ks.animations[0].fcurves[1].data_path == "location");
  CHECK(ks.animations[0].fcurves[1].index == 1);
  CHECK(ks.animations[0].fcurves[2].keys.size() == 13 * 7);
  // the keys of the tetrahedron: C0 at each of the 13 LINEAR keys
  const mo::TimeMap tm = mt::time24(73);
  const auto keys = mo::declared_keys(ks, 1, tm, -2 * balla::time::kWindowNs, tm.t_end_ns());
  REQUIRE(keys.size() == 13);
  CHECK(keys[1].t_ns == tm.to_ns(7.0f));
  ex.anim_keys.clear();
  CHECK_THROWS_WITH_AS(mo::key_scene(ex), doctest::Contains("no key table"), std::invalid_argument);
}

TEST_CASE("run_session: columns, survey, Simplify, static camera, a keyed body fitted, outputs") {
  const bl::ExchangeIn ex = exchange();
  const Fake fake;
  mo::SessionOptions opt;
  opt.workers = 2;
  const mo::SessionResult r = mo::run_session(ex, opt, fake.io());
  REQUIRE_FALSE(r.refused);
  CHECK(*fake.columns == std::vector<int32_t>{1, 0});   // the body, then the camera
  CHECK(*fake.simplify == std::vector<bool>{false, true});
  CHECK(r.simplify);
  // D-P3-12: the trajectories start at -5 W; the bundle at -2 W
  const int64_t W = balla::time::kWindowNs;
  REQUIRE(r.trajectories.size() == 2);
  CHECK(r.trajectories[0].traj.span_begin_ns == -5 * W);
  CHECK(r.trajectories[0].traj.span_end_ns == r.time->t_end_ns());
  CHECK(r.trajectories[0].traj.pieces.size() >= 2);
  CHECK(r.trajectories[1].traj.pieces.empty());   // the camera's rig: one spline, held
  CHECK(r.trajectories[1].holds.size() == 1);
  CHECK(r.body_trajectory == std::vector<int32_t>{0});
  CHECK(r.camera_trajectory == std::vector<int32_t>{1});
  CHECK(r.camera_t_rig_cam[0][5] == -1.0);   // diag(1, -1, -1, 1)
  const Tree& bodies = *r.session.find("bodies");
  REQUIRE(bodies.items().size() == 2);
  const Tree& tet = bodies.items()[0];
  CHECK(tet.find("mode")->as_string() == "keyed");
  CHECK(tet.find("breakpoints")->items().size() >= 12);
  CHECK(r.perf.find("calls")->as_int() >= 100);
  CHECK(r.perf.find("messages")->as_int() == static_cast<int64_t>(r.log.poses.calls.size()));
  CHECK(r.log.bodies == std::vector<int32_t>{1, 0});
  // replay: the same decisions, bitwise
  mo::SessionResult again = mo::run_session(ex, opt, mo::replaying_io(r.log));
  REQUIRE_FALSE(again.refused);
  REQUIRE(again.trajectories.size() == r.trajectories.size());
  for (size_t i = 0; i < r.trajectories.size(); ++i) {
    CHECK(again.trajectories[i].traj.rot == r.trajectories[i].traj.rot);
    CHECK(again.trajectories[i].traj.pos == r.trajectories[i].traj.pos);
  }
  CHECK(again.origin_world == r.origin_world);
  // other options ask other questions
  opt.survey_samples = 17;
  CHECK_THROWS_WITH_AS(mo::run_session(ex, opt, mo::replaying_io(r.log)), doctest::Contains("BRG-REPLAY"),
                       mo::MotionError);
}

TEST_CASE("run_session: the export instant must hold every body (BRG-STATE) and Simplify must not move a matrix") {
  const bl::ExchangeIn ex = exchange();
  Fake fake;
  fake.absent_first = true;
  CHECK_THROWS_WITH_AS(mo::run_session(ex, {}, fake.io()), doctest::Contains("BRG-STATE: Tet is absent"),
                       mo::MotionError);
  Fake moved;
  *moved.shift_with_simplify = 1;
  const mo::SessionResult r = mo::run_session(ex, {}, moved.io());
  CHECK_FALSE(r.simplify);
  CHECK(*moved.simplify == std::vector<bool>{false, true, false});
}

TEST_CASE("run_session: a refused conversion makes no oracle call") {
  bl::ExchangeIn ex = exchange();
  ex.objects[1].visibility = Tree::dict();
  ex.objects[1].visibility["holdout"] = Tree::boolean(true);   // VIS: refused by the statics
  const Fake fake;
  int calls = 0;
  mo::SessionIO io = fake.io();
  const mo::Oracle inner = io.oracle;
  io.oracle = [&](std::span<const int32_t> F, std::span<const float> s) {
    ++calls;
    return inner(F, s);
  };
  const mo::SessionResult r = mo::run_session(ex, {}, io);
  CHECK(r.refused);
  CHECK(calls == 0);
  CHECK(fake.columns->empty());
}

TEST_CASE("replaying_io refuses another set of columns and other mesh probes") {
  mo::SessionLog log;
  log.bodies = {1, 0};
  mo::MeshCall mc;
  mc.objects = {3};
  mc.F = {1, 2};
  mc.s = {0.0f, 0.5f};
  log.meshes.push_back(mc);
  const mo::SessionIO io = mo::replaying_io(log);
  const std::vector<int32_t> bad{0, 1}, good{1, 0}, F{1, 2}, other{1, 3};
  const std::vector<float> s{0.0f, 0.5f}, s2{0.0f, 0.25f};
  CHECK_THROWS_WITH_AS(io.bodies(bad), doctest::Contains("BRG-REPLAY"), mo::MotionError);
  CHECK_NOTHROW(io.bodies(good));
  CHECK_THROWS_WITH_AS(io.meshes(std::vector<int32_t>{4}, F, s), doctest::Contains("BRG-REPLAY"), mo::MotionError);
  CHECK_THROWS_WITH_AS(io.meshes(std::vector<int32_t>{3}, other, s), doctest::Contains("BRG-REPLAY"), mo::MotionError);
  CHECK_THROWS_WITH_AS(io.meshes(std::vector<int32_t>{3}, F, s2), doctest::Contains("BRG-REPLAY"), mo::MotionError);
  CHECK_NOTHROW(io.meshes(std::vector<int32_t>{3}, F, s));
  CHECK_THROWS_WITH_AS(io.meshes(std::vector<int32_t>{3}, F, s), doctest::Contains("BRG-REPLAY"), mo::MotionError);
}
