// The Blender motion session's pieces (src/scene/motion; docs/design/phase3/blender.md §4.2, §4.7.2, §4.7.3): the
// TimeMap (T1's C++ part; its exactness against Python's fractions is tests/python/test_motion_time_map.py), decode,
// the static distance, the camera gains and the error model, the base requests and the midpoint test.
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "core/camera/injective_domain.h"
#include "core/camera/radtan.h"
#include "core/math/so3.h"
#include "core/time/time.h"
#include "scene/motion/error_model.h"
#include "scene/motion/pipeline.h"
#include "scene/motion/time_map.h"

namespace mo = balla::scene::motion;
namespace rot = balla::scene::rot;

namespace {

constexpr int64_t W = balla::time::kWindowNs;

mo::TimeSettings settings(int32_t fps, float fps_base, int32_t f0, int32_t f1) {
  mo::TimeSettings s;
  s.fps = fps;
  s.fps_base_bits = std::bit_cast<uint32_t>(fps_base);
  s.frame_start = f0;
  s.frame_end = f1;
  return s;
}

double half_ulp(float x) {
  x = std::fabs(x);
  return 0.5 * (static_cast<double>(std::nextafter(x, 2.0f * x + 1.0f)) - static_cast<double>(x));
}

// One raw answer of B bodies at the requests (F, s), each body's matrix given.
mo::RawPoses raw_of(const mo::TimeMap& tm, const std::vector<int32_t>& F, const std::vector<float>& s,
                    const std::vector<std::array<float, 12>>& m) {
  mo::RawPoses r;
  r.n = static_cast<int32_t>(F.size());
  r.bodies = static_cast<int32_t>(m.size());
  r.F = F;
  r.s = s;
  for (size_t i = 0; i < F.size(); ++i) {
    r.c_final.push_back(static_cast<double>(tm.reported(mo::TimeMap::realized(F[i], s[i]))));
    for (const auto& mb : m) r.m.insert(r.m.end(), mb.begin(), mb.end());
  }
  r.present.assign(F.size() * m.size(), 1);
  r.unknown.assign(F.size(), 0);
  return r;
}

constexpr std::array<double, 9> kUnitS{1, 0, 0, 0, 1, 0, 0, 0, 1};

// n columns of the default role (a mesh body) at unit scale: the S_ref every mesh body and light carrier needs
mo::MotionScene unit_bodies(size_t n) {
  mo::MotionScene scene;
  scene.bodies.resize(n);
  for (mo::MotionBody& b : scene.bodies) b.S_ref = kUnitS;
  return scene;
}

}  // namespace

TEST_CASE("TimeMap: frames to ns exactly, frame_start is t = 0") {
  const mo::TimeMap tm(settings(24, 1.0f, 1, 250));
  CHECK(tm.to_ns(1.0f) == 0);
  CHECK(tm.to_ns(25.0f) == 1000000000);
  CHECK(tm.to_ns(1.5f) == 20833333);   // 1/48 s = 20833333.33 ns
  CHECK(tm.to_ns(0.0f) == -41666667);  // -1/24 s
  CHECK(tm.frames_to_ns(1, 8) == 5208333);
  CHECK(tm.frames_to_ns(-3, 16) == -7812500);
  // rounded half to even: half a frame at 3 fps is 166666666.67 ns; at 8000 fps 1/250000 frame is 0.5 ns
  const mo::TimeMap t3(settings(3, 1.0f, 0, 10));
  CHECK(t3.frames_to_ns(1, 2) == 166666667);
  const mo::TimeMap t8(settings(8000, 1.0f, 0, 10));
  CHECK(t8.frames_to_ns(1, 2) == 62500);
  CHECK(t8.frames_to_ns(1, 250000) == 0);  // 0.5 ns -> 0 (even)
  CHECK(t8.frames_to_ns(3, 250000) == 2);  // 1.5 ns -> 2 (even)
}

TEST_CASE("TimeMap: requests round trip within the time resolution, stamped from the realized frame") {
  const mo::TimeMap tm(settings(24, 1.0f, 1, 250));
  std::mt19937_64 rng(7);
  std::uniform_int_distribution<int64_t> dist(-2 * W, 10400000000);
  for (int i = 0; i < 20000; ++i) {
    const int64_t t = dist(rng);
    int32_t F = 0;
    float s = 0.0f;
    tm.request(t, &F, &s);
    CHECK((s == 0.0f || (s >= 0x1p-30f && s < 1.0f)));
    const int64_t back = tm.to_ns(mo::TimeMap::realized(F, s));
    CHECK(static_cast<double>(std::llabs(back - t)) <= tm.resolution_ns(F) + 0.5);
  }
  // q(F) = 1/2 ulp32(F) P: 0.02 us at frame 10 and 1.3 us at frame 1000 (blender.md §1)
  CHECK(tm.resolution_ns(10) == doctest::Approx(19.8682).epsilon(1e-4));
  CHECK(tm.resolution_ns(1000) == doctest::Approx(1271.57).epsilon(1e-4));
}

TEST_CASE("TimeMap: a -2W pre-roll from frame_start 0 is stamped from realized frames and passes BRG-TIME") {
  const mo::TimeMap tm(settings(24, 1.0f, 0, 48));
  const mo::MotionScene scene = unit_bodies(1);
  int differ = 0, n = 0;
  for (int64_t t = -2 * W; t < 0; t += 9973) {
    int32_t F = 0;
    float s = 0.0f;
    tm.request(t, &F, &s);
    const float c = mo::TimeMap::realized(F, s);
    // the target's own float32 frame c0 and the realized c' = f32(F + s)
    const auto x = static_cast<float>(static_cast<double>(t) * 24e-9);
    differ += (c != x) ? 1 : 0;
    ++n;
    const std::array<float, 12> id{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    const mo::PoseBatch b = mo::decode(raw_of(tm, {F}, {s}, {id}), tm, scene);
    CHECK(b.t_ns[0] == tm.to_ns(c));
    CHECK(b.c[0] == c);
  }
  CHECK(differ > n / 4);  // RI2: 70.6 % of such requests realize c' != c
}

TEST_CASE("TimeMap: framelen, the end of the bundle, the record") {
  mo::TimeSettings s = settings(24, 1.0f, 1, 250);
  s.frame_map_old = 100;
  s.frame_map_new = 200;
  const mo::TimeMap half(s);
  CHECK(half.reported(10.0f) == 5.0f);
  CHECK(half.to_ns(25.0f) == 1000000000);  // frame_map changes the evaluation time only
  s.frame_map_new = 50;
  CHECK(mo::TimeMap(s).reported(20.5f) == 41.0f);
  // t(frame_end) a multiple of W: 16384 frames at 15625 fps = 1.048576 s = 125 W
  const mo::TimeMap mult(settings(15625, 1.0f, 0, 16384));
  CHECK(mult.to_ns(16384.0f) == 125 * W);
  CHECK(mult.t_end_ns() == 126 * W);
  // a single-frame span gets one window
  const mo::TimeMap single(settings(24, 1.0f, 7, 7));
  CHECK(single.t_end_ns() == W);
  const balla::scene::Tree rec = mult.record();
  CHECK(rec.find("run_span_extends_ns")->as_int() == W);
  CHECK(rec.find("bundle_span")->items()[0].as_int() == -2 * W);
  const mo::TimeMap back = mo::TimeMap::from_record(half.record());
  CHECK(back.reported(10.0f) == 5.0f);
  CHECK(back.t_end_ns() == half.t_end_ns());
  CHECK(back.to_ns(13.25f) == half.to_ns(13.25f));
  balla::scene::Tree bad = half.record();
  bad["t_frame_end_ns"] = balla::scene::Tree::integer(5);
  CHECK_THROWS_AS(mo::TimeMap::from_record(bad), std::invalid_argument);
  // every field of its kind: an int shutter is its value; an int `use`, a hex string with trailing characters or more
  // than 32 bits are refused
  using balla::scene::Tree;
  Tree edited = half.record();
  edited["motion_blur"]["use"] = Tree::boolean(true);
  edited["motion_blur"]["shutter"] = Tree::integer(1);
  const Tree again = mo::TimeMap::from_record(edited).record();
  CHECK(again.find("motion_blur")->find("use")->as_bool());
  CHECK(again.find("motion_blur")->find("shutter")->as_float() == 1.0);
  auto refused = [&](const char* key, Tree value, const char* sub = nullptr) {
    Tree r = half.record();
    (sub ? r[sub][key] : r[key]) = std::move(value);
    CHECK_THROWS_AS(mo::TimeMap::from_record(r), std::invalid_argument);
  };
  refused("use", Tree::integer(1), "motion_blur");
  refused("shutter", Tree::string("0.5"), "motion_blur");
  refused("shutter", Tree::real(0.1), "motion_blur");  // not a float32
  refused("position", Tree::integer(0), "motion_blur");
  refused("framelen_bits", Tree::string("0x3f000000xyz"));
  refused("framelen_bits", Tree::string("0x13f000000"));
  refused("fps_base_bits", Tree::string("3f800000"));
  refused("fps_base_bits", Tree::string("0x"));
  refused("fps", Tree::real(24.0));
  refused("fps", Tree::integer((int64_t{1} << 32) + 24));  // not truncated to 24
  refused("run_span", Tree::ints({0}));
}

TEST_CASE("TimeMap: from_record checks the inputs against each other, q_max_ns included") {
  using balla::scene::Tree;
  // derived fields copied from another record make a consistent-looking record whose inputs disagree
  auto with_derived = [](Tree r, const Tree& from) {
    for (const char* k : {"t_frame_end_ns", "bundle_span", "run_span", "run_span_extends_ns", "q_max_ns"}) {
      r[k] = *from.find(k);
    }
    return r;
  };
  auto without = [](const Tree& r, const char* key) {
    Tree out = Tree::dict();
    for (const auto& [k, v] : r.entries()) {
      if (k != key) out[k] = v;
    }
    return out;
  };
  auto refuses = [](const Tree& r, const char* what) {
    CHECK_THROWS_WITH_AS(mo::TimeMap::from_record(r), doctest::Contains(what), std::invalid_argument);
  };
  const mo::TimeMap t24(settings(24, 1.0f, 1, 250));
  // base_num / base_den: 1/2 under fps_base_bits 1.0 with a 48 fps map's spans; an unreduced 2/2; 3/1 under 2.0
  Tree r = with_derived(t24.record(), mo::TimeMap(settings(48, 1.0f, 1, 250)).record());
  r["base_num"] = Tree::integer(1);
  r["base_den"] = Tree::integer(2);
  refuses(r, "inconsistent base_num / base_den");
  r = t24.record();
  r["base_num"] = Tree::integer(2);
  r["base_den"] = Tree::integer(2);
  refuses(r, "inconsistent base_num / base_den");
  const mo::TimeMap t2(settings(24, 2.0f, 7, 7));
  CHECK(mo::TimeMap::from_record(t2.record()).period_ns() == t2.period_ns());
  r = t2.record();
  r["base_num"] = Tree::integer(3);
  refuses(r, "inconsistent base_num / base_den");
  // a snapped base and the raw float32 base are both the fps_base_bits' own
  mo::TimeSettings ntsc = settings(30, 1.001f, 0, 1000);
  const mo::TimeMap raw(ntsc);
  ntsc.fps_base_snap = true;
  const mo::TimeMap snapped(ntsc);
  CHECK(mo::TimeMap::from_record(raw.record()).base_den() == (int64_t{1} << 23));
  const mo::TimeMap back = mo::TimeMap::from_record(snapped.record());
  CHECK(back.base_num() == 1001);
  CHECK(back.base_den() == 1000);
  CHECK(back.to_ns(1000.0f) == snapped.to_ns(1000.0f));
  // frame_ref is frame_start: a 25 fps record with frame_ref 2 and t_ref_ns 40 ms keeps every time and is refused,
  // as is frame_ref 0 under frame_start 1 with frame_start 0's spans; t_ref_ns is 0
  r = mo::TimeMap(settings(25, 1.0f, 1, 100)).record();
  r["frame_ref"] = Tree::integer(2);
  r["t_ref_ns"] = Tree::integer(40000000);
  refuses(r, "frame_ref is not frame_start");
  r = mo::TimeMap(settings(24, 1.0f, 0, 250)).record();
  r["frame_start"] = Tree::integer(1);
  refuses(r, "frame_ref is not frame_start");
  r = t24.record();
  r["t_ref_ns"] = Tree::integer(5);
  refuses(r, "t_ref_ns is not 0");
  // q_max_ns: present, a number, the recomputed value (an int of that value is that number)
  refuses(without(t24.record(), "q_max_ns"), "missing q_max_ns");
  r = t24.record();
  r["q_max_ns"] = Tree::string("x");
  refuses(r, "q_max_ns is not a number");
  r = t24.record();
  r["q_max_ns"] = Tree::real(2.0 * r.find("q_max_ns")->as_float());
  refuses(r, "inconsistent q_max_ns");
  r = t24.record();
  CHECK(mo::TimeMap::from_record(r).record().find("q_max_ns")->as_float() == r.find("q_max_ns")->as_float());
}

TEST_CASE("TimeMap: 30000/1001 fps with and without fps_base_snap") {
  mo::TimeSettings s = settings(30, 1.001f, 0, 100000);
  const mo::TimeMap raw(s);
  CHECK(raw.base_den() == (int64_t{1} << 23));
  s.fps_base_snap = true;
  const mo::TimeMap snapped(s);
  CHECK(snapped.base_num() == 1001);
  CHECK(snapped.base_den() == 1000);
  CHECK(snapped.to_ns(30000.0f) == 1001000000000);
  // float32 1.001 is 1.0010000467: 46.7 us later after 1001 s
  CHECK(raw.to_ns(30000.0f) - snapped.to_ns(30000.0f) == 46730);
}

TEST_CASE("decode: BRG-TIME when frame_current_final differs from f32(c' framelen)") {
  const mo::TimeMap tm(settings(24, 1.0f, 1, 48));
  const mo::MotionScene scene = unit_bodies(1);
  const std::array<float, 12> id{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  mo::RawPoses r = raw_of(tm, {5}, {0.25f}, {id});
  CHECK_NOTHROW(mo::decode(r, tm, scene));
  r.c_final[0] = static_cast<double>(std::nextafter(5.25f, 10.0f));  // 1 ulp32 off, shown in full
  CHECK_THROWS_WITH_AS(mo::decode(r, tm, scene),
                       doctest::Contains("BRG-TIME: frame_current_final 5.250000476837158 at F = 5, s = 0.25 differs "
                                         "from the expected 5.25"),
                       mo::MotionError);
}

TEST_CASE("decode: BRG-PROTO for a non-finite matrix_world, GEO-SCALEANIM for a singular one, naming the body") {
  const mo::TimeMap tm(settings(24, 1.0f, 1, 48));
  const mo::MotionScene scene = unit_bodies(2);
  const std::array<float, 12> id{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  for (const float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    for (const int entry : {3, 5}) {  // a translation and a rotation entry
      std::array<float, 12> m = id;
      m[static_cast<size_t>(entry)] = bad;
      CHECK_THROWS_WITH_AS(mo::decode(raw_of(tm, {7}, {0.5f}, {id, m}), tm, scene),
                           doctest::Contains("BRG-PROTO: body column 1 has a non-finite matrix_world at t = 270833333"),
                           mo::MotionError);
    }
  }
  // an absent body's matrix is not read
  mo::RawPoses r = raw_of(tm, {7}, {0.5f}, {id, {std::numeric_limits<float>::infinity()}});
  r.present[1] = 0;
  CHECK_NOTHROW(mo::decode(r, tm, scene));
  const std::array<float, 12> flat{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
  CHECK_THROWS_WITH_AS(mo::decode(raw_of(tm, {7}, {0.0f}, {flat, id}), tm, scene),
                       doctest::Contains("GEO-SCALEANIM: body column 0 has a singular matrix_world"), mo::MotionError);
}

TEST_CASE("decode: polar decomposition, mirrors, recentring and the float32 rounding bounds") {
  const mo::TimeMap tm(settings(24, 1.0f, 1, 48));
  mo::MotionScene scene = unit_bodies(2);
  scene.scale_length = 0.5;
  scene.bodies[0].S_ref = std::array<double, 9>{-1, 0, 0, 0, -1, 0, 0, 0, -2};
  scene.bodies[1].S_ref = std::array<double, 9>{2, 0, 0, 0, 2, 0, 0, 0, 2};
  scene.bodies[1].centre_m = {0.1, 0.0, 0.0};
  // body 0: the design's mirror A = diag(-1, 1, 2) -> R = diag(1, -1, -1), S = diag(-1, -1, -2)
  // body 1: 2 R_z(30 deg), t = (3, -1, 0.25)
  const double c30 = std::cos(M_PI / 6), s30 = std::sin(M_PI / 6);
  const std::array<float, 12> mirror{-1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 2, 3};
  const std::array<float, 12> scaled{static_cast<float>(2 * c30), static_cast<float>(-2 * s30), 0, 3,
                                     static_cast<float>(2 * s30), static_cast<float>(2 * c30), 0, -1,
                                     0, 0, 2, 0.25f};
  const mo::PoseBatch b = mo::decode(raw_of(tm, {3}, {0.0f}, {mirror, scaled}), tm, scene);
  const rot::Q q0 = b.q[0];
  CHECK(std::fabs(q0.x) == doctest::Approx(1.0));  // R = diag(1, -1, -1): pi about x
  CHECK(q0.w == doctest::Approx(0.0).epsilon(1e-15));
  const auto& S0 = b.S[0];
  CHECK(S0[0] == doctest::Approx(-1.0));
  CHECK(S0[4] == doctest::Approx(-1.0));
  CHECK(S0[8] == doctest::Approx(-2.0));
  CHECK(b.p[0].x == doctest::Approx(0.5));
  CHECK(b.p[0].z == doctest::Approx(1.5));
  // body 1: R_z(30 deg) and S = 2 I (to float32 rounding); p = 0.5 t + R c
  const rot::Q q1 = b.q[1];
  CHECK(q1.w == doctest::Approx(std::cos(M_PI / 12)).epsilon(1e-7));
  CHECK(q1.z == doctest::Approx(std::sin(M_PI / 12)).epsilon(1e-7));
  CHECK(b.S[1][0] == doctest::Approx(2.0).epsilon(1e-7));
  CHECK(b.p[1].x == doctest::Approx(1.5 + 0.1 * c30).epsilon(1e-7));
  CHECK(b.p[1].y == doctest::Approx(-0.5 + 0.1 * s30).epsilon(1e-7));
  // rounding bounds: dtheta = 3 * 1/2 ulp32(max |A|) / sigma_min(S); dp = scale |1/2 ulp32(t)| + |c| dtheta
  const double dth = 3.0 * half_ulp(2.0f) / 2.0;
  CHECK(b.dtheta_round[1] == doctest::Approx(dth).epsilon(1e-6));
  const double dp = 0.5 * std::sqrt(half_ulp(3.0f) * half_ulp(3.0f) + half_ulp(1.0f) * half_ulp(1.0f) +
                                    half_ulp(0.25f) * half_ulp(0.25f));
  CHECK(b.dp_round[1] == doctest::Approx(dp + 0.1 * dth).epsilon(1e-9));
}

TEST_CASE("decode: GEO-SCALEANIM when S departs from the export frame by more than 1e-6; S_ref is required") {
  const mo::TimeMap tm(settings(24, 1.0f, 1, 48));
  mo::MotionScene scene = unit_bodies(1);
  const std::array<float, 12> same{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  const std::array<float, 12> grown{1.001f, 0, 0, 0, 0, 1.001f, 0, 0, 0, 0, 1.001f, 0};
  const std::array<float, 12> twice{2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0};
  CHECK_NOTHROW(mo::decode(raw_of(tm, {2}, {0.0f}, {same}), tm, scene));
  CHECK_THROWS_WITH_AS(mo::decode(raw_of(tm, {2}, {0.0f}, {grown}), tm, scene), doctest::Contains("GEO-SCALEANIM"),
                       mo::MotionError);
  // a mesh body or a light carrier without S_ref is refused before any answer is read, so a keyed scale 1 -> 2 can
  // never pass unchecked; a rig needs none (its scale moves no ray), and one that has it is checked
  scene.bodies[0].S_ref.reset();
  CHECK_THROWS_WITH_AS(mo::decode(raw_of(tm, {2}, {0.0f}, {twice}), tm, scene),
                       doctest::Contains("body column 0 (a mesh body) has no S_ref"), std::invalid_argument);
  CHECK_THROWS_AS(mo::decode(raw_of(tm, {2}, {0.0f}, {same}), tm, scene), std::invalid_argument);
  scene.bodies[0].role = mo::Role::kCarrier;
  CHECK_THROWS_WITH_AS(mo::decode(raw_of(tm, {2}, {0.0f}, {same}), tm, scene),
                       doctest::Contains("body column 0 (a light carrier) has no S_ref"), std::invalid_argument);
  scene.bodies[0].S_ref = std::array<double, 9>{0, 0, 0, 0, 0, 0, 0, 0, 0};
  CHECK_THROWS_WITH_AS(mo::decode(raw_of(tm, {2}, {0.0f}, {same}), tm, scene),
                       doctest::Contains("non-finite or zero S_ref"), std::invalid_argument);
  scene.bodies[0].role = mo::Role::kRig;
  scene.bodies[0].S_ref.reset();
  CHECK_NOTHROW(mo::decode(raw_of(tm, {2}, {0.0f}, {twice}), tm, scene));
  scene.bodies[0].S_ref = kUnitS;
  CHECK_THROWS_WITH_AS(mo::decode(raw_of(tm, {2}, {0.0f}, {twice}), tm, scene), doctest::Contains("GEO-SCALEANIM"),
                       mo::MotionError);
  // the error model refuses the same scene at construction, before any oracle call
  mo::MotionScene no_ref = unit_bodies(2);
  no_ref.bodies[0].role = mo::Role::kRig;
  no_ref.cameras.push_back({0, {0.0, 0.0, 0.0}, 1000.0, 0.0});
  CHECK_NOTHROW(mo::ErrorModel{no_ref});
  no_ref.bodies[1].S_ref.reset();
  CHECK_THROWS_WITH_AS(mo::ErrorModel{no_ref}, doctest::Contains("body column 1 (a mesh body) has no S_ref"),
                       std::invalid_argument);
}

TEST_CASE("StaticDistance is the exact minimum over every triangle") {
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> u(-5.0, 5.0), small(-0.3, 0.3);
  std::vector<double> v;
  std::vector<uint32_t> idx;
  for (uint32_t t = 0; t < 700; ++t) {
    const double cx = u(rng), cy = u(rng), cz = u(rng);
    for (int k = 0; k < 3; ++k) {
      v.push_back(cx + small(rng));
      v.push_back(cy + small(rng));
      v.push_back(cz + small(rng));
    }
    idx.insert(idx.end(), {3 * t, 3 * t + 1, 3 * t + 2});
  }
  const mo::StaticDistance sd(v, idx);
  for (int i = 0; i < 300; ++i) {
    const rot::V x{2 * u(rng), 2 * u(rng), 2 * u(rng)};
    double brute = std::numeric_limits<double>::infinity();
    for (size_t t = 0; t < idx.size() / 3; ++t) {
      auto at = [&](uint32_t k) { return rot::V{v[3 * k], v[3 * k + 1], v[3 * k + 2]}; };
      brute = std::min(brute, mo::point_triangle_distance(x, at(idx[3 * t]), at(idx[3 * t + 1]), at(idx[3 * t + 2])));
    }
    CHECK(sd(x) == brute);
  }
  CHECK(std::isinf(mo::StaticDistance()({0, 0, 0})));
  // the closest point of a triangle by region: a vertex, an edge, the face
  const rot::V a{0, 0, 0}, b{1, 0, 0}, c{0, 1, 0};
  CHECK(mo::point_triangle_distance({-1, -1, 0}, a, b, c) == doctest::Approx(std::sqrt(2.0)));
  CHECK(mo::point_triangle_distance({0.5, -2, 0}, a, b, c) == doctest::Approx(2.0));
  CHECK(mo::point_triangle_distance({0.2, 0.2, 3}, a, b, c) == doctest::Approx(3.0));
  CHECK(mo::point_triangle_distance({1, 1, 0}, a, b, c) == doctest::Approx(std::sqrt(0.5)));
  // degenerate triangles are their segment (or point), with any vertex repeated, also inside StaticDistance
  const rot::V o{0, 0, 0}, e{1, 0, 0}, p{0.5, 0, 1};
  for (const auto& [x, y, z] : {std::array<rot::V, 3>{o, o, e}, std::array<rot::V, 3>{o, e, o},
                                std::array<rot::V, 3>{e, o, o}, std::array<rot::V, 3>{o, e, e}}) {
    CHECK(mo::point_triangle_distance(p, x, y, z) == 1.0);
    CHECK(mo::point_triangle_distance({2, 0, 0}, x, y, z) == 1.0);
    CHECK(mo::point_triangle_distance({-1, 0, 0}, x, y, z) == 1.0);
  }
  CHECK(mo::point_triangle_distance({0, 3, 4}, o, o, o) == 5.0);
  const std::vector<double> sliver{0, 0, 0, 1, 0, 0};
  const std::vector<uint32_t> repeated{0, 0, 1};
  CHECK(mo::StaticDistance(sliver, repeated)(p) == 1.0);
}

TEST_CASE("angular_gain: pinhole f (1 + r_max^2); equisolid at 180 deg is finite, 1.414 f") {
  balla::render::Camera pin;
  pin.width = 1920;
  pin.height = 1080;
  pin.fx = pin.fy = 50.0 / 36.0 * 1920.0;
  pin.cx = 959.5;
  pin.cy = 539.5;
  pin.dist = {0, 0, 0, 0, 0};
  const double r2 = (960.0 / pin.fx) * (960.0 / pin.fx) + (540.0 / pin.fy) * (540.0 / pin.fy);
  CHECK(mo::angular_gain(pin) == doctest::Approx(pin.fx * (1.0 + r2)).epsilon(1e-9));
  // equisolid r = 2 f sin(theta / 2) as Kannala-Brandt (its Taylor coefficients, DS d10); a 2f x 2f image puts the
  // corners at theta = 90 degrees
  balla::render::Camera eq;
  eq.model = balla::render::CameraModelKind::kKannalaBrandt;
  eq.width = eq.height = 373;
  eq.fx = eq.fy = 186.5;
  eq.cx = eq.cy = 186.0;
  eq.dist = {-1.0 / 24.0, 1.0 / 1920.0, -1.0 / 322560.0, 1.0 / 92897280.0};
  const double g = mo::angular_gain(eq);
  CHECK(std::isfinite(g));
  CHECK(g / eq.fx == doctest::Approx(std::sqrt(2.0)).epsilon(2e-4));
}

TEST_CASE("ErrorModel: body, rig and carrier weights of blender.md 4.7.2") {
  mo::MotionScene scene = unit_bodies(3);
  scene.bodies[0].role = mo::Role::kRig;
  scene.bodies[1].role = mo::Role::kBody;
  scene.bodies[1].rho_m = 0.5;
  scene.bodies[2].role = mo::Role::kCarrier;
  scene.bodies[2].lights.push_back({{0.1, 0.0, 0.0}, false, false, true});
  scene.cameras.push_back({0, {0.0, 0.0, 0.0}, 1000.0, 0.0});
  // a static plane z = -0.3 under the camera
  scene.static_vertices = {-50, -50, -0.3, 50, -50, -0.3, 0, 50, -0.3};
  scene.static_indices = {0, 1, 2};
  scene.scene_box = {-1, -1, -1, 1, 1, 1};
  const mo::TimeMap tm(settings(24, 1.0f, 1, 48));
  const std::array<float, 12> rig{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  const std::array<float, 12> body{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2};
  const std::array<float, 12> carrier{1, 0, 0, 0.9f, 0, 1, 0, 0, 0, 0, 1, 0};
  mo::PoseBatch b = mo::decode(raw_of(tm, {4}, {0.0f}, {rig, body, carrier}), tm, scene);
  const mo::ErrorModel model(scene);
  model.weigh(b);
  CHECK(model.gain() == 1000.0);
  // the body: d_min = 2 - 0.5
  CHECK(b.w_pos[1] == doctest::Approx(1000.0 / 1.5));
  CHECK(b.w_rot[1] == doctest::Approx(1000.0 * 0.5 / 1.5));
  // the rig: d_scene = min(static 0.3, body 1.5), l = 0
  CHECK(b.w_pos[0] == doctest::Approx(1000.0 / 0.3));
  CHECK(b.w_rot[0] == doctest::Approx(1000.0));
  // the carrier: light at (1, 0, 0); D_max to the farthest box corner; d_occ = min(static 0.3, body sqrt(5) - 0.5);
  // d_recv: the camera is inside the box, so the 5 cm floor; s = 0 (meshless, not a spot)
  const double dmax = std::sqrt(4.0 + 1.0 + 1.0), lambda = dmax / 0.3;
  CHECK(b.w_pos[2] == doctest::Approx(1000.0 * (1.0 + lambda) / 0.05));
  CHECK(b.w_rot[2] == doctest::Approx(1000.0 * (1.0 + lambda) * 0.1 / 0.05));
  CHECK(b.light_near[2] == 0);
  CHECK(b.floor1[1] == doctest::Approx(b.w_pos[1] * b.dp_round[1] + b.w_rot[1] * b.dtheta_round[1]));
  CHECK_THROWS_AS(mo::ErrorModel([] {
                    mo::MotionScene s;
                    s.bodies.resize(1);
                    s.bodies[0].role = mo::Role::kRig;
                    return s;
                  }()),
                  std::invalid_argument);
}

TEST_CASE("ErrorModel: a directional light turns every shadow, on a meshless carrier too") {
  // a Sun lamp object (a meshless carrier, its light at the origin) outside and at the centre of the scene box; the
  // camera inside the box puts d_recv at the 5 cm floor
  mo::MotionScene scene = unit_bodies(3);
  scene.bodies[0].role = mo::Role::kRig;
  for (const size_t b : {size_t{1}, size_t{2}}) {
    scene.bodies[b].role = mo::Role::kCarrier;
    scene.bodies[b].lights.push_back({{0.0, 0.0, 0.0}, false, true, true});
  }
  scene.cameras.push_back({0, {0.0, 0.0, 0.0}, 1000.0, 0.0});
  scene.scene_box = {-1, -1, -1, 1, 1, 1};
  const mo::TimeMap tm(settings(24, 1.0f, 1, 48));
  const std::array<float, 12> rig{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  const std::array<float, 12> outside{1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0};
  const std::array<float, 12> centre{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  mo::PoseBatch b = mo::decode(raw_of(tm, {4}, {0.0f}, {rig, outside, centre}), tm, scene);
  mo::ErrorModel(scene).weigh(b);
  // s = 1: w_rot = G D_max / d_recv, D_max the farthest box corner from (5, 0, 0), or the diagonal from the centre
  CHECK(b.w_rot[1] == doctest::Approx(1000.0 * std::sqrt(36.0 + 1.0 + 1.0) / 0.05));
  CHECK(b.w_rot[2] == doctest::Approx(1000.0 * std::sqrt(12.0) / 0.05));
  CHECK(b.w_pos[1] == doctest::Approx(1000.0 / 0.05));  // no occluder near: lambda = 0
  // a point light on the same meshless carrier: s = 0, its offset 0, so its rotation moves nothing
  scene.bodies[1].lights[0].directional = false;
  mo::PoseBatch p = mo::decode(raw_of(tm, {4}, {0.0f}, {rig, outside, centre}), tm, scene);
  mo::ErrorModel(scene).weigh(p);
  CHECK(p.w_rot[1] == 0.0);
}

TEST_CASE("angular_gain: radtan with distortion is f max|J_D| (1 + r_img^2), J_D by finite differences") {
  // fx = 2 fy and a pincushion with tangential terms: the gain peaks at the corners, where J_D's off-diagonal term is
  // large, so scaling J_D's second row by fx instead of fy would show
  balla::render::Camera cam;
  cam.width = 640;
  cam.height = 480;
  cam.fx = 900.0;
  cam.fy = 450.0;
  cam.cx = 319.5;
  cam.cy = 239.5;
  const double k1 = 0.3, k2 = 0.02, p1 = 0.01, p2 = -0.008;
  cam.dist = {k1, k2, p1, p2, 0.0};
  const double G = mo::angular_gain(cam);
  // OpenCV's radial-tangential map, written out here
  auto distort = [&](double x, double y, double* xd, double* yd) {
    const double r2 = x * x + y * y, f = 1.0 + r2 * (k1 + r2 * k2);
    *xd = x * f + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    *yd = y * f + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
  };
  balla::camera::RadtanCamera<double> rc{cam.fx, cam.fy, cam.cx, cam.cy, k1, k2, p1, p2, 0.0, 0.0, 0.0};
  const balla::camera::RadtanDomain dom = balla::camera::radtan_injective_domain(rc, cam.width, cam.height, 0.5);
  rc.r_lim = dom.r_lim;
  // sigma_max of [[a, b], [c, d]]
  auto sigma_max = [](double a, double b, double c, double d) {
    const double fro = a * a + b * b + c * c + d * d, det = a * d - b * c;
    return std::sqrt(0.5 * (fro + std::sqrt(std::max(0.0, fro * fro - 4.0 * det * det))));
  };
  double smax = 0.0, smax_fx = 0.0, true_gain = 0.0;
  constexpr double h = 1e-6, dth = 1e-6;
  for (int a = 0; a <= 32; ++a) {
    for (int b = 0; b <= 24; ++b) {
      const double u = -0.5 + (cam.width - 1 + 1.0) * a / 32.0, v = -0.5 + (cam.height - 1 + 1.0) * b / 24.0;
      balla::Vec3<double> ray{0.0, 0.0, 1.0};
      REQUIRE(balla::camera::unproject(rc, balla::Vec2<double>{u, v}, &ray));
      double xp, yp, xm, ym, j[2][2];
      distort(ray.x + h, ray.y, &xp, &yp);
      distort(ray.x - h, ray.y, &xm, &ym);
      j[0][0] = (xp - xm) / (2 * h);
      j[1][0] = (yp - ym) / (2 * h);
      distort(ray.x, ray.y + h, &xp, &yp);
      distort(ray.x, ray.y - h, &xm, &ym);
      j[0][1] = (xp - xm) / (2 * h);
      j[1][1] = (yp - ym) / (2 * h);
      smax = std::max(smax, sigma_max(cam.fx * j[0][0], cam.fx * j[0][1], cam.fy * j[1][0], cam.fy * j[1][1]));
      smax_fx = std::max(smax_fx, sigma_max(cam.fx * j[0][0], cam.fx * j[0][1], cam.fx * j[1][0], cam.fy * j[1][1]));
      // the pixel motion per radian of ray rotation, in 16 directions: never above G
      const rot::V d = (1.0 / std::sqrt(ray.x * ray.x + ray.y * ray.y + 1.0)) * rot::V{ray.x, ray.y, 1.0};
      const rot::V e1 = (1.0 / norm(cross(d, rot::V{0, 1, 0}))) * cross(d, rot::V{0, 1, 0}), e2 = cross(d, e1);
      double x0, y0;
      distort(ray.x, ray.y, &x0, &y0);
      for (int k = 0; k < 16; ++k) {
        const double phi = M_PI * k / 8.0;
        const rot::V w = std::cos(dth) * d + std::sin(dth) * (std::cos(phi) * e1 + std::sin(phi) * e2);
        double x1, y1;
        distort(w.x / w.z, w.y / w.z, &x1, &y1);
        true_gain = std::max(true_gain, std::hypot(cam.fx * (x1 - x0), cam.fy * (y1 - y0)) / dth);
      }
    }
  }
  const double r2 = dom.r_img * dom.r_img;
  CHECK(G == doctest::Approx(smax * (1.0 + r2)).epsilon(1e-8));
  CHECK(std::fabs(smax_fx * (1.0 + r2) / G - 1.0) > 1e-3);  // fx in J_D's second row would show
  CHECK(true_gain <= G * (1.0 + 1e-6));
  CHECK(true_gain >= 0.8 * G);  // the bound is tight: 0.91 G here
  MESSAGE("radtan gain: G = " << G << " px/rad, the measured maximum " << true_gain << " px/rad, fx-row variant "
                              << smax_fx * (1.0 + r2));
}

TEST_CASE("ErrorModel: a rig's camera offset, LGT-MOVE-NEAR and a spot on a mesh body") {
  // the rig turned 90 deg about z with its camera 0.1 m along the rig's x: the camera centre is o = (0, 0.1, 0); a
  // mesh body (rho 0.5) at z = 1.625 carries a shadow-casting spot 0.2 m along x; a meshless carrier's shadow-casting
  // point light sits 3.125 cm from that body's bound; a static plane z = -0.3; the scene box z in [5, 7] is 5 m from o
  mo::MotionScene scene = unit_bodies(3);
  scene.bodies[0].role = mo::Role::kRig;
  scene.bodies[1].rho_m = 0.5;
  scene.bodies[1].lights.push_back({{0.2, 0.0, 0.0}, true, false, true});
  scene.bodies[2].role = mo::Role::kCarrier;
  scene.bodies[2].lights.push_back({{0.0, 0.0, 0.0}, false, false, true});
  scene.cameras.push_back({0, {0.1, 0.0, 0.0}, 1000.0, 0.0});
  scene.static_vertices = {-50, -50, -0.3, 50, -50, -0.3, 0, 50, -0.3};
  scene.static_indices = {0, 1, 2};
  scene.scene_box = {-1, -1, 5, 1, 1, 7};
  const mo::TimeMap tm(settings(24, 1.0f, 1, 48));
  const std::array<float, 12> rig{0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0};
  const std::array<float, 12> body{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1.625f};
  const std::array<float, 12> carrier{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2.15625f};
  mo::PoseBatch b = mo::decode(raw_of(tm, {4}, {0.0f}, {rig, body, carrier}), tm, scene);
  mo::ErrorModel(scene).weigh(b);
  const double G = 1000.0, d_recv = 5.0;
  auto box_far = [](double x, double y, double z) {
    const double dx = std::max(std::fabs(x + 1), std::fabs(x - 1)), dy = std::max(std::fabs(y + 1), std::fabs(y - 1)),
                 dz = std::max(std::fabs(z - 5), std::fabs(z - 7));
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };
  // the rig: d_scene = min(static 0.3, body |(0, -0.1, 1.625)| - 0.5) = 0.3, l = 0.1: w_rot = G (1 + l / d)
  CHECK(b.w_pos[0] == doctest::Approx(G / 0.3));
  CHECK(b.w_rot[0] == doctest::Approx(G * (1.0 + 0.1 / 0.3)));
  // the mesh body and its spot (s = 1): w_pos stays the body's G / d_min, w_rot is the spot's
  const double d_min = std::sqrt(0.01 + 1.625 * 1.625) - 0.5;
  const double dmax_spot = box_far(0.2, 0.0, 1.625), lambda_spot = dmax_spot / (1.625 + 0.3);
  const double wp_light = G * (1.0 + lambda_spot) / d_recv;
  const double wr_light = G * ((1.0 + lambda_spot) * 0.2 + dmax_spot) / d_recv;
  REQUIRE(G / d_min > wp_light);
  REQUIRE(wr_light > G * 0.5 / d_min);
  CHECK(b.w_pos[1] == doctest::Approx(G / d_min));
  CHECK(b.w_rot[1] == doctest::Approx(wr_light));
  CHECK(b.light_near[1] == 0);
  // the carrier's light 0.53125 - 0.5 m from the body's bound: LGT-MOVE-NEAR, lambda = D_max / 5 cm; s = 0, offset 0
  const double dmax_pt = box_far(0.0, 0.0, 2.15625);
  CHECK(b.light_near[2] == 1);
  CHECK(b.w_pos[2] == doctest::Approx(G * (1.0 + dmax_pt / 0.05) / d_recv));
  CHECK(b.w_rot[2] == 0.0);
  CHECK(b.floor1[2] == doctest::Approx(b.w_pos[2] * b.dp_round[2]));
  // the light carried farther than 5 cm from every body: no warning
  const std::array<float, 12> away{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2.25f};
  mo::PoseBatch c = mo::decode(raw_of(tm, {4}, {0.0f}, {rig, body, away}), tm, scene);
  mo::ErrorModel(scene).weigh(c);
  CHECK(c.light_near[2] == 0);
  CHECK(c.w_pos[2] == doctest::Approx(G * (1.0 + box_far(0.0, 0.0, 2.25) / 0.125) / d_recv));
}

TEST_CASE("base_requests: the h0 grid with half the aliasing density, the span ends, the keys, the pre-key sample") {
  // the frame-aligned grid of `per_frame` samples per frame strictly inside (-2W, t_end), each one requested
  auto grid = [](const mo::TimeMap& tm, const std::vector<int64_t>& req, int64_t per_frame) {
    size_t n = 0;
    for (int64_t j = -4 * per_frame; j < 30 * per_frame; ++j) {
      const int64_t t = tm.frames_to_ns(j, per_frame);
      if (t <= -2 * W || t >= tm.t_end_ns()) continue;
      ++n;
      CHECK(std::binary_search(req.begin(), req.end(), t));
    }
    return n;
  };
  const mo::TimeMap tm(settings(24, 1.0f, 1, 25));
  const std::vector<mo::DeclaredKey> keys{{tm.to_ns(13.0f), mo::Continuity::kStep, true, 1e-4},
                                          {tm.to_ns(7.0f), mo::Continuity::kC0, true, 1e-4},
                                          {tm.to_ns(9.1f), mo::Continuity::kC0, true, 1e-4}};
  const std::vector<int64_t> r = mo::base_requests(tm, -2 * W, tm.t_end_ns(), keys, mo::SampleOptions{});
  CHECK(r.front() == -2 * W);
  CHECK(r.back() == tm.t_end_ns());
  CHECK(std::is_sorted(r.begin(), r.end()));
  CHECK(std::adjacent_find(r.begin(), r.end()) == r.end());
  // 24 fps: h0 = 10.4 ms (1/96 s) and half the 8 per frame of the aliasing pre-pass are both 4 per frame; the keys at
  // frames 13 and 7 lie on that grid, the one at 9.1 does not
  const size_t n4 = grid(tm, r, 4);
  CHECK(r.size() == n4 + 2 + 1 + 1);  // the grid, the span ends, the key at 9.1 and the pre-key sample
  CHECK(std::binary_search(r.begin(), r.end(), tm.to_ns(9.1f)));
  // key - 1e-4 frames - 2 q(13): 4166.67 ns + 2 x 19.87 ns
  CHECK(std::binary_search(r.begin(), r.end(), tm.to_ns(13.0f) - 4167 - 40));
  // a request at a frame eighth realizes exactly
  int32_t F = 0;
  float s = 0.0f;
  tm.request(tm.frames_to_ns(37, 8), &F, &s);
  CHECK(mo::TimeMap::realized(F, s) == 1.0f + 37.0f / 8.0f);
  // 12 fps: h0 = 10.4 ms sets 8 per frame; 60 fps: h0 = P / 2, and the pre-pass asks for 4 per frame
  const mo::TimeMap t12(settings(12, 1.0f, 1, 25)), t60(settings(60, 1.0f, 1, 25));
  const std::vector<int64_t> r12 = mo::base_requests(t12, -2 * W, t12.t_end_ns(), {}, mo::SampleOptions{});
  const std::vector<int64_t> r60 = mo::base_requests(t60, -2 * W, t60.t_end_ns(), {}, mo::SampleOptions{});
  CHECK(r12.size() == grid(t12, r12, 8) + 2);
  CHECK(r60.size() == grid(t60, r60, 4) + 2);
  mo::SampleOptions no_alias;
  no_alias.alias_per_frame = 0;
  const std::vector<int64_t> r60b = mo::base_requests(t60, -2 * W, t60.t_end_ns(), {}, no_alias);
  CHECK(r60b.size() == grid(t60, r60b, 2) + 2);
  // an explicit h0 finer than both sets the density: 1 ms at 24 fps is 42 per frame
  mo::SampleOptions fine;
  fine.h0_ns = 1000000;
  CHECK(mo::base_requests(tm, 0, tm.to_ns(2.0f), {}, fine).size() == 42 + 1);
}

TEST_CASE("base_requests: a base grid above max_base_samples is refused before it is allocated") {
  // 32767 fps at fps_base 1e-5 (Blender's smallest): P = 0.305 ns, about 3.3e8 base samples over [-2W, W]
  const mo::TimeMap tiny(settings(32767, 1e-5f, 1, 250));
  CHECK(tiny.period_ns() < 1.0);
  CHECK_THROWS_WITH_AS(mo::base_requests(tiny, -2 * W, tiny.t_end_ns(), {}, mo::SampleOptions{}),
                       doctest::Contains("a frame period of 0.305"), std::invalid_argument);
  CHECK_THROWS_AS(mo::BodyPipeline(0, mo::Mode::kKeyed, tiny, {}, mo::SampleOptions{}), std::invalid_argument);
  CHECK(mo::h0_ns(tiny, mo::SampleOptions{}) == 1);  // never 0 (the split filter's cap is 2 h0)
  // the cap counts before allocating: 24 fps over two frames is 9 samples, estimated as 10
  const mo::TimeMap tm(settings(24, 1.0f, 1, 25));
  mo::SampleOptions cap;
  CHECK(mo::base_requests(tm, 0, tm.to_ns(3.0f), {}, cap).size() == 9);
  cap.max_base_samples = 10;
  CHECK(mo::base_requests(tm, 0, tm.to_ns(3.0f), {}, cap).size() == 9);
  cap.max_base_samples = 8;
  CHECK_THROWS_AS(mo::base_requests(tm, 0, tm.to_ns(3.0f), {}, cap), std::invalid_argument);
  cap.max_base_samples = 0;
  CHECK_THROWS_AS(mo::BodyPipeline(0, mo::Mode::kKeyed, tm, {}, cap), std::invalid_argument);
}

TEST_CASE("midpoint_test: exact on cubics, refused across a kink, the floor scales with the stencil gain") {
  mo::Track fit, mid;
  auto push = [](mo::Track& t, int64_t ns, double x, double f1) {
    t.t_ns.push_back(ns);
    t.q.push_back(balla::so3::exp(rot::V{0.0, 0.0, 0.3 * static_cast<double>(ns) * 1e-9}));
    t.p.push_back({x, 0.0, 0.0});
    t.w_pos.push_back(1000.0);
    t.w_rot.push_back(1000.0);
    t.floor1.push_back(f1);
  };
  auto cubic = [](double s) { return 0.3 + s * (0.2 - s * (1.5 - 4.0 * s)); };
  for (int k = 0; k < 6; ++k) push(fit, k * 10000000, cubic(k * 0.01), 0.0);
  push(mid, 25000000, cubic(0.025), 0.0);
  mo::MidpointTest r = mo::midpoint_test(fit, 2, mid, 0, 0.01);
  CHECK(r.accepted);
  CHECK(r.residual_px < 1e-9);
  // a kink at 20 ms: the centred and left stencils straddle it, the right one does not
  mo::Track kink, km;
  auto kinked = [](double s) { return s < 0.02 ? 0.5 * s : 0.01 + 2.0 * (s - 0.02); };
  for (int k = 0; k < 7; ++k) push(kink, k * 10000000, kinked(k * 0.01), 0.0);
  push(km, 15000000, kinked(0.015), 0.0);
  CHECK_FALSE(mo::midpoint_test(kink, 1, km, 0, 0.01).accepted);
  // an error e accepted only with the floor: e = 1000 * 1e-5 = 0.01 px > eps/4; floor1 = 0.004 at every sample, the
  // centred stencil's g = 2.25 -> floor 0.009, threshold 0.0115
  mo::Track noisy, nm;
  for (int k = 0; k < 6; ++k) push(noisy, k * 10000000, 0.1, 0.004);
  push(nm, 25000000, 0.1 + 1e-5, 0.004);
  r = mo::midpoint_test(noisy, 2, nm, 0, 0.01);
  CHECK(r.accepted);
  CHECK(r.floor_px == doctest::Approx(2.25 * 0.004));
  nm.floor1[0] = 0.0;
  for (double& f : noisy.floor1) f = 0.0;
  CHECK_FALSE(mo::midpoint_test(noisy, 2, nm, 0, 0.01).accepted);
}
