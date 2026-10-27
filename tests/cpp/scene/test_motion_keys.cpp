// Declared keys (src/scene/motion/keys.h; docs/design/phase3/blender.md §4.7.4): the code-table rules, the F-modifier
// breaks, the NLA mapping with blends, repeats and reversal, plateaus scaled by the strip and by framelen, the key
// chain, and the exact frame-to-ns conversion they use. The expected NLA, BOUNCE, STEPPED and Cycles times are the
// BL-7 probe's Blender measurements (keys.h provenance).
#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "scene/motion/keys.h"
#include "scene/motion/time_map.h"

namespace mo = balla::scene::motion;
using balla::scene::Tree;
using C = mo::Continuity;

namespace {

mo::TimeMap time24(int32_t f0, int32_t f1, int32_t map_old = 100, int32_t map_new = 100) {
  mo::TimeSettings s;
  s.fps = 24;
  s.frame_start = f0;
  s.frame_end = f1;
  s.frame_map_old = map_old;
  s.frame_map_new = map_new;
  return mo::TimeMap(s);
}

// One key row: frame, value, interpolation, easing, left and right handle types, collinear.
void key(mo::KeyCurve& c, double f, double v, mo::Interp ip, mo::Handle h = mo::Handle::kAutoClamped,
         mo::Easing e = mo::Easing::kAuto, bool collinear = true) {
  c.keys.insert(c.keys.end(), {f, v, static_cast<double>(ip), static_cast<double>(e), static_cast<double>(h),
                               static_cast<double>(h), collinear ? 1.0 : 0.0});
}

mo::KeyCurve curve(const std::vector<std::pair<double, double>>& fv, mo::Interp ip,
                   mo::Handle h = mo::Handle::kAutoClamped, mo::Easing e = mo::Easing::kAuto) {
  mo::KeyCurve c;
  c.data_path = "location";
  for (const auto& [f, v] : fv) key(c, f, v, ip, h, e);
  return c;
}

struct F {
  double frame;
  C hint;
  bool plateau;
};

void check_features(const std::vector<mo::KeyFeature>& got, const std::vector<F>& want) {
  REQUIRE(got.size() == want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    INFO("feature " << i << " at " << got[i].frame);
    CHECK(got[i].frame == doctest::Approx(want[i].frame).epsilon(1e-12));
    CHECK(got[i].hint == want[i].hint);
    CHECK(got[i].plateau == want[i].plateau);
  }
}

}  // namespace

TEST_CASE("TimeMap::frame_to_ns is to_ns for float32 frames, exact on doubles") {
  for (const auto& tm : {time24(1, 100), time24(-3, 50), time24(0, 10, 100, 200)}) {
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<float> u(-500.0f, 5000.0f);
    for (int i = 0; i < 2000; ++i) {
      const float c = u(rng);
      CHECK(tm.frame_to_ns(static_cast<double>(c)) == tm.to_ns(c));
    }
    CHECK(tm.frame_to_ns(tm.frame_ref()) == 0);
  }
  mo::TimeSettings s;  // 30000/1001 fps: frame 1 + 1/3 is 1e9 * 1001 / 90000 ns
  s.fps = 30;
  s.fps_base_bits = std::bit_cast<uint32_t>(1.001f);
  s.fps_base_snap = true;
  s.frame_start = 1;
  s.frame_end = 10;
  const mo::TimeMap ntsc(s);
  CHECK(ntsc.frame_to_ns(4.0) == 100100000);
  CHECK(time24(1, 10).frame_to_ns(1.0 + 1.0 / 3.0) == 13888889);
}

TEST_CASE("curve_features: the code tables (CONSTANT, LINEAR, BEZIER handles, extrapolation)") {
  using I = mo::Interp;
  // CONSTANT: C0 where the hold starts, a step (plateau) at the next key
  check_features(mo::curve_features(curve({{1, 0}, {13, 1}, {25, 0}}, I::kConstant), -1e9, 1e9),
                 {{1, C::kC0, true}, {13, C::kStep, true}, {25, C::kStep, true}});
  // LINEAR: C0 at every key; under LINEAR extrapolation the end keys continue their slope
  check_features(mo::curve_features(curve({{1, 0}, {13, 1}, {25, 0}}, I::kLinear), -1e9, 1e9),
                 {{1, C::kC0, true}, {13, C::kC0, true}, {25, C::kC0, true}});
  mo::KeyCurve lin = curve({{1, 0}, {13, 1}, {25, 0}}, I::kLinear);
  lin.extrapolation = 1;
  check_features(mo::curve_features(lin, -1e9, 1e9), {{1, C::kC1, true}, {13, C::kC0, true}, {25, C::kC1, true}});
  // BEZIER: AUTO_CLAMPED interior keys are C1 hints; the ends hold (CONSTANT extrapolation): C0
  check_features(mo::curve_features(curve({{1, 0}, {13, 1}, {25, 0}}, I::kBezier), -1e9, 1e9),
                 {{1, C::kC0, true}, {13, C::kC1, true}, {25, C::kC0, true}});
  mo::KeyCurve bez = curve({{1, 0}, {13, 1}, {25, 0}}, I::kBezier);
  bez.extrapolation = 1;
  check_features(mo::curve_features(bez, -1e9, 1e9), {{1, C::kC1, true}, {13, C::kC1, true}, {25, C::kC1, true}});
  for (const auto h : {mo::Handle::kVector, mo::Handle::kFree}) {
    check_features(mo::curve_features(curve({{1, 0}, {13, 1}, {25, 0}}, I::kBezier, h), -1e9, 1e9),
                   {{1, C::kC0, true}, {13, C::kC0, true}, {25, C::kC0, true}});
  }
  mo::KeyCurve broken;  // ALIGNED but not collinear (an edited handle): C0
  broken.data_path = "location";
  key(broken, 1, 0, I::kBezier);
  key(broken, 13, 1, I::kBezier, mo::Handle::kAligned, mo::Easing::kAuto, false);
  key(broken, 25, 0, I::kBezier);
  CHECK(mo::curve_features(broken, -1e9, 1e9)[1].hint == C::kC0);
  // mixed: a LINEAR segment into a BEZIER one kinks at the shared key; the window clips
  mo::KeyCurve mix;
  mix.data_path = "location";
  key(mix, 1, 0, I::kLinear);
  key(mix, 13, 1, I::kBezier);
  key(mix, 25, 0, I::kBezier);
  key(mix, 37, 1, I::kBezier);
  check_features(mo::curve_features(mix, 5, 30), {{13, C::kC0, true}, {25, C::kC1, true}});
}

TEST_CASE("curve_features: easing kinks at the segment ends, BOUNCE at Penner's breaks (BL-7 probe fractions)") {
  using I = mo::Interp;
  using E = mo::Easing;
  const double out[3] = {1 / 2.75, 2 / 2.75, 2.5 / 2.75};
  auto bounce = [&](E e) { return mo::curve_features(curve({{1, 0.3}, {49, 0}}, I::kBounce, mo::Handle::kAutoClamped,
                                                           e), -1e9, 1e9); };
  for (const E e : {E::kAuto, E::kEaseOut}) {  // AUTO breaks as EASE_OUT does (probe: 0.36363, 0.72727, 0.90908)
    check_features(bounce(e), {{1, C::kC0, true}, {1 + 48 * out[0], C::kC0, false}, {1 + 48 * out[1], C::kC0, false},
                               {1 + 48 * out[2], C::kC0, false}, {49, C::kC0, true}});
  }
  check_features(bounce(E::kEaseIn), {{1, C::kC0, true}, {1 + 48 * (1 - out[2]), C::kC0, false},
                                      {1 + 48 * (1 - out[1]), C::kC0, false}, {1 + 48 * (1 - out[0]), C::kC0, false},
                                      {49, C::kC0, true}});
  const std::vector<mo::KeyFeature> io = bounce(E::kEaseInOut);
  const double probe[6] = {0.04544, 0.13635, 0.31817, 0.68181, 0.86362, 0.95454};
  REQUIRE(io.size() == 8);
  for (size_t k = 0; k < 6; ++k) CHECK((io[k + 1].frame - 1) / 48 == doctest::Approx(probe[k]).epsilon(2e-4));
  // every other easing: C0 at both ends only
  check_features(mo::curve_features(curve({{1, 0}, {13, 1}}, I::kSine), -1e9, 1e9),
                 {{1, C::kC0, true}, {13, C::kC0, true}});
}

TEST_CASE("curve_features: STEPPED, Cycles, restricted ranges, ENVELOPE and malformed tables") {
  using I = mo::Interp;
  mo::KeyCurve st = curve({{1, 1}, {25, 25}}, I::kLinear);
  mo::KeyModifier m;
  m.type = "STEPPED";
  m.fields["frame_step"] = Tree::real(3.0);
  m.fields["frame_offset"] = Tree::real(1.0);
  m.fields["use_frame_start"] = Tree::boolean(true);
  m.fields["frame_start"] = Tree::real(5.0);
  st.modifiers.push_back(m);
  // steps exactly at 1 + 3k from frame 5 on (probe: 6.99999 -> 4, 7.0 -> 7), no plateau; the keys stay
  std::vector<F> want{{1, C::kC0, true}};
  for (double f = 7; f <= 20; f += 3) want.push_back({f, C::kStep, false});
  check_features(mo::curve_features(st, 0, 20), want);

  // Cycles REPEAT after keys 1 (0), 5 (1), 9 (0.5): period 8 (probe: kinks at 5, 9, 13, 17, ...), a step at each
  // junction since 0.5 != 0
  mo::KeyCurve cy = curve({{1, 0}, {5, 1}, {9, 0.5}}, I::kLinear);
  mo::KeyModifier mc;
  mc.type = "CYCLES";
  mc.fields["mode_before"] = Tree::string("NONE");
  mc.fields["mode_after"] = Tree::string("REPEAT");
  cy.modifiers.push_back(mc);
  check_features(mo::curve_features(cy, 0, 22), {{1, C::kC0, true}, {5, C::kC0, true}, {9, C::kStep, true},
                                                 {13, C::kC0, true}, {17, C::kStep, true}, {21, C::kC0, true}});
  // MIRROR: the odd cycles run backwards (key 5 maps to 9 + 9 - 5 = 13), junctions kink; cycles_after limits them
  cy.modifiers[0].fields["mode_after"] = Tree::string("MIRROR");
  cy.modifiers[0].fields["cycles_after"] = Tree::integer(1);
  check_features(mo::curve_features(cy, 0, 40), {{1, C::kC0, true}, {5, C::kC0, true}, {9, C::kC0, true},
                                                 {13, C::kC0, true}, {17, C::kC0, true}});
  // a restricted range switches the modifier on and off: steps without blends, kinks with them
  mo::KeyCurve rr = curve({{1, 0}, {49, 1}}, I::kLinear);
  mo::KeyModifier noise;
  noise.type = "NOISE";
  noise.use_restricted_range = true;
  noise.frame_start = 10;
  noise.frame_end = 30;
  noise.blend_out = 4;
  rr.modifiers.push_back(noise);
  check_features(mo::curve_features(rr, -1e9, 1e9), {{1, C::kC0, true}, {10, C::kStep, false}, {26, C::kC0, false},
                                                     {30, C::kC0, false}, {49, C::kC0, true}});
  // ENVELOPE (and any unknown type) is MOT-SOURCE unless lenient; a muted one is ignored
  mo::KeyCurve env = curve({{1, 0}, {49, 1}}, I::kLinear);
  mo::KeyModifier me;
  me.type = "ENVELOPE";
  env.modifiers.push_back(me);
  CHECK_THROWS_WITH_AS(mo::curve_features(env, -1e9, 1e9), doctest::Contains("MOT-SOURCE"), mo::MotionError);
  CHECK(mo::curve_features(env, -1e9, 1e9, true).size() == 2);
  env.modifiers[0].mute = true;
  CHECK(mo::curve_features(env, -1e9, 1e9).size() == 2);
  // malformed tables
  mo::KeyCurve bad = curve({{1, 0}, {13, 1}}, I::kLinear);
  bad.keys.pop_back();
  CHECK_THROWS_WITH_AS(mo::curve_features(bad, -1e9, 1e9), doctest::Contains("BRG-PROTO"), mo::MotionError);
  bad = curve({{13, 0}, {1, 1}}, I::kLinear);
  CHECK_THROWS_WITH_AS(mo::curve_features(bad, -1e9, 1e9), doctest::Contains("BRG-PROTO"), mo::MotionError);
  bad = curve({{1, 0}, {13, 1}}, I::kLinear);
  bad.keys[2] = 13;  // interpolation code past ELASTIC
  CHECK_THROWS_WITH_AS(mo::curve_features(bad, -1e9, 1e9), doctest::Contains("BRG-PROTO"), mo::MotionError);
}

TEST_CASE("declared_keys: NLA mapping, strip ends and blends, plateaus scaled by the strip (BL-7 probe)") {
  const mo::TimeMap tm = time24(1, 80);
  mo::KeyScene sc;
  sc.objects.resize(1);
  sc.objects[0].animations = {0};
  sc.animations.resize(2);
  mo::KeyAnimation& action = sc.animations[1];  // the strip's action: LINEAR keys 1 (0), 5 (1), 9 (0), 10 (2)
  action.action = "act";
  action.fcurves.push_back(curve({{1, 0}, {5, 1}, {9, 0}, {10, 2}}, mo::Interp::kLinear));
  mo::KeyStrip s;
  s.action = "act";
  s.frame_start = 20;
  s.frame_end = 56;
  s.action_frame_start = 1;
  s.action_frame_end = 10;
  s.scale = 2;
  s.repeat = 2;
  s.extrapolation = 1;  // HOLD
  sc.animations[0].nla.push_back({false, false, {s}});
  auto frames_of = [&](const std::vector<mo::DeclaredKey>& k) {
    std::vector<double> f;
    for (const auto& d : k) f.push_back(1.0 + static_cast<double>(d.t_ns) / tm.period_ns());
    return f;
  };
  const int64_t t0 = -2 * balla::time::kWindowNs, t1 = tm.t_end_ns();
  // forward: keys at 20 + (a - 1) 2 + 18 r (probe kinks 20, 28, 36, 38, 46, 54, 56); the repeat boundary at 38
  std::vector<mo::DeclaredKey> k = mo::declared_keys(sc, 0, tm, t0, t1);
  std::vector<double> want{20, 28, 36, 38, 46, 54, 56};
  REQUIRE(k.size() == want.size());
  for (size_t i = 0; i < k.size(); ++i) {
    CHECK(frames_of(k)[i] == doctest::Approx(want[i]));
    CHECK(k[i].hint == C::kC0);
    CHECK(k[i].plateau);
    CHECK(k[i].plateau_frames == doctest::Approx(2e-4));  // 1e-4 N_s / framelen (probe: 4e-4 at scale 4)
    CHECK(k[i].t_ns == tm.frame_to_ns(want[i]));
  }
  // reversed: mirrored about the strip's middle (probe kinks 20, 22, 30, 38, 40, 48, 56)
  sc.animations[0].nla[0].strips[0].use_reverse = true;
  k = mo::declared_keys(sc, 0, tm, t0, t1);
  want = {20, 22, 30, 38, 40, 48, 56};
  REQUIRE(k.size() == want.size());
  for (size_t i = 0; i < k.size(); ++i) CHECK(frames_of(k)[i] == doctest::Approx(want[i]));
  // NOTHING without blends: the strip's ends step (no plateau of their own; 20 and 56 are keys too, so it stays);
  // blends kink at the ramps' ends; HOLD_FORWARD holds the end only
  mo::KeyStrip& st = sc.animations[0].nla[0].strips[0];
  st.use_reverse = false;
  st.extrapolation = 0;
  k = mo::declared_keys(sc, 0, tm, t0, t1);
  CHECK(k.front().hint == C::kStep);
  CHECK(k.back().hint == C::kStep);
  st.extrapolation = 2;
  st.blend_in = 3;
  k = mo::declared_keys(sc, 0, tm, t0, t1);
  CHECK(k.front().hint == C::kC0);
  CHECK(k.back().hint == C::kC0);
  CHECK(frames_of(k)[1] == doctest::Approx(23));  // the end of the blend-in ramp
  CHECK_FALSE(k[1].plateau);
  CHECK(k[1].plateau_frames == 0.0);
  // animated strip time: the strip's ends and blend ramp only (no repeat wraps, probe p4); animated influence: its
  // ends (by extrapolation, the blends replaced: probe p4) and the repeat boundary, no key mapped
  st.use_animated_time = true;
  k = mo::declared_keys(sc, 0, tm, t0, t1);
  want = {20, 23, 56};
  REQUIRE(k.size() == want.size());
  for (size_t i = 0; i < k.size(); ++i) CHECK(frames_of(k)[i] == doctest::Approx(want[i]));
  for (const auto& d : k) CHECK_FALSE(d.plateau);
  st.use_animated_time = false;
  st.use_animated_influence = true;
  k = mo::declared_keys(sc, 0, tm, t0, t1);
  want = {20, 38, 56};
  REQUIRE(k.size() == want.size());
  for (size_t i = 0; i < k.size(); ++i) CHECK(frames_of(k)[i] == doctest::Approx(want[i]));
  CHECK(k.front().hint == C::kStep);  // HOLD_FORWARD: the underlying value before the start
  CHECK(k.back().hint == C::kC0);     // and the end held
  for (const auto& d : k) CHECK_FALSE(d.plateau);
  st.use_animated_influence = false;
  // a strip property keyed through the owner (probe p3): nothing of this animation's strips; its moving curves stay
  mo::KeyCurve keyed_blend = curve({{1, 0}, {30, 8}}, mo::Interp::kLinear);
  keyed_blend.data_path = "animation_data.nla_tracks[\"T\"].strips[\"s\"].blend_in";
  sc.animations[0].fcurves.push_back(keyed_blend);
  CHECK(mo::declared_keys(sc, 0, tm, t0, t1).empty());
  sc.animations[0].fcurves.push_back(curve({{60, 0}, {70, 1}}, mo::Interp::kLinear));
  k = mo::declared_keys(sc, 0, tm, t0, t1);
  REQUIRE(k.size() == 2);
  CHECK(frames_of(k)[0] == doctest::Approx(60));
  CHECK(frames_of(k)[1] == doctest::Approx(70));
  sc.animations[0].fcurves.clear();
  CHECK(mo::declared_keys(sc, 0, tm, t0, t1).size() == 8);
  sc.animations[0].nla.push_back({false, true, {}});
  CHECK(mo::declared_keys(sc, 0, tm, t0, t1).empty());
  sc.animations[0].nla.pop_back();
  sc.animations[0].nla[0].mute = true;
  CHECK(mo::declared_keys(sc, 0, tm, t0, t1).empty());
  sc.animations[0].nla[0].mute = false;
  sc.animations[0].nla[0].strips[0].action = "missing";
  CHECK_THROWS_WITH_AS(mo::declared_keys(sc, 0, tm, t0, t1), doctest::Contains("BRG-PROTO"), mo::MotionError);
}

TEST_CASE("declared_keys: strip properties animated from a strip's action or by a driver (BL-7 probes p5, p6)") {
  // probe p5's scene: a LINEAR action over frames 1 .. 21 in strip "s" of track "T" (NOTHING), and the owner's control
  // action "ctl" (the strip property curve, pushed down) in a strip of track "T2" over frames 1 .. 20 (HOLD)
  const mo::TimeMap tm = time24(1, 80);
  mo::KeyCurve keyed = curve({{1, 0}, {20, 8}}, mo::Interp::kLinear);
  keyed.data_path = "animation_data.nla_tracks[\"T\"].strips[\"s\"].blend_out";
  auto scene = [&](bool below, bool with_curve) {
    mo::KeyScene sc;
    sc.objects.resize(1);
    sc.objects[0].animations = {0};
    sc.animations.resize(3);
    sc.animations[1].action = "act";
    sc.animations[1].fcurves.push_back(curve({{1, 0}, {21, 1}}, mo::Interp::kLinear));
    sc.animations[2].action = "ctl";
    if (with_curve) sc.animations[2].fcurves.push_back(keyed);
    mo::KeyStrip s;
    s.action = "act";
    s.frame_start = 1;
    s.frame_end = 21;
    s.action_frame_start = 1;
    s.action_frame_end = 21;
    s.extrapolation = 0;
    mo::KeyStrip c;
    c.action = "ctl";
    c.frame_start = 1;
    c.frame_end = 20;
    c.action_frame_start = 1;
    c.action_frame_end = 20;
    const mo::KeyTrack t{false, false, {s}}, t2{false, false, {c}};
    sc.animations[0].nla = below ? std::vector<mo::KeyTrack>{t2, t} : std::vector<mo::KeyTrack>{t, t2};
    return sc;
  };
  const int64_t t0 = -2 * balla::time::kWindowNs, t1 = tm.t_end_ns();
  auto frames_of = [&](const std::vector<mo::DeclaredKey>& k) {
    std::vector<double> f;
    for (const auto& d : k) f.push_back(1.0 + static_cast<double>(d.t_ns) / tm.period_ns());
    return f;
  };
  // without the curve: s steps at 1 and 21 (its keys there), the control strip's held ends at 1 and 20
  for (const bool below : {false, true}) {
    const std::vector<mo::DeclaredKey> k = mo::declared_keys(scene(below, false), 0, tm, t0, t1);
    const std::vector<double> want{1, 20, 21};
    REQUIRE(k.size() == want.size());
    for (size_t i = 0; i < k.size(); ++i) CHECK(frames_of(k)[i] == doctest::Approx(want[i]));
    CHECK(k[0].hint == C::kStep);
    CHECK(k[0].plateau);
    CHECK(k[2].hint == C::kStep);
  }
  // the curve pushed down, above or below, its track or its strip muted, or the solo track the other one: nothing
  for (const bool below : {false, true}) {
    mo::KeyScene sc = scene(below, true);
    CHECK(mo::declared_keys(sc, 0, tm, t0, t1).empty());
    mo::KeyTrack& t2 = sc.animations[0].nla[below ? 0 : 1];
    t2.mute = true;
    CHECK(mo::declared_keys(sc, 0, tm, t0, t1).empty());
    t2.mute = false;
    t2.strips[0].mute = true;
    CHECK(mo::declared_keys(sc, 0, tm, t0, t1).empty());
    t2.strips[0].mute = false;
    sc.animations[0].nla[below ? 1 : 0].is_solo = true;
    CHECK(mo::declared_keys(sc, 0, tm, t0, t1).empty());
  }
  // in the keyed strip's own action: nothing; the active action's moving curves stay
  {
    mo::KeyScene sc = scene(false, false);
    sc.animations[1].fcurves.push_back(keyed);
    sc.animations[0].fcurves.push_back(curve({{60, 0}, {70, 1}}, mo::Interp::kLinear));
    const std::vector<mo::DeclaredKey> k = mo::declared_keys(sc, 0, tm, t0, t1);
    REQUIRE(k.size() == 2);
    CHECK(frames_of(k)[0] == doctest::Approx(60));
    CHECK(frames_of(k)[1] == doctest::Approx(70));
  }
  // a strip whose action the exchange lacks does not stop the scan (the strip is muted, so it is not refused)
  {
    mo::KeyScene sc = scene(false, true);
    sc.animations[0].nla.push_back({false, false, {mo::KeyStrip{}}});
    sc.animations[0].nla.back().strips[0].action = "missing";
    sc.animations[0].nla.back().strips[0].mute = true;
    std::swap(sc.animations[0].nla.front(), sc.animations[0].nla.back());
    CHECK(mo::declared_keys(sc, 0, tm, t0, t1).empty());
  }
  // a driver on a strip property (p6): nothing; a driver on the motion leaves the strips declared (detection finds it)
  {
    mo::KeyScene sc = scene(false, false);
    sc.animations[0].driver_paths = {"animation_data.nla_tracks[\"T\"].strips[\"s\"].scale"};
    CHECK(mo::declared_keys(sc, 0, tm, t0, t1).empty());
    sc.animations[0].driver_paths = {"location", "animation_data.action_influence"};
    CHECK(mo::declared_keys(sc, 0, tm, t0, t1).size() == 3);
  }
}

TEST_CASE("declared_keys: a repeat beyond the strip's end visits only the repeats within it; too many: MOT-SOURCE") {
  // a 2-key LINEAR action (frames 1 .. 2) in a strip over frames 1 .. 50: Blender's repeat would be 49
  const mo::TimeMap tm = time24(1, 80);
  mo::KeyScene sc;
  sc.objects.resize(1);
  sc.objects[0].animations = {0};
  sc.animations.resize(2);
  sc.animations[1].action = "act";
  sc.animations[1].fcurves.push_back(curve({{1, 0}, {2, 1}}, mo::Interp::kLinear));
  mo::KeyStrip s;
  s.action = "act";
  s.frame_start = 1;
  s.frame_end = 50;
  s.action_frame_start = 1;
  s.action_frame_end = 2;
  s.repeat = 49;
  sc.animations[0].nla.push_back({false, false, {s}});
  const int64_t t0 = -2 * balla::time::kWindowNs, t1 = tm.t_end_ns();
  const std::vector<mo::DeclaredKey> want = mo::declared_keys(sc, 0, tm, t0, t1);
  CHECK(want.size() == 50);  // frames 1 .. 50
  for (const double repeat : {3e9, 1e300}) {  // inconsistent with frame_end; 1e300 > 2^63
    sc.animations[0].nla[0].strips[0].repeat = repeat;
    const std::vector<mo::DeclaredKey> k = mo::declared_keys(sc, 0, tm, t0, t1);
    REQUIRE(k.size() == want.size());
    for (size_t i = 0; i < k.size(); ++i) {
      CHECK(k[i].t_ns == want[i].t_ns);
      CHECK(k[i].hint == want[i].hint);
    }
  }
  // a span of 1e-9 frames per repeat: 4.9e10 repeats within the strip
  sc.animations[0].nla[0].strips[0].scale = 1e-9;
  sc.animations[0].nla[0].strips[0].repeat = 1e15;
  CHECK_THROWS_WITH_AS(mo::declared_keys(sc, 0, tm, t0, t1), doctest::Contains("MOT-SOURCE"), mo::MotionError);
  // only the repeats that meet the span are visited: a strip starting long before it is fine
  mo::KeyStrip& st = sc.animations[0].nla[0].strips[0];
  st.scale = 1;
  st.frame_start = -1e9;
  st.frame_end = 50;
  st.repeat = 1e9 + 49;
  const std::vector<mo::DeclaredKey> late = mo::declared_keys(sc, 0, tm, t0, t1);
  REQUIRE(late.size() == want.size());  // the span [-2W, t_end] (2W = 16.8 ms) holds frames 1 .. 50
  for (size_t i = 0; i < late.size(); ++i) CHECK(late[i].t_ns == want[i].t_ns);
}

TEST_CASE("declared_keys: the chain (parents, constraint targets and their data), moving curves only, framelen") {
  mo::KeyScene sc;
  sc.objects.resize(5);
  sc.animations.resize(5);
  // 0: the body, parented to 1, with a constraint targeting 2; 3 is unrelated; 2's data block (a curve) is animated
  sc.objects[0].parent = 1;
  sc.objects[0].animations = {0};
  sc.objects[0].constraint_targets = {2};
  sc.objects[1].animations = {1};
  sc.objects[2].animations = {2};
  sc.objects[2].data_animations = {3};
  sc.objects[3].animations = {4};
  sc.animations[0].fcurves = {curve({{10, 0}, {20, 1}}, mo::Interp::kLinear)};
  mo::KeyCurve color = curve({{15, 0}, {16, 1}}, mo::Interp::kConstant);
  color.data_path = "color";  // not a motion: ignored on an object
  sc.animations[0].fcurves.push_back(color);
  sc.animations[1].fcurves = {curve({{30, 0}, {40, 1}}, mo::Interp::kLinear)};
  sc.animations[1].fcurves[0].data_path = "pose.bones[\"arm\"].rotation_quaternion";
  sc.animations[2].fcurves = {curve({{50, 0}, {60, 1}}, mo::Interp::kLinear)};
  sc.animations[2].fcurves[0].data_path = "constraints[\"Track\"].influence";
  sc.animations[3].fcurves = {curve({{70, 0}, {80, 1}}, mo::Interp::kLinear)};
  sc.animations[3].fcurves[0].data_path = "eval_time";  // the target's data block: kept
  sc.animations[4].fcurves = {curve({{5, 0}, {6, 1}}, mo::Interp::kLinear)};
  const mo::TimeMap tm = time24(1, 100);
  const std::vector<mo::DeclaredKey> k = mo::declared_keys(sc, 0, tm, 0, tm.t_end_ns());
  std::vector<int64_t> want;
  for (const double f : {10, 20, 30, 40, 50, 60, 70, 80}) want.push_back(tm.frame_to_ns(f));
  REQUIRE(k.size() == want.size());
  for (size_t i = 0; i < k.size(); ++i) CHECK(k[i].t_ns == want[i]);
  // framelen 0.5 (frame_map 100 -> 200): scene time c is output frame 2c, the plateau 2e-4 output frames
  const mo::TimeMap half = time24(1, 200, 100, 200);
  const std::vector<mo::DeclaredKey> kh = mo::declared_keys(sc, 0, half, 0, half.t_end_ns());
  REQUIRE(kh.size() == want.size());
  CHECK(kh[0].t_ns == half.frame_to_ns(20.0));
  CHECK(kh[0].plateau_frames == doctest::Approx(2e-4));
  // the span clips by time; a loop in the parent chain is BRG-PROTO
  CHECK(mo::declared_keys(sc, 0, tm, tm.frame_to_ns(35), tm.frame_to_ns(65)).size() == 3);
  sc.objects[1].parent = 0;
  CHECK_THROWS_WITH_AS(mo::declared_keys(sc, 0, tm, 0, tm.t_end_ns()), doctest::Contains("BRG-PROTO"),
                       mo::MotionError);
}

TEST_CASE("declared_keys: equal times merge to the lowest class and the widest plateau") {
  mo::KeyScene sc;
  sc.objects.resize(1);
  sc.objects[0].animations = {0, 1};
  sc.animations.resize(3);
  sc.animations[0].fcurves = {curve({{1, 0}, {13, 1}, {25, 0}}, mo::Interp::kBezier),      // C1 at 13
                              curve({{1, 0}, {13, 1}}, mo::Interp::kConstant)};             // step at 13
  sc.animations[1].action = "";
  mo::KeyStrip s;  // a strip of scale 3 whose key lands on 13 as well: plateau 3e-4
  s.action = "a3";
  s.frame_start = 10;
  s.frame_end = 16;
  s.action_frame_start = 0;
  s.action_frame_end = 2;
  s.scale = 3;
  sc.animations[1].nla.push_back({false, false, {s}});
  sc.animations[2].action = "a3";
  sc.animations[2].fcurves = {curve({{0, 0}, {1, 1}, {2, 0}}, mo::Interp::kBezier)};
  const mo::TimeMap tm = time24(1, 30);
  const std::vector<mo::DeclaredKey> k = mo::declared_keys(sc, 0, tm, 0, tm.t_end_ns());
  const auto it = std::find_if(k.begin(), k.end(), [&](const mo::DeclaredKey& d) { return d.t_ns == tm.frame_to_ns(13); });
  REQUIRE(it != k.end());
  CHECK(it->hint == C::kStep);
  CHECK(it->plateau);
  CHECK(it->plateau_frames == doctest::Approx(3e-4));
  CHECK(std::adjacent_find(k.begin(), k.end(), [](const auto& a, const auto& b) { return a.t_ns >= b.t_ns; }) ==
        k.end());
}
