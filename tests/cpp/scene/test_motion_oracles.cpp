// The keyed sampler against synthetic and recorded oracles (docs/design/phase3/blender.md §8.1): T6's sampling part
// (the prototype's bounce, linear, step and smooth motions with Blender's float32 time, float32 matrices and key
// plateau; the float32 noise floor and POS-RES) and T7 (a recorded oracle replays bitwise at 1 and 8 workers, the
// samples and detection's breakpoints alike, and a changed request list gives BRG-REPLAY).
#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <numbers>
#include <vector>

#include "core/time/time.h"
#include "motion_oracles.h"
#include "scene/motion/pipeline.h"

namespace mo = balla::scene::motion;
namespace rot = balla::scene::rot;
namespace syn = balla::scene::motion::testing;

namespace {

using syn::declared;
using syn::kEps;
using syn::kHmin;
using syn::run;
using syn::Run;
using syn::scene_of;
using syn::time24;

// Every unresolved interval lies within `slack` of one of the breakpoints.
bool unresolved_near(const mo::SampleSet& s, const std::vector<int64_t>& breaks, int64_t slack) {
  return std::all_of(s.unresolved.begin(), s.unresolved.end(), [&](const auto& iv) {
    return std::any_of(breaks.begin(), breaks.end(),
                       [&](int64_t b) { return b >= iv.first - slack && b <= iv.second + slack; });
  });
}

bool track_equal(const mo::Track& a, const mo::Track& b) {
  auto same = [](const auto& x, const auto& y) {
    return x.size() == y.size() && (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0);
  };
  return same(a.t_ns, b.t_ns) && same(a.q, b.q) && same(a.p, b.p) && same(a.w_pos, b.w_pos) && same(a.w_rot, b.w_rot) &&
         same(a.floor1, b.floor1);
}

bool same_bits(double x, double y) { return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y); }

// detection's results bitwise: nothing after the last burst round asks the oracle, so the replay alone cannot show them
bool breakpoint_equal(const mo::Breakpoint& x, const mo::Breakpoint& y) {
  return x.t_ns == y.t_ns && x.cls == y.cls && x.declared == y.declared && x.snapped == y.snapped &&
         same_bits(x.jump_px, y.jump_px) && same_bits(x.dv_pxps, y.dv_pxps) && same_bits(x.dt_bound_ns, y.dt_bound_ns);
}

bool set_equal(const mo::SampleSet& a, const mo::SampleSet& b) {
  return track_equal(a.fit, b.fit) && track_equal(a.held_out, b.held_out) &&
         a.residual_px.size() == b.residual_px.size() &&
         std::equal(a.residual_px.begin(), a.residual_px.end(), b.residual_px.begin(), same_bits) &&
         a.unresolved == b.unresolved && same_bits(a.floor_px, b.floor_px) && a.rounds == b.rounds &&
         same_bits(a.alias_max_rad, b.alias_max_rad) && a.alias_warnings == b.alias_warnings &&
         a.light_near == b.light_near && a.breakpoints.size() == b.breakpoints.size() &&
         std::equal(a.breakpoints.begin(), a.breakpoints.end(), b.breakpoints.begin(), breakpoint_equal) &&
         a.detect == b.detect;
}

// fit and V ascending in time, disjoint; V's residuals within eps/4 + floor_px; |dtheta| < 0.6 pi between consecutive
// samples of both (the aliasing pre-pass) when nothing is unresolved
void check_accepted(const mo::SampleSet& s) {
  for (size_t i = 0; i < s.residual_px.size(); ++i) CHECK(s.residual_px[i] <= kEps / 4.0 + s.floor_px);
  CHECK(s.residual_px.size() == s.held_out.size());
  std::vector<int64_t> all(s.fit.t_ns);
  all.insert(all.end(), s.held_out.t_ns.begin(), s.held_out.t_ns.end());
  std::sort(all.begin(), all.end());
  auto increasing = [](const std::vector<int64_t>& t) {
    return std::adjacent_find(t.begin(), t.end(), [](int64_t a, int64_t b) { return a >= b; }) == t.end();
  };
  CHECK(increasing(s.fit.t_ns));
  CHECK(increasing(s.held_out.t_ns));
  CHECK(increasing(all));
  if (s.unresolved.empty()) CHECK(s.alias_max_rad < 0.6 * std::numbers::pi);
}

}  // namespace

TEST_CASE("T6 sampling: smooth motion converges without unresolved intervals") {
  const mo::TimeMap tm = time24(73);
  const auto calls = std::make_shared<int64_t>(0);
  const mo::Oracle o = syn::synthetic_oracle(tm, {{syn::smooth, {}}, {syn::camera, {}}}, calls);
  const Run r = run(o, tm, scene_of(1), {}, 2);
  check_accepted(r.sets[0]);
  CHECK(r.sets[0].unresolved.empty());
  CHECK(r.sets[1].unresolved.empty());
  CHECK(r.sets[0].rounds < 40);
  CHECK_FALSE(mo::pos_res(r.sets[0], kEps));
  MESSAGE("smooth: " << *calls << " calls (" << r.stats.calls << " unique), " << r.stats.rounds << " rounds, fit "
                     << r.sets[0].fit.size() << ", held out " << r.sets[0].held_out.size());
  CHECK(r.stats.calls == *calls);
  // every base interval's midpoint is accepted in the first refinement round: 2 n - 1 calls for n base samples, the
  // aliasing pre-pass's 8 per frame (192 per second at 24 fps) and no more
  const size_t base =
      mo::base_requests(tm, -2 * balla::time::kWindowNs, tm.t_end_ns(), {}, mo::SampleOptions{}).size();
  CHECK(r.stats.calls == static_cast<int64_t>(2 * base - 1));
  CHECK(r.stats.calls_by_stage.at("base") == static_cast<int64_t>(base));
  CHECK(r.stats.calls_by_stage.at("refine") == static_cast<int64_t>(base - 1));
  CHECK(r.stats.rounds == 2);
  CHECK(r.sets[0].held_out.size() == base - 1);
}

TEST_CASE("T6 sampling: a fast spin densifies until |dtheta| < 0.6 pi between samples") {
  // 40 rev/s about a tilted axis: 0.83 pi per base interval (P / 4), so the stencils' logs wrap and the base grid
  // alone would alias; the refinement splits until the motion is resolved
  const mo::TimeMap tm = time24(25);
  auto spin = [](double frame) {
    const double t = (frame - 1.0) / 24.0, a = 2.0 * std::numbers::pi * 40.0 * t;
    return syn::Pose{syn::qexp({0.6 * a, 0.0, 0.8 * a}), {0.0, 0.0, 0.0}};
  };
  const Run r = run(syn::synthetic_oracle(tm, {{spin, {}}, {syn::camera, {}}}), tm, scene_of(1), {}, 2);
  check_accepted(r.sets[0]);
  CHECK(r.sets[0].unresolved.empty());
  CHECK(r.sets[0].alias_warnings == 0);
  const size_t base =
      mo::base_requests(tm, -2 * balla::time::kWindowNs, tm.t_end_ns(), {}, mo::SampleOptions{}).size();
  CHECK(r.sets[0].fit.size() > base);
  MESSAGE("spin: " << r.stats.calls << " calls, " << r.stats.rounds << " rounds, fit " << r.sets[0].fit.size()
                   << ", held out " << r.sets[0].held_out.size() << ", largest turn " << r.sets[0].alias_max_rad
                   << " rad");
}

TEST_CASE("T6 sampling: bounce impacts, LINEAR kinks and CONSTANT steps stay unresolved only near their breakpoints") {
  const mo::TimeMap tm = time24(73);
  std::vector<double> impacts;
  syn::bounce(1.0, &impacts);
  std::vector<int64_t> imp_ns;
  for (const double t : impacts) imp_ns.push_back(std::llround(t * 1e9));
  // the step happens 1e-4 frames before each CONSTANT key (Blender's key plateau)
  const std::vector<double> lin = syn::linear_keys(), stp = syn::step_keys();
  std::vector<int64_t> lin_ns, step_ns;
  for (const double f : lin) lin_ns.push_back(tm.to_ns(static_cast<float>(f)));
  for (size_t k = 1; k < stp.size(); ++k) step_ns.push_back(tm.to_ns(static_cast<float>(stp[k])) - 4167);
  std::vector<mo::DeclaredKey> step_keys = declared(tm, {stp.begin() + 1, stp.end()}, mo::Continuity::kStep);
  const std::vector<mo::DeclaredKey> lin_keys = declared(tm, lin, mo::Continuity::kC0);

  const auto calls = std::make_shared<int64_t>(0);
  const mo::Oracle o = syn::synthetic_oracle(
      tm, {{[](double f) { return syn::bounce(f); }, {}}, {syn::linear, lin}, {syn::step, stp}, {syn::step, stp},
           {syn::camera, {}}},
      calls);
  mo::DetectOptions keep_steps;  // detection would refuse the steps (MOT-STEP); this case checks the sampling
  keep_steps.refuse_steps = false;
  const Run r = run(o, tm, scene_of(4), {{}, lin_keys, step_keys, {}}, 4, {}, keep_steps);
  for (const auto& s : r.sets) check_accepted(s);
  const mo::SampleSet &bounce = r.sets[0], &linear = r.sets[1], &step = r.sets[2], &step_undeclared = r.sets[3];
  CHECK(unresolved_near(bounce, imp_ns, 1000000));
  CHECK(unresolved_near(linear, lin_ns, 1000000));
  CHECK(unresolved_near(step, step_ns, 1000000));
  CHECK(unresolved_near(step_undeclared, step_ns, 1000000));
  // declared LINEAR keys are samples; a step lies inside an unresolved interval of at most 2 h_min, declared or not
  for (const int64_t k : lin_ns) CHECK(std::binary_search(linear.fit.t_ns.begin(), linear.fit.t_ns.end(), k));
  for (const mo::SampleSet* s : {&step, &step_undeclared}) {
    for (const int64_t t : step_ns) {
      CHECK(std::any_of(s->unresolved.begin(), s->unresolved.end(), [&](const auto& iv) {
        return iv.first < t && t <= iv.second && iv.second - iv.first <= 2 * kHmin;
      }));
    }
  }
  // samples strictly inside a declared key's plateau are dropped; the key's own sample stays
  for (const int64_t k : lin_ns) {
    for (const int64_t t : linear.fit.t_ns) CHECK_FALSE((std::llabs(t - k) > 21 && std::llabs(t - k) <= 4167));
  }
  CHECK(r.sets[4].unresolved.empty());
  MESSAGE("bounce/linear/step/step(undeclared) + camera: " << r.stats.calls << " calls, " << r.stats.rounds
          << " rounds; unresolved " << bounce.unresolved.size() << " / " << linear.unresolved.size() << " / "
          << step.unresolved.size() << " / " << step_undeclared.unresolved.size());
}

TEST_CASE("T6 noise floor: a float32 camera 3 m from the origin at G = 2667 px, d at the floor, raises POS-RES") {
  // RI2's dolly (r2_f32_noise_floor.py): 1 m/s along x from 3 m, no rotation; clip_start 0.1 m and a static plane 5 cm
  // below put d_scene at its floor, 0.1 m
  const mo::TimeMap tm = time24(13);
  mo::MotionScene scene;
  scene.bodies.resize(1);
  scene.bodies[0].role = mo::Role::kRig;
  scene.cameras.push_back({0, {0, 0, 0}, 2667.0, 0.1});
  scene.static_vertices = {-100, -100, 1.45, 100, -100, 1.45, 0, 100, 1.45};
  scene.static_indices = {0, 1, 2};
  auto dolly = [](double frame) {
    const double t = (frame - 1.0) / 24.0;
    return syn::Pose{{1, 0, 0, 0}, {3.0 + t + 0.05 * std::sin(3.0 * t), 0.3 * std::cos(2.0 * t), 1.5}};
  };
  const mo::Oracle o = syn::synthetic_oracle(tm, {{dolly, {}}});
  const Run r = run(o, tm, scene, {}, 1);
  const mo::SampleSet& s = r.sets[0];
  check_accepted(s);
  CHECK(s.unresolved.empty());
  const double worst = s.residual_px.empty() ? 0.0 : *std::max_element(s.residual_px.begin(), s.residual_px.end());
  CHECK(worst > kEps / 4.0);  // the float32 noise alone exceeds eps/4: without the floor this would not converge
  const auto pr = mo::pos_res(s, kEps);
  REQUIRE(pr);
  CHECK(pr->floor_px >= worst);
  CHECK(pr->bound_px == doctest::Approx(kEps / 2.0 + 2.0 * pr->floor_px));
  MESSAGE("noise floor: worst midpoint residual " << worst << " px, floor_px " << pr->floor_px << " px, "
                                                  << s.held_out.size() << " midpoints");
}

TEST_CASE("T6: an equisolid camera in the error model gives finite bounds") {
  balla::render::Camera eq;
  eq.model = balla::render::CameraModelKind::kKannalaBrandt;
  eq.width = eq.height = 373;
  eq.fx = eq.fy = 186.5;
  eq.cx = eq.cy = 186.0;
  eq.dist = {-1.0 / 24.0, 1.0 / 1920.0, -1.0 / 322560.0, 1.0 / 92897280.0};
  const mo::TimeMap tm = time24(25);
  mo::MotionScene scene = scene_of(1);
  scene.cameras[0].gain_px_per_rad = mo::angular_gain(eq);
  const mo::Oracle o = syn::synthetic_oracle(tm, {{syn::smooth, {}}, {syn::camera, {}}});
  const Run r = run(o, tm, scene, {}, 1);
  for (const auto& s : r.sets) {
    for (size_t i = 0; i < s.fit.size(); ++i) {
      CHECK(std::isfinite(s.fit.w_pos[i]));
      CHECK(std::isfinite(s.fit.w_rot[i]));
      CHECK(std::isfinite(s.fit.floor1[i]));
    }
  }
  CHECK(r.sets[0].unresolved.empty());
}

TEST_CASE("T6: MOT-ALIAS above 0.9 pi between samples at h_min; VIS-ANIM for an absent body") {
  // a rotation that jumps by pi at one instant (a CONSTANT key on rotation) stays at pi down to h_min
  const mo::TimeMap tm = time24(25);
  auto flip = [](double frame) {
    return syn::Pose{frame < 13.0 ? rot::Q{1, 0, 0, 0} : rot::Q{0, 0, 0, 1}, {0, 0, 0}};
  };
  const mo::Oracle o = syn::synthetic_oracle(tm, {{flip, {}}, {syn::camera, {}}});
  CHECK_THROWS_WITH_AS(run(o, tm, scene_of(1), {}, 1), doctest::Contains("MOT-ALIAS"), mo::MotionError);
  mo::Oracle absent = [&](std::span<const int32_t> F, std::span<const float> s) {
    mo::RawPoses r = syn::synthetic_oracle(tm, {{syn::smooth, {}}, {syn::camera, {}}})(F, s);
    r.present[0] = 0;
    return r;
  };
  CHECK_THROWS_WITH_AS(run(absent, tm, scene_of(1), {}, 1), doctest::Contains("VIS-ANIM"), mo::MotionError);
}

TEST_CASE("T6: GEO-SCALEANIM for a keyed scale; a body without S_ref is refused before the first call") {
  // the smooth body's scale keyed from 1 at frame 13 to 2 at frame 25 (linear); scene_of gives it S_ref = I
  const mo::TimeMap tm = time24(25);
  auto grows = [&](std::shared_ptr<int64_t> calls) -> mo::Oracle {
    return [&tm, calls](std::span<const int32_t> F, std::span<const float> s) {
      mo::RawPoses r = syn::synthetic_oracle(tm, {{syn::smooth, {}}, {syn::camera, {}}}, calls)(F, s);
      for (size_t i = 0; i < F.size(); ++i) {
        const double k = 1.0 + std::clamp((r.c_final[i] - 13.0) / 12.0, 0.0, 1.0);
        for (int row = 0; row < 3; ++row) {
          for (int j = 0; j < 3; ++j) {
            float& a = r.m[i * 24 + static_cast<size_t>(4 * row + j)];
            a = static_cast<float>(static_cast<double>(a) * k);
          }
        }
      }
      return r;
    };
  };
  const auto calls = std::make_shared<int64_t>(0);
  CHECK_THROWS_WITH_AS(run(grows(calls), tm, scene_of(1), {}, 1),
                       doctest::Contains("GEO-SCALEANIM: body 0 changes scale"), mo::MotionError);
  CHECK(*calls > 0);
  // without S_ref the same scene cannot sample at all (no silent rigid fit of the growing body)
  mo::MotionScene missing = scene_of(1);
  missing.bodies[0].S_ref.reset();
  const auto none = std::make_shared<int64_t>(0);
  CHECK_THROWS_WITH_AS(run(grows(none), tm, missing, {}, 1),
                       doctest::Contains("body column 0 (a mesh body) has no S_ref"), std::invalid_argument);
  CHECK(*none == 0);
}

TEST_CASE("T7: a recorded oracle replays bitwise at 1 and 8 workers; a changed request list gives BRG-REPLAY") {
  // bounce (undeclared kinks: bursts), linear (declared kinks: snapped) and steps (bisection), detection's results
  // included
  const mo::TimeMap tm = time24(49);
  const std::vector<double> lin = syn::linear_keys(), stp = syn::step_keys();
  const std::vector<std::vector<mo::DeclaredKey>> keys{
      {}, declared(tm, lin, mo::Continuity::kC0), declared(tm, {stp.begin() + 1, stp.end()}, mo::Continuity::kStep)};
  const mo::MotionScene scene = scene_of(3);
  mo::DetectOptions keep_steps;
  keep_steps.refuse_steps = false;
  mo::OracleLog log;
  const mo::Oracle rec = mo::recording(
      syn::synthetic_oracle(tm, {{[](double f) { return syn::bounce(f); }, {}}, {syn::linear, lin}, {syn::step, stp},
                                 {syn::camera, {}}}),
      log);
  const Run first = run(rec, tm, scene, keys, 1, {}, keep_steps);
  CHECK(log.calls.size() == static_cast<size_t>(first.stats.messages));
  CHECK(std::all_of(log.calls.begin(), log.calls.end(), [](const mo::RawPoses& c) { return c.n <= 512; }));
  CHECK(first.stats.calls_by_stage.count("detect") == 1);
  CHECK(first.stats.calls_by_stage.count("burst") == 1);
  for (size_t b = 0; b < 3; ++b) CHECK_FALSE(first.sets[b].breakpoints.empty());
  for (const int workers : {8, 1}) {
    const auto n = std::make_shared<size_t>(0);
    const Run again = run(mo::replay(log, n), tm, scene, keys, workers, {}, keep_steps);
    CHECK(*n == log.calls.size());
    CHECK(again.stats == first.stats);
    for (size_t b = 0; b < first.sets.size(); ++b) CHECK(set_equal(again.sets[b], first.sets[b]));
  }
  // other options ask for other times: the replay stops at the first differing request
  mo::SampleOptions tighter;
  tighter.eps_px = 0.005;
  CHECK_THROWS_WITH_AS(run(mo::replay(log, std::make_shared<size_t>(0)), tm, scene, keys, 8, tighter, keep_steps),
                       doctest::Contains("BRG-REPLAY"), mo::MotionError);
}

TEST_CASE("drive: messages of at most the pipelines' smallest max_per_message, the same samples") {
  const mo::TimeMap tm = time24(25);
  const mo::MotionScene scene = scene_of(1);
  const mo::Oracle inner = syn::synthetic_oracle(tm, {{syn::smooth, {}}, {syn::camera, {}}});
  mo::OracleLog wide, narrow;
  const Run a = run(mo::recording(inner, wide), tm, scene, {}, 2);
  mo::SampleOptions small;
  small.max_per_message = 64;
  const Run b = run(mo::recording(inner, narrow), tm, scene, {}, 2, small);
  CHECK(std::all_of(narrow.calls.begin(), narrow.calls.end(), [](const mo::RawPoses& c) { return c.n <= 64; }));
  CHECK(b.stats.messages > a.stats.messages);
  CHECK(b.stats.calls == a.stats.calls);
  CHECK(b.stats.rounds == a.stats.rounds);
  for (size_t k = 0; k < a.sets.size(); ++k) CHECK(set_equal(a.sets[k], b.sets[k]));
  for (const int bad : {0, 513}) {
    mo::SampleOptions o;
    o.max_per_message = bad;
    CHECK_THROWS_AS(mo::BodyPipeline(0, mo::Mode::kKeyed, tm, {}, o), std::invalid_argument);
  }
}
