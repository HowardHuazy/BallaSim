// Breakpoint detection (src/scene/motion/detect.h, pipeline.h; docs/design/phase3/blender.md §4.7.5, §8.1 T6): the
// pure pieces on exact tracks, then the pipeline end to end against the synthetic oracles of motion_oracles.h (the
// prototype's t_pipeline.py motions with Blender's float32 time and matrices, and its key plateau where keys exist):
// bounce 8/8 and linear 11/11 with their MF timings, declared keys snapped exactly, undeclared kinks within
// (eps/8)/dv, steps C-1 and MOT-STEP, no breakpoint on smooth or C1 Bezier-rig motion.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/time/time.h"
#include "motion_oracles.h"
#include "scene/motion/detect.h"

namespace mo = balla::scene::motion;
namespace rot = balla::scene::rot;
namespace syn = balla::scene::motion::testing;
using C = mo::Continuity;
using syn::kEps;
using syn::kHmin;

namespace {

// A track of a point moving along x with position x(t) (m), sampled at the given times; rho 0.05 m at 0.95 m: the
// rig weights of MF (w_pos 674 px/m).
mo::Track track(const std::vector<int64_t>& t, double (*x)(double)) {
  mo::Track tr;
  for (const int64_t ti : t) {
    tr.t_ns.push_back(ti);
    tr.q.push_back({1, 0, 0, 0});
    tr.p.push_back({x(static_cast<double>(ti) * 1e-9), 0.0, 0.0});
    tr.w_pos.push_back(640.0 / 0.95);
    tr.w_rot.push_back(640.0 * 0.05 / 0.95);
    tr.floor1.push_back(0.0);
  }
  return tr;
}

std::vector<int64_t> grid(int64_t a, int64_t b, int64_t h) {
  std::vector<int64_t> t;
  for (int64_t x = a; x <= b; x += h) t.push_back(x);
  return t;
}

constexpr int64_t kKink = 503000000;  // 0.503 s
double kinked(double t) { return t < 0.503 ? 0.4 * t : 0.4 * 0.503 - 0.7 * (t - 0.503); }
double stepped(double t) { return t < 0.503 ? 0.1 * t : 0.1 * t + 0.002; }
double wavy(double t) { return 0.05 * std::sin(3.0 * t); }

int64_t calls_in(const syn::Run& r, const char* stage) {
  const auto it = r.stats.calls_by_stage.find(stage);
  return it == r.stats.calls_by_stage.end() ? 0 : it->second;
}

int64_t dist(int64_t a, int64_t b) { return a > b ? a - b : b - a; }

mo::TimeMap time_of(int32_t fps, int32_t frame_start, int32_t frame_end, int32_t map_old = 100, int32_t map_new = 100) {
  mo::TimeSettings s;
  s.fps = fps;
  s.frame_start = frame_start;
  s.frame_end = frame_end;
  s.frame_map_old = map_old;
  s.frame_map_new = map_new;
  return mo::TimeMap(s);
}

// the breakpoint nearest t, or nullptr
const mo::Breakpoint* near(const std::vector<mo::Breakpoint>& bps, int64_t t) {
  const mo::Breakpoint* best = nullptr;
  for (const auto& b : bps) {
    if (!best || dist(b.t_ns, t) < dist(best->t_ns, t)) best = &b;
  }
  return best;
}

}  // namespace

TEST_CASE("classify: a kink at its crossing, a step in its bracket, smooth motion smooth") {
  std::vector<int64_t> t = grid(0, 1000000000, 10416667);
  {
    const mo::Track tr = track(t, kinked);
    const mo::Classified c = mo::classify(tr, kKink, kEps, {kHmin}, 6.0);
    CHECK(c.kind == mo::Classified::Kind::kKink);
    CHECK(dist(c.t_ns, kKink) <= 2);  // exact lines cross at the kink
    CHECK(c.dv_pxps == doctest::Approx(1.1 * 640.0 / 0.95).epsilon(1e-6));
  }
  {  // a 2 mm step inside a 20 us bracket
    t.push_back(kKink - 10000);
    t.push_back(kKink + 10000);
    std::sort(t.begin(), t.end());
    const mo::Track tr = track(t, stepped);
    const mo::Classified c = mo::classify(tr, kKink, kEps, {kHmin}, 6.0);
    REQUIRE(c.kind == mo::Classified::Kind::kStep);
    CHECK(c.t_ns == kKink + 10000);
    CHECK(tr.t_ns[c.a] == kKink - 10000);
    CHECK(c.jump_px == doctest::Approx(0.002 * 640.0 / 0.95).epsilon(1e-3));
  }
  {
    const mo::Track tr = track(grid(0, 1000000000, 10416667), wavy);
    CHECK(mo::classify(tr, 500000000, kEps, {kHmin}, 6.0).kind == mo::Classified::Kind::kSmooth);
  }
  {  // a 2 mm step inside a 40.7 us bracket: wider than 2 h_min, but one float32 frame step past frame 2^13 at 24 fps
    std::vector<int64_t> tb = grid(0, 1000000000, 10416667);
    tb.push_back(kKink - 20345);
    tb.push_back(kKink + 20345);
    std::sort(tb.begin(), tb.end());
    const mo::Track tr = track(tb, stepped);
    const mo::TimeMap big = time_of(24, 9001, 9073);
    CHECK(mo::classify(tr, kKink, kEps, {kHmin}, 6.0).kind != mo::Classified::Kind::kStep);
    const mo::Classified c = mo::classify(tr, kKink, kEps, {kHmin, &big}, 6.0);
    REQUIRE(c.kind == mo::Classified::Kind::kStep);
    CHECK(c.t_ns == kKink + 20345);
    CHECK(c.v_pxps == doctest::Approx(0.1 * 640.0 / 0.95).epsilon(1e-6));  // 0.1 m/s on both sides
  }
}

TEST_CASE("StepBracket: 2 h_min; one float32 frame step past 2^13 frames; a dropped plateau adds 2 w") {
  const mo::TimeMap tm24 = syn::time24(73), big = time_of(24, 9001, 9073), slow = time_of(24, 1, 73, 100, 1000);
  CHECK(mo::StepBracket{kHmin}.limit_ns(0, 1000000) == 2 * kHmin);
  CHECK(mo::StepBracket{kHmin, &tm24}.limit_ns(0, 1000000) == 2 * kHmin);  // 2 q(F) < 2 h_min
  const int64_t g = static_cast<int64_t>(std::ceil(2.0 * big.resolution_ns(9013))) + 2;
  CHECK(g > 2 * kHmin);
  CHECK(mo::StepBracket{kHmin, &big}.limit_ns(500000000, 500040690) == g);
  // framelen 0.1: a declared key's plateau of 1e-3 output frames, 41.7 us
  const int64_t key = slow.frame_to_ns(2.0 / slow.framelen());
  const std::vector<mo::DeclaredKey> keys{{key, C::kStep, true, 1e-4 / slow.framelen()}};
  const mo::Plateaus pl(slow, keys);
  const int64_t w = std::llround(1e-4 / slow.framelen() * slow.period_ns());
  CHECK(w == 41667);
  CHECK(pl.dropped_ns(key - w - 20000, key) == 2 * w);
  CHECK(pl.dropped_ns(key, key + w + 20000) == 2 * w);
  CHECK(pl.dropped_ns(key, key + 2) == 0);              // key + 1 is within the key's resolution: kept
  CHECK(pl.dropped_ns(key - 3 * w, key - w - 1) == 0);  // (a, c) ends before the plateau
  CHECK(pl.dropped_ns(key + w, key + 3 * w) == 0);      // and starts at its end
  CHECK(mo::StepBracket{kHmin, &slow, &pl}.limit_ns(key - w - 20000, key) == 2 * kHmin + 2 * w);
}

TEST_CASE("candidates: unresolved intervals, spacing spikes, the node scan and declared step and C0 keys") {
  mo::SampleSet s;
  std::vector<int64_t> t = grid(0, 1000000000, 10416667);
  t.push_back(698516689);  // a spike: 0.6 ms after the sample at 697916689
  std::sort(t.begin(), t.end());
  s.fit = track(t, kinked);
  s.unresolved = {{200000000, 200020000}};
  const std::vector<mo::DeclaredKey> keys{{100000000, C::kC0, true, 1e-4}, {300000000, C::kC1, true, 1e-4},
                                          {400000000, C::kStep, true, 1e-4}};
  const std::vector<int64_t> c = mo::candidates(s, keys, kEps);
  auto has = [&](int64_t x, int64_t tol) {
    return std::any_of(c.begin(), c.end(), [&](int64_t v) { return dist(v, x) <= tol; });
  };
  CHECK(has(200010000, 0));           // the unresolved interval's midpoint
  CHECK(has(100000000, 0));           // a declared C0 key
  CHECK(has(400000000, 0));           // a declared step
  CHECK_FALSE(has(300000000, 0));     // C1 hints only after a validation failure (BL-8)
  CHECK(has(kKink, 10416667));        // the node scan, at the sample next to the kink
  CHECK(has(698216689, 0));           // the spike's midpoint
  CHECK(std::is_sorted(c.begin(), c.end()));
}

TEST_CASE("burst_requests and burst_crossing: 2n samples in [t-W, t-W/16] u [t+W/16, t+W]; the crossing") {
  const std::vector<int64_t> r = mo::burst_requests(1000000, 64000, 8);
  REQUIRE(r.size() == 16);
  CHECK(r.front() == 1000000 - 64000);
  CHECK(r[7] == 1000000 - 4000);
  CHECK(r[8] == 1000000 + 4000);
  CHECK(r.back() == 1000000 + 64000);
  // quadratics on either side of an exact kink: the crossing lands on it from 100 us away
  const mo::Track tr = track(mo::burst_requests(kKink + 100000, 2000000, 8), kinked);
  const mo::Crossing x = mo::burst_crossing(tr, kKink + 100000, 2000000, tr.w_pos[0], tr.w_rot[0]);
  REQUIRE(x.ok);
  CHECK(dist(x.t_ns, kKink) <= 1);
  CHECK(x.dv_pxps == doctest::Approx(1.1 * 640.0 / 0.95).epsilon(1e-6));
  CHECK_FALSE(mo::burst_crossing(track({1, 2, 3, 4}, kinked), 3, 10, 1.0, 1.0).ok);  // < 3 samples on a side
}

TEST_CASE("merge_breakpoints and snap_breakpoints") {
  const mo::TimeMap tm = syn::time24(49);
  std::vector<mo::Breakpoint> b{{500000600, C::kC0, 0, 800, false, false, 1500},
                                {500050000, C::kC0, 0, 900, false, false, 1300},
                                {500090000, C::kStep, 30, 5, false, false, 20},
                                {800000000, C::kC0, 0, 50, false, false, 25000}};
  std::vector<mo::Breakpoint> m = mo::merge_breakpoints(b, 100000);
  REQUIRE(m.size() == 2);  // within 100 us of the first kept one: the step wins
  CHECK(m[0].cls == C::kStep);
  CHECK(m[0].t_ns == 500090000);
  const int64_t key = tm.to_ns(13.0f);  // 0.5 s
  const std::vector<mo::DeclaredKey> keys{{key, C::kC0, true, 1e-4}, {800010000, C::kC0, false, 0.0}};
  std::vector<mo::Breakpoint> s{{key - 4166 - 1000, C::kC0, 0, 800, false, false, 1500},
                                {800001000, C::kC0, 0, 50, false, false, 25000},
                                {800020000, C::kC0, 0, 50, false, false, 25000}};
  CHECK(mo::snap_breakpoints(s, keys, tm) == 1);  // plateau 4167 ns + 1 us; no plateau: 1 us only
  CHECK(s[0].t_ns == key);
  CHECK(s[0].snapped);
  CHECK(s[0].declared);
  CHECK(s[0].dt_bound_ns == doctest::Approx(tm.resolution_ns(13)));
  CHECK_FALSE(s[1].snapped);
  CHECK_FALSE(s[2].snapped);
  std::vector<mo::Breakpoint> s2{{key + 4167 + 1001, C::kC0, 0, 800, false, false, 1500}};
  CHECK(mo::snap_breakpoints(s2, keys, tm) == 0);
  const std::vector<mo::DeclaredKey> unsorted{keys[1], keys[0]};
  CHECK_THROWS_AS(mo::snap_breakpoints(s2, unsorted, tm), std::invalid_argument);
}

TEST_CASE("snap_breakpoints visits only the keys near each breakpoint and agrees with a scan of every key") {
  const mo::TimeMap tm = syn::time24(49);
  std::mt19937_64 rng(7);
  std::vector<mo::DeclaredKey> keys;
  int64_t t = 0;
  for (int i = 0; i < 3000; ++i) {  // keys 0.5 .. 20 us apart, plateaus of 0 to 3e-4 frames (up to 12.5 us)
    t += 500 + static_cast<int64_t>(rng() % 19500);
    const bool plateau = rng() % 2 == 0;
    keys.push_back({t, C::kC0, plateau, plateau ? 3e-4 * static_cast<double>(rng() % 1001) / 1000.0 : 0.0});
  }
  std::vector<mo::Breakpoint> bps;
  for (int i = 0; i < 2000; ++i) {
    const int64_t at = keys[rng() % keys.size()].t_ns + static_cast<int64_t>(rng() % 30001) - 15000;
    bps.push_back({at, C::kC0, 0, 100, false, false, 1000});
  }
  std::vector<mo::Breakpoint> want = bps;
  int want_n = 0;
  for (mo::Breakpoint& b : want) {  // the nearest key within its plateau + 1 us, the first one on a tie
    const mo::DeclaredKey* best = nullptr;
    int64_t best_d = std::numeric_limits<int64_t>::max();
    for (const mo::DeclaredKey& k : keys) {
      const int64_t w = (k.plateau ? std::llround(k.plateau_frames * tm.period_ns()) : 0) + 1000;
      const int64_t d = std::abs(b.t_ns - k.t_ns);
      if (d <= w && d < best_d) {
        best = &k;
        best_d = d;
      }
    }
    if (best) {
      b.t_ns = best->t_ns;
      ++want_n;
    }
  }
  CHECK(mo::snap_breakpoints(bps, keys, tm) == want_n);
  CHECK(want_n > 500);
  CHECK(want_n < 2000);
  for (size_t i = 0; i < bps.size(); ++i) CHECK(bps[i].t_ns == want[i].t_ns);
}

TEST_CASE("T6 detection: bounce 8/8 undeclared impacts within (eps/8)/dv, MF t_pipeline timing (0-1 ns)") {
  const mo::TimeMap tm = syn::time24(73);
  std::vector<double> impacts;
  syn::bounce(1.0, &impacts);
  std::vector<int64_t> imp_ns;
  for (const double t : impacts) {
    if (t < 3.0) imp_ns.push_back(std::llround(t * 1e9));
  }
  CHECK(imp_ns.size() == 8);
  const auto calls = std::make_shared<int64_t>(0);
  const mo::Oracle o =
      syn::synthetic_oracle(tm, {{[](double f) { return syn::bounce(f); }, {}}, {syn::camera, {}}}, calls);
  const syn::Run r = syn::run(o, tm, syn::scene_of(1), {}, 2);
  const mo::SampleSet& s = r.sets[0];
  int64_t worst = 0;
  for (const int64_t t : imp_ns) {
    const mo::Breakpoint* b = near(s.breakpoints, t);
    REQUIRE(b);
    CHECK(b->cls == C::kC0);
    CHECK_FALSE(b->snapped);
    CHECK(static_cast<double>(dist(b->t_ns, t)) <= b->dt_bound_ns);
    CHECK(b->dt_bound_ns == doctest::Approx(kEps / 8.0 / b->dv_pxps * 1e9));
    worst = std::max(worst, dist(b->t_ns, t));
  }
  CHECK(s.breakpoints.size() == imp_ns.size());  // no false join
  CHECK(r.sets[1].breakpoints.empty());
  CHECK(worst <= 100);
  MESSAGE("bounce: " << s.breakpoints.size() << "/" << imp_ns.size() << " impacts, worst " << worst << " ns; "
                     << r.stats.calls << " calls (detect " << calls_in(r, "detect") << ", burst "
                     << calls_in(r, "burst") << "), " << r.stats.rounds << " rounds; candidates "
                     << s.detect.candidates << ", kinks " << s.detect.kinks << ", smooth " << s.detect.smooth
                     << ", unclear " << s.detect.unclear << ", insignificant " << s.detect.insignificant);
  // a kink costs one burst level of 2 x 8 samples per burst: 64 calls (MF E7: about 64 per candidate)
  CHECK(calls_in(r, "burst") <= 64 * (s.detect.kinks + s.detect.insignificant));
}

TEST_CASE("T6 detection: linear 11/11 - undeclared kinks within (eps/8)/dv, declared ones snapped exactly") {
  const mo::TimeMap tm = syn::time24(73);
  const std::vector<double> keys = syn::linear_keys();  // frames 1, 7, ..., 73
  std::vector<int64_t> interior;
  for (size_t k = 1; k + 1 < keys.size(); ++k) interior.push_back(tm.to_ns(static_cast<float>(keys[k])));
  REQUIRE(interior.size() == 11);
  const std::vector<mo::DeclaredKey> decl = syn::declared(tm, keys, C::kC0);
  // bodies: 0 no plateau, undeclared (t_pipeline); 1 Blender's plateau, undeclared (as E7 before snapping);
  // 2 Blender's plateau, declared
  const mo::Oracle o = syn::synthetic_oracle(
      tm, {{syn::linear, {}}, {syn::linear, keys}, {syn::linear, keys}, {syn::camera, {}}});
  const syn::Run r = syn::run(o, tm, syn::scene_of(3), {{}, {}, decl}, 4);
  auto ends_or_interior = [&](const mo::SampleSet& s) {  // every breakpoint at a key (the ends kink as well)
    return std::all_of(s.breakpoints.begin(), s.breakpoints.end(), [&](const mo::Breakpoint& b) {
      return std::any_of(keys.begin(), keys.end(), [&](double f) {
        return static_cast<double>(dist(b.t_ns, tm.to_ns(static_cast<float>(f)))) <= 5000.0 + b.dt_bound_ns;
      });
    });
  };
  for (size_t body = 0; body < 3; ++body) {
    const mo::SampleSet& s = r.sets[body];
    int64_t worst = 0;
    for (const int64_t t : interior) {
      const mo::Breakpoint* b = near(s.breakpoints, t);
      REQUIRE(b);
      CHECK(b->cls == C::kC0);
      const int64_t err = dist(b->t_ns, t);
      worst = std::max(worst, err);
      if (body == 2) {
        CHECK(b->snapped);
        CHECK(err == 0);
        CHECK(b->dt_bound_ns <= tm.resolution_ns(73));
      } else {
        CHECK_FALSE(b->snapped);
        CHECK(static_cast<double>(err) <= b->dt_bound_ns);
      }
    }
    CHECK(ends_or_interior(s));
    MESSAGE("linear body " << body << ": " << s.breakpoints.size() << " breakpoints (11 interior), worst interior "
                           << worst << " ns; kinks " << s.detect.kinks << ", snapped " << s.detect.snapped);
    if (body == 0) CHECK(worst <= 100);  // MF t_pipeline: <= 18 ns
  }
  MESSAGE("linear: " << r.stats.calls << " calls (burst " << calls_in(r, "burst") << "), "
                     << r.stats.rounds << " rounds");
}

TEST_CASE("T6 detection: steps are C-1, snapped to their keys, and refused (MOT-STEP) in v1") {
  const mo::TimeMap tm = syn::time24(73);
  const std::vector<double> stp = syn::step_keys();
  const std::vector<mo::DeclaredKey> decl = syn::declared(tm, {stp.begin() + 1, stp.end()}, C::kStep);
  const mo::Oracle o = syn::synthetic_oracle(tm, {{syn::step, stp}, {syn::step, stp}, {syn::camera, {}}});
  CHECK_THROWS_WITH_AS(syn::run(o, tm, syn::scene_of(2), {decl, {}}, 2), doctest::Contains("MOT-STEP"),
                       mo::MotionError);
  mo::DetectOptions keep;
  keep.refuse_steps = false;
  const syn::Run r = syn::run(o, tm, syn::scene_of(2), {decl, {}}, 2, {}, keep);
  for (size_t body = 0; body < 2; ++body) {
    const mo::SampleSet& s = r.sets[body];
    std::vector<mo::Breakpoint> steps;
    for (const auto& b : s.breakpoints) {
      if (b.cls == C::kStep) steps.push_back(b);
    }
    REQUIRE(steps.size() == 5);
    for (size_t k = 0; k < 5; ++k) {
      const int64_t key = tm.to_ns(static_cast<float>(stp[k + 1]));
      if (body == 0) {  // declared: exactly at the key (Q2)
        CHECK(steps[k].t_ns == key);
        CHECK(steps[k].snapped);
      } else {  // undeclared: where Blender steps, 1e-4 frames early, to the bisection's last bracket
        int32_t F = 0;
        float sub = 0.0f;
        tm.request(key, &F, &sub);
        const double early = static_cast<double>(key) - 1e-4 * tm.period_ns();
        CHECK(std::fabs(static_cast<double>(steps[k].t_ns) - early) <= steps[k].dt_bound_ns + 2 * tm.resolution_ns(F));
        CHECK_FALSE(steps[k].snapped);
      }
      CHECK(steps[k].jump_px > kEps / 4.0);
    }
    for (const auto& b : s.breakpoints) {  // anything else is the rotation's smooth ends: none
      CHECK(b.cls == C::kStep);
    }
  }
  MESSAGE("step: " << r.stats.calls << " calls (detect " << calls_in(r, "detect") << ", burst "
                   << calls_in(r, "burst") << "), " << r.stats.rounds << " rounds");
}

namespace {

// A step body whose 5 CONSTANT keys (scene frames keys[1..5]) end in brackets wider than 2 h_min: Blender's plateau
// (`plateau` scene frames, 1e-4 N_s) is dropped around a declared key, or the float32 time grid is coarser than h_min.
struct WideStep {
  const char* name;
  mo::TimeMap tm;
  std::function<syn::Pose(double)> motion;
  std::vector<double> keys;  // scene frames; keys[0] starts the motion
  double plateau = 1e-4;
};

}  // namespace

TEST_CASE("T6 detection: steps in brackets wider than 2 h_min - a dropped plateau, Blender's coarse time grid") {
  const auto shifted = [](double by, syn::Pose (*m)(double)) { return [by, m](double f) { return m(f - by); }; };
  std::vector<double> at_9001;
  for (const double k : syn::step_keys()) at_9001.push_back(k + 9000.0);
  const std::vector<WideStep> scenes{
      // framelen 0.1 (frame_map 100 / 1000): the plateau is 1e-3 output frames, 41.7 us
      {"framelen 0.1", time_of(24, 1, 73, 100, 1000),
       [](double f) {  // steps at scene frames 2 .. 6 (output frames 20 .. 60)
         const int k = std::clamp(static_cast<int>(std::floor(f - 1.0)), 0, 5);
         return syn::Pose{syn::qexp({0.5 * std::sin(f / 2.4), 0.0, 0.0}), {0.02 * k, 0.0, 0.01 * (k % 2)}};
       },
       {1, 2, 3, 4, 5, 6}},
      // 3 fps: 1e-4 frames are 33.3 us
      {"3 fps", time_of(3, 1, 73), syn::step, syn::step_keys()},
      // a key under an NLA strip of scale 8 at 24 fps: 8e-4 frames, 33.3 us
      {"NLA scale 8", time_of(24, 1, 73), syn::step, syn::step_keys(), 8e-4},
      // frames past 2^13 at 24 fps: one float32 frame step (2 q(F)) is 40.7 us
      {"frame 9001", time_of(24, 9001, 9073), shifted(9000.0, syn::step), at_9001},
  };
  for (const WideStep& sc : scenes) {
    const std::string name = sc.name;
    CAPTURE(name);
    const mo::TimeMap& tm = sc.tm;
    const double fl = tm.framelen();
    std::vector<mo::DeclaredKey> decl;
    std::vector<int64_t> key_ns;
    for (size_t k = 1; k < sc.keys.size(); ++k) {  // as declared_keys gives them
      key_ns.push_back(tm.frame_to_ns(sc.keys[k] / fl));
      decl.push_back({key_ns.back(), C::kStep, true, sc.plateau / fl});
    }
    const syn::SyntheticBody stepper{sc.motion, sc.keys, 1.0, sc.plateau};
    for (const bool declared : {true, false}) {  // v1 refuses either
      CAPTURE(declared);
      CHECK_THROWS_WITH_AS(syn::run(syn::synthetic_oracle(tm, {stepper, {syn::camera, {}}}), tm, syn::scene_of(1),
                                    {declared ? decl : std::vector<mo::DeclaredKey>{}}, 1),
                           doctest::Contains("MOT-STEP"), mo::MotionError);
    }
    const mo::Oracle o = syn::synthetic_oracle(tm, {stepper, stepper, {syn::camera, {}}});
    mo::DetectOptions keep;
    keep.refuse_steps = false;
    const syn::Run r = syn::run(o, tm, syn::scene_of(2), {decl, {}}, 2, {}, keep);
    for (size_t body = 0; body < 2; ++body) {
      CAPTURE(body);
      const mo::SampleSet& s = r.sets[body];
      std::vector<mo::Breakpoint> steps;
      for (const auto& b : s.breakpoints) {
        CHECK(b.cls == C::kStep);  // the rotation is smooth
        if (b.cls == C::kStep) steps.push_back(b);
      }
      CHECK(steps.size() == 5);
      for (size_t k = 0; k < std::min<size_t>(5, steps.size()); ++k) {
        int32_t F = 0;
        float sub = 0.0f;
        tm.request(key_ns[k], &F, &sub);
        if (body == 0) {
          CHECK(steps[k].t_ns == key_ns[k]);
          CHECK(steps[k].snapped);
        } else {  // Blender steps a plateau early, found to the bisection's last bracket (one grid step at most)
          const double early = static_cast<double>(key_ns[k]) - sc.plateau / fl * tm.period_ns();
          CHECK(std::fabs(static_cast<double>(steps[k].t_ns) - early) <=
                steps[k].dt_bound_ns + 2 * tm.resolution_ns(F));
          CHECK(steps[k].dt_bound_ns <= 2 * tm.resolution_ns(F) + 2);
          CHECK_FALSE(steps[k].snapped);
        }
        CHECK(steps[k].jump_px > 10.0);  // 2 cm at 1 m
      }
      MESSAGE(name << " body " << body << ": candidates " << s.detect.candidates << ", steps " << s.detect.steps
                      << ", smooth " << s.detect.smooth << ", demoted " << s.detect.demoted << "; "
                      << r.stats.calls << " calls (detect " << calls_in(r, "detect") << ")");
    }
  }
}

TEST_CASE("T6 detection: kinks between frames past 2^13 stay kinks (no step from the grid's coarse brackets)") {
  // One float32 frame step is 40.7 us there. LINEAR keys at whole frames are grid points; a bounce's impacts fall
  // inside a grid step, where the one-sided fits disagree by up to dv 2 q(F) (> eps/4) and bisection cannot split the
  // bracket: the motion over it explains the end poses' distance, so it is a kink (demoted), never a step.
  const mo::TimeMap tm = time_of(24, 9001, 9073);
  std::vector<double> keys;
  for (const double k : syn::linear_keys()) keys.push_back(k + 9000.0);
  std::vector<int64_t> interior;
  for (size_t k = 1; k + 1 < keys.size(); ++k) interior.push_back(tm.to_ns(static_cast<float>(keys[k])));
  std::vector<double> impacts;
  syn::bounce(1.0, &impacts);
  std::vector<int64_t> imp_ns;
  for (const double t : impacts) {
    if (t < 3.0) imp_ns.push_back(std::llround(t * 1e9));
  }
  const auto lin = [](double f) { return syn::linear(f - 9000.0); };
  const auto bnc = [](double f) { return syn::bounce(f - 9000.0); };
  const mo::Oracle o = syn::synthetic_oracle(tm, {{lin, keys}, {lin, keys}, {bnc, {}}, {syn::camera, {}}});
  const syn::Run r = syn::run(o, tm, syn::scene_of(3), {syn::declared(tm, keys, C::kC0), {}, {}}, 2);
  for (size_t body = 0; body < 3; ++body) {
    CAPTURE(body);
    const mo::SampleSet& s = r.sets[body];
    const std::vector<int64_t>& breaks = body < 2 ? interior : imp_ns;
    int64_t worst = 0;
    for (const int64_t t : breaks) {
      const mo::Breakpoint* b = near(s.breakpoints, t);
      REQUIRE(b);
      CHECK(b->cls == C::kC0);
      const int64_t err = dist(b->t_ns, t);
      worst = std::max(worst, err);
      if (body == 0) CHECK(err == 0);
      if (body == 2) CHECK(static_cast<double>(err) <= std::max(b->dt_bound_ns, 2 * tm.resolution_ns(9073)));
    }
    for (const auto& b : s.breakpoints) CHECK(b.cls == C::kC0);
    CHECK(s.detect.steps == 0);
    if (body == 2) CHECK(s.breakpoints.size() == imp_ns.size());
    MESSAGE("frame 9001, body " << body << ": " << s.breakpoints.size() << " breakpoints, worst " << worst
                                << " ns; candidates " << s.detect.candidates << ", demoted " << s.detect.demoted);
  }
}

TEST_CASE("T6 detection: LINEAR keys under a wide dropped plateau (framelen 0.1) stay kinks, snapped when declared") {
  // keys at scene frames 1 + 0.6 k, output frames 10 + 6 k; the plateau is 1e-3 output frames (41.7 us)
  const mo::TimeMap tm = time_of(24, 1, 73, 100, 1000);
  const double fl = tm.framelen();
  std::vector<double> keys;
  for (int k = 0; k <= 10; ++k) keys.push_back(1.0 + 0.6 * k);
  std::vector<mo::DeclaredKey> decl;
  std::vector<int64_t> interior;
  for (size_t k = 0; k < keys.size(); ++k) {
    decl.push_back({tm.frame_to_ns(keys[k] / fl), C::kC0, true, 1e-4 / fl});
    if (k > 0 && k + 1 < keys.size()) interior.push_back(decl.back().t_ns);
  }
  const auto lin = [](double f) { return syn::linear(1.0 + (f - 1.0) * 10.0); };
  const mo::Oracle o = syn::synthetic_oracle(tm, {{lin, keys}, {lin, keys}, {syn::camera, {}}});
  const syn::Run r = syn::run(o, tm, syn::scene_of(2), {decl, {}}, 2);
  for (size_t body = 0; body < 2; ++body) {
    CAPTURE(body);
    const mo::SampleSet& s = r.sets[body];
    int64_t worst = 0;
    for (const int64_t t : interior) {
      const mo::Breakpoint* b = near(s.breakpoints, t);
      REQUIRE(b);
      CHECK(b->cls == C::kC0);
      const int64_t err = dist(b->t_ns, t);
      worst = std::max(worst, err);
      if (body == 0) CHECK(err == 0);
    }
    for (const auto& b : s.breakpoints) CHECK(b.cls == C::kC0);
    CHECK(s.detect.steps == 0);
    MESSAGE("framelen 0.1 linear, body " << body << ": " << s.breakpoints.size() << " breakpoints, worst " << worst
                                         << " ns; candidates " << s.detect.candidates << ", demoted "
                                         << s.detect.demoted << "; " << calls_in(r, "detect") << " detect calls");
  }
}

TEST_CASE("T6 detection: no breakpoint on smooth motion or a C1 Bezier rig") {
  const mo::TimeMap tm = syn::time24(73);
  const std::vector<mo::DeclaredKey> rig_keys = syn::declared(tm, syn::rig_keys(), C::kC1);
  const mo::Oracle o = syn::synthetic_oracle(tm, {{syn::smooth, {}}, {syn::rig, syn::rig_keys()}, {syn::camera, {}}});
  const syn::Run r = syn::run(o, tm, syn::scene_of(2), {{}, rig_keys}, 2);
  int insignificant = 0;
  for (const auto& s : r.sets) {
    CHECK(s.breakpoints.empty());
    CHECK(s.detect.kinks == 0);
    CHECK(s.detect.steps == 0);
    insignificant += s.detect.insignificant;
    MESSAGE("candidates " << s.detect.candidates << ": smooth " << s.detect.smooth << ", unclear " << s.detect.unclear
                          << ", kinks " << s.detect.kinks << ", insignificant " << s.detect.insignificant);
  }
  CHECK(r.sets[0].detect.insignificant == 0);  // smooth: classification alone, no oracle call
  // a curvature jump next to a rig key can look like a kink through the cubics; its bursts show dv below the split
  // filter (MF E7's cameras: one "C0-insignificant" each); one burst level per kink
  CHECK(calls_in(r, "detect") == 0);
  CHECK(calls_in(r, "burst") <= 64 * insignificant);
}

TEST_CASE("T6 detection: options are checked; burst_iters 0 keeps the classification's estimate") {
  const mo::TimeMap tm = syn::time24(25);
  for (const auto& bad : {mo::DetectOptions{0.0}, mo::DetectOptions{6.0, 0.1, 2}}) {
    CHECK_THROWS_AS(mo::BodyPipeline(0, mo::Mode::kKeyed, tm, {}, {}, bad), std::invalid_argument);
  }
  mo::DetectOptions none;
  none.burst_iters = 0;
  const std::vector<double> keys = syn::linear_keys();
  const syn::Run r = syn::run(syn::synthetic_oracle(tm, {{syn::linear, {}}, {syn::camera, {}}}), tm,
                              syn::scene_of(1), {}, 1, {}, none);
  CHECK(r.stats.calls_by_stage.count("burst") == 0);
  CHECK(r.sets[0].breakpoints.size() >= 3);  // keys 7, 13, 19 within frames 1 .. 25
}
