// The piecewise fit and its validation against synthetic and recorded oracles (docs/design/phase3/blender.md §4.7.6,
// §8.1 T6 and T7; BL-8): fits within eps/2 of the truth (smooth, bounce, LINEAR keys, a Bezier rig split at its C1
// keys), holds as bitwise-still pieces, joins that keep the pose, the breakpoint-probe case of DS d14 (refused or
// fixed by halving the left piece, accepted above eps/2 without the probes), a recorded oracle replaying the fits
// bitwise at 1 and 8 workers; the rigs (§4.7.9), the mode by provenance (§4.7.1) and a fitted body's report (§6.1).
#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/time/time.h"
#include "core/traj/pieces.h"
#include "motion_oracles.h"
#include "render/thread_pool.h"
#include "scene/motion/checks.h"
#include "scene/motion/fit_piecewise.h"
#include "scene/motion/pipeline.h"

namespace mo = balla::scene::motion;
namespace rot = balla::scene::rot;
namespace syn = balla::scene::motion::testing;
namespace bl = balla::scene::blender;
using balla::scene::Tree;

namespace {

using syn::declared;
using syn::kEps;
using syn::run;
using syn::Run;
using syn::scene_of;
using syn::time24;

constexpr int64_t kW = balla::time::kWindowNs;

double angle(const rot::Q& a, const rot::Q& b) { return norm(balla::so3::log(conjugate(a) * b)); }
bool same_bits(double x, double y) { return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y); }

int64_t stage_calls(const Run& r, const char* stage) {
  const auto it = r.stats.calls_by_stage.find(stage);
  return it == r.stats.calls_by_stage.end() ? 0 : it->second;
}

// The truth without Blender's key plateau (the motion itself), float32 matrices as Blender gives them, decoded and
// weighed as the pipeline sees its samples: every 1/per_frame frame over [t0, t1], and around each time in `near`
// (+-1 ns .. +-4 us, where a fit's error at a join peaks).
mo::Track truth(const mo::TimeMap& tm, const mo::MotionScene& scene, const std::vector<syn::SyntheticBody>& bodies,
                int32_t body, int64_t t0, int64_t t1, int per_frame, const std::vector<int64_t>& near = {}) {
  std::vector<syn::SyntheticBody> plain = bodies;
  for (auto& b : plain) b.keys.clear();
  std::vector<int64_t> times;
  const double step = tm.period_ns() / per_frame;
  for (double t = static_cast<double>(t0); t <= static_cast<double>(t1); t += step) times.push_back(std::llround(t));
  for (const int64_t b : near) {
    for (const int64_t d : {-4000, -1000, -300, -100, -10, -1, 0, 1, 10, 100, 300, 1000, 4000}) {
      if (b + d >= t0 && b + d <= t1) times.push_back(b + d);
    }
  }
  std::sort(times.begin(), times.end());
  std::vector<int32_t> F;
  std::vector<float> s;
  for (const int64_t t : times) {
    int32_t f = 0;
    float sf = 0.0f;
    tm.request(t, &f, &sf);
    if (!F.empty() && F.back() == f && std::bit_cast<uint32_t>(s.back()) == std::bit_cast<uint32_t>(sf)) continue;
    F.push_back(f);
    s.push_back(sf);
  }
  mo::RawPoses raw = syn::synthetic_oracle(tm, plain)(F, s);
  raw.F = F;
  raw.s = s;
  mo::PoseBatch b = mo::decode(raw, tm, scene);
  const mo::ErrorModel model(scene);
  model.weigh(b);
  mo::Track tr;
  for (size_t i = 0; i < b.size(); ++i) tr.push(b, i, body);
  return tr;
}

// The fit's error (px) at sample i of a track: the piece serving its time (right-continuous).
double fit_error(const mo::PiecewiseFit& f, const mo::Track& tr, size_t i) {
  size_t k = 0;
  while (k + 1 < f.pieces.size() && f.pieces[k + 1].begin_ns <= tr.t_ns[i]) ++k;
  const mo::Pose p = mo::PieceEval(f.pieces[k])(tr.t_ns[i]);
  return tr.w_pos[i] * norm(p.p - tr.p[i]) + tr.w_rot[i] * angle(tr.q[i], p.q);
}

struct Dense {
  double max_px = 0.0;
  int64_t at_ns = 0;
};
// The fit against the truth, leaving out the samples within skip_ns of a time in `skip` (Blender's key plateaus).
Dense dense(const mo::PiecewiseFit& f, const mo::Track& tr, const std::vector<int64_t>& skip = {}, int64_t skip_ns = 0) {
  Dense d;
  for (size_t i = 0; i < tr.size(); ++i) {
    if (std::any_of(skip.begin(), skip.end(), [&](int64_t k) { return std::llabs(tr.t_ns[i] - k) <= skip_ns; })) continue;
    const double e = fit_error(f, tr, i);
    if (e > d.max_px) {
      d.max_px = e;
      d.at_ns = tr.t_ns[i];
    }
  }
  return d;
}

std::vector<int64_t> begins(const mo::PiecewiseFit& f) {
  std::vector<int64_t> b;
  for (const auto& p : f.pieces) b.push_back(p.begin_ns);
  return b;
}

// Every join labelled C0 keeps the pose (core/traj/pieces.h kJoinTol*), as the engine checks it.
void check_joins(const mo::PiecewiseFit& f) {
  for (size_t k = 1; k < f.pieces.size(); ++k) {
    if (f.pieces[k].join == mo::Continuity::kStep) continue;
    const int64_t b = f.pieces[k].begin_ns;
    const mo::Pose l = mo::PieceEval(f.pieces[k - 1])(b), r = mo::PieceEval(f.pieces[k])(b);
    CHECK(norm(l.p - r.p) <= balla::traj::kJoinTolM);
    CHECK(angle(l.q, r.q) <= balla::traj::kJoinTolRad);
  }
}

// C1's rules for the trajectory (CONTRACTS.md C1 `pieces`, bundle.py Trajectory.validate): one spline or P >= 2
// pieces, begins increasing from span_begin, rows following each other (>= K each), supports covering the intervals,
// piece 0's t0 and h on top, consecutive controls within pi/2, unit quaternions; a hold's piece holds one control.
void check_c1(const balla::render::Trajectory& t, const std::vector<std::pair<int64_t, int64_t>>& holds) {
  const auto rows = static_cast<int32_t>(t.rot.size() / 4);
  CHECK(t.order == 6);
  CHECK(t.pos.size() == 3 * t.rot.size() / 4);
  auto q = [&](int32_t c) {
    const auto i = static_cast<size_t>(4 * c);
    return rot::Q{t.rot[i], t.rot[i + 1], t.rot[i + 2], t.rot[i + 3]};
  };
  std::vector<balla::render::TrajectoryPiece> table = t.pieces;
  CHECK((t.pieces.empty() || t.pieces.size() >= 2));
  if (table.empty()) table.push_back({t.span_begin_ns, t.t0_ns, t.h_ns, 0, rows, 0});
  CHECK(table[0].begin_ns == t.span_begin_ns);
  CHECK(table[0].t0_ns == t.t0_ns);
  CHECK(table[0].h_ns == t.h_ns);
  int32_t next = 0;
  for (size_t k = 0; k < table.size(); ++k) {
    const auto& r = table[k];
    const int64_t end = k + 1 < table.size() ? table[k + 1].begin_ns : t.span_end_ns;
    CHECK(r.begin_ns < end);
    CHECK(r.ctrl_begin == next);
    CHECK(r.ctrl_end - r.ctrl_begin >= 6);
    CHECK(r.t0_ns <= r.begin_ns);
    CHECK(r.t0_ns + (r.ctrl_end - r.ctrl_begin - 5) * r.h_ns >= end);
    CHECK((k == 0 ? r.join == 0 : (r.join >= -1 && r.join <= 1)));
    for (int32_t c = r.ctrl_begin + 1; c < r.ctrl_end; ++c) CHECK(angle(q(c - 1), q(c)) <= std::numbers::pi / 2);
    next = r.ctrl_end;
  }
  CHECK(next == rows);
  for (int32_t c = 0; c < rows; ++c) {
    const rot::Q x = q(c);
    CHECK(std::fabs(std::sqrt(x.w * x.w + x.x * x.x + x.y * x.y + x.z * x.z) - 1.0) <= 1e-12);
  }
  for (const auto& h : holds) {
    const auto it = std::find_if(table.begin(), table.end(), [&](const auto& r) { return r.begin_ns == h.first; });
    REQUIRE(it != table.end());
    for (int32_t c = it->ctrl_begin; c < it->ctrl_end; ++c) {
      const auto i = static_cast<size_t>(c), i0 = static_cast<size_t>(it->ctrl_begin);
      CHECK(std::memcmp(&t.rot[4 * i], &t.rot[4 * i0], 4 * sizeof(double)) == 0);
      CHECK(std::memcmp(&t.pos[3 * i], &t.pos[3 * i0], 3 * sizeof(double)) == 0);
    }
  }
}

mo::FitOptions fit_on() { return mo::FitOptions{}; }

// MF E7's spin_child as a synthetic motion: a parent bobbing in z through AUTO_CLAMPED keys (flat handles at each
// extreme key: smoothstep segments, the curvature jumping at every key, C1) carries a child spinning at a constant
// rate from frame 1 to 49 (LINEAR rotation keys, held after 49: a C0 kink); after frame 49 the parent drifts on in x,
// so no hold follows the kink.
double bob(double frame) {
  const double k[5] = {1, 13, 25, 37, 49}, v[5] = {0, 0.3, 0, 0.3, 0};
  if (frame <= k[0]) return v[0];
  if (frame >= k[4]) return v[4];
  int i = 0;
  while (frame > k[i + 1]) ++i;
  const double u = (frame - k[i]) / (k[i + 1] - k[i]);
  return v[i] + (v[i + 1] - v[i]) * u * u * (3.0 - 2.0 * u);
}
syn::Pose spin_child(double frame) {
  const double th = 4.0 * std::numbers::pi * std::clamp((frame - 1.0) / 48.0, 0.0, 1.0);
  const double drift = frame > 49.0 ? 0.004 * (frame - 49.0) : 0.0;
  return {syn::qexp({0.0, 0.0, th}), {-0.2 + drift, 0.0, bob(frame)}};
}
// The same child with nothing after frame 49: a hold follows the kink.
syn::Pose spin_child_still(double frame) {
  const double th = 4.0 * std::numbers::pi * std::clamp((frame - 1.0) / 48.0, 0.0, 1.0);
  return {syn::qexp({0.0, 0.0, th}), {-0.2, 0.0, bob(frame)}};
}

// Piecewise-linear motion through n LINEAR keys `spacing` frames apart from `first` (Bake to Keyframes writes one per
// frame), constant outside them; steps of up to 0.08 m per key (about 0.6 m/s on average at one key per frame)
syn::Pose dense_linear(double frame, double first, double spacing, int n, uint32_t seed) {
  auto value = [&](int k, int axis) {
    const uint32_t h = static_cast<uint32_t>(k) * 2654435761u + static_cast<uint32_t>(axis) * 40503u + seed * 97u + 12345u;
    return -0.04 + 0.08 * static_cast<double>(h % 10007u) / 10006.0;
  };
  const double u = std::clamp((frame - first) / spacing, 0.0, static_cast<double>(n - 1));
  const int k = std::min(static_cast<int>(std::floor(u)), n - 2);
  const double a = u - k;
  const rot::V p{(1 - a) * value(k, 0) + a * value(k + 1, 0), (1 - a) * value(k, 1) + a * value(k + 1, 1),
                 (1 - a) * value(k, 2) + a * value(k + 1, 2)};
  return {syn::qexp({0.0, 0.0, 0.3 * (frame - 1.0) / 24.0}), p};
}
std::vector<double> frames_from(double first, double spacing, int n) {
  std::vector<double> f;
  for (int i = 0; i < n; ++i) f.push_back(first + spacing * i);
  return f;
}
// Parabolic arcs of 0.75 frames (impacts in between: velocity kinks, BOUNCE's breaks, which are no F-curve keys) from
// `first` on, still outside them.
syn::Pose short_arcs(double frame, double first, int n) {
  const double u = std::clamp((frame - first) / 0.75, 0.0, static_cast<double>(n));
  const double w = u - std::floor(u);
  const double z = u >= n ? 0.0 : 0.015 * 4.0 * w * (1.0 - w);
  return {syn::qexp({0.1, 0.0, 0.02 * (frame - 1.0)}), {0.0, 0.0, z}};
}

}  // namespace

TEST_CASE("T6 fit: smooth motion is one piece within eps/2 of the truth; a static camera one hold") {
  const mo::TimeMap tm = time24(73);
  const std::vector<syn::SyntheticBody> bodies{{syn::smooth, {}}, {syn::camera, {}}};
  const mo::MotionScene scene = scene_of(1);
  const Run r = run(syn::synthetic_oracle(tm, bodies), tm, scene, {}, 2, {}, {}, fit_on());
  const mo::SampleSet& s = r.sets[0];
  REQUIRE(s.fitted);
  const mo::PiecewiseFit& f = s.motion;
  CHECK(f.pieces.size() == 1);  // no joins on smooth motion
  CHECK(f.breakpoints.empty());
  CHECK(f.holds.empty());
  CHECK(f.span_begin_ns == -2 * kW);
  CHECK(f.span_end_ns == tm.t_end_ns());
  CHECK(f.max_validation_px <= 0.4 * kEps + f.floor_px);
  CHECK(f.validation_samples >= static_cast<int64_t>(s.held_out.size()));
  const Dense d = dense(f, truth(tm, scene, bodies, 0, -2 * kW, tm.t_end_ns(), 64));
  MESSAGE("smooth: h " << f.pieces[0].h_ns << " ns, validation " << f.max_validation_px << " px, dense " << d.max_px
                       << " px; calls " << r.stats.calls << " (validate " << stage_calls(r, "validate") << ")");
  CHECK(d.max_px <= kEps / 2.0);
  // h = clamp(2 x median spacing, 1 ms, 2 h0): the base grid's 10.4 ms at 24 fps gives 2 h0 = one frame, shortened so
  // that a whole number of knot spacings ends at the span's end
  const int64_t h2 = 2 * mo::h0_ns(tm, mo::SampleOptions{}), len = f.span_end_ns - f.span_begin_ns;
  CHECK(f.pieces[0].h_ns <= h2);
  CHECK(f.pieces[0].h_ns > h2 - h2 / 100);
  CHECK(f.pieces[0].t0_ns + (static_cast<int64_t>(f.pieces[0].rot.size()) - 5) * f.pieces[0].h_ns - f.span_end_ns <
        static_cast<int64_t>(f.pieces[0].rot.size()));
  (void)len;
  check_c1(mo::to_trajectory(f, {0.0, 0.0, 0.0}), f.holds);
  // the static camera: one hold piece over the span, K equal controls at knot spacing 2^30 ns (a static rig's)
  const mo::PiecewiseFit& cam = r.sets[1].motion;
  REQUIRE(cam.pieces.size() == 1);
  CHECK(cam.pieces[0].hold);
  REQUIRE(cam.holds.size() == 1);
  CHECK(cam.holds[0] == std::pair<int64_t, int64_t>{-2 * kW, tm.t_end_ns()});
  CHECK(cam.pieces[0].h_ns == mo::kHoldKnotNs);
  const balla::render::Trajectory ct = mo::to_trajectory(cam, {0.0, -1.0, 0.0});
  CHECK(ct.pieces.empty());
  CHECK(ct.pos[1] == 0.0);  // relative to origin_world
  check_c1(ct, cam.holds);
}

TEST_CASE("T6 fit: bounce impacts, LINEAR keys and a Bezier rig split at its C1 keys, within eps/2 of the truth") {
  const mo::TimeMap tm = time24(73);
  const std::vector<double> lin = syn::linear_keys(), rk = syn::rig_keys();
  const std::vector<syn::SyntheticBody> bodies{
      {[](double f) { return syn::bounce(f); }, {}}, {syn::linear, lin}, {syn::rig, rk}, {syn::camera, {}}};
  const mo::MotionScene scene = scene_of(3);
  const std::vector<std::vector<mo::DeclaredKey>> keys{
      {}, declared(tm, lin, mo::Continuity::kC0), declared(tm, rk, mo::Continuity::kC1)};
  const Run r = run(syn::synthetic_oracle(tm, bodies), tm, scene, keys, 4, {}, {}, fit_on());
  std::vector<int64_t> lin_ns, rig_ns;
  for (const double f : lin) lin_ns.push_back(tm.to_ns(static_cast<float>(f)));
  for (const double f : rk) rig_ns.push_back(tm.to_ns(static_cast<float>(f)));
  const char* names[] = {"bounce", "linear", "rig"};
  for (size_t b = 0; b < 3; ++b) {
    REQUIRE(r.sets[b].fitted);
    const mo::PiecewiseFit& f = r.sets[b].motion;
    check_joins(f);
    check_c1(mo::to_trajectory(f, {0.1, 0.0, 0.0}), f.holds);
    const std::vector<int64_t>& skip = b == 1 ? lin_ns : b == 2 ? rig_ns : std::vector<int64_t>{};
    const Dense d = dense(f, truth(tm, scene, bodies, static_cast<int32_t>(b), -2 * kW, tm.t_end_ns(), 64, begins(f)),
                          skip, 4200);
    MESSAGE(std::string(names[b]) << ": " << f.pieces.size() << " pieces, " << f.breakpoints.size()
                                  << " breakpoints, validation " << f.max_validation_px << " px, dense " << d.max_px
                                  << " px at " << d.at_ns << " ns");
    CHECK(d.max_px <= kEps / 2.0);
    CHECK(f.max_validation_px <= 0.4 * kEps + f.floor_px);
    CHECK(f.max_join_px <= 1e-6 * kEps);  // the C0 pseudo-samples' residuals (§4.7.6)
  }
  // bounce: 8 C0 impacts, a piece each; linear: the 13 keys snapped; every piece begins at a breakpoint
  const mo::PiecewiseFit &bounce = r.sets[0].motion, &linear = r.sets[1].motion, &rig = r.sets[2].motion;
  CHECK(bounce.breakpoints.size() == 8);
  CHECK(bounce.pieces.size() == 9);
  CHECK(linear.breakpoints.size() == 13);
  for (const auto& bp : linear.breakpoints) CHECK(bp.snapped);
  for (size_t k = 1; k < linear.pieces.size(); ++k) {
    CHECK(std::binary_search(lin_ns.begin(), lin_ns.end(), linear.pieces[k].begin_ns));
  }
  // the rig: no breakpoint (C1 keys only); the fallback split it with C0 joins at its declared C1 keys
  CHECK(rig.breakpoints.empty());
  CHECK(rig.pieces.size() > 1);
  for (size_t k = 1; k < rig.pieces.size(); ++k) {
    CHECK(rig.pieces[k].split);
    CHECK(rig.pieces[k].join == mo::Continuity::kC0);
    CHECK(std::binary_search(rig_ns.begin(), rig_ns.end(), rig.pieces[k].begin_ns));
  }
  // every unsnapped impact got a join sample; the fill and the probes were asked
  CHECK(stage_calls(r, "join") >= 8);
  CHECK(stage_calls(r, "validate") > 0);
  MESSAGE("calls " << r.stats.calls << ": join " << stage_calls(r, "join") << ", fill " << stage_calls(r, "fill")
                   << ", validate " << stage_calls(r, "validate") << "; rounds " << r.stats.rounds);
}

TEST_CASE("T6 fit: CONSTANT steps (BL-12's case, refuse_steps off) are free joins; a still body steps between holds") {
  const mo::TimeMap tm = time24(73);
  const std::vector<double> stp = syn::step_keys();
  auto still_steps = [](double frame) {  // the step motion without its rotation: still between the steps
    const int k = std::clamp(static_cast<int>(std::floor((frame - 1.0) / 12.0)), 0, 5);
    return syn::Pose{syn::qexp({0.2, 0.0, 0.0}), {0.02 * k, 0.0, 0.01 * (k % 2)}};
  };
  const std::vector<syn::SyntheticBody> bodies{{syn::step, stp}, {still_steps, stp}, {syn::camera, {}}};
  const mo::MotionScene scene = scene_of(2);
  const std::vector<mo::DeclaredKey> keys = declared(tm, {stp.begin() + 1, stp.end()}, mo::Continuity::kStep);
  mo::DetectOptions det;
  det.refuse_steps = false;
  const Run r = run(syn::synthetic_oracle(tm, bodies), tm, scene, {keys, keys}, 2, {}, det, fit_on());
  std::vector<int64_t> step_ns;
  for (size_t k = 1; k < stp.size(); ++k) step_ns.push_back(tm.to_ns(static_cast<float>(stp[k])));
  for (size_t b = 0; b < 2; ++b) {
    const mo::PiecewiseFit& f = r.sets[b].motion;
    std::vector<int64_t> steps;
    for (const auto& p : f.pieces) {
      if (p.join == mo::Continuity::kStep) steps.push_back(p.begin_ns);
    }
    CHECK(steps == step_ns);  // a step join at each key (snapped)
    const Dense d = dense(f, truth(tm, scene, bodies, static_cast<int32_t>(b), -2 * kW, tm.t_end_ns(), 64, begins(f)),
                          step_ns, 4200);
    MESSAGE("steps body " << b << ": " << f.pieces.size() << " pieces, " << f.holds.size() << " holds, dense " << d.max_px
                          << " px");
    CHECK(d.max_px <= kEps / 2.0);
    check_c1(mo::to_trajectory(f, {0.0, 0.0, 0.0}), f.holds);
  }
  // the still body: a hold per level, each reaching the next step (its pre-key sample ends the run)
  const mo::PiecewiseFit& still = r.sets[1].motion;
  CHECK(still.pieces.size() == step_ns.size() + 1);
  for (const auto& p : still.pieces) CHECK(p.hold);
}

TEST_CASE("T6 fit: the breakpoint-probe case of DS d14 refuses rather than accepts, and halving the left piece fixes it") {
  // detection alone (no declared key, as for keys under an animated NLA time): the kink at frame 49 stays unsnapped and
  // Blender's plateau samples just before it stay in the fit, so the piece left of it ends off the motion and the
  // piece right of it inherits that pose (DS d14's spin_child, "found (unsnapped, plateau kept)"). DS d14's fill (gaps
  // of h/2 up to a piece's end, end_fill_div 0) reproduces it; the default fill (h/8 within one knot spacing of the
  // end) does not leave the error
  const mo::TimeMap tm = time24(61);
  const std::vector<syn::SyntheticBody> bodies{{spin_child, {1, 49}}, {syn::camera, {}}};
  mo::MotionScene scene = scene_of(1);
  scene.bodies[0].rho_m = 0.28;
  auto attempt = [&](bool probes, double accept, int halvings, int end_fill_div = 0) {
    mo::FitOptions fo;
    fo.breakpoint_probes = probes;
    fo.accept_frac = accept;
    fo.halvings = halvings;
    fo.end_fill_div = end_fill_div;
    return run(syn::synthetic_oracle(tm, bodies), tm, scene, {}, 2, {}, {}, fo);
  };
  {
    const Run dflt = attempt(true, 0.4, 5, mo::FitOptions{}.end_fill_div);
    const mo::PiecewiseFit& f = dflt.sets[0].motion;
    const Dense d = dense(f, truth(tm, scene, bodies, 0, -2 * kW, tm.t_end_ns(), 256, begins(f)));
    MESSAGE("default fill: validation " << f.max_validation_px << " px, dense " << d.max_px << " px");
    CHECK(d.max_px <= kEps / 2.0);
    for (const auto& p : f.pieces) CHECK(p.halvings == 0);
  }
  const int64_t b49 = tm.to_ns(49.0f);
  // with the breakpoint probes (b and b-) and no halving: MOT-FIT next to b, at 0.4 eps and at eps/2 alike
  for (const double accept : {0.4, 0.5}) {
    try {
      (void)attempt(true, accept, 0);
      FAIL("accepted");
    } catch (const mo::MotionError& e) {
      CHECK(e.code == "MOT-FIT");
      MESSAGE(std::string(e.what()));
    }
  }
  // with halvings: the left piece of the join at 49 is halved, the fit then lies within eps/2 of the truth
  const Run fixed = attempt(true, 0.4, 5);
  const mo::PiecewiseFit& f = fixed.sets[0].motion;
  const auto right = std::find_if(f.pieces.begin(), f.pieces.end(),
                                  [&](const mo::FitPiece& p) { return std::llabs(p.begin_ns - b49) < 100; });
  REQUIRE(right != f.pieces.end());
  REQUIRE(right != f.pieces.begin());
  CHECK((right - 1)->halvings >= 1);
  CHECK(right->halvings == 0);
  check_joins(f);
  const Dense dok = dense(f, truth(tm, scene, bodies, 0, -2 * kW, tm.t_end_ns(), 256, begins(f)));
  MESSAGE("halved: validation " << f.max_validation_px << " px, dense " << dok.max_px << " px");
  CHECK(dok.max_px <= kEps / 2.0);
  // without the breakpoint probes, validation (V and the seeded probes) accepts the unhalved fit while it errs above
  // eps/2 next to b: DS d14's silent failure
  const Run silent = attempt(false, 0.5, 0);
  const mo::PiecewiseFit& g = silent.sets[0].motion;
  const Dense dbad = dense(g, truth(tm, scene, bodies, 0, -2 * kW, tm.t_end_ns(), 256, begins(g)));
  MESSAGE("no breakpoint probes: validation " << g.max_validation_px << " px, dense " << dbad.max_px << " px at "
                                             << dbad.at_ns << " ns");
  CHECK(g.max_validation_px <= 0.5 * kEps + g.floor_px);
  CHECK(dbad.max_px > kEps / 2.0);
  CHECK(std::llabs(dbad.at_ns - b49) < 20000);
}

TEST_CASE("T6 fit: a hold's failure is charged to the piece on its left") {
  // spin_child held after frame 49: the hold piece takes the left piece's end pose, so its own samples show the error
  // the left piece leaves at b; without breakpoint probes that piece passes its own validation, the hold fails and
  // the left piece is halved
  const mo::TimeMap tm = time24(61);
  const std::vector<syn::SyntheticBody> bodies{{spin_child_still, {1, 49}}, {syn::camera, {}}};
  mo::MotionScene scene = scene_of(1);
  scene.bodies[0].rho_m = 0.2;
  mo::FitOptions fo;
  fo.breakpoint_probes = false;
  fo.end_fill_div = 0;  // DS d14's fill
  const Run r = run(syn::synthetic_oracle(tm, bodies), tm, scene, {}, 2, {}, {}, fo);
  const mo::PiecewiseFit& f = r.sets[0].motion;
  REQUIRE(f.pieces.size() == 3);
  CHECK(f.pieces[0].hold);
  CHECK(f.pieces[2].hold);
  CHECK(std::llabs(f.pieces[2].begin_ns - tm.to_ns(49.0f)) < 100);
  CHECK(f.pieces[1].halvings >= 1);
  check_joins(f);
  const Dense d = dense(f, truth(tm, scene, bodies, 0, -2 * kW, tm.t_end_ns(), 256, begins(f)));
  MESSAGE("hold charged left: " << f.pieces[1].halvings << " halvings, dense " << d.max_px << " px");
  CHECK(d.max_px <= kEps / 2.0);
  fo.halvings = 0;
  CHECK_THROWS_WITH_AS(run(syn::synthetic_oracle(tm, bodies), tm, scene, {}, 2, {}, {}, fo),
                       doctest::Contains("MOT-FIT"), mo::MotionError);
}

TEST_CASE("T6 fit: holds become pieces of their own, bitwise still, joined C0, declared as C1 holds") {
  // still until frame 13, LINEAR to x = +0.1 at frame 25, still after (declared C0 keys); a fixed tilt
  const mo::TimeMap tm = time24(37);
  auto move = [](double frame) {
    const double u = std::clamp((frame - 13.0) / 12.0, 0.0, 1.0);
    return syn::Pose{syn::qexp({0.1, 0.2, 0.3}), {0.2 * u - 0.1, 0.05, 0.0}};
  };
  const std::vector<syn::SyntheticBody> bodies{{move, {13, 25}}, {syn::camera, {}}};
  const mo::MotionScene scene = scene_of(1);
  const Run r = run(syn::synthetic_oracle(tm, bodies), tm, scene, {declared(tm, {13, 25}, mo::Continuity::kC0)}, 2,
                    {}, {}, fit_on());
  const mo::PiecewiseFit& f = r.sets[0].motion;
  const int64_t t13 = tm.to_ns(13.0f), t25 = tm.to_ns(25.0f);
  REQUIRE(f.pieces.size() == 3);
  CHECK(f.pieces[0].hold);
  CHECK_FALSE(f.pieces[1].hold);
  CHECK(f.pieces[2].hold);
  CHECK(f.pieces[1].begin_ns == t13);
  CHECK(f.pieces[2].begin_ns == t25);
  CHECK(f.pieces[1].join == mo::Continuity::kC0);
  CHECK(f.pieces[2].join == mo::Continuity::kC0);
  REQUIRE(f.holds.size() == 2);
  CHECK(f.holds[0] == std::pair<int64_t, int64_t>{-2 * kW, t13});
  CHECK(f.holds[1] == std::pair<int64_t, int64_t>{t25, tm.t_end_ns()});
  check_joins(f);
  // bitwise still over each hold (equal controls: zero increments)
  for (const size_t k : {size_t{0}, size_t{2}}) {
    const mo::PieceEval e(f.pieces[k]);
    const mo::Pose p0 = e(f.pieces[k].begin_ns);
    for (const int64_t t : {f.pieces[k].begin_ns + 1, (f.pieces[k].begin_ns + f.pieces[k].end_ns) / 2,
                            f.pieces[k].end_ns - 1}) {
      const mo::Pose p = e(t);
      CHECK(std::memcmp(&p.p, &p0.p, sizeof p.p) == 0);
      CHECK(std::memcmp(&p.q, &p0.q, sizeof p.q) == 0);
    }
  }
  const Dense d = dense(f, truth(tm, scene, bodies, 0, -2 * kW, tm.t_end_ns(), 64, begins(f)), {t13, t25}, 4200);
  CHECK(d.max_px <= kEps / 2.0);
  check_c1(mo::to_trajectory(f, {0.0, 0.0, 0.0}), f.holds);
}

TEST_CASE("T6 fit: keys closer than 1.5 frames (per-frame LINEAR keys, short bounce arcs) fit by splitting at them") {
  // detection classifies a key from the samples on either side, which reach the neighbouring kinks when the keys are
  // closer than about 1.5 frames, so these kinks stay unclassified; the fit fails validation and the split at every
  // declared non-step key (C0 joins) fixes it, where a split at C1 keys alone ended in MOT-FIT
  const mo::TimeMap tm = time24(30);
  const mo::MotionScene scene = scene_of(4);
  const std::vector<double> f10 = frames_from(8.0, 1.0, 7), f05 = frames_from(8.0, 0.5, 9);
  const std::vector<double> f025 = frames_from(8.0, 0.25, 9);
  const std::vector<double> arcs = frames_from(8.0, 0.75, 9);
  const std::vector<syn::SyntheticBody> bodies{
      {[](double f) { return dense_linear(f, 8.0, 1.0, 7, 1); }, f10},
      {[](double f) { return dense_linear(f, 8.0, 0.5, 9, 2); }, f05},
      {[](double f) { return dense_linear(f, 8.0, 0.25, 9, 3); }, f025},
      {[](double f) { return short_arcs(f, 8.0, 8); }, {}},
      {syn::camera, {}}};
  std::vector<std::vector<mo::DeclaredKey>> keys{declared(tm, f10, mo::Continuity::kC0),
                                                 declared(tm, f05, mo::Continuity::kC0),
                                                 declared(tm, f025, mo::Continuity::kC0)};
  {  // BOUNCE's breaks are no F-curve keys: no plateau
    std::vector<mo::DeclaredKey> breaks;
    for (const double f : arcs) breaks.push_back({tm.to_ns(static_cast<float>(f)), mo::Continuity::kC0, false, 1e-4});
    keys.push_back(breaks);
  }
  const Run r = run(syn::synthetic_oracle(tm, bodies), tm, scene, keys, 4, {}, {}, fit_on());
  const char* names[] = {"linear 1.0", "linear 0.5", "linear 0.25", "short arcs"};
  int split = 0;
  for (size_t b = 0; b < 4; ++b) {
    REQUIRE(r.sets[b].fitted);
    const mo::PiecewiseFit& f = r.sets[b].motion;
    check_joins(f);
    check_c1(mo::to_trajectory(f, {0.0, 0.0, 0.0}), f.holds);
    std::vector<int64_t> key_ns;
    for (const mo::DeclaredKey& k : keys[b]) key_ns.push_back(k.t_ns);
    const Dense d = dense(f, truth(tm, scene, bodies, static_cast<int32_t>(b), -2 * kW, tm.t_end_ns(), 64, begins(f)),
                          b < 3 ? key_ns : std::vector<int64_t>{}, 4200);
    int halved = 0;
    bool any_split = false;
    for (const auto& p : f.pieces) {
      halved += p.halvings;
      any_split = any_split || p.split;
    }
    split += any_split ? 1 : 0;
    MESSAGE(std::string(names[b]) << ": " << f.pieces.size() << " pieces, " << f.breakpoints.size()
                                  << " breakpoints, split " << any_split << ", halvings " << halved << ", validation "
                                  << f.max_validation_px << " px, dense " << d.max_px << " px");
    CHECK(d.max_px <= kEps / 2.0);
    CHECK(f.max_validation_px <= 0.4 * kEps + f.floor_px);
  }
  CHECK(split >= 1);  // at least one of them took the fallback
}

TEST_CASE("T6 fit: a fast spinner fails C1's pi/2 rule, not validation: MOT-FIT names the rad, a halving fixes it") {
  // 20 turns/s about z: 1.05 rad (0.33 pi) between the base samples, so the aliasing pre-pass leaves it alone, and the
  // knots 2 x the median spacing (16.7 ms) apart turn by 2.1 rad > pi/2. The fit lies on the samples (a constant rate
  // is in the spline's space), so validation passes and only the increment rule fails
  const mo::TimeMap tm = time24(25);
  auto spin = [](double f) {
    return syn::Pose{syn::qexp({0.0, 0.0, 2.0 * std::numbers::pi * 20.0 * (f - 1.0) / 24.0}), {0.0, 0.0, 0.1}};
  };
  const std::vector<syn::SyntheticBody> bodies{{spin, {}}, {syn::camera, {}}};
  const mo::MotionScene scene = scene_of(1);
  mo::FitOptions none;
  none.halvings = 0;
  try {
    (void)run(syn::synthetic_oracle(tm, bodies), tm, scene, {}, 2, {}, {}, none);
    FAIL("accepted");
  } catch (const mo::MotionError& e) {
    const std::string what = e.what();
    MESSAGE(what);
    CHECK(e.code == "MOT-FIT");
    CHECK(what.find("control rotations turn by") != std::string::npos);
    CHECK(what.find("rad") != std::string::npos);
    CHECK(what.find("px at") == std::string::npos);  // not blamed on a pixel error
    CHECK(what.find("0 halvings") != std::string::npos);
  }
  const Run r = run(syn::synthetic_oracle(tm, bodies), tm, scene, {}, 2, {}, {}, fit_on());
  const mo::PiecewiseFit& f = r.sets[0].motion;
  REQUIRE(f.pieces.size() == 1);
  CHECK(f.pieces[0].halvings >= 1);
  CHECK(mo::max_increment_rad(f.pieces[0]) <= std::numbers::pi / 2);
  const Dense d = dense(f, truth(tm, scene, bodies, 0, -2 * kW, tm.t_end_ns(), 64));
  MESSAGE("spinner: " << f.pieces[0].halvings << " halvings, h " << f.pieces[0].h_ns << " ns, dense " << d.max_px
                      << " px");
  CHECK(d.max_px <= kEps / 2.0);
}

TEST_CASE("T6 fit: the breakpoint's left-limit probe stays off the key on Blender's coarse float32 time grid") {
  // at frame 5020 the float32 time grid is 2 q = 20 us (1/2 ulp(5020) = 2.4e-4 frames), wider than the key's plateau
  // (1e-4 frames) plus the 1 us offset b- = b - w - 1 us used to have: that request realized on the key itself and the
  // left piece had no left-limit probe. The probe is now offset by 2 q(F), as the sampler's pre-key sample is
  mo::TimeSettings ts;
  ts.fps = 24;
  ts.frame_start = 5000;
  ts.frame_end = 5040;
  const mo::TimeMap tm(ts);
  const double key = 5020.0;
  auto kink = [key](double f) {  // 0.6 m/s along x, then 0.4 m/s along y: a velocity kink at the key
    const double t = (f - key) / 24.0;
    return syn::Pose{syn::qexp({0.0, 0.0, 0.1 * t}), {t < 0 ? 0.6 * t : 0.0, t < 0 ? 0.0 : 0.4 * t, 0.0}};
  };
  const std::vector<syn::SyntheticBody> bodies{{kink, {key}}, {syn::camera, {}}};
  const mo::MotionScene scene = scene_of(1);
  const Run r = run(syn::synthetic_oracle(tm, bodies), tm, scene, {declared(tm, {key}, mo::Continuity::kC0)}, 2, {}, {},
                    fit_on());
  const mo::SampleSet& set = r.sets[0];
  REQUIRE(set.fitted);
  REQUIRE(set.motion.pieces.size() == 2);
  const int64_t b = tm.to_ns(static_cast<float>(key));
  CHECK(set.motion.pieces[1].begin_ns == b);
  int32_t F = 0;
  float sub = 0.0f;
  tm.request(b, &F, &sub);
  const double q = tm.resolution_ns(F);
  const double w = 1e-4 * tm.period_ns();
  CHECK(w < q);
  // a probe left of the key by more than q, within the 2 q the offset asks for (plus the request's own rounding)
  int found = 0;
  for (size_t i = 0; i < set.probes.size(); ++i) {
    const double d = static_cast<double>(b - set.probes.t_ns[i]);
    if (d > q && d <= w + 3.0 * q + 2.0) ++found;
  }
  MESSAGE("q(F) " << q << " ns, plateau " << w << " ns, left-limit probes " << found << ", validation "
                  << set.motion.max_validation_px << " px");
  CHECK(found >= 1);
  CHECK(set.motion.max_validation_px <= 0.4 * kEps + set.motion.floor_px);
}

TEST_CASE("drive: on_batch sees every round's weighed batch; a camera that leaves its rig in a later round ends the "
          "run with CAM-RIG") {
  const mo::TimeMap tm = time24(25);
  // the base grid holds per_frame samples per frame from the frame reference: a drift of
  // sin^2(pi per_frame (f - F_ref)) is zero there (to float32 rounding) and one at the first round's midpoints, so the
  // rigs form on the base batch and break on the next; the span's ends (the base grid's first and last samples) are off
  // that grid, so the drift lives between frames 3 and 20 only
  const mo::SampleOptions opt;
  const double per_frame = std::max(std::ceil(tm.period_ns() / static_cast<double>(mo::h0_ns(tm, opt))),
                                    std::ceil(opt.alias_per_frame / 2.0));
  auto lead = [](double f) { return syn::Pose{syn::qexp({0.0, 0.01 * f, 0.02 * f}), {0.01 * f, -1.0, 0.2}}; };
  const rot::Q off_q = syn::qexp({0.05, -0.02, 0.1});
  const rot::V off_p{0.12, 0.0, 0.01};
  auto follower_with = [&](double amp) {
    return [=](double f) {
      const syn::Pose l = lead(f);
      const double w = f >= 3.0 && f <= 20.0 ? std::sin(std::numbers::pi * per_frame * (f - 1.0)) : 0.0;
      return syn::Pose{l.q * off_q, l.p + rotate(l.q, off_p + rot::V{amp * w * w, 0.0, 0.0})};
    };
  };
  mo::MotionScene scene;
  scene.bodies.resize(3);
  scene.bodies[0].rho_m = 0.05;
  scene.bodies[0].S_ref = std::array<double, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1};
  scene.bodies[1].role = mo::Role::kRig;
  scene.bodies[2].role = mo::Role::kRig;
  scene.cameras.push_back({1, {0, 0, 0}, 640.0, 0.0});
  scene.cameras.push_back({2, {0, 0, 0}, 640.0, 0.0});
  const std::vector<int32_t> cams{1, 2};
  // `amp` > 0: the follower drifts in its rig at the midpoints only
  auto attempt = [&](double amp, int* batches, bool* weighed, std::optional<mo::RigGroups>* groups) {
    const mo::Oracle oracle = syn::synthetic_oracle(tm, {{syn::smooth, {}}, {lead, {}}, {follower_with(amp), {}}});
    const mo::ErrorModel model(scene);
    std::vector<mo::BodyPipeline> pipes;
    for (int32_t b = 0; b < 3; ++b) pipes.emplace_back(b, mo::Mode::kKeyed, tm, std::vector<mo::DeclaredKey>{}, opt);
    balla::render::ThreadPool pool(2);
    auto hook = [&](const mo::PoseBatch& batch) {
      ++*batches;
      *weighed = *weighed && batch.w_pos.size() == batch.size() * static_cast<size_t>(batch.bodies);
      for (size_t i = 0; i < batch.w_pos.size(); ++i) *weighed = *weighed && batch.w_pos[i] > 0.0;
      if (!*groups) *groups = mo::group_rigs(batch, cams);
      mo::check_rigs(batch, **groups);
    };
    return mo::drive(oracle, tm, model, pipes, pool, {}, hook);
  };
  {  // a rigid follower: the hook sees every round (all weighed) and nothing departs
    int batches = 0;
    bool weighed = true;
    std::optional<mo::RigGroups> groups;
    const mo::RoundStats st = attempt(0.0, &batches, &weighed, &groups);
    CHECK(batches == st.rounds);
    CHECK(batches >= 2);
    CHECK(weighed);
    REQUIRE(groups);
    CHECK(groups->rig == std::vector<int32_t>{1, 1});
  }
  {  // the drift appears with the first midpoint round: CAM-RIG ends the run there
    int batches = 0;
    bool weighed = true;
    std::optional<mo::RigGroups> groups;
    try {
      (void)attempt(1e-3, &batches, &weighed, &groups);
      FAIL("the run finished");
    } catch (const mo::MotionError& e) {
      CHECK(e.code == "CAM-RIG");
      MESSAGE(std::string(e.what()));
    }
    CHECK(batches == 2);  // the base grid (the rig forms on it), then the midpoints
    CHECK(weighed);
    REQUIRE(groups);
    CHECK(groups->rig == std::vector<int32_t>{1, 1});
  }
}

TEST_CASE("T7 fit: a recorded oracle replays the fits bitwise at 1 and 8 workers") {
  const mo::TimeMap tm = time24(49);
  const std::vector<double> lin = syn::linear_keys(), rk = syn::rig_keys();
  const std::vector<syn::SyntheticBody> bodies{
      {[](double f) { return syn::bounce(f); }, {}}, {syn::linear, lin}, {syn::rig, rk}, {syn::camera, {}}};
  const mo::MotionScene scene = scene_of(3);
  const std::vector<std::vector<mo::DeclaredKey>> keys{
      {}, declared(tm, lin, mo::Continuity::kC0), declared(tm, rk, mo::Continuity::kC1)};
  mo::OracleLog log;
  const Run first = run(mo::recording(syn::synthetic_oracle(tm, bodies), log), tm, scene, keys, 1, {}, {}, fit_on());
  auto same_fit = [](const mo::PiecewiseFit& a, const mo::PiecewiseFit& b) {
    if (a.pieces.size() != b.pieces.size() || a.holds != b.holds || a.span_begin_ns != b.span_begin_ns ||
        a.span_end_ns != b.span_end_ns || !same_bits(a.max_validation_px, b.max_validation_px) ||
        !same_bits(a.floor_px, b.floor_px) || a.validation_samples != b.validation_samples) {
      return false;
    }
    for (size_t k = 0; k < a.pieces.size(); ++k) {
      const mo::FitPiece &x = a.pieces[k], &y = b.pieces[k];
      if (x.begin_ns != y.begin_ns || x.end_ns != y.end_ns || x.t0_ns != y.t0_ns || x.h_ns != y.h_ns ||
          x.join != y.join || x.hold != y.hold || x.halvings != y.halvings || x.rot.size() != y.rot.size() ||
          std::memcmp(x.rot.data(), y.rot.data(), x.rot.size() * sizeof x.rot[0]) != 0 ||
          std::memcmp(x.pos.data(), y.pos.data(), x.pos.size() * sizeof x.pos[0]) != 0) {
        return false;
      }
    }
    return true;
  };
  for (const int workers : {8, 1}) {
    auto n = std::make_shared<size_t>(0);
    const Run again = run(mo::replay(log, n), tm, scene, keys, workers, {}, {}, fit_on());
    CHECK(*n == log.calls.size());
    CHECK(again.stats == first.stats);
    for (size_t b = 0; b < first.sets.size(); ++b) {
      REQUIRE(again.sets[b].fitted);
      CHECK(same_fit(first.sets[b].motion, again.sets[b].motion));
    }
  }
  // another probe seed asks other probes: the replay stops with BRG-REPLAY
  mo::FitOptions other;
  other.probe_seed = 7;
  CHECK_THROWS_WITH_AS(run(mo::replay(log, std::make_shared<size_t>(0)), tm, scene, keys, 2, {}, {}, other),
                       doctest::Contains("BRG-REPLAY"), mo::MotionError);
}

TEST_CASE("fit pieces: holds, plans, knot spacing and a polynomial fit") {
  // find_holds: runs of at least 3 bitwise-identical poses
  mo::Track t;
  const rot::Q q0{1, 0, 0, 0}, q1 = syn::qexp({0.0, 0.1, 0.0});
  const rot::V p0{0, 0, 0}, p1{1, 0, 0};
  const std::vector<std::pair<rot::Q, rot::V>> poses{{q0, p0}, {q0, p0}, {q0, p0}, {q1, p1}, {q1, p1},
                                                     {q1, p0}, {q1, p0}, {q1, p0}, {q1, p0}};
  for (size_t i = 0; i < poses.size(); ++i) {
    t.t_ns.push_back(static_cast<int64_t>(10 * i));
    t.q.push_back(poses[i].first);
    t.p.push_back(poses[i].second);
    t.w_pos.push_back(1.0);
    t.w_rot.push_back(1.0);
    t.floor1.push_back(0.0);
  }
  const auto holds = mo::find_holds(t);
  REQUIRE(holds.size() == 2);
  CHECK(holds[0] == std::pair<int64_t, int64_t>{0, 20});
  CHECK(holds[1] == std::pair<int64_t, int64_t>{50, 80});
  // plan_pieces: a breakpoint in (the sample before a hold, its first sample] begins it, one in [its last sample, the
  // next sample) ends it, one inside is dropped; a hold from the first sample reaches the span's start
  const std::vector<int64_t> times{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  const mo::Breakpoint b1{25, mo::Continuity::kC0}, b2{45, mo::Continuity::kC0}, b3{65, mo::Continuity::kC0},
      b4{72, mo::Continuity::kC0};
  const std::vector<mo::Breakpoint> bps{b1, b2, b3, b4};
  const std::vector<std::pair<int64_t, int64_t>> runs{{0, 20}, {50, 70}};
  const std::vector<mo::PiecePlan> plan = mo::plan_pieces(-5, 105, bps, runs, times);
  REQUIRE(plan.size() == 4);
  CHECK((plan[0].begin_ns == -5 && plan[0].end_ns == 25 && plan[0].hold));
  CHECK((plan[1].begin_ns == 25 && plan[1].end_ns == 45 && !plan[1].hold));
  CHECK((plan[2].begin_ns == 45 && plan[2].end_ns == 72 && plan[2].hold));
  CHECK((plan[3].begin_ns == 72 && plan[3].end_ns == 105 && !plan[3].hold));
  // without breakpoints a hold piece spans exactly its run
  const std::vector<mo::PiecePlan> plain = mo::plan_pieces(-5, 105, {}, runs, times);
  REQUIRE(plain.size() == 4);
  CHECK((plain[1].begin_ns == 20 && plain[2].begin_ns == 50 && plain[2].end_ns == 70 && plain[2].hold));
  for (size_t k = 1; k < plan.size(); ++k) CHECK(plan[k].join == mo::Continuity::kC0);
  const mo::Breakpoint step{90, mo::Continuity::kStep};
  const std::vector<mo::PiecePlan> plan2 = mo::plan_pieces(0, 100, std::vector<mo::Breakpoint>{step}, {}, times);
  REQUIRE(plan2.size() == 2);
  CHECK(plan2[1].join == mo::Continuity::kStep);
  // a CONSTANT key's step at the sample after a run (its pre-key sample ends the run): the hold reaches the step
  const std::vector<std::pair<int64_t, int64_t>> before_step{{50, 80}};
  const std::vector<mo::PiecePlan> plan3 = mo::plan_pieces(0, 100, std::vector<mo::Breakpoint>{step}, before_step, times);
  REQUIRE(plan3.size() == 3);
  CHECK((plan3[1].begin_ns == 50 && plan3[1].end_ns == 90 && plan3[1].hold));
  CHECK(plan3[2].join == mo::Continuity::kStep);
  // knot spacing: 2 x the median spacing of the middle 80 %, clamped, at most the piece's length
  std::vector<int64_t> grid;
  for (int64_t x = 0; x <= 1000000000; x += 5000000) grid.push_back(x);
  CHECK(mo::knot_spacing(grid, 0, 1000000000, 1000000, 20833334) == 10000000);
  CHECK(mo::knot_spacing(grid, 0, 1000000000, 1000000, 8000000) == 8000000);
  CHECK(mo::knot_spacing(grid, 0, 3000000, 1000000, 20833334) == 3000000);
  CHECK(mo::knot_spacing(grid, 0, 1, 1000000, 20833334) == 1);
  // fit_piece: a quintic in position and a constant-rate turn lie in the spline's space, so the fit reproduces them;
  // a C0 start is kept to rounding
  mo::Track s;
  auto pose_at = [](double ts) {
    const double x = ts - 0.3;
    return std::pair{syn::qexp({0.2 * ts, -0.1 * ts, 0.4 * ts}),
                     rot::V{0.1 * x * x * x * x * x - 0.2 * x * x + 0.3, 0.05 * x * x * x, -0.4 * x}};
  };
  for (int64_t x = 0; x <= 600000000; x += 2500000) {
    const auto [qq, pp] = pose_at(static_cast<double>(x) * 1e-9);
    s.t_ns.push_back(x);
    s.q.push_back(qq);
    s.p.push_back(pp);
    s.w_pos.push_back(2000.0);
    s.w_rot.push_back(100.0);
    s.floor1.push_back(0.0);
  }
  const mo::FitPiece pc = mo::fit_piece(s, 0, 600000000, 10000000, std::nullopt, false, mo::FitOptions{});
  const std::vector<double> e = mo::piece_errors(pc, s);
  CHECK(*std::max_element(e.begin(), e.end()) < 1e-6);
  CHECK(mo::max_increment_rad(pc) < 0.1);
  const mo::Pose start{syn::qexp({0.0, 0.0, 1e-7}), {0.3 + 1e-9, 0.0, 0.12}};
  const mo::FitPiece pj = mo::fit_piece(s, 0, 600000000, 10000000, start, false, mo::FitOptions{});
  const mo::Pose at0 = mo::PieceEval(pj)(0);
  CHECK(norm(at0.p - start.p) <= 1e-3 * balla::traj::kJoinTolM);
  CHECK(angle(at0.q, start.q) <= 1e-3 * balla::traj::kJoinTolRad);
  CHECK(pj.join_px > 0.0);
  // a hold piece: K equal controls at the start pose
  const mo::FitPiece ph = mo::fit_piece(s, 0, 600000000, 0, start, true, mo::FitOptions{});
  CHECK(ph.rot.size() == 6);
  CHECK(std::all_of(ph.pos.begin(), ph.pos.end(),
                    [&](const rot::V& v) { return std::memcmp(&v, &start.p, sizeof v) == 0; }));
}

TEST_CASE("fit pieces: a C0 start far from Blender's origin keeps the pose to fp64 rounding; PieceEval no copy") {
  static_assert(!std::is_copy_constructible_v<mo::PieceEval> && !std::is_copy_assignable_v<mo::PieceEval>);
  static_assert(std::is_nothrow_move_constructible_v<mo::PieceEval>);
  // an action 1e5 m or more from the origin (MOT-FAR and POS-RES warn, the fit goes on): there one ulp (1.5e-11 m at
  // 1e5 m) exceeds the 1e-3 kJoinTolM the rigid move was checked against, and about 0.5 % of the joins at 3e5 m and
  // beyond came out an ulp apart (a shifted control rounds on the coarser grid of the next binade), which threw
  // std::logic_error. The check now allows 16 eps |p| on top
  uint64_t state = 0x9e3779b97f4a7c15ull;
  auto unit = [&] {  // uniform in [-1, 1)
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return 2.0 * static_cast<double>((z ^ (z >> 31)) >> 11) / 9007199254740992.0 - 1.0;
  };
  int fits = 0, past_old = 0;
  double worst = 0.0;  // the join error in eps |p|
  for (const double far : {3e5, 1e6}) {
    for (int k = 0; k < 300; ++k) {
      const double ox = far * (0.5 + unit()), oy = far * unit(), oz = far * unit();
      mo::Track s;
      for (int64_t x = 0; x <= 600000000; x += 2500000) {
        const double ts = static_cast<double>(x) * 1e-9;
        s.t_ns.push_back(x);
        s.q.push_back(syn::qexp({0.2 * ts, -0.1 * ts, 0.4 * ts}));
        s.p.push_back({ox + 0.3 * ts * ts + 1e-5 * unit(), oy + 0.05 * ts, oz - 0.4 * ts});
        s.w_pos.push_back(2000.0);
        s.w_rot.push_back(100.0);
        s.floor1.push_back(0.0);
      }
      const mo::Pose start{syn::qexp({0.0, 0.0, 1e-7}), {ox + 1e-6 * unit(), oy + 1e-6 * unit(), oz + 1e-6 * unit()}};
      mo::FitPiece pj;
      REQUIRE_NOTHROW(pj = mo::fit_piece(s, 0, 600000000, 10000000, start, false, mo::FitOptions{}));
      ++fits;
      const mo::Pose at0 = mo::PieceEval(pj)(0);
      const double err = norm(at0.p - start.p);
      const double tol = 1e-3 * balla::traj::kJoinTolM + 16.0 * std::numeric_limits<double>::epsilon() * norm(start.p);
      CHECK(err <= tol);
      CHECK(angle(at0.q, start.q) <= 1e-3 * balla::traj::kJoinTolRad);
      CHECK(tol < balla::traj::kJoinTolM);  // still below the engine's own tolerance out to 1e6 m
      past_old += err > 1e-3 * balla::traj::kJoinTolM ? 1 : 0;
      worst = std::max(worst, err / (std::numeric_limits<double>::epsilon() * norm(start.p)));
      // moving a PieceEval keeps it valid
      mo::PieceEval a(pj);
      const mo::Pose want = a(1000000);
      mo::PieceEval b(std::move(a));
      const mo::Pose got = b(1000000);
      CHECK(std::memcmp(&want, &got, sizeof want) == 0);
    }
  }
  MESSAGE(fits << " fits: " << past_old << " joins past the old absolute 1e-11 m, the worst " << worst << " eps |p|");
  CHECK(past_old >= 1);  // the old check would have thrown
}

TEST_CASE("rigs: cameras with a constant relative pose share one; CAM-RIG when a declared rig or a later sample departs") {
  const mo::TimeMap tm = time24(25);
  auto lead = [](double f) { return syn::Pose{syn::qexp({0.0, 0.01 * f, 0.02 * f}), {0.01 * f, -1.0, 0.2}}; };
  const rot::Q off_q = syn::qexp({0.05, -0.02, 0.1});
  const rot::V off_p{0.12, 0.0, 0.01};
  auto follower = [&](double f) {
    const syn::Pose l = lead(f);
    return syn::Pose{l.q * off_q, l.p + rotate(l.q, off_p)};
  };
  auto drifting = [&](double f) {
    const syn::Pose l = lead(f);
    return syn::Pose{l.q * off_q, l.p + rotate(l.q, off_p + rot::V{1e-4 * f, 0.0, 0.0})};
  };
  auto other = [](double f) { return syn::smooth(f); };
  mo::MotionScene scene;
  scene.bodies.resize(3);
  for (auto& b : scene.bodies) b.role = mo::Role::kRig;
  std::vector<int32_t> F(48);
  std::vector<float> s(48);
  for (size_t i = 0; i < F.size(); ++i) tm.request(static_cast<int64_t>(i) * 20000000, &F[i], &s[i]);
  auto batch = [&](std::vector<syn::SyntheticBody> bodies) {
    mo::RawPoses raw = syn::synthetic_oracle(tm, std::move(bodies))(F, s);
    raw.F = F;
    raw.s = s;
    return mo::decode(raw, tm, scene);
  };
  const mo::PoseBatch good = batch({{lead, {}}, {follower, {}}, {other, {}}});
  const std::vector<int32_t> cams{0, 1, 2};
  mo::RigGroups g = mo::group_rigs(good, cams);
  CHECK(g.rig == std::vector<int32_t>{0, 0, 2});
  CHECK(angle(g.q_rig_cam[1], off_q) < 1e-6);
  CHECK(norm(g.p_rig_cam[1] - off_p) < 1e-6);
  CHECK(g.dev_rad[1] <= mo::kRigTol);
  CHECK_NOTHROW(mo::check_rigs(good, g));
  // a scene that declares all three one rig: the third moves on its own
  CHECK_THROWS_WITH_AS(mo::group_rigs(good, cams, std::vector<int32_t>{4, 4, 4}), doctest::Contains("CAM-RIG"),
                       mo::MotionError);
  // a later batch where the follower drifts in its rig
  const mo::PoseBatch later = batch({{lead, {}}, {drifting, {}}, {other, {}}});
  CHECK_THROWS_WITH_AS(mo::check_rigs(later, g), doctest::Contains("CAM-RIG"), mo::MotionError);
}

TEST_CASE("mode by provenance: keyed, frame-state, static, PHY-MIXED, PHY-UNBAKED, PHY-REMAP, MOT-SOURCE") {
  bl::ExchangeIn ex;
  auto obj = [&](const std::string& name) -> bl::ObjectIn& {
    bl::ObjectIn o;
    o.name_full = name;
    o.key = "[\"" + name + "\"]";
    ex.objects.push_back(o);
    return ex.objects.back();
  };
  auto rigid = [](const char* type, bool kinematic, bool animated) {
    Tree t = Tree::dict();
    t["type"] = Tree::string(type);
    t["enabled"] = Tree::boolean(true);
    t["kinematic"] = Tree::boolean(kinematic);
    t["kinematic_animated"] = Tree::boolean(animated);
    return t;
  };
  auto constraint = [](const char* type) {
    Tree c = Tree::dict();
    c["type"] = Tree::string(type);
    c["mute"] = Tree::boolean(false);
    Tree l = Tree::list();
    l.push_back(c);
    return l;
  };
  obj("keyed").animation = 0;                                   // 0
  obj("ball").rigid_body = rigid("ACTIVE", false, false);       // 1
  obj("on_ball").parent = 1;                                    // 2: a pure child of the rigid body
  {
    bl::ObjectIn& o = obj("mixed");                             // 3: a keyed child of the rigid body
    o.parent = 1;
    o.animation = 1;
  }
  obj("passive").rigid_body = rigid("PASSIVE", false, false);   // 4
  {
    bl::ObjectIn& o = obj("switch");                            // 5: kinematic keyed on and off
    o.rigid_body = rigid("ACTIVE", true, true);
    o.animation = 2;
  }
  obj("alembic").constraints = constraint("TRANSFORM_CACHE");   // 6
  obj("plain");                                                 // 7
  {
    bl::ObjectIn& o = obj("bone_child");                        // 8
    o.parent = 7;
    o.parent_type = "BONE";
  }
  obj("tracked").constraints = constraint("TRACK_TO");          // 9
  Tree world = Tree::dict(), cache = Tree::dict();
  cache["is_baked"] = Tree::boolean(true);
  cache["is_outdated"] = Tree::boolean(false);
  world["enabled"] = Tree::boolean(true);
  world["point_cache"] = cache;
  ex.rigidbody_world = world;

  CHECK(mo::choose_mode(ex, 0).source == mo::Source::kKeyed);  // a LINEAR-keyed body is keyed (never frame-state)
  const mo::ModeChoice ball = mo::choose_mode(ex, 1);
  CHECK(ball.source == mo::Source::kFrameState);
  CHECK_FALSE(ball.cache);
  CHECK(mo::choose_mode(ex, 2).source == mo::Source::kFrameState);  // a child of an ACTIVE rigid body
  CHECK(mo::choose_mode(ex, 2).frame_state_object == 1);
  CHECK(mo::choose_mode(ex, 3).code == "PHY-MIXED");
  CHECK(mo::choose_mode(ex, 4).source == mo::Source::kNone);  // a passive rigid body without keys does not move
  const mo::ModeChoice sw = mo::choose_mode(ex, 5);
  CHECK(sw.source == mo::Source::kFrameState);
  CHECK(sw.kinematic_split);  // an animated kinematic flag splits the span (BL-10)
  CHECK(sw.code.empty());
  const mo::ModeChoice abc = mo::choose_mode(ex, 6);
  CHECK(abc.source == mo::Source::kFrameState);
  CHECK(abc.cache);
  CHECK(mo::choose_mode(ex, 7).source == mo::Source::kNone);
  CHECK(mo::choose_mode(ex, 8).source == mo::Source::kKeyed);
  CHECK(mo::choose_mode(ex, 9).source == mo::Source::kKeyed);
  CHECK(mo::choose_mode(ex, 1, 0.5).code == "PHY-REMAP");
  CHECK(mo::choose_mode(ex, 6, 0.5).code.empty());  // frame_map refuses rigid bodies only
  // an unbaked world refuses its rigid bodies, not its transform caches
  bl::ExchangeIn unbaked = ex;
  unbaked.rigidbody_world["point_cache"]["is_baked"] = Tree::boolean(false);
  CHECK(mo::choose_mode(unbaked, 1).code == "PHY-UNBAKED");
  CHECK(mo::choose_mode(unbaked, 6).code.empty());
  // MOT-SOURCE: no source but varying samples
  const mo::ModeChoice none = mo::choose_mode(ex, 7);
  CHECK_FALSE(mo::motion_source(none, false, bl::Policy::kStrict));
  const auto strict = mo::motion_source(none, true, bl::Policy::kStrict);
  REQUIRE(strict);
  CHECK(strict->code == "MOT-SOURCE");
  CHECK(strict->severity == "refuse");
  CHECK(mo::motion_source(none, true, bl::Policy::kLenient)->severity == "warn");
  CHECK_FALSE(mo::motion_source(mo::choose_mode(ex, 0), true, bl::Policy::kStrict));
  // an index out of range is refused
  ex.objects[7].parent = 99;
  CHECK_THROWS_AS(mo::choose_mode(ex, 8), std::invalid_argument);
}

TEST_CASE("a fitted body's report: POS-RES, TIME-RES, MOT-FAR, LGT-MOVE-NEAR") {
  const mo::TimeMap tm = time24(25);
  mo::SampleSet s;
  CHECK(mo::body_report(s, tm, kEps).empty());
  s.floor_px = 0.003;  // > eps/8
  const mo::Breakpoint fast{tm.to_ns(20.0f), mo::Continuity::kC0, 0.0, 1e5};  // 1e5 px/s x q(20) = 40 ns: 4e-3 px
  const mo::Breakpoint slow{tm.to_ns(21.0f), mo::Continuity::kC0, 0.0, 10.0};
  s.breakpoints = {fast, slow};
  s.fit.t_ns = {0};
  s.fit.q = {{1, 0, 0, 0}};
  s.fit.p = {{70.0, 0.0, 0.0}};
  s.fit.w_pos = {1.0};
  s.fit.w_rot = {1.0};
  s.fit.floor1 = {0.0};
  s.light_near = true;
  const std::vector<bl::ReportItem> items = mo::body_report(s, tm, kEps);
  std::vector<std::string> codes;
  for (const auto& i : items) {
    codes.push_back(i.code);
    CHECK(i.severity == "warn");
    CHECK(i.bound_kind == bl::BoundKind::kMeasured);
  }
  CHECK(codes == std::vector<std::string>{"POS-RES", "TIME-RES", "MOT-FAR", "LGT-MOVE-NEAR"});
  REQUIRE(items.size() == 4);
  CHECK(items[0].bound == doctest::Approx(kEps / 2.0 + 2 * 0.003));
  CHECK(items[2].bound == doctest::Approx(70.0));
}
