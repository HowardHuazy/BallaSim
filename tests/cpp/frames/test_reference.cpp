// U4 (docs/design/phase3/frames.md §8.1): the exposure reference (§4.4) through the renderer, on Embree. K = 1 under a
// global shutter equals dense() bitwise; a moving 10:1 edge (N = 64) at D = 1, 4, 16, 64 px stays within 0.006 (p99.9
// over the moving-edge mask) of the same renderer at K = 4096; a moving point light makes the probe raise K; K <= T
// (K = 1 at T = 1 ns); e_fact is 0 for a constant waveform, and the factorization test holds >= 4 samples per
// sub-window. Also: rolling rows' exact waveform means, each sample's own window means at K > 1, e_fact's sub-windows
// by sample midpoint, levels that stay multiples of n_sub under the cap, sample reuse (and the same frame without
// it), ROI crops, motion facts, visibility, validation.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "core/traj/spline_build.h"
#include "frames/reference.h"
#include "frames/timing.h"
#include "render/renderer.h"
#include "scenes.h"
#include "sensor/params.h"
#include "sensor/waveform.h"

namespace {

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;
using frames::ExposureReference;
using frames::RefFrame;
using frames::RefOptions;
using frames::Schedule;

constexpr int64_t kSpanEnd = 60'000'000;  // bundles span [0, 60 ms]
constexpr int64_t kSoe = 20'000'000;      // the frame's start of exposure
constexpr int64_t kT = 10'000'000;        // 10 ms exposure
constexpr double kF = 100.0;

Vec3<double> spline_position(const Trajectory& tr, int64_t t) {
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
  return traj::evaluate<6>(v, st).p;
}

Schedule one_frame(int32_t height, int64_t exposure, frames::Shutter shutter = frames::Shutter::kGlobal,
                   int64_t row_dt = 0) {
  return Schedule::uniform({exposure, row_dt, height, shutter, frames::Readout::kTopDown}, kSoe, 30'000'000, 1,
                           {0, kSpanEnd});
}

std::vector<sensor::Waveform> constant(size_t groups) {
  return std::vector<sensor::Waveform>(groups, sensor::make_constant_waveform());
}

RendererOptions embree() { return options(BackendKind::kEmbree, 4); }

// A plane at z = 1 m, albedo 1 for x < 0 and 0.1 for x > 0 (a 10:1 edge), under an ambient light, seen by a camera
// translating along +x so that the edge's image sweeps d_px during [kSoe, kSoe + exposure] and is at the image centre
// at the frame's middle. One group; waveforms are the caller's.
SceneBundle edge_scene(double d_px, int64_t exposure, int width, int height) {
  const double v = d_px / (kF * 1e-9 * static_cast<double>(exposure));
  SceneBundle s;
  s.t_start_ns = 0;
  s.t_end_ns = kSpanEnd;
  s.light_groups = {"default"};
  const Quat<double> id = quat_identity<double>();
  s.trajectories = {moving_trajectory(0, kSpanEnd, id, {0, 0, 0}, {v, 0, 0}, {0, 0, 0})};
  const double xc = spline_position(s.trajectories[0], kSoe + exposure / 2).x;
  s.cameras.push_back(pinhole(width, height, kF, 0));
  Body plane = make_body(1, plane_mesh(4.0, 2));
  Material dark;
  dark.base_color = {0.1f, 0.1f, 0.1f};
  plane.materials.push_back(dark);
  for (int q = 0; q < 4; ++q) plane.mesh.material_ids.insert(plane.mesh.material_ids.end(), 2, uint16_t(q % 2));
  plane.T_world_body = translation(xc, 0, 1.0);
  s.bodies.push_back(plane);
  s.lights.push_back(ambient_light(1.0f));
  return s;
}

// A static camera over a white plane at z = 1 m, lit by a point light 0.1 m in front of it that a meshless body
// carries along +x at 10 m/s (0.1 m per 10 ms exposure, centred on the image at the frame's middle), plus a dim
// ambient term.
SceneBundle light_scene() {
  SceneBundle s;
  s.t_start_ns = 0;
  s.t_end_ns = kSpanEnd;
  s.light_groups = {"default"};
  const Quat<double> id = quat_identity<double>();
  s.trajectories = {static_trajectory(0, kSpanEnd, id, {0, 0, 0}),
                    moving_trajectory(0, kSpanEnd, id, {0, 0, 0.9}, {10.0, 0, 0}, {0, 0, 0})};
  const double xl = spline_position(s.trajectories[1], kSoe + kT / 2).x;
  s.trajectories[1] = moving_trajectory(0, kSpanEnd, id, {-xl, 0, 0.9}, {10.0, 0, 0}, {0, 0, 0});
  s.cameras.push_back(pinhole(32, 8, kF, 0));
  Body plane = make_body(1, plane_mesh(4.0, 2));
  plane.T_world_body = translation(0, 0, 1.0);
  s.bodies.push_back(plane);
  Body carrier;
  carrier.id = 2;
  carrier.name = "carrier";
  carrier.trajectory = 1;
  s.bodies.push_back(carrier);
  Light l = point_light({0, 0, 0}, 0.05f, false);
  l.body = 1;
  s.lights.push_back(l);
  s.lights.push_back(ambient_light(0.01f));
  return s;
}

// A textured floor plane, a box (static or moving) casting shadows, a point light (group 0), a spot light (group 1)
// and ambient terms in both groups.
SceneBundle mixed_scene(bool box_moves, int width = 40, int height = 12) {
  SceneBundle s;
  s.t_start_ns = 0;
  s.t_end_ns = kSpanEnd;
  s.light_groups = {"default", "spot"};
  const Quat<double> id = quat_identity<double>();
  s.trajectories = {static_trajectory(0, kSpanEnd, id, {0, 0, 0}),
                    box_moves ? moving_trajectory(0, kSpanEnd, id, {-0.2, 0.0, 1.4}, {4.0, 0.0, 0.0}, {0, 0, 2.0})
                              : static_trajectory(0, kSpanEnd, id, {-0.05, 0.0, 1.4})};
  s.cameras.push_back(pinhole(width, height, 40.0, 0));
  std::vector<uint8_t> tex(64 * 64 * 3);
  for (int j = 0; j < 64; ++j) {
    for (int i = 0; i < 64; ++i) {
      for (int c = 0; c < 3; ++c) {
        tex[(static_cast<size_t>(j) * 64 + static_cast<size_t>(i)) * 3 + static_cast<size_t>(c)] =
            static_cast<uint8_t>(((i / 8 + j / 8) % 2) ? 200 - 20 * c : 50 + 10 * c);
      }
    }
  }
  Body floor = make_body(1, plane_mesh(2.0, 4), {0.9f, 0.8f, 0.7f});
  floor.materials[0].texture = tex;
  floor.materials[0].texture_width = floor.materials[0].texture_height = 64;
  floor.materials[0].texture_channels = 3;
  floor.T_world_body = translation(0, 0, 2.0);
  s.bodies.push_back(floor);
  Body box = make_body(2, box_mesh(0.08f, 0.08f, 0.08f), {0.4f, 0.6f, 0.5f});
  box.trajectory = 1;
  s.bodies.push_back(box);
  s.lights.push_back(point_light({-0.3, -0.4, 0.6}, 2.0f));
  Light spot;
  spot.kind = BundleLightKind::kSpot;
  spot.group = 1;
  spot.position = {0.4, 0.3, 0.5};
  spot.direction = {-0.2, -0.1, 1.0};
  spot.spot_inner_rad = 0.3f;
  spot.spot_outer_rad = 0.7f;
  spot.intensity = 3.0f;
  s.lights.push_back(spot);
  s.lights.push_back(ambient_light(0.05f));
  s.lights.push_back(ambient_light(0.02f, 1));
  return s;
}

// p99.9 (nearest rank) and max of |ln a - ln b| over the pixels where `on` holds.
std::pair<double, double> log_error(const std::vector<float>& a, const std::vector<float>& b,
                                    const std::vector<bool>& on) {
  std::vector<double> e;
  for (size_t i = 0; i < a.size(); ++i) {
    if (on[i]) e.push_back(std::abs(std::log(static_cast<double>(a[i])) - std::log(static_cast<double>(b[i]))));
  }
  REQUIRE(!e.empty());
  std::sort(e.begin(), e.end());
  const size_t r = std::min(e.size(), static_cast<size_t>(std::ceil(0.999 * static_cast<double>(e.size())))) - 1;
  return {e[r], e.back()};
}

bool has_warning(const RefFrame& f, const std::string& code) {
  return std::any_of(f.warnings.begin(), f.warnings.end(), [&](const frames::Warning& w) { return w.code == code; });
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

}  // namespace

TEST_CASE("U4: K = 1 under a global shutter equals dense() bitwise; ROI crops equal the full frame's pixels") {
  const SceneBundle s = mixed_scene(true);
  Renderer r(s, embree());
  ExposureReference ref(r, 0, constant(2), {0, 0, 40, 12});
  const Schedule sched = one_frame(12, kT);
  RefOptions o;
  o.k_fixed = 1;
  const RefFrame f = ref.frame(sched, 0, o);
  CHECK(f.k_used == 1);
  CHECK(f.mask_px == 0);
  const int64_t t = kSoe + kT / 2;
  CHECK(sched.frame_mid(0) == t);
  const DenseFrame d = r.dense(0, t, true, true);
  REQUIRE(d.irradiance.size() == f.per_group.size());
  bool equal = true;
  for (size_t i = 0; i < f.per_group.size(); ++i) equal = equal && same_bits(d.irradiance[i], f.per_group[i]);
  CHECK(equal);
  bool mono = true, gt = true;
  for (size_t p = 0; p < f.mono.size(); ++p) {
    const double sum = static_cast<double>(d.irradiance[2 * p]) + static_cast<double>(d.irradiance[2 * p + 1]);
    mono = mono && same_bits(static_cast<float>(sum), f.mono[p]);
    gt = gt && same_bits(d.gt[p].depth, f.depth[p]) && d.gt[p].instance == f.instance[p];
  }
  CHECK(mono);
  CHECK(gt);
  for (double m : f.m_mean) CHECK(m == 1.0);

  // An ROI equals the full frame's pixels at the same K.
  o.k_fixed = 3;
  const RefFrame full = ref.frame(sched, 0, o);
  ExposureReference crop(r, 0, constant(2), {5, 3, 20, 9});
  const RefFrame c = crop.frame(sched, 0, o);
  REQUIRE(c.mono.size() == 15u * 6u);
  bool roi = true;
  for (int y = 3; y < 9; ++y) {
    for (int x = 5; x < 20; ++x) {
      const size_t pc = static_cast<size_t>((y - 3) * 15 + (x - 5)), pf = static_cast<size_t>(y * 40 + x);
      roi = roi && same_bits(c.mono[pc], full.mono[pf]) && same_bits(c.per_group[2 * pc], full.per_group[2 * pf]) &&
            same_bits(c.per_group[2 * pc + 1], full.per_group[2 * pf + 1]);
    }
  }
  CHECK(roi);
}

TEST_CASE("U4: a static scene takes one sample per row; rolling rows carry their own exact waveform means") {
  const SceneBundle s = mixed_scene(false);
  Renderer r(s, embree());
  const sensor::Waveform flicker = sensor::make_fourier_waveform(120.0, {0.4, 0.1}, {0.3, 1.1});
  const std::vector<sensor::Waveform> waves{sensor::make_constant_waveform(), flicker};
  ExposureReference ref(r, 0, waves, {0, 0, 40, 12});
  const Schedule sched = one_frame(12, kT, frames::Shutter::kRolling, 700'000);
  CHECK_FALSE(ref.motion(sched, 0).any_motion);
  const RefFrame f = ref.frame(sched, 0, {});
  CHECK(f.k_used == 1);
  CHECK(f.mask_px == 0);
  CHECK(f.fact_p99 == 0.0f);
  bool means = true, values = true;
  for (int y = 0; y < 12; ++y) {
    const int64_t soe = sched.soe(0, y);
    CHECK(soe == kSoe + 700'000 * y);
    means = means && f.m_mean[2 * static_cast<size_t>(y)] == 1.0 &&
            f.m_mean[2 * static_cast<size_t>(y) + 1] == sensor::waveform_mean<double>(flicker, soe, soe + kT);
    const DenseFrame d = r.dense(0, soe + kT / 2, true, false);
    for (int x = 0; x < 40; ++x) {
      const size_t p = static_cast<size_t>(y * 40 + x);
      for (size_t g = 0; g < 2; ++g) {
        const double expect = f.m_mean[2 * static_cast<size_t>(y) + g] * static_cast<double>(d.irradiance[2 * p + g]);
        values = values && same_bits(static_cast<float>(expect), f.per_group[2 * p + g]);
      }
    }
  }
  CHECK(means);
  CHECK(values);
}

TEST_CASE("U4: at K > 1 each sample carries its own window's exact waveform means (rolling, T not divisible by K)") {
  // F_g = (sum_j mbar_{g,j} I_g(t_j)) / K bitwise, with t_j = soe + floor((2j + 1) T / (2K)) and mbar_{g,j} the
  // exact mean over [soe + floor(j T / K), soe + floor((j + 1) T / K)], for a pulse train and a 2-harmonic flicker.
  const SceneBundle s = mixed_scene(false);
  Renderer r(s, embree());
  const std::vector<sensor::Waveform> waves{sensor::make_pulse_waveform(170.0, 0.3, 4e-4, 1e-3, 0.2),
                                            sensor::make_fourier_waveform(120.0, {0.4, 0.1}, {0.3, 1.1})};
  ExposureReference ref(r, 0, waves, {0, 0, 40, 12});
  constexpr int64_t t = 4'000'003;
  constexpr int32_t kk = 7;
  const Schedule sched = one_frame(12, t, frames::Shutter::kRolling, 700'000);
  RefOptions o;
  o.k_fixed = kk;
  const RefFrame f = ref.frame(sched, 0, o);
  REQUIRE(f.k_used == kk);
  CHECK(f.rendered == 40 * 12 * kk);
  bool equal = true, varies = false;
  for (int y = 0; y < 12; ++y) {
    const int64_t soe = sched.soe(0, y);
    std::vector<double> acc(40 * 2, 0.0);
    for (int32_t j = 0; j < kk; ++j) {
      const int64_t a = soe + j * t / kk, b = soe + (j + 1) * t / kk;
      const double m[2] = {sensor::waveform_mean<double>(waves[0], a, b),
                           sensor::waveform_mean<double>(waves[1], a, b)};
      varies = varies || m[0] != sensor::waveform_mean<double>(waves[0], soe, soe + t);
      const DenseFrame d = r.dense(0, soe + (2 * j + 1) * t / (2 * kk), true, false);
      for (size_t i = 0; i < acc.size(); ++i) {
        acc[i] += m[i % 2] * static_cast<double>(d.irradiance[static_cast<size_t>(y) * 80 + i]);
      }
    }
    for (size_t i = 0; i < acc.size(); ++i) {
      equal = equal && same_bits(static_cast<float>(acc[i] / kk), f.per_group[static_cast<size_t>(y) * 80 + i]);
    }
  }
  CHECK(varies);
  CHECK(equal);
}

TEST_CASE("U4: a moving 10:1 edge (N = 64) at D = 1, 4, 16, 64 px is within 0.006 of K = 4096 on the edge mask") {
  for (const double d_px : {1.0, 4.0, 16.0, 64.0}) {
    const int width = static_cast<int>(d_px) + 10;
    const SceneBundle s = edge_scene(d_px, kT, width, 1);
    Renderer r(s, embree());
    REQUIRE(r.options().scene.n_subsamples == 64u);
    ExposureReference ref(r, 0, constant(1), {0, 0, width, 1});
    const Schedule sched = one_frame(1, kT);
    const frames::MotionFacts mf = ref.motion(sched, 0);
    CHECK(mf.any_motion);
    CHECK_FALSE(mf.light_motion);
    CHECK(mf.d_max_px == doctest::Approx(d_px).epsilon(1e-4));
    const RefFrame f = ref.frame(sched, 0, {});
    RefOptions truth_opt;
    truth_opt.k_fixed = 4096;
    const RefFrame truth = ref.frame(sched, 0, truth_opt);
    REQUIRE(truth.k_used == 4096);
    const auto [lo, hi] = std::minmax_element(truth.mono.begin(), truth.mono.end());
    CHECK(*hi / *lo == doctest::Approx(10.0).epsilon(1e-3));
    std::vector<bool> edge(truth.mono.size());
    size_t n_edge = 0;
    for (size_t p = 0; p < edge.size(); ++p) {
      const double v = truth.mono[p];
      edge[p] = std::log(v) > std::log(static_cast<double>(*lo)) + 1e-3 &&
                std::log(v) < std::log(static_cast<double>(*hi)) - 1e-3;
      n_edge += edge[p] ? 1 : 0;
    }
    CHECK(static_cast<double>(n_edge) >= d_px);
    const auto [p999, worst] = log_error(f.mono, truth.mono, edge);
    MESSAGE("D = " << d_px << " px: K0 = " << f.k_used << ", mask K = " << f.k_mask << " on " << f.mask_px
                   << " px, probe p99.9 " << f.probe_p999 << "; |ln F - ln F_4096| on " << n_edge
                   << " edge px: p99.9 " << p999 << ", max " << worst);
    CHECK(p999 <= 0.006);
    CHECK(f.probe_p999 <= 0.005f);
    CHECK_FALSE(has_warning(f, "FRM-REF-K"));
  }
}

TEST_CASE("U4: a moving point light makes the probe raise K; its motion facts bound the light's rate") {
  const SceneBundle s = light_scene();
  Renderer r(s, embree());
  ExposureReference ref(r, 0, constant(1), {0, 0, 32, 8});
  const Schedule sched = one_frame(8, kT);
  const frames::MotionFacts mf = ref.motion(sched, 0);
  CHECK(mf.any_motion);
  CHECK(mf.light_motion);
  CHECK(mf.d_max_px == 0.0);
  // rho = (v / d)(2 + 3) with v = 10 m/s and d >= 0.1 m (the nearest visible point lies just off the foot of the
  // light, half a pixel away in x and y: d = hypot(0.1, 0.005, 0.005)).
  CHECK(mf.light_rate_per_s <= 500.0 * (1 + 1e-9));
  CHECK(mf.light_rate_per_s >= 500.0 * 0.1 / std::sqrt(0.01 + 2 * 0.005 * 0.005) * (1 - 1e-6));
  const RefFrame f = ref.frame(sched, 0, {});
  RefOptions truth_opt;
  truth_opt.k_fixed = 4096;
  const RefFrame truth = ref.frame(sched, 0, truth_opt);
  const auto [p999, worst] = log_error(f.mono, truth.mono, std::vector<bool>(f.mono.size(), true));
  MESSAGE("moving light: K0 = " << f.k_used << ", mask K = " << f.k_mask << " on " << f.mask_px << " px, probe p99.9 "
                                << f.probe_p999 << "; |ln F - ln F_4096|: p99.9 " << p999 << ", max " << worst);
  CHECK(f.k_used > 2);  // K0 = 2 (no flow), raised by the probe
  CHECK(f.mask_px > 0);
  CHECK(p999 <= 0.006);
  // Every level reuses the samples below it (3j + 1 <- j; the mask level's pixels whole), so each pixel's samples
  // are rendered once.
  CHECK(f.rendered == 32 * 8 * int64_t{f.k_used} + f.mask_px * (f.k_mask - f.k_used));
  // Without the reuse budget nothing is kept for reuse: more renders, the same frame.
  RefOptions no_reuse;
  no_reuse.reuse_budget_mb = 0.0;
  const RefFrame g = ref.frame(sched, 0, no_reuse);
  CHECK(g.rendered > f.rendered);
  CHECK(g.k_used == f.k_used);
  CHECK(g.k_mask == f.k_mask);
  CHECK(g.mask_px == f.mask_px);
  CHECK(same_bits(g.probe_p999, f.probe_p999));
  bool same = g.per_group.size() == f.per_group.size();
  for (size_t i = 0; same && i < f.per_group.size(); ++i) same = same_bits(g.per_group[i], f.per_group[i]);
  for (size_t p = 0; same && p < f.mono.size(); ++p) same = same_bits(g.mono[p], f.mono[p]);
  CHECK(same);
  // The same frame with K fixed at 2 is far off: the probe's raise was needed.
  RefOptions two;
  two.k_fixed = 2;
  CHECK(log_error(ref.frame(sched, 0, two).mono, truth.mono, std::vector<bool>(f.mono.size(), true)).first > 0.006);
}

TEST_CASE("U4: K <= T; K = 1 at T = 1 ns and at T = 0") {
  for (const int64_t t : {int64_t{0}, int64_t{1}, int64_t{3}}) {
    const SceneBundle s = edge_scene(4.0, kT, 14, 1);
    Renderer r(s, embree());
    ExposureReference ref(r, 0, constant(1), {0, 0, 14, 1});
    const Schedule sched = one_frame(1, t);
    const RefFrame f = ref.frame(sched, 0, {});
    CHECK(f.k_used <= std::max<int64_t>(t, 1));
    CHECK(f.k_mask <= std::max<int64_t>(t, 1));
    if (t <= 1) CHECK(f.k_used == 1);
    RefOptions o;
    o.k_fixed = 8;
    CHECK(ref.frame(sched, 0, o).k_used == std::max<int64_t>(t, 1));
    if (t == 0) {  // instantaneous: the frame is the irradiance at t_soe
      o.k_fixed = 0;
      const DenseFrame d = r.dense(0, kSoe, true, false);
      bool equal = true;
      for (size_t p = 0; p < f.mono.size(); ++p) equal = equal && same_bits(d.irradiance[p], f.per_group[p]);
      CHECK(equal);
    }
  }
}

TEST_CASE("U4: factorization: e_fact = 0 for a constant waveform; >= 4 samples per sub-window before the test") {
  const SceneBundle s = edge_scene(1.0, kT, 11, 1);
  Renderer r(s, embree());
  const Schedule sched = one_frame(1, kT);
  RefOptions o;
  o.n_sub = 4;
  {
    ExposureReference ref(r, 0, constant(1), {0, 0, 11, 1});
    const RefFrame f = ref.frame(sched, 0, o);
    CHECK(f.fact_p99 == 0.0f);
    CHECK(f.k_used == ref.frame(sched, 0, {}).k_used);  // no raise without flicker
  }
  const std::vector<sensor::Waveform> flicker{sensor::make_fourier_waveform(120.0, {0.5}, {0.2})};
  ExposureReference ref(r, 0, flicker, {0, 0, 11, 1});
  const RefFrame f = ref.frame(sched, 0, o);
  MESSAGE("flicker, n_sub = 4: K0 = " << f.k_used << ", mask K = " << f.k_mask << ", fact_p99 " << f.fact_p99);
  CHECK(f.k_used >= 16);
  CHECK(f.k_used % 4 == 0);
  CHECK(f.k_mask % 4 == 0);
  CHECK(f.fact_p99 > 0.0f);
  CHECK_FALSE(has_warning(f, "FRM-FACTOR"));
  o.k_max = 8;  // the cap leaves 2 samples per sub-window
  const RefFrame capped = ref.frame(sched, 0, o);
  CHECK(capped.k_used == 8);
  CHECK(has_warning(capped, "FRM-FACTOR"));
}

TEST_CASE("U4: e_fact takes each sub-window's samples by their midpoints; capped levels stay multiples of n_sub") {
  const SceneBundle s = edge_scene(1.0, kT, 11, 1);
  Renderer r(s, embree());
  const Schedule sched = one_frame(1, kT);
  const sensor::Waveform w = sensor::make_fourier_waveform(120.0, {0.5}, {0.2});
  ExposureReference ref(r, 0, std::vector<sensor::Waveform>{w}, {0, 0, 11, 1});

  // K = 10 is not a multiple of n_sub = 4: the midpoints (2j + 1) / 20 of the exposure put samples 0-1, 2-4, 5-6 and
  // 7-9 in the four sub-windows (floor(j n_sub / K) would take 0-2, 3-4, 5-7, 8-9).
  constexpr int32_t kk = 10, n_sub = 4;
  RefOptions o;
  o.k_fixed = kk;
  o.n_sub = n_sub;
  const RefFrame f = ref.frame(sched, 0, o);
  REQUIRE(f.k_used == kk);
  std::vector<double> m(kk);
  std::vector<DenseFrame> d;
  for (int32_t j = 0; j < kk; ++j) {
    m[static_cast<size_t>(j)] = sensor::waveform_mean<double>(w, kSoe + j * kT / kk, kSoe + (j + 1) * kT / kk);
    d.push_back(r.dense(0, kSoe + (2 * j + 1) * kT / (2 * kk), true, false));
  }
  // p99 (nearest rank) of 11 pixels is their maximum.
  const auto fact_max = [&](const auto& sub_of) {
    double e_max = 0.0;
    for (size_t p = 0; p < 11; ++p) {
      double fp = 0.0;
      for (size_t j = 0; j < static_cast<size_t>(kk); ++j) fp += m[j] * static_cast<double>(d[j].irradiance[p]);
      fp /= kk;
      for (int32_t sw = 0; sw < n_sub; ++sw) {
        double smi = 0.0, sm = 0.0, si = 0.0;
        int32_t cnt = 0;
        for (int32_t j = 0; j < kk; ++j) {
          if (sub_of(j) != sw) continue;
          const double ii = d[static_cast<size_t>(j)].irradiance[p];
          smi += m[static_cast<size_t>(j)] * ii;
          sm += m[static_cast<size_t>(j)];
          si += ii;
          ++cnt;
        }
        if (cnt > 0) e_max = std::max(e_max, std::abs(smi / cnt - (sm / cnt) * (si / cnt)) / fp);
      }
    }
    return e_max;
  };
  const double by_mid = fact_max([](int32_t j) { return (2 * j + 1) * n_sub / (2 * kk); });
  const double by_index = fact_max([](int32_t j) { return j * n_sub / kk; });
  MESSAGE("K = 10, n_sub = 4: fact_p99 " << f.fact_p99 << "; by midpoint " << by_mid << ", by floor(j n / K) "
                                         << by_index);
  CHECK(same_bits(f.fact_p99, static_cast<float>(by_mid)));
  CHECK(static_cast<float>(by_index) != static_cast<float>(by_mid));

  // A probe that fails at every level: K0 = 12 (4 n_sub), then 36, then k_max = 50 rounded down to 48, a multiple of
  // n_sub = 3; the mask levels are 3K.
  RefOptions capped;
  capped.n_sub = 3;
  capped.k_max = 50;
  capped.probe_tol = 1e-9;
  const RefFrame c = ref.frame(sched, 0, capped);
  MESSAGE("k_max = 50, n_sub = 3, failing probe: K = " << c.k_used << ", mask K = " << c.k_mask);
  CHECK(c.k_used == 48);
  CHECK(c.k_mask == 144);
  CHECK(has_warning(c, "FRM-REF-K"));
  CHECK_FALSE(has_warning(c, "FRM-FACTOR"));
}

TEST_CASE("visibility: R / R_noshadow, 0 in the umbra, exactly 1 where nothing occludes") {
  const SceneBundle s = mixed_scene(false, 60, 30);
  Renderer r(s, embree());
  ExposureReference ref(r, 0, constant(2), {0, 0, 60, 30});
  const std::vector<float> v = ref.visibility(kSoe);
  REQUIRE(v.size() == 60u * 30u * 2u);
  size_t ones = 0, shadow = 0;
  float lowest = 1.0f;
  bool in_range = true;
  for (size_t p = 0; p < 60u * 30u; ++p) {
    in_range = in_range && v[2 * p] >= 0.0f && v[2 * p] <= 1.0f + 1e-6f;
    ones += v[2 * p] == 1.0f ? 1 : 0;
    shadow += v[2 * p] < 0.5f ? 1 : 0;  // the box's shadow keeps group 0's ambient term
    lowest = std::min(lowest, v[2 * p]);
  }
  MESSAGE("visibility of group 0: " << ones << " px exactly 1, " << shadow << " px below 0.5, lowest " << lowest);
  CHECK(in_range);
  CHECK(ones > 1000u);
  CHECK(shadow >= 9u);
  CHECK(lowest > 0.05f);  // the ambient term is never shadowed
}

TEST_CASE("exposure reference validation") {
  const SceneBundle s = edge_scene(4.0, kT, 14, 2);
  Renderer r(s, embree());
  for (const frames::Roi roi : {frames::Roi{0, 0, 15, 2}, frames::Roi{3, 0, 3, 2}, frames::Roi{-1, 0, 4, 2}}) {
    try {
      ExposureReference bad(r, 0, constant(1), roi);
      FAIL("ROI accepted");
    } catch (const frames::FramesError& e) {
      CHECK(e.code == "FRM-VALUE");
    }
  }
  CHECK_THROWS_AS(ExposureReference(r, 1, constant(1), {0, 0, 14, 2}), std::invalid_argument);
  CHECK_THROWS_AS(ExposureReference(r, 0, constant(2), {0, 0, 14, 2}), std::invalid_argument);
  ExposureReference ref(r, 0, constant(1), {0, 0, 14, 2});
  CHECK_THROWS_AS(ref.frame(one_frame(3, kT), 0, {}), std::invalid_argument);  // the camera has 2 rows
  CHECK_THROWS_AS(ref.frame(one_frame(2, kT), 1, {}), std::invalid_argument);
  for (const RefOptions o : {RefOptions{0.0, 0.005, 256, 0, 1}, RefOptions{0.25, 0.0, 256, 0, 1},
                             RefOptions{0.25, 0.005, 1, 0, 1}, RefOptions{0.25, 0.005, 4097, 0, 1},
                             RefOptions{0.25, 0.005, 256, -1, 1}, RefOptions{0.25, 0.005, 256, 12289, 1},
                             RefOptions{0.25, 0.005, 256, 0, 0}, RefOptions{0.25, 0.005, 256, 0, 65},
                             RefOptions{0.25, 0.005, 256, 0, 1, -1.0}}) {
    CHECK_THROWS_AS(ref.frame(one_frame(2, kT), 0, o), std::invalid_argument);
  }
}
