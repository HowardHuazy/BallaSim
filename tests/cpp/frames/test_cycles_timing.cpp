// U3 (docs/design/phase3/frames.md §8.1): the Cycles timing (§4.2) and motion steps (§4.3). E1's closed forms
// against E1's Blender 4.2.1 measurements; global and rolling snaps at frames 0, 30, 300, 9000 and -120 (30 and
// 24/1.001 fps): the realized centre is to_ns(c) - round(delta_c f32(d) f32(S) P) (checked in exact integer
// arithmetic), the rows that Cycles exposes (its forward model at float32 S and d) land on the snapped schedule's rows,
// the derivation puts row i at t_b0 + i Delta_b + T'/2 within 1e-9 frames, d in {0.25, 0.6} and a readout-dominated
// case on routes F/O and S (s = 4, 8 with margins); T' = 0 (global: motion blur off, t_hat = to_ns(c); rolling: on,
// d = 0); the realized length within 6e-8 / 1.2e-7 relative + 0.5 ns; bands; the expected ctime under framelen 0.5;
// node_times bitwise against Cycles' float32 expression; the motion steps (nlerp and slerp chord errors against closed
// forms, the pivot, minimality under the chord budget, the rig probe, Blender's time resolution, which never asks for
// more sub-renders, bodies whose AABB corners all leave the image, splines built over the call's segments only) and
// the sub-render and row-band rules.
#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/traj/spline.h"
#include "core/traj/spline_build.h"
#include "frames/cycles_timing.h"
#include "frames/timing.h"
#include "scenes.h"

namespace {

using namespace balla;
using namespace balla::frames;
using balla::render::test::box_mesh;
using balla::render::test::make_body;
using balla::render::test::make_trajectory;
using balla::render::test::moving_trajectory;
using balla::render::test::pinhole;
using balla::render::test::static_trajectory;
using scene::motion::TimeMap;
using scene::motion::TimeSettings;

__extension__ typedef __int128 i128;

TimeMap make_time(int32_t fps, float fps_base, int32_t frame_start = 1, int32_t frame_end = 20000,
                  int32_t frame_map_old = 100, int32_t frame_map_new = 100) {
  TimeSettings s;
  s.fps = fps;
  s.fps_base_bits = std::bit_cast<uint32_t>(fps_base);
  s.frame_start = frame_start;
  s.frame_end = frame_end;
  s.frame_map_old = frame_map_old;
  s.frame_map_new = frame_map_new;
  return TimeMap(s);
}

long double period(const TimeMap& t) {
  return 1e9L * static_cast<long double>(t.base_num()) /
         (static_cast<long double>(t.fps()) * static_cast<long double>(t.base_den()));
}
// The true time of a (real) Blender frame value: t_ref + (frame - F_ref) P.
long double frame_ns(const TimeMap& t, long double frame) {
  return static_cast<long double>(t.t_ref_ns()) + (frame - static_cast<long double>(t.frame_ref())) * period(t);
}

i128 div_round_even(i128 n, i128 d) {
  i128 q = n / d, r = n % d;
  if (r < 0) {
    q -= 1;
    r += d;
  }
  if (2 * r > d || (2 * r == d && (q & 1) != 0)) q += 1;
  return q;
}

// round(delta_c x P_ns) for x = f32(d) f32(S), exactly: x = m 2^e (m < 2^53), P = 1e9 B_num / (fps B_den).
int64_t exact_delta(const TimeMap& t, float d32, float s32) {
  const double x = static_cast<double>(d32) * static_cast<double>(s32);  // exact: 24 + 24 bits
  if (x == 0.0) return 0;
  int e = 0;
  const double fr = std::frexp(x, &e);
  const auto m = static_cast<i128>(std::ldexp(fr, 53));
  e -= 53;
  i128 num = m * 1'000'000'000 * t.base_num();
  i128 den = i128{512} * t.fps() * t.base_den();
  if (e >= 0) {
    num <<= e;
  } else {
    den <<= -e;
  }
  return static_cast<int64_t>(div_round_even(num, den));
}

double ulp32(float x) {
  x = std::fabs(x);
  return static_cast<double>(std::nextafter(x, INFINITY)) - static_cast<double>(x);
}

// Cycles' realized centre of raster row i (ns, real): the forward model of camera.h at float32 S and d, rolling TOP
// (d = 1 and no row term for a global shutter).
long double cycles_row_centre(const TimeMap& t, const CyclesTiming& ct, int32_t rows, int32_t i) {
  const long double s = ct.shutter_frames, d = ct.rolling ? static_cast<long double>(ct.rolling_duration) : 1.0L;
  long double frames = static_cast<long double>(ct.expected_c);
  if (ct.motion_blur) frames -= static_cast<long double>(kDeltaC) * d * s;
  if (ct.rolling) frames += ((i + 1.0L) / rows - 0.5L) * (1.0L - d) * s;
  return frame_ns(t, frames);
}

}  // namespace

TEST_CASE("U3 Cycles timing: shutter constants") {
  CHECK(kShutterWidth == 65025.0 / 65536.0);
  CHECK(kDeltaC == 0.5 - (kShutterA + kShutterB) / 2);
  CHECK(kDeltaC == 1.0 / 512);
  CHECK(std::fabs(kShutterWidth - 0.99220276) < 1e-8);
}

TEST_CASE("U3 Cycles timing: E1's closed forms against Blender 4.2.1") {
  // CT e1_out/e1_results.json cases.<name>: frame 50, H = 40, 4096 spp; t_mean_frames, exposure_frames.
  struct Case {
    const char* name;
    float subframe, shutter, d;
    bool rolling;
    double t_mean, exposure;  // exposure < 0: not compared (d = 0: E1's width there is the 1 px box)
  };
  const Case cases[] = {
      {"center_s05", 0.0f, 0.5f, 1.0f, false, 49.99895538330078, 0.49635078976876484},
      {"center_s10", 0.0f, 1.0f, 1.0f, false, 49.998046875, 0.9921567159008791},
      {"center_s05_sub03", 0.3f, 0.5f, 1.0f, false, 50.299091491699215, 0.4963184505543873},
      {"roll_top_d0_s1", 0.0f, 1.0f, 0.0f, true, 50.01248779296875, -1.0},
      {"roll_top_d025_s1", 0.0f, 1.0f, 0.25f, true, 50.00886108398438, 0.24798947371585103},
  };
  constexpr int32_t kH = 40;
  for (const Case& cs : cases) {
    CAPTURE(cs.name);
    const double c = static_cast<double>(TimeMap::realized(50, cs.subframe));
    const double s = cs.shutter, d = cs.d;
    double mean = 0.0, mean_rev2 = 0.0, mean_half = 0.0;
    for (int32_t r = 0; r < kH; ++r) {
      const double rows = cs.rolling ? ((r + 1.0) / kH - 0.5) * (1.0 - d) * s : 0.0;
      mean += c - kDeltaC * d * s + rows;
      mean_rev2 += c - kDeltaC * s + rows;  // rev 2: delta_c S, not delta_c d S
      mean_half += c - kDeltaC * d * s + (cs.rolling ? ((r + 0.5) / kH - 0.5) * (1.0 - d) * s : 0.0);
    }
    mean /= kH;
    mean_rev2 /= kH;
    mean_half /= kH;
    CHECK(std::fabs(mean - cs.t_mean) < 1e-4);  // E1's noise: about 7e-5 frames
    if (cs.rolling) {
      CHECK(std::fabs(mean_half - cs.t_mean) > 1e-3);  // the (r + 1/2)/H rows are rejected
      if (d > 0.0 && d < 1.0) CHECK(std::fabs(mean_rev2 - cs.t_mean) > 1e-3);  // and so is rev 2's shift
    }
    if (cs.exposure > 0.0) CHECK(std::fabs(kShutterWidth * d * s / cs.exposure - 1.0) < 1e-3);
  }
}

TEST_CASE("U3 Cycles timing: rasters, bands and sub-renders") {
  const RowTiming rolling{10'000'000, 20'000, 48, Shutter::kRolling, Readout::kTopDown};
  const Raster full = blender_raster(rolling);
  CHECK(full.rows == 48);
  CHECK(full.row_dt_ns == 20'000.0);
  CHECK(full.first_row_offset_ns == 0.0);
  const Raster band = blender_raster(rolling, {10, 5});
  CHECK(band.rows == 5);
  CHECK(band.first_row_offset_ns == 200'000.0);
  const Raster tail = blender_raster(rolling, {40, -1});
  CHECK(tail.rows == 8);
  for (const int32_t s : {4, 8}) {
    const int32_t m = 2;
    const Raster rs = blender_raster(rolling, {0, -1}, s, m);
    CHECK(rs.rows == s * (48 + 2 * m));
    CHECK(rs.row_dt_ns == 20'000.0 / s);
    CHECK(rs.first_row_offset_ns == doctest::Approx(-(s * m + (s - 1) / 2.0) * 20'000.0 / s).epsilon(1e-15));
    // The s sub-rows of image row y start, on average, at row y's own start of exposure.
    for (const int32_t y : {0, 23, 47}) {
      long double mean = 0.0L;
      for (int32_t q = 0; q < s; ++q) {
        mean += rs.first_row_offset_ns + (s * (m + y) + q) * static_cast<long double>(rs.row_dt_ns);
      }
      CHECK(static_cast<double>(std::fabs(mean / s - y * 20'000.0L)) < 1e-6);
    }
  }
  const RowTiming global{10'000'000, 20'000, 48, Shutter::kGlobal, Readout::kTopDown};
  const Raster g = blender_raster(global, {0, -1}, 4, 2);
  CHECK(g.row_dt_ns == 0.0);
  CHECK(g.first_row_offset_ns == 0.0);
  CHECK_THROWS_AS(blender_raster(rolling, {45, 5}), std::invalid_argument);
  CHECK_THROWS_AS(blender_raster(rolling, {-1, 5}), std::invalid_argument);
  CHECK_THROWS_AS(blender_raster(rolling, {0, -1}, 0), std::invalid_argument);

  for (const int32_t n : {1, 3, 7, 48}) {
    const std::vector<RowBand> bands = row_bands(48, n);
    REQUIRE(bands.size() == static_cast<size_t>(n));
    int32_t next = 0;
    for (const RowBand& b : bands) {
      CHECK(b.row0 == next);
      CHECK(b.rows >= 48 / n);
      CHECK(b.rows <= 48 / n + 1);
      next = b.row0 + b.rows;
    }
    CHECK(next == 48);
  }
  CHECK_THROWS_AS(row_bands(48, 49), std::invalid_argument);
  CHECK_THROWS_AS(row_bands(48, 0), std::invalid_argument);

  for (const int64_t t : {int64_t{0}, int64_t{1}, int64_t{7}, int64_t{10'000'003}}) {
    for (const int32_t n : {1, 3, 64}) {
      int64_t sum = 0;
      for (int32_t j = 0; j < n; ++j) {
        const SubWindow w = sub_window(t, n, j);
        CHECK(w.offset_ns == static_cast<int64_t>(static_cast<i128>(j) * t / n));
        CHECK(w.offset_ns == sum);
        sum += w.length_ns;
      }
      CHECK(sum == t);
    }
  }
  CHECK_THROWS_AS(sub_window(10, 4, 4), std::invalid_argument);
  CHECK_THROWS_AS(sub_window(-1, 4, 0), std::invalid_argument);
}

TEST_CASE("U3 Cycles timing: global shutter at frames 0, 30, 300, 9000, -120") {
  for (const float base : {1.0f, 1.001f}) {
    const TimeMap time = make_time(base == 1.0f ? 30 : 24, base);
    const long double p = period(time);
    for (const int32_t frame : {0, 30, 300, 9000, -120}) {
      for (const int64_t t1 : {int64_t{10'000'000}, int64_t{1}, int64_t{33'333'333}, int64_t{0}}) {
        CAPTURE(base);
        CAPTURE(frame);
        CAPTURE(t1);
        const int64_t centre = time.to_ns(static_cast<float>(frame)) + 1234;
        const int64_t soe = centre - t1 / 2;
        const RowTiming timing{t1, 0, 32, Shutter::kGlobal, Readout::kTopDown};
        const Raster raster = blender_raster(timing);
        const CyclesTiming ct = cycles_timing(time, timing, raster, soe, t1);
        CHECK(!ct.rolling);
        CHECK(ct.rolling_duration == 1.0f);
        CHECK(ct.expected_c == TimeMap::realized(ct.frame, ct.subframe));
        CHECK(ct.expected_ctime == time.reported(ct.expected_c));
        CHECK(ct.q_ns == time.resolution_ns(ct.frame));
        const bool warn = ct.q_ns > 1000.0 || (t1 > 0 && ct.q_ns > 0.01 * static_cast<double>(t1));
        CHECK((!ct.warnings.empty() && ct.warnings[0].code == "FRM-TIME-RES") == warn);
        if (t1 == 0) {
          // Motion blur off: Cycles renders the frame's own time; the request is the start of exposure itself.
          CHECK(!ct.motion_blur);
          CHECK(ct.shutter_frames == 0.0f);
          CHECK(ct.realized_center_ns == time.to_ns(ct.expected_c));
          CHECK(ct.snap_shift_ns == ct.realized_center_ns - soe);
          CHECK(ct.realized_row_exposure_ns == 0.0);
          int32_t f = 0;
          float s = 0.0f;
          time.request(soe, &f, &s);
          CHECK(f == ct.frame);
          CHECK(s == ct.subframe);
          continue;
        }
        CHECK(ct.motion_blur);
        // S = F_r T' / (1e9 (b - a)), rounded once to float32.
        const long double s_exact = static_cast<long double>(t1) * 65536.0L / 65025.0L / p;
        CHECK(ct.shutter_frames == static_cast<float>(s_exact));
        // The realized centre, exactly; the request includes delta_t = round(128 T' / 65025).
        CHECK(ct.realized_center_ns == time.to_ns(ct.expected_c) - exact_delta(time, 1.0f, ct.shutter_frames));
        const auto delta_t = static_cast<int64_t>(div_round_even(static_cast<i128>(t1) * 128, 65025));
        const int64_t t_req = soe + t1 / 2 + delta_t;
        CHECK(ct.snap_shift_ns == ct.realized_center_ns - (t_req - delta_t));
        CHECK(std::fabs(static_cast<double>(ct.snap_shift_ns)) <= ct.q_ns + 2.0);
        // Cycles' forward model: the exposure centre lands on the snapped window's centre.
        const long double fwd = cycles_row_centre(time, ct, 1, 0);
        const long double mid = static_cast<long double>(soe + ct.snap_shift_ns) + t1 / 2.0L;
        CHECK(static_cast<double>(std::fabs(fwd - mid)) <= 1.5);
        // The realized length: (b - a) f32(S) P within 6e-8 T' + 0.5 ns of T'.
        CHECK(ct.realized_row_exposure_ns ==
              doctest::Approx(static_cast<double>(65025.0L / 65536.0L * ct.shutter_frames * p)).epsilon(1e-15));
        CHECK(std::fabs(ct.realized_row_exposure_ns - static_cast<double>(t1)) <= 6e-8 * static_cast<double>(t1) + 0.5);
      }
    }
  }
}

TEST_CASE("U3 Cycles timing: rolling snap, d in {0.25, 0.6} and readout-dominated, routes F/O and S") {
  constexpr int32_t kH = 48;
  struct Route {
    int32_t s, m;
  };
  for (const float base : {1.0f, 1.001f}) {
    const TimeMap time = make_time(base == 1.0f ? 30 : 24, base);
    const long double p = period(time);
    for (const Route route : {Route{1, 0}, Route{4, 2}, Route{8, 3}}) {
      for (const double target_d : {0.25, 0.6, -10.0}) {  // -10: H_b Delta_b = 10 T'
        for (const int32_t frame : {0, 30, 300, 9000, -120}) {
          for (const Readout readout : {Readout::kTopDown, Readout::kBottomUp}) {
            CAPTURE(base);
            CAPTURE(route.s);
            CAPTURE(target_d);
            CAPTURE(frame);
            const int64_t t1 = 4'000'003;
            const long double x = t1 * 65536.0L / 65025.0L;
            const int32_t rows_eff = kH + 2 * route.m;  // H_b Delta_b = (H + 2m) Delta
            const long double r_b = target_d > 0 ? x * (1 - target_d) / target_d : 10.0L * t1;
            const auto delta = static_cast<int64_t>(std::llround(r_b / rows_eff));
            const RowTiming timing{t1, delta, kH, Shutter::kRolling, readout};
            const Raster raster = blender_raster(timing, {0, -1}, route.s, route.m);
            const int64_t soe = time.to_ns(static_cast<float>(frame)) - 777;
            const CyclesTiming ct = cycles_timing(time, timing, raster, soe, t1);
            REQUIRE(ct.rolling);
            CHECK(ct.motion_blur);
            if (target_d > 0) CHECK(std::fabs(static_cast<double>(ct.rolling_duration) - target_d) < 0.01);
            if (target_d < 0) CHECK(ct.rolling_duration < 0.1f);
            const long double hd = static_cast<long double>(raster.rows) * raster.row_dt_ns;
            CHECK(ct.shutter_frames == static_cast<float>((x + hd) / p));
            CHECK(ct.rolling_duration ==
                  static_cast<float>(x / (static_cast<long double>(ct.shutter_frames) * p)));
            // Realized centre = to_ns(c) - round(delta_c f32(d) f32(S) P), exactly.
            CHECK(ct.realized_center_ns ==
                  time.to_ns(ct.expected_c) - exact_delta(time, ct.rolling_duration, ct.shutter_frames));
            // The derivation, in real arithmetic at exact S and d: row i at t_b0 + i Delta_b + T'/2 within 1e-9 frames.
            const long double t_b0 = soe + static_cast<long double>(raster.first_row_offset_ns);
            const long double s_e = (x + hd) / p, d_e = x / (s_e * p);
            const long double t_c = t_b0 + t1 / 2.0L + raster.row_dt_ns * (raster.rows / 2.0L - 1) + kDeltaC * x;
            const long double c_e = (t_c - time.t_ref_ns()) / p + time.frame_ref();
            double worst = 0.0;
            for (int32_t i = 0; i < raster.rows; ++i) {
              const long double cyc = c_e - kDeltaC * d_e * s_e + ((i + 1.0L) / raster.rows - 0.5L) * (1 - d_e) * s_e;
              const long double want = (t_b0 + i * static_cast<long double>(raster.row_dt_ns) + t1 / 2.0L) / p +
                                       time.frame_ref();
              worst = std::max(worst, static_cast<double>(std::fabs(cyc - want)));
            }
            CHECK(worst <= 1e-9);
            // Cycles' forward model at float32 S and d: every raster row lands on the snapped row window's centre.
            const double tol = 1.2e-7 * static_cast<double>(ct.shutter_frames * p) + 2.0;
            double off = 0.0;
            for (int32_t i = 0; i < raster.rows; ++i) {
              const long double want =
                  t_b0 + ct.snap_shift_ns + i * static_cast<long double>(raster.row_dt_ns) + t1 / 2.0L;
              off = std::max(off, static_cast<double>(std::fabs(cycles_row_centre(time, ct, raster.rows, i) - want)));
            }
            CHECK(off <= tol);
            CHECK(std::fabs(static_cast<double>(ct.snap_shift_ns)) <= ct.q_ns + 2.0);
            // The realized row exposure: within 1.2e-7 T' + 0.5 ns of T'.
            CHECK(std::fabs(ct.realized_row_exposure_ns - static_cast<double>(t1)) <=
                  1.2e-7 * static_cast<double>(t1) + 0.5);
          }
        }
      }
    }
  }
}

TEST_CASE("U3 Cycles timing: T' = 0 and row bands under a rolling shutter") {
  const TimeMap time = make_time(30, 1.0f);
  const long double p = period(time);
  constexpr int32_t kH = 60;
  const RowTiming timing{0, 25'000, kH, Shutter::kRolling, Readout::kTopDown};
  for (const int32_t frame : {0, 30, 300, 9000, -120}) {
    CAPTURE(frame);
    const int64_t soe = time.to_ns(static_cast<float>(frame)) + 5;
    const Raster raster = blender_raster(timing);
    const CyclesTiming ct = cycles_timing(time, timing, raster, soe, 0);
    CHECK(ct.rolling);
    CHECK(ct.motion_blur);  // rows are time-shifted only with motion blur on
    CHECK(ct.rolling_duration == 0.0f);
    CHECK(ct.shutter_frames == static_cast<float>(kH * 25'000.0L / p));
    CHECK(ct.realized_center_ns == time.to_ns(ct.expected_c));
    CHECK(ct.realized_row_exposure_ns == 0.0);
    for (int32_t i = 0; i < kH; ++i) {
      const long double want = static_cast<long double>(soe + ct.snap_shift_ns) + i * 25'000.0L;
      CHECK(static_cast<double>(std::fabs(cycles_row_centre(time, ct, kH, i) - want)) <=
            1.2e-7 * static_cast<double>(ct.shutter_frames * p) + 2.0);
    }
  }
  // Bands: each band is its own sub-raster, whose rows land on the frame's rows (read order), here bottom-up.
  const int64_t t1 = 2'000'000;
  const RowTiming bu{t1, 25'000, kH, Shutter::kRolling, Readout::kBottomUp};
  const Schedule sch = Schedule::uniform(bu, time.to_ns(31.0f), 100'000'000, 1, {0, 10'000'000'000});
  for (const RowBand& band : row_bands(kH, 7)) {
    CAPTURE(band.row0);
    const Raster r = blender_raster(bu, band);
    CHECK(r.rows == band.rows);
    const CyclesTiming ct = cycles_timing(time, bu, r, sch.frame_soe(0), t1);
    for (int32_t i = 0; i < band.rows; ++i) {
      const int32_t image_row = kH - 1 - (band.row0 + i);  // read index -> image row (bottom-up)
      const long double want = static_cast<long double>(sch.soe(0, image_row) + ct.snap_shift_ns) + t1 / 2.0L;
      CHECK(static_cast<double>(std::fabs(cycles_row_centre(time, ct, r.rows, i) - want)) <=
            1.2e-7 * static_cast<double>(ct.shutter_frames * p) + 2.0);
    }
  }
  // A rolling shutter without readout (Delta = 0) is a global one.
  const RowTiming flat{t1, 0, kH, Shutter::kRolling, Readout::kTopDown};
  const CyclesTiming g = cycles_timing(time, flat, blender_raster(flat), 1'000'000'000, t1);
  CHECK(!g.rolling);
  CHECK(g.rolling_duration == 1.0f);
  CHECK_THROWS_AS(cycles_timing(time, flat, blender_raster(flat), 0, -1), std::invalid_argument);
  CHECK_THROWS_AS(cycles_timing(time, flat, Raster{0, 0.0, 0.0}, 0, 1), std::invalid_argument);
}

TEST_CASE("U3 Cycles timing: the expected ctime under time remapping (framelen 0.5)") {
  // frame_map_old 50, frame_map_new 100: framelen = f32(50) / f32(100) = 0.5. Blender reports frame_current_final =
  // f32(c framelen), which FR-5's FRM-BLENDER-TIME compares within 1 ulp, so the plan must carry that, not c.
  const TimeMap remap = make_time(30, 1.0f, 1, 20000, 50, 100);
  const TimeMap plain = make_time(30, 1.0f);
  REQUIRE(remap.framelen_bits() == std::bit_cast<uint32_t>(0.5f));
  const RowTiming global{10'000'000, 0, 32, Shutter::kGlobal, Readout::kTopDown};
  const RowTiming rolling{4'000'003, 20'000, 48, Shutter::kRolling, Readout::kBottomUp};
  for (const RowTiming& timing : {global, rolling}) {
    for (const int32_t frame : {30, 301, 9000, -120}) {
      CAPTURE(frame);
      CAPTURE(static_cast<int>(timing.shutter));
      const int64_t soe = remap.to_ns(static_cast<float>(frame)) + 4321;
      const CyclesTiming ct = cycles_timing(remap, timing, blender_raster(timing), soe, timing.exposure_ns);
      CHECK(ct.rolling == (timing.shutter == Shutter::kRolling));
      volatile float c = ct.expected_c;
      volatile float framelen = 0.5f;
      const float want = c * framelen;
      CHECK(std::bit_cast<uint32_t>(ct.expected_ctime) == std::bit_cast<uint32_t>(want));
      CHECK(ct.expected_ctime != ct.expected_c);
      // The remap changes only the reported time: the request, shutter, centre and snap are those without it.
      const CyclesTiming p = cycles_timing(plain, timing, blender_raster(timing), soe, timing.exposure_ns);
      CHECK(p.expected_ctime == p.expected_c);
      CHECK(ct.frame == p.frame);
      CHECK(ct.subframe == p.subframe);
      CHECK(ct.expected_c == p.expected_c);
      CHECK(ct.shutter_frames == p.shutter_frames);
      CHECK(ct.rolling_duration == p.rolling_duration);
      CHECK(ct.realized_center_ns == p.realized_center_ns);
      CHECK(ct.snap_shift_ns == p.snap_shift_ns);
    }
  }
}

TEST_CASE("U3 node_times equal Cycles' float32 expression bitwise") {
  // Count: (2 << (steps - 1)) + 1 positions (E1: steps 1, 2, 3, 5, 7 give 3, 5, 9, 33, 129).
  const int32_t counts[] = {3, 5, 9, 17, 33, 65, 129};
  for (int32_t steps = 1; steps <= 7; ++steps) {
    CHECK(node_times(30.0f, 0.5f, steps).size() == size_t(counts[steps - 1]));
  }
  int64_t checked = 0;
  for (const float frame : {0.0f, 30.0f, 300.0f, 9000.0f, -120.0f, -1.0f}) {
    for (const float sub : {0.0f, 0.3f, 0.71f, 0.999f}) {
      for (const float shutter : {0.1f, 0.5f, 1.0f, 2.5f}) {
        for (int32_t steps = 1; steps <= 7; ++steps) {
          const float c = TimeMap::realized(static_cast<int32_t>(frame), sub);
          const std::vector<float> got = node_times(c, shutter, steps);
          const int32_t n = (2 << (steps - 1)) + 1;
          for (int32_t i = 0; i < n; ++i) {
            // object->motion_time(step) = 2.0f * step / (motion.size() - 1) - 1.0f, then
            // frame_center + subframe_center + frame_center_delta + relative_time * shuttertime * 0.5f.
            volatile float step_f = static_cast<float>(i);
            volatile float twice = 2.0f * step_f;
            volatile float size1 = static_cast<float>(static_cast<size_t>(n) - 1);
            volatile float q = twice / size1;
            volatile float rel = q - 1.0f;
            volatile float a = static_cast<float>(static_cast<int32_t>(frame)) + sub;
            volatile float b = a + 0.0f;
            volatile float rs = rel * shutter;
            volatile float half = rs * 0.5f;
            volatile float t = b + half;
            float want = t;
            if (rel != 0.0f) {  // frame_set(floorf(time), time - frame), realized f32(f32(frame) + subframe)
              const int f = static_cast<int>(std::floor(static_cast<float>(t)));
              volatile float subf = t - static_cast<float>(f);
              volatile float realized = static_cast<float>(f) + subf;
              want = realized;
            }
            CHECK(std::bit_cast<uint32_t>(got[static_cast<size_t>(i)]) == std::bit_cast<uint32_t>(want));
            ++checked;
          }
          // Nested: node i of steps n is node 2i of steps n + 1.
          if (steps < 7) {
            const std::vector<float> finer = node_times(c, shutter, steps + 1);
            for (int32_t i = 0; i < n; ++i) CHECK(finer[2 * static_cast<size_t>(i)] == got[static_cast<size_t>(i)]);
          }
          CHECK(got[static_cast<size_t>(n / 2)] == c);
          CHECK(std::is_sorted(got.begin(), got.end()));
        }
      }
    }
  }
  CHECK(checked > 10000);
  // Near frame 0 from below, frame_set realizes f32(-1 + f32(time + 1)), not time itself.
  const std::vector<float> neg = node_times(-0.3f, 0.5f, 1);
  volatile float c_neg = -0.3f;
  volatile float t_neg = c_neg + -0.25f;
  volatile float sub_neg = t_neg - -1.0f;
  CHECK(neg[0] == TimeMap::realized(-1, sub_neg));
  CHECK_THROWS_AS(node_times(0.0f, 0.5f, 0), std::invalid_argument);
  CHECK_THROWS_AS(node_times(0.0f, 0.5f, 8), std::invalid_argument);
  CHECK_THROWS_AS(node_times(0.0f, -0.5f, 1), std::invalid_argument);
}

namespace {

constexpr double kF = 500.0;

// A static pinhole camera at the origin looking along +z (640 x 480, f = 500 px) and one body, both over
// [t_begin, t_end]; trajectory 0 is the rig, 1 the body's.
render::SceneBundle motion_scene(int64_t t_begin, int64_t t_end, const render::Trajectory& body_traj,
                                 const render::Trajectory* rig = nullptr,
                                 const render::Mesh& mesh = box_mesh(0.2f, 0.2f, 0.01f)) {
  render::SceneBundle s;
  s.t_start_ns = t_begin;
  s.t_end_ns = t_end;
  s.light_groups = {"default"};
  s.trajectories = {rig != nullptr ? *rig : static_trajectory(t_begin, t_end, quat_identity<double>(), {0, 0, 0}),
                    body_traj};
  s.cameras.push_back(pinhole(640, 480, kF, 0));
  render::Body b = make_body(1, mesh);
  b.trajectory = 1;
  s.bodies.push_back(b);
  return s;
}

struct Pass {
  float c, s;
  int64_t a, b;  // the shutter window: the outer node times
};
Pass pass_at(const TimeMap& time, float frame, int64_t exposure) {
  const RowTiming g{exposure, 0, 480, Shutter::kGlobal, Readout::kTopDown};
  const CyclesTiming ct = cycles_timing(time, g, blender_raster(g), time.to_ns(frame) - exposure / 2, exposure);
  const std::vector<float> nodes = node_times(ct.expected_c, ct.shutter_frames, 1);
  return {ct.expected_c, ct.shutter_frames, time.to_ns(nodes.front()), time.to_ns(nodes.back())};
}

// A trajectory's pose at t (the K = 6 spline, as the renderer evaluates it).
traj::Kinematics<double> pose_at(const render::Trajectory& tr, int64_t t) {
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
  return traj::evaluate<6>(v, st);
}

// The probe times of interval [ta, tb]: ta + floor(k (tb - ta) / 16), k = 0..15.
std::vector<int64_t> probes(int64_t ta, int64_t tb) {
  std::vector<int64_t> out;
  for (int k = 0; k < 16; ++k) out.push_back(ta + static_cast<int64_t>(static_cast<i128>(tb - ta) * k / 16));
  return out;
}

}  // namespace

TEST_CASE("U3 motion steps: translation, spin (slerp and nlerp), pivot, orbit, rig, time resolution") {
  const TimeMap time = make_time(30, 1.0f);
  const int64_t t_mid = time.to_ns(31.0f);
  const int64_t t_begin = t_mid - 100'000'000, t_end = t_mid + 100'000'000;
  const Pass ps = pass_at(time, 31.0f, 20'000'000);
  const auto q_id = quat_identity<double>();

  SUBCASE("linear translation is exact for every model") {
    const render::SceneBundle b =
        motion_scene(t_begin, t_end,
                     moving_trajectory(t_begin, t_end, q_id, {-0.5, 0.1, 2.0}, {5.0, 0.0, 0.0}, {0, 0, 0}));
    const MotionProbe pr = motion_probe(b, 0, 0, {}, t_mid);
    CHECK(pr.points.size() == 8);
    for (const QuatInterp m : {QuatInterp::kCpu, QuatInterp::kGpuRt, QuatInterp::kWorst}) {
      CHECK(motion_chord_px(pr, ps.a, ps.b, ps.c, ps.s, time, 1, m) < 1e-7);
      MotionStepsInfo info;
      CHECK(motion_steps(pr, ps.a, ps.b, ps.c, ps.s, time, 0.05, m, &info) == 1);
      CHECK(info.speed_px_s == doctest::Approx(kF * 5.0 / 1.99).epsilon(1e-3));  // the near corners, z = 1.99 m
      CHECK(info.warnings.empty());
    }
  }

  SUBCASE("spin about the body origin: slerp is exact, nlerp matches its closed form, kWorst takes the larger") {
    const double omega = 100.0;  // rad/s about the optical axis, through (0, 0, 2)
    const render::SceneBundle b =
        motion_scene(t_begin, t_end,
                     moving_trajectory(t_begin, t_end, q_id, {0.0, 0.0, 2.0}, {0, 0, 0}, {0, 0, omega}));
    const MotionProbe pr = motion_probe(b, 0, 0, {}, t_mid);
    const double cpu = motion_chord_px(pr, ps.a, ps.b, ps.c, ps.s, time, 1, QuatInterp::kCpu);
    const double gpu = motion_chord_px(pr, ps.a, ps.b, ps.c, ps.s, time, 1, QuatInterp::kGpuRt);
    const double worst = motion_chord_px(pr, ps.a, ps.b, ps.c, ps.s, time, 1, QuatInterp::kWorst);
    CHECK(cpu < 1e-7);  // per interval 1 rad: dot = cos(0.5) < 0.9995, so slerp
    CHECK(worst == gpu);
    // nlerp of q(theta) to q(theta + phi): angle 2 atan2(u sin(phi/2), 1 - u + u cos(phi/2)); a corner at radius
    // rho and depth z moves by 2 rho sin(|delta| / 2) f / z px.
    const std::vector<float> nodes = node_times(ps.c, ps.s, 1);
    double want = 0.0;
    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
      const int64_t ta = time.to_ns(nodes[i]), tb = time.to_ns(nodes[i + 1]);
      const double phi = omega * 1e-9 * static_cast<double>(tb - ta);
      for (const int64_t t : probes(ta, tb)) {
        const double u = static_cast<double>(t - ta) / static_cast<double>(tb - ta);
        const double delta = 2.0 * std::atan2(u * std::sin(phi / 2), 1 - u + u * std::cos(phi / 2)) - u * phi;
        const double rho = std::hypot(0.2, 0.2), z = 2.0 - 0.01;
        want = std::max(want, 2.0 * rho * std::sin(std::fabs(delta) / 2) * kF / z);
      }
    }
    CHECK(want > 0.05);
    CHECK(gpu == doctest::Approx(want).epsilon(1e-6));
    CHECK(motion_steps(pr, ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kCpu) == 1);
    const int32_t n_gpu = motion_steps(pr, ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kGpuRt);
    CHECK(n_gpu > 1);
    CHECK(motion_steps(pr, ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kWorst) == n_gpu);

    // A pivot 0.5 m off the body origin: Cycles interpolates the pivot's circle linearly, so slerp is no longer
    // exact; the chord at u = 1/2 is r (1 - cos(phi/2)), the same for every corner.
    MotionProbe off = pr;
    off.pivot = {0.5, 0.0, 0.0};
    const double cpu_off = motion_chord_px(off, ps.a, ps.b, ps.c, ps.s, time, 1, QuatInterp::kCpu);
    double chord = 0.0;
    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
      const int64_t ta = time.to_ns(nodes[i]), tb = time.to_ns(nodes[i + 1]);
      const double phi = omega * 1e-9 * static_cast<double>(tb - ta);
      for (const int64_t t : probes(ta, tb)) {
        const double u = static_cast<double>(t - ta) / static_cast<double>(tb - ta);
        // |(1 - u) e^{i a} + u e^{i b} - e^{i (a + u phi)}| r for a point on the circle
        const double re = (1 - u) + u * std::cos(phi) - std::cos(u * phi), im = u * std::sin(phi) - std::sin(u * phi);
        chord = std::max(chord, 0.5 * std::hypot(re, im) * kF / (2.0 - 0.01));
      }
    }
    CHECK(cpu_off == doctest::Approx(chord).epsilon(1e-6));
  }

  SUBCASE("orbit: the smallest steps that meet the tolerance; chord falls about 4x per step") {
    const double r = 0.3, w = 20.0;
    const render::Trajectory orbit = make_trajectory(t_begin, t_end, [&](double s, Quat<double>* q, Vec3<double>* p) {
      *q = quat_identity<double>();
      *p = {r * std::cos(w * s), r * std::sin(w * s), 2.0};
    });
    const render::SceneBundle b = motion_scene(t_begin, t_end, orbit);
    const MotionProbe pr = motion_probe(b, 0, 0, {}, t_mid);
    std::vector<double> chord;
    for (int32_t n = 1; n <= 7; ++n) {
      chord.push_back(motion_chord_px(pr, ps.a, ps.b, ps.c, ps.s, time, n, QuatInterp::kWorst));
    }
    for (int32_t n = 3; n <= 7; ++n) {
      CHECK(chord[size_t(n - 2)] / chord[size_t(n - 1)] == doctest::Approx(4.0).epsilon(0.1));
    }
    for (const double tol : {0.5, 0.05, 0.005}) {
      MotionStepsInfo info;
      const int32_t n = motion_steps(pr, ps.a, ps.b, ps.c, ps.s, time, tol, QuatInterp::kWorst, &info);
      REQUIRE(n >= 1);
      CHECK(info.time_res_px <= tol / 2);  // frame 31: the budget is tol less the time-resolution term
      CHECK(info.budget_px == tol - info.time_res_px);
      CHECK(info.chord_px == chord[size_t(n - 1)]);
      CHECK(info.chord_px + info.time_res_px <= tol);
      if (n > 1) CHECK(chord[size_t(n - 2)] + info.time_res_px > tol);
    }
    CHECK(motion_steps(pr, ps.a, ps.b, ps.c, ps.s, time, 1e-6, QuatInterp::kWorst) == 0);  // 7 is not enough
    // Probes only inside [t_a, t_b]: an empty window sees no chord.
    CHECK(motion_chord_px(pr, ps.a - 10, ps.a - 1, ps.c, ps.s, time, 1, QuatInterp::kWorst) == 0.0);
  }

  SUBCASE("the rig probe: image corners and centre at z_near; a panning camera over a static world") {
    const render::Trajectory pan =
        moving_trajectory(t_begin, t_end, q_id, {0, 0, 0}, {0, 0, 0}, {0.0, 40.0, 0.0});  // 40 rad/s about y
    const render::SceneBundle b =
        motion_scene(t_begin, t_end, static_trajectory(t_begin, t_end, q_id, {0, 0, 5}), &pan);
    std::vector<float> depth(16, std::numeric_limits<float>::quiet_NaN());
    depth[3] = 2.5f;
    depth[7] = 1.5f;
    const MotionProbe rig = motion_probe(b, 0, -1, depth, t_mid);
    CHECK(rig.z_near_m == 1.5);
    REQUIRE(rig.points.size() == 5);
    // The points are world points that the camera at t_mid sees at the image corners and centre, 1.5 m deep.
    const traj::Kinematics<double> k = pose_at(b.trajectories[0], t_mid);
    const Vec2<double> want[5] = {{0, 0}, {639, 0}, {0, 479}, {639, 479}, {319.5, 239.5}};
    for (size_t i = 0; i < 5; ++i) {
      const Vec3<double> xc =
          rotate_inverse(k.q, Vec3<double>{rig.points[i][0], rig.points[i][1], rig.points[i][2]} - k.p);
      CHECK(xc.z == doctest::Approx(1.5).epsilon(1e-12));
      CHECK(std::fabs(kF * xc.x / xc.z + 319.5 - want[i].x) < 1e-9);
      CHECK(std::fabs(kF * xc.y / xc.z + 239.5 - want[i].y) < 1e-9);
    }
    // The probe's poses come from the spline built over the call's segments only, bitwise those of the whole spline:
    // the centre pixel's ray is the optical axis, so its point is R (0, 0, z_near) + p of the whole spline's pose.
    const Vec3<double> centre = rotate(k.q, Vec3<double>{0.0, 0.0, 1.5}) + k.p;
    CHECK(rig.points[4][0] == centre.x);
    CHECK(rig.points[4][1] == centre.y);
    CHECK(rig.points[4][2] == centre.z);
    const MotionProbe none = motion_probe(b, 0, -1, std::vector<float>(4, NAN), t_mid);
    CHECK(none.points.empty());
    CHECK(std::isnan(none.z_near_m));
    CHECK(motion_steps(none, ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kWorst) == 1);
    const double cpu = motion_chord_px(rig, ps.a, ps.b, ps.c, ps.s, time, 1, QuatInterp::kCpu);
    const double gpu = motion_chord_px(rig, ps.a, ps.b, ps.c, ps.s, time, 1, QuatInterp::kGpuRt);
    CHECK(cpu < 1e-6);  // slerp about the camera centre is the true pan
    CHECK(gpu > 0.05);
    CHECK(motion_steps(rig, ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kCpu) == 1);
    CHECK(motion_steps(rig, ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kWorst) > 1);
  }

  SUBCASE("Blender's time resolution: FRM-TIME-RES at frame 2^20, none at frame 31") {
    const TimeMap t24 = make_time(24, 1.0f, 1, 1 << 21);
    const int64_t mid = t24.to_ns(static_cast<float>(1 << 20));
    const render::SceneBundle b = motion_scene(
        mid - 300'000'000, mid + 300'000'000,
        moving_trajectory(mid - 300'000'000, mid + 300'000'000, q_id, {0.0, 0.0, 2.0}, {2.0, 0, 0}, {0, 0, 0}));
    const MotionProbe pr = motion_probe(b, 0, 0, {}, mid);
    const Pass far = pass_at(t24, static_cast<float>(1 << 20), 20'000'000);
    MotionStepsInfo info;
    // The time-resolution term (> 1 px) exceeds tol at every n and in every sub-render, so it does not ask for more
    // sub-renders (0): FRM-TIME-RES reports it and the chord, exact for a linear translation, keeps tol / 2.
    CHECK(motion_steps(pr, far.a, far.b, far.c, far.s, t24, 0.05, QuatInterp::kWorst, &info) == 1);
    CHECK(info.time_res_px ==
          doctest::Approx(info.speed_px_s * 2.0 * ulp32(far.c + 0.5f * far.s) * t24.period_ns() * 1e-9).epsilon(1e-12));
    CHECK(info.time_res_px > 1.0);
    CHECK(info.budget_px == 0.025);
    CHECK(info.chord_px <= info.budget_px);
    REQUIRE(info.warnings.size() == 1);
    CHECK(info.warnings[0].code == "FRM-TIME-RES");
    const render::SceneBundle near = motion_scene(
        t_begin, t_end, moving_trajectory(t_begin, t_end, q_id, {0.0, 0.0, 2.0}, {2.0, 0, 0}, {0, 0, 0}));
    MotionStepsInfo near_info;
    CHECK(motion_steps(motion_probe(near, 0, 0, {}, t_mid), ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kWorst,
                       &near_info) == 1);
    CHECK(near_info.warnings.empty());
    CHECK(near_info.time_res_px < 1e-3);
  }

  SUBCASE("frame 9000: the time-resolution floor takes at most tol / 2 and never asks for more sub-renders") {
    // 30 fps, frame 9000 (5 min): ulp32 = 2^-10 frames, so 2 ulp32 P = 65.1 us. A plate translating at about
    // 800 px/s puts the term just above tol = 0.05 px: the old rule (chord + term <= tol) failed at every n and in
    // every sub-render and doubled n_sub to 64.
    const int64_t mid = time.to_ns(9000.0f);
    const int64_t b0 = mid - 100'000'000, b1 = mid + 100'000'000;
    const Pass p9 = pass_at(time, 9000.0f, 20'000'000);
    const render::SceneBundle slide =
        motion_scene(b0, b1, moving_trajectory(b0, b1, q_id, {-0.368, 0.0, 2.0}, {3.2, 0.0, 0.0}, {0, 0, 0}));
    MotionStepsInfo info;
    CHECK(motion_steps(motion_probe(slide, 0, 0, {}, mid), p9.a, p9.b, p9.c, p9.s, time, 0.05, QuatInterp::kWorst,
                       &info) == 1);
    CHECK(info.speed_px_s == doctest::Approx(kF * 3.2 / 1.99).epsilon(1e-3));
    CHECK(info.time_res_px > 0.05);
    CHECK(info.budget_px == 0.025);
    REQUIRE(info.warnings.size() == 1);
    CHECK(info.warnings[0].code == "FRM-TIME-RES");
    // An orbit there: the smallest n with chord(n) <= max(tol - term, tol / 2), in both regimes of the budget.
    const double r = 0.3, w = 20.0;
    const render::SceneBundle orbit = motion_scene(b0, b1, make_trajectory(b0, b1, [&](double s, Quat<double>* q,
                                                                                         Vec3<double>* p) {
      *q = quat_identity<double>();
      *p = {r * std::cos(w * s), r * std::sin(w * s), 2.0};
    }));
    const MotionProbe pr = motion_probe(orbit, 0, 0, {}, mid);
    std::vector<double> chord;
    for (int32_t n = 1; n <= 7; ++n) {
      chord.push_back(motion_chord_px(pr, p9.a, p9.b, p9.c, p9.s, time, n, QuatInterp::kWorst));
    }
    for (const double tol : {0.5, 0.05}) {
      CAPTURE(tol);
      MotionStepsInfo oi;
      const int32_t n = motion_steps(pr, p9.a, p9.b, p9.c, p9.s, time, tol, QuatInterp::kWorst, &oi);
      REQUIRE(n >= 1);
      CHECK(oi.budget_px == std::max(tol - oi.time_res_px, tol / 2));
      CHECK((oi.time_res_px > tol / 2) == (tol == 0.05));
      CHECK(oi.chord_px == chord[size_t(n - 1)]);
      CHECK(oi.chord_px <= oi.budget_px);
      if (n > 1) CHECK(chord[size_t(n - 2)] > oi.budget_px);
    }
    // 0 still means that 7 steps miss the chord's budget, which more sub-renders (closer nodes) can lower.
    CHECK(motion_steps(pr, p9.a, p9.b, p9.c, p9.s, time, 1e-6, QuatInterp::kWorst) == 0);
  }

  SUBCASE("a body whose AABB corners all leave the image still measures its visible interior") {
    // A 2 m x 2 m plate 0.5 m in front of the camera covers the image; its corners (normalized radius >= 1.86) lie
    // outside the pinhole's injective domain (r_lim <= 1.2 r_img = 0.96). It orbits r = 5 cm at 20 rad/s in the
    // image plane, so Cycles' linear node interpolation misses the visible interior by r (1 - cos(phi / 2)) per
    // node interval: about 0.25 px at 1 step. A 10 cm plate (corners inside the domain) on the same orbit measures
    // that exactly, and the bound f_max |dX| / z of the big plate's corners equals it (pure translation at fixed z).
    REQUIRE((1.0 - 0.05) / 0.51 > 1.2 * std::hypot(320.0, 240.0) / kF);
    const double r = 0.05, w = 20.0;
    const auto orbit_at = [&](double z) {
      return make_trajectory(t_begin, t_end, [&, z](double s, Quat<double>* q, Vec3<double>* p) {
        *q = quat_identity<double>();
        *p = {r * std::cos(w * s), r * std::sin(w * s), z};
      });
    };
    const render::Trajectory front = orbit_at(0.5);
    const render::SceneBundle big = motion_scene(t_begin, t_end, front, nullptr, box_mesh(1.0f, 1.0f, 0.01f));
    const render::SceneBundle small = motion_scene(t_begin, t_end, front, nullptr, box_mesh(0.05f, 0.05f, 0.01f));
    const MotionProbe pb = motion_probe(big, 0, 0, {}, t_mid), psm = motion_probe(small, 0, 0, {}, t_mid);
    for (int32_t n = 1; n <= 3; ++n) {
      CAPTURE(n);
      const double cb = motion_chord_px(pb, ps.a, ps.b, ps.c, ps.s, time, n, QuatInterp::kWorst);
      const double cs = motion_chord_px(psm, ps.a, ps.b, ps.c, ps.s, time, n, QuatInterp::kWorst);
      CHECK(cs > 0.0);
      CHECK(cb == doctest::Approx(cs).epsilon(1e-9));
      if (n == 1) CHECK(cb > 0.2);
    }
    MotionStepsInfo ib, is;
    const int32_t nb = motion_steps(pb, ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kWorst, &ib);
    CHECK(nb > 1);
    CHECK(nb == motion_steps(psm, ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kWorst, &is));
    CHECK(ib.speed_px_s == doctest::Approx(is.speed_px_s).epsilon(1e-9));
    CHECK(ib.speed_px_s == doctest::Approx(kF * r * w / 0.49).epsilon(1e-2));
    // z_near (the ROI's nearest centre-ray depth) floors the bound's depth: visible points lie no nearer.
    std::vector<float> depth(8, std::numeric_limits<float>::quiet_NaN());
    depth[5] = 0.6f;
    const MotionProbe pz = motion_probe(big, 0, 0, depth, t_mid);
    CHECK(pz.z_near_m == static_cast<double>(0.6f));
    CHECK(motion_chord_px(pz, ps.a, ps.b, ps.c, ps.s, time, 1, QuatInterp::kWorst) ==
          doctest::Approx(motion_chord_px(psm, ps.a, ps.b, ps.c, ps.s, time, 1, QuatInterp::kWorst) * 0.49 /
                          pz.z_near_m)
              .epsilon(1e-9));
    // Wholly behind the camera: nothing is visible, no chord and no speed.
    const render::SceneBundle behind =
        motion_scene(t_begin, t_end, orbit_at(-0.5), nullptr, box_mesh(1.0f, 1.0f, 0.01f));
    const MotionProbe pbh = motion_probe(behind, 0, 0, {}, t_mid);
    CHECK(motion_chord_px(pbh, ps.a, ps.b, ps.c, ps.s, time, 1, QuatInterp::kWorst) == 0.0);
    MotionStepsInfo ih;
    CHECK(motion_steps(pbh, ps.a, ps.b, ps.c, ps.s, time, 0.05, QuatInterp::kWorst, &ih) == 1);
    CHECK(ih.speed_px_s == 0.0);
  }

  SUBCASE("no shutter, no points, bad arguments, poses outside the trajectories") {
    const render::SceneBundle b =
        motion_scene(t_begin, t_end, moving_trajectory(t_begin, t_end, q_id, {0, 0, 2.0}, {1.0, 0, 0}, {0, 0, 0}));
    const MotionProbe pr = motion_probe(b, 0, 0, {}, t_mid);
    CHECK(motion_steps(pr, ps.a, ps.b, ps.c, 0.0f, time, 0.05, QuatInterp::kWorst) == 1);
    CHECK_THROWS_AS(motion_steps(pr, ps.a, ps.b, ps.c, ps.s, time, 0.0, QuatInterp::kWorst), std::invalid_argument);
    CHECK_THROWS_AS(motion_steps(pr, ps.b, ps.a, ps.c, ps.s, time, 0.05, QuatInterp::kWorst), std::invalid_argument);
    CHECK_THROWS_AS(motion_probe(b, 1, 0, {}, t_mid), std::invalid_argument);
    CHECK_THROWS_AS(motion_probe(b, 0, 1, {}, t_mid), std::invalid_argument);
    const Pass late = pass_at(time, 40.0f, 20'000'000);  // 300 ms after the trajectories end
    try {
      motion_steps(pr, late.a, late.b, late.c, late.s, time, 0.05, QuatInterp::kWorst);
      FAIL("expected FRM-SPAN");
    } catch (const FramesError& e) {
      CHECK(e.code == "FRM-SPAN");
    }
    // A body span that runs past its spline's own end (rig valid there): the pose there is FRM-SPAN as well.
    render::Trajectory longer = moving_trajectory(t_begin, t_end, q_id, {0, 0, 2.0}, {1.0, 0, 0}, {0, 0, 0});
    longer.span_end_ns = t_end + 1'000'000'000;
    const render::Trajectory rig_long = static_trajectory(t_begin, t_end + 1'000'000'000, q_id, {0, 0, 0});
    const render::SceneBundle past = motion_scene(t_begin, t_end, longer, &rig_long);
    try {
      motion_steps(motion_probe(past, 0, 0, {}, t_mid), late.a, late.b, late.c, late.s, time, 0.05,
                   QuatInterp::kWorst);
      FAIL("expected FRM-SPAN");
    } catch (const FramesError& e) {
      CHECK(e.code == "FRM-SPAN");
    }
    // A window over the whole int64 range: the speed steps are taken in 128 bits, and the first pose refuses.
    try {
      motion_steps(pr, INT64_MIN, INT64_MAX, ps.c, ps.s, time, 0.05, QuatInterp::kWorst);
      FAIL("expected FRM-SPAN");
    } catch (const FramesError& e) {
      CHECK(e.code == "FRM-SPAN");
    }
  }
}

TEST_CASE("U3 sub-render and row-band rules") {
  const RowTiming global{10'000'000, 0, 480, Shutter::kGlobal, Readout::kTopDown};
  const RowTiming rolling{10'000'000, 20'000, 480, Shutter::kRolling, Readout::kTopDown};
  const LightSplit none = light_split(0.0, rolling, 0.005);
  CHECK(none.n_sub == 1);
  CHECK(none.n_band == 1);
  CHECK(none.bound == 0.0);
  CHECK(none.warnings.empty());
  // Global: n_sub = ceil(rho T / tol) = ceil(10 * 0.01 / 0.005) = 20.
  const LightSplit g = light_split(10.0, global, 0.005);
  CHECK(g.n_sub == 20);
  CHECK(g.n_band == 1);
  CHECK(g.bound <= 0.005 + 1e-15);
  CHECK(g.warnings.empty());
  // Rolling: n_sub = ceil(2 rho T / tol) = 40, n_band = ceil(2 rho R / tol) = ceil(38.32) = 39; the longest pass
  // (T/40 + 12 rows of 20 us) keeps rho (T'_j + R_band) <= tol.
  const LightSplit r = light_split(10.0, rolling, 0.005);
  CHECK(r.n_sub == 40);
  CHECK(r.n_band == 39);
  CHECK(r.bound == doctest::Approx(10.0 * (250e-6 + 12 * 20e-6)).epsilon(1e-12));
  CHECK(r.bound <= 0.005);
  CHECK(r.warnings.empty());
  // Caps: n_sub 64, n_band min(H, 64); FRM-LIGHT-MOTION with the bound and its T and R terms.
  const LightSplit cap = light_split(1000.0, rolling, 0.005);
  CHECK(cap.n_sub == 64);
  CHECK(cap.n_band == 64);
  REQUIRE(cap.warnings.size() == 1);
  CHECK(cap.warnings[0].code == "FRM-LIGHT-MOTION");
  CHECK(cap.bound > 0.005);
  CHECK(cap.warnings[0].message.find("R term") != std::string::npos);
  const RowTiming short_roll{10'000'000, 20'000, 16, Shutter::kRolling, Readout::kTopDown};
  CHECK(light_split(1000.0, short_roll, 0.005).n_band == 16);  // 120 bands wanted, H = 16 rows
  CHECK_THROWS_AS(light_split(-1.0, global, 0.005), std::invalid_argument);
  CHECK_THROWS_AS(light_split(1.0, global, 0.0), std::invalid_argument);

  // n_sub doubles (cap 64) until motion steps and the factorization test pass.
  std::vector<int32_t> seen;
  const SubRenderChoice a = choose_n_sub(
      3, 0.005, [&](int32_t n) { seen.push_back(n); return n >= 10; }, [](int32_t n) { return 0.04 / n; });
  CHECK(a.n_sub == 12);
  CHECK(seen == std::vector<int32_t>{3, 6, 12});
  CHECK(a.fact_p99 == doctest::Approx(0.04 / 12));
  CHECK(a.warnings.empty());
  const SubRenderChoice f = choose_n_sub(1, 0.005, {}, [](int32_t n) { return 0.1 / n; });
  CHECK(f.n_sub == 32);
  CHECK(choose_n_sub(5, 0.005, {}, {}).n_sub == 5);
  const SubRenderChoice fail = choose_n_sub(48, 0.005, [](int32_t) { return false; }, [](int32_t) { return 1.0; });
  CHECK(fail.n_sub == 64);
  REQUIRE(fail.warnings.size() == 2);
  CHECK(fail.warnings[0].code == "FRM-MOTION");
  CHECK(fail.warnings[1].code == "FRM-FACTOR");
  CHECK_THROWS_AS(choose_n_sub(0, 0.005, {}, {}), std::invalid_argument);
  CHECK_THROWS_AS(choose_n_sub(65, 0.005, {}, {}), std::invalid_argument);
}
