// U2 (docs/design/phase3/frames.md §8.1): frame schedules (§4.1). Both timelines and readouts; the blender
// positions START/CENTER/END reproduce Blender's exposure windows (BL §4.2); exposure null gives sigma P or 0 per
// motion_blur.use; end frames outside the bundle span are dropped and listed, none left is refused (FRM-SPAN);
// sensor_time equals reported_time without delay and quantum.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/time/time.h"
#include "frames/timing.h"
#include "sensor/readout.h"

namespace {

using namespace balla::frames;

__extension__ typedef __int128 i128;

constexpr int64_t kW = balla::time::kWindowNs;

// BL §4.2: t_ns(F) = round_half_even((F - F_ref) 1e9 base_num / (fps base_den)), t_ref = 0.
int64_t t_ns_frame(int64_t frame, int64_t frame_ref, const FrameRate& r) {
  const i128 num = i128{frame - frame_ref} * 1'000'000'000 * r.base_num;
  const i128 den = i128{r.fps} * r.base_den;
  i128 q = num / den, rem = num % den;
  if (rem < 0) {
    --q;
    rem += den;
  }
  if (2 * rem > den || (2 * rem == den && (q & 1) != 0)) ++q;
  return static_cast<int64_t>(q);
}

// sigma P in long double, for the window checks.
long double sigma_p(float sigma, const FrameRate& r) {
  return static_cast<long double>(sigma) * 1e9L * static_cast<long double>(r.base_num) /
         (static_cast<long double>(r.fps) * static_cast<long double>(r.base_den));
}

std::vector<int64_t> blender_frames(int64_t start, int64_t end, int64_t stride, const FrameRate& r) {
  std::vector<int64_t> t;
  for (int64_t f = start; f <= end; f += stride) t.push_back(t_ns_frame(f, start, r));
  return t;
}

bool has_warning(const Schedule& s, const std::string& code) {
  for (const Warning& w : s.warnings()) {
    if (w.code == code) return true;
  }
  return false;
}

}  // namespace

TEST_CASE("U2 uniform timeline: global and rolling, both readouts") {
  const Span support{-2 * kW, 2'000'000'000};
  const int64_t t0 = 1000, period = 33'333'333, exposure = 10'000'000, dt = 20'000;
  for (Readout readout : {Readout::kTopDown, Readout::kBottomUp}) {
    const RowTiming rolling{exposure, dt, 480, Shutter::kRolling, readout};
    const Schedule s = Schedule::uniform(rolling, t0, period, 5, support);
    REQUIRE(s.size() == 5);
    CHECK(s.dropped().empty());
    CHECK(s.warnings().empty());
    CHECK(s.readout_span_ns() == 479 * dt);
    for (int32_t k = 0; k < 5; ++k) {
      const int64_t a = t0 + k * period;
      CHECK(s.candidate(k) == k);
      CHECK(s.frame_soe(k) == a);
      CHECK(s.frame_eoe(k) == a + 479 * dt + exposure);
      CHECK(s.frame_mid(k) == a + (479 * dt + exposure) / 2);
      for (int32_t r : {0, 1, 17, 240, 479}) {
        const int32_t rho = readout == Readout::kTopDown ? r : 479 - r;
        CHECK(s.read_order(r) == rho);
        CHECK(s.soe(k, r) == a + rho * dt);
        CHECK(s.eoe(k, r) == a + rho * dt + exposure);
      }
    }
    // The first-read row starts the frame, the last-read row ends it.
    const int32_t first = readout == Readout::kTopDown ? 0 : 479, last = 479 - first;
    CHECK(s.soe(2, first) == s.frame_soe(2));
    CHECK(s.eoe(2, last) == s.frame_eoe(2));

    // A global shutter ignores the row delay: every row shares the window.
    const RowTiming global{exposure, dt, 480, Shutter::kGlobal, readout};
    const Schedule g = Schedule::uniform(global, t0, period, 3, support);
    CHECK(g.readout_span_ns() == 0);
    for (int32_t r : {0, 239, 479}) {
      CHECK(g.soe(1, r) == t0 + period);
      CHECK(g.eoe(1, r) == t0 + period + exposure);
    }
  }
}

TEST_CASE("U2 uniform timeline: count null fills the support, frames before it are dropped and listed") {
  const RowTiming timing{5'000'000, 0, 260, Shutter::kGlobal, Readout::kTopDown};
  const int64_t period = 25'000'000;
  // t0 two periods before the support: frames 0 and 1 start before it.
  const Schedule s = Schedule::uniform(timing, -2 * period, period, std::nullopt, Span{0, 1'000'000'000});
  // k >= 0 with -2P + k P + T <= 1e9: k <= 41.
  REQUIRE(s.size() == 40);
  REQUIRE(s.dropped().size() == 2);
  CHECK(s.dropped()[0] == 0);
  CHECK(s.dropped()[1] == 1);
  CHECK(s.candidate(0) == 2);
  CHECK(s.frame_soe(0) == 0);
  CHECK(s.frame_eoe(39) <= 1'000'000'000);
  CHECK(has_warning(s, "FRM-SPAN"));

  // An explicit count past the support drops the frames that end after it.
  const Schedule c = Schedule::uniform(timing, 0, period, 50, Span{0, 1'000'000'000});
  CHECK(c.size() == 40);
  CHECK(c.dropped().size() == 10);
  CHECK(c.dropped()[0] == 40);
}

TEST_CASE("U2 blender timeline: positions reproduce Blender's exposure windows") {
  const FrameRate rates[] = {{24, 1, 1}, {30, 1001, 1000}, {25, 1, 1}};
  for (const FrameRate& rate : rates) {
    for (float sigma : {0.5f, 0.3f, 1.0f}) {
      for (Position pos : {Position::kStart, Position::kCenter, Position::kEnd}) {
        const BlenderShutter shutter{true, sigma, pos};
        const Exposure e = resolve_exposure(std::nullopt, &shutter, &rate);
        CHECK(e.source == ExposureSource::kBlenderShutter);
        const long double sp = sigma_p(sigma, rate);
        CHECK(std::abs(static_cast<long double>(e.ns) - sp) <= 0.5L);
        const std::vector<int64_t> frames = blender_frames(1, 250, 3, rate);
        const RowTiming timing{e.ns, 0, 480, Shutter::kGlobal, Readout::kTopDown};
        const Schedule s = Schedule::blender(timing, frames, pos, Span{-10 * kW, frames.back() + 10 * kW});
        REQUIRE(s.size() == static_cast<int32_t>(frames.size()));
        for (int32_t k = 0; k < s.size(); ++k) {
          const long double t = static_cast<long double>(frames[static_cast<size_t>(k)]);
          // BL §4.2: CENTER [t - sigma P/2, t + sigma P/2], START [t, t + sigma P], END [t - sigma P, t].
          const long double lo = pos == Position::kStart ? t : (pos == Position::kEnd ? t - sp : t - sp / 2);
          CHECK(std::abs(static_cast<long double>(s.frame_soe(k)) - lo) <= 1.0L);
          CHECK(std::abs(static_cast<long double>(s.frame_eoe(k)) - (lo + sp)) <= 1.0L);
          if (pos == Position::kStart) CHECK(s.frame_soe(k) == frames[static_cast<size_t>(k)]);
          if (pos == Position::kEnd) CHECK(s.frame_eoe(k) == frames[static_cast<size_t>(k)]);
        }
      }
    }
  }
  // Rolling: the readout span R joins the exposure, so the frame's first-row start to last-row end is centred.
  const FrameRate rate{30, 1, 1};
  const std::vector<int64_t> frames = blender_frames(1, 30, 1, rate);
  const RowTiming rolling{8'000'001, 15'000, 480, Shutter::kRolling, Readout::kBottomUp};
  const Schedule s = Schedule::blender(rolling, frames, Position::kCenter, Span{-2 * kW, frames.back() + kW});
  const int64_t length = 479 * 15'000 + 8'000'001;
  for (int32_t k = 0; k < s.size(); ++k) {
    const int64_t t = frames[static_cast<size_t>(k)];
    CHECK(s.frame_soe(k) == t - length / 2);
    CHECK(s.frame_eoe(k) == t - length / 2 + length);
    CHECK(s.soe(k, 479) == s.frame_soe(k));  // bottom-up reads the last row first
  }
  const Schedule end = Schedule::blender(rolling, frames, Position::kEnd, Span{-2 * kW, frames.back() + kW});
  CHECK(end.frame_eoe(4) == frames[4]);
}

TEST_CASE("U2 exposure: set, sigma P or 0 per motion_blur.use, 10 ms otherwise") {
  const FrameRate r24{24, 1, 1};
  const BlenderShutter on{true, 0.5f, Position::kCenter}, off{false, 0.5f, Position::kCenter};
  // Set wins over every source.
  CHECK(resolve_exposure(1234, &on, &r24).ns == 1234);
  CHECK(resolve_exposure(1234, &on, &r24).source == ExposureSource::kSet);
  CHECK(resolve_exposure(0, nullptr, nullptr).ns == 0);
  // Motion blur on: round_half_even(sigma P), exact. 0.5 * 1e9 / 24 = 20833333.33.
  CHECK(resolve_exposure(std::nullopt, &on, &r24).ns == 20'833'333);
  // Motion blur off: instantaneous.
  const Exposure e_off = resolve_exposure(std::nullopt, &off, &r24);
  CHECK(e_off.ns == 0);
  CHECK(e_off.source == ExposureSource::kBlenderShutter);
  // Not a Blender source: 10 ms.
  const Exposure e_def = resolve_exposure(std::nullopt, nullptr, nullptr);
  CHECK(e_def.ns == 10'000'000);
  CHECK(e_def.source == ExposureSource::kDefault);
  CHECK(std::string(exposure_source_name(e_def.source)) == "default");
  CHECK(std::string(exposure_source_name(ExposureSource::kBlenderShutter)) == "blender_shutter");
  // Ties round to even: 2^-10 * 1e9 = 976562.5 -> 976562; 3 * 2^-10 * 1e9 = 2929687.5 -> 2929688.
  const FrameRate r1{1, 1, 1};
  const BlenderShutter tie1{true, 0x1p-10f, Position::kCenter}, tie3{true, 0x3p-10f, Position::kCenter};
  CHECK(resolve_exposure(std::nullopt, &tie1, &r1).ns == 976'562);
  CHECK(resolve_exposure(std::nullopt, &tie3, &r1).ns == 2'929'688);
  // A float32 sigma is used exactly: 0.3f = 0.300000011920928955078125; 0.3f * 1e9 / 30 * 1001/1000.
  const FrameRate ntsc{30, 1001, 1000};
  const BlenderShutter s03{true, 0.3f, Position::kCenter};
  CHECK(resolve_exposure(std::nullopt, &s03, &ntsc).ns == 10'010'000);  // 10010000.398 -> 10010000
  // A small fps_base: B_num / B_den is the float32's exact dyadic value (BL §4.2), so B_den reaches 2^33 at 0.001f
  // (0.5 1e9 0.001000000047497451 / 24 = 20833.33) and 2^38 at Blender's minimum 1e-5f (208.33).
  const FrameRate milli{24, 8'589'935, int64_t{1} << 33};
  CHECK(resolve_exposure(std::nullopt, &on, &milli).ns == 20'833);
  const FrameRate min_base{24, 2'748'779, int64_t{1} << 38};
  CHECK(resolve_exposure(std::nullopt, &on, &min_base).ns == 208);
  // Any int64 base stays exact: 0.5 1e9 (2^62 + 1) / (30 (2^62 - 1)) = 16666666.67.
  const FrameRate wide_base{30, (int64_t{1} << 62) + 1, (int64_t{1} << 62) - 1};
  CHECK(resolve_exposure(std::nullopt, &on, &wide_base).ns == 16'666'667);
  const FrameRate zero_base{24, 0, 1};
  CHECK_THROWS_AS(resolve_exposure(std::nullopt, &on, &zero_base), std::invalid_argument);
  const BlenderShutter tiny{true, 1e-30f, Position::kCenter};
  CHECK(resolve_exposure(std::nullopt, &tiny, &r24).ns == 0);
  const BlenderShutter huge{true, 1e30f, Position::kCenter};
  CHECK_THROWS_AS(resolve_exposure(std::nullopt, &huge, &r24), std::invalid_argument);
  CHECK_THROWS_AS(resolve_exposure(std::nullopt, &on, nullptr), std::invalid_argument);
  CHECK_THROWS_AS(resolve_exposure(-1, nullptr, nullptr), std::invalid_argument);
}

TEST_CASE("U2 end frames outside the bundle span are dropped and listed; none left is refused") {
  // A Blender timeline at 30 fps, frames 1..61; the bundle spans [-2W, t_end] with t_end = (floor(t_e/W) + 1) W.
  const FrameRate rate{30, 1, 1};
  const std::vector<int64_t> frames = blender_frames(1, 61, 1, rate);
  const int64_t t_e = frames.back();
  const int64_t t_end = (balla::time::floor_div(t_e, kW) + 1) * kW;
  const int64_t d = t_end - t_e;
  REQUIRE(d > 0);
  const Span bundle{-2 * kW, t_end};
  const int32_t n = static_cast<int32_t>(frames.size());
  auto centre = [&](int64_t exposure, int64_t dt) {
    const RowTiming rt{exposure, dt, 2, dt > 0 ? Shutter::kRolling : Shutter::kGlobal, Readout::kTopDown};
    return Schedule::blender(rt, frames, Position::kCenter, bundle);
  };
  // T + R = 4W keeps frame 0 at -2W; T + R > 4W (floor((T + R)/2) > 2W) drops it.
  const Schedule first_kept = centre(4 * kW, 0);
  CHECK(first_kept.candidate(0) == 0);
  CHECK(first_kept.frame_soe(0) == -2 * kW);
  const Schedule first_dropped = centre(4 * kW + 2, 0);
  REQUIRE(!first_dropped.dropped().empty());
  CHECK(first_dropped.dropped()[0] == 0);
  CHECK(first_dropped.candidate(0) == 1);
  CHECK(has_warning(first_dropped, "FRM-SPAN"));
  // With rolling readout: T + R = 4W + 2 with R = W (two rows, Delta = W) drops frame 0 the same way.
  CHECK(centre(3 * kW + 2, kW).dropped()[0] == 0);
  // The last frame: kept while ceil((T + R)/2) <= t_end - t_ns(frame_end), dropped above.
  const Schedule last_kept = centre(2 * d, 0);
  CHECK(last_kept.size() == n);
  CHECK(last_kept.dropped().empty());
  CHECK(last_kept.frame_eoe(n - 1) == t_end);
  const Schedule last_dropped = centre(2 * d + 1, 0);
  CHECK(last_dropped.size() == n - 1);
  REQUIRE(last_dropped.dropped().size() == 1);
  CHECK(last_dropped.dropped()[0] == n - 1);
  CHECK(has_warning(last_dropped, "FRM-SPAN"));
  const Schedule last_dropped_rolling = centre(2 * d + 1 - 1000, 1000);
  CHECK(last_dropped_rolling.dropped().size() == 1);
  // None left: refused with FRM-SPAN.
  try {
    (void)centre(t_end + 2 * kW + 1, 0);
    FAIL("expected FRM-SPAN");
  } catch (const FramesError& e) {
    CHECK(e.code == "FRM-SPAN");
  }
  try {
    (void)Schedule::uniform(RowTiming{10, 0, 1, Shutter::kGlobal, Readout::kTopDown}, 5'000, 100, 3, Span{0, 1000});
    FAIL("expected FRM-SPAN");
  } catch (const FramesError& e) {
    CHECK(e.code == "FRM-SPAN");
  }
}

TEST_CASE("U2 instantaneous rows, mid on negative times, shifted schedules") {
  const RowTiming instant{0, 7, 3, Shutter::kRolling, Readout::kTopDown};
  const Schedule s = Schedule::uniform(instant, -1'001, 1'000, 4, Span{-10'000, 10'000});
  for (int32_t k = 0; k < s.size(); ++k) {
    for (int32_t r = 0; r < 3; ++r) CHECK(s.eoe(k, r) == s.soe(k, r));
  }
  // frame 0: soe -1001, eoe -1001 + 14 = -987; mid = floor(-1988 / 2) = -994. frame 1: [-1, 13] -> floor(12/2) = 6.
  CHECK(s.frame_mid(0) == -994);
  const RowTiming odd{3, 0, 1, Shutter::kGlobal, Readout::kTopDown};
  const Schedule o = Schedule::uniform(odd, -1'001, 1'000, 2, Span{-10'000, 10'000});
  CHECK(o.frame_mid(0) == -1'000);  // floor((-1001 - 998) / 2) = floor(-999.5)
  CHECK(o.frame_mid(1) == 0);       // floor((-1 + 2) / 2) = floor(0.5)
}

TEST_CASE("U2 sensor_time equals reported_time without delay and quantum") {
  std::mt19937_64 rng(17);
  std::uniform_int_distribution<int64_t> t(-50'000'000'000, 50'000'000'000);
  const balla::sensor::Readout clock{1000, 5000, 37, -123'456'789, 200.0};
  const balla::sensor::Readout plain{0, 0, 0, -123'456'789, 200.0};
  for (int i = 0; i < 10000; ++i) {
    const int64_t x = t(rng);
    CHECK(sensor_time(clock, x) == balla::sensor::reported_time(plain, x));
    CHECK(sensor_time(clock, x) == x - 123'456'789 + std::llround(200.0 * 1e-6 * static_cast<double>(x)));
  }
}

TEST_CASE("U2 shifted schedules, FRM-EXPOSURE, invalid input") {
  const RowTiming rt{1'000, 10, 4, Shutter::kRolling, Readout::kBottomUp};
  const Schedule s = Schedule::uniform(rt, 0, 2'000, 5, Span{0, 9'000});
  REQUIRE(s.size() == 4);
  const std::vector<int64_t> shift{3, -2, 0, 7};
  const Schedule t = s.shifted(shift);
  for (int32_t k = 0; k < 4; ++k) {
    for (int32_t r = 0; r < 4; ++r) {
      CHECK(t.soe(k, r) == s.soe(k, r) + shift[static_cast<size_t>(k)]);
      CHECK(t.eoe(k, r) == s.eoe(k, r) + shift[static_cast<size_t>(k)]);
    }
  }
  CHECK(t.dropped().size() == 1);
  CHECK(t.dropped()[0] == 4);
  CHECK_THROWS_AS(s.shifted(std::vector<int64_t>{1, 2}), std::invalid_argument);
  CHECK(!has_warning(s, "FRM-EXPOSURE"));

  // T > P, and R + T > P with T <= P.
  const Span wide{0, 1'000'000};
  CHECK(has_warning(Schedule::uniform({2'500, 0, 4, Shutter::kGlobal, Readout::kTopDown}, 0, 2'000, 3, wide),
                    "FRM-EXPOSURE"));
  CHECK(has_warning(Schedule::uniform({1'990, 10, 4, Shutter::kRolling, Readout::kTopDown}, 0, 2'000, 3, wide),
                    "FRM-EXPOSURE"));
  const std::vector<int64_t> frames{0, 2'000, 4'001};
  CHECK(has_warning(Schedule::blender({1'990, 10, 4, Shutter::kRolling, Readout::kTopDown}, frames, Position::kStart,
                                      wide),
                    "FRM-EXPOSURE"));

  const RowTiming ok{10, 0, 1, Shutter::kGlobal, Readout::kTopDown};
  CHECK_THROWS_AS(Schedule::uniform(ok, 0, 0, 3, wide), std::invalid_argument);
  CHECK_THROWS_AS(Schedule::uniform(ok, 0, 100, 0, wide), std::invalid_argument);
  CHECK_THROWS_AS(Schedule::uniform({10, 0, 0, Shutter::kGlobal, Readout::kTopDown}, 0, 100, 3, wide),
                  std::invalid_argument);
  CHECK_THROWS_AS(Schedule::uniform({-1, 0, 1, Shutter::kGlobal, Readout::kTopDown}, 0, 100, 3, wide),
                  std::invalid_argument);
  CHECK_THROWS_AS(Schedule::blender(ok, std::vector<int64_t>{0, 0}, Position::kStart, wide), std::invalid_argument);
  CHECK_THROWS_AS(Schedule::blender(ok, std::vector<int64_t>{}, Position::kStart, wide), std::invalid_argument);
}

TEST_CASE("U2 extreme uniform schedules: exact anchors, bounded drop lists") {
  const RowTiming g{100, 0, 1, Shutter::kGlobal, Readout::kTopDown};
  const Span second{0, 1'000'000'000};
  // t0 + k P past int64: candidate 4 at 4 (2^62 + 1000) must not wrap back into the support.
  const Schedule far = Schedule::uniform(g, 0, (int64_t{1} << 62) + 1'000, 5, second);
  REQUIRE(far.size() == 1);
  CHECK(far.frame_soe(0) == 0);
  REQUIRE(far.dropped().size() == 4);
  CHECK(far.dropped()[0] == 1);
  CHECK(far.dropped()[3] == 4);
  // Anchors below and above int64 on both sides of the one kept frame.
  const int64_t lo = std::numeric_limits<int64_t>::min() + 3, p = std::numeric_limits<int64_t>::max();
  const Schedule wrap = Schedule::uniform(g, lo, p, 5, Span{0, 1'000});
  REQUIRE(wrap.size() == 1);
  CHECK(wrap.candidate(0) == 1);
  CHECK(wrap.frame_soe(0) == 2);
  CHECK(wrap.dropped().size() == 4);
  // A count far past the support is refused before any list is formed; up to kMaxDroppedFrames drops are listed.
  try {
    (void)Schedule::uniform(g, 0, 1'000'000, 2'000'000'000, second);
    FAIL("expected FRM-SPAN");
  } catch (const FramesError& e) {
    CHECK(e.code == "FRM-SPAN");
  }
  const Schedule capped = Schedule::uniform(g, 0, 1'000'000, 1'000 + kMaxDroppedFrames, second);
  CHECK(capped.size() == 1'000);
  REQUIRE(capped.dropped().size() == static_cast<size_t>(kMaxDroppedFrames));
  CHECK(capped.dropped().front() == 1'000);
  CHECK(capped.dropped().back() == 999 + kMaxDroppedFrames);
  CHECK_THROWS_AS(Schedule::uniform(g, 0, 1'000'000, 1'001 + kMaxDroppedFrames, second), FramesError);
  // Count null with t0 far before the support: the front drops are bounded the same way.
  const Schedule front = Schedule::uniform(g, -1'000 * 1'000'000, 1'000'000, std::nullopt, second);
  CHECK(front.size() == 1'000);
  CHECK(front.dropped().size() == 1'000);
  CHECK_THROWS_AS(Schedule::uniform(g, -(int64_t{kMaxDroppedFrames} + 1) * 1'000, 1'000, std::nullopt, second),
                  FramesError);
  CHECK_THROWS_AS(Schedule::uniform(g, -(int64_t{1} << 62), 1, std::nullopt, Span{0, 10}), std::invalid_argument);
  // R + T past int64 is refused.
  const int64_t big = std::numeric_limits<int64_t>::max();
  CHECK_THROWS_AS(Schedule::uniform({big, 1, 2, Shutter::kRolling, Readout::kTopDown}, 0, 1, 1, second),
                  std::invalid_argument);
  CHECK_THROWS_AS(Schedule::uniform({0, big, 3, Shutter::kRolling, Readout::kTopDown}, 0, 1, 1, second),
                  std::invalid_argument);
  // frame_mid at anchors next to the int64 ends, where soe + eoe itself would overflow.
  const int64_t top = std::numeric_limits<int64_t>::max(), bottom = std::numeric_limits<int64_t>::min();
  const Schedule high = Schedule::uniform(g, top - 1'000, 1'000, 1, Span{top - 1'000, top});
  REQUIRE(high.size() == 1);
  CHECK(high.frame_eoe(0) == top - 900);
  CHECK(high.frame_mid(0) == top - 950);
  const RowTiming odd_rolling{4, 3, 2, Shutter::kRolling, Readout::kBottomUp};  // R + T = 7
  const Schedule low = Schedule::uniform(odd_rolling, bottom, 10, 2, Span{bottom, bottom + 100});
  REQUIRE(low.size() == 2);
  CHECK(low.frame_mid(0) == bottom + 3);   // floor((2 min + 7) / 2)
  CHECK(low.frame_mid(1) == bottom + 13);
  const Schedule edge = Schedule::uniform(odd_rolling, top - 7, 1, 1, Span{top - 7, top});
  REQUIRE(edge.size() == 1);
  CHECK(edge.frame_eoe(0) == top);
  CHECK(edge.frame_mid(0) == top - 4);  // floor((2 max - 7) / 2) = max - 4
  const Schedule moved = high.shifted(std::vector<int64_t>{900});
  CHECK(moved.frame_eoe(0) == top);
  CHECK(moved.frame_mid(0) == top - 50);
  const std::vector<int64_t> late{top - 100};
  const Schedule at_end = Schedule::blender(g, late, Position::kEnd, Span{top - 1'000, top});
  REQUIRE(at_end.size() == 1);
  CHECK(at_end.frame_mid(0) == top - 150);
  // Blender: an END anchor below int64 is dropped, and spacings past int64 are fine.
  const std::vector<int64_t> frames{std::numeric_limits<int64_t>::min() + 5, 0, 1'000};
  const Schedule b = Schedule::blender(g, frames, Position::kEnd, Span{0, 10'000});
  REQUIRE(b.size() == 1);
  CHECK(b.candidate(0) == 2);
  CHECK(b.frame_soe(0) == 900);
  REQUIRE(b.dropped().size() == 2);
  CHECK(b.dropped()[0] == 0);
  CHECK(b.dropped()[1] == 1);
}
