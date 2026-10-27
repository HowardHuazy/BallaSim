// The rate limiter (sensor/limiter.h; DESIGN D13-5, F043; docs/dev/EVIMO2_LIMITER.md §5): its keys and validate()
// rules, the exact refill, and the oracle rows L1-L4 with the cut invariance (P2-H §6) on Poisson streams drawn from
// the test tags.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/rng/purpose.h"
#include "sensor/limiter.h"
#include "sensor/param_fields.h"
#include "sensor/params.h"

using namespace balla::sensor;
namespace rng = balla::rng;

namespace {

constexpr int64_t kUs = 1000;
constexpr int64_t kEpochUs = 65'600;
constexpr uint32_t kWidth = 640, kHeight = 480;

// An output event as the engine's (engine::OutEvent's names), with an id to tell the events apart.
struct Ev {
  int64_t t_reported, t_cross;
  uint16_t x, y;
  uint8_t p;
  uint32_t id;
  bool operator==(const Ev&) const = default;
};

// Test draws (S1 §5.2: tags 0x7F00-0x7FFF are the tests').
rng::U32x4 test_draw(uint64_t seed, uint32_t stream, uint64_t index) {
  return rng::draw(seed, static_cast<rng::Purpose>(0x7F21), 0, stream, index);
}
double uniform(uint32_t w) { return (static_cast<double>(w) + 0.5) * 0x1p-32; }

// Poisson(lambda) by inversion of one uniform.
int poisson(double lambda, double u) {
  double p = std::exp(-lambda), cdf = p;
  int k = 0;
  while (u > cdf && k < 1000) {
    ++k;
    p *= lambda / k;
    cdf += p;
  }
  return k;
}

// A Poisson stream of `rate` events per us over the microseconds [u0, u1), uniform over the pixels and polarities and
// sorted as the engine sorts (t_reported, then x, y). quantum_ns > 0: stamps on the counter grid phase + k q (q = 1 us
// here); 0: times anywhere in the microsecond. t_cross precedes the stamp by 300-301 us.
std::vector<Ev> poisson_stream(uint64_t seed, double rate, int64_t u0, int64_t u1, int64_t quantum_ns,
                               int64_t readout_phase_ns, uint32_t* next_id) {
  std::vector<Ev> out;
  for (int64_t u = u0; u < u1; ++u) {
    const int n = poisson(rate, uniform(test_draw(seed, 0, static_cast<uint64_t>(u)).v[0]));
    for (int i = 0; i < n; ++i) {
      const rng::U32x4 d = test_draw(seed, 1 + static_cast<uint32_t>(i), static_cast<uint64_t>(u));
      Ev e{};
      e.t_reported = quantum_ns > 0 ? u * kUs + readout_phase_ns : u * kUs + d.v[3] % kUs;
      e.t_cross = e.t_reported - 300 * kUs - (d.v[3] >> 10) % kUs;
      e.x = static_cast<uint16_t>(d.v[0] % kWidth);
      e.y = static_cast<uint16_t>(d.v[1] % kHeight);
      e.p = static_cast<uint8_t>(d.v[2] & 1u);
      e.id = (*next_id)++;
      out.push_back(e);
    }
  }
  std::sort(out.begin(), out.end(), [](const Ev& a, const Ev& b) {
    return std::tie(a.t_reported, a.x, a.y, a.id) < std::tie(b.t_reported, b.x, b.y, b.id);
  });
  return out;
}

// A limiter on prophesee_gen3's values with the given cap, depth, quantum and phase.
SensorParams limiter_params(double r0, double r1, int64_t depth, double quantum_s, std::optional<double> phase_s) {
  SensorParams p = preset("prophesee_gen3");
  p.limiter = LimiterMode::kEpoch;
  p.limiter_rate0_hz = r0;
  p.limiter_rate1_hz = r1;
  p.limiter_depth = depth;
  p.readout_quantum_s = quantum_s;
  p.readout_delay_s = std::max(p.readout_delay_s, quantum_s);
  p.limiter_phase_s = phase_s;
  return p;
}

LimiterConfig config_of(double r0, double r1, int64_t depth, double quantum_s, std::optional<double> phase_s,
                        uint64_t seed = 7) {
  return limiter_config(limiter_params(r0, r1, depth, quantum_s, phase_s), seed, 1, kWidth);
}

// The events of `all` through one limiter, in calls cut before the times `cuts` (each call: the events reported before
// its cut, complete up to it), then flushed.
std::vector<Ev> limit_in_calls(const LimiterConfig& c, const std::vector<Ev>& all, const std::vector<int64_t>& cuts,
                               LimiterState* state = nullptr) {
  Limiter<Ev> lim(c);
  std::vector<Ev> out;
  size_t i = 0;
  for (const int64_t cut : cuts) {
    std::vector<Ev> call;
    while (i < all.size() && all[i].t_reported < cut) call.push_back(all[i++]);
    lim.apply(call, cut);
    out.insert(out.end(), call.begin(), call.end());
  }
  std::vector<Ev> rest(all.begin() + static_cast<std::ptrdiff_t>(i), all.end());
  lim.flush(rest);
  out.insert(out.end(), rest.begin(), rest.end());
  if (state) *state = lim.state();
  return out;
}

}  // namespace

TEST_CASE("limiter keys: off in every family prior, inert while off, refusals while on") {
  for (const std::string& n : preset_names()) {
    INFO(n);
    const SensorParams s = preset(n);
    CHECK(s.limiter == LimiterMode::kOff);
    CHECK_FALSE(s.limiter_on());
    CHECK(find_prior(n, "sensor.limiter")->status == balla::config::Status::kChoice);
    bool inherited = true;
    CHECK(find_prior(n, "sensor.limiter", &inherited) != nullptr);
    CHECK_FALSE(inherited);  // stated in every family
  }
  const SensorParams g = preset("prophesee_gen3");
  CHECK((g.limiter_rate0_hz == 6.0e6 && g.limiter_rate1_hz == 6.0e6 && g.limiter_period_s == 65.6e-3));
  CHECK((g.limiter_depth == 32 && !g.limiter_phase_s.has_value()));
  for (const char* k : {"sensor.limiter_rate0_hz", "sensor.limiter_rate1_hz", "sensor.limiter_period_s"}) {
    CHECK(find_prior("prophesee_gen3", k)->evimo2_derived);  // flagged when on with the prior's values
  }
  CHECK_FALSE(find_prior("prophesee_gen3", "sensor.limiter_depth")->evimo2_derived);
  // The rows: stage sensor, inert while off; the depth warns below 16 (its soft range); the phase nullable.
  for (const balla::config::Param& r : param_rows()) {
    if (!r.key.starts_with("sensor.limiter")) continue;
    INFO(r.key);
    CHECK(r.stage == balla::config::Stage::kSensor);
    CHECK(r.when.size() == (r.key == "sensor.limiter" ? 0u : 1u));
  }
  const balla::config::Param& depth = *param_row("limiter_depth");
  CHECK((depth.soft.contains(16) && !depth.soft.contains(15) && depth.hard.contains(1) && !depth.hard.contains(0)));
  CHECK(param_row("limiter_phase_s")->nullable);

  // Off: the rows are inert (no cross-key rule, no fingerprint). On: the rules.
  SensorParams off = preset("prophesee_gen3");
  off.limiter_depth = 1;
  off.limiter_rate1_hz = 7e6;
  off.limiter_phase_s = 1.0;
  CHECK_NOTHROW(off.validate());
  CHECK(fingerprint(off, {}) == fingerprint(preset("prophesee_gen3"), {}));
  SensorParams on = limiter_params(6e6, 5.64e6, 7, 1e-6, std::nullopt);
  CHECK_NOTHROW(on.validate());
  CHECK(on.limiter_min_depth() == 7);  // ceil(6 Mev/s x 1 us) + 1
  CHECK(fingerprint(on, {}) != fingerprint(preset("prophesee_gen3"), {}));
  SensorParams deeper = on;
  deeper.limiter_depth = 8;
  CHECK(fingerprint(deeper, {}) != fingerprint(on, {}));
  on.limiter_depth = 6;
  CHECK_THROWS_WITH_AS(on.validate(), doctest::Contains("limiter_depth = 6 must be >= ceil"), std::invalid_argument);
  on.limiter_depth = 7;
  on.limiter_rate1_hz = 6.1e6;
  CHECK_THROWS_WITH_AS(on.validate(), doctest::Contains("limiter_rate1_hz must be <= limiter_rate0_hz"),
                       std::invalid_argument);
  on.limiter_rate1_hz = 6e6;  // a flat cap
  CHECK_NOTHROW(on.validate());
  on.limiter_phase_s = 65.6e-3;
  CHECK_THROWS_WITH_AS(on.validate(), doctest::Contains("limiter_phase_s must lie in [0, limiter_period_s)"),
                       std::invalid_argument);
  on.limiter_phase_s = 0.0;
  CHECK_NOTHROW(on.validate());
  // The bin is max(quantum, 1 us): q = 0 and q = 1 us give 7, q = 2 us 13; 6.5 Mev/s at 1 us 8.
  CHECK(limiter_params(6e6, 6e6, 7, 0.0, std::nullopt).limiter_min_depth() == 7);
  CHECK(limiter_params(6e6, 6e6, 13, 2e-6, std::nullopt).limiter_min_depth() == 13);
  CHECK(limiter_params(6.5e6, 6e6, 8, 1e-6, std::nullopt).limiter_min_depth() == 8);
  CHECK_THROWS_AS(limiter_params(6e6, 6e6, 12, 2e-6, std::nullopt).validate(), std::invalid_argument);
  for (const int64_t bad : {int64_t{0}, int64_t{1'000'000'001}}) {
    SensorParams p = limiter_params(6e6, 6e6, 32, 1e-6, std::nullopt);
    p.limiter_depth = bad;
    CHECK_THROWS_WITH_AS(p.validate(), doctest::Contains("limiter_depth"), std::invalid_argument);
  }
}

TEST_CASE("limiter config: integer units, the phase given or drawn from tag 0x0023") {
  const LimiterConfig c = config_of(6e6, 5.64e6, 32, 1e-6, std::nullopt, 0x0123456789abcdefull);
  CHECK((c.on && c.period_ns == 65'600'000 && c.quantum_ns == 1000 && c.depth_q32 == (int64_t{32} << 32)));
  CHECK(c.rate0_q48 == std::llround(6e6 * 0x1p48 / 1e9));
  CHECK(c.rate1_q48 == std::llround(5.64e6 * 0x1p48 / 1e9));
  const rng::U32x4 d = rng::draw(0x0123456789abcdefull, rng::Purpose::kLimiterPhase, 1, 0, 0);
  CHECK(c.phase_ns == static_cast<int64_t>(((uint64_t{d.v[1]} << 32) | d.v[0]) % 65'600'000u));
  // Another camera, another phase; a given phase is taken mod P.
  const LimiterConfig c2 = limiter_config(limiter_params(6e6, 5.64e6, 32, 1e-6, std::nullopt),
                                          0x0123456789abcdefull, 2, kWidth);
  CHECK(c2.phase_ns != c.phase_ns);
  CHECK(config_of(6e6, 5.64e6, 32, 1e-6, 0.0123456789).phase_ns == 12'345'679);
  CHECK_FALSE(limiter_config(preset("prophesee_gen3"), 1, 0, kWidth).on);
  SensorParams bad = limiter_params(6e6, 5.64e6, 3, 1e-6, std::nullopt);
  CHECK_THROWS_AS(limiter_config(bad, 1, 0, kWidth), std::invalid_argument);
}

TEST_CASE("limiter refill: exact telescoping, whole epochs, the integral of the cap") {
  const LimiterConfig c = config_of(6e6, 5.64e6, 32, 1e-6, 0.0213);
  const int64_t big_p = c.period_ns;
  const LimiterI128 epoch = limiter_refill_q32(c, c.phase_ns, c.phase_ns + big_p);
  // E = floor(P (r0 + r1) / 2) in Q32 events (the rates Q48): 381 792 events per epoch (5.82 Mev/s x 65.6 ms).
  CHECK(epoch == (static_cast<LimiterI128>(big_p) * (c.rate0_q48 + c.rate1_q48)) >> 17);
  // The rates are rounded once to Q48 per ns: 1e-7 events per epoch, plus E's floor (2^-32).
  CHECK(std::fabs(static_cast<double>(epoch) * 0x1p-32 - 5.82e6 * 65.6e-3) < 1e-6);
  // Down to the rows' 1 Hz floor the cap is the one set, to 1e-5 (Q32 rates would make 1 Hz 0.93 Hz): 160 epochs
  // (10.5 s) at 1 Hz, and at 1.5 Hz falling to 1 Hz (a 1.25 Hz mean).
  for (const auto& [r0, r1] : {std::pair{1.0, 1.0}, std::pair{1.5, 1.0}}) {
    const LimiterConfig low = config_of(r0, r1, 2, 1e-6, 0.0);
    const double events = (r0 + r1) / 2 * 160 * 65.6e-3;
    CHECK(std::fabs(static_cast<double>(limiter_refill_q32(low, 0, 160 * low.period_ns)) * 0x1p-32 / events - 1) <
          1e-5);
  }
  uint64_t s = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    const rng::U32x4 d = test_draw(11, 99, s++);
    // Times from -1 s to +1 s; a <= b <= e.
    int64_t a = static_cast<int64_t>(d.v[0] % 2'000'000'000u) - 1'000'000'000;
    int64_t b = a + d.v[1] % 200'000'000u, e = b + d.v[2] % 200'000'000u;
    INFO(a << " " << b << " " << e);
    CHECK(limiter_refill_q32(c, a, e) == limiter_refill_q32(c, a, b) + limiter_refill_q32(c, b, e));
    CHECK(limiter_refill_q32(c, a, a + big_p) == epoch);  // one period from any phase
    CHECK(limiter_refill_q32(c, a, a) == 0);
    // Against the integral of R in double (to a few events in 1e9).
    const auto cap_integral = [&](int64_t t) {  // integral from the reset before 0 to t, in events
      const double k = std::floor(static_cast<double>(t - c.phase_ns) / static_cast<double>(big_p));
      const double phi = static_cast<double>(t - c.phase_ns) - k * static_cast<double>(big_p);
      const double r0 = 6e-3, dr = 0.36e-3;  // events per ns
      const double pd = static_cast<double>(big_p);
      return k * (r0 - dr / 2) * pd + r0 * phi - dr * phi * phi / (2 * pd);
    };
    CHECK(std::fabs(static_cast<double>(limiter_refill_q32(c, a, b)) * 0x1p-32 - (cap_integral(b) - cap_integral(a))) <
          1e-6 * static_cast<double>(b - a) * 6e-3 + 2);
  }
}

TEST_CASE("limiter: a hand-checked stream, the smallest keys kept, held bins, misuse refused") {
  // A flat 6 Mev/s cap, B = 7, q = 1 us: ten events in each of four stamps. The first bin finds the bucket full (7),
  // the next two get one microsecond's refill each (6), and 10 idle microseconds fill the bucket to 7 again.
  const LimiterConfig c = config_of(6e6, 6e6, 7, 1e-6, 0.0);
  std::vector<Ev> all;
  uint32_t id = 0;
  for (const int64_t t : {int64_t{5000}, int64_t{6000}, int64_t{7000}, int64_t{17000}}) {
    for (uint16_t x = 0; x < 10; ++x) all.push_back(Ev{t, t - 300'000 - x, x, 3, static_cast<uint8_t>(x & 1), id++});
  }
  LimiterState st;
  const std::vector<Ev> out = limit_in_calls(c, all, {}, &st);
  std::vector<int> per_stamp(4, 0);
  for (const Ev& e : out) ++per_stamp[e.id / 10];
  CHECK(per_stamp == std::vector<int>{7, 6, 6, 7});
  CHECK((st.kept == 26 && st.dropped == 14 && st.capped_bins == 4));
  // The kept of each stamp are its smallest keys, in input order.
  for (int b = 0; b < 4; ++b) {
    std::vector<std::pair<uint64_t, uint32_t>> keys;
    for (int k = 0; k < 10; ++k) {
      const Ev& e = all[static_cast<size_t>(10 * b + k)];
      keys.emplace_back(limiter_key(c, e.y * kWidth + e.x, e.t_cross), e.id);
    }
    std::sort(keys.begin(), keys.end());
    std::vector<uint32_t> want;
    for (int k = 0; k < per_stamp[static_cast<size_t>(b)]; ++k) want.push_back(keys[static_cast<size_t>(k)].second);
    std::sort(want.begin(), want.end());
    std::vector<uint32_t> got;
    for (const Ev& e : out) {
      if (static_cast<int>(e.id / 10) == b) got.push_back(e.id);
    }
    CHECK(got == want);
  }
  // A bin that later events may still join is held: q = 0 and the call complete up to 5.5 us holds the bin [5, 6) us.
  const LimiterConfig c0 = config_of(6e6, 6e6, 7, 0.0, 0.0);
  Limiter<Ev> lim(c0);
  std::vector<Ev> call{Ev{4200, 0, 1, 1, 0, 0}, Ev{5100, 0, 2, 1, 0, 1}};
  lim.apply(call, 5500);
  CHECK((call.size() == 1 && call[0].id == 0 && lim.held().size() == 1));
  std::vector<Ev> next{Ev{5600, 0, 3, 1, 0, 2}};
  lim.apply(next, 6000);
  CHECK((next.size() == 2 && next[0].id == 1 && next[1].id == 2 && lim.held().empty()));
  std::vector<Ev> late{Ev{5900, 0, 4, 1, 0, 3}};  // a bin already decided
  CHECK_THROWS_WITH_AS(lim.apply(late, 7000), doctest::Contains("already decided"), std::invalid_argument);
  Limiter<Ev> lim2(c0);
  std::vector<Ev> unsorted{Ev{6100, 0, 1, 1, 0, 0}, Ev{5100, 0, 2, 1, 0, 1}};
  CHECK_THROWS_WITH_AS(lim2.apply(unsorted, 7000), doctest::Contains("out of time order"), std::invalid_argument);
  // Off: nothing moves, nothing is held.
  Limiter<Ev> none(limiter_config(preset("prophesee_gen3"), 7, 1, kWidth));
  std::vector<Ev> same = all;
  none.apply(same, 0);
  CHECK((same == all && none.held().empty()));
}

TEST_CASE("limiter L1: Poisson 10 Mev/s over 15 whole epochs, B = 32: the mean cap, its eighths, fold and line") {
  // 1 us stamps from t = 0, the epochs reset at 0: 984 000 us, about 9.8 M events offered in 120 calls of 8.2 ms.
  const LimiterConfig c = config_of(6e6, 5.64e6, 32, 1e-6, 0.0);
  const int64_t n_us = 15 * kEpochUs, chunk_us = 8200;
  Limiter<Ev> lim(c);
  std::vector<double> count(static_cast<size_t>(n_us), 0.0);
  uint32_t id = 0;
  uint64_t offered = 0;
  for (int64_t u0 = 0; u0 < n_us; u0 += chunk_us) {
    std::vector<Ev> ev = poisson_stream(101, 10.0, u0, std::min(u0 + chunk_us, n_us), 1000, 0, &id);
    offered += ev.size();
    if (u0 + chunk_us >= n_us) lim.flush(ev);
    else lim.apply(ev, (u0 + chunk_us) * kUs);
    for (const Ev& e : ev) count[static_cast<size_t>(e.t_reported / kUs)] += 1;
  }
  CHECK(lim.state().kept + lim.state().dropped == offered);
  double total = 0, first = 0, last = 0, sxy = 0, sx = 0, sxx = 0;
  std::complex<double> line = 0;
  for (int64_t u = 0; u < n_us; ++u) {
    const double c_u = count[static_cast<size_t>(u)];
    const double phi = (static_cast<double>(u % kEpochUs) + 0.5) / kEpochUs;  // the bin's mid-phase in [0, 1)
    total += c_u;
    if (phi < 0.125) first += c_u;
    if (phi >= 0.875) last += c_u;
    sx += phi;
    sxx += phi * phi;
    sxy += phi * c_u;
    line += c_u * std::polar(1.0, -2 * std::numbers::pi * phi);
  }
  const double n = static_cast<double>(n_us);
  const double mean = total / n;                         // events per us = Mev/s
  const double eighth = n / 8;
  const double slope = (sxy - sx * total / n) / (sxx - sx * sx / n);  // per unit phase
  const double fold = -slope / mean, amp = 2 * std::abs(line) / total;
  MESSAGE("L1: offered " << offered / 0.984e6 << " Mev/s, mean " << mean << ", eighths " << first / eighth << " / "
                         << last / eighth << ", fold " << fold << ", line " << amp << ", dropped "
                         << lim.state().dropped << ", capped bins " << lim.state().capped_bins);
  CHECK(std::fabs(mean - 5.820) <= 0.002);
  CHECK(std::fabs(first / eighth - 5.978) <= 0.005);
  CHECK(std::fabs(last / eighth - 5.663) <= 0.005);
  CHECK(std::fabs(fold - 0.0619) <= 0.002);   // (R0 - R1) / mean R
  CHECK(std::fabs(amp - 0.0197) <= 0.002);    // (R0 - R1) / (pi mean R): a sawtooth's fundamental
}

TEST_CASE("limiter L2: offered 3 Mev/s, B = 32: every event out, bitwise") {
  for (const double q : {1e-6, 0.0}) {
    INFO(q);
    uint32_t id = 0;
    const std::vector<Ev> all = poisson_stream(202, 3.0, 0, 200'000, q > 0 ? 1000 : 0, 0, &id);
    LimiterState st;
    const LimiterConfig c = config_of(6e6, 5.64e6, 32, q, std::nullopt);
    const std::vector<Ev> out = limit_in_calls(c, all, {50'000'000, 100'000'000, 150'000'000}, &st);
    CHECK(out == all);
    CHECK((st.dropped == 0 && st.capped_bins == 0 && st.kept == all.size()));
  }
}

TEST_CASE("limiter L3: arbitrary window cuts give the same stream, both quantum modes") {
  for (const double q : {1e-6, 0.0}) {
    INFO(q);
    uint32_t id = 0;
    // 30 ms at 9 Mev/s across a reset (the phase at 12.345 ms), stamps on a counter grid of phase 437 ns.
    const std::vector<Ev> all = poisson_stream(303, 9.0, 0, 30'000, q > 0 ? 1000 : 0, q > 0 ? 437 : 0, &id);
    const LimiterConfig c = config_of(6e6, 5.64e6, 32, q, 12.345e-3);
    LimiterState whole_state;
    const std::vector<Ev> whole = limit_in_calls(c, all, {}, &whole_state);
    CHECK(whole_state.dropped > 0);
    // Engine windows (2^23 ns), cuts inside microseconds and on stamps, an empty call, random cuts.
    std::vector<std::vector<int64_t>> cut_sets{{int64_t{1} << 23, int64_t{2} << 23, int64_t{3} << 23},
                                               {5'500'500, 5'501'437, 5'501'437, 12'345'000, 12'345'001, 29'999'999}};
    std::vector<int64_t> random_cuts;
    for (uint64_t k = 0; k < 400; ++k) random_cuts.push_back(test_draw(304, 7, k).v[0] % 30'000'000u);
    std::sort(random_cuts.begin(), random_cuts.end());
    cut_sets.push_back(random_cuts);
    for (const auto& cuts : cut_sets) {
      LimiterState st;
      CHECK(limit_in_calls(c, all, cuts, &st) == whole);
      CHECK((st.kept == whole_state.kept && st.dropped == whole_state.dropped &&
             st.capped_bins == whole_state.capped_bins && st.tokens_q32 == whole_state.tokens_q32));
    }
    // Calls that hand over events beyond their completeness: held until a later call or the flush, same stream.
    Limiter<Ev> lim(c);
    std::vector<Ev> out;
    for (size_t i = 0; i < all.size(); i += 5000) {
      std::vector<Ev> call(all.begin() + static_cast<std::ptrdiff_t>(i),
                           all.begin() + static_cast<std::ptrdiff_t>(std::min(i + 5000, all.size())));
      lim.apply(call, call.front().t_reported);  // complete only before its first event
      out.insert(out.end(), call.begin(), call.end());
    }
    std::vector<Ev> rest;
    lim.flush(rest);
    out.insert(out.end(), rest.begin(), rest.end());
    CHECK(out == whole);
  }
}

TEST_CASE("limiter L4: the dropped events are blind to polarity and position (2 SE), readout order is not") {
  // 60 ms at 10 Mev/s, q = 1 us: about 40 % dropped.
  uint32_t id = 0;
  const std::vector<Ev> all = poisson_stream(404, 10.0, 0, 60'000, 1000, 0, &id);
  const LimiterConfig c = config_of(6e6, 5.64e6, 32, 1e-6, std::nullopt);
  const std::vector<Ev> out = limit_in_calls(c, all, {});
  std::vector<uint8_t> kept(all.size(), 0);
  for (const Ev& e : out) kept[e.id] = 1;
  // The readout-order control: per stamp, the first a_u events in (x, y) order, a_u as the limiter's.
  std::vector<uint8_t> first(all.size(), 0);
  for (size_t i = 0; i < all.size();) {
    size_t j = i, a = 0;
    while (j < all.size() && all[j].t_reported == all[i].t_reported) a += kept[all[j++].id];
    for (size_t k = i; k < i + a; ++k) first[all[k].id] = 1;
    i = j;
  }
  const auto z = [&](const std::vector<uint8_t>& keep, auto value) {  // (kept mean - dropped mean) / SE
    double s[2] = {0, 0}, ss[2] = {0, 0}, n[2] = {0, 0};
    for (const Ev& e : all) {
      const double v = value(e);
      const int g = keep[e.id];
      s[g] += v;
      ss[g] += v * v;
      n[g] += 1;
    }
    const double m0 = s[0] / n[0], m1 = s[1] / n[1];
    const double v0 = ss[0] / n[0] - m0 * m0, v1 = ss[1] / n[1] - m1 * m1;
    return (m1 - m0) / std::sqrt(v0 / n[0] + v1 / n[1]);
  };
  const auto on = [](const Ev& e) { return static_cast<double>(e.p); };
  const auto x = [](const Ev& e) { return static_cast<double>(e.x); };
  const auto y = [](const Ev& e) { return static_cast<double>(e.y); };
  const double dropped_share = 1.0 - static_cast<double>(out.size()) / static_cast<double>(all.size());
  MESSAGE("L4: dropped share " << dropped_share << "; z (kept - dropped) ON " << z(kept, on) << ", x " << z(kept, x)
                               << ", y " << z(kept, y) << "; readout-order control x " << z(first, x));
  CHECK(dropped_share > 0.3);
  CHECK(std::fabs(z(kept, on)) <= 2);
  CHECK(std::fabs(z(kept, x)) <= 2);
  CHECK(std::fabs(z(kept, y)) <= 2);
  CHECK(std::fabs(z(first, x)) > 10);  // the test sees a positional policy
}
