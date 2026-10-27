// UI-4b: real resolver values through classify, plus an exhaustive grid oracle for the inverse clocks (ui.md T1/T2).
#include <doctest/doctest.h>

#include "config/tier.h"
#include "sensor/readout.h"

using namespace balla;
namespace {
config::Node leaf(config::Value v) { config::Node n; n.leaf = std::move(v); return n; }
config::Node map(std::vector<std::pair<std::string, config::Node>> entries) {
  config::Node n; n.is_map = true; n.map = std::move(entries); return n;
}
config::Resolved resolve(std::vector<std::pair<std::string, config::Node>> cli = {}, bool fidelity = false) {
  config::Inputs in;
  in.config = map({{"format", leaf(std::string("ballasim.run/2"))},
                   {"output", map({{"dir", leaf(std::string("out"))}})},
                   {"accuracy", map({{"profile", leaf(std::string(fidelity ? "fidelity" : "phase1"))}})},
                   {"cameras", map({{"default", map({{"sensor", map({{"family", leaf(std::string("prophesee_gen3"))}})},
                                                     {"photometry", map({{"illuminance_scale", leaf(200.0)}})}})}})}});
  in.cli = map(std::move(cli));
  config::BundleInfo b;
  b.cameras.push_back({"left", "prophesee_gen3", 64, 48, 123, 15e-6, 2});
  b.t_end_ns = 10 * (int64_t{1} << 23);
  b.supplied = true;
  b.bundle_digest = std::string(64, 'b'); b.engine_digest = std::string(64, 'e');
  return config::resolve(in, b);
}
}
TEST_CASE("tier: the main resolver's event-only edits mask the probe and classify E") {
  const auto base = resolve();
  for (const auto& [key, v] : std::vector<std::pair<std::string, config::Value>>{
           {"cameras.left.sensor.readout_delay_s", 0.0001}, {"cameras.left.sensor.readout_quantum_s", 2e-6},
           {"cameras.left.sensor.clock_offset_ns", int64_t{500}}, {"cameras.left.sensor.clock_drift_ppm", 200.0},
           {"output.label_cut", 0.8}}) {
    const auto edit = resolve({{key, leaf(v)}});
    CHECK(config::classify(base, edit, "left", {}).tier == config::Tier::kE);
    CHECK(config::probe_digest(base, "left") == config::probe_digest(edit, "left"));
  }
  CHECK(config::classify(base, base, "left", {}).tier == config::Tier::kNone);
  CHECK(config::classify(base, resolve({{"output.gt", leaf(false)}}), "left", {}).tier == config::Tier::kO);
  CHECK(config::classify(base, resolve({{"cameras.left.sensor.theta_on", leaf(0.3)}}), "left", {}).tier ==
        config::Tier::kS);
  CHECK(config::classify(base, resolve({{"accuracy.n_subsamples", leaf(int64_t{128})}}), "left", {}).tier ==
        config::Tier::kR);
  config::CacheFacts fp64; fp64.precision = "fp64";
  CHECK(config::classify(base, resolve({{"output.label_cut", leaf(0.8)}}), "left", fp64).tier == config::Tier::kS);
}
TEST_CASE("tier: adequacy is conservative and directional under fidelity") {
  const auto base = resolve({}, true);
  CHECK(config::classify(base, resolve({{"cameras.left.sensor.dark_current_fA", leaf(8.0)}}, true), "left", {}).tier ==
        config::Tier::kSTilde);
  CHECK(config::classify(base, resolve({{"cameras.left.sensor.dark_current_fA", leaf(0.001)}}, true), "left", {}).tier ==
        config::Tier::kR);
  CHECK(config::classify(base, resolve({{"cameras.left.sensor.readout_quantum_s", leaf(2e-6)}}, true), "left", {}).tier ==
        config::Tier::kR);
}
TEST_CASE("clocks: inverse clock and crossing are minimal over signed times and counter phases") {
  for (double drift : {-200.0, 0.0, 37.5, 200.0}) for (int64_t q : {0, 1000, 1024}) {
    sensor::Readout r{q, q + 113, q / 3, -137, drift};
    for (int64_t s : {-1000001, -1001, -1, 0, 1, 137, 1000, 1000001}) {
      const int64_t u = config::sensor_time_lower_bound(r, s), t = config::crossing_lower_bound(r, s);
      CHECK(config::sensor_clock_ns(r, u) >= s);
      CHECK(config::sensor_clock_ns(r, u - 1) < s);
      CHECK(sensor::reported_time(r, t) >= s);
      CHECK(sensor::reported_time(r, t - 1) < s);
      // Exhaustively cover a surrounding 20 us window, including the negative-time floor-grid behaviour.
      for (int64_t a = t - 10000; a <= t + 10000; ++a) {
        CHECK((sensor::reported_time(r, a) >= s) == (a >= t));
      }
    }
  }
}
