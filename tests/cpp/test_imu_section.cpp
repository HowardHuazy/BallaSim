// The IMU's rows in the resolver (docs/design/phase3/outputs.md §4.10, §4.11, §6; P3-A A3): the digest skip while
// imu.enabled is false (D-P3-11), imu::validate at resolve time, and config::resolve_section, the overlay of `ballasim
// imu`: layers, refusals, warnings, the imu digest. U13's schema part is tests/cpp/imu/test_imu_params.cpp.
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "config/diagnostics.h"
#include "config/resolve.h"
#include "config/schema.h"
#include "config/sha256.h"
#include "core/time/time.h"
#include "imu/params.h"

using namespace balla;
using config::DiagKind;
using config::Value;

namespace {

using Map = std::vector<std::pair<std::string, config::Node>>;
config::Node leaf(Value v) {
  config::Node n;
  n.leaf = std::move(v);
  return n;
}
config::Node mapping(Map m) {
  config::Node n;
  n.is_map = true;
  n.map = std::move(m);
  return n;
}
config::Node str(const char* s) { return leaf(std::string(s)); }

// A run configuration on prophesee_gen3 cameras: the `default` block's extra keys, and per-camera blocks.
config::Inputs run_inputs(Map defaults = {}, std::vector<std::pair<std::string, Map>> blocks = {}) {
  Map block{{"sensor.family", str("prophesee_gen3")}, {"photometry.illuminance_scale", leaf(200.0)}};
  for (auto& d : defaults) block.push_back(std::move(d));
  Map cams{{"default", mapping(std::move(block))}};
  for (auto& [name, keys] : blocks) cams.emplace_back(name, mapping(std::move(keys)));
  config::Inputs in;
  in.config = mapping({{"format", str("ballasim.run/2")}, {"output", mapping({{"dir", str("out")}})},
                       {"cameras", mapping(std::move(cams))}});
  return in;
}

// A supplied bundle with 64 x 48 cameras left and right.
config::BundleInfo bundle() {
  config::BundleInfo b;
  b.cameras = {{"left", "prophesee_gen3", 64, 48, 0, 15e-6, 2.0}, {"right", "prophesee_gen3", 64, 48, 0, 15e-6, 2.0}};
  b.t_end_ns = 10 * time::kWindowNs;
  b.supplied = true;
  b.bundle_digest = std::string(64, 'b');
  b.engine_digest = std::string(64, 'e');
  return b;
}

config::Diagnostic error_of(const std::function<void()>& f) {
  try {
    f();
  } catch (const config::DiagnosticError& e) {
    CHECK(std::string(e.what()) == e.diag.message);
    return e.diag;
  }
  FAIL("no DiagnosticError");
  return {};
}

bool has(const std::vector<std::string>& v, const std::string& part) {
  return std::any_of(v.begin(), v.end(), [&](const std::string& s) { return s.find(part) != std::string::npos; });
}

const config::Diagnostic* diag_of(const std::vector<config::Diagnostic>& v, const std::string& part) {
  for (const config::Diagnostic& d : v) {
    if (d.message.find(part) != std::string::npos) return &d;
  }
  return nullptr;
}

using Digests = std::array<std::string, 5>;
Digests digests_of(const config::Resolved& r, size_t camera) { return r.digests.at(camera); }

TEST_CASE("digests: the IMU's rows enter no digest while imu.enabled is false") {
  const config::Resolved base = config::resolve(run_inputs(), bundle());
  REQUIRE(base.digests.size() == 2);
  // Every imu.* entry is in the record (content only), at its default.
  CHECK(std::get<bool>(base.find("imu.enabled", "left")->value) == false);
  CHECK(std::get<double>(base.find("imu.rate_hz", "right")->value) == 1000.0);
  // Values set while the IMU is off move no digest (and are inert, with a warning).
  const config::Resolved off = config::resolve(
      run_inputs({{"imu.rate_hz", leaf(500.0)}, {"imu.gyro.noise_density", leaf(1e-4)}, {"imu.enabled", leaf(false)}}),
      bundle());
  CHECK(off.digests == base.digests);
  CHECK(has(off.warnings, "cameras.left.imu.rate_hz is set but inert"));
  // On, only that camera's output digest moves: no earlier stage, no other camera.
  const config::Resolved one = config::resolve(run_inputs({}, {{"right", {{"imu.enabled", leaf(true)}}}}), bundle());
  CHECK(one.digests[0] == base.digests[0]);
  for (int s = 0; s < 4; ++s) CHECK(digests_of(one, 1)[static_cast<size_t>(s)] == digests_of(base, 1)[static_cast<size_t>(s)]);
  CHECK(digests_of(one, 1)[4] != digests_of(base, 1)[4]);
  // And the values of an enabled IMU enter it.
  const config::Resolved fast =
      config::resolve(run_inputs({}, {{"right", {{"imu.enabled", leaf(true)}, {"imu.rate_hz", leaf(2000.0)}}}}), bundle());
  for (int s = 0; s < 4; ++s) CHECK(digests_of(fast, 1)[static_cast<size_t>(s)] == digests_of(one, 1)[static_cast<size_t>(s)]);
  CHECK(digests_of(fast, 1)[4] != digests_of(one, 1)[4]);
  CHECK(fast.digests[0] == base.digests[0]);
}

TEST_CASE("validate() at resolve time: errors fail with their keys, warnings are collected, a disabled IMU is not checked") {
  const auto resolve_error = [](Map defaults) {
    return error_of([&] { config::resolve(run_inputs(std::move(defaults)), bundle()); });
  };
  config::Diagnostic d = resolve_error({{"imu.enabled", leaf(true)}, {"imu.noise_mode", str("shaped")}});
  CHECK(d.kind == DiagKind::kCrossKey);
  CHECK(d.key == "imu.noise_mode");
  CHECK(d.camera == "left");
  CHECK(d.layer == "config");
  CHECK(std::find(d.keys.begin(), d.keys.end(), "imu.gyro.filter") != d.keys.end());
  CHECK(d.message.rfind("cameras.left: ", 0) == 0);
  // A transform that is not rigid.
  d = resolve_error({{"imu.enabled", leaf(true)},
                     {"imu.T_cam_imu", leaf(std::vector<double>{1, 1.1e-5, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1})}});
  CHECK(d.kind == DiagKind::kCrossKey);
  CHECK(d.key == "imu.T_cam_imu");
  // The DLPF's delay below its time constant names both keys.
  d = resolve_error({{"imu.enabled", leaf(true)},
                     {"imu.gyro.filter", str("first_order")},
                     {"imu.gyro.group_delay_s", leaf(1e-4)}});
  CHECK(d.key == "imu.gyro.group_delay_s");
  CHECK(std::find(d.keys.begin(), d.keys.end(), "imu.gyro.cutoff_hz") != d.keys.end());
  // The same combination with the IMU off is not checked.
  const config::Resolved off = config::resolve(
      run_inputs({{"imu.noise_mode", str("shaped")}, {"imu.T_cam_imu", leaf(std::vector<double>(16, 0.0))}}), bundle());
  CHECK(off.digests.size() == 2);
  // Warnings: a cutoff above Nyquist, a gravity outside [9.7, 9.9].
  const config::Resolved w = config::resolve(
      run_inputs({{"imu.enabled", leaf(true)},
                  {"imu.gyro.filter", str("first_order")},
                  {"imu.gyro.cutoff_hz", leaf(600.0)},
                  {"imu.gravity_mps2", leaf(std::vector<double>{0, 0, -1.6})}},
                 {{"right", {{"imu.enabled", leaf(false)}}}}),
      bundle());
  const config::Diagnostic* nyquist = diag_of(w.diagnostics, "Nyquist");
  REQUIRE(nyquist != nullptr);
  CHECK(nyquist->kind == DiagKind::kCrossKey);
  CHECK(nyquist->key == "imu.gyro.cutoff_hz");
  CHECK(nyquist->camera == "left");
  CHECK(std::find(nyquist->keys.begin(), nyquist->keys.end(), "imu.rate_hz") != nyquist->keys.end());
  CHECK(has(w.warnings, "imu.gravity_mps2 has |g|"));
  REQUIRE(w.warnings.size() == w.diagnostics.size());
}

TEST_CASE("validate() at resolve time: a gravity that differs between enabled cameras is a warning naming them") {
  const auto warnings = [](std::vector<std::pair<std::string, Map>> blocks, Map defaults = {}) {
    defaults.emplace_back("imu.enabled", leaf(true));
    return config::resolve(run_inputs(std::move(defaults), std::move(blocks)), bundle()).warnings;
  };
  CHECK(!has(warnings({}), "differs between"));  // both null: one value
  const Map g_down{{"imu.gravity_mps2", leaf(std::vector<double>{0, 0, -9.81})}};
  CHECK(!has(warnings({}, g_down), "differs between"));  // the same explicit value on both
  const std::vector<std::string> w = warnings({{"right", {{"imu.gravity_mps2", leaf(std::vector<double>{0, 0, 9.81})}}}}, g_down);
  REQUIRE(has(w, "imu.gravity_mps2 differs between cameras.left ([0, 0, -9.81]) and cameras.right ([0, 0, 9.81])"));
  // Null against an explicit value differs too (the explicit one is not the bundle's by construction).
  CHECK(has(warnings({{"right", g_down}}), "differs between cameras.left (null) and cameras.right ([0, 0, -9.81])"));
  // A disabled camera is not compared.
  const config::Resolved r = config::resolve(
      run_inputs({}, {{"left", {{"imu.enabled", leaf(true)}}},
                      {"right", {{"imu.gravity_mps2", leaf(std::vector<double>{1, 2, 3})}}}}),
      bundle());
  CHECK(!has(r.warnings, "differs between"));
}

// ---------------------------------------------------------------------------------------------------------------
// resolve_section
// ---------------------------------------------------------------------------------------------------------------

config::Node profile(const char* id, const char* mount, Map values) {
  Map entries;
  for (auto& [key, v] : values) {
    entries.emplace_back(key, mapping({{"value", std::move(v)}, {"status", str("measured")}, {"source", str("test")}}));
  }
  return mapping({{"format", str("ballasim.profile/1")}, {"id", str(id)}, {"mount", str(mount)},
                  {"description", str("test")}, {"values", mapping(std::move(entries))}});
}

config::Entry recorded(Value v, const char* layer = "config", const char* status = "user") {
  return {std::move(v), layer, status, ""};
}

// The run's two cameras: left recorded with an enabled IMU at 1026.45 Hz (a profile's value), right as an older run (no
// imu.* entries).
config::SectionInputs section() {
  config::SectionInputs in;
  in.cameras = {{"left", {{"imu.enabled", recorded(true, "profile:evimo2/x@v1", "measured")},
                          {"imu.rate_hz", recorded(1026.45, "profile:evimo2/x@v1", "measured")},
                          {"imu.gyro.lsb", recorded(1.3315805450396192e-4, "config")}},
                 std::string(64, 'a')},
                {"right", {}, std::string(64, 'c')}};
  return in;
}

const config::SectionEntry& entry(const config::SectionResult& r, size_t camera, std::string_view key) {
  for (const auto& [k, e] : r.cameras.at(camera).second) {
    if (k == key) return e;
  }
  FAIL("no entry " << key);
  return r.cameras.at(0).second.front().second;
}

std::string canon(const Value& v) {
  char buf[40];
  if (const auto* d = std::get_if<double>(&v)) {
    const auto r = std::to_chars(buf, buf + sizeof(buf), *d, std::chars_format::hex);
    return std::string(buf, r.ptr);
  }
  if (const auto* b = std::get_if<bool>(&v)) return *b ? "true" : "false";
  if (const auto* i = std::get_if<int64_t>(&v)) return std::to_string(*i);
  if (const auto* s = std::get_if<std::string>(&v)) return "\"" + *s + "\"";
  if (std::holds_alternative<std::monostate>(v)) return "null";
  std::string out = "[";
  const auto& l = std::get<std::vector<double>>(v);
  for (size_t i = 0; i < l.size(); ++i) out += (i ? "," : "") + canon(l[i]);
  return out + "]";
}

TEST_CASE("resolve_section: schema literals, then the recorded values, with their layers, status and source") {
  const config::SectionResult r = config::resolve_section(section());
  REQUIRE(r.cameras.size() == 2);
  REQUIRE(r.cameras[0].second.size() == 37);
  CHECK(r.cameras[0].first == "left");
  CHECK(r.cameras[1].first == "right");
  // Recorded values keep their layer and status; the older run's camera takes the schema literals.
  const config::SectionEntry& rate = entry(r, 0, "imu.rate_hz");
  CHECK(std::get<double>(rate.value) == 1026.45);
  CHECK(rate.layer == "profile:evimo2/x@v1");
  CHECK(rate.status == "measured");
  CHECK(rate.source == "run.json (profile:evimo2/x@v1)");
  CHECK(std::get<double>(entry(r, 0, "imu.gyro.lsb").value) == 1.3315805450396192e-4);
  const config::SectionEntry& def = entry(r, 0, "imu.accel.cutoff_hz");  // not recorded: the row's default
  CHECK(std::get<double>(def.value) == 99.0);
  CHECK(def.layer == "default");
  CHECK(def.status == "prior");
  CHECK(def.source.find("RM-MPU-9250A-00") != std::string::npos);
  CHECK(std::get<bool>(entry(r, 1, "imu.enabled").value) == false);
  CHECK(std::get<double>(entry(r, 1, "imu.rate_hz").value) == 1000.0);
  // Keys in schema order.
  size_t i = 0;
  for (const config::Param& p : config::schema()) {
    if (p.key.rfind("imu.", 0) == 0) CHECK(r.cameras[0].second[i++].first == p.key);
  }
  CHECK(r.warnings.empty());
  // imu_digest = sha256(sensor digest + "\nimu\n" + the canonical key=value lines).
  std::string text = std::string(64, 'a') + "\nimu\n";
  for (const auto& [k, e] : r.cameras[0].second) text += k + "=" + canon(e.value) + "\n";
  REQUIRE(r.imu_digests.size() == 2);
  CHECK(r.imu_digests[0] == config::sha256_hex(text));
  CHECK(r.imu_digests[0] != r.imu_digests[1]);
  CHECK(r.rotation_change == std::vector<double>{0.0, 0.0});
}

TEST_CASE("resolve_section: root profiles, camera profiles and --set over the recorded values, lowest first") {
  config::SectionInputs in = section();
  in.profiles = {"evimo2/root@v1"};
  in.camera_profiles = {{"left", {"evimo2/cam@v1"}}};
  in.profile_docs.emplace("evimo2/root@v1",
                          profile("evimo2/root@v1", "root",
                                  {{"cameras.default.imu.rate_hz", leaf(1000.0)},
                                   {"cameras.left.imu.gyro.noise_density", leaf(1e-4)},
                                   {"cameras.right.imu.accel.noise_density", leaf(1.2e-3)},
                                   {"cameras.default.imu.gyro.bias", leaf(std::vector<double>{1e-3, 0, 0})}}));
  in.profile_docs.emplace("evimo2/cam@v1", profile("evimo2/cam@v1", "camera",
                                                   {{"imu.rate_hz", leaf(2000.0)}, {"imu.stamp_offset_ns", leaf(int64_t{250})}}));
  in.sets = mapping({{"cameras.left.imu.stamp_offset_ns", leaf(int64_t{-5})}, {"cameras.default.imu.seed", leaf(int64_t{7})}});
  const config::SectionResult r = config::resolve_section(in);
  // left: the camera profile beats the root's and the recorded rate; --set beats the camera profile.
  CHECK(std::get<double>(entry(r, 0, "imu.rate_hz").value) == 2000.0);
  CHECK(entry(r, 0, "imu.rate_hz").layer == "profile:evimo2/cam@v1");
  CHECK(entry(r, 0, "imu.rate_hz").source == "profile evimo2/cam@v1: test");
  CHECK(entry(r, 0, "imu.rate_hz").status == "measured");
  CHECK(std::get<int64_t>(entry(r, 0, "imu.stamp_offset_ns").value) == -5);
  CHECK(entry(r, 0, "imu.stamp_offset_ns").layer == "cli");
  CHECK(entry(r, 0, "imu.stamp_offset_ns").status == "user");
  CHECK(entry(r, 0, "imu.stamp_offset_ns").source == "--set");
  CHECK(std::get<double>(entry(r, 0, "imu.gyro.noise_density").value) == 1e-4);  // the root profile's left block
  CHECK(entry(r, 0, "imu.gyro.noise_density").layer == "profile:evimo2/root@v1");
  // The default block of a root profile reaches both cameras; the camera profile only its camera.
  CHECK(std::get<std::vector<double>>(entry(r, 1, "imu.gyro.bias").value) == std::vector<double>{1e-3, 0, 0});
  CHECK(std::get<double>(entry(r, 1, "imu.rate_hz").value) == 1000.0);
  CHECK(entry(r, 1, "imu.rate_hz").layer == "profile:evimo2/root@v1");
  CHECK(std::get<double>(entry(r, 1, "imu.accel.noise_density").value) == 1.2e-3);
  CHECK(std::get<int64_t>(entry(r, 1, "imu.stamp_offset_ns").value) == 0);  // not the left camera's profile or set
  CHECK(std::get<int64_t>(entry(r, 0, "imu.seed").value) == 7);
  CHECK(std::get<int64_t>(entry(r, 1, "imu.seed").value) == 7);
  // A value that did not change keeps the recorded layer.
  CHECK(entry(r, 0, "imu.gyro.lsb").layer == "config");
  // The digest follows the values: a different --set gives another digest, the same inputs the same one.
  const config::SectionResult again = config::resolve_section(in);
  CHECK(again.imu_digests == r.imu_digests);
  in.sets = mapping({{"cameras.left.imu.stamp_offset_ns", leaf(int64_t{-6})}, {"cameras.default.imu.seed", leaf(int64_t{7})}});
  const config::SectionResult other = config::resolve_section(in);
  CHECK(other.imu_digests[0] != r.imu_digests[0]);
  CHECK(other.imu_digests[1] == r.imu_digests[1]);
}

TEST_CASE("resolve_section: refusals") {
  const auto refusal = [](const std::function<void(config::SectionInputs&)>& edit) {
    config::SectionInputs in = section();
    in.profile_docs.emplace("evimo2/root@v1", profile("evimo2/root@v1", "root", {{"cameras.default.imu.rate_hz", leaf(500.0)}}));
    in.profile_docs.emplace("evimo2/cam@v1", profile("evimo2/cam@v1", "camera", {{"imu.rate_hz", leaf(500.0)}}));
    in.profile_docs.emplace("evimo2/run@v1", profile("evimo2/run@v1", "root", {{"run.device", str("cpu")}}));
    in.profile_docs.emplace("evimo2/sens@v1", profile("evimo2/sens@v1", "camera", {{"sensor.k", leaf(1e17)}}));
    edit(in);
    return error_of([&] { config::resolve_section(in); });
  };
  // Keys outside imu.*: a --set of a sensor key or a run key, a root profile's run key, a camera profile's sensor key.
  config::Diagnostic d = refusal([](config::SectionInputs& in) {
    in.sets = mapping({{"cameras.left.sensor.k", leaf(1e17)}});
  });
  CHECK(d.kind == DiagKind::kWrongScope);
  CHECK(d.message == "ballasim imu changes imu.* keys only; re-run for others: cli sets cameras.left.sensor.k");
  d = refusal([](config::SectionInputs& in) { in.sets = mapping({{"run.device", str("cpu")}}); });
  CHECK(d.message == "ballasim imu changes imu.* keys only; re-run for others: cli sets run.device");
  d = refusal([](config::SectionInputs& in) { in.profiles = {"evimo2/run@v1"}; });
  CHECK(d.message.find("imu.* keys only") != std::string::npos);
  d = refusal([](config::SectionInputs& in) { in.camera_profiles = {{"left", {"evimo2/sens@v1"}}}; });
  CHECK(d.message.find("imu.* keys only") != std::string::npos);
  CHECK(d.message.find("profile evimo2/sens@v1") != std::string::npos);
  // Mounts: a camera-mounted profile as a root profile and the reverse.
  d = refusal([](config::SectionInputs& in) { in.profiles = {"evimo2/cam@v1"}; });
  CHECK(d.kind == DiagKind::kProfile);
  CHECK(d.message.find("camera-mounted profile listed in the root profiles") != std::string::npos);
  d = refusal([](config::SectionInputs& in) { in.camera_profiles = {{"left", {"evimo2/root@v1"}}}; });
  CHECK(d.kind == DiagKind::kProfile);
  CHECK(d.message.find("root-mounted profile listed in a camera block's profiles") != std::string::npos);
  // Unknown cameras and profiles.
  d = refusal([](config::SectionInputs& in) { in.sets = mapping({{"cameras.rigth.imu.seed", leaf(int64_t{2})}}); });
  CHECK(d.kind == DiagKind::kUnknownCamera);
  CHECK(d.message == "cameras.rigth: not a camera of the run (left, right)");
  d = refusal([](config::SectionInputs& in) { in.camera_profiles = {{"middle", {"evimo2/cam@v1"}}}; });
  CHECK(d.kind == DiagKind::kUnknownCamera);
  d = refusal([](config::SectionInputs& in) { in.profiles = {"evimo2/missing@v1"}; });
  CHECK(d.kind == DiagKind::kProfile);
  d = refusal([](config::SectionInputs& in) { in.profiles = {"not an id"}; });
  CHECK(d.kind == DiagKind::kProfile);
  // Types and hard ranges, as the full resolver.
  d = refusal([](config::SectionInputs& in) { in.sets = mapping({{"cameras.left.imu.rate_hz", leaf(0.0)}}); });
  CHECK(d.kind == DiagKind::kHardRange);
  CHECK(d.key == "imu.rate_hz");
  d = refusal([](config::SectionInputs& in) { in.sets = mapping({{"cameras.left.imu.rate_hz", str("fast")}}); });
  CHECK(d.kind == DiagKind::kType);
  d = refusal([](config::SectionInputs& in) { in.sets = mapping({{"cameras.left.imu.rate_hzz", leaf(5.0)}}); });
  CHECK(d.kind == DiagKind::kUnknownKey);
  CHECK(d.nearest == "imu.rate_hz");
  // A recorded entry this version does not know, or of the wrong type, is the run's.
  d = refusal([](config::SectionInputs& in) { in.cameras[0].recorded.emplace_back("imu.bogus", recorded(1.0)); });
  CHECK(d.kind == DiagKind::kUnknownKey);
  CHECK(d.message.find("run.json cameras.left.imu.bogus") != std::string::npos);
  d = refusal([](config::SectionInputs& in) { in.cameras[0].recorded.emplace_back("sensor.k", recorded(1.0)); });
  CHECK(d.kind == DiagKind::kUnknownKey);
  // imu::validate on an enabled camera.
  d = refusal([](config::SectionInputs& in) { in.sets = mapping({{"cameras.left.imu.noise_mode", str("shaped")}}); });
  CHECK(d.kind == DiagKind::kCrossKey);
  CHECK(d.key == "imu.noise_mode");
  CHECK(d.camera == "left");
  CHECK(d.layer == "cli");
}

TEST_CASE("resolve_section: warnings: soft ranges, inert values, other cameras' blocks, validate, the gravity") {
  config::SectionInputs in = section();
  in.profiles = {"evimo2/root@v1"};
  in.profile_docs.emplace("evimo2/root@v1", profile("evimo2/root@v1", "root",
                                                    {{"cameras.middle.imu.seed", leaf(int64_t{5})},
                                                     {"cameras.default.imu.rate_hz", leaf(20.0)}}));
  // right stays disabled: --set of its rate is inert (a user layer), left's profile-set rate 20 Hz is below the soft range.
  in.sets = mapping({{"cameras.right.imu.gyro.noise_density", leaf(1e-4)},
                     {"cameras.left.imu.gyro.filter", str("first_order")},
                     {"cameras.left.imu.gyro.cutoff_hz", leaf(600.0)},
                     {"cameras.left.imu.rate_hz", leaf(900.0)}});
  config::SectionResult r = config::resolve_section(in);
  CHECK(has(r.warnings, "profile evimo2/root@v1: cameras.middle is not a camera of the run (left, right); its values are unused"));
  CHECK(has(r.warnings, "cameras.right.imu.gyro.noise_density is set but inert: it applies when imu.enabled is true"));
  CHECK(has(r.warnings, "cameras.right.imu.rate_hz = 20 (profile evimo2/root@v1): below the physical range 50-8000"));
  CHECK(!has(r.warnings, "cameras.left.imu.rate_hz = 900"));  // inside the range, set by the user on an enabled IMU
  CHECK(has(r.warnings, "cameras.left: imu.gyro.cutoff_hz = 600 is at or above the Nyquist rate"));
  REQUIRE(r.warnings.size() == r.diagnostics.size());
  const config::Diagnostic* inert = diag_of(r.diagnostics, "is set but inert");
  REQUIRE(inert != nullptr);
  CHECK(inert->kind == DiagKind::kInert);
  CHECK(inert->camera == "right");
  CHECK(inert->layer == "cli");
  // The gravity: left enabled with one value, right enabled with another.
  in = section();
  in.sets = mapping({{"cameras.right.imu.enabled", leaf(true)},
                     {"cameras.left.imu.gravity_mps2", leaf(std::vector<double>{0, 0, -9.81})}});
  r = config::resolve_section(in);
  CHECK(has(r.warnings, "imu.gravity_mps2 differs between cameras.left ([0, 0, -9.81]) and cameras.right (null)"));
  // The re-orthonormalization of an accepted, not exactly rigid T_cam_imu is reported.
  in = section();
  in.sets = mapping({{"cameras.left.imu.T_cam_imu",
                      leaf(std::vector<double>{1, 5e-7, 0, 0.1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1})}});
  r = config::resolve_section(in);
  REQUIRE(r.rotation_change.size() == 2);
  CHECK(r.rotation_change[0] > 1e-8);
  CHECK(r.rotation_change[0] < 1e-6);
  CHECK(r.rotation_change[1] == 0.0);  // right is disabled: not checked
}

}  // namespace
