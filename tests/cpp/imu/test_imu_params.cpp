// U13 (docs/design/phase3/outputs.md §8.1): the struct part (ImuParams <-> values, every key read exactly once, the
// hard ranges and every §6 validate() refusal and warning; rotation_of's polar re-orthonormalization) and, since A3,
// the schema part: the 37 rows of param_fields.h are kKeys' keys, and from_values(schema defaults) == ImuParams{}.
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "config/schema.h"
#include "imu/params.h"

using namespace balla;
using namespace balla::imu;

namespace {

bool contains(const std::vector<std::string>& msgs, const std::string& part) {
  for (const std::string& m : msgs) {
    if (m.find(part) != std::string::npos) return true;
  }
  return false;
}

std::string refusal(const Values& v) {
  try {
    (void)from_values(v);
  } catch (const Refused& e) {
    return e.what();
  }
  return "";
}

// A struct whose every field differs from its default and from every other field of its type.
ImuParams distinct() {
  ImuParams p;
  p.enabled = true;
  p.rate_hz = 1026.45;
  p.start_phase = StartPhase::kZero;
  p.clock_adev = 4.5e-5;
  p.T_cam_imu = {-1, 0, 0, -1.3e-3, 0, -1, 0, -0.5e-3, 0, 0, 1, -40.9e-3, 0, 0, 0, 1};
  p.gravity_mps2 = std::array<double, 3>{0.1, 0.2, -9.81};
  p.stamp_offset_ns = -1234;
  p.noise_mode = NoiseMode::kShaped;
  p.clip_stage = ClipStage::kBoth;
  p.adc_bits = 12;
  p.seed = 77;
  double x = 1e-6;
  for (SensorErrors* e : {&p.gyro, &p.accel}) {
    e->noise_density = x *= 1.1;
    e->random_walk = x *= 1.1;
    e->bias_instability = x *= 1.1;
    e->bias_instability_tau_s = x *= 1.1;
    for (auto* a : {&e->bias, &e->bias_sd, &e->scale, &e->nonorth, &e->misalign_deg}) {
      for (double& c : *a) c = x *= 1.1;
    }
    e->filter = Filter::kFirstOrder;
    e->cutoff_hz = x *= 1.1;
    e->group_delay_s = x *= 1.1;
    e->lsb = x *= 1.1;
  }
  return p;
}

}  // namespace

TEST_CASE("U13: ImuParams round-trips through its values, one key per field") {
  const Values defaults = default_values();
  CHECK(defaults.size() == 37);
  CHECK(kKeys.size() == 37);
  std::set<std::string> names;
  for (size_t i = 0; i < kKeys.size(); ++i) names.insert(std::string(key_name(i)));
  CHECK(names.size() == 37);
  for (const auto& [k, v] : defaults) CHECK(names.count(k) == 1);
  CHECK(from_values(defaults) == ImuParams{});
  CHECK(defaults.at("imu.gyro.cutoff_hz") == config::Value{92.0});
  CHECK(defaults.at("imu.accel.group_delay_s") == config::Value{2.88e-3});
  CHECK(defaults.at("imu.start_phase") == config::Value{std::string("random")});
  CHECK(std::holds_alternative<std::monostate>(defaults.at("imu.gyro.lsb")));
  CHECK(std::holds_alternative<std::monostate>(defaults.at("imu.gravity_mps2")));

  const ImuParams d = distinct();
  const Values dv = to_values(d);
  CHECK(from_values(dv) == d);
  // Every key carries a value of its own: none equals its default, so no field is read twice or skipped.
  for (const auto& [k, v] : dv) {
    INFO(k);
    CHECK(v != defaults.at(k));
  }
  // Changing one key changes exactly that key's field: the round trip of the changed values gives them back.
  for (const auto& [k, v] : dv) {
    INFO(k);
    Values one = defaults;
    one[k] = v;
    const ImuParams p = from_values(one);
    CHECK(p != ImuParams{});
    CHECK(to_values(p) == one);
  }
}

TEST_CASE("U13: from_values refuses missing, unknown and mistyped keys and values outside the hard ranges") {
  const Values defaults = default_values();
  for (const auto& [k, v] : defaults) {
    Values missing = defaults;
    missing.erase(k);
    CHECK(refusal(missing) == "imu: missing key " + k);
  }
  Values extra = defaults;
  extra["imu.gyro.cutof_hz"] = 92.0;
  CHECK(refusal(extra) == "imu: unknown key imu.gyro.cutof_hz");

  const auto with = [&](const char* key, config::Value v) {
    Values out = defaults;
    out[key] = std::move(v);
    return refusal(out);
  };
  CHECK(with("imu.rate_hz", int64_t{500}).empty());  // an integer is a float
  CHECK(with("imu.T_cam_imu", std::vector<int64_t>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).empty());
  CHECK(with("imu.enabled", int64_t{1}).find("expected true or false") != std::string::npos);
  CHECK(with("imu.rate_hz", true).find("imu.rate_hz") != std::string::npos);
  CHECK(with("imu.adc_bits", 16.0).find("expected an integer") != std::string::npos);
  CHECK(with("imu.start_phase", std::string("uniform")).find("one of random, zero") != std::string::npos);
  CHECK(with("imu.gyro.filter", std::string("butterworth")).find("one of none, first_order") != std::string::npos);
  CHECK(with("imu.gyro.bias", std::vector<double>{0, 0}).find("expected 3 values") != std::string::npos);
  CHECK(with("imu.T_cam_imu", std::vector<double>(9, 0.0)).find("expected 16 values") != std::string::npos);
  CHECK(with("imu.rate_hz", std::monostate{}).find("a value is required") != std::string::npos);
  CHECK(with("imu.gyro.lsb", 1e-4).empty());
  CHECK(with("imu.gravity_mps2", std::vector<double>{0, 0, -9.81}).empty());
  // Hard ranges and finiteness.
  CHECK(with("imu.rate_hz", 0.0) == "imu.rate_hz = 0: outside (0, 1e+05]");
  CHECK(with("imu.rate_hz", 1e5).empty());
  CHECK(!with("imu.rate_hz", std::nan("")).empty());
  CHECK(!with("imu.clock_adev", 1e-2).empty());
  CHECK(!with("imu.adc_bits", int64_t{7}).empty());
  CHECK(with("imu.adc_bits", int64_t{32}).empty());
  CHECK(!with("imu.seed", int64_t{-1}).empty());
  CHECK(!with("imu.stamp_offset_ns", int64_t{10'000'000}).empty());
  CHECK(!with("imu.gyro.scale", std::vector<double>{0, 0.5, 0}).empty());
  CHECK(!with("imu.accel.nonorth", std::vector<double>{0, 0, -0.1}).empty());
  CHECK(!with("imu.accel.misalign_deg", std::vector<double>{10, 0, 0}).empty());
  CHECK(!with("imu.gyro.bias_sd", std::vector<double>{0, -1e-3, 0}).empty());
  CHECK(!with("imu.gyro.bias", std::vector<double>{0, std::numeric_limits<double>::infinity(), 0}).empty());
  CHECK(!with("imu.gyro.cutoff_hz", 0.0).empty());
  CHECK(!with("imu.accel.lsb", 0.0).empty());
  CHECK(!with("imu.gyro.bias_instability_tau_s", 0.0).empty());
}

TEST_CASE("U13: validate() states every section 6 refusal and warning") {
  CHECK(validate(ImuParams{}).errors.empty());
  CHECK(validate(ImuParams{}).warnings.empty());

  // T_cam_imu rigidity: |(R^T R - I)_ij| = 0.9e-6 accepted, 1.1e-6 refused; the last row exactly (0, 0, 0, 1).
  const auto rigid = [](std::array<double, 16> t) { return validate([&] { ImuParams p; p.T_cam_imu = t; return p; }()); };
  const std::array<double, 16> eye = ImuParams{}.T_cam_imu;
  {
    auto t = eye;
    t[0] = std::sqrt(1 + 0.9e-6);  // (R^T R)_00 = 1 + 0.9e-6
    CHECK(rigid(t).errors.empty());
    t[0] = std::sqrt(1 + 1.1e-6);
    CHECK(contains(rigid(t).errors, "imu.T_cam_imu rotation is not orthonormal"));
  }
  {
    auto t = eye;  // off-diagonal (R^T R)_01 = R_00 R_01 = 0.9e-6 and 1.1e-6
    t[1] = 0.9e-6;
    CHECK(rigid(t).errors.empty());
    t[1] = 1.1e-6;
    CHECK(contains(rigid(t).errors, "not orthonormal"));
  }
  {
    auto t = eye;
    t[15] = 1 + 1e-12;
    CHECK(contains(rigid(t).errors, "last row is not exactly (0, 0, 0, 1)"));
    t = eye;
    t[12] = 1e-300;
    CHECK(contains(rigid(t).errors, "last row"));
    t = eye;
    t[5] = std::nan("");
    CHECK(contains(rigid(t).errors, "non-finite"));
    t = eye;
    t[10] = -1;  // a reflection: orthonormal, det -1
    CHECK(contains(rigid(t).errors, "det R - 1"));
  }

  ImuParams p;
  p.noise_mode = NoiseMode::kShaped;
  p.accel.filter = Filter::kFirstOrder;
  CHECK(contains(validate(p).errors, "shaped noise is white noise through the DLPF: set imu.gyro.filter=first_order"));
  CHECK(!contains(validate(p).errors, "imu.accel.filter"));
  p.gyro.filter = Filter::kFirstOrder;
  CHECK(validate(p).errors.empty());

  // First-order filters: the delay, the cutoff against the rate; nothing is checked for filter none.
  p = ImuParams{};
  p.gyro.filter = Filter::kFirstOrder;
  p.gyro.group_delay_s = tau_c_s(p.gyro);
  CHECK(validate(p).errors.empty());  // d = 0
  p.gyro.group_delay_s = std::nextafter(tau_c_s(p.gyro), 0.0);
  CHECK(contains(validate(p).errors, "imu.gyro.group_delay_s"));
  p = ImuParams{};
  p.rate_hz = 90;
  CHECK(validate(p).errors.empty());  // both filters none: cutoff 92 and 99 are inert
  CHECK(validate(p).warnings.empty());
  p.gyro.filter = Filter::kFirstOrder;
  CHECK(contains(validate(p).errors, "imu.gyro.cutoff_hz = 92 is at or above imu.rate_hz = 90"));
  p.rate_hz = 92;
  CHECK(contains(validate(p).errors, "imu.gyro.cutoff_hz"));
  p.rate_hz = 100;
  CHECK(validate(p).errors.empty());
  CHECK(contains(validate(p).warnings, "imu.gyro.cutoff_hz = 92 is at or above the Nyquist rate"));
  p.rate_hz = 184;
  CHECK(contains(validate(p).warnings, "Nyquist"));
  p.rate_hz = 184.5;
  CHECK(validate(p).warnings.empty());

  // clip_stage both needs an lsb; one null is inert for that sensor.
  p = ImuParams{};
  p.clip_stage = ClipStage::kBoth;
  CHECK(contains(validate(p).errors, "imu.clip_stage both needs imu.gyro.lsb or imu.accel.lsb"));
  p.gyro.lsb = 1.3315805450396192e-4;
  CHECK(validate(p).errors.empty());
  CHECK(contains(validate(p).warnings, "the pre-clip is inert for imu.accel"));
  p.accel.lsb = 5.987548828125e-4;
  CHECK(validate(p).warnings.empty());

  // |gravity| outside [9.7, 9.9] warns (a zero-g scene is allowed).
  p = ImuParams{};
  p.gravity_mps2 = std::array<double, 3>{0, 0, -9.81};
  CHECK(validate(p).warnings.empty());
  p.gravity_mps2 = std::array<double, 3>{0, 0, 0};
  CHECK(validate(p).errors.empty());
  CHECK(contains(validate(p).warnings, "imu.gravity_mps2 has |g| = 0"));

  // Hard ranges hold for a struct built in C++ too.
  p = ImuParams{};
  p.rate_hz = -1;
  p.adc_bits = 40;
  const Check c = validate(p);
  CHECK(contains(c.errors, "imu.rate_hz"));
  CHECK(contains(c.errors, "imu.adc_bits"));
}

TEST_CASE("rotation_of: polar re-orthonormalization keeps exact rotations and repairs near ones") {
  const auto rot = [](const std::array<double, 16>& t) {
    ImuParams p;
    p.T_cam_imu = t;
    return rotation_of(p);
  };
  const Mat3d eye = rot(ImuParams{}.T_cam_imu);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) CHECK(eye.m[i][j] == (i == j ? 1.0 : 0.0));
  }
  const Mat3d flip = rot({-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});  // EVIMO2's diag(-1, -1, 1)
  CHECK(flip.m[0][0] == -1.0);
  CHECK(flip.m[1][1] == -1.0);
  CHECK(flip.m[2][2] == 1.0);
  CHECK(flip.m[0][1] == 0.0);
  // A rotation about z by 0.3 rad with 1e-7 of skew and scale: the polar factor is orthonormal to ulps and
  // within the perturbation of the input.
  const double c = std::cos(0.3), s = std::sin(0.3);
  const std::array<double, 16> t = {c * (1 + 1e-7), -s + 1e-7, 0, 0.01, s, c, 2e-7, 0, 0, 0, 1 - 1e-7, 0, 0, 0, 0, 1};
  const Mat3d r = rot(t);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double g = (r.m[0][i] * r.m[0][j] + r.m[1][i] * r.m[1][j]) + r.m[2][i] * r.m[2][j];
      CHECK(std::abs(g - (i == j ? 1.0 : 0.0)) < 4e-16);
      CHECK(std::abs(r.m[i][j] - t[4 * i + j]) < 3e-7);
    }
  }
  const Vec3d p = translation_of([&] { ImuParams q; q.T_cam_imu = t; return q; }());
  CHECK(p.x == 0.01);
  CHECK(p.y == 0.0);
}

TEST_CASE("U13: the schema's imu rows are kKeys' keys in order; from_values(schema defaults) == ImuParams{}") {
  std::vector<const config::Param*> rows;
  for (const config::Param& p : config::schema()) {
    if (p.key.rfind("imu.", 0) == 0) rows.push_back(&p);
  }
  REQUIRE(rows.size() == kKeys.size());
  Values defaults;
  std::set<std::string> logs, with_filter;
  for (size_t i = 0; i < rows.size(); ++i) {
    const config::Param& p = *rows[i];
    INFO(p.key);
    CHECK(p.key == key_name(i));
    // One camera's IMU, stage output, no target: no IMU key invalidates a render or a sensor result.
    CHECK(p.scope == config::Scope::kCamera);
    CHECK(p.stage == config::Stage::kOutput);
    CHECK(p.target == config::Target::kNone);
    CHECK(p.def == config::Default::kLiteral);
    CHECK(p.cls == (p.key == "imu.seed" ? config::Class::kTRun : config::Class::kTSensor));
    // The type, shape and hard range are kKeys' (the single source of from_values).
    CHECK(p.type == kKeys[i].type);
    CHECK(p.length == kKeys[i].length);
    CHECK(p.nullable == kKeys[i].nullable);
    CHECK(p.hard.lo == kKeys[i].hard.lo);
    CHECK(p.hard.hi == kKeys[i].hard.hi);
    CHECK(p.hard.lo_open == kKeys[i].hard.lo_open);
    CHECK(p.hard.hi_open == kKeys[i].hard.hi_open);
    // Relevant only while the IMU is on; the DLPF's two keys only while that sensor's filter is first_order.
    if (p.key == "imu.enabled") {
      CHECK(p.when.empty());
    } else {
      REQUIRE(!p.when.empty());
      CHECK(p.when[0].key == "imu.enabled");
      CHECK(p.when[0].value == config::Value{true});
      if (p.when.size() > 1) {
        with_filter.insert(p.key);
        CHECK(p.when.size() == 2);
        CHECK(p.when[1].key == p.key.substr(0, p.key.rfind('.')) + ".filter");
        CHECK(p.when[1].value == config::Value{std::string("first_order")});
      }
    }
    if (p.scale == config::Scale::kLog) logs.insert(p.key);
    // The default is ImuParams{}'s, the status says how it was chosen: the MPU-9250 DLPF_CFG 2 priors, else choices.
    const bool prior = p.key.ends_with(".cutoff_hz") || p.key.ends_with(".group_delay_s");
    REQUIRE(p.status.has_value());
    CHECK(*p.status == (prior ? config::Status::kPrior : config::Status::kChoice));
    if (prior) CHECK(p.source.find("RM-MPU-9250A-00 r1.6") != std::string::npos);
    // Numbers carry a unit; the soft range is the rate's alone.
    CHECK((p.type == config::Type::kBool || p.type == config::Type::kEnum) == p.unit.empty());
    CHECK(p.soft.unset() == (p.key != "imu.rate_hz"));
    defaults.emplace(p.key, p.value);
  }
  CHECK(from_values(defaults) == ImuParams{});
  CHECK(to_values(ImuParams{}) == defaults);
  CHECK(logs == std::set<std::string>{"imu.rate_hz", "imu.clock_adev", "imu.gyro.noise_density", "imu.gyro.random_walk",
                                      "imu.gyro.bias_instability", "imu.gyro.bias_instability_tau_s", "imu.gyro.cutoff_hz",
                                      "imu.gyro.lsb", "imu.accel.noise_density", "imu.accel.random_walk",
                                      "imu.accel.bias_instability", "imu.accel.bias_instability_tau_s",
                                      "imu.accel.cutoff_hz", "imu.accel.lsb"});
  CHECK(with_filter == std::set<std::string>{"imu.gyro.cutoff_hz", "imu.gyro.group_delay_s", "imu.accel.cutoff_hz",
                                             "imu.accel.group_delay_s"});
  const config::Param& rate = *config::find("imu.rate_hz");
  CHECK(rate.soft.lo == 50);
  CHECK(rate.soft.hi == 8000);
  // The rows' defaults are checked by the hard range, and a value the row refuses is the key's refusal.
  CHECK_THROWS_AS(config::check_value(rate, 0.0, "imu.rate_hz"), std::invalid_argument);
}
