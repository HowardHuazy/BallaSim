#include "sensor/params.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <variant>

#include "config/diagnostics.h"
#include "sensor/param_fields.h"

namespace balla::sensor {

namespace {

// A cross-key rule (ui.md §3.10, UI-4a): its message as before, thrown as a config::DiagnosticError (a
// std::invalid_argument) naming the schema keys of the members it reads, so that a form marks every one.
[[noreturn]] void refuse(std::string what, std::initializer_list<const char*> fields) {
  config::Diagnostic d;
  d.kind = config::DiagKind::kCrossKey;
  d.message = "SensorParams: " + std::move(what);
  for (const char* f : fields) {
    const config::Param* r = param_row(f);
    d.keys.push_back(r != nullptr ? r->key : "sensor." + std::string(f));
  }
  d.key = d.keys.front();
  throw config::DiagnosticError(std::move(d));
}

void require(bool ok, const char* what, std::initializer_list<const char*> fields) {
  if (!ok) refuse(what, fields);
}

}  // namespace

void SensorParams::validate() const {
  // D2: no family, no prior (a default-constructed SensorParams).
  if (find_family(family) == nullptr) {
    std::string known;
    for (const Family& f : families()) known += (known.empty() ? "" : ", ") + f.name;
    throw std::invalid_argument("SensorParams: family = '" + family + "': one of " + known +
                                (family.empty() ? " (SensorParams() carries no prior: use preset(name) or a resolved "
                                                  "configuration)"
                                                : ""));
  }
  // Single-key rules: every number finite and inside its row's hard range (enums are valid by type).
  for (const config::Param& r : param_rows()) {
    if (r.type == config::Type::kFloat || r.type == config::Type::kInt) {
      config::check_value(r, r.get(this), "SensorParams: " + config::display_name(r));
    }
  }
  // Cross-key rules.
  // Phase 1 solves stages 2-3 with distinct poles, in either order; near-confluent poles would put
  // cancelling amplitudes ~ tau / |tau_sf - tau_o| into the fp32 comparator (docs/dev/SENSOR.md).
  require(std::max(tau_o, tau_sf) >= 1.1 * std::min(tau_o, tau_sf),
          "tau_sf and tau_o must differ by at least 10 % (the larger >= 1.1 x the smaller)", {"tau_sf", "tau_o"});
  // Output-node noise needs a resolvable pole after its injection point (M3 §A.3).
  require(!(noise && sigma_o > 0) || tau_o >= 1e-6, "sigma_o > 0 requires tau_o >= 1 us",
          {"sigma_o", "tau_o", "noise"});
  require(theta_on > theta_floor, "theta_on must exceed theta_floor", {"theta_on", "theta_floor"});
  require(theta_off > theta_floor, "theta_off must exceed theta_floor", {"theta_off", "theta_floor"});
  // A release must never fire immediately (DESIGN §8.4).
  require(std::fabs(eps_ci) < 0.5 * theta_floor, "|eps_ci| must be < theta_floor / 2", {"eps_ci", "theta_floor"});
  require(hot_rate_max_hz >= hot_rate_min_hz, "hot rates must satisfy 0 < hot_rate_min_hz <= hot_rate_max_hz",
          {"hot_rate_max_hz", "hot_rate_min_hz"});
  // No pixel can fire faster than once per refractory period.
  require(!hot_pixels() || hot_rate_max_hz * refractory_s < 1.0, "hot_rate_max_hz must be < 1 / refractory_s",
          {"hot_rate_max_hz", "refractory_s", "hot_fraction"});
  // The floor puts t_reported - t_cross in (d0 - q, d0]; causality needs d0 >= q (DESIGN §8.7).
  require(readout_quantum_s == 0 || readout_delay_s >= readout_quantum_s,
          "readout_delay_s must be >= readout_quantum_s when quantization is on",
          {"readout_delay_s", "readout_quantum_s"});
  // The high-pass filter (DESIGN §8.4 A.14): a rate well below the cascade's poles, and a reference node to filter.
  if (hpf_on()) {
    require(tau_hp >= hpf_floor_s(),
            "tau_hp must be 0 (off) or >= max(1 ms, 10 max(tau_o, tau_sf)) (SENSOR.md \"High-pass filter\")",
            {"tau_hp", "tau_o", "tau_sf"});
    require(photoreceptor == Photoreceptor::kCascade,
            "tau_hp > 0 needs photoreceptor cascade (the instantaneous response has no filtered reference)",
            {"tau_hp", "photoreceptor"});
  }
  // The rate limiter (sensor/limiter.h): a falling cap, a bucket that holds a saturated bin's refill, and a reset time
  // inside the epoch. Its rows are inert while it is off.
  if (limiter_on()) {
    require(limiter_rate1_hz <= limiter_rate0_hz,
            "limiter_rate1_hz must be <= limiter_rate0_hz (the cap falls over the epoch)",
            {"limiter_rate1_hz", "limiter_rate0_hz"});
    if (limiter_depth < limiter_min_depth()) {
      refuse("limiter_depth = " + std::to_string(limiter_depth) + " must be >= ceil(limiter_rate0_hz x " +
                 "max(readout_quantum_s, 1 us)) + 1 = " + std::to_string(limiter_min_depth()) +
                 " (a shallower bucket clips the refill of saturated bins: the output falls below the cap)",
             {"limiter_depth", "limiter_rate0_hz", "readout_quantum_s"});
    }
    require(!limiter_phase_s || *limiter_phase_s < limiter_period_s,
            "limiter_phase_s must lie in [0, limiter_period_s) (a reset time mod the epoch)",
            {"limiter_phase_s", "limiter_period_s"});
  }
}

double SensorParams::hpf_floor_s() const { return std::max(1e-3, 10.0 * std::max(tau_o, tau_sf)); }

int64_t SensorParams::limiter_min_depth() const {
  // In integer ns, so that 6 Mev/s x 1 us is 6 exactly; the 1e-9 forgives a product's last bit (a clip of < 1e-9
  // events per bin).
  const auto q_bin_ns = static_cast<double>(std::max<int64_t>(std::llround(readout_quantum_s * 1e9), 1000));
  return static_cast<int64_t>(std::ceil(limiter_rate0_hz * q_bin_ns / 1e9 - 1e-9)) + 1;
}

SensorParams preset(const std::string& name) {
  if (!find_family(name)) {
    std::string known;
    for (const Family& f : families()) known += (known.empty() ? "" : ", ") + f.name;
    throw std::invalid_argument("unknown sensor preset '" + name + "' (known: " + known + ")");
  }
  SensorParams p;
  for (const config::Param& r : param_rows()) {
    if (r.def == config::Default::kFamily) {
      const Prior* v = find_prior(name, r.key);
      if (!v) throw std::logic_error("sensor family " + name + " has no prior for " + r.key);
      r.set(r, &p, v->value);
    } else if (r.def == config::Default::kLiteral) {
      r.set(r, &p, r.value);
    }
  }
  p.family = name;
  p.illuminance_scale = 1.0;  // the low-level API's neutral scale; runs set their own (required)
  p.clock_offset_ns = 0;
  return p;
}

std::vector<std::string> preset_names() {
  std::vector<std::string> out;
  for (const Family& f : families()) out.push_back(f.name);
  return out;
}

Waveform make_constant_waveform() { return Waveform{}; }

Waveform make_fourier_waveform(double f0_hz, const std::vector<double>& a, const std::vector<double>& phi) {
  if (!(std::isfinite(f0_hz) && f0_hz > 0)) throw std::invalid_argument("fourier waveform: f0_hz must be positive");
  if (a.size() != phi.size()) throw std::invalid_argument("fourier waveform: a and phi need the same length");
  if (a.size() > static_cast<size_t>(kMaxHarmonics)) {
    throw std::invalid_argument("fourier waveform: at most 8 harmonics");
  }
  double sum_abs = 0;
  Waveform w;
  w.type = WaveformType::kFourier;
  w.f0_hz = f0_hz;
  w.n_harmonics = static_cast<int32_t>(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(phi[i])) {
      throw std::invalid_argument("fourier waveform: non-finite coefficient");
    }
    w.a[i] = a[i];
    w.phi[i] = phi[i];
    sum_abs += std::fabs(a[i]);
  }
  // A negative emitter power is unphysical; m >= 0 everywhere iff sum |a_k| <= 1 is sufficient.
  if (sum_abs > 1.0) throw std::invalid_argument("fourier waveform: sum |a_k| must be <= 1 so that m >= 0");
  return w;
}

Waveform make_pulse_waveform(double f_hz, double duty, double rise_s, double phase_s, double floor_level) {
  if (!(std::isfinite(f_hz) && f_hz > 0)) throw std::invalid_argument("pulse waveform: f_hz must be positive");
  const double period = 1.0 / f_hz;
  if (!(std::isfinite(duty) && duty > 0 && duty <= 1)) {
    throw std::invalid_argument("pulse waveform: duty must lie in (0, 1]");
  }
  if (!(std::isfinite(rise_s) && rise_s >= 1e-9)) throw std::invalid_argument("pulse waveform: rise_s must be >= 1 ns");
  const double top = duty * period - rise_s;
  if (!(top >= 0 && 2 * rise_s + top <= period)) {
    throw std::invalid_argument("pulse waveform: need rise_s <= duty / f and duty / f + rise_s <= 1 / f");
  }
  if (!(std::isfinite(floor_level) && floor_level >= 0 && floor_level < 1)) {
    throw std::invalid_argument("pulse waveform: floor must lie in [0, 1)");
  }
  if (!std::isfinite(phase_s)) throw std::invalid_argument("pulse waveform: phase_s must be finite");
  Waveform w;
  w.type = WaveformType::kPulse;
  w.period_ns = period * 1e9;
  w.phase_ns = phase_s * 1e9;
  w.rise_ns = rise_s * 1e9;
  w.top_ns = top * 1e9;
  w.floor_level = floor_level;
  w.peak_level = floor_level + (1.0 - floor_level) / duty;  // mean-normalized (M3 V7 X3)
  return w;
}

void validate_waveform(const Waveform& w) {
  switch (w.type) {
    case WaveformType::kConstant: return;
    case WaveformType::kFourier:
      if (w.n_harmonics < 0 || w.n_harmonics > kMaxHarmonics) {
        throw std::invalid_argument("fourier waveform: n_harmonics must lie in [0, 8]");
      }
      make_fourier_waveform(w.f0_hz, {w.a, w.a + w.n_harmonics}, {w.phi, w.phi + w.n_harmonics});
      return;
    case WaveformType::kPulse:
      // Periods of 1 ns to 2^61 ns and phases within 2^61 ns keep the period index and the snapped corners inside
      // int64 for |t| < 2^62 ns (each snapped period non-empty); 2^-40 absorbs the factory's s-to-ns rounding.
      if (!(w.period_ns >= 1 && w.period_ns <= 0x1p61 && std::fabs(w.phase_ns) <= 0x1p61 && w.rise_ns >= 0 &&
            w.top_ns >= 0 && 2 * w.rise_ns + w.top_ns <= w.period_ns * (1 + 0x1p-40))) {
        throw std::invalid_argument("pulse waveform: need period_ns in [1, 2^61], |phase_ns| <= 2^61, rise_ns >= 0, "
                                    "top_ns >= 0 and 2 rise_ns + top_ns <= period_ns");
      }
      if (!(w.floor_level >= 0 && w.floor_level < 1 && std::isfinite(w.peak_level) && w.peak_level >= w.floor_level)) {
        throw std::invalid_argument("pulse waveform: need floor_level in [0, 1) and a finite peak_level >= floor_level");
      }
      return;
  }
  throw std::invalid_argument("waveform: unknown type");
}

namespace {

struct Fnv1a {
  uint64_t h = 0xcbf29ce484222325ull;
  void bytes(const void* p, size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) {
      h ^= b[i];
      h *= 0x100000001b3ull;
    }
  }
  template <class T>
  void value(const T& v) {
    bytes(&v, sizeof(v));
  }
  void add(std::monostate) {}
  void add(bool b) { value(b); }
  void add(int64_t i) { value(i); }
  void add(double d) { value(d); }
  void add(const std::string& s) {
    value(s.size());
    bytes(s.data(), s.size());
  }
  template <class T>
  void add(const std::vector<T>& l) {
    value(l.size());
    for (const T& x : l) add(x);
  }
};

}  // namespace

uint64_t fingerprint(const SensorParams& p, const std::vector<Waveform>& groups) {
  Fnv1a f;
  for (const config::Param& r : param_rows()) {
    if (!p.hpf_on() && (r.key == "sensor.tau_hp" || r.key == "sensor.tau_hp_sigma_ln")) continue;  // as before them
    if (!p.limiter_on() && r.key.starts_with("sensor.limiter")) continue;                        // likewise
    const config::Value v = r.get(&p);
    f.value(v.index());
    std::visit([&](const auto& x) { f.add(x); }, v);
  }
  for (const Waveform& w : groups) {
    f.value(w.type);
    f.value(w.n_harmonics);
    f.value(w.f0_hz);
    for (int i = 0; i < kMaxHarmonics; ++i) {
      f.value(w.a[i]);
      f.value(w.phi[i]);
    }
    for (double v : {w.period_ns, w.phase_ns, w.rise_ns, w.top_ns, w.floor_level, w.peak_level}) f.value(v);
  }
  return f.h;
}

}  // namespace balla::sensor
