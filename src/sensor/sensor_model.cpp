#include "sensor/sensor_model.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/rng/purpose.h"
#include "core/time/time.h"
#include "sensor/noise.h"
#if BALLASIM_SENSOR_CUDA
#include "sensor/sensor_cuda.h"
#endif

namespace balla::sensor {

namespace {

constexpr double kElementaryCharge = 1.602176634e-19;
constexpr size_t kPixelBlock = 64;

constexpr uint32_t kStateMagic = 0x534e5342;  // "BSNS"
// 2: the fingerprint covers every parameter row (P0a); 3: and sensor.photoreceptor (P0d); 4: and the initial-state
// rows (I6'); 5: and accuracy.noise_diffusion, and PixelState::rho_m1 takes four of the reserved bytes (PHASE2
// D-P2-2; 0 under frozen diffusion, so frozen states keep their bytes). The size is unchanged. A gain map (D1)
// enters the fingerprint without a bump (state_fingerprint). 6: the high-pass filter's section (one HpfState per
// pixel) follows the pixel states while its kernels run (filter on, or the b = 0 test hook); without it a state differs
// from version 5's in the version alone (the fingerprint skips the filter's rows while it is off), and version 5 loads
// as filter-off.
constexpr uint32_t kStateVersion = 6;
constexpr uint32_t kStateVersionNoHpf = 5;

struct StateHeader {
  uint32_t magic, version;
  uint32_t width, height;
  uint32_t groups, camera;
  uint64_t seed;
  uint64_t fingerprint;
  int64_t t_state;
  uint32_t initialized;
  uint32_t state_size;
};

static_assert(sizeof(PixelState) == 200, "PixelState layout is part of the saved-state format");
static_assert(sizeof(HpfState) == 16, "HpfState layout is part of the saved-state format");

// A saved state's fingerprint: the parameters and waveforms, and with a gain map (set_photometry: each pixel's input
// and state depend on its gain) the gains, FNV-1a; without one (gain 1 everywhere) the parameters' alone, so that the
// states of relative photometry stay as they were.
uint64_t state_fingerprint(const SensorParams& p, const std::vector<Waveform>& groups,
                           const std::vector<PixelMap>& pixels) {
  uint64_t h = fingerprint(p, groups);
  if (std::all_of(pixels.begin(), pixels.end(), [](const PixelMap& m) { return m.gain == 1.0f; })) return h;
  for (const PixelMap& m : pixels) {
    unsigned char b[sizeof(float)];
    std::memcpy(b, &m.gain, sizeof(b));
    for (const unsigned char c : b) h = (h ^ c) * 0x100000001b3ull;
  }
  return h;
}

struct VectorSink {
  std::vector<RawEvent>* out;
  void push(const RawEvent& e) { out->push_back(e); }
};

struct NullSink {
  void push(const RawEvent&) {}
};

// The stationary transfer's features of a pixel, in units of their tolerance: levels 0.01 log-e (0.2 sigma_v),
// photocurrent 10 %, modulated share 0.1; and the noise sigma_v(I).
struct Feature {
  double on, off, ln_i, share, sigma;
};

// One snapshot of a probe's comparator (relative to the snapshot's time).
struct Snap {
  int64_t blind_until, t_search;
  double c;  // the offset without s3's current value
};

constexpr int kSnapshots = 64;
constexpr uint32_t kProbeMagic = 0x52505342;  // "BSPR"
constexpr uint32_t kProbeVersion = 1;

struct ProbeHeader {
  uint32_t magic, version;
  uint32_t precision, hpf;
  uint32_t probes, snapshots;
  uint64_t key;        // the inputs' key (SensorModel::probe_key)
  int64_t t0, history_ns;
  double ms;           // the probe's wall time
};

// FNV-1a 64 over bytes (the probe's input key).
struct Hash64 {
  uint64_t h = 0xcbf29ce484222325ull;
  void bytes(const void* p, size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
  }
  template <class T>
  void value(const T& v) {
    bytes(&v, sizeof(v));
  }
};

template <class T>
Feature pixel_feature(const KernelConstants<T>& kc, const SensorParams& params, const std::vector<Waveform>& groups,
                      const NoiseModel& nm, const PixelMap& m, const PixelState& s) {
  const double I = s.i_scale;  // the mean photocurrent (fA) init set
  double mod = 0, veil = 0;
  for (size_t g = 0; g < groups.size(); ++g) {
    if (groups[g].type != WaveformType::kConstant) {
      mod += static_cast<double>(s.held[g]);
      if (kc.glare.on) veil += static_cast<double>(kc.glare.a[g]);
    }
  }
  mod *= params.illuminance_scale * static_cast<double>(m.gain);
  if (kc.glare.on) mod += params.illuminance_scale * veil;  // the glare of modulated groups is modulated too
  double cov[3][3];
  noise_stationary_cov(nm, params.k * I * 1e-15, cov);
  return {m.theta_on / 0.01, m.theta_off / 0.01, pmath::log(I) / 0.1, mod / I / 0.1, std::sqrt(cov[2][2])};
}

NoiseModel probe_noise_model(const SensorParams& p) {
  NoiseModel nm{};
  nm.kq = p.k * kElementaryCharge;
  nm.a2 = 1.0 / p.tau_o;
  nm.a3 = 1.0 / p.tau_sf;
  nm.sigma_o = p.sigma_o;
  return nm;
}

// The glare ramp of init and the stationary probe's history: constant at g times the frame means at init.
template <class T>
GlareRamp<T> init_ramp(double g, const std::vector<double>& means, int64_t t0) {
  GlareRamp<T> r{};
  r.on = g > 0;
  r.t_a = r.t_b = t0;
  for (size_t k = 0; r.on && k < means.size(); ++k) r.a[k] = r.b[k] = static_cast<T>(g * means[k]);
  return r;
}

}  // namespace

// The stationary probe (stationary_init's first part), and the content of a ProbeResult.
struct SensorModel::ProbeData {
  std::vector<uint32_t> id;           // the probe pixels (frame ids), one per stratum
  std::vector<Feature> f;             // their features
  std::vector<double> theta_on, theta_off;  // their levels
  std::vector<PixelState> ps;         // their states at t0, re-based (F034)
  std::vector<HpfState> ph;           // and their filter states (kHpf)
  std::vector<Snap> snap;             // kSnapshots per probe
  int64_t history_ns = 0;
  double ms = 0;
};

int64_t readout_phase_ns(uint64_t seed, uint32_t camera, int64_t quantum_ns) {
  if (quantum_ns <= 0) return 0;
  // Per-camera phase of the free-running microsecond counter (M3 F.1).
  return static_cast<int64_t>(rng::draw(seed, rng::Purpose::kReadoutPhase, camera, 0, 0).v[0] %
                              static_cast<uint64_t>(quantum_ns));
}

Readout readout_of(const SensorParams& p, uint64_t seed, uint32_t camera) {
  const int64_t q_ns = std::llround(p.readout_quantum_s * 1e9);
  return {q_ns, std::llround(p.readout_delay_s * 1e9), readout_phase_ns(seed, camera, q_ns), p.clock_offset_ns,
          p.clock_drift_ppm};
}

bool sensor_cuda_available() {
#if BALLASIM_SENSOR_CUDA
  return sensor_cuda_device_count() > 0;
#else
  return false;
#endif
}

#if !BALLASIM_SENSOR_CUDA
// Without CUDA the backend type stays incomplete; the unique_ptr still needs a deleter.
class CudaSensorKernel {};
#endif

SensorModel::SensorModel(const SensorParams& params, uint32_t width, uint32_t height, uint64_t seed,
                         std::vector<Waveform> light_groups, uint32_t camera, int num_threads,
                         Precision precision, Device device, std::optional<Roi> roi)
    : params_(params), width_(width), height_(height), seed_(seed), camera_(camera),
      groups_(std::move(light_groups)), precision_(precision), device_(device), roi_(roi) {
  if (device_ == Device::kCuda) {
    if (precision_ != Precision::kFp32) throw std::invalid_argument("SensorModel: the CUDA device runs fp32 only");
    if (!sensor_cuda_available()) {
      throw std::runtime_error("SensorModel: no CUDA device available (or built without CUDA)");
    }
  }
  params_.validate();
  if (width == 0 || height == 0 || width > 65535 || height > 65535) {
    throw std::invalid_argument("SensorModel: width and height must lie in [1, 65535]");
  }
  if (camera > 255) throw std::invalid_argument("SensorModel: camera index must be < 256 (purpose.h layout)");
  if (groups_.empty() || groups_.size() > static_cast<size_t>(kMaxLightGroups)) {
    throw std::invalid_argument("SensorModel: need 1 to 8 light groups");
  }
  for (const Waveform& w : groups_) validate_waveform(w);
  pool_ = std::make_unique<ThreadPool>(num_threads);
  uint32_t n_pixels = width * height;
  if (roi_) {
    const Roi& r = *roi_;
    if (!(r.x0 >= 0 && r.x0 < r.x1 && r.x1 <= static_cast<int64_t>(width) && r.y0 >= 0 && r.y0 < r.y1 &&
          r.y1 <= static_cast<int64_t>(height))) {
      throw std::invalid_argument("SensorModel: roi must satisfy 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height");
    }
#if BALLA_SENSOR_CENSUS
    throw std::invalid_argument("SensorModel: a census build has no region-of-interest models");
#endif
    pixel_ids_.reserve(static_cast<size_t>(r.x1 - r.x0) * static_cast<size_t>(r.y1 - r.y0));
    for (int32_t y = r.y0; y < r.y1; ++y) {
      for (int32_t x = r.x0; x < r.x1; ++x) pixel_ids_.push_back(static_cast<uint32_t>(y) * width + x);
    }
    n_pixels = static_cast<uint32_t>(pixel_ids_.size());
    maps_ = draw_sensor_maps(params_, seed_, camera_, pixel_ids_);
  } else {
    maps_ = draw_sensor_maps(params_, seed_, camera_, n_pixels);
  }
#if BALLA_SENSOR_CENSUS
  census_px_.assign(n_pixels, CensusPixel{});
#endif
  // The kernel needs k * 1 ns * I^2 normal in its precision (try_substep's xi0; also k h I > 0 in z_step and a finite
  // 1 / I at init), and the dark current bounds every photocurrent I below: refuse pixel dark currents under that floor
  // (an underflow there gave a storm of quarantines and solver fallbacks).
  const bool fp32 = precision_ == Precision::kFp32;
  const double floor_fA = std::sqrt((fp32 ? double{FLT_MIN} : DBL_MIN) / (params_.k * 1e-15 * 1e-9));
  double lowest = std::numeric_limits<double>::infinity();
  for (const PixelMap& m : maps_.pixels) lowest = std::min(lowest, m.dark_fA);
  if (roi_) {
    // An ROI model refuses what the full model refuses (its stationary probe also initializes the whole frame): the
    // floor over every pixel of the frame, from the dark draws alone.
    std::vector<double> low(pool_->size(), lowest);
    pool_->parallel_for(static_cast<size_t>(width) * height, 1 << 14, [&](int w, size_t b, size_t e) {
      const double l = lowest_dark_fA(params_, seed_, camera_, static_cast<uint32_t>(b), static_cast<uint32_t>(e));
      low[w] = std::min(low[w], l);
    });
    for (const double l : low) lowest = std::min(lowest, l);
  }
  if (!(lowest >= floor_fA)) {
    char msg[256];
    std::snprintf(msg, sizeof msg,
                  "SensorModel: sensor.dark_current_fA at sensor.temperature_K gives a pixel dark current of %g fA, "
                  "below the %s kernel's floor of %g fA%s",
                  lowest, fp32 ? "fp32" : "fp64", floor_fA, fp32 ? " (use precision fp64 on the CPU)" : "");
    throw std::invalid_argument(msg);
  }
  if (params_.noise_on()) {
    NoiseModel nm{};
    nm.kq = params_.k * kElementaryCharge;
    nm.a2 = 1.0 / params_.tau_o;
    nm.a3 = 1.0 / params_.tau_sf;
    nm.sigma_o = params_.sigma_o;
    nm.levels = params_.noise_leaf_level;
    nm.margin_z = params_.noise_margin_z;
    nm.split = params_.noise_diffusion == NoiseDiffusion::kTransient;
    tables_ = std::make_unique<NoiseTables>(nm);
  }
  states_.assign(n_pixels, PixelState{});
  if (params_.hpf_on()) hpf_.assign(n_pixels, HpfState{});
  NoiseTableView<double> v64{};
  NoiseTableView<float> v32{};
  if (tables_) {
    v64 = {tables_->data64().data(), tables_->n_nodes(), tables_->layout().stride(), params_.noise_leaf_level,
           tables_->ln_a1_min(), tables_->inv_dln()};
    v32 = {tables_->data32().data(), tables_->n_nodes(), tables_->layout().stride(), params_.noise_leaf_level,
           static_cast<float>(tables_->ln_a1_min()), static_cast<float>(tables_->inv_dln())};
  }
  kc64_ = make_constants<double>(v64);
  kc32_ = make_constants<float>(v32);
  readout_ = readout_of(params_, seed_, camera_);
  readout_phase_ns_ = readout_.phase_ns;
#if BALLASIM_SENSOR_CUDA
  if (device_ == Device::kCuda) {
    cuda_ = std::make_unique<CudaSensorKernel>(kc32_, maps_.pixels, tables_ ? tables_->data32() : std::vector<float>{});
    if (kc32_.hpf) cuda_->enable_hpf(kc32_, maps_.hpf_b);
    if (roi_) cuda_->set_pixel_ids(pixel_ids_);
    cuda_->upload_states(states_, hpf_);
  }
#endif
}

int64_t SensorModel::local_index(uint32_t id) const {
  if (!roi_) return id < states_.size() ? static_cast<int64_t>(id) : -1;
  const Roi& r = *roi_;
  const int64_t x = id % width_, y = id / width_;
  if (x < r.x0 || x >= r.x1 || y < r.y0 || y >= r.y1) return -1;
  return (y - r.y0) * (r.x1 - r.x0) + (x - r.x0);
}

void SensorModel::set_frame_means(std::vector<double> means) {
  if (!roi_) throw std::logic_error("SensorModel.set_frame_means: a full model computes its own frame means");
  if (means.size() % groups_.size() != 0 || means.empty()) {
    throw std::invalid_argument("SensorModel.set_frame_means: (calls + 1) x groups values");
  }
  for (const double v : means) {
    if (!std::isfinite(v) || v < 0) throw std::invalid_argument("SensorModel.set_frame_means: finite means >= 0");
  }
  given_means_ = std::move(means);
}

void SensorModel::set_track_frame_means(bool on) {
  if (roi_) throw std::logic_error("SensorModel.set_track_frame_means: an ROI model takes its frame means as given");
  track_means_ = on;
}

std::vector<PixelState> SensorModel::states() const {
  std::vector<PixelState> out = states_;
#if BALLASIM_SENSOR_CUDA
  if (cuda_) cuda_->download_states(out);
#endif
  return out;
}

std::vector<HpfState> SensorModel::hpf_states() const {
  std::vector<HpfState> out = hpf_;
#if BALLASIM_SENSOR_CUDA
  if (cuda_ && !out.empty()) cuda_->download_hpf(out);
#endif
  return out;
}

void SensorModel::force_hpf_kernel() {
  if (params_.hpf_on()) throw std::logic_error("SensorModel.force_hpf_kernel: the filter is on");
  if (kc32_.hpf) return;
  hpf_test_ = true;
  hpf_.assign(states_.size(), HpfState{});
  kc64_ = make_constants<double>(kc64_.tables);
  kc32_ = make_constants<float>(kc32_.tables);
  initialized_ = false;  // the side states start at the next init
#if BALLASIM_SENSOR_CUDA
  if (cuda_) cuda_->enable_hpf(kc32_, maps_.hpf_b);
#endif
}

SensorModel::~SensorModel() = default;

void SensorModel::set_photometry(double illuminance_scale, const std::vector<float>& gain) {
  if (!gain.empty() && gain.size() != static_cast<size_t>(width_) * height_) {
    throw std::invalid_argument("SensorModel::set_photometry: one gain per pixel of the frame (or none)");
  }
  SensorParams p = params_;
  p.illuminance_scale = illuminance_scale;
  p.validate();
  double g_max = gain.empty() ? 1.0 : 0.0;
  for (const float g : gain) {
    if (!(g > 0.0f) || !std::isfinite(g)) throw std::invalid_argument("SensorModel::set_photometry: gains must be > 0");
    g_max = std::max(g_max, static_cast<double>(g));
  }
  params_ = p;
  gain_max_ = g_max;
  for (size_t i = 0; i < maps_.pixels.size(); ++i) {
    maps_.pixels[i].gain = gain.empty() ? 1.0f : gain[roi_ ? pixel_ids_[i] : i];
  }
  if (roi_) frame_gain_ = gain;
  kc64_ = make_constants<double>(kc64_.tables);
  kc32_ = make_constants<float>(kc32_.tables);
#if BALLASIM_SENSOR_CUDA
  if (cuda_) cuda_->set_photometry(kc32_, maps_.pixels);
#endif
}

template <class T>
CascadeConstants<T> cascade_constants(const SensorParams& p) {
  const double eps_h = p.eps_L / 8.0;
  CascadeConstants<T> cc{};
  cc.k = static_cast<T>(p.k * 1e-15);  // (fA s)^-1
  cc.a2 = static_cast<T>(1.0 / p.tau_o);
  cc.a3 = static_cast<T>(1.0 / p.tau_sf);
  cc.tau_o = static_cast<T>(p.tau_o);
  cc.tau_sf = static_cast<T>(p.tau_sf);
  cc.eps_accept = static_cast<T>(eps_h / 1.5);  // DESIGN §8.3: accept at eps_h / 1.5
  // Negligible terms: 2^-10 eps_h, but at least a few ulp of an O(1) log state in fp32.
  cc.eps_drop = static_cast<T>(std::max(eps_h / 1024.0, 32.0 * static_cast<double>(FloatTraits<T>::kEpsilon)));
  return cc;
}

template CascadeConstants<float> cascade_constants<float>(const SensorParams&);
template CascadeConstants<double> cascade_constants<double>(const SensorParams&);

template <class T>
KernelConstants<T> SensorModel::make_constants(const NoiseTableView<T>& view) const {
  KernelConstants<T> kc{};
  kc.cc = cascade_constants<T>(params_);
  kc.eps_pl_base = static_cast<T>(params_.eps_L / 8.0);
  kc.scale = static_cast<T>(params_.illuminance_scale);
  // One clamp per camera, at the brightest pixel's gain (1 without a gain map): no pixel's photocurrent passes the cap.
  kc.irr_max = static_cast<float>(kMaxSamplePhotocurrent_fA / (params_.illuminance_scale * gain_max_));
  kc.groups = static_cast<int32_t>(groups_.size());
  kc.modulated = false;
  kc.breakpoints = false;
  kc.wave_mask = 0;
  for (size_t g = 0; g < groups_.size(); ++g) {
    kc.waves[g] = groups_[g];
    kc.modulated = kc.modulated || groups_[g].type != WaveformType::kConstant;
    kc.breakpoints = kc.breakpoints || groups_[g].type == WaveformType::kPulse;
    if (groups_[g].type != WaveformType::kConstant) kc.wave_mask |= 1u << g;
  }
  kc.table = {};
  kc.noise = params_.noise_on();
  kc.static_fast_path = true;
  kc.instant = params_.photoreceptor == Photoreceptor::kInstantaneous;
  kc.levels = params_.noise_leaf_level;
  kc.margin_z = static_cast<T>(params_.noise_margin_z);
  kc.tables = view;
  kc.transient = kc.noise && params_.noise_diffusion == NoiseDiffusion::kTransient;
  kc.ideal_reset = params_.reset_mode == ResetMode::kIdeal;
  kc.refractory_ns = std::llround(params_.refractory_s * 1e9);
  kc.eps_ci = static_cast<T>(params_.eps_ci);
  kc.label_cut = static_cast<T>(params_.label_share_cut);
  kc.seed = seed_;
  kc.camera = camera_;
  // The high-pass filter (hpf.h): on, or forced at b = 0 (force_hpf_kernel); per-pixel rates from the side map.
  kc.hpf = params_.hpf_on() || hpf_test_;
  kc.hpf_b = params_.hpf_on() ? static_cast<T>(1.0 / params_.tau_hp) : T(0);
  kc.hpf_b_map = maps_.hpf_b.empty() ? nullptr : maps_.hpf_b.data();
  kc.eps_hp = static_cast<T>(params_.eps_L / 32.0);
  kc.pixel_ids = roi_ ? pixel_ids_.data() : nullptr;
  return kc;
}

template <class T>
void SensorModel::init_states(const KernelConstants<T>& kc, const std::vector<PixelMap>& maps, int64_t t0,
                              const float* irr, std::vector<PixelState>& states, std::vector<HpfState>& hpf,
                              KernelStats* total) {
  const int G = groups();
  std::vector<KernelStats> stats(pool_->size());
#if BALLA_SENSOR_CENSUS
  for (KernelStats& s : stats) s.census.px = census_px_.data();
#endif
  // The glare-free instantiation without glare (pixel.h kGlare); the filter's with glare compiled in (kHpf).
  const auto run = [&](auto glare, auto filter) {
    pool_->parallel_for(states.size(), kPixelBlock, [&](int w, size_t begin, size_t end) {
      NullSink sink;
      const InputRun none{nullptr, nullptr, 0};
      for (size_t p = begin; p < end; ++p) {
        states[p] = PixelState{};
        HpfState* h = nullptr;
        if constexpr (decltype(filter)::value) h = &(hpf[p] = HpfState{});
        PixelKernel<T, NullSink, true, KernelStats, decltype(glare)::value, decltype(filter)::value> k(
            kc, maps[p], static_cast<uint32_t>(p), states[p], t0, none, sink, stats[w], -1, h);
        k.init(irr + p * G);
      }
    });
  };
  if (kc.hpf) run(std::true_type{}, std::true_type{});
  else if (kc.glare.on) run(std::true_type{}, std::false_type{});
  else run(std::false_type{}, std::false_type{});
  if (total != nullptr) {
    for (const KernelStats& s : stats) total->add(s);
  }
}

template <class T>
void SensorModel::run_init(const KernelConstants<T>& kc, int64_t t0, const float* irradiance0) {
  init_states(kc, maps_.pixels, t0, irradiance0, states_, hpf_, &stats_);
  stationary_ms_ = 0;
  stationary_probes_ = 0;
  stationary_history_ns_ = 0;
  stationary_phase_kept_ = 0;
}

// The probe's history T_b (accuracy.stationary_burnin_ns; 0 derives it): 1.5 / R_L, R_L the median leak rate at the
// run's temperature, clamped to [60, 600] s, in whole noise cells. The comparator's slow mode is the leak cycle
// (~0.9 / R_L); from the phase law the rate is within a few % of the long run after 0.25-0.4 / R_L at theta_off 0.25
// and 0.375 and R_L 0.002-0.02 (overshoot <= 6 % until ~0.65 / R_L), so the snapshots over the last 2/3 start after
// 0.5 / R_L and span about one cycle (ENGINE_CPP "I6p review resolution"). The high-pass filter forgets the
// reference's history in tau_hp (its leak cycle, where it still fires, is shorter than 1 / R_L): with it on, at most
// max(1 s, 10 tau_hp).
int64_t probe_history_ns(const SensorParams& p) {
  int64_t t = p.stationary_burnin_ns;
  if (t == 0) {
    const double r = p.leak_rate_hz * arrhenius(p.leak_ea_eV, p.temperature_K, p.reference_temperature_K);
    double s = r > 0 ? std::min(std::max(1.5 / r, 60.0), 600.0) : 600.0;
    if (p.hpf_on()) s = std::min(s, std::max(1.0, 10.0 * p.tau_hp));
    t = static_cast<int64_t>(s * 1e9);
  }
  return std::max<int64_t>(t / kNoiseCellNs, 1) * kNoiseCellNs;
}

// The stationary initial state (accuracy.initial_state = stationary; SENSOR.md "Initial state", ENGINE_CPP I6').
//
// The comparator is the slow part of a pixel: its offset c = Delta - n3 - (s3 - its mean) drifts with the leak and
// jumps at every reset, while the cascade and the noise forget in ~1 / (k I). Its stationary law depends on the
// pixel's own levels and leak (noise ping-pong where the window is narrow, leak-driven cycles of ~1 / R_L where it is
// wide) and is reached only after ~0.5 / R_L of history; so it is measured, by the kernel itself:
//   1. The probe: P pixels, one per stratum of equal count in ln(mean photocurrent) (the one with the smallest
//      kStationaryProbe key in its stratum), run their own history: init at t0 - T_b (probe_history_ns) with
//      the phase law and this frame held, advanced to t0 by this model's kernel on the host (its precision; events
//      dropped), with snapshots of the comparator at kSnapshots times over the last 2/3. Their noise is their own
//      lattice at those absolute times, so a probe pixel is at t0 what a run started T_b earlier makes of it, and
//      keeps all of it, its reference re-based at t0 (F034: PixelKernel::rebase_history).
//   2. Every other pixel (hot pixels keep the phase law, which is theirs: leak-driven, or relaxing in ms) takes the
//      comparator of its nearest probe in (theta_on, theta_off, ln I, modulated share) at one of the probe's snapshots
//      (a keyed draw: the pixels sharing a probe take different draws of its stationary law): c, moved to keep the
//      distance to the probe's nearer level in units of the noise sigma_v(I) (the ratio clamped to [1/2, 2]). Its
//      noise and cascade stay its own init, and so does its reference (its own s3 and n3 at t0, F034), whose offset
//      becomes Delta(t0) = c + n3(t0) + s3(t0). A draw that would put Delta(t0) on or past a level (a state the law
//      cannot hold: the pixel would have fired) takes the probe's next snapshot instead, cyclically; if none fits, the
//      pixel keeps the phase law.
// Without noise (or under the instantaneous response) the comparator is deterministic and init keeps the phase law.
bool SensorModel::stationary_on() const {
  return params_.initial_state == InitialState::kStationary && params_.noise_on() &&
         params_.photoreceptor != Photoreceptor::kInstantaneous;
}

// Part 1, the probe, over the full frame (maps and init states of every pixel, ids = indices).
template <class T>
SensorModel::ProbeData SensorModel::probe_of(const KernelConstants<T>& kc, const std::vector<PixelMap>& maps,
                                             const std::vector<PixelState>& states, int64_t t0,
                                             const float* irradiance0) {
  ProbeData d;
  const size_t n = states.size();
  const int G = groups();
  // pmath, not the host libm: ln I orders the strata, so it chooses the probes (determinism across platforms).
  const NoiseModel nm = probe_noise_model(params_);
  std::vector<Feature> f(n);
  pool_->parallel_for(n, 4096, [&](int, size_t b, size_t e) {
    for (size_t p = b; p < e; ++p) f[p] = pixel_feature(kc, params_, groups_, nm, maps[p], states[p]);
  });
  std::vector<uint32_t> eligible;
  for (size_t p = 0; p < n; ++p) {
    if (maps[p].hot == HotType::kNone) eligible.push_back(static_cast<uint32_t>(p));
  }
  if (eligible.empty()) return d;
  std::sort(eligible.begin(), eligible.end(),
            [&](uint32_t a, uint32_t b) { return f[a].ln_i != f[b].ln_i ? f[a].ln_i < f[b].ln_i : a < b; });
  const size_t m = eligible.size();
  const size_t P = std::min(m, static_cast<size_t>(params_.stationary_probe_pixels));
  const auto key = [&](uint32_t p, int i) {
    return rng::draw(seed_, rng::Purpose::kStationaryProbe, camera_, p, static_cast<uint64_t>(t0)).v[i];
  };
  std::vector<uint32_t> probe(P);
  for (size_t s = 0; s < P; ++s) {
    uint32_t best = eligible[s * m / P], best_key = UINT32_MAX;
    for (size_t i = s * m / P; i < (s + 1) * m / P; ++i) {
      const uint32_t k = key(eligible[i], 0);
      if (k < best_key) best = eligible[i], best_key = k;
    }
    probe[s] = best;
  }
  // The probe's histories: snapshot j at t0 - (kSnapshots - 1 - j) dt, dt whole noise cells over the last 2/3 of T_b
  // (the last at t0). A static pixel's law is the same at every time, so each snapshot is a draw of it.
  const int64_t T_b = probe_history_ns(params_);
  const int64_t dt = std::max<int64_t>(T_b * 2 / 3 / (kSnapshots - 1) / kNoiseCellNs, 1) * kNoiseCellNs;
  const int64_t t_p = t0 - std::max<int64_t>(T_b, (kSnapshots - 1) * dt);
  std::vector<PixelState>& ps = d.ps;
  std::vector<HpfState>& ph = d.ph;
  std::vector<Snap>& snap = d.snap;
  ps.assign(P, PixelState{});
  ph.assign(kc.hpf ? P : 0, HpfState{});  // the probes' filter states (pixel.h kHpf)
  snap.assign(P * kSnapshots, Snap{});
  // The glare-free instantiation without glare (pixel.h kGlare); the filter's with glare compiled in (kHpf).
  const auto histories = [&](auto glare, auto hpf) {
    using K = PixelKernel<T, NullSink, true, KernelStats, decltype(glare)::value, decltype(hpf)::value>;
    pool_->parallel_for(P, 1, [&](int, size_t b, size_t e) {
      NullSink sink;
      KernelStats st;
      const InputRun none{nullptr, nullptr, 0};
      for (size_t i = b; i < e; ++i) {
        const uint32_t q = probe[i];
        const PixelMap& mq = maps[q];
        HpfState* h = nullptr;
        if constexpr (decltype(hpf)::value) h = &(ph[i] = HpfState{});
        K(kc, mq, q, ps[i], t_p, none, sink, st, -1, h).init(irradiance0 + static_cast<size_t>(q) * G);
        int64_t t = t_p;
        for (int j = 0; j < kSnapshots; ++j) {
          const int64_t t_j = t0 - static_cast<int64_t>(kSnapshots - 1 - j) * dt;
          K(kc, mq, q, ps[i], t, none, sink, st, -1, h).advance(t_j);
          t = t_j;
          const PixelState& x = ps[i];
          // Under the high-pass filter Delta also loses the memory m (hpf.h).
          snap[i * kSnapshots + j] = {x.blind_until == kNotBlind ? kNotBlind : x.blind_until - t_j, x.t_search - t_j,
                                      -x.s3_ref - x.n3_ref + mq.leak * static_cast<double>(t_j - x.t_ref) * 1e-9 +
                                          x.dr_ref - (h != nullptr ? h->m : 0.0)};
        }
        K(kc, mq, q, ps[i], t0, none, sink, st, -1, h).rebase_history();  // F034: starts at t0 as init's states do
      }
    });
  };
  if (kc.hpf) histories(std::true_type{}, std::true_type{});
  else if (kc.glare.on) histories(std::true_type{}, std::false_type{});
  else histories(std::false_type{}, std::false_type{});
  d.id = probe;
  d.f.resize(P);
  d.theta_on.resize(P);
  d.theta_off.resize(P);
  for (size_t i = 0; i < P; ++i) {
    d.f[i] = f[probe[i]];
    d.theta_on[i] = maps[probe[i]].theta_on;
    d.theta_off[i] = maps[probe[i]].theta_off;
  }
  d.history_ns = T_b;
  return d;
}

// The probe from a fresh init of the whole frame (the model's state untouched): the model's own maps, or an ROI
// model's frame drawn by global id with the frame's gains and its own constants (identity ids, the frame's tau_hp map).
template <class T>
SensorModel::ProbeData SensorModel::probe_frame(const KernelConstants<T>& kc, int64_t t0, const float* frame) {
  const size_t n = static_cast<size_t>(width_) * height_;
  std::vector<PixelState> st(n);
  std::vector<HpfState> h(kc.hpf ? n : 0);
  if (!roi_) {
    init_states(kc, maps_.pixels, t0, frame, st, h, nullptr);
    return probe_of(kc, maps_.pixels, st, t0, frame);
  }
  SensorMaps full = draw_sensor_maps(params_, seed_, camera_, static_cast<uint32_t>(n));
  if (!frame_gain_.empty()) {
    for (size_t i = 0; i < n; ++i) full.pixels[i].gain = frame_gain_[i];
  }
  KernelConstants<T> kf = kc;
  kf.pixel_ids = nullptr;
  kf.hpf_b_map = full.hpf_b.empty() ? nullptr : full.hpf_b.data();
  init_states(kf, full.pixels, t0, frame, st, h, nullptr);
  return probe_of(kf, full.pixels, st, t0, frame);
}

// Part 2, the transfer to the model's own pixels: the nearest probe, a keyed snapshot of it (or the next that fits),
// re-timed to t0.
template <class T>
void SensorModel::transfer(const KernelConstants<T>& kc, const ProbeData& d, int64_t t0) {
  const size_t P = d.id.size();
  if (P == 0) return;  // no eligible pixel in the frame: the phase law stays
  const size_t n = states_.size();
  const NoiseModel nm = probe_noise_model(params_);
  const auto key = [&](uint32_t p, int i) {
    return rng::draw(seed_, rng::Purpose::kStationaryProbe, camera_, p, static_cast<uint64_t>(t0)).v[i];
  };
  std::vector<int32_t> probe_of_px(n, -1);
  for (size_t i = 0; i < P; ++i) {
    const int64_t l = local_index(d.id[i]);
    if (l >= 0) probe_of_px[static_cast<size_t>(l)] = static_cast<int32_t>(i);
  }
  std::vector<uint32_t> kept(pool_->size(), 0);
  pool_->parallel_for(n, 1024, [&](int w, size_t b, size_t e) {
    for (size_t p = b; p < e; ++p) {
      const PixelMap& mp = maps_.pixels[p];
      if (mp.hot != HotType::kNone) continue;  // hot pixels keep the phase law
      PixelState& s = states_[p];
      if (probe_of_px[p] >= 0) {
        const size_t i = static_cast<size_t>(probe_of_px[p]);
        s = d.ps[i];
        if (kc.hpf) hpf_[p] = d.ph[i];
        continue;
      }
      const uint32_t id = roi_ ? pixel_ids_[p] : static_cast<uint32_t>(p);
      const Feature a = pixel_feature(kc, params_, groups_, nm, mp, s);
      size_t j = 0;
      double best = INFINITY;
      for (size_t i = 0; i < P; ++i) {
        const Feature& o = d.f[i];
        const double dd = (a.on - o.on) * (a.on - o.on) + (a.off - o.off) * (a.off - o.off) +
                          (a.ln_i - o.ln_i) * (a.ln_i - o.ln_i) + (a.share - o.share) * (a.share - o.share);
        if (dd < best) best = dd, j = i;
      }
      const double r = std::min(std::max(a.sigma / d.f[j].sigma, 0.5), 2.0);
      const double n3_0 = s.n3_ref, s3_0 = s.s3;  // init's own n3(t0) and s3(t0): its reference, set at t0
      const int first = static_cast<int>((static_cast<uint64_t>(key(id, 1)) * kSnapshots) >> 32);
      bool placed = false;
      for (int k = 0; k < kSnapshots && !placed; ++k) {
        const Snap& q = d.snap[j * kSnapshots + (first + k) % kSnapshots];
        if (q.blind_until != kNotBlind) {  // in the refractory: the release sets the reference
          s.t_search = std::max(t0 + q.t_search, t0 + 1);
          s.blind_until = t0 + q.blind_until;
          placed = true;
          continue;
        }
        const double u_on = d.theta_on[j] - q.c, u_off = d.theta_off[j] + q.c;
        const double c = u_on <= u_off ? mp.theta_on - u_on * r : -mp.theta_off + u_off * r;
        const double delta = c + n3_0 + s3_0;
        if (!(delta < mp.theta_on && delta > -mp.theta_off)) continue;
        s.t_search = std::max(t0 + q.t_search, t0 + 1);
        s.dr_ref = delta;  // F034: the offset holds Delta(t0) whole, as phi0 does under the phase law
        s.blind_until = kNotBlind;
        placed = true;
      }
      if (!placed) ++kept[w];  // no snapshot fits: the phase law stays
    }
  });
  stationary_probes_ = P;
  stationary_history_ns_ = d.history_ns;
  stationary_phase_kept_ = 0;
  for (const uint32_t k : kept) stationary_phase_kept_ += k;
}

// The key of a probe's inputs: the parameters without the readout and the label cut (which leave the dynamics
// bit-identical), the waveforms, the frame size, seed, camera, precision, the gains of the frame, the frame means at
// init (with glare), t0 and the frame.
uint64_t SensorModel::probe_key(int64_t t0, const float* frame, const std::vector<double>& means) const {
  SensorParams p = params_;
  p.readout_delay_s = 0;
  p.readout_quantum_s = 0;
  p.clock_offset_ns = 0;
  p.clock_drift_ppm = 0;
  p.label_share_cut = 0.5;
  Hash64 h;
  h.value(fingerprint(p, groups_));
  h.value(width_);
  h.value(height_);
  h.value(seed_);
  h.value(camera_);
  h.value(precision_);
  h.value(hpf_test_);
  h.value(t0);
  const size_t n = static_cast<size_t>(width_) * height_;
  // The frame's gains, none when all are 1 (a full model's or an ROI model's map: the same key either way).
  std::vector<float> gains;
  if (roi_) {
    if (std::any_of(frame_gain_.begin(), frame_gain_.end(), [](float g) { return g != 1.0f; })) gains = frame_gain_;
  } else if (std::any_of(maps_.pixels.begin(), maps_.pixels.end(), [](const PixelMap& m) { return m.gain != 1.0f; })) {
    gains.resize(n);
    for (size_t i = 0; i < n; ++i) gains[i] = maps_.pixels[i].gain;
  }
  h.value(gains.size());
  h.bytes(gains.data(), gains.size() * sizeof(float));
  if (params_.veiling_glare > 0) h.bytes(means.data(), means.size() * sizeof(double));
  h.bytes(frame, n * groups_.size() * sizeof(float));
  return h.h;
}

std::vector<double> SensorModel::frame_mean(const float* frame) const {
  const size_t G = groups_.size(), n = static_cast<size_t>(width_) * height_;
  std::vector<double> m(G, 0.0);
  for (size_t p = 0; p < n; ++p) {
    // An ROI model's frame: the frame's gains (set_photometry), 1 without a gain map.
    const double gain = !roi_ ? maps_.pixels[p].gain : frame_gain_.empty() ? 1.0f : frame_gain_[p];
    for (size_t g = 0; g < G; ++g) {
      m[g] += gain * static_cast<double>(clamp_irradiance(frame[p * G + g], kc32_.irr_max));
    }
  }
  for (double& v : m) v /= static_cast<double>(n);
  return m;
}

std::vector<float> SensorModel::held_frame(const uint32_t* pixel_ids, size_t n, const float* irr,
                                           const int64_t* row_start, const int32_t* row_count) const {
  if (roi_) throw std::logic_error("SensorModel: an ROI model holds no frame (its frame means are given)");
  const size_t G = groups_.size();
  std::vector<float> frame;
#if BALLASIM_SENSOR_CUDA
  if (cuda_) frame = cuda_->end_frame(DeviceRuns{}, nullptr);
#endif
  if (frame.empty()) {
    frame.resize(states_.size() * G);
    for (size_t p = 0; p < states_.size(); ++p) std::copy_n(states_[p].held, G, frame.data() + p * G);
  }
  for (size_t i = 0; i < n; ++i) {
    if (row_count[i] > 0) {
      std::copy_n(irr + static_cast<size_t>(row_start[i] + row_count[i] - 1) * G, G, frame.data() + pixel_ids[i] * G);
    }
  }
  return frame;
}

void SensorModel::set_glare(int64_t t_end, const std::vector<double>& end) {
  GlareRamp<double> r{};
  GlareRamp<float> r32{};
  r.on = r32.on = params_.veiling_glare > 0;
  if (r.on && (frame_mean_.size() != groups_.size() || end.size() != groups_.size())) {
    throw std::logic_error("SensorModel: veiling glare without the frame means of the held inputs");
  }
  r.t_a = r32.t_a = t_state_;
  r.t_b = r32.t_b = t_end;
  for (size_t g = 0; r.on && g < groups_.size(); ++g) {
    const double a = params_.veiling_glare * frame_mean_[g];
    const double b = t_end <= t_state_ ? a : params_.veiling_glare * end[g];
    r.a[g] = a;
    r.b[g] = b;
    r32.a[g] = static_cast<float>(a);
    r32.b[g] = static_cast<float>(b);
  }
  kc64_.glare = r;
  kc32_.glare = r32;
#if BALLASIM_SENSOR_CUDA
  if (cuda_) cuda_->set_glare(r32);
#endif
}

void SensorModel::check_frame(const float* frame, const char* what) const {
  // Non-finite irradiance is an interface error of the renderer: no clamp gives it a meaning.
  const size_t G = groups_.size(), n = static_cast<size_t>(width_) * height_ * G;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(frame[i])) {
      throw std::invalid_argument(std::string("SensorModel.") + what + ": non-finite irradiance at pixel " +
                                  std::to_string(i / G));
    }
  }
}

std::vector<double> SensorModel::init_means(const float* frame) const {
  if (!given_means_.empty()) return {given_means_.begin(), given_means_.begin() + groups_.size()};
  if (params_.veiling_glare > 0 || track_means_) return frame_mean(frame);  // the held inputs' (frame_mean clamps)
  return {};
}

void SensorModel::init(int64_t t0_ns, const float* irradiance0) { init_with(t0_ns, irradiance0, nullptr); }

ProbeResult SensorModel::stationary_probe(int64_t t0_ns, const float* irradiance0) {
  if (params_.initial_state != InitialState::kStationary) {
    throw std::logic_error("SensorModel.stationary_probe: accuracy.initial_state is not stationary");
  }
  check_frame(irradiance0, "stationary_probe");
  const auto start = std::chrono::steady_clock::now();
  const std::vector<double> means = init_means(irradiance0);
  ProbeData d;
  if (stationary_on()) {
    const auto make = [&](auto kc) {
      kc.glare = init_ramp<decltype(kc.scale)>(params_.veiling_glare, means, t0_ns);
      return probe_frame(kc, t0_ns, irradiance0);
    };
    d = precision_ == Precision::kFp32 ? make(kc32_) : make(kc64_);
  }
  d.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  const size_t P = d.id.size();
  ProbeHeader h{};
  h.magic = kProbeMagic;
  h.version = kProbeVersion;
  h.precision = static_cast<uint32_t>(precision_);
  // The model's own filter flag, also for an empty probe (noise off, or no eligible pixel): init_kernel checks it.
  h.hpf = (precision_ == Precision::kFp32 ? kc32_.hpf : kc64_.hpf) ? 1 : 0;
  h.probes = static_cast<uint32_t>(P);
  h.snapshots = kSnapshots;
  h.key = probe_key(t0_ns, irradiance0, means);
  h.t0 = t0_ns;
  h.history_ns = d.history_ns;
  h.ms = d.ms;
  ProbeResult out;
  std::vector<uint8_t>& b = out.blob;
  const auto put = [&](const void* src, size_t bytes) {
    const size_t at = b.size();
    b.resize(at + bytes);
    if (bytes > 0) std::memcpy(b.data() + at, src, bytes);
  };
  put(&h, sizeof(h));
  put(d.id.data(), P * sizeof(uint32_t));
  put(d.f.data(), P * sizeof(Feature));
  put(d.theta_on.data(), P * sizeof(double));
  put(d.theta_off.data(), P * sizeof(double));
  put(d.ps.data(), P * sizeof(PixelState));
  put(d.ph.data(), d.ph.size() * sizeof(HpfState));
  put(d.snap.data(), d.snap.size() * sizeof(Snap));
  return out;
}

void SensorModel::stationary_init(int64_t t0_ns, const float* irradiance0, const ProbeResult& probe) {
  if (params_.initial_state != InitialState::kStationary) {
    throw std::logic_error("SensorModel.stationary_init: accuracy.initial_state is not stationary");
  }
  init_with(t0_ns, irradiance0, &probe);
}

void SensorModel::init_with(int64_t t0_ns, const float* irradiance0, const ProbeResult* probe) {
  check_frame(irradiance0, "init");
  initialized_ = false;  // until the states are complete: a failure past the checks leaves them part-way
  frame_mean_ = init_means(irradiance0);
  set_glare(t0_ns, frame_mean_);  // constant: held over the init and the stationary probe's history
  // An ROI model's own pixels' rows of the frame.
  std::vector<float> own;
  if (roi_) {
    const size_t G = groups_.size();
    own.resize(pixel_ids_.size() * G);
    for (size_t i = 0; i < pixel_ids_.size(); ++i) {
      std::copy_n(irradiance0 + static_cast<size_t>(pixel_ids_[i]) * G, G, own.data() + i * G);
    }
  }
  const float* rows = roi_ ? own.data() : irradiance0;
  // The GPU starts from the CPU fp32 initialization: the same code, so the same bits.
  if (precision_ == Precision::kFp32) init_kernel(kc32_, t0_ns, irradiance0, rows, probe);
  else init_kernel(kc64_, t0_ns, irradiance0, rows, probe);
#if BALLASIM_SENSOR_CUDA
  if (cuda_) cuda_->upload_states(states_, hpf_);
#endif
  t_state_ = t0_ns;
  calls_ = 0;
  initialized_ = true;
}

template <class T>
void SensorModel::init_kernel(const KernelConstants<T>& kc, int64_t t0, const float* frame, const float* own,
                              const ProbeResult* probe) {
  run_init(kc, t0, own);
  if (probe != nullptr) {  // check the given probe before using any of it
    ProbeHeader h{};
    const std::vector<uint8_t>& b = probe->blob;
    if (b.size() < sizeof(h)) throw std::invalid_argument("SensorModel.stationary_init: not a stationary probe");
    std::memcpy(&h, b.data(), sizeof(h));
    const size_t P = h.probes;
    const size_t ph = h.hpf != 0 ? P : 0;
    const size_t per_probe = sizeof(uint32_t) + sizeof(Feature) + 2 * sizeof(double) + sizeof(PixelState);
    const size_t bytes = sizeof(h) + P * per_probe + ph * sizeof(HpfState) + P * kSnapshots * sizeof(Snap);
    if (h.magic != kProbeMagic || h.version != kProbeVersion || h.snapshots != kSnapshots || b.size() != bytes) {
      throw std::invalid_argument("SensorModel.stationary_init: not a stationary probe of this version");
    }
    if (h.precision != static_cast<uint32_t>(precision_) || (h.hpf != 0) != kc.hpf || h.t0 != t0 ||
        h.key != probe_key(t0, frame, frame_mean_)) {
      throw std::invalid_argument(
          "SensorModel.stationary_init: the probe was made from other parameters, photometry, frame means, frame or "
          "t0");
    }
    if (!stationary_on()) return;  // the phase law (noise off or the instantaneous response): an empty probe
    const auto start = std::chrono::steady_clock::now();
    ProbeData d;
    const uint8_t* at = b.data() + sizeof(h);
    const auto get = [&](auto& v, size_t count) {
      v.resize(count);
      std::memcpy(v.data(), at, count * sizeof(v[0]));
      at += count * sizeof(v[0]);
    };
    get(d.id, P);
    get(d.f, P);
    get(d.theta_on, P);
    get(d.theta_off, P);
    get(d.ps, P);
    get(d.ph, ph);
    get(d.snap, P * kSnapshots);
    d.history_ns = h.history_ns;
    for (const uint32_t id : d.id) {
      if (id >= static_cast<size_t>(width_) * height_) {
        throw std::invalid_argument("SensorModel.stationary_init: a probe pixel outside the frame");
      }
    }
    transfer(kc, d, t0);
    if (P > 0) {
      stationary_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    return;
  }
  if (!stationary_on()) return;
  const auto start = std::chrono::steady_clock::now();
  // A full model probes its own init states; an ROI model a fresh init of the whole frame.
  const ProbeData d = roi_ ? probe_frame(kc, t0, frame) : probe_of(kc, maps_.pixels, states_, t0, frame);
  transfer(kc, d, t0);
  if (!d.id.empty()) {
    stationary_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
}

template <class T>
std::vector<RawEvent> SensorModel::run_advance(const KernelConstants<T>& kc, const std::vector<int32_t>& row_of,
                                               const int64_t* t, const float* irr, const int64_t* row_start,
                                               const int32_t* row_count, int64_t t_end) {
  const int G = groups();
  const int n_workers = pool_->size();
  std::vector<std::vector<RawEvent>> events(n_workers);
  std::vector<KernelStats> stats(n_workers);
#if BALLA_SENSOR_CENSUS
  for (KernelStats& s : stats) s.census.px = census_px_.data();
#endif
  const int64_t t_state = t_state_;
  // The glare-free instantiation keeps its code (pixel.h kGlare) when the call has no glare; the filter's has glare
  // compiled in (kHpf).
  const auto run = [&](auto glare, auto hpf) {
    pool_->parallel_for(states_.size(), kPixelBlock, [&](int w, size_t begin, size_t end) {
      VectorSink sink{&events[w]};
      for (size_t p = begin; p < end; ++p) {
        const int32_t row = row_of[p];
        InputRun in{nullptr, nullptr, 0};
        if (row >= 0) {
          in.t = t + row_start[row];
          in.irr = irr + static_cast<size_t>(row_start[row]) * G;
          in.n = row_count[row];
        }
        HpfState* h = nullptr;
        if constexpr (decltype(hpf)::value) h = &hpf_[p];
        PixelKernel<T, VectorSink, true, KernelStats, decltype(glare)::value, decltype(hpf)::value> k(
            kc, maps_.pixels[p], static_cast<uint32_t>(p), states_[p], t_state, in, sink, stats[w], -1, h);
        k.advance(t_end);
      }
    });
  };
  if (kc.hpf) run(std::true_type{}, std::true_type{});
  else if (kc.glare.on) run(std::true_type{}, std::false_type{});
  else run(std::false_type{}, std::false_type{});
  size_t total = 0;
  for (const auto& v : events) total += v.size();
  std::vector<RawEvent> all;
  all.reserve(total);
  for (auto& v : events) all.insert(all.end(), v.begin(), v.end());
  for (const KernelStats& s : stats) stats_.add(s);
  return all;
}

Events SensorModel::finish_events(std::vector<RawEvent>&& all) const {
  // Deterministic order whatever the thread count or device: a pixel never fires twice in one ns.
  std::sort(all.begin(), all.end(), [](const RawEvent& a, const RawEvent& b) {
    return a.t_cross != b.t_cross ? a.t_cross < b.t_cross : a.pixel < b.pixel;
  });
  Events out;
  const size_t n = all.size();
  out.x.resize(n);
  out.y.resize(n);
  out.p.resize(n);
  out.t_cross_ns.resize(n);
  out.t_reported_ns.resize(n);
  out.label.resize(n);
  out.flags.resize(n);
  out.ds.resize(n);
  out.dn.resize(n);
  out.dl.resize(n);
  out.l_first.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const RawEvent& e = all[i];
    out.x[i] = static_cast<uint16_t>(e.pixel % width_);
    out.y[i] = static_cast<uint16_t>(e.pixel / width_);
    out.p[i] = e.polarity;
    out.t_cross_ns[i] = e.t_cross;
    out.t_reported_ns[i] = reported_time(e.t_cross);
    out.label[i] = e.label;
    out.flags[i] = e.flags;
    out.ds[i] = e.ds;
    out.dn[i] = e.dn;
    out.dl[i] = e.dl;
    out.l_first[i] = e.l_first;
  }
  return out;
}

Events SensorModel::advance(const uint32_t* pixel_ids, size_t n, const int64_t* t, const float* irr, size_t m,
                            const int32_t* counts, int64_t t_end_ns) {
  if (m > static_cast<size_t>(INT32_MAX)) {
    throw std::invalid_argument("SensorModel.advance: too many samples per pixel");
  }
  std::vector<int64_t> start(n);
  std::vector<int32_t> count(n);
  for (size_t i = 0; i < n; ++i) {
    const int32_t c = counts ? counts[i] : static_cast<int32_t>(m);
    if (c < 0 || static_cast<size_t>(c) > m) throw std::invalid_argument("SensorModel.advance: count out of range");
    start[i] = static_cast<int64_t>(i * m);
    count[i] = c;
  }
  return advance_runs(pixel_ids, n, t, irr, n * m, start.data(), count.data(), t_end_ns);
}

Events SensorModel::advance_runs(const uint32_t* pixel_ids, size_t n, const int64_t* t, const float* irr,
                                 size_t n_samples, const int64_t* row_start, const int32_t* row_count,
                                 int64_t t_end_ns) {
  return finish_events(advance_raw(pixel_ids, n, t, irr, n_samples, row_start, row_count, t_end_ns));
}

std::vector<RawEvent> SensorModel::advance_raw(const uint32_t* pixel_ids, size_t n, const int64_t* t,
                                               const float* irr, size_t n_samples, const int64_t* row_start,
                                               const int32_t* row_count, int64_t t_end_ns) {
  if (!initialized_) {
    throw std::logic_error(
        "SensorModel.advance: call init() or load_state() first (a call that failed part-way leaves the model "
        "uninitialized)");
  }
  if (t_end_ns < t_state_) throw std::invalid_argument("SensorModel.advance: t_end_ns is before the model time");
  const size_t n_pixels = states_.size();
  const int G = groups();
  // An ROI model's pixel ids are the frame's: its own index of each.
  std::vector<uint32_t> own;
  if (roi_) {
    own.resize(n);
    for (size_t i = 0; i < n; ++i) {
      const int64_t l = pixel_ids[i] < static_cast<uint64_t>(width_) * height_ ? local_index(pixel_ids[i]) : -1;
      if (l < 0) throw std::invalid_argument("SensorModel.advance: pixel id outside the model's region of interest");
      own[i] = static_cast<uint32_t>(l);
    }
    pixel_ids = own.data();
  }
  std::vector<int32_t> row_of(n_pixels, -1);
  uint64_t clamps = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t p = pixel_ids[i];
    if (p >= n_pixels) throw std::invalid_argument("SensorModel.advance: pixel id out of range");
    if (row_of[p] >= 0) throw std::invalid_argument("SensorModel.advance: duplicate pixel id");
    row_of[p] = static_cast<int32_t>(i);
    const int64_t s = row_start[i];
    const int32_t c = row_count[i];
    if (c < 0 || s < 0 || static_cast<size_t>(s) + static_cast<size_t>(c) > n_samples) {
      throw std::invalid_argument("SensorModel.advance: sample run out of range");
    }
    const int64_t* ti = t + s;
    for (int32_t j = 0; j < c; ++j) {
      if (j == 0 ? ti[0] < t_state_ : ti[j] <= ti[j - 1]) {
        throw std::invalid_argument(
            "SensorModel.advance: sample times must be strictly increasing and start at or after the model time");
      }
    }
    if (c > 0 && ti[c - 1] > t_end_ns) throw std::invalid_argument("SensorModel.advance: sample after t_end_ns");
    const float* vi = irr + static_cast<size_t>(s) * G;
    for (size_t j = 0; j < static_cast<size_t>(c) * G; ++j) {
      if (!std::isfinite(vi[j])) {
        throw std::invalid_argument("SensorModel.advance: non-finite irradiance (pixel " + std::to_string(p) +
                                    ", sample " + std::to_string(j / G) + ")");
      }
      if (clamp_irradiance(vi[j], kc32_.irr_max) != vi[j]) ++clamps;
    }
  }
  stats_.input_clamps += clamps;
  std::vector<double> end;  // veiling glare: the frame means of the inputs held at t_end (given, or computed)
  if (!given_means_.empty()) {
    const size_t G0 = groups_.size(), at = (calls_ + 1) * G0;
    if (at + G0 > given_means_.size()) {
      throw std::logic_error("SensorModel.advance: no frame means given for this call (set_frame_means)");
    }
    end.assign(given_means_.begin() + static_cast<std::ptrdiff_t>(at),
               given_means_.begin() + static_cast<std::ptrdiff_t>(at + G0));
  } else if (params_.veiling_glare > 0 || track_means_) {
    if (roi_) throw std::logic_error("SensorModel.advance: an ROI model needs set_frame_means under veiling glare");
    end = frame_mean(held_frame(pixel_ids, n, irr, row_start, row_count).data());
  }
  set_glare(t_end_ns, end);
  std::vector<RawEvent> raw;
  initialized_ = false;  // until the kernel completes: a failure part-way leaves the pixels part-way
#if BALLASIM_SENSOR_CUDA
  if (cuda_) {
    raw = cuda_->advance(pixel_ids, n, t, irr, n_samples, row_start, row_count, t_state_, t_end_ns, stats_, gpu_stats_);
  } else
#endif
  {
    raw = precision_ == Precision::kFp32 ? run_advance(kc32_, row_of, t, irr, row_start, row_count, t_end_ns)
                                         : run_advance(kc64_, row_of, t, irr, row_start, row_count, t_end_ns);
  }
  t_state_ = t_end_ns;
  ++calls_;
  initialized_ = true;
  if (!end.empty()) frame_mean_ = std::move(end);
  return raw;
}

std::vector<RawEvent> SensorModel::advance_runs_raw(const uint32_t* pixel_ids, size_t n, const int64_t* t,
                                                    const float* irr, size_t n_samples, const int64_t* row_start,
                                                    const int32_t* row_count, int64_t t_end_ns) {
  std::vector<RawEvent> all = advance_raw(pixel_ids, n, t, irr, n_samples, row_start, row_count, t_end_ns);
  std::sort(all.begin(), all.end(), [](const RawEvent& a, const RawEvent& b) {
    return a.t_cross != b.t_cross ? a.t_cross < b.t_cross : a.pixel < b.pixel;
  });
  return all;
}

DeviceEvents SensorModel::advance_device(const DeviceRuns& runs, int64_t t_end_ns, void* stream) {
  if (!initialized_) {
    throw std::logic_error(
        "SensorModel.advance: call init() or load_state() first (a call that failed part-way leaves the model "
        "uninitialized)");
  }
  if (t_end_ns < t_state_) throw std::invalid_argument("SensorModel.advance: t_end_ns is before the model time");
  if (roi_) throw std::logic_error("SensorModel.advance_device: an ROI model advances host runs (advance_runs)");
#if BALLASIM_SENSOR_CUDA
  if (cuda_) {
    std::vector<double> end;  // veiling glare: the frame means of the inputs held at t_end, from the device
    if (params_.veiling_glare > 0 || track_means_) end = frame_mean(cuda_->end_frame(runs, stream).data());
    set_glare(t_end_ns, end);
    initialized_ = false;  // until the launches complete (as advance_raw)
    DeviceEvents ev;
    try {
      ev = cuda_->advance_device(runs, t_state_, t_end_ns, stream, true, stats_, gpu_stats_);
    } catch (const std::invalid_argument&) {  // the device's non-finite check: thrown before any state changes
      initialized_ = true;
      throw;
    }
    t_state_ = t_end_ns;
    ++calls_;
    initialized_ = true;
    if (!end.empty()) frame_mean_ = std::move(end);
    return ev;
  }
#endif
  (void)stream;
  host_events_ = advance_raw(runs.pixel, runs.n, runs.t, runs.irr, runs.n_samples, runs.row_start, runs.row_count,
                             t_end_ns);
  return {host_events_.data(), host_events_.size()};
}

Events SensorModel::download_events(const DeviceEvents& events) const {
#if BALLASIM_SENSOR_CUDA
  if (cuda_) return finish_events(cuda_->download(events));
#endif
  return finish_events(std::vector<RawEvent>(events.data, events.data + events.count));
}

void SensorModel::set_gpu_launch_policy(const GpuLaunchPolicy& policy) {
#if BALLASIM_SENSOR_CUDA
  if (cuda_) {
    cuda_->set_policy(policy);
    return;
  }
#endif
  (void)policy;
  throw std::logic_error("SensorModel.set_gpu_launch_policy: the model does not run on the CUDA device");
}

size_t SensorModel::state_bytes() const {
  if (roi_) throw std::logic_error("SensorModel: an ROI model has no saved states");
  return sizeof(StateHeader) + states_.size() * sizeof(PixelState) + hpf_.size() * sizeof(HpfState);
}

std::vector<uint8_t> SensorModel::save_state() const {
  StateHeader h{};
  h.magic = kStateMagic;
  h.version = kStateVersion;
  h.width = width_;
  h.height = height_;
  h.groups = static_cast<uint32_t>(groups_.size());
  h.camera = camera_;
  h.seed = seed_;
  h.fingerprint = state_fingerprint(params_, groups_, maps_.pixels);
  h.t_state = t_state_;
  h.initialized = initialized_ ? 1 : 0;
  h.state_size = sizeof(PixelState);
  std::vector<uint8_t> out(state_bytes());
  std::memcpy(out.data(), &h, sizeof(h));
  const std::vector<PixelState> current = states();
  std::memcpy(out.data() + sizeof(h), current.data(), current.size() * sizeof(PixelState));
  if (!hpf_.empty()) {  // the high-pass filter's section
    const std::vector<HpfState> filter = hpf_states();
    std::memcpy(out.data() + sizeof(h) + current.size() * sizeof(PixelState), filter.data(),
                filter.size() * sizeof(HpfState));
  }
  return out;
}

void SensorModel::load_state(const uint8_t* data, size_t size) {
  if (size != state_bytes()) throw std::invalid_argument("SensorModel.load_state: size does not match this model");
  StateHeader h{};
  std::memcpy(&h, data, sizeof(h));
  // Version 5 has no filter section: it loads into a model without the filter's kernels (the size check above).
  const bool version_ok = h.version == kStateVersion || (h.version == kStateVersionNoHpf && hpf_.empty());
  if (h.magic != kStateMagic || !version_ok || h.state_size != sizeof(PixelState)) {
    throw std::invalid_argument("SensorModel.load_state: not a sensor state of this version");
  }
  if (h.width != width_ || h.height != height_ || h.groups != groups_.size() || h.camera != camera_ ||
      h.seed != seed_ || h.fingerprint != state_fingerprint(params_, groups_, maps_.pixels)) {
    throw std::invalid_argument("SensorModel.load_state: state was saved by a model with other parameters");
  }
  initialized_ = false;  // until the upload completes
  std::memcpy(states_.data(), data + sizeof(h), states_.size() * sizeof(PixelState));
  if (!hpf_.empty()) {
    std::memcpy(hpf_.data(), data + sizeof(h) + states_.size() * sizeof(PixelState), hpf_.size() * sizeof(HpfState));
  }
#if BALLASIM_SENSOR_CUDA
  if (cuda_) cuda_->upload_states(states_, hpf_);
#endif
  t_state_ = h.t_state;
  initialized_ = h.initialized != 0;
  // Veiling glare: the frame means at the model time are the held inputs' (as an uninterrupted model has them).
  frame_mean_.clear();
  if (params_.veiling_glare > 0 || track_means_) {
    frame_mean_ = frame_mean(held_frame(nullptr, 0, nullptr, nullptr, nullptr).data());
  }
}

}  // namespace balla::sensor
