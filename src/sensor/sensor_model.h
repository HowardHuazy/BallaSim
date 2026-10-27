// SensorModel: contract C3 (docs/dev/CONTRACTS.md).
//
// Owns the frozen sensor maps, the noise tables, the per-pixel states and a thread pool, and
// advances every pixel of one camera with the shared kernel of pixel.h: on the CPU in fp64 or fp32,
// or on the GPU in fp32 (bitwise equal to the CPU fp32 path). Listed pixels follow their sample
// runs; unlisted pixels are advanced with their held input. Events are returned sorted by
// (t_cross, pixel) with the readout applied (M3 F.1).
//
// A region-of-interest model (roi; phase3/ui.md §3.10, §4.8) holds only the pixels of one rectangle of the frame and
// keys every draw (maps, noise, hot pixels, the tau_hp spread, the stationary probe's strata) by the frame's pixel ids,
// so that its events are bitwise those of the full model with only the rectangle's pixels listed.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sensor/launch.h"
#include "sensor/maps.h"
#include "sensor/noise_tables.h"
#include "sensor/params.h"
#include "sensor/pixel.h"
#include "sensor/readout.h"
#include "sensor/thread_pool.h"

namespace balla::sensor {

enum class Precision : int32_t { kFp64 = 0, kFp32 = 1 };
enum class Device : int32_t { kCpu = 0, kCuda = 1 };  // kCuda runs the fp32 kernel on the GPU

class CudaSensorKernel;

// A rectangle of the frame: x0 <= x < x1, y0 <= y < y1 (end-exclusive), inside [0, width) x [0, height).
struct Roi {
  int32_t x0, y0, x1, y1;
};

// The readout of camera `camera` (M3 F.1): the delay and the counter quantum of the parameters, the counter's phase
// draw(seed, kReadoutPhase, camera, 0, 0) mod q (0 without a quantum), the clock offset and drift.
int64_t readout_phase_ns(uint64_t seed, uint32_t camera, int64_t quantum_ns);
Readout readout_of(const SensorParams& p, uint64_t seed, uint32_t camera);

// A stationary probe (SensorModel::stationary_probe): the probe pixels' states and comparator snapshots per stratum,
// and the key of the inputs it was made from. Opaque.
struct ProbeResult {
  std::vector<uint8_t> blob;
};

// True if this build contains the CUDA sensor kernel and a CUDA device is visible.
bool sensor_cuda_available();

// The stationary probe's history T_b in whole noise cells (accuracy.stationary_burnin_ns; 0: 1.5 / R_L at the run's
// temperature, clamped to [60, 600] s, and with the high-pass filter on at most max(1 s, 10 tau_hp); SENSOR.md
// "Initial state").
int64_t probe_history_ns(const SensorParams& p);

// Cascade constants in kernel units (fA, s, log-e) with the tolerances of DESIGN §8.3:
// sub-steps accepted at eps_h / 1.5 with eps_h = eps_L / 8.
template <class T>
CascadeConstants<T> cascade_constants(const SensorParams& p);
extern template CascadeConstants<float> cascade_constants<float>(const SensorParams&);
extern template CascadeConstants<double> cascade_constants<double>(const SensorParams&);

struct Events {
  std::vector<uint16_t> x, y;
  std::vector<uint8_t> p;      // 1 = ON
  std::vector<int64_t> t_cross_ns, t_reported_ns;
  std::vector<uint8_t> label;  // 0 signal, 1 noise, 2 leak, 3 hot
  std::vector<uint16_t> flags;
  std::vector<float> ds, dn, dl;
  std::vector<float> l_first;  // RawEvent::l_first (FIRST_AFTER_INIT events; 0 otherwise)
  size_t size() const { return t_cross_ns.size(); }
};

// Sample runs in the model's memory (ENGINE_CPP §1.5, §1.7: device memory on the CUDA device, host
// memory on the CPU): listed pixel pixel[i] takes the row_count[i] samples starting at sample
// row_start[i] of t (int64 ns) and irr (n_samples x groups). The runs follow the rules of advance_runs
// by construction; the CUDA device checks only non-finite samples, the CPU checks them as advance_runs.
struct DeviceRuns {
  const uint32_t* pixel = nullptr;
  const int64_t* row_start = nullptr;
  const int32_t* row_count = nullptr;
  size_t n = 0;
  const int64_t* t = nullptr;
  const float* irr = nullptr;
  size_t n_samples = 0;
};

// Raw events left in the model's memory by advance_device, in no particular order; valid until the
// model advances again.
struct DeviceEvents {
  const RawEvent* data = nullptr;
  size_t count = 0;
};

class SensorModel {
 public:
  // camera is the camera's index in the scene bundle: it keys every random draw (maps, noise,
  // readout phase) together with the seed, so the cameras of a rig must pass distinct indices.
  // Every light group must pass validate_waveform (params.h; std::invalid_argument otherwise).
  // roi: a region-of-interest model of the width x height frame (file comment). Its pixel ids (advance, events) are
  // the frame's; init, stationary_probe and stationary_init take the full frame (the probe's strata); set_photometry
  // takes the frame's gains (or none: the clamp is the frame's); veiling glare needs set_frame_means. It has no
  // advance_device, save_state or load_state (std::logic_error), nor a census build.
  SensorModel(const SensorParams& params, uint32_t width, uint32_t height, uint64_t seed,
              std::vector<Waveform> light_groups, uint32_t camera, int num_threads = 0,
              Precision precision = Precision::kFp64, Device device = Device::kCpu,
              std::optional<Roi> roi = std::nullopt);
  ~SensorModel();

  // Photometry (D1; the states take it at the next init): the photocurrent scale (params().illuminance_scale, fA per renderer unit) and each
  // pixel's static gain (natural vignetting; empty: 1 everywhere), so that I = scale gain_p sum_g m_g E_g + I_dk. The
  // input clamp is one per camera, at the largest gain (irr_max()).
  void set_photometry(double illuminance_scale, const std::vector<float>& gain);

  // Steady state at t0 for every pixel; irradiance0 is (width * height) x groups, renderer units.
  // Non-finite irradiance is refused (std::invalid_argument); samples outside [0, irr_max()] are
  // clamped and counted (KernelStats::input_clamps). With veiling glare the frame means at t0 are irradiance0's.
  void init(int64_t t0_ns, const float* irradiance0);

  // Veiling glare (params().veiling_glare = g > 0; D13 item 3, SENSOR.md "Veiling glare"): every pixel's input gains
  // the uniform irradiance g x E_g(t) per light group, E_g the frame mean of gain x irradiance (frame_mean), so that
  // I = scale (gain_p E_gp + g E_g(t)) summed with m_g(t), + I_dk. E_g at a call boundary is the frame mean of what the
  // pixels hold there (PixelState::held: init's frame, then each listed pixel's last sample), a function of the
  // model's own inputs (a loaded state carries it; nothing is recorded or set); linear over each call between the
  // means at its start (glare_frame_mean) and at its end (computed from the runs before the call).
  std::vector<double> frame_mean(const float* frame) const;  // frame: (width * height) x groups
  const std::vector<double>& glare_frame_mean() const { return frame_mean_; }
  // The frame means from outside (an ROI model's: a signal cache's /frame_mean, f64; phase3/ui.md §3.8): row 0 at
  // init, row k + 1 at the end of the k-th advance after it, (calls + 1) x groups. They feed the glare as the full
  // model's own do, so a glare edit replays without the other pixels. std::logic_error on a full model.
  void set_frame_means(std::vector<double> means);
  // Computes the frame means at every init and advance also without glare (glare_frame_mean(); the signal tap's
  // input, f64 at any g): off by default, no effect on states or events. std::logic_error on an ROI model.
  void set_track_frame_means(bool on);

  // The stationary initial state (accuracy.initial_state stationary; std::logic_error otherwise) in two parts, whose
  // sequence equals init's own bit for bit: the probe, from the full frame at t0 (the model's state is untouched), and
  // init with that probe. The probe serves every model, ROI or full, with the same parameters (readout and label cut
  // aside: they leave the dynamics bit-identical), seed, camera, frame size, precision, photometry and frame means at
  // init, on the same frame and t0; stationary_init refuses any other (std::invalid_argument).
  ProbeResult stationary_probe(int64_t t0_ns, const float* irradiance0);
  void stationary_init(int64_t t0_ns, const float* irradiance0, const ProbeResult& probe);

  // Advances to t_end_ns. Row i of (t, irr) is pixel pixel_ids[i]: counts[i] samples (all m when
  // counts is null), strictly increasing, the first >= the model time, all <= t_end_ns; irr is
  // n x m x groups, finite (clamped as in init). Returns every crossing with t_cross < t_end_ns.
  Events advance(const uint32_t* pixel_ids, size_t n, const int64_t* t, const float* irr, size_t m,
                 const int32_t* counts, int64_t t_end_ns);

  // advance() with ragged runs in one sample array: pixel pixel_ids[i] takes the row_count[i]
  // samples starting at sample row_start[i] of t (n_samples entries) and irr (n_samples x groups).
  // The integrator's windows mix lattice levels 0-13, so a rectangle would be padded to the longest
  // run. Every run follows the rules of advance(), and the result is the same bit for bit.
  Events advance_runs(const uint32_t* pixel_ids, size_t n, const int64_t* t, const float* irr, size_t n_samples,
                      const int64_t* row_start, const int32_t* row_count, int64_t t_end_ns);

  // advance_runs on runs in the model's memory, with the events left there. On the CUDA device a
  // device check counts non-finite samples, and throws std::invalid_argument before any state changes
  // when there are any, and counts clamped samples (KernelStats::input_clamps); work is queued on
  // `stream` (a cudaStream_t) and the launch planner synchronizes after every launch. On the CPU it is
  // advance_runs on host runs without the sort (the engine's host path, ENGINE_CPP §1.7).
  // download_events copies the events to the host in the order and form of advance().
  DeviceEvents advance_device(const DeviceRuns& runs, int64_t t_end_ns, void* stream);
  Events download_events(const DeviceEvents& events) const;
  // advance_runs' events as the kernel made them (RawEvent: global pixel ids, the increments and l_first), sorted by
  // (t_cross, pixel), on either device (the re-simulation's held buffer, resim.h).
  std::vector<RawEvent> advance_runs_raw(const uint32_t* pixel_ids, size_t n, const int64_t* t, const float* irr,
                                         size_t n_samples, const int64_t* row_start, const int32_t* row_count,
                                         int64_t t_end_ns);

  // Largest irradiance sample taken as is (renderer units): kMaxSamplePhotocurrent_fA / (scale x the largest gain).
  float irr_max() const { return kc32_.irr_max; }

  // load_state takes a save_state blob of a model with the same parameters (std::invalid_argument otherwise):
  // only its header is checked, so a corrupted payload gives undefined results. A call of init, advance,
  // advance_device or load_state that throws after its input checks leaves the model uninitialized
  // (initialized() false: advance refuses) until init or load_state, since its pixels may be part-way.
  size_t state_bytes() const;
  std::vector<uint8_t> save_state() const;
  void load_state(const uint8_t* data, size_t size);

  const SensorParams& params() const { return params_; }
  const SensorMaps& maps() const { return maps_; }
  const KernelStats& stats() const { return stats_; }
  const GpuLaunchStats& gpu_launch_stats() const { return gpu_stats_; }
  void reset_stats() {
    stats_ = KernelStats{};
    gpu_stats_ = GpuLaunchStats{};
  }
  // Launch policy of the CUDA device (std::logic_error on the CPU).
  void set_gpu_launch_policy(const GpuLaunchPolicy& policy);
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  // The region of interest (none: the full frame), the model's pixels (its own index order: maps(), states()) and
  // their frame pixel ids (an ROI model's, row-major; empty for a full model).
  const std::optional<Roi>& roi() const { return roi_; }
  size_t n_pixels() const { return states_.size(); }
  const std::vector<uint32_t>& pixel_ids() const { return pixel_ids_; }
  int groups() const { return static_cast<int>(groups_.size()); }
  int64_t time_ns() const { return t_state_; }
  bool initialized() const { return initialized_; }
  int num_threads() const { return pool_->size(); }
  Precision precision() const { return precision_; }
  Device device() const { return device_; }
  int64_t readout_phase_ns() const { return readout_phase_ns_; }
  // Wall time of the last init's stationary probe and transfer (ms; 0 without one), its probe pixels, its history
  // T_b (ns) and the pixels that kept the phase law because no snapshot of their probe fits inside their levels.
  double stationary_ms() const { return stationary_ms_; }
  size_t stationary_probes() const { return stationary_probes_; }
  int64_t stationary_history_ns() const { return stationary_history_ns_; }
  size_t stationary_phase_kept() const { return stationary_phase_kept_; }

  // Readout of one crossing (M3 F.1): floor to the counter quantum after the delay d0, then the
  // sensor clock t + offset + drift t (sensor/readout.h).
  int64_t reported_time(int64_t t_cross_ns) const { return balla::sensor::reported_time(readout_, t_cross_ns); }
  const Readout& readout() const { return readout_; }

  // Kernel constants of either precision and a copy of the pixel states (for tests that drive
  // PixelKernel directly or compare devices).
  const KernelConstants<double>& constants64() const { return kc64_; }
  const KernelConstants<float>& constants32() const { return kc32_; }
  std::vector<PixelState> states() const;
  // The high-pass filter's side states (hpf.h; empty while the filter is off), and whether its kernels run.
  std::vector<HpfState> hpf_states() const;
  bool hpf_kernel() const { return kc32_.hpf; }
  // Test hook (ORACLES.md "HPF oracle" D): with the filter off, run its kernels at b = 0 from the next init on (the
  // filter-free sensor's dynamics through the filter's code path). std::logic_error with the filter on.
  void force_hpf_kernel();

 private:
  struct ProbeData;  // sensor_model.cpp
  template <class T>
  KernelConstants<T> make_constants(const NoiseTableView<T>& view) const;
  // init and stationary_init: the model's pixels' states from the full frame, then (stationary) a probe, given or made.
  void init_with(int64_t t0, const float* irradiance0, const ProbeResult* probe);
  template <class T>
  void init_kernel(const KernelConstants<T>& kc, int64_t t0, const float* frame, const float* own,
                   const ProbeResult* probe);
  // The phase law's init of `states` (and the filter's `hpf`) of these maps from irr (one row per state).
  template <class T>
  void init_states(const KernelConstants<T>& kc, const std::vector<PixelMap>& maps, int64_t t0, const float* irr,
                   std::vector<PixelState>& states, std::vector<HpfState>& hpf, KernelStats* total);
  template <class T>
  void run_init(const KernelConstants<T>& kc, int64_t t0, const float* irradiance0);
  // accuracy.initial_state = stationary (SENSOR.md "Initial state"): the probe from the full frame's init states
  // (probe_of), from a fresh full-frame init (probe_frame: stationary_probe, and an ROI model's init), and the
  // transfer to the model's own pixels.
  bool stationary_on() const;
  template <class T>
  ProbeData probe_of(const KernelConstants<T>& kc, const std::vector<PixelMap>& maps,
                     const std::vector<PixelState>& states, int64_t t0, const float* irradiance0);
  template <class T>
  ProbeData probe_frame(const KernelConstants<T>& kc, int64_t t0, const float* frame);
  template <class T>
  void transfer(const KernelConstants<T>& kc, const ProbeData& d, int64_t t0);
  uint64_t probe_key(int64_t t0, const float* frame, const std::vector<double>& means) const;
  // The frame means at init: given (set_frame_means), or the frame's when glare (or tracking) needs them.
  std::vector<double> init_means(const float* frame) const;
  void check_frame(const float* frame, const char* what) const;
  // An ROI model's index of frame pixel id `id` (-1 outside the region; a full model: id itself).
  int64_t local_index(uint32_t id) const;
  template <class T>
  std::vector<RawEvent> run_advance(const KernelConstants<T>& kc, const std::vector<int32_t>& row_of,
                                    const int64_t* t, const float* irr, const int64_t* row_start,
                                    const int32_t* row_count, int64_t t_end);
  // advance_runs without the final sort: the events in no particular order.
  std::vector<RawEvent> advance_raw(const uint32_t* pixel_ids, size_t n, const int64_t* t, const float* irr,
                                    size_t n_samples, const int64_t* row_start, const int32_t* row_count,
                                    int64_t t_end_ns);
  Events finish_events(std::vector<RawEvent>&& all) const;
  // Veiling glare: the input each pixel holds after a call on these host runs (its run's last sample, else its held
  // input), (width * height) x groups; the CUDA device's held inputs are read from the device.
  std::vector<float> held_frame(const uint32_t* pixel_ids, size_t n, const float* irr, const int64_t* row_start,
                                const int32_t* row_count) const;
  // The kernels' glare ramp from the model time (frame_mean_) to t_end (end: the frame means there; off without
  // glare); std::logic_error when glare is on and the means are missing.
  void set_glare(int64_t t_end, const std::vector<double>& end);

  SensorParams params_;
  uint32_t width_, height_;
  uint64_t seed_;
  uint32_t camera_;
  std::vector<Waveform> groups_;
  Precision precision_;
  Device device_;
  std::optional<Roi> roi_;
  std::vector<uint32_t> pixel_ids_;   // an ROI model's frame pixel ids (KernelConstants::pixel_ids points here)
  std::vector<float> frame_gain_;     // an ROI model's gains of the whole frame (set_photometry; empty: 1)
  std::vector<double> given_means_;   // set_frame_means; empty: computed
  size_t calls_ = 0;                  // advance calls since the last init (the row of given_means_)
  bool track_means_ = false;          // set_track_frame_means
  SensorMaps maps_;
  double gain_max_ = 1.0;  // the largest PixelMap::gain (set_photometry)
  std::vector<double> frame_mean_;  // veiling glare: the frame means of the held inputs at t_state_
  std::unique_ptr<NoiseTables> tables_;
  std::vector<PixelState> states_;
  std::vector<HpfState> hpf_;  // the high-pass filter's side states (one per pixel while kc.hpf; empty otherwise)
#if BALLA_SENSOR_CENSUS
  std::vector<CensusPixel> census_px_;  // the census's per-pixel side state (census.h; CPU census builds only)
#endif
  bool hpf_test_ = false;      // force_hpf_kernel()
  std::unique_ptr<ThreadPool> pool_;
  KernelConstants<double> kc64_;
  KernelConstants<float> kc32_;
  std::unique_ptr<CudaSensorKernel> cuda_;  // owns the device copy of the states when device_ is kCuda
  int64_t readout_phase_ns_ = 0;
  Readout readout_{};
  std::vector<RawEvent> host_events_;  // advance_device's events on the CPU
  int64_t t_state_ = 0;
  bool initialized_ = false;
  double stationary_ms_ = 0;
  size_t stationary_probes_ = 0;
  int64_t stationary_history_ns_ = 0;
  size_t stationary_phase_kept_ = 0;
  KernelStats stats_;
  GpuLaunchStats gpu_stats_;
};

}  // namespace balla::sensor
