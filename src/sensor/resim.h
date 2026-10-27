// Sensor-only re-simulation from a signal cache (phase3/ui.md §3.8, §3.10, §4.8; UI-10's part: the replay of one
// cache through an ROI model with the engine's held buffer, output order and cut; UI-11 adds the session, its device
// residency and the worker).
//
// The cache holds every sensor call the writer run's camera made, in order: the hold (t_init -> t_m, empty runs), the
// motion warm-up windows [t_m, t_s) and the run windows, each with the cache ROI's listed pixels and their runs as the
// engine passed them. A ResimRun initializes an ROI model at t_init from the full frame, advances it through the calls
// up to t1 with the cache's frame means, drops the events of hold and warm-up calls (the engine does), and releases
// each run window's final events as Camera::output does: the window's events join the held ones, all are sorted into
// the output order and those reported before the window end on the sensor clock are released; finish() releases the
// rest (Camera::flush). The concatenated stream equals the writer run's event file inside the ROI on the compared
// columns. A spec with sensor.limiter on is refused (the camera-wide rate limiter needs every pixel's events: such an
// edit is tier R), and so is veiling glare on a cache without frame means.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "sensor/output_order.h"
#include "sensor/rederive.h"
#include "sensor/sensor_model.h"

namespace balla::sensor {

enum class CallKind : uint8_t { kHold = 0, kWarmup = 1, kWindow = 2 };

// One recorded sensor call (§3.8 /calls/{k}), decoded.
struct CacheCall {
  CallKind kind = CallKind::kWindow;
  int64_t t_w_ns = 0, t_end_ns = 0;
  std::vector<uint32_t> ids;   // the listed pixels' frame ids, ascending
  std::vector<int32_t> count;  // samples per listed pixel (the carried-over first sample included)
  std::vector<int64_t> t;      // the samples' times (ns), run after run
  std::vector<float> irr;      // samples x groups (every group; a dead one holds 0)
};

// A decoded signal cache of one camera (§3.8), in host memory.
struct CacheView {
  uint32_t width = 0, height = 0, camera = 0;
  Roi roi{};
  int64_t t_init_ns = 0;
  std::vector<float> init_irr;     // the frame rendered at t_m, (width * height) x groups
  std::vector<float> gain;         // the gain map, width * height (empty: 1 everywhere)
  std::vector<double> frame_mean;  // (calls + 1) x groups: row 0 at init, row k + 1 at the end of call k
  std::vector<CacheCall> calls;
};

// The sensor of a replay (§4.8): the edited parameters (illuminance_scale by the photometry rule of §4.7), the run's
// sensor seed, its light groups (the cache's, or rebuilt for EVIMO2 lighting edits) and the replay's precision.
struct SensorSpec {
  SensorParams params;
  uint64_t seed = 0;
  std::vector<Waveform> groups;
  Precision precision = Precision::kFp32;
};

// kPhase: the phase-start preview pass; kStationary: the stationary start (with a given probe, or its own).
enum class StartMode { kPhase, kStationary };

// Released events in the engine's output order.
struct RawEvents {
  std::vector<RawEvent> ev;         // as the sensor made them: frame pixel ids, C3 labels, increments, l_first
  std::vector<int64_t> t_reported;  // on the sensor clock
  std::vector<uint8_t> label_s1;    // the S1 labels the event file holds
  size_t size() const { return ev.size(); }
  void clear() {
    ev.clear();
    t_reported.clear();
    label_s1.clear();
  }
};

class ResimRun {
 public:
  // roi: inside the cache's ROI (a sub-ROI replays only its pixels: pixels are independent given the frame means and
  // the global-keyed draws). t1_ns: the end of a run window (replay every call that ends at or before it), or at least
  // the last call's end (every call). probe: a stationary probe for kStationary (stationary_probe of a model with the
  // same inputs; null: the replay probes). std::invalid_argument on an inconsistent cache, ROI or t1, on
  // spec.params.limiter_on(), and on veiling glare without the cache's frame means.
  ResimRun(const CacheView& cache, const SensorSpec& spec, Roi roi, int64_t t1_ns, StartMode mode,
           const ProbeResult* probe = nullptr, Device device = Device::kCpu, int num_threads = 0);
  ~ResimRun();
  ResimRun(const ResimRun&) = delete;
  ResimRun& operator=(const ResimRun&) = delete;

  // One recorded call; out (cleared) takes the events it releases (none for hold and warm-up calls). False once every
  // call up to t1 is done, or after cancel().
  bool step(RawEvents* out);
  // The held rest, in output order (Camera::flush): every event not yet released, all with t_cross < t1.
  void finish(RawEvents* out);
  // Stops at the next step (from any thread).
  void cancel() { cancelled_.store(true); }
  bool cancelled() const { return cancelled_.load(); }
  size_t calls_done() const { return next_; }
  size_t calls() const { return n_calls_; }
  const SensorModel& model() const { return *model_; }

 private:
  const CacheView& cache_;
  std::unique_ptr<SensorModel> model_;
  Roi roi_;
  int y_bits_;
  size_t n_calls_ = 0, next_ = 0;
  std::atomic<bool> cancelled_{false};
  RawEvents held_;                    // reported at or after the last cut, output order
  std::vector<uint32_t> ids_;         // a call's listing restricted to roi_
  std::vector<int64_t> start_;
  std::vector<int32_t> count_;
};

}  // namespace balla::sensor
