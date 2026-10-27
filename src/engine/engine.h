// The C++/CUDA engine of a run (ENGINE_CPP §1.1, §1.3, §1.7, §1.9, §3 I1, I3 and I4): one renderer, one
// WindowEngine shared by the cameras (they run one after the other), and per camera its sensor, planner
// and carried samples. Two instantiations of one implementation (engine_impl.h): on the device (OptiX,
// the CUDA sensor, CUB) and on the host (Embree, the CPU sensor, serial primitives); the renderer's
// backend chooses. Python drives it window by window: start(), then step(), which plans, samples, senses
// and returns the events that are final in output order (flush() returns the rest at the end). window()
// samples given active pixels and returns the sensor's events, and gt() the centre-ray ground truth
// (tests). The calls of all cameras of an Engine are serialized (one stream, one renderer).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "engine/cancel.h"
#include "engine/motion.h"
#include "engine/output.h"
#include "engine/window.h"
#include "render/renderer.h"
#include "sensor/sensor_model.h"

namespace balla::engine {

// P2-B B6c (accuracy.band_growth): the band N's map, the tiles within a seed's reach (B4a) or the pixels (band_map.h).
enum class BandGrowth : int32_t { kTiles = 0, kPixels = 1 };
// P2-D D1 (accuracy.shadow_obliquity, F021): the planner's shadow-edge rates as built (off) or with the receivers'
// obliquity factors from the plan pass's geometric normals (projected; plan.h obliquity, capped at kObliquityCap).
enum class ShadowObliquity : int32_t { kOff = 0, kProjected = 1 };
// P2-D D3a (accuracy.localization): how signature flips are localized. bisect: bisection of whole pixel-samples (as
// built); shadow: predict-and-verify beside it on the band's pixels, counted only (engine/localize.h; every output as
// bisect's).
enum class Localization : int32_t { kBisect = 0, kShadow = 1 };

// The parameter rows of EngineConfig and SensorSpec are in engine/param_fields.h.
struct EngineConfig {
  render::RendererOptions render;              // OptiX: the device engine; Embree: the host engine
  std::vector<sensor::Waveform> light_groups;  // C1 waveforms, one per light group
  bool refine_signature = true;
  std::optional<double> refine_tol_log = 0.002;  // eta_L tolerance; none: no midpoint test
  int32_t max_refine_levels = 4;
  // The economy schedule's switches (ENGINE_CPP I7; sampling.h SampleRules); the defaults are Phase 1's schedule.
  std::optional<int32_t> mid_levels;    // midpoint flags split intervals < this many levels below the lattice; none: all
  std::optional<int64_t> tau_jump_ns;   // signature splits stop below 2 tau max(1, delta_ref / eta_J); none: no stop
  double delta_ref = 0.0;               // log-e; 0: tau absolute
  double half_ratio = 0.0;              // a midpoint-flagged half refines if eta_J >= this x the halves' larger; 0: both
  std::optional<double> eps_jump;       // a signature flag needs eta_J > eps_jump; none: no gate
  int64_t min_dt_ns = 1024;
  int64_t fixed_period_ns = 0;  // > 0: the fixed-rate grid
  double device_budget_mb = 0;  // <= 0: 70 % of the device memory free once renderer and sensors are built
                                // (host: of the available memory, MemAvailable on Linux; the physical memory
                                // elsewhere); the windows' samples get it less the engine's other buffers and
                                // the renderer's held sets (I3b) at each window
  // step(): the planner (l_min and l_max follow refine_tol_log and min_dt_ns), the trajectories' holds
  // (C1; MOVING and held windows), the crop and the GT columns.
  double d_px = 0.25;
  std::optional<int64_t> max_samples_per_window = int64_t{1} << 25;  // none: no clamp (D9)
  std::vector<Holds> holds;
  std::optional<Crop> crop;
  bool gt = true;
  bool held_world = true;  // I3b: bodies held over a window are traced in the renderer's world structure
  // I8 (motion.h PlanSchedule): the moving-shadow boundary band.
  bool shadow_band = false;
  // P2-B B2 and B4a (PHASE2 D-P2-4's band N; accuracy.n_band_max, 0 for null): 0 one N, render.scene.n_subsamples, for
  // every pixel; else the band's N, a power of two above n_subsamples (itself a power of two) and at most 1024. Each
  // window then samples every active pixel at N_base = n_subsamples or N_band = n_band, each N in render calls of its
  // own: the planner's band tiles (TilePlan::band: within the reach of a plan-pass seed) at N_band, the rest at N_base;
  // on the fixed-rate grid and in window() every pixel at N_band (CameraEngine::force_n overrides both). A pixel whose
  // carried sample has the other N re-renders it at t_w first (a switch); start() renders every pixel at N_band. P2-B
  // B4b: after the window's pass, a pixel at N_base with an edge sample (two or more clusters or a shadow boundary)
  // re-renders the window at N_band (an escalation; SwitchStats), so its runs are the uniform N_band run's; B6a: so
  // does one whose active 8-neighbour has an edge sample, and the crop's 1 px border ring samples at N_band.
  uint32_t n_band = 0;
  // P2-B B6c, with the band (n_band > 0; ignored without it): its map, tile (B4a's tile map) or pixel (each seed's
  // L-inf ball of radius floor(rho + s + max(sigma_cam M_static, sigma_mov M_mov)), band_map.h), and the reach scales
  // [sigma_cam, sigma_mov] (>= 0; pixel only) of a seed's static (camera) and movers' motion. Under T2's conditions
  // the escalation makes the outputs independent of both: they move cost only (ENGINE_CPP §0.1 "P2-B B6c notes").
  BandGrowth band_growth = BandGrowth::kTiles;
  std::array<double, 2> band_reach{1.0, 1.0};
  // P2-C C2 (execution.jump_counters): the window pass's counters of what the jump rules would change (sampling.h
  // kJump*, kSib*; WindowStats::jump); outputs unchanged.
  bool jump_counters = false;
  // P2-D D1 (motion.h PlanSchedule::obliquity): the shadow-edge speed bound with the obliquity factors.
  ShadowObliquity shadow_obliquity = ShadowObliquity::kOff;
  // P2-D D3a (window.h SampleConfig::pv): P&V's shadow mode on the band's pixels (N >= 256, at most 8 shadow-casting
  // lights; else off), counters in WindowStats::pv; outputs unchanged.
  Localization localization = Localization::kBisect;
  // P2-H H9 (accuracy.level_policy; motion.h LevelPolicy, F004): per-tile lattice levels (Phase 1's), or the window's
  // largest tile level on every active tile (global) or on the dynamic tiles (global_certified); F004's thesis arms.
  LevelPolicy level_policy = LevelPolicy::kPerTile;
};

// The renderer's held set at a window (I3b): bodies held over it, and whether and how long it was rebuilt.
struct HeldSet {
  uint32_t bodies = 0;
  bool rebuilt = false;
  double build_ms = 0;
};

// How a camera's photocurrent scale is set (photometry.mode; D1, D13 item 2; SENSOR.md "Photometry").
enum class Photometry : int32_t {
  kRelative = 0,   // params.illuminance_scale as given (Phase 1); vignetting only when asked (D1 option 2)
  kPhysical = 1,   // params.illuminance_scale = the lens and pixel gain the resolver derived (DESIGN §7.1); vignetting
  kReference = 2,  // the scale solved at start from the frame: a stated median photocurrent; vignetting
};

// The natural-vignetting law of a camera's gain map (optics.vignetting_law; P2-F F2, PHASE2 F079; SENSOR.md
// "Photometry"): cos^4 of each pixel's chief ray (Phase 1's, the pinhole's), or cos(theta) dOmega/dA of the camera
// model (core/camera/solid_angle.h; cos^4 / det J_d for radtan).
enum class VignettingLaw : int32_t {
  kCos4 = 0,
  kJacobian = 1,
};

// One camera's sensor (SensorModel arguments): the CUDA device in fp32 with the device engine, the CPU
// at `precision` with the host engine; and its photometry.
struct SensorSpec {
  sensor::SensorParams params;
  uint64_t seed = 1;
  int num_threads = 0;
  sensor::Precision precision = sensor::Precision::kFp64;
  Photometry photometry = Photometry::kRelative;
  std::optional<bool> vignetting;              // natural vignetting by `vignetting_law`; none: off under relative
  VignettingLaw vignetting_law = VignettingLaw::kCos4;
  // The lens's relative illumination on the gain map (P2-F F2b; core/camera/relative_illumination.h): [] (none) or
  // [c2, c4], G = 1 + c2 r~^2 + c4 r~^4 times the law's gain.
  std::vector<double> relative_illumination;
  // The fraction of this camera's photo-electrons from the near infrared, per light group (P2-F F11a; p2f.md §4.9,
  // DESIGN §7.2): [] (0) or one value in [0, 1] per group of the bundle. make_engine hands it to the renderer
  // (render::RendererOptions::nir_fraction), which mixes each material's visible and NIR albedo by it.
  std::vector<double> nir_fraction;
  std::optional<double> reference_current_fA;  // reference: median signal photocurrent of the reference pixels
  std::string reference_instance;              // reference: the body whose pixels set it ("": every pixel that hits)
};

struct StartResult {
  double warmup_ms = 0;         // wall time of the sensor's init and warm-up (the motion windows included)
  double probe_ms = 0;          // of which the stationary probe (SensorModel::stationary_ms; 0 without one)
  int64_t probe_history_ns = 0; // the probe's history T_b (SensorModel::stationary_history_ns; 0 without one)
  double illuminance_scale = 0; // the sensor's photocurrent scale, fA per renderer unit on axis (reference: solved)
  int32_t motion_windows = 0;   // windows of the motion warm-up (rendered and sensed, events dropped)
  render::RenderStats render;   // the frames and the motion windows
  HeldSet held;                 // the held set at t_s (rebuilt and build_ms: over the whole start)
  std::vector<double> frame_mean;  // veiling glare: the sensor's frame means per group at t_s (empty without glare)
  // Scanned rooms (scans.md §4.2, "Start diagnostic"): in a scene with baked proxies, per proxy shadow light (bundle
  // order) the share of its start-frame rays from baked receivers that the scan blocks (Renderer::proxy_visibility at
  // t_s; high where the proxy sits inside scanned fixture geometry), and the static bodies split from the scan
  // geometry (Renderer::split_geometries; cost grows linearly with them). Empty and 0 otherwise.
  std::vector<float> proxy_blocked;
  int32_t split_geometries = 0;
};

// The band's N switches of a window (P2-B B2; EngineConfig::n_band): its active pixels at N_band (and their
// pixel-samples, WindowStats::band_samples); the switches, active pixels whose carried sample has the other N (each
// re-renders its t_w node at the new one: a sample each, in render calls of their own, not in pixel_samples), of
// which those up to N_band; and each switch's step, eta_J between the carried and the re-rendered value, in units of
// eps_L / 8 (refine_tol_log / 8, the switch tolerance eps_jump of D-P2-4; 2.5e-4 without a midpoint test) in the bins
// [0, 1), [1, 2), [2, 4), ..., [64, inf) (sampling.h kDiagBins), unquiescent above 1 (n_switch_unquiescent).
// A re-rendered node is the uniform run's carried node at that N bit for bit, except in a window whose held set was
// rebuilt (HeldSet::rebuilt): the uniform run rendered it under the previous window's set, and a held body's hits
// differ from a moving one's in the last bits (ENGINE_CPP §0.1 "P2-B B2 notes"; held_world false has no exception).
// P2-B B4a: the active pixels with a sample of edge bits (sampling.h kBandEdge, the carried sample at t_w included) at
// N_band and at N_base (none on the fixed-rate grid, which has no flags); the band's overfill is band_pixels -
// band_edge, its misses base_edge. P2-B B4b escalates the misses (Camera::escalate): escalated (outside the fixed-rate
// grid = base_edge at B4b, base_edge + escalated_pooled since B6a, below) pixels re-render their window at N_band,
// escalated_samples pixel-samples at N_band (their t_w nodes and the pass's rounds; neither in pixel_samples nor
// band_samples, which count the window's pass). The window's other counters (WindowStats: samples_by_round, flags,
// refine_cap_hits, batches, the P2-0b diag words) describe the window's pass, the escalated pixels' runs at N_base
// included, which the sensor does not read; the escalation pass's own are below. changed: the switches whose
// re-rendered value differs from the carried one in any bit; escalated_changed, escalated_unquiescent and
// escalated_eta: the escalated pixels' t_w steps, from the node the window's pass used (the previous window's N_base
// end node) to the N_band re-render, as the switches' (none for a pixel switched down in the window, whose step the
// switch counted). Under T2's conditions (fixed64, powers of two, no held-set rebuild) a switch's or an escalation's
// re-render equals the node it replaces but for T2's gap nodes (an N_base end node that differs from the uniform N_band
// run's; ENGINE_CPP §0.1 "P2-B B4b notes"), so changed and escalated_changed are defect detectors finer than
// unquiescent, and lower bounds (a gap node the next window neither switches nor escalates, and a gap inside a window,
// are not seen). P2-B B6a: an N_base pixel also escalates when an active 8-neighbour has an edge sample (pooled
// escalation; escalated_pooled: those without one of their own, so escalated = base_edge + escalated_pooled), and the
// crop's outermost 1 px ring samples at N_band (border: its active pixels outside the plan's band tiles, among
// band_pixels); force_n's map overrides the ring, not the pooling. P2-B B6c: plan, the active pixels the plan's band
// (tile or pixel map; EngineConfig::band_growth) puts at N_band, the ring and force_n aside (0 without a plan:
// window(), the fixed-rate grid); escalation_calls, the escalation's render calls (its t_w nodes and its pass's rounds,
// the last ones of WindowResult::render_ms_by_call). P2-C C2: the escalation pass's own counters are
// WindowResult::escalation.
struct SwitchStats {
  bool on = false;  // the band is on (the counters below are counted)
  uint64_t band_pixels = 0, switches = 0, up = 0, unquiescent = 0;
  std::array<uint64_t, kDiagBins> eta{};
  uint64_t band_edge = 0, base_edge = 0;
  uint64_t escalated = 0, escalated_samples = 0, changed = 0;
  uint64_t escalated_changed = 0, escalated_unquiescent = 0;
  std::array<uint64_t, kDiagBins> escalated_eta{};
  uint64_t escalated_pooled = 0, border = 0;
  uint64_t plan = 0, escalation_calls = 0;
};

// A copy of a window's runs (CameraEngine::last_runs): its active pixels (rank order), their offsets into t and irr
// (n + 1), the sample times and the irradiance (samples x groups), each pixel's run contiguous in rank order; with the
// band (EngineConfig::n_band) the log2 N of each pixel's run (N_band for an escalated pixel, P2-B B4b), else empty.
struct WindowRuns {
  std::vector<uint32_t> pixel;
  std::vector<int64_t> offsets, t;
  std::vector<float> irr;
  std::vector<uint8_t> n_log2;
};

struct WindowResult {
  HeldSet held;  // the renderer's held set, set before the window (not in sample_ms; step(): before the plan)
  SwitchStats band;  // P2-B B2's per-pixel N (EngineConfig::n_band)
  WindowStats stats;
  // P2-C C2: the escalation pass's own counters (P2-B B4b; Camera::escalate): its samples by round, render calls by
  // round (call_round) and, with SampleRules::counters, the jump counters of its rounds. Empty without an escalation.
  WindowStats escalation;
  uint64_t pixel_samples = 0;  // the rounds' samples (stats.samples_by_round), the band's switches not among them
  render::RenderStats render;  // the window's render_into calls
  // GPU time (trace + shade) of each render call (I7): per round with one batch; with the band, the switches' calls
  // (SwitchStats) first, then a call per N class of each round, then the escalation's (its t_w nodes, then a call per
  // round; stats.render_calls counts them all).
  std::vector<double> render_ms_by_call;
  // Veiling glare (D13 item 3): the sensor's frame means per group at t_w + W (SensorModel::glare_frame_mean: of the
  // inputs its pixels hold there, derived from its runs; a diagnostic, nothing re-simulation needs). Empty without glare.
  std::vector<double> frame_mean;
  double sample_ms = 0;        // wall time of sampling (from the call to the runs)
  double sense_ms = 0;         // wall time of the sensor (advance_device and the event download)
  sensor::Events events;       // as SensorModel::advance returns them
  sensor::KernelStats sensor_stats;
  sensor::GpuLaunchStats launch_stats;
};

// P2-C C2: a window's render time (trace + shade, ms; WindowResult::render_ms_by_call) split by what each call
// rendered. render_ms_by_call has an entry per call made: the switches' t_w nodes (SwitchStats), the window pass's
// calls in WindowStats::call_round's order, then with an escalation its t_w nodes' call and its pass's calls (in
// WindowResult::escalation's call_round). pass and escalation: by round of each pass; redone_ms: the calls of batches
// that overflowed after rendering and were redone (kRedoneCall, either pass); switches_ms and escalation_nodes_ms: the
// t_w nodes' calls. ok false, everything else empty, where the calls do not add up.
struct RoundTimes {
  bool ok = false;
  std::vector<double> pass, escalation;
  double switches_ms = 0, escalation_nodes_ms = 0, redone_ms = 0;
};

inline RoundTimes round_times(const WindowResult& r) {
  RoundTimes out;
  const std::vector<double>& calls = r.render_ms_by_call;
  const std::vector<uint32_t>& pass = r.stats.call_round;
  const std::vector<uint32_t>& esc = r.escalation.call_round;
  const auto redone = [](const std::vector<uint32_t>& c) {
    return static_cast<size_t>(std::count(c.begin(), c.end(), kRedoneCall));
  };
  const size_t esc_calls = r.band.escalation_calls == 0 ? 0 : 1 + esc.size();
  // stats.render_calls counts the calls of the batches kept (the escalation's included), so with the redone ones it
  // is every call made.
  if (calls.size() != r.stats.render_calls + redone(pass) + redone(esc) || calls.size() < pass.size() + esc_calls) {
    return out;
  }
  const size_t first = calls.size() - pass.size() - esc_calls;
  const auto bin = [&](const std::vector<uint32_t>& rounds, size_t at, size_t n_rounds, std::vector<double>& ms) {
    ms.assign(n_rounds, 0.0);
    for (size_t i = 0; i < rounds.size(); ++i) {
      if (rounds[i] == kRedoneCall) {
        out.redone_ms += calls[at + i];
        continue;
      }
      if (rounds[i] >= ms.size()) ms.resize(static_cast<size_t>(rounds[i]) + 1, 0.0);
      ms[rounds[i]] += calls[at + i];
    }
  };
  for (size_t i = 0; i < first; ++i) out.switches_ms += calls[i];
  bin(pass, first, r.stats.samples_by_round.size(), out.pass);
  if (esc_calls > 0) {
    out.escalation_nodes_ms = calls[first + pass.size()];
    bin(esc, first + pass.size() + 1, r.escalation.samples_by_round.size(), out.escalation);
  }
  out.ok = true;
  return out;
}

// The plan's kappa (PHASE2 F010; ENGINE_CPP §0.1 P2-0b): over the window's active pixels whose plan-pass centre ray
// hits, their number and the sums of the true image speed |flow| at t_w, of their tile's bound v_hat and of their
// lattice's speed 2^level d_px / W, px/s (all zero without a plan pass).
struct PlanCounters {
  uint64_t pixels = 0;
  double flow = 0, v_hat = 0, lattice = 0;
};

// P2-H H4b: the rate limiter's counters (sensor.limiter; sensor/limiter.h, applied to a camera's final events at the
// end of the output stage, on the sensor clock). Over the bins it decided in a step (StepResult::limiter) or since
// start() (CameraEngine::limiter_counters, the flush's bins included): the events kept and dropped and the bins that
// dropped any; held, the events of the bin later events may still join (with a readout quantum q > 0 none: a bin is one
// stamp); ms, the wall time of the limiter (since start(): the flush's included). All zero, and on false, while the
// limiter is off.
struct LimiterCounters {
  bool on = false;
  uint64_t kept = 0, dropped = 0, capped_bins = 0, held = 0;
  double ms = 0;
};

// One step(): the window's plan, samples, sensing and output.
struct StepResult {
  WindowResult window;          // sampling and sensing counters (no events)
  bool planned = false;         // false on the fixed-rate grid
  PlanStats plan;
  PlanCounters kappa;
  std::optional<int> level_cap;
  double plan_ms = 0;           // wall time of planning (plan pass, pixel and tile stages, active list)
  double output_ms = 0;         // wall time of GT, sort, split and download
  uint64_t events = 0;          // the window's events inside the crop (before the rate limiter)
  std::vector<OutEvent> ready;  // the events final after this window, in output order (after the rate limiter)
  LimiterCounters limiter;      // P2-H H4b: the rate limiter's decisions in this step
};

// One camera of an Engine.
class CameraEngine {
 public:
  virtual ~CameraEngine() = default;

  // Renders every pixel at t_s (each pixel's carried sample and the sensor's frame), initializes the
  // sensor at t_s - warmup_ns and advances it to t_s with the frame held; warm-up events are dropped.
  // motion (accuracy.warmup = motion; D12): the warm-up's whole windows, rounded up, that the bundle covers are
  // planned, rendered and sensed as the run's windows are (events dropped), the frame at their start held before
  // them; every pixel is then rendered at t_s as above, so the runs from t_s on are those of a static warm-up.
  virtual StartResult start(int64_t t_s, int64_t warmup_ns, bool motion) = 0;
  // Window [t_w, t_w + W) of n distinct active pixels (host arrays): pixel ids, lattice levels (null
  // on the fixed-rate grid) and (Morton, pixel) ranks. Windows follow start() in sequence: t_w is the
  // sensor's time (std::logic_error before start(), std::invalid_argument otherwise). Its events are the sensor's,
  // before the output stage (no crop, sort or rate limiter).
  virtual WindowResult window(int64_t t_w, const uint32_t* pixel, const uint8_t* level, const int64_t* rank,
                              size_t n) = 0;
  // The window [t_w, t_w + W) end to end (ENGINE_CPP §1.3): plan, active pixels, samples, sensing,
  // and the output stage; the events not yet final stay held until a later step() or flush().
  virtual StepResult step(int64_t t_w) = 0;
  // The held events (the end of a run), in output order. With the rate limiter on (sensor.limiter), step() and
  // flush() return its kept events: every bin it decides is final (P2-H H4b).
  virtual std::vector<OutEvent> flush() = 0;
  // The rate limiter's counters since start() (LimiterCounters; on false while it is off).
  virtual LimiterCounters limiter_counters() const = 0;
  // Centre-ray ground truth at pixel positions and times (host arrays), with every mover traced as such.
  virtual std::vector<render::GtSample> gt(const float* x, const float* y, const int64_t* t_ns, size_t n) = 0;

  virtual const sensor::SensorModel& sensor() const = 0;
  // Test hook of the stage audit (docs/dev/P0_PARAMS.md §5), host engine only (std::logic_error on the device):
  // 64-bit hashes of the start frame (every pixel's carried sample from start()) and of the runs of the last window
  // this camera sampled (pixels, offsets, times, irradiance; none once a window of any camera of the engine began
  // since, or the output stage freed them). With the band (EngineConfig::n_band) both also hash every pixel's
  // carried N (the runs' hash once start() has run).
  virtual std::pair<uint64_t, uint64_t> runs_digest() const = 0;
  // Test hooks of P2-B B2, both engines. force_n: every following window samples pixel p (y * width + x) at
  // N = 2^n_log2[p], log2 n_subsamples or log2 n_band, instead of the band's own choice (B4a's tile map); empty: back
  // to it.
  // std::logic_error without the band, std::invalid_argument for another size or value. last_runs: a copy of the runs
  // of the last window this camera sampled, as the sensor read them (an escalated pixel's from its escalation pass,
  // P2-B B4b; runs_digest hashes the same copy), empty when there is none, once a window of any camera of the engine
  // began since, or once the output stage freed them.
  virtual void force_n(const std::vector<uint8_t>& n_log2) = 0;
  virtual WindowRuns last_runs() const = 0;
  // Test hook of P2-B B6c, both engines: the band's map of the last window step() planned, by pixel id (1: N_band),
  // as the plan made it (EngineConfig::band_growth pixel: its pixel map, exact on the crop, beyond it the balls of the
  // seeds within the largest radius of the crop and 0 further out (band_map.h); tile: its tiles' pixels), before the
  // border ring, force_n and the escalation; empty without the band, on the fixed-rate grid and after window().
  virtual std::vector<uint8_t> last_band() const = 0;
  // BL-0b (cancel.h; blender.md §3.8): the token start(), step() and window() check before their work and before
  // sensing, and their windows at each pixel batch and refinement round; a set token throws Cancelled there (abort
  // path only). Null (the default): no check. The token must outlive the calls that read it.
  virtual void set_cancel(const CancelToken* token) = 0;
};

class Engine {
 public:
  virtual ~Engine() = default;
  virtual CameraEngine& camera(size_t i) = 0;
  virtual size_t budget_bytes() const = 0;  // the engine's memory budget (EngineConfig::device_budget_mb)
  virtual bool on_device() const = 0;
};

// The engine of a run; sensors[i]: camera i's sensor, or none for a camera the run does not simulate.
// config.render.backend kOptix builds the device engine (std::invalid_argument without
// BALLA_ENGINE_HAVE_DEVICE), kEmbree the host engine.
std::unique_ptr<Engine> make_engine(render::SceneBundle bundle, EngineConfig config,
                                    const std::vector<std::optional<SensorSpec>>& sensors);

// True if the device engine can run in this process: the build has it (BALLA_ENGINE_HAVE_DEVICE), a CUDA
// device is visible and OptiX is usable. The rule `auto` resolves by (ENGINE_CPP §1.7).
bool device_available();

}  // namespace balla::engine
