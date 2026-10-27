// The motion session's per-body pipelines and their round driver (docs/design/phase3/blender.md §3.8, §4.7.3, §4.7.5).
//
// A BodyPipeline is a resumable state machine that never calls the oracle: it says which times it needs this round
// (requests), takes the answers at those times (feed) and does the pure work they enable (step). The round driver
// merges every unfinished pipeline's requests (one oracle call evaluates every body), converts them with the TimeMap,
// sends them in messages of at most SampleOptions::max_per_message (<= 512) samples, decodes and weighs the answers,
// gives each pipeline its own column and runs the steps on a thread pool. Requests depend only on the pipelines'
// states, so the oracle log and every result are bitwise independent of the worker count.
//
// Blender's key plateau (a key's value within 1e-4 frames of it, so a CONSTANT step comes 1e-4 frames early) is the
// headless oracles' measurement (MF E8, blender.md §2.3; PHASE3.md §0 rule 4), not taken from Blender's sources.
//
// Keyed mode (stages kBase .. kBurst):
// - kBase, kRefine (BL-6, §4.7.3): the base grid with the declared keys, then level-synchronous midpoint refinement.
//   The aliasing pre-pass (|dtheta| < 0.6 pi between samples at least alias_per_frame per frame) is the base grid with
//   the first round's midpoints: the base grid holds at least half that density, and a midpoint is accepted only when
//   both of its halves turn by less than 0.6 pi, so every interval that turns faster is split (§7: one call per sample
//   of that density, not two).
// - kDetect, kBurst (BL-7, §4.7.5): candidates (unresolved intervals, spacing spikes, the node scan, declared step and
//   C0 keys) are classified on the samples (detect.h); a value step is confirmed by pose-distance bisection (kDetect,
//   one sample per step per round) and a kink localized by bursts (kBurst, every kink's burst of one level in the same
//   round); then the split significance filter, the merge and the snap to declared keys. A confirmed step refuses the
//   body (MOT-STEP) until BL-12 unless DetectOptions::refuse_steps is off.
// - kJoin, kFill, kValidate (BL-8, §4.7.6, fit_piecewise.h): the pieces (breakpoints and hold runs), a sample at each
//   piece begin the samples lack (kJoin: a piece then begins at that sample's realized time), the knot-rate fill
//   (every gap of a piece <= h/2), the validation probes (FitOptions::probes seeded per piece, and the breakpoint
//   probes at b- = b - w - max(1 us, 2 q(F)): off the key's plateau and its own sample on Blender's float32 time
//   grid; the probe at b is the sample there), then the fits left to right, each accepted when
//   every validation sample (V, the probes, the sample at its begin, a hold piece's own samples) errs by at most
//   accept_frac eps + its float32 floor. A failing piece is charged to its left neighbour when the probe at its begin
//   fails (the error a C0 join inherits, DS d14), or when it is a hold; the charged piece is split once at the
//   declared keys inside it (C0 joins; not the CONSTANT keys, which are steps), then its knot spacing halved
//   (FitOptions::halvings), then MOT-FIT (naming the validation error, or the C1 increment that broke the pi/2 rule).
//   The pieces from the charged one on are refitted after the round that fills and probes them.
// Frame-state sampling (BL-10) extends it.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "render/thread_pool.h"
#include "scene/motion/error_model.h"
#include "scene/motion/fit_piecewise.h"
#include "scene/motion/keys.h"
#include "scene/motion/poses.h"
#include "scene/motion/time_map.h"
#include "scene/motion/track.h"

namespace balla::scene::motion {

struct SampleOptions {
  double eps_px = 0.01;        // scene.blender.motion_tol_px: eps/4 for sampling (§4.7.2)
  int64_t h0_ns = 0;           // base spacing; 0: P / max(2, ceil(P / 10.4 ms)), frame-aligned (h0_ns below)
  int64_t h_min_ns = 16384;    // 2^14 ns, the lattice (DESIGN.md:881)
  int max_rounds = 40;
  int max_per_message = 512;   // samples per `sample` message, 1 .. 512 (§3.5); drive takes its pipelines' smallest
  int alias_per_frame = 8;     // the aliasing pre-pass: |dtheta| < 0.6 pi between samples this dense
  // the base grid's size cap per body (2^24: 16.8 M samples, about 80 min of Blender time at 0.29 ms per call, §7):
  // a frame period this short for the span (fps_base can be 1e-5) is refused before anything is allocated
  int64_t max_base_samples = int64_t{1} << 24;
  // The span the pipeline samples and fits: [-2 W, TimeMap::t_end_ns()] unless set. The session widens it
  // (PHASE3.md D-P3-12: [-5 W, t_end + ceil(m / W) W]) so the trajectories also serve the IMU's filter warm-up and the
  // frames' exposure margins.
  std::optional<int64_t> t0_ns, t1_ns;
};

// The base grid's spacing h0 (§4.7.3) unless SampleOptions::h0_ns sets it: P / max(2, ceil(P / 10.4 ms)), the
// frame-aligned spacing at most min(P / 2, 10.4 ms = 1/96 s) (8.33 ms at 30 fps, not 10.4 ms). Detection's split
// filter and BL-8's knot cap (2 h0, §4.7.6) take h0 from here, not from the min formula.
int64_t h0_ns(const TimeMap& time, const SampleOptions& opt);

// Detection's options (§4.7.5; the scene.blender.detect_* rows of §3.11 once BL-3b binds them).
struct DetectOptions {
  double k_sig = 6.0;              // detect_k_sigma: slope or value jumps above this many standard errors
  double unsplit_coeff = 0.1;      // an unsplit kink costs about 0.1 dv h (MF a6c, t_syn): split when > eps/8
  int burst_n = 8;                 // detect_burst_samples: samples per side of one burst (>= 3)
  int burst_iters = 4;             // detect_burst_rounds: bursts per kink, the window shrinking 4x each
  int64_t burst_floor_ns = 64000;  // detect_burst_floor_ns: the smallest burst window
  int64_t merge_ns = 100000;       // detect_merge_ns: candidates and breakpoints closer than this are merged
  int64_t h_cap_ns = 0;            // the knot spacing of the significance filter; 0: 2 h0_ns() (the fit's cap, §4.7.6)
  bool refuse_steps = true;        // MOT-STEP for a confirmed step (v1); BL-12 lifts it
};

struct DetectStats {  // per body, for session_perf.json and the tests
  int candidates = 0;                 // candidate times
  int smooth = 0, unclear = 0;        // candidates classified as neither a step nor a kink
  int steps = 0, kinks = 0;           // confirmed steps, kinks that pass the split filter (before the merge)
  int demoted = 0, insignificant = 0; // steps whose jump vanished under bisection; kinks below the split filter
  int snapped = 0;                    // breakpoints snapped to a declared key
  bool operator==(const DetectStats&) const = default;
};

enum class Mode : uint8_t { kKeyed, kFrameState };
enum class Stage : uint8_t { kBase, kRefine, kDetect, kBurst, kJoin, kFill, kValidate, kDone };
const char* stage_name(Stage s);

struct SampleSet {
  Track fit;                       // base grid, keys and split points; samples strictly inside a plateau dropped
  Track held_out;                  // V: the accepted midpoints, ascending in time like fit
  std::vector<double> residual_px; // V's residuals (the accepting stencil's), in V's order
  std::vector<std::pair<int64_t, int64_t>> unresolved;  // failing at h_min, or past the time grid: candidates
  double floor_px = 0.0;           // the largest floor_px at an accepted midpoint
  double alias_max_rad = 0.0;      // the largest |dtheta| between consecutive samples of fit and V together
  int alias_warnings = 0;          // such intervals in [0.6 pi, 0.9 pi] that h_min (or the time grid) left unsplit
  bool light_near = false;
  int rounds = 0;                  // refinement rounds
  std::vector<Breakpoint> breakpoints;  // §4.7.5, ascending
  DetectStats detect;
  bool fitted = false;             // FitOptions::enabled: the fit below (BL-8, §4.7.6)
  PiecewiseFit motion;
  Track probes;                    // the validation probes answered (seeded and at the breakpoints), ascending
};

// POS-RES (§4.7.2): floor_px > eps/8 at an accepted sample; the body's bound is then eps/2 + 2 floor_px.
struct PosRes {
  double floor_px, bound_px;
};
std::optional<PosRes> pos_res(const SampleSet& set, double eps_px);

// --- pure pieces (unit-tested directly; detection's in detect.h) ---
// The base requests over [t0, t1]: a frame-aligned grid of max(ceil(P / h0), ceil(alias_per_frame / 2)) samples per
// frame (the h0 grid; with the first round's midpoints it holds the aliasing pre-pass's density), t0, t1, the declared
// keys and, before every key that ends a CONSTANT segment (hint kStep), one sample at key - plateau - 2 q(F).
// Ascending, unique. std::invalid_argument, naming the frame period, when the grid and the keys would exceed
// SampleOptions::max_base_samples.
std::vector<int64_t> base_requests(const TimeMap& time, int64_t t0, int64_t t1, std::span<const DeclaredKey> keys,
                                   const SampleOptions& opt);
// The midpoint test of §4.7.3 for the sample m of `mid` in (fit[i], fit[i+1]): three cubic Lagrange stencils (in R^3
// for position, in Log(q_i^T q) for rotation), in the order centred, left, right; accepted when a stencil errs by at
// most eps/4 + g * max floor1 (g = 1 + sum |w_k| of that stencil). Returns the accepting (or best) residual.
struct MidpointTest {
  bool accepted = false;
  double residual_px = 0.0, floor_px = 0.0;
};
MidpointTest midpoint_test(const Track& fit, size_t i, const Track& mid, size_t m, double eps_px);

// The declared plateaus of a key list (§4.7.4): a sample strictly inside one (farther from its key than the key's time
// resolution, at most plateau_frames P away) shows the key's value, not the motion.
class Plateaus {
 public:
  Plateaus(const TimeMap& time, std::span<const DeclaredKey> keys);
  bool inside(int64_t t) const;
  // 2 w summed over the plateaus with samples strictly inside them in (a, c): a bracket of the fit samples (which have
  // none) is that much wider than the sampler left it.
  int64_t dropped_ns(int64_t a, int64_t c) const;

 private:
  struct P {
    int64_t t, w, tol;
  };
  std::vector<P> p_;  // ascending in t
  int64_t w_max_ = 0;
};

class BodyPipeline {
 public:
  // body: the oracle's body column. Samples [-2W, t_end] of the TimeMap (the bundle span, §4.2).
  BodyPipeline(int32_t body, Mode mode, const TimeMap& time, std::vector<DeclaredKey> keys, const SampleOptions& opt,
               const DetectOptions& detect = {}, const FitOptions& fit = {});
  int32_t body() const { return body_; }
  const SampleOptions& options() const { return opt_; }
  std::vector<int64_t> requests() const { return pending_; }  // this round's target times: ascending, unique
  void feed(const Track& answers);  // the answers to requests(), in that order (realized times)
  void step();                      // pure, deterministic; pool-safe
  bool done() const { return stage_ == Stage::kDone; }
  Stage stage() const { return stage_; }
  SampleSet take() &&;              // after done()

 private:
  // one step or kink being localized (§4.7.5)
  struct Loc {
    Continuity cls;
    int64_t t;                 // the current estimate
    double jump_px, dv_pxps;
    double w_pos, w_rot, f1;   // the signal's weights and the float32 floor (g = 1) near the candidate
    double v_pxps;             // a step's larger one-sided speed (the pose distance's rate), px/s
    int64_t a, c;              // a step's bracket (a, c] and its end poses
    rot::Q qa, qc;
    rot::V pa, pc;
    int iters = 0;             // bisection steps, or bursts done
    int64_t w_ns = 0;          // a kink's burst window
    bool active = true;
    std::vector<int64_t> req;  // this round's target times
  };
  void insert(const Track& add);
  void finish_sampling();
  void start_detect();
  void start_kink(Loc& l);
  void end_step(Loc& l);
  void step_bisect();
  void start_bursts();
  void step_burst();
  void finish_detect();
  void set_pending();
  size_t answer_of(int64_t target) const;
  // the fit (§4.7.6)
  struct Piece {
    FitPiece fit;               // its plan (begin, end, join, hold, h, halvings, split) and, once accepted, controls
    int64_t h_rule = 0;         // §4.7.6's spacing (2 x median, clamped; halved on failure); fit.h_ns aligns it
    bool need_fill = true, need_probes = true;
  };
  void start_fit();
  void step_join();
  void enter_fill();
  void enter_validate();
  void validate_pieces();
  // why a piece failed: a validation error (value: px at t_bad), the pi/2 rule (value: the largest increment, rad) or
  // too few samples for its knots (the fit was singular)
  enum class Failure : uint8_t { kError, kIncrement, kUnderdetermined };
  void charge(size_t j, int64_t t_bad, double value, Failure why);
  void finish_fit();
  int64_t plateau_ns(int64_t t) const;
  bool has_sample(int64_t t) const;
  int32_t body_;
  TimeMap time_;
  std::vector<DeclaredKey> keys_;
  SampleOptions opt_;
  DetectOptions dopt_;
  Plateaus plateaus_;
  int64_t t0_, t1_;
  Stage stage_ = Stage::kBase;
  std::vector<int64_t> pending_;
  std::vector<std::pair<int64_t, int64_t>> open_;  // the intervals whose midpoints pending_ holds
  std::vector<Loc> locs_;
  std::vector<Breakpoint> found_;
  Track answers_;
  bool fed_ = false;
  SampleSet set_;
  FitOptions fopt_;
  std::vector<Piece> pieces_;
  size_t dirty_ = 0;                 // the first piece to (re)fit
  std::vector<int64_t> join_req_;    // kJoin: the piece begins requested, in pending_'s order
};

// --- the oracle and the round driver ---
// Moves bytes over the pipe only: one `sample` message (n <= max_per_message <= 512) and its `poses` answer.
using Oracle = std::function<RawPoses(std::span<const int32_t> F, std::span<const float> s)>;

struct RoundStats {
  int64_t rounds = 0, calls = 0, messages = 0;
  std::map<std::string, int64_t> calls_by_stage;  // each call counted once, for the first pipeline requesting it
  bool operator==(const RoundStats&) const = default;
};

// Runs every pipeline to done(), in messages of at most the smallest max_per_message of the pipelines' options.
// VIS-ANIM when a sampled body is absent or an unknown instance appears; BRG-TIME, BRG-PROTO and GEO-SCALEANIM from
// decode; a pipeline's error is rethrown for the lowest failing pipeline. on_batch sees every round's decoded and
// weighed answers, all body columns (the rig check, checks.h check_rigs: CAM-RIG).
// name_of(column) names a body in the error that ends the run (the pipeline's, "(body NAME)" appended).
RoundStats drive(const Oracle& oracle, const TimeMap& time, const ErrorModel& model, std::span<BodyPipeline> pipelines,
                 render::ThreadPool& pool, const std::function<void(double, std::string_view)>& progress = {},
                 const std::function<void(const PoseBatch&)>& on_batch = {},
                 const std::function<std::string(int32_t)>& name_of = {});

// --- the recorded oracle (blender.md §3.6 oracle_log.npz) ---
struct OracleLog {
  std::vector<RawPoses> calls;  // every message in order: the requested (F, s) and the raw answers
};
// Answers through `inner` and appends each call to `log` (which must outlive the oracle).
Oracle recording(Oracle inner, OracleLog& log);
// Answers from `log` (which must outlive the oracle); BRG-REPLAY at the first request that differs from the log's or
// goes past its end. *calls counts the calls answered.
Oracle replay(const OracleLog& log, std::shared_ptr<size_t> calls);

}  // namespace balla::scene::motion
