// WindowEngine<Space>: the samples of one camera window (ENGINE_CPP §1.2-§1.3, K4-K13): lattice or
// fixed-rate grid, refinement rounds, run assembly, in pixel batches that bound the memory footprint.
// The host and device instantiations differ only in par.h's primitives and buffers; rendering is a
// hook (Renderer::render_into on the device, a Python callable for sampling.py's functions).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/localize.h"
#include "engine/cancel.h"
#include "engine/par.h"
#include "engine/sampling.h"
#include "sensor/waveform.h"

namespace balla::engine {

// Range [min, max] of a light group's waveform over all time (sampling.py waveform_range).
std::pair<double, double> waveform_range(const sensor::Waveform& w);
// eta_L parameters (sampling.py MidpointTest): tolerance, dark current (renderer units) and the
// waveform range of every light group.
MidpointParams midpoint_params(double tol, double dark, const std::vector<std::pair<double, double>>& ranges);

// The economy schedule's switches into r (ENGINE_CPP I7; accuracy.mid_levels, tau_jump_ns, delta_ref, half_ratio and
// eps_jump): none (delta_ref and half_ratio: 0) leaves Phase 1's rule; mid_levels 0 turns the midpoint test off and
// eps_jump 0 still gates (drops flips with eta_J 0). std::invalid_argument outside their ranges.
inline void economy_rules(SampleRules& r, std::optional<int32_t> mid_levels, std::optional<int64_t> tau_jump_ns,
                          double delta_ref, double half_ratio, std::optional<double> eps_jump) {
  if ((mid_levels && *mid_levels < 0) || (tau_jump_ns && (*tau_jump_ns < 1 || *tau_jump_ns > time::kWindowNs)) ||
      !(delta_ref >= 0) || !(half_ratio >= 0 && half_ratio <= 1) || (eps_jump && !(*eps_jump >= 0))) {
    throw std::invalid_argument("economy rules: mid_levels >= 0, tau_jump_ns in [1, 2^23], delta_ref >= 0, "
                                "half_ratio in [0, 1] and eps_jump >= 0");
  }
  r.mid_levels = mid_levels.value_or(INT32_MAX);
  r.tau2 = tau_jump_ns ? 2.0f * static_cast<float>(*tau_jump_ns) : 0.0f;
  r.delta_ref = static_cast<float>(delta_ref);
  r.half_ratio = static_cast<float>(half_ratio);
  r.gate = eps_jump.has_value();
  r.eps_jump = static_cast<float>(eps_jump.value_or(0.0));
}

// Finest level that keeps a window's lattice within `budget` samples (sampling.py level_cap): none
// when the levels fit, else the largest cap >= l_min with sum(2^min(l, cap) + 1) <= budget.
// std::invalid_argument when even the coarsest lattice does not fit.
std::optional<int> level_cap(const int64_t* levels, size_t n, int64_t budget, int l_min);
// The same from the number of pixels at each level (counts[l]), n pixels in all.
std::optional<int> level_cap_counts(std::vector<int64_t> counts, size_t n, int64_t budget, int l_min);
// Samples per pixel of the fixed-rate grid in the window at t_w: the multiples of the period inside
// the window, plus the window end.
uint32_t fixed_rate_samples(int64_t t_w, int64_t period_ns);

// A window that does not fit the device budget even in one-pixel batches, or whose pixel offsets
// alone exceed it (ENGINE_CPP §1.2); raised before the allocation that would exceed it.
class MemoryBudgetError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Renders n requests (pose_slot: the time slot of the window clock): irradiance row out_index[i]
// (groups floats), signature out_index[i] and, when klass is not null, the class word out_index[i]
// (render/types.h kClass*) receive request i. Every request of a call has k_count == subsamples (one N per
// call: P2-B B2's two N classes take a call each). Pointers are in the engine's space.
class RenderHook {
 public:
  virtual ~RenderHook() = default;
  virtual void render(SampleRequest* requests, size_t n, const uint32_t* out_index, float* irr, uint64_t* sig,
                      uint32_t* klass, uint32_t subsamples) = 0;
  // P2-D D3a (SampleConfig::pv): render() with P&V's outputs (render/shade/codes.h PvOut; its rows as irr's), and
  // probes (request i, one sub-sample at a slot of the window clock, into out[i]; Renderer::probe_into). The engine
  // calls them only under SampleConfig::pv; a hook without them refuses.
  virtual void render_pv(SampleRequest*, size_t, const uint32_t*, float*, uint64_t*, uint32_t*, uint32_t,
                         const render::PvOut&) {
    throw std::logic_error("window: this render hook has no P&V outputs (SampleConfig::pv)");
  }
  virtual void probe(SampleRequest*, size_t, render::ProbeOut*) {
    throw std::logic_error("window: this render hook has no probes (SampleConfig::pv)");
  }
};

// Requests go to the renderer in slot order: pixel-major in active-pixel order (the caller's rank
// order), lattice slots by time, refinement midpoints in interval order (ENGINE_CPP §2.3, I1 notes).
struct SampleConfig {
  SampleRules rules;
  int32_t groups = 1;
  uint16_t cam = 0;
  uint32_t subsamples = 64;
  // P2-B B2: the band's N (0: none). With Input::n_log2 a pixel at log2 band_subsamples renders at it, the others at
  // subsamples (both powers of two); each round renders the two classes in a call each, the base class first.
  uint32_t band_subsamples = 0;
  // P2-D D3a (accuracy.localization shadow; pv.n 0: off): P&V in shadow mode on the pixels at N = pv.n (localize.h).
  // Round 0's call at that N writes P&V's outputs, P&V runs beside bisection after round 0's flags and counts
  // (WindowStats::pv); every sample and run stays as without it. Not on the fixed-rate grid. D3b: where P&V found
  // flips, each refinement round's call at that N writes its midpoints' outputs too, for the state check (PvCheck).
  PvParams pv{};
  // BL-0b (cancel.h): checked at each pixel batch and refinement round (Cancelled); null: none.
  const CancelToken* cancel = nullptr;
};

// WindowStats::call_round's entry of a render call whose batch overflowed the budget after rendering and was redone
// (its samples dropped): the call was made and timed, but belongs to no round of the runs.
inline constexpr uint32_t kRedoneCall = 0xFFFFFFFFu;

// Counters of one window (perf.jsonl; sampling.sample_window's stats).
struct WindowStats {
  uint64_t pixels = 0;
  std::vector<uint64_t> samples_by_round;
  uint64_t flagged_signature = 0, flagged_midpoint = 0, refine_cap_hits = 0;
  uint64_t render_calls = 0;  // RenderHook::render calls: one per round and batch, one per N class with per-pixel N
  uint32_t batches = 0;
  std::array<uint64_t, kDiagWords> diag{};  // the P2-0b counters (sampling.h kDiag*; zero without class words)
  uint64_t band_samples = 0;  // P2-B B2: rendered pixel-samples at SampleConfig::band_subsamples
  // P2-C C2: per round its pixel-samples at band_subsamples (zero without per-pixel N), and the round of each
  // RenderHook::render call in call order (batches one after the other; kRedoneCall for the calls of a batch that
  // overflowed after rendering and was redone, so that call_round has one entry per call made, render_calls counting
  // the others); the counters of SampleRules::counters (sampling.h kJump*, kSib*; jump_on: counted, which needs
  // refinement).
  std::vector<uint64_t> band_samples_by_round;
  std::vector<uint32_t> call_round;
  bool jump_on = false;
  std::array<uint64_t, kJumpWords> jump{};
  // P2-D D3a (SampleConfig::pv): P&V's shadow-mode counters (localize.h kPv*; pv_on: counted).
  bool pv_on = false;
  std::array<uint64_t, kPvWords> pv{};
};

template <Space S>
class WindowEngine {
 public:
  // n active pixels at t_w (Space memory), in the order their requests should go to the renderer
  // (the device engine passes them in rank order). Each pixel's carried sample (its value at t_w) is row
  // carried_at[a] (a when null) of carried_irr / carried_sig, which receive its last sample (t_w + W).
  struct Input {
    int64_t t_w = 0;
    const uint32_t* pixel = nullptr;
    const uint8_t* level = nullptr;  // unused by the fixed-rate grid
    size_t n = 0;
    float* carried_irr = nullptr;
    uint64_t* carried_sig = nullptr;
    const uint32_t* carried_at = nullptr;
    uint32_t* carried_class = nullptr;  // the class words of the carried samples (null: no P2-0b counters)
    // P2-B B2: per active pixel the log2 of its N, log2 cfg.subsamples or log2 cfg.band_subsamples (null: every pixel
    // at cfg.subsamples). The carried samples are at that N already (the engine re-renders a switching pixel's).
    const uint8_t* n_log2 = nullptr;
    // P2-B B4a, with carried_class (null: none): per active pixel, set to 1 when a sample of its window (its carried
    // sample at t_w included) has edge bits (sampling.h kBandEdge: two or more clusters or a shadow boundary), left
    // as it is otherwise (the caller zeroes it); the fixed-rate grid flags none.
    uint8_t* edge = nullptr;
    // P2-B B4b (null: none): a pass appended to the window's runs (the escalation pass). Its runs follow those already
    // held (samples() grows; t() and irr() keep their samples) and its pixels' offsets, into the same t() and irr(),
    // go to append_offsets (n + 1 entries, Space memory; [n] the new end) instead of offsets().
    int64_t* append_offsets = nullptr;
    // P2-D D3a (SampleConfig::pv): the carried samples' P&V codes and tables (rows as carried_class; pv.n's code words
    // and the table words each, a table header kPvNone where a row has none; each pixel's last sample's are written
    // back), and the plan data by pixel id: the plan bits (plan.h kPv*; null: none), the pixels' tiles and the tiles'
    // v_hat (null: PvParams::vhat).
    uint32_t* pv_codes = nullptr;
    uint32_t* pv_table = nullptr;
    const uint8_t* pv_plan = nullptr;
    const uint32_t* pixel_tile = nullptr;
    const float* tile_vhat = nullptr;
  };

  // Samples a window into runs(). budget_bytes bounds every byte this engine holds (its buffers, the
  // primitives' temporary storage, and a buffer held twice while it grows): a batch of active pixels
  // whose projected footprint exceeds it is redone as two halves split at its middle pixel (any split
  // is exact), after a retry without the capacity earlier windows left (MemoryBudgetError below one
  // pixel).
  WindowStats run(const Input& in, const SampleConfig& cfg, RenderHook& render, size_t budget_bytes, Stream stream);

  // The window's runs, in active-pixel order: pixel a has samples [offsets[a], offsets[a + 1]) of t
  // (int64 ns) and irr (samples x groups).
  const int64_t* offsets() const { return offsets_.data(); }
  const int64_t* t() const { return t_.data(); }
  const float* irr() const { return fin_irr_.data(); }
  size_t samples() const { return samples_; }
  // Bytes this engine holds: its buffers and the primitives' temporary storage.
  size_t held_bytes() const;
  // Frees every buffer, the runs included (between windows, once the sensor has read them).
  void release();
  // P2-D D3a (a test hook): with keep_pv_flips(true) before run(), the flips of the last batch that ran P&V
  // (localize.h PvFlip; empty: none), downloaded when the stage ends (its device buffers are freed then).
  void keep_pv_flips(bool on) { pv_keep_flips_ = on; }
  const std::vector<PvFlip>& pv_flips() const { return pv_flips_host_; }

 private:
  struct BatchResult {
    WindowStats stats;
    bool overflow = false;
    size_t need = 0;  // the projected footprint that overflowed
  };
  BatchResult run_batch(const Input& in, size_t a0, size_t a1, const SampleConfig& cfg, RenderHook& render,
                        size_t budget, Stream stream);
  size_t footprint(const SampleConfig& cfg, size_t nb, size_t s0, size_t samples, size_t kept, size_t requests,
                   size_t list, size_t candidates, bool headroom) const;
  bool fits(const SampleConfig& cfg, size_t nb, size_t s0, size_t samples, size_t kept, size_t requests, size_t list,
            size_t candidates, size_t budget, BatchResult& out);
  size_t batch_bytes() const;  // what batches hold between them: their buffers and the temporary storage
  void release_batches();
  // P2-D D3a: what round 0's P&V outputs are checked against (the batch's first footprint check).
  struct PvRound0 {
    size_t nb, s0, reserve, budget;
    BatchResult* out;
  };
  // Renders the m requests of a round (out_index_); st counts the render calls and the samples at the band's N. With
  // pv (round 0 under SampleConfig::pv) the call at pv.n writes P&V's outputs; false: they do not fit the budget
  // (nothing rendered, pv->out's overflow set).
  bool render_samples(const SampleConfig& cfg, const Input& in, size_t a0, size_t m, RenderHook& render, Stream stream,
                      WindowStats& st, const PvRound0* pv = nullptr);
  // P2-D D3a: round 0's P&V outputs of `count` requests (fits checked), and P&V's shadow mode after round 0's flags
  // (localize.h; false: the batch overflows).
  bool pv_outputs(const SampleConfig& cfg, const PvRound0& r, size_t count, Stream stream);
  bool localize(const Input& in, size_t a0, const SampleConfig& cfg, RenderHook& render, const PvRound0& r,
                Stream stream, WindowStats& st);
  void pv_release_stage(Stream stream);  // frees what only localize reads, once it ends
  size_t pv_bytes() const;  // footprint's share of the P&V buffers at pv_sz_
  // P2-D D3b: what P&V's functors read of the batch (stage: with J and the thin nodes, localize's own), and the state
  // check of a refinement round's m midpoints from sample `first` (after their render), then the check's buffers freed
  // once the rounds end.
  PvBatch pv_batch(const Input& in, size_t a0, const SampleConfig& cfg, bool stage);
  void pv_check(const Input& in, size_t a0, const SampleConfig& cfg, uint32_t first, size_t m, Stream stream);
  void pv_release_check();

  using P = Par<S>;
  template <class T>
  using Buf = Buffer<T, S>;

  // Window: lattice offsets of all active pixels, the runs.
  Buf<uint64_t> count64_, off64_;
  std::vector<uint64_t> off64_host_;  // only once a batch was split
  uint64_t s0_total_ = 0;
  Buf<int64_t> offsets_, t_;
  Buf<float> fin_irr_;
  size_t samples_ = 0;
  bool runs_reserved_ = false;  // the window is batched and its runs were reserved, or the pass is appended
  bool runs_headroom_ = false;  // the last check let the runs grow with headroom
  bool classes_ = false;        // the window has class words (Input::carried_class)
  bool mixed_ = false;          // the window has per-pixel N (Input::n_log2): its rounds are partitioned by N
  // Batch: samples, requests, flags and intervals, assembly.
  // With mixed_, select_ and index_alt_ also hold a round's N partition and its scan (at most requests + 1 each).
  Buf<uint32_t> off0_, dt_, owner_, before_, out_index_, index_alt_, select_;
  Buf<float> irr_;
  Buf<uint64_t> sig_, keys_, keys_alt_, candidates_, list_;  // keys: the assembly's sort
  Buf<SampleRequest> requests_;
  Buf<uint8_t> sel_;
  Buf<uint32_t> counts_, ex_;
  Buf<unsigned long long> counters_;
  Buf<uint32_t> cls_;            // the samples' class words (with classes_)
  Buf<unsigned long long> diag_;  // the P2-0b counters: the window's, then the current batch's
  // P2-C C2 (SampleRules::counters): the listed intervals' outer nodes and tags, the candidates' outer nodes, and the
  // counters (the window's, then the current batch's).
  Buf<uint64_t> outer_, cand_outer_;
  Buf<uint8_t> tag_;
  Buf<unsigned long long> jump_;
  // P2-D D3a (SampleConfig::pv): round 0's outputs (slot per sample, entries' codes and tables, the entry count), per
  // slot the route bits, J, R2's thin nodes, the flips and (flips, localized, probes) of each interval, per pixel the
  // flip counts and offsets, the flips, their groups' fits, the rounds' requests and results, and the counters (the
  // window's, then the current batch's). pv_sz_: the sizes the current batch needs (footprint; stage: localize's own
  // arrays, J, thin nodes, flip counts and offsets, from round 0's check until the stage ends and frees them with the
  // flips' arrays), pv_r0_: round 0 wrote.
  // D3b: the groups' fits (ng), the nodes' sort (sort: after the rounds, in place of their requests and results), the
  // brackets kept through the refinement rounds with the pixels' flip offsets (brackets), each round's midpoint
  // outputs (mid: its requests) and the output rows' entries (rows: the lattice's, then the round's samples too).
  struct PvSizes {
    size_t slots = 0, entries = 0, code_words = 0, table_words = 0, pixels = 0, flips = 0, groups = 1;
    size_t ng = 0, brackets = 0, mid = 0, rows = 0;
    bool stage = false, sort = false;
  };
  Buf<uint32_t> pv_slot_, pv_codes_, pv_table_, pv_count_, pv_route_, pv_flipc_, pv_stats_, pv_off_, pv_nreq_, pv_reqoff_;
  Buf<float> pv_j_;
  Buf<uint8_t> pv_thin_;
  Buf<PvFlip> pv_flips_;
  Buf<PvFit> pv_fits_;
  Buf<SampleRequest> pv_req_;
  Buf<render::ProbeOut> pv_probe_;
  Buf<unsigned long long> pv_counts_;
  Buf<PvBracket> pv_brk_;                                  // D3b
  Buf<uint32_t> pv_mcodes_, pv_mtable_, pv_mcount_, pv_nval_;  // D3b: the midpoints' outputs, the nodes' sort values
  Buf<uint64_t> pv_nkey_;                                  // D3b: the nodes' sort keys
  PvSizes pv_sz_;
  render::PvOut pv_out_{};
  render::PvOut pv_mid_out_{};  // D3b: the refinement round's midpoint outputs (pv_mid_: its band call writes them)
  bool pv_r0_ = false;
  bool pv_mid_ = false;
  bool pv_keep_flips_ = false;          // keep_pv_flips (a test hook)
  std::vector<PvFlip> pv_flips_host_;  // pv_flips()
  Temp<S> temp_;
  HostWords<S> words_;  // [0] a select's count, [1..3] a round's flag counters
};

extern template class WindowEngine<Space::kHost>;
#if BALLA_ENGINE_HAVE_DEVICE
extern template class WindowEngine<Space::kDevice>;
#endif

}  // namespace balla::engine
