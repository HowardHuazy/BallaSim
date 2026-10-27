// The Engine and its CameraEngines on one memory space (engine.h; ENGINE_CPP §1.3, §1.7): the window's
// launch sequence. step() plans (H0, K1 plan pass, K2 pixel stage, H1 tile stages, K4 active list), samples
// (K5-K13), senses (S) and runs the output stage (K14-K15); only the tile results and the final events cross
// back to the host. window() samples active pixels given by the caller (I1's form, for tests).
//
// Included by engine.cu (the device instantiation: OptiX, the CUDA sensor, CUB) and engine_host.cpp (the
// host instantiation: Embree, the CPU sensor, serial primitives) only; the two differ in par.h's primitives
// and the renderer's and sensor's backends, never in the code below.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "core/camera/relative_illumination.h"
#include "core/camera/solid_angle.h"
#include "core/time/time.h"
#include "engine/band_map.h"
#include "engine/engine.h"
#include "engine/window_impl.h"
#include "render/shade/lights.h"
#include "sensor/limiter.h"

namespace balla::engine {

namespace engine_detail {

// Internal linkage in each including translation unit: BALLA_HD expands differently under nvcc (engine.cu)
// and g++ (engine_host.cpp), so no definition below may be shared between them (one-definition rule).
namespace {

using Clock = std::chrono::steady_clock;

// DESIGN §8.4's minimum of a static warm-up (resolve.cpp's validate_run refuses a shorter static one).
constexpr int64_t kMinStaticWarmupNs = 50'000'000;

inline double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct Iota {
  uint32_t* out;
  BALLA_HD void operator()(size_t i) const { out[i] = static_cast<uint32_t>(i); }
};

// The active pixels in rank order (K4 of window()).
struct GatherActive {
  const uint32_t* order;
  const uint32_t* pixel;
  const uint8_t* level;
  uint32_t* act_pixel;
  uint8_t* act_level;
  BALLA_HD void operator()(size_t i) const {
    act_pixel[i] = pixel[order[i]];
    if (level != nullptr) act_level[i] = level[order[i]];
  }
};

// K4 of step(): pixel by rank, inside the crop and on a dynamic tile (every tile on the fixed-rate grid).
struct ActiveFlags {
  const uint32_t* pixel_by_rank;
  const uint32_t* pixel_tile;
  const int8_t* tile_level;  // null on the fixed-rate grid
  uint32_t width;
  Crop crop;
  uint8_t* flags;
  BALLA_HD void operator()(size_t r) const {
    const uint32_t p = pixel_by_rank[r];
    const auto x = static_cast<int32_t>(p % width), y = static_cast<int32_t>(p / width);
    const bool in = x >= crop.x0 && x < crop.x1 && y >= crop.y0 && y < crop.y1;
    flags[r] = (in && (tile_level == nullptr || tile_level[pixel_tile[p]] >= 0)) ? 1 : 0;
  }
};

struct GatherSelected {
  const uint32_t* select;
  const uint32_t* pixel_by_rank;
  const uint32_t* pixel_tile;
  const int8_t* tile_level;
  int32_t cap;
  uint32_t* act_pixel;
  uint8_t* act_level;
  BALLA_HD void operator()(size_t i) const {
    const uint32_t p = pixel_by_rank[select[i]];
    act_pixel[i] = p;
    if (tile_level != nullptr) {
      const int32_t l = tile_level[pixel_tile[p]];
      act_level[i] = static_cast<uint8_t>(l < cap ? l : cap);
    }
  }
};

// K1: the plan pass's centre rays at t_w, every pixel.
struct PlanQueries {
  int64_t t;
  uint32_t width;
  render::GtQuery* q;
  BALLA_HD void operator()(size_t i) const {
    q[i] = {t, static_cast<float>(i % width), static_cast<float>(i / width)};
  }
};

struct FrameRequests {
  uint16_t cam;
  uint32_t subsamples;
  SampleRequest* requests;
  BALLA_HD void operator()(size_t p) const { requests[p] = {cam, 0, static_cast<uint32_t>(p), 0, subsamples, 0, 0}; }
};

// The plan's kappa (StepResult::kappa): per active pixel whose plan-pass centre ray hits, its |flow| at t_w, its tile's
// v_hat and its lattice's speed 2^level d_px / W, in 1/16 px/s (exact integer sums).
struct PlanCounts {
  const uint32_t* pixel;
  const uint8_t* level;
  const uint32_t* pixel_tile;
  const float* v_hat;
  const render::GtSample* gt;
  double lattice;  // d_px / W, px/s
  unsigned long long* out;
  BALLA_HD void operator()(size_t a) const {
    const uint32_t p = pixel[a];
    const render::GtSample g = gt[p];
    const double fx = g.flow_x, fy = g.flow_y, f = sqrt(fx * fx + fy * fy);
    const double v = v_hat[pixel_tile[p]];
    const bool ok = isfinite(g.depth) && isfinite(f) && isfinite(v);
    const auto q = [ok](double x) { return ok ? static_cast<unsigned long long>(llrint(16.0 * x)) : 0ull; };
    count_if(&out[0], ok);
    add_count(&out[1], q(f));
    add_count(&out[2], q(v));
    add_count(&out[3], q(lattice * static_cast<double>(1u << level[a])));
  }
};

struct RunCounts {
  const int64_t* offsets;
  int32_t* counts;
  BALLA_HD void operator()(size_t a) const { counts[a] = static_cast<int32_t>(offsets[a + 1] - offsets[a]); }
};

struct FillBytes {
  uint8_t* out;
  uint8_t value;
  BALLA_HD void operator()(size_t i) const { out[i] = value; }
};

// P2-B B2: the band's switch words (SwitchStats), counted on the device: [0] active pixels at N_band, [1] unquiescent
// switches, [2 ...] the steps' bins; B4a: then the edge pixels at N_band and at N_base; B4b: the switches whose value
// changed, then the escalated pixels' t_w steps (changed, unquiescent, bins); B6a: the pixels escalated by pooling
// alone and those the border ring alone puts at N_band; B6c: the active pixels the plan's band puts at N_band.
inline constexpr int kSwitchBandPixels = 0, kSwitchUnquiescent = 1, kSwitchHist = 2, kSwitchBandEdge = 2 + kDiagBins,
                     kSwitchBaseEdge = 3 + kDiagBins, kSwitchChanged = 4 + kDiagBins, kEscChanged = 5 + kDiagBins,
                     kEscUnquiescent = 6 + kDiagBins, kEscHist = 7 + kDiagBins, kEscPooled = 7 + 2 * kDiagBins,
                     kSwitchBorder = 8 + 2 * kDiagBins, kSwitchPlan = 9 + 2 * kDiagBins,
                     kSwitchWords = 10 + 2 * kDiagBins;

// A t_w node's step from `old` to `now` (a switch's or an escalation's re-render, at pixel p): eta_J in units of `unit`
// (refine_tol_log / 8, eps_L / 8 where the two agree; run.json's eps_switch) into the bins of sampling.h count_jump at
// counts[hist], counts[unquiescent] above 1, counts[changed] when any bit differs.
BALLA_HD inline void count_step(const MidpointParams& mid, uint32_t p, const float* old, const float* now, float unit,
                                int32_t groups, unsigned long long* counts, int unquiescent, int hist, int changed) {
  const float x = endpoint_eta(mid, mid.dark_of(p), old, now) / unit;
  int32_t bin = 0;
  for (float edge = 1.0f; bin < kDiagBins - 1 && x >= edge; edge *= 2.0f) ++bin;
  count_if(&counts[unquiescent], x > 1.0f);
  for (int32_t k = 0; k < kDiagBins; ++k) count_if(&counts[hist + k], bin == k);
  bool differ = false;
  for (int32_t g = 0; g < groups; ++g) differ = differ || old[g] != now[g];
  count_if(&counts[changed], differ);
}

// Each active pixel's log2 N: the forced map's (CameraEngine::force_n), else the plan's (P2-B B4a: N_band in its band
// tiles, N_base elsewhere; B6c: in its band pixels instead under band_growth pixel; B6a: and N_band on the crop's
// outermost 1 px ring, whose pixels have neighbours beyond it that pooled escalation cannot read), else N_band (the
// fixed-rate grid, window()). Counted: the plan's band (kSwitchPlan, forced or not), the ring alone, N_band.
struct ActiveN {
  const uint32_t* pixel;
  const uint8_t* forced;      // by pixel id (null: none)
  const uint8_t* tile_band;   // by tile (null with pixel_band null: band_log2 for every pixel)
  const uint8_t* pixel_band;  // P2-B B6c: by pixel id (null: tile_band)
  const uint32_t* pixel_tile;
  uint32_t width;
  Crop crop;  // the image without one
  uint8_t base_log2, band_log2;
  uint8_t* act_n;
  unsigned long long* counts;
  BALLA_HD void operator()(size_t i) const {
    const uint32_t p = pixel[i];
    const bool planned = pixel_band != nullptr || tile_band != nullptr;
    const bool in_band = pixel_band != nullptr ? pixel_band[p] != 0
                         : tile_band != nullptr ? tile_band[pixel_tile[p]] != 0
                                                : true;
    count_if(&counts[kSwitchPlan], planned && in_band);
    uint8_t n = forced != nullptr ? forced[p] : in_band ? band_log2 : base_log2;
    if (forced == nullptr && n != band_log2) {
      const auto x = static_cast<int32_t>(p % width), y = static_cast<int32_t>(p / width);
      const bool ring = x == crop.x0 || x + 1 == crop.x1 || y == crop.y0 || y + 1 == crop.y1;
      n = ring ? band_log2 : n;
      count_if(&counts[kSwitchBorder], ring);
    }
    act_n[i] = n;
    count_if(&counts[kSwitchBandPixels], n == band_log2);
  }
};

// P2-B B4a: the window's edge pixels (WindowEngine::Input::edge) by N class.
struct EdgeCounts {
  const uint8_t* act_n;
  const uint8_t* edge;
  uint8_t band_log2;
  unsigned long long* counts;
  BALLA_HD void operator()(size_t i) const {
    count_if(&counts[kSwitchBandEdge], edge[i] != 0 && act_n[i] == band_log2);
    count_if(&counts[kSwitchBaseEdge], edge[i] != 0 && act_n[i] != band_log2);
  }
};

// The active pixels that switch to `to` (log2 N): their carried sample has another N.
struct SwitchFlags {
  const uint32_t* pixel;
  const uint8_t* act_n;
  const uint8_t* carried_n;  // by pixel id
  uint8_t to;
  uint8_t* flags;
  BALLA_HD void operator()(size_t i) const { flags[i] = act_n[i] == to && carried_n[pixel[i]] != to ? 1 : 0; }
};

// Switch j: active pixel sel[j] (j < k0, to N_base) or sel[n + j - k0] (to N_band).
struct SwitchList {
  const uint32_t* sel;
  size_t n;
  uint32_t k0;
  BALLA_HD uint32_t operator()(size_t j) const { return j < k0 ? sel[j] : sel[n + j - k0]; }
};

// The switches' t_w nodes: slot 0 of the window clock, at the new N.
struct SwitchRequests {
  SwitchList list;
  const uint32_t* pixel;
  uint16_t cam;
  uint32_t base, band;  // N
  SampleRequest* requests;
  BALLA_HD void operator()(size_t j) const {
    requests[j] = {cam, 0, pixel[list(j)], 0, j < list.k0 ? base : band, 0, 0};
  }
};

// A switch's re-rendered t_w node becomes the pixel's carried sample (row = pixel id), at its new N; its step from the
// old carried value is counted (count_step).
struct SwitchApply {
  SwitchList list;
  const uint32_t* pixel;
  const uint8_t* act_n;
  MidpointParams mid;
  float unit;
  int32_t groups;
  const float* irr;  // the re-rendered nodes, row j
  const uint64_t* sig;
  const uint32_t* cls;
  float* carried_irr;
  uint64_t* carried_sig;
  uint32_t* carried_class;
  uint8_t* carried_n;
  unsigned long long* counts;
  BALLA_HD void operator()(size_t j) const {
    const uint32_t i = list(j), p = pixel[i];
    float* old = carried_irr + static_cast<size_t>(p) * groups;
    const float* now = irr + j * static_cast<size_t>(groups);
    count_step(mid, p, old, now, unit, groups, counts, kSwitchUnquiescent, kSwitchHist, kSwitchChanged);
    for (int32_t g = 0; g < groups; ++g) old[g] = now[g];
    carried_sig[p] = sig[j];
    carried_class[p] = cls[j];
    carried_n[p] = act_n[i];
  }
};

// P2-B B6a: the window's edge pixels (s.edge, by active index) as a byte image by pixel id, cleared before.
struct EdgeImage {
  const uint32_t* pixel;
  const uint8_t* edge;
  uint8_t* image;
  BALLA_HD void operator()(size_t i) const {
    if (edge[i] != 0) image[pixel[i]] = 1;
  }
};

// P2-B B4b, escalation (Camera::escalate): the active pixels at N_base with an edge sample in their window; B6a
// (pooled): or with an active 8-neighbour that has one (the edge image), counted (kEscPooled) when a neighbour's alone
// does. The fixed 3 x 3 test closes T2's tail gap only while the 64 points' hulls reach past +-0.5 px (escalate()).
struct EscalateFlags {
  const uint8_t* act_n;
  const uint8_t* edge;
  const uint32_t* pixel;
  const uint8_t* image;  // EdgeImage's
  int32_t width, height;
  uint8_t band_log2;
  uint8_t* flags;
  unsigned long long* counts;
  BALLA_HD void operator()(size_t i) const {
    bool any = false;
    if (act_n[i] != band_log2) {
      const auto p = static_cast<int32_t>(pixel[i]);
      const int32_t x = p % width, y = p / width;
      for (int32_t v = y > 0 ? y - 1 : y; v <= y + 1 && v < height; ++v) {
        for (int32_t u = x > 0 ? x - 1 : x; u <= x + 1 && u < width; ++u) any = any || image[v * width + u] != 0;
      }
    }
    flags[i] = any ? 1 : 0;
    count_if(&counts[kEscPooled], any && edge[i] == 0);
  }
};

// The escalated pixels (active pixels sel[j]) and their lattice levels, in rank order.
struct GatherEscalated {
  const uint32_t* sel;
  const uint32_t* act_pixel;
  const uint8_t* act_level;
  uint32_t* pixel;
  uint8_t* level;
  BALLA_HD void operator()(size_t j) const {
    pixel[j] = act_pixel[sel[j]];
    level[j] = act_level[sel[j]];
  }
};

// The escalated pixels' t_w nodes at N_band: slot 0 of the window clock.
struct NodeRequests {
  const uint32_t* pixel;
  uint16_t cam;
  uint32_t subsamples;
  SampleRequest* requests;
  BALLA_HD void operator()(size_t j) const { requests[j] = {cam, 0, pixel[j], 0, subsamples, 0, 0}; }
};

// An escalated pixel's re-rendered t_w node becomes its carried sample (row = pixel id), at N_band. Unless the pixel
// switched down in this window (the window's switch list down[0, k0), ascending), the t_w node the window's pass used
// (slot 0 of its run, main_irr row main_offsets[sel[j]]) is its carried end node of the previous window at N_base, and
// the step from it is counted (count_step, the kEsc words): T2's gap nodes that an escalation, not an up-switch,
// re-renders (ENGINE_CPP §0.1 "P2-B B4b notes"). A switch down's step is SwitchApply's, from an N_band node.
struct NodeApply {
  const uint32_t* pixel;
  const uint32_t* sel;
  const uint32_t* down;
  uint32_t k0;
  const int64_t* main_offsets;
  const float* main_irr;
  MidpointParams mid;
  float unit;
  int32_t groups;
  const float* irr;
  const uint64_t* sig;
  const uint32_t* cls;
  uint8_t band_log2;
  float* carried_irr;
  uint64_t* carried_sig;
  uint32_t* carried_class;
  uint8_t* carried_n;
  unsigned long long* counts;
  BALLA_HD void operator()(size_t j) const {
    const uint32_t p = pixel[j], a = sel[j];
    const float* now = irr + j * static_cast<size_t>(groups);
    uint32_t lo = 0, hi = k0;  // a in down[0, k0)?
    while (lo < hi) {
      const uint32_t mid_i = lo + (hi - lo) / 2;
      if (down[mid_i] < a) {
        lo = mid_i + 1;
      } else {
        hi = mid_i;
      }
    }
    if (lo == k0 || down[lo] != a) {
      count_step(mid, p, main_irr + static_cast<size_t>(main_offsets[a]) * groups, now, unit, groups, counts,
                 kEscUnquiescent, kEscHist, kEscChanged);
    }
    for (int32_t g = 0; g < groups; ++g) carried_irr[static_cast<size_t>(p) * groups + g] = now[g];
    carried_sig[p] = sig[j];
    carried_class[p] = cls[j];
    carried_n[p] = band_log2;
  }
};

// The sensor's rows of the escalated pixels (active pixels sel[j]) point to the escalation pass's runs, and their N
// (act_n) becomes N_band's.
struct Redirect {
  const uint32_t* sel;
  const int64_t* offsets;  // the escalation pass's, k + 1
  uint8_t band_log2;
  int64_t* row_start;
  int32_t* row_count;
  uint8_t* act_n;
  BALLA_HD void operator()(size_t j) const {
    const uint32_t a = sel[j];
    row_start[a] = offsets[j];
    row_count[a] = static_cast<int32_t>(offsets[j + 1] - offsets[j]);
    act_n[a] = band_log2;
  }
};

// P2-D D3a: a re-rendered t_w node's P&V codes and table (call row j - first; slot null or j < first: none) become its
// pixel's carried ones (row(j), a pixel id), so that interval 1 of the next pass reads them (localize.h).
struct SwitchRow {
  SwitchList list;
  const uint32_t* pixel;
  BALLA_HD uint32_t operator()(size_t j) const { return pixel[list(j)]; }
};
struct DirectRow {
  const uint32_t* pixel;
  BALLA_HD uint32_t operator()(size_t j) const { return pixel[j]; }
};
template <class Row>
struct PvNodeCopy {
  Row row;
  size_t first;
  const uint32_t* slot;
  const uint32_t* codes;
  const uint32_t* table;
  uint32_t code_words, table_words;
  uint32_t* carried_codes;
  uint32_t* carried_table;
  BALLA_HD void operator()(size_t j) const {
    const size_t r = row(j);
    pv_copy_entry(carried_codes + r * code_words, carried_table + r * table_words, codes, table,
                  slot != nullptr && j >= first ? slot[j - first] : render::kPvNone, code_words, table_words);
  }
};
// P2-D D3a: no carried codes (the start frame's, an unplanned window's; localize.h reads an edge-class node without
// them as R0c), on row i or on pixel id pixel[i].
struct PvClearCarried {
  uint32_t* table;
  uint32_t table_words;
  const uint32_t* pixel = nullptr;
  BALLA_HD void operator()(size_t i) const {
    table[(pixel != nullptr ? static_cast<size_t>(pixel[i]) : i) * table_words] = render::kPvNone;
  }
};

// Window requests through the renderer's window pose table and render_into (out_index null: row i, the switches'
// t_w nodes of Camera::switch_nodes).
class RenderInto final : public RenderHook {
 public:
  RenderInto(render::Renderer& r, Stream s) : renderer_(r), stream_(s) {}
  void render(SampleRequest* requests, size_t n, const uint32_t* out_index, float* irr, uint64_t* sig,
              uint32_t* klass, uint32_t subsamples) override {
    if (calls_++ > 0) take();  // the previous call's stage times (its round synced since)
    renderer_.assign_pose_rows(requests, n, stream_);
    renderer_.render_into(requests, n, out_index, irr, sig, stream_, klass, subsamples);
  }
  // P2-D D3a: the same with P&V's outputs, and the probes (timed apart: RenderStats::probe_ms).
  void render_pv(SampleRequest* requests, size_t n, const uint32_t* out_index, float* irr, uint64_t* sig,
                 uint32_t* klass, uint32_t subsamples, const render::PvOut& pv) override {
    if (calls_++ > 0) take();
    renderer_.assign_pose_rows(requests, n, stream_);
    renderer_.render_into(requests, n, out_index, irr, sig, stream_, klass, subsamples, &pv);
  }
  void probe(SampleRequest* requests, size_t n, render::ProbeOut* out) override {
    renderer_.assign_pose_rows(requests, n, stream_);
    renderer_.probe_into(requests, n, out, stream_);
  }
  // The window's stage times, and the GPU time (trace + shade) of each call (WindowResult::render_ms_by_call).
  render::RenderStats finish(std::vector<double>& ms_by_call) {
    take();
    ms_by_call = std::move(by_call_);
    return total_;
  }

 private:
  void take() {
    const render::RenderStats r = renderer_.take_render_stats();
    total_.trace_ms += r.trace_ms;
    total_.shade_ms += r.shade_ms;
    total_.host_ms += r.host_ms;
    total_.probe_ms += r.probe_ms;  // P2-D D3a's probes
    total_.census_on = r.census_on;
    for (size_t i = 0; i < r.census.size(); ++i) total_.census[i] += r.census[i];
    total_.pv_on = r.pv_on;  // P2-D D2 (execution.pv_outputs)
    total_.pv_entries += r.pv_entries;
    total_.pv_bytes += r.pv_bytes;
    if (calls_ > 0) by_call_.push_back(r.trace_ms + r.shade_ms);  // finish() of a window without a call: none
  }
  render::Renderer& renderer_;
  Stream stream_;
  size_t calls_ = 0;
  render::RenderStats total_{};
  std::vector<double> by_call_;
};

template <Space S>
class Camera;

// What the cameras of an engine share: the renderer, the scene's motion, the stream, the WindowEngine and
// the scratch of window(), step() and gt().
template <Space S>
struct Shared {
  template <class T>
  using Buf = Buffer<T, S>;

  Shared(render::SceneBundle bundle, EngineConfig c) : config(std::move(c)), renderer(std::move(bundle), config.render) {}
  ~Shared() { Par<S>::destroy_stream(stream); }
  Shared(const Shared&) = delete;
  Shared& operator=(const Shared&) = delete;

  EngineConfig config;
  render::Renderer renderer;
  std::unique_ptr<SceneMotion> motion;
  Stream stream = nullptr;
  std::mutex mutex;          // start, window, step, flush and gt of all cameras, one at a time
  WindowEngine<S> core;      // shared: the cameras run one after the other
  // The camera whose last window's runs core and act_pixel hold (runs_digest, last_runs); null: none (a window began
  // or failed since, or the output stage released them).
  const void* runs_of = nullptr;
  size_t budget = 0;
  std::vector<const Camera<S>*> cameras;
  // Scratch: O(pixels) (the plan pass's 36 B per pixel, K4's 5 B) and O(events) (about 132 B per held or new event and
  // 41 B per new one at the peak of a window; released by flush()). Counted in the budget (outside_bytes).
  Buf<uint32_t> in_pixel, order, order_alt, act_pixel, select;
  Buf<uint8_t> in_level, act_level, flags;
  Buf<uint64_t> in_rank, rank_alt, keys, keys_alt;
  Temp<S> temp;
  std::vector<uint64_t> seen;  // the duplicate check's bitmap
  Buf<int32_t> run_counts;
  Buf<SampleRequest> frame_requests;
  Buf<render::GtQuery> gt_queries;
  Buf<render::GtSample> gt_out;
  Buf<uint32_t> plan_states;  // I8's band: the plan pass's shadow states (8 B per pixel and shadow light)
  Buf<float> plan_normals;    // P2-D D1: the plan pass's geometric normals (12 B per pixel; obliquity or plan bits)
  // P2-D D3a: the sub-sample warp table (SampleConfig::pv's positions) and the re-rendered t_w nodes' P&V outputs.
  Buf<float> pv_warp_cdf, pv_warp_slope;
  Buf<uint16_t> pv_warp_guide;
  Buf<uint32_t> pv_node_slot, pv_node_codes, pv_node_table, pv_node_count;
  Buf<unsigned long long> counter, plan_counts;
  HostWords<S> plan_words;  // the plan's kappa sums (PlanCounts), read after the window's sampling syncs
  HostWords<S> pv_words;    // P2-D D1: the plan bits' counts (PvPlanPass), read after K4's sync
  // P2-B B2's per-pixel N: the active pixels' log2 N, the switches' requests and re-rendered t_w nodes (their lists in
  // select, two halves of 2 n), the switch words and the switch selects' counts.
  Buf<uint8_t> act_n, edge;  // edge: P2-B B4a's edge pixels (WindowEngine::Input::edge)
  Buf<uint8_t> edge_image;   // P2-B B6a: the edge pixels by pixel id (EdgeImage), pooled escalation's input
  Buf<SampleRequest> switch_requests;
  Buf<float> switch_irr;
  Buf<uint64_t> switch_sig;
  Buf<uint32_t> switch_class;
  Buf<unsigned long long> switch_counts;
  HostWords<S> switch_words;
  // P2-B B4b's escalation: the escalated pixels and levels, their pass's offsets, and the sensor's row starts (the
  // window's offsets, the escalated pixels' redirected; `redirected`: the last window's rows are row_start's).
  Buf<uint32_t> esc_pixel;
  Buf<uint8_t> esc_level;
  Buf<int64_t> esc_offsets, row_start;
  bool redirected = false;
  Buf<OutEvent> out_alt;
  // MOVING: the bodies' hold intervals.
  Buf<uint32_t> body_id, hold_off;
  Buf<uint8_t> body_traj;
  Buf<int64_t> holds;
  MovingTable moving{};

  // The bytes the engine holds beside the WindowEngine: the scratch above, every camera's buffers (the planner's, the
  // carried samples and the held events) and the device memory the renderer keeps for the held set (I3b). The budget
  // counts them (ENGINE_CPP I3 open item (d), I3b (b)), so that a window they crowd out raises MemoryBudgetError.
  size_t outside_bytes() const;

  // I3b: the bodies held over [t_w, t_w + W] join the renderer's world structure (rebuilt when the set changes); not in
  // a scene with baked proxies, whose movers stay apart from the scan (Renderer::holds_movers, scans.md §4.2).
  HeldSet hold(int64_t t_w) {
    HeldSet h;
    // P2-F F9b: under accuracy.mover_ao traced every mover stays a mover (the ambient rays look at movers only).
    // S3a-1: a baked-proxy scene keeps its movers in the scan world, whose proxy shadows would vanish during holds.
    if (!config.held_world || !renderer.holds_movers() || config.render.scene.mover_ao == render::MoverAo::kTraced) return h;
    const Clock::time_point t0 = Clock::now();
    const std::vector<uint32_t> ids = motion->held_bodies(t_w, t_w + time::kWindowNs);
    h.bodies = static_cast<uint32_t>(ids.size());
    h.rebuilt = renderer.hold(ids, t_w);
    h.build_ms = h.rebuilt ? ms_since(t0) : 0.0;
    return h;
  }
};

template <Space S>
class Camera final : public CameraEngine {
 public:
  using P = Par<S>;
  template <class T>
  using Buf = Buffer<T, S>;

  Camera(Shared<S>& s, uint16_t c, const SensorSpec& spec)
      : shared(s),
        cam(c),
        width(s.renderer.bundle().cameras[c].width),
        height(s.renderer.bundle().cameras[c].height),
        groups(s.renderer.num_groups()),
        sensor_(spec.params, static_cast<uint32_t>(width), static_cast<uint32_t>(height), spec.seed,
                s.config.light_groups, c, spec.num_threads,
                S == Space::kDevice ? sensor::Precision::kFp32 : spec.precision,
                S == Space::kDevice ? sensor::Device::kCuda : sensor::Device::kCpu),
        geo(s.renderer.bundle().cameras[c], render::support_px(s.renderer.options().scene)),
        limiter_(sensor::limiter_config(spec.params, spec.seed, c, static_cast<uint32_t>(width))) {
    init_photometry(spec);
  }

  StartResult start(int64_t t_s, int64_t warmup_ns, bool motion) override;
  WindowResult window(int64_t t_w, const uint32_t* pixel, const uint8_t* level, const int64_t* rank,
                      size_t n) override;
  StepResult step(int64_t t_w) override;
  std::vector<OutEvent> flush() override;
  LimiterCounters limiter_counters() const override;
  std::vector<render::GtSample> gt(const float* x, const float* y, const int64_t* t_ns, size_t n) override;
  const sensor::SensorModel& sensor() const override { return sensor_; }
  std::pair<uint64_t, uint64_t> runs_digest() const override;
  void force_n(const std::vector<uint8_t>& n_log2) override;
  WindowRuns last_runs() const override;
  std::vector<uint8_t> last_band() const override;
  void set_cancel(const CancelToken* token) override { sample.cancel = token; }  // read by every window's copy
  size_t bytes() const {
    return carried_irr.bytes() + carried_sig.bytes() + carried_class.bytes() + carried_n.bytes() + forced_n.bytes() +
           tile_band.bytes() + pixel_band.bytes() + plan.bytes() + pixel_tile.bytes() + pixel_by_rank.bytes() +
           tile_level.bytes() + tile_vhat.bytes() + held.bytes() + mid_dark.bytes() + pv_codes.bytes() + pv_table.bytes();
  }

  void init_photometry(const SensorSpec& spec);
  void dark_terms();
  void solve_reference(int64_t t, const std::vector<float>& frame);
  void render_frame(int64_t t, StartResult& out);
  size_t plan_window(int64_t t_w, StepResult& out);
  WindowResult sample_and_sense(int64_t t_w, size_t n, sensor::DeviceEvents* events);
  uint32_t switch_nodes(size_t n, RenderInto& hook, SwitchStats& out);
  const int64_t* escalate(int64_t t_w, size_t n, const SampleConfig& cfg, RenderInto& hook, WindowResult& out);
  void pixel_band_map(const TilePlan& tp, PlanStats& stats);
  WindowRuns runs_copy() const;
  std::vector<OutEvent> output(const sensor::DeviceEvents& ev, int64_t t_w, uint64_t* kept);
  int64_t output_cut(int64_t t_w) const;
  LimiterCounters limit(std::vector<OutEvent>& events, int64_t complete_ns);
  void check_time(int64_t t_w, const char* caller) const;

  Shared<S>& shared;
  uint16_t cam;
  int32_t width, height, groups;
  sensor::SensorModel sensor_;
  SampleConfig sample;
  int32_t rank_bits = 1;  // ranks < 2^rank_bits
  Buf<float> carried_irr;
  Buf<uint64_t> carried_sig;
  Buf<uint32_t> carried_class;  // the class words of the carried samples (the P2-0b counters)
  // P2-B B2 (sample.band_subsamples > 0): the log2 N of every pixel's carried sample, and force_n's map (empty: none).
  Buf<uint8_t> carried_n, forced_n;
  bool forced = false;
  // P2-B B4a: the plan's band tiles (TilePlan::band), valid for the window plan_window planned (band_planned); B6c:
  // under band_growth pixel its band pixels instead (pixel_band_map; band_pixel).
  Buf<uint8_t> tile_band, pixel_band;
  bool band_planned = false, band_pixel = false;
  // The planner and its data.
  CameraGeometry geo;
  std::unique_ptr<Planner> planner;
  PlanBuffers<S> plan;
  Buf<uint32_t> pixel_tile, pixel_by_rank;
  Buf<int8_t> tile_level;
  Buf<float> tile_vhat;  // the plan's v_hat per tile (PlanCounts)
  Crop crop{};
  int y_bits = 1;  // y < 2^y_bits
  // Events not yet final (output order; grown by a quarter beyond the need, released by flush()).
  Buf<OutEvent> held;
  size_t n_held = 0;
  // runs_digest(): the start frame's hash (host engine) and the active pixels of the last sampled window.
  uint64_t frame_digest = 0;
  size_t last_active = 0;
  // Photometry (SensorSpec; D1, SENSOR.md "Photometry"): the mode, the reference body's id (0: every hit) and
  // photocurrent, the gain map (natural vignetting; empty: none) and eta_L's per-pixel dark term (dark_terms()).
  Photometry photometry = Photometry::kRelative;
  std::optional<double> reference_fA;
  uint32_t reference_id = 0;
  std::vector<float> gain;
  Buf<float> mid_dark;
  // P2-H H4b: the rate limiter on the final events (sensor.limiter; off: it passes them through untouched), and its
  // wall time since start() (the flush's included).
  sensor::Limiter<OutEvent> limiter_;
  double limiter_ms_ = 0;
  // P2-D D3a (sample.pv): the carried samples' P&V codes and tables by pixel id (WindowEngine::Input::pv_codes), and
  // whether the planned window's plan bits and tile v_hat are current (plan_window; window() has none, so P&V is off
  // in its windows: sample_and_sense).
  Buf<uint32_t> pv_codes, pv_table;
  bool pv_plan_ok = false, pv_vhat_ok = false;
  // The P&V outputs of a re-render of `count` t_w nodes at sample.pv.n (Shared scratch; capacity count).
  render::PvOut pv_node_out(size_t count) {
    Shared<S>& s = shared;
    const auto cw = render::pv_code_words(sample.pv.n, sample.pv.lights);
    const auto tw = render::pv_table_words(groups, sample.pv.lights);
    uint32_t* count_w = s.pv_node_count.ensure(1);
    P::zero(count_w, sizeof(uint32_t), s.stream);
    return {s.pv_node_slot.ensure(count), s.pv_node_codes.ensure(count * cw), s.pv_node_table.ensure(count * tw), count_w,
            static_cast<uint32_t>(count), 0};
  }
  void pv_input(typename WindowEngine<S>::Input& in) {
    if (sample.pv.n == 0) return;
    in.pv_codes = pv_codes.data();
    in.pv_table = pv_table.data();
    in.pv_plan = pv_plan_ok ? plan.pv_plan.data() : nullptr;
    in.pixel_tile = pv_vhat_ok ? pixel_tile.data() : nullptr;
    in.tile_vhat = pv_vhat_ok ? tile_vhat.data() : nullptr;
  }
};

template <Space S>
size_t Shared<S>::outside_bytes() const {
  size_t n = renderer.held_device_bytes() + temp.bytes() + counter.bytes() + plan_counts.bytes() + run_counts.bytes() +
             frame_requests.bytes() + gt_queries.bytes() + gt_out.bytes() + plan_states.bytes() + plan_normals.bytes() +
             out_alt.bytes() +
             act_n.bytes() + edge.bytes() + edge_image.bytes() + switch_requests.bytes() + switch_irr.bytes() +
             switch_sig.bytes() + switch_class.bytes() + switch_counts.bytes() + esc_pixel.bytes() + esc_level.bytes() +
             esc_offsets.bytes() + row_start.bytes() + pv_warp_cdf.bytes() + pv_warp_slope.bytes() +
             pv_warp_guide.bytes() + pv_node_slot.bytes() + pv_node_codes.bytes() + pv_node_table.bytes() +
             pv_node_count.bytes();
  for (const auto* b : {&in_pixel, &order, &order_alt, &act_pixel, &select, &body_id, &hold_off}) n += b->bytes();
  for (const auto* b : {&in_level, &act_level, &flags, &body_traj}) n += b->bytes();
  for (const auto* b : {&in_rank, &rank_alt, &keys, &keys_alt}) n += b->bytes();
  n += holds.bytes();
  for (const Camera<S>* c : cameras) n += c->bytes();
  return n;
}

// FNV-1a over 8-byte words (and the tail bytes): the stage audit's hashes (a test hook, not a checksum format).
inline uint64_t hash_bytes(uint64_t h, const void* data, size_t n) {
  const auto* b = static_cast<const unsigned char*>(data);
  for (; n >= 8; n -= 8, b += 8) {
    uint64_t w;
    std::memcpy(&w, b, 8);
    h = (h ^ w) * 0x100000001b3ull;
  }
  for (; n > 0; --n, ++b) h = (h ^ *b) * 0x100000001b3ull;
  return h;
}

template <Space S>
class EngineOn final : public Engine {
 public:
  EngineOn(render::SceneBundle bundle, EngineConfig config, const std::vector<std::optional<SensorSpec>>& sensors);
  CameraEngine& camera(size_t i) override {
    if (i >= cameras_.size() || !cameras_[i]) {
      throw std::invalid_argument("engine: camera " + std::to_string(i) + " has no sensor in this engine");
    }
    return *cameras_[i];
  }
  size_t budget_bytes() const override { return shared_->budget; }
  bool on_device() const override { return S == Space::kDevice; }

 private:
  std::unique_ptr<Shared<S>> shared_;
  std::vector<std::unique_ptr<Camera<S>>> cameras_;
};

template <Space S>
EngineOn<S>::EngineOn(render::SceneBundle bundle, EngineConfig config,
                      const std::vector<std::optional<SensorSpec>>& sensors)
    : shared_(std::make_unique<Shared<S>>(std::move(bundle), std::move(config))) {
  Shared<S>& s = *shared_;
  const EngineConfig& cfg = s.config;
  const render::BackendKind backend = S == Space::kDevice ? render::BackendKind::kOptix : render::BackendKind::kEmbree;
  if (cfg.render.backend != backend) {
    throw std::invalid_argument(S == Space::kDevice ? "engine: the device engine needs the OptiX renderer"
                                                    : "engine: the host engine needs the Embree renderer");
  }
  const int64_t min_dt = cfg.min_dt_ns;
  if (min_dt < 1 || min_dt > time::kWindowNs || (min_dt & (min_dt - 1)) != 0) {
    throw std::invalid_argument("engine: min_dt_ns must be a power of two in [1, 2^23]");
  }
  if (!(cfg.d_px > 0)) throw std::invalid_argument("engine: d_px must be > 0");
  if (cfg.n_band != 0) {
    const auto pow2 = [](uint32_t v) { return v != 0 && (v & (v - 1)) == 0; };
    const uint32_t base = cfg.render.scene.n_subsamples;
    if (!pow2(cfg.n_band) || !pow2(base) || cfg.n_band <= base ||
        cfg.n_band > static_cast<uint32_t>(render::kMaxSubsamples)) {
      throw std::invalid_argument("engine: n_band (the band's N) must be a power of two above n_subsamples, itself a "
                                  "power of two, and at most " + std::to_string(render::kMaxSubsamples));
    }
  }
  const render::SceneBundle& b = s.renderer.bundle();
  if (cfg.light_groups.size() != b.light_groups.size()) {
    throw std::invalid_argument("engine: one waveform per light group of the bundle is needed");
  }
  if (sensors.size() > b.cameras.size()) throw std::invalid_argument("engine: more sensors than cameras");
  if (cfg.holds.size() > b.trajectories.size()) throw std::invalid_argument("engine: more holds than trajectories");
  s.renderer.build();
  // The planner's mesh distances are built now (not in the loop), and not on the fixed-rate grid.
  s.motion = std::make_unique<SceneMotion>(b, cfg.holds, cfg.fixed_period_ns <= 0);
  s.stream = Par<S>::create_stream();
  const MovingHolds& mh = s.motion->moving_holds();  // MOVING
  s.moving = {upload_vector(s.body_id, mh.id, s.stream), upload_vector(s.body_traj, mh.has_trajectory, s.stream),
              upload_vector(s.hold_off, mh.hold_off, s.stream), upload_vector(s.holds, mh.holds, s.stream),
              static_cast<uint32_t>(mh.id.size())};
  std::vector<std::pair<double, double>> ranges;
  for (const sensor::Waveform& w : cfg.light_groups) ranges.push_back(waveform_range(w));
  int32_t shift = 0;
  while ((int64_t{1} << shift) < min_dt) ++shift;
  PlanSchedule schedule = plan_schedule(cfg.d_px, cfg.refine_tol_log, min_dt);
  schedule.band = cfg.shadow_band;
  schedule.level_policy = cfg.level_policy;  // P2-H H9 (F004): per-tile, global or global_certified levels
  schedule.seeds = cfg.n_band != 0;  // P2-B B4a: the band N's tile map
  // P2-B B6c: the per-pixel band and its reach scales (ignored without the band N).
  for (const double sigma : cfg.band_reach) {
    if (!(sigma >= 0) || !std::isfinite(sigma)) {
      throw std::invalid_argument("engine: band_reach must be finite and >= 0");
    }
  }
  schedule.pixel = cfg.n_band != 0 && cfg.band_growth == BandGrowth::kPixels;
  schedule.sigma_cam = cfg.band_reach[0];
  schedule.sigma_mov = cfg.band_reach[1];
  // P2-D D1 (F021): the shadow-edge rates with the receivers' obliquity factors; the sub-sample pitch of R4's width
  // test at the run's largest N (P&V's plan bits: PlanSchedule::pv, set below with P&V's shadow mode).
  schedule.obliquity = cfg.shadow_obliquity == ShadowObliquity::kProjected;
  schedule.bits = true;  // F021's flag on the as-built rate (plan_oblique_pixels), rates unchanged
  schedule.pitch = 1.0 / std::sqrt(static_cast<double>(cfg.n_band != 0 ? cfg.n_band : cfg.render.scene.n_subsamples));
  // P2-F F9b: the moving bodies' AO receivers (plan.h AoMover) under accuracy.mover_ao proxy or traced.
  schedule.ao = cfg.render.scene.mover_ao != render::MoverAo::kOff;
  // P2-D D3a (accuracy.localization shadow): P&V's shadow mode on the pixels at the run's largest N if it is >= 256
  // (N_band, or a uniform N), with at most kPvMaxLights shadow-casting lights, not under accuracy.mover_ao traced (else
  // off: the counters stay empty; traced: P2-F F9b's code has no AO bit, so render_into refuses P&V outputs and
  // probe_into refuses probes; likewise in a scene with a specular lobe (P2-F F6): the table holds Lambert shares; and
  // where the renderer has no P&V build, Renderer::pv_supported: the device with two-channel NIR, P2-F F11a; and not with
  // baked proxies, !Renderer::holds_movers: the table and the probes carry no U2 factor or static word, S3a-2). The
  // sub-sample positions are the renderer's (its warp table, rebuilt here from the same settings, and pattern seeds).
  PvParams pv{};
  const uint32_t n_top = cfg.n_band != 0 ? cfg.n_band : cfg.render.scene.n_subsamples;
  if (cfg.localization == Localization::kShadow && cfg.fixed_period_ns <= 0 && n_top >= 256 &&
      cfg.render.scene.mover_ao != render::MoverAo::kTraced && !s.renderer.has_specular() &&
      s.renderer.pv_supported() && s.renderer.holds_movers() && s.renderer.num_shadow_lights() <= render::kPvMaxLights) {
    const render::SceneSettings& ss = cfg.render.scene;
    const rng::WarpTable<float> warp(rng::PixelResponse(ss.aperture_px, ss.blur_sigma_px, ss.truncation_sigmas),
                                     ss.warp_cells);
    pv.n = n_top;
    pv.lights = s.renderer.num_shadow_lights();
    int32_t bit = 0;
    for (const render::Light& l : b.lights) {
      if (l.kind == render::BundleLightKind::kAmbient || l.kind == render::BundleLightKind::kBackground) continue;
      if (l.casts_shadows && bit < render::kPvMaxLights) pv.light_group[bit++] = static_cast<int8_t>(l.group);
    }
    pv.seed = ss.seed;
    pv.warp = {upload_vector(s.pv_warp_cdf, warp.cdf(), s.stream), upload_vector(s.pv_warp_slope, warp.slope(), s.stream),
               upload_vector(s.pv_warp_guide, warp.guide(), s.stream), warp.cells(), warp.view().dx};
    pv.obliquity = cfg.shadow_obliquity == ShadowObliquity::kProjected;
  }
  // P2-D D3b: R4's thin-moving-shadow width in the plan bits (PlanSchedule::pv: the movers' thickness at start-up, the
  // plan pass's widths); it moves no rate, level or band decision (the band's region (d) already reads the thickness).
  schedule.pv = pv.n != 0;
  cameras_.resize(b.cameras.size());
  for (size_t i = 0; i < sensors.size(); ++i) {
    if (!sensors[i]) continue;
    const auto cam = static_cast<uint16_t>(i);
    auto c = std::make_unique<Camera<S>>(s, cam, *sensors[i]);
    SampleConfig& sc = c->sample;
    sc.groups = c->groups;
    sc.cam = cam;
    sc.subsamples = cfg.render.scene.n_subsamples;
    sc.band_subsamples = cfg.n_band;
    const uint64_t n_pixels = static_cast<uint64_t>(c->width) * static_cast<uint64_t>(c->height);
    while ((uint64_t{1} << c->rank_bits) < n_pixels) ++c->rank_bits;
    while ((1 << c->y_bits) < c->height) ++c->y_bits;
    SampleRules& r = sc.rules;
    r.fixed = cfg.fixed_period_ns > 0;
    r.period_ns = cfg.fixed_period_ns;
    r.refine_signature = cfg.refine_signature;
    r.max_refine_levels = cfg.max_refine_levels;
    r.min_split = static_cast<uint32_t>(2 * min_dt);
    r.shift = shift;
    r.midpoint = cfg.refine_tol_log.has_value();
    // eta_L's (and eta_J's) dark term: the median dark current at the sensor's temperature, in renderer units.
    const sensor::SensorParams& p = c->sensor_.params();
    if (cfg.render.scene.mover_ao == render::MoverAo::kTraced) {
      // P2-F F9b, F108's step rule (render/shade/ao_traced.h): an AO flip moves a whole-pixel cluster's ambient by at
      // most 1/N of it, so ln(1 + 1/N) <= theta_d / 4 at ambient share 1 needs N >= n_s = ceil(1 / (exp(theta_d / 4) -
      // 1)), theta_d the camera's smaller mean threshold.
      const double theta = std::min(p.theta_on, p.theta_off);
      const double n_s = std::ceil(1.0 / std::expm1(0.25 * theta));
      if (!(static_cast<double>(cfg.render.scene.n_subsamples) >= n_s)) {
        throw std::invalid_argument("engine: accuracy.mover_ao traced needs accuracy.n_subsamples >= " +
                                    std::to_string(static_cast<int64_t>(n_s)) + " at camera " +
                                    b.cameras[i].name + "'s threshold " + std::to_string(theta) +
                                    " (F108's step rule: ln(1 + 1/N) <= theta / 4)");
      }
    }
    r.mid = midpoint_params(cfg.refine_tol_log.value_or(0.0), sensor::median_dark_fA(p) / p.illuminance_scale, ranges);
    c->dark_terms();  // the sensor's gain map, per pixel (reference photometry: again once the scale is solved)
    economy_rules(r, cfg.mid_levels, cfg.tau_jump_ns, cfg.delta_ref, cfg.half_ratio, cfg.eps_jump);
    r.counters = cfg.jump_counters;  // P2-C C2
    sc.pv = pv;  // P2-D D3a; the bracket floor's quantum from this camera's readout quantum (DESIGN §6.2)
    if (pv.n != 0) {
      const double q = p.readout_quantum_s > 0 ? p.readout_quantum_s * 1e9 : 0.0;
      sc.pv.t_q = static_cast<uint32_t>(std::max(512.0, std::min(0.5 * q, 1e4)));
    }
    c->planner = std::make_unique<Planner>(*s.motion, c->geo, cam, schedule, cfg.crop);
    c->crop = cfg.crop.value_or(Crop{0, 0, c->width, c->height});
    upload_vector(c->pixel_tile, c->geo.pixel_tile, s.stream);
    upload_vector(c->pixel_by_rank, c->geo.pixel_by_rank, s.stream);
    s.cameras.push_back(c.get());
    cameras_[i] = std::move(c);
  }
  const double bytes = cfg.device_budget_mb * 1048576.0;  // beyond size_t: no limit
  s.budget = cfg.device_budget_mb > 0
                 ? (bytes < static_cast<double>(SIZE_MAX) ? static_cast<size_t>(bytes) : SIZE_MAX)
                 : static_cast<size_t>(0.7 * static_cast<double>(Par<S>::free_bytes()));
}

// The camera's photometry (D1, D13 item 2; SENSOR.md "Photometry"): with optics.vignetting (unset: on under physical
// and reference, off under relative) the sensor's gain map, natural vignetting at each pixel centre's chief ray (the
// camera model's unit rays, optical axis z) by optics.vignetting_law: cos^4, or (P2-F F2) cos(theta) dOmega/dA of the
// camera model (core/camera/solid_angle.h), times (P2-F F2b) optics.relative_illumination's radial G; one path for
// every mode; the reference body by name.
template <Space S>
void Camera<S>::init_photometry(const SensorSpec& spec) {
  const render::SceneBundle& b = shared.renderer.bundle();
  const std::string where = "engine: camera " + b.cameras[cam].name + ": ";
  photometry = spec.photometry;
  reference_fA = spec.reference_current_fA;
  if (photometry == Photometry::kReference) {
    if (!reference_fA || !(*reference_fA > 0)) {
      throw std::invalid_argument(where + "reference photometry needs a photocurrent > 0");
    }
    if (!spec.reference_instance.empty()) {
      std::string names;
      for (const render::Body& body : b.bodies) {
        if (body.name == spec.reference_instance) reference_id = body.id;
        names += (names.empty() ? "" : ", ") + body.name;
      }
      if (reference_id == 0) {
        throw std::invalid_argument(where + "photometry.reference_instance '" + spec.reference_instance +
                                    "' names no body (" + names + ")");
      }
    }
  }
  if (!spec.vignetting.value_or(photometry != Photometry::kRelative)) return;
  gain.resize(geo.n_pixels);
  const auto off_axis = [&](const char* law) {
    return std::invalid_argument(where + "a pixel looks 90 degrees or more off axis, where " + law +
                                 " vignetting does not apply (optics.vignetting: false)");
  };
  if (spec.vignetting_law == VignettingLaw::kJacobian) {
    // The model without a domain limit (r_lim, theta_lim infinite): CameraGeometry unprojected every pixel centre's
    // ray inside the injective domain, where the projection's Jacobian has full rank.
    const render::Camera& bc = b.cameras[cam];
    const auto k = [&](size_t i) { return i < bc.dist.size() ? bc.dist[i] : 0.0; };
    const double open = infinity<double>();
    const auto fill = [&](const auto& model) {
      for (size_t p = 0; p < geo.n_pixels; ++p) {
        const Vec3<double> w{geo.rays[3 * p], geo.rays[3 * p + 1], geo.rays[3 * p + 2]};
        double g = 0.0;
        if (!camera::vignetting_gain(model, w, &g)) throw off_axis("cos(theta) dOmega/dA");
        gain[p] = static_cast<float>(g);
      }
    };
    if (bc.model == render::CameraModelKind::kRadtan) {
      fill(camera::RadtanCamera<double>{bc.fx, bc.fy, bc.cx, bc.cy, k(0), k(1), k(2), k(3), k(4), open, 0.0});
    } else {
      fill(camera::KannalaBrandtCamera<double>{bc.fx, bc.fy, bc.cx, bc.cy, k(0), k(1), k(2), k(3), open});
    }
  } else {
    for (size_t p = 0; p < geo.n_pixels; ++p) {
      const double c = geo.rays[3 * p + 2];  // cos theta of the unit ray
      if (!(c > 0)) throw off_axis("cos^4");
      gain[p] = static_cast<float>((c * c) * (c * c));
    }
  }
  // P2-F F2b: the lens's relative illumination G(r~) on the law's gain (core/camera/relative_illumination.h); [] leaves
  // the map as the law made it.
  const std::vector<double>& ri = spec.relative_illumination;
  if (const char* why = camera::relative_illumination_error(ri)) {
    throw std::invalid_argument(where + "optics.relative_illumination: " + why);
  }
  if (!ri.empty()) {
    const render::Camera& bc = b.cameras[cam];
    const double r2c = camera::corner_radius_sq(bc.width, bc.height, bc.cx, bc.cy);
    const auto w = static_cast<size_t>(bc.width);
    for (size_t p = 0; p < geo.n_pixels; ++p) {
      const double dx = static_cast<double>(p % w) - bc.cx, dy = static_cast<double>(p / w) - bc.cy;
      const double u = r2c > 0.0 ? (dx * dx + dy * dy) / r2c : 0.0;  // r~^2 (a one-pixel image: the axis)
      gain[p] = static_cast<float>(static_cast<double>(gain[p]) * camera::relative_illumination(ri[0], ri[1], u));
    }
  }
  sensor_.set_photometry(sensor_.params().illuminance_scale, gain);
}

// eta_L's (and eta_J's) dark term at the sensor's photometry, in renderer units: the median dark current at the
// sensor's temperature over its scale, and with a gain map over scale x gain per pixel (the sensor's per-pixel scale).
template <Space S>
void Camera<S>::dark_terms() {
  const sensor::SensorParams& p = sensor_.params();
  const double dark = sensor::median_dark_fA(p);
  MidpointParams& mid = sample.rules.mid;
  mid.dark = static_cast<float>(dark / p.illuminance_scale);
  if (gain.empty()) return;
  std::vector<float> d(gain.size());
  for (size_t i = 0; i < d.size(); ++i) {
    d[i] = static_cast<float>(dark / (p.illuminance_scale * static_cast<double>(gain[i])));
  }
  mid.dark_px = upload_vector(mid_dark, d, shared.stream);
}

// Reference photometry (D1 c): the scale that puts the median signal photocurrent (the frame's sum over the light
// groups, at their mean, times the gain) of the reference pixels at reference_fA; reference pixels: the centre rays at
// t that hit the reference body (every body when none is named), from a GT pass at the frame's time. Then the sensor's
// scale and eta_L's dark terms follow.
template <Space S>
void Camera<S>::solve_reference(int64_t t, const std::vector<float>& frame) {
  Shared<S>& s = shared;
  const size_t n = geo.n_pixels;
  render::GtQuery* q = s.gt_queries.ensure(n);
  render::GtSample* gt = s.gt_out.ensure(n);
  P::for_each(n, PlanQueries{t, static_cast<uint32_t>(width), q}, s.stream);
  s.renderer.gt_into(cam, q, n, gt, s.stream);
  std::vector<render::GtSample> hits(n);
  P::download(hits.data(), gt, n * sizeof(render::GtSample), s.stream);
  std::vector<double> v;
  for (size_t p = 0; p < n; ++p) {
    if (hits[p].instance == 0 || (reference_id != 0 && hits[p].instance != reference_id)) continue;
    double e = 0;
    for (int32_t g = 0; g < groups; ++g) e += static_cast<double>(frame[p * static_cast<size_t>(groups) + g]);
    v.push_back(e * (gain.empty() ? 1.0 : static_cast<double>(gain[p])));
  }
  const std::string where = "engine: camera " + s.renderer.bundle().cameras[cam].name + ": reference photometry: ";
  if (v.empty()) throw std::invalid_argument(where + "no pixel of the start frame sees the reference");
  const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);  // the upper median
  std::nth_element(v.begin(), mid, v.end());
  if (!(*mid > 0)) throw std::invalid_argument(where + "the reference pixels' median radiance is 0 (unlit)");
  sensor_.set_photometry(*reference_fA / *mid, gain);
  dark_terms();
}

// Every pixel's carried sample at t (the sensor's frame): one render request per pixel, the window clock's slot 0 at t.
// With the band (P2-B B2) every pixel at N_band, its carried N: the frame at t_s after a motion warm-up is then a
// static start's, whatever N its windows switched to.
template <Space S>
void Camera<S>::render_frame(int64_t t, StartResult& out) {
  Shared<S>& s = shared;
  const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
  const uint32_t band = sample.band_subsamples, n_sub = band > 0 ? band : sample.subsamples;
  float* irr = carried_irr.ensure(n * static_cast<size_t>(groups));
  uint64_t* sig = carried_sig.ensure(n);
  uint32_t* cls = carried_class.ensure(n);
  SampleRequest* req = s.frame_requests.ensure(n);
  s.renderer.take_render_stats();  // drops the stage times a failed window left (no wait when none are pending)
  s.renderer.begin_window(cam, {t, time::kWindowLog2, 0, 0}, s.stream);  // slot 0 at t
  P::for_each(n, FrameRequests{cam, n_sub, req}, s.stream);
  s.renderer.assign_pose_rows(req, n, s.stream);
  s.renderer.render_into(req, n, nullptr, irr, sig, s.stream, cls, n_sub);
  if (band > 0) P::for_each(n, FillBytes{carried_n.ensure(n), static_cast<uint8_t>(std::countr_zero(band))}, s.stream);
  if (sample.pv.n != 0) {  // P2-D D3a: the frame's nodes carry no P&V codes (R0c on an edge-class one)
    const auto tw = render::pv_table_words(groups, sample.pv.lights);
    pv_codes.ensure(n * render::pv_code_words(sample.pv.n, sample.pv.lights));
    P::for_each(n, PvClearCarried{pv_table.ensure(n * tw), tw, nullptr}, s.stream);
  }
  const render::RenderStats r = s.renderer.take_render_stats();
  out.render.trace_ms += r.trace_ms;
  out.render.shade_ms += r.shade_ms;
  out.render.host_ms += r.host_ms;
}

template <Space S>
StartResult Camera<S>::start(int64_t t_s, int64_t warmup_ns, bool motion) {
  Shared<S>& s = shared;
  const std::lock_guard<std::mutex> lock(s.mutex);
  check_cancel(sample.cancel, "start");  // BL-0b
  const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
  // The motion warm-up's windows [t_m, t_s): whole windows, rounded up, inside the bundle's span (D12: "when the bundle
  // covers it"); before t_m (and throughout a static warm-up) the frame at t_m is held. A motion warm-up the bundle
  // covers only in part is completed by that static one, to DESIGN §8.4's 50 ms in all at least (validate_run warns).
  int64_t t_m = t_s, span = warmup_ns;
  if (motion) {
    const int64_t k = (warmup_ns + time::kWindowNs - 1) / time::kWindowNs;
    const int64_t covered = -time::floor_div(-s.renderer.bundle().t_start_ns, time::kWindowNs) * time::kWindowNs;
    t_m = std::min(t_s, std::max(t_s - k * time::kWindowNs, covered));
    if (t_m > t_s - k * time::kWindowNs) span = std::max(warmup_ns, kMinStaticWarmupNs);
  }
  const int64_t t_init = std::min(t_s - span, t_m);
  StartResult out;
  HeldSet held_set = s.hold(t_m);
  out.held = held_set;
  // A scene with baked proxies: how much of each proxy the scan hides from the baked receivers, and the split bodies
  // (empty and 0 for every other scene); before the start frame, outside the warm-up's timing.
  if (!s.renderer.holds_movers()) {
    out.proxy_blocked = s.renderer.proxy_visibility(cam, t_s);
    out.split_geometries = s.renderer.split_geometries();
  }
  render_frame(t_m, out);
  std::vector<float> frame(n * static_cast<size_t>(groups));
  P::download(frame.data(), carried_irr.data(), frame.size() * sizeof(float), s.stream);
  if (photometry == Photometry::kReference) solve_reference(t_m, frame);
  out.illuminance_scale = sensor_.params().illuminance_scale;
  check_cancel(sample.cancel, "start: before sensing");  // BL-0b
  const Clock::time_point t0 = Clock::now();
  sensor_.init(t_init, frame.data());
  out.probe_ms = sensor_.stationary_ms();
  out.probe_history_ns = sensor_.stationary_history_ns();
  if (t_init < t_m) sensor_.advance_device(sensor::DeviceRuns{}, t_m, s.stream);
  for (int64_t t_w = t_m; t_w < t_s; t_w += time::kWindowNs) {  // the motion windows; their events are dropped
    held_set = s.hold(t_w);
    out.held.rebuilt = out.held.rebuilt || held_set.rebuilt;
    out.held.build_ms += held_set.build_ms;
    StepResult plan_out;
    const size_t n_active = plan_window(t_w, plan_out);
    sensor::DeviceEvents ev;
    const WindowResult w = sample_and_sense(t_w, n_active, &ev);
    out.render.trace_ms += w.render.trace_ms;
    out.render.shade_ms += w.render.shade_ms;
    out.render.host_ms += w.render.host_ms;
    ++out.motion_windows;
  }
  if (t_m < t_s) {  // the carried samples at t_s are the static warm-up's: the runs from t_s on are unchanged
    held_set = s.hold(t_s);
    out.held.bodies = held_set.bodies;
    out.held.rebuilt = out.held.rebuilt || held_set.rebuilt;
    out.held.build_ms += held_set.build_ms;
    render_frame(t_s, out);
  }
  out.warmup_ms = ms_since(t0);
  out.frame_mean = sensor_.glare_frame_mean();
  if constexpr (S == Space::kHost) {
    frame_digest = hash_bytes(hash_bytes(0xcbf29ce484222325ull, carried_irr.data(),
                                         n * static_cast<size_t>(groups) * sizeof(float)),
                              carried_sig.data(), n * sizeof(uint64_t));
    if (sample.band_subsamples > 0) frame_digest = hash_bytes(frame_digest, carried_n.data(), n);
  }
  sensor_.reset_stats();
  n_held = 0;
  limiter_ = sensor::Limiter<OutEvent>(limiter_.config());  // a run's first bin finds the bucket full
  limiter_ms_ = 0;
  return out;
}

template <Space S>
std::pair<uint64_t, uint64_t> Camera<S>::runs_digest() const {
  if constexpr (S != Space::kHost) {
    throw std::logic_error("runs_digest: a test hook of the host engine");
  } else {
    const std::lock_guard<std::mutex> lock(shared.mutex);  // the buffers step() reallocates
    uint64_t h = 0xcbf29ce484222325ull;
    if (const WindowRuns r = runs_copy(); !r.pixel.empty()) {  // without escalations the runs as the engine holds them
      h = hash_bytes(h, r.pixel.data(), r.pixel.size() * sizeof(uint32_t));
      h = hash_bytes(h, r.offsets.data(), r.offsets.size() * sizeof(int64_t));
      h = hash_bytes(h, r.t.data(), r.t.size() * sizeof(int64_t));
      h = hash_bytes(h, r.irr.data(), r.irr.size() * sizeof(float));
    }
    if (sample.band_subsamples > 0 && carried_n.data() != nullptr) h = hash_bytes(h, carried_n.data(), geo.n_pixels);
    return {frame_digest, h};
  }
}

template <Space S>
void Camera<S>::force_n(const std::vector<uint8_t>& n_log2) {
  Shared<S>& s = shared;
  const std::lock_guard<std::mutex> lock(s.mutex);
  const uint32_t band = sample.band_subsamples;
  if (band == 0) throw std::logic_error("force_n: the band is off (EngineConfig::n_band is 0)");
  if (n_log2.empty()) {
    forced = false;
    forced_n.reset();
    return;
  }
  if (n_log2.size() != geo.n_pixels) throw std::invalid_argument("force_n: one log2 N per pixel (width x height)");
  const auto lo = std::countr_zero(sample.subsamples), hi = std::countr_zero(band);
  for (const uint8_t v : n_log2) {
    if (v != lo && v != hi) {
      throw std::invalid_argument("force_n: every value is log2 n_subsamples (" + std::to_string(lo) +
                                  ") or log2 n_band (" + std::to_string(hi) + "), not " + std::to_string(v));
    }
  }
  upload_vector(forced_n, n_log2, s.stream);
  forced = true;
}

template <Space S>
WindowRuns Camera<S>::last_runs() const {
  const std::lock_guard<std::mutex> lock(shared.mutex);
  return runs_copy();
}

template <Space S>
std::vector<uint8_t> Camera<S>::last_band() const {
  const std::lock_guard<std::mutex> lock(shared.mutex);
  if (!band_planned) return {};
  std::vector<uint8_t> out(geo.n_pixels);
  if (band_pixel) {
    P::download(out.data(), pixel_band.data(), out.size(), shared.stream);
    return out;
  }
  std::vector<uint8_t> tiles(static_cast<size_t>(geo.tiles_x) * static_cast<size_t>(geo.tiles_y));
  P::download(tiles.data(), tile_band.data(), tiles.size(), shared.stream);
  for (size_t p = 0; p < out.size(); ++p) out[p] = tiles[geo.pixel_tile[p]];
  return out;
}

// The camera's last window's runs as the sensor read them (rows s.row_start or the window's offsets, s.run_counts),
// each pixel's contiguous in rank order: without an escalation exactly the engine's runs (they are contiguous from 0);
// with one, the main pass's runs of the escalated pixels left out. The caller holds the mutex.
template <Space S>
WindowRuns Camera<S>::runs_copy() const {
  const Shared<S>& s = shared;
  WindowRuns r;
  const size_t n = s.runs_of == this ? last_active : 0;
  if (n == 0) return r;
  const WindowEngine<S>& c = s.core;
  const size_t samples = c.samples(), g = static_cast<size_t>(groups);
  r.pixel.resize(n);
  r.offsets.resize(n + 1);
  P::download(r.pixel.data(), s.act_pixel.data(), n * sizeof(uint32_t), s.stream);
  if (sample.band_subsamples > 0) {
    r.n_log2.resize(n);
    P::download(r.n_log2.data(), s.act_n.data(), n, s.stream);
  }
  if (!s.redirected) {
    r.t.resize(samples);
    r.irr.resize(samples * g);
    P::download(r.offsets.data(), c.offsets(), (n + 1) * sizeof(int64_t), s.stream);
    P::download(r.t.data(), c.t(), samples * sizeof(int64_t), s.stream);
    P::download(r.irr.data(), c.irr(), r.irr.size() * sizeof(float), s.stream);
    return r;
  }
  std::vector<int64_t> start(n), t(samples);
  std::vector<int32_t> count(n);
  std::vector<float> irr(samples * g);
  P::download(start.data(), s.row_start.data(), n * sizeof(int64_t), s.stream);
  P::download(count.data(), s.run_counts.data(), n * sizeof(int32_t), s.stream);
  P::download(t.data(), c.t(), samples * sizeof(int64_t), s.stream);
  P::download(irr.data(), c.irr(), irr.size() * sizeof(float), s.stream);
  r.offsets[0] = 0;
  for (size_t a = 0; a < n; ++a) {
    const auto b = static_cast<size_t>(start[a]), m = static_cast<size_t>(count[a]);
    r.offsets[a + 1] = r.offsets[a] + count[a];
    r.t.insert(r.t.end(), t.begin() + static_cast<std::ptrdiff_t>(b), t.begin() + static_cast<std::ptrdiff_t>(b + m));
    r.irr.insert(r.irr.end(), irr.begin() + static_cast<std::ptrdiff_t>(b * g),
                 irr.begin() + static_cast<std::ptrdiff_t>((b + m) * g));
  }
  return r;
}

template <Space S>
void Camera<S>::check_time(int64_t t_w, const char* caller) const {
  if (carried_irr.data() == nullptr) throw std::logic_error(std::string(caller) + ": call start() first");
  if (t_w != sensor_.time_ns()) {
    throw std::invalid_argument(std::string(caller) + ": t_w = " + std::to_string(t_w) +
                                " ns, but the camera's sensor is at " + std::to_string(sensor_.time_ns()) +
                                " ns (windows follow start() in sequence)");
  }
}

// K5-K13 and S for the n active pixels in shared.act_pixel / act_level (rank order).
template <Space S>
WindowResult Camera<S>::sample_and_sense(int64_t t_w, size_t n, sensor::DeviceEvents* events) {
  Shared<S>& s = shared;
  const Clock::time_point t0 = Clock::now();
  SampleConfig cfg = sample;
  // P2-D D3a: P&V counts in planned windows only (D_bound and the brackets read the plan's tile v_hat; window() has
  // none). In an unplanned window it is off (no counters), and the active pixels' carried codes are cleared once the
  // window is sampled, since no PvCarry wrote their last nodes' (R0c in the next window).
  const bool pv_unplanned = sample.pv.n != 0 && !pv_vhat_ok;
  if (pv_unplanned) cfg.pv.n = 0;
  render::WindowClock clock{t_w, cfg.rules.shift, 0, 0};
  if (cfg.rules.fixed) {
    cfg.rules.m = fixed_rate_samples(t_w, cfg.rules.period_ns);
    clock = {t_w, 0, cfg.rules.m, cfg.rules.period_ns};
  }
  s.renderer.take_render_stats();  // drops the stage times a failed window left (no wait when none are pending)
  s.renderer.begin_window(cam, clock, s.stream);
  RenderInto hook(s.renderer, s.stream);
  uint32_t* act_pixel = s.act_pixel.ensure(n);
  WindowResult out;
  // P2-B B2: each active pixel's N, the switches' t_w nodes re-rendered at it first.
  const uint8_t* act_n = nullptr;
  uint32_t switch_calls = 0;
  out.band.on = sample.band_subsamples > 0;
  if (out.band.on && n > 0) {
    switch_calls = switch_nodes(n, hook, out.band);
    act_n = s.act_n.data();
  }
  uint8_t* edge = nullptr;  // P2-B B4a: the window's edge pixels
  if (act_n != nullptr) {
    edge = s.edge.ensure(n);
    P::zero(edge, n, s.stream);
    if (!cfg.rules.fixed) s.edge_image.ensure(geo.n_pixels);  // B6a's, before the budget counts it
  }
  typename WindowEngine<S>::Input in{t_w,       act_pixel,          s.act_level.ensure(n),
                                     n,         carried_irr.data(), carried_sig.data(),
                                     act_pixel, carried_class.data(), act_n,
                                     edge};
  pv_input(in);  // P2-D D3a
  int32_t* counts = s.run_counts.ensure(n);
  // The window's samples get the budget less what the engine holds beside them (Shared::outside_bytes).
  const size_t outside = s.outside_bytes();
  if (outside >= s.budget) {
    throw MemoryBudgetError("window at " + std::to_string(t_w) + " ns: the planner's and output stage's buffers and the "
                            "held world hold " + detail::mib(outside) + " MiB, the device budget is " +
                            detail::mib(s.budget) + " MiB (execution.device_budget_mb)");
  }
  out.stats = s.core.run(in, cfg, hook, s.budget - outside, s.stream);
  out.stats.render_calls += switch_calls;
  P::for_each(n, RunCounts{s.core.offsets(), counts}, s.stream);
  const int64_t* row_start = s.core.offsets();
  s.redirected = false;
  if (act_n != nullptr) {
    P::for_each(n,
                EdgeCounts{act_n, edge, static_cast<uint8_t>(std::countr_zero(sample.band_subsamples)),
                           s.switch_counts.data()},
                s.stream);
    if (!cfg.rules.fixed) row_start = escalate(t_w, n, cfg, hook, out);  // the fixed-rate grid has no edge pixels
  }
  if (pv_unplanned && n > 0) {
    const auto tw = render::pv_table_words(groups, sample.pv.lights);
    P::for_each(n, PvClearCarried{pv_table.data(), tw, act_pixel}, s.stream);
  }
  last_active = n;
  s.runs_of = this;
  if (act_n != nullptr) {
    std::array<unsigned long long, kSwitchWords> w{};
    P::download(w.data(), s.switch_counts.data(), sizeof(w), s.stream);
    out.band.band_pixels = w[kSwitchBandPixels];
    out.band.unquiescent = w[kSwitchUnquiescent];
    for (int k = 0; k < kDiagBins; ++k) out.band.eta[static_cast<size_t>(k)] = w[static_cast<size_t>(kSwitchHist + k)];
    out.band.band_edge = w[kSwitchBandEdge];
    out.band.base_edge = w[kSwitchBaseEdge];
    out.band.changed = w[kSwitchChanged];
    out.band.escalated_changed = w[kEscChanged];
    out.band.escalated_unquiescent = w[kEscUnquiescent];
    for (int k = 0; k < kDiagBins; ++k) {
      out.band.escalated_eta[static_cast<size_t>(k)] = w[static_cast<size_t>(kEscHist + k)];
    }
    out.band.escalated_pooled = w[kEscPooled];
    out.band.border = w[kSwitchBorder];
    out.band.plan = w[kSwitchPlan];
  }
  out.render = hook.finish(out.render_ms_by_call);
  for (const uint64_t k : out.stats.samples_by_round) out.pixel_samples += k;
  out.sample_ms = ms_since(t0);
  // S: the sensor on the runs in the engine's memory; its events stay there.
  check_cancel(sample.cancel, "window: before sensing");  // BL-0b
  const Clock::time_point t1 = Clock::now();
  const sensor::DeviceRuns runs{act_pixel, row_start, counts, n, s.core.t(), s.core.irr(), s.core.samples()};
  sensor_.reset_stats();
  *events = sensor_.advance_device(runs, t_w + time::kWindowNs, s.stream);
  out.frame_mean = sensor_.glare_frame_mean();  // veiling glare: the sensor's at t_w + W, from its runs
  out.sensor_stats = sensor_.stats();
  out.launch_stats = sensor_.gpu_launch_stats();
  out.sense_ms = ms_since(t1);
  return out;
}

// P2-B B2, before a window's samples: the log2 N of each of its n active pixels (s.act_n; ActiveN), and the switches,
// active pixels whose carried sample has the other N: their t_w nodes (slot 0 of the window clock) are re-rendered at
// the new N, a call per N, and become their carried samples, so that the window's signatures and refinement compare
// samples of one N. The switch lists take s.select (to N_base in its first n entries, to N_band in the next n); one
// sync reads their counts. The device words (band pixels, unquiescent switches, the steps' bins) are read after the
// window. Returns the render calls (0-2). The re-render uses this window's held set: where it was rebuilt, a held
// body's hits differ in the last bits from the previous set's, under which the uniform run rendered its carried node
// (SwitchStats; the exception to "a uniform run's node at that N").
template <Space S>
uint32_t Camera<S>::switch_nodes(size_t n, RenderInto& hook, SwitchStats& out) {
  Shared<S>& s = shared;
  const uint32_t base = sample.subsamples, band = sample.band_subsamples;
  const auto base_log2 = static_cast<uint8_t>(std::countr_zero(base));
  const auto band_log2 = static_cast<uint8_t>(std::countr_zero(band));
  const uint32_t* pixel = s.act_pixel.data();
  uint8_t* act_n = s.act_n.ensure(n);
  unsigned long long* counts = s.switch_counts.ensure(kSwitchWords);
  P::zero(counts, kSwitchWords * sizeof(unsigned long long), s.stream);
  P::for_each(n,
              ActiveN{pixel, forced ? forced_n.data() : nullptr,
                      band_planned && !band_pixel ? tile_band.data() : nullptr,
                      band_planned && band_pixel ? pixel_band.data() : nullptr, pixel_tile.data(),
                      static_cast<uint32_t>(width), crop, base_log2, band_log2, act_n, counts},
              s.stream);
  uint8_t* flags = s.flags.ensure(n);
  uint32_t* sel = s.select.ensure(2 * n);
  unsigned long long* words = s.switch_words.data();
  P::for_each(n, SwitchFlags{pixel, act_n, carried_n.data(), base_log2, flags}, s.stream);
  P::select_flagged(flags, sel, n, &words[0], s.temp, s.stream);
  P::for_each(n, SwitchFlags{pixel, act_n, carried_n.data(), band_log2, flags}, s.stream);
  P::select_flagged(flags, sel + n, n, &words[1], s.temp, s.stream);
  P::sync(s.stream);
  const auto k0 = static_cast<uint32_t>(words[0]);
  const size_t k = k0 + static_cast<size_t>(words[1]), g = static_cast<size_t>(groups);
  out.switches = k;
  out.up = words[1];
  if (k == 0) return 0;
  const SwitchList list{sel, n, k0};
  SampleRequest* req = s.switch_requests.ensure(k);
  float* irr = s.switch_irr.ensure(k * g);
  uint64_t* sig = s.switch_sig.ensure(k);
  uint32_t* cls = s.switch_class.ensure(k);
  P::for_each(k, SwitchRequests{list, pixel, cam, base, band, req}, s.stream);
  if (k0 > 0) hook.render(req, k0, nullptr, irr, sig, cls, base);
  // P2-D D3a: an up-switch's node at P&V's N carries its codes and table (the switches down carry none).
  const bool pv = sample.pv.n != 0 && band == sample.pv.n;
  const render::PvOut pv_out = pv && k > k0 ? pv_node_out(k - k0) : render::PvOut{};
  if (k > k0 && pv) hook.render_pv(req + k0, k - k0, nullptr, irr + k0 * g, sig + k0, cls + k0, band, pv_out);
  if (k > k0 && !pv) hook.render(req + k0, k - k0, nullptr, irr + k0 * g, sig + k0, cls + k0, band);
  const MidpointParams& mid = sample.rules.mid;
  P::for_each(k,
              SwitchApply{list, pixel, act_n, mid, mid.tol > 0.0f ? 0.125f * mid.tol : 2.5e-4f, groups, irr, sig, cls,
                          carried_irr.data(), carried_sig.data(), carried_class.data(), carried_n.data(), counts},
              s.stream);
  if (sample.pv.n != 0) {
    P::for_each(k,
                PvNodeCopy<SwitchRow>{{list, pixel}, k0, pv_out.slot, pv_out.codes, pv_out.table,
                                      render::pv_code_words(sample.pv.n, sample.pv.lights),
                                      render::pv_table_words(groups, sample.pv.lights), pv_codes.data(), pv_table.data()},
                s.stream);
  }
  return (k0 > 0 ? 1u : 0u) + (k > k0 ? 1u : 0u);
}

// P2-B B4b, after a window's pass (PHASE2 D-P2-4: "a pixel that shows an unpredicted boundary at the lower N re-renders
// that window at N_band"): every active pixel at N_base with an edge sample (s.edge: kBandEdge, its carried t_w node
// and its end node included) is escalated, and (B6a, pooled escalation) every one with an active 8-neighbour that has
// one, at either N: an edge on p's tail sub-samples k >= N_base alone lies within about 0.5 px of a neighbour's centre,
// so inside its 64 points' hull, which then shows it. That premise needs the 64 points' hull to reach past +-0.5 px,
// which the default pixel response (render::support_px 0.95) meets; a narrow one (sensor.aperture_px 0.5 with
// optics.blur_sigma_px <= 0.15) leaves a strip between neighbouring hulls that no 64-prefix sees, so a straight edge
// there stays T2's gap (none missed at support_px >= 0.75 in phase2-checks/p2_b6_pool_hull.py; ENGINE_CPP §0.1 "P2-B B6
// notes"). Its t_w node is re-rendered at N_band (slot 0 of the window clock) and becomes its carried sample; then one
// WindowEngine pass at N_band samples its window again, at its lattice level, from that node through every round,
// appended to the window's runs; the sensor's rows of these pixels are redirected to them (s.row_start, s.run_counts)
// and their carried samples, class words and N are the pass's (the next window carries from it). The main pass's runs
// of these pixels stay in the runs unused. Returns the sensor's row starts: the window's offsets, or s.row_start with
// escalations. Counters: SwitchStats::escalated (escalated_pooled of them by a neighbour's edge alone),
// escalated_samples (the re-rendered nodes and the pass's samples; not in pixel_samples), the nodes' steps
// (escalated_changed, _unquiescent, _eta; NodeApply) and the pass's render calls (in render_calls). The pass's own
// WindowStats go to out.escalation (P2-C C2: its rounds' samples, calls and jump counters); out.stats is the window's
// pass, the escalated pixels' unused runs included.
// T2 (ENGINE_CPP §0.1 "P2-B B4b notes"): an escalated pixel's runs are a uniform N_band run's, as a switched one's.
template <Space S>
const int64_t* Camera<S>::escalate(int64_t t_w, size_t n, const SampleConfig& cfg, RenderInto& hook,
                                   WindowResult& out) {
  Shared<S>& s = shared;
  const uint32_t band = sample.band_subsamples;
  const auto band_log2 = static_cast<uint8_t>(std::countr_zero(band));
  uint8_t* flags = s.flags.ensure(n);
  // s.select (2 n since switch_nodes) keeps the window's switches down in its first k0 entries; the escalated pixels
  // go to its second half.
  const uint32_t* down = s.select.ensure(2 * n);
  const auto k0 = static_cast<uint32_t>(out.band.switches - out.band.up);
  uint32_t* sel = s.select.data() + n;
  unsigned long long* words = s.switch_words.data();
  uint8_t* image = s.edge_image.data();  // P2-B B6a: pooled escalation (sample_and_sense sized it)
  P::zero(image, geo.n_pixels, s.stream);
  P::for_each(n, EdgeImage{s.act_pixel.data(), s.edge.data(), image}, s.stream);
  P::for_each(n,
              EscalateFlags{s.act_n.data(), s.edge.data(), s.act_pixel.data(), image, width, height, band_log2, flags,
                            s.switch_counts.data()},
              s.stream);
  P::select_flagged(flags, sel, n, &words[0], s.temp, s.stream);
  P::sync(s.stream);
  const auto k = static_cast<size_t>(words[0]);
  out.band.escalated = k;
  if (k == 0) return s.core.offsets();
  const size_t g = static_cast<size_t>(groups);
  uint32_t* pixel = s.esc_pixel.ensure(k);
  uint8_t* level = s.esc_level.ensure(k);
  P::for_each(k, GatherEscalated{sel, s.act_pixel.data(), s.act_level.data(), pixel, level}, s.stream);
  SampleRequest* req = s.switch_requests.ensure(k);
  float* irr = s.switch_irr.ensure(k * g);
  uint64_t* sig = s.switch_sig.ensure(k);
  uint32_t* cls = s.switch_class.ensure(k);
  P::for_each(k, NodeRequests{pixel, cam, band, req}, s.stream);
  const bool pv = sample.pv.n != 0 && band == sample.pv.n;  // P2-D D3a: the nodes' codes and tables
  const render::PvOut pv_out = pv ? pv_node_out(k) : render::PvOut{};
  if (pv) {
    hook.render_pv(req, k, nullptr, irr, sig, cls, band, pv_out);
  } else {
    hook.render(req, k, nullptr, irr, sig, cls, band);
  }
  const MidpointParams& mid = sample.rules.mid;
  P::for_each(k,
              NodeApply{pixel, sel, down, k0, s.core.offsets(), s.core.irr(), mid,
                        mid.tol > 0.0f ? 0.125f * mid.tol : 2.5e-4f, groups, irr, sig, cls, band_log2,
                        carried_irr.data(), carried_sig.data(), carried_class.data(), carried_n.data(),
                        s.switch_counts.data()},
              s.stream);
  if (sample.pv.n != 0) {
    P::for_each(k,
                PvNodeCopy<DirectRow>{{pixel}, 0, pv_out.slot, pv_out.codes, pv_out.table,
                                      render::pv_code_words(sample.pv.n, sample.pv.lights),
                                      render::pv_table_words(groups, sample.pv.lights), pv_codes.data(), pv_table.data()},
                s.stream);
  }
  int64_t* offsets = s.esc_offsets.ensure(k + 1);
  int64_t* rows = s.row_start.ensure(n);
  SampleConfig at_band = cfg;
  at_band.subsamples = band;
  at_band.band_subsamples = 0;
  typename WindowEngine<S>::Input in{t_w,     pixel,   level,   k,       carried_irr.data(), carried_sig.data(),
                                     pixel,   carried_class.data(), nullptr, nullptr, offsets};
  pv_input(in);  // P2-D D3a
  const size_t outside = s.outside_bytes();
  if (outside >= s.budget) {
    throw MemoryBudgetError("window at " + std::to_string(t_w) + " ns: the escalation of " + std::to_string(k) +
                            " pixels: the planner's and output stage's buffers and the held world hold " +
                            detail::mib(outside) + " MiB, the device budget is " + detail::mib(s.budget) +
                            " MiB (execution.device_budget_mb)");
  }
  out.escalation = s.core.run(in, at_band, hook, s.budget - outside, s.stream);
  const WindowStats& st = out.escalation;
  P::copy(rows, s.core.offsets(), n * sizeof(int64_t), s.stream);
  P::for_each(k, Redirect{sel, offsets, band_log2, rows, s.run_counts.data(), s.act_n.data()}, s.stream);
  s.redirected = true;
  out.stats.render_calls += 1 + st.render_calls;
  out.band.escalation_calls = 1 + st.render_calls;
  out.band.escalated_samples = k;
  for (const uint64_t m : st.samples_by_round) out.band.escalated_samples += m;
  return rows;
}

template <Space S>
WindowResult Camera<S>::window(int64_t t_w, const uint32_t* pixel, const uint8_t* level, const int64_t* rank,
                               size_t n) {
  Shared<S>& s = shared;
  const std::lock_guard<std::mutex> lock(s.mutex);
  const Clock::time_point t0 = Clock::now();
  const uint64_t n_pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  const SampleRules& rules = sample.rules;
  check_time(t_w, "window");
  check_cancel(sample.cancel, "window");  // BL-0b
  const HeldSet held_set = s.hold(t_w);
  if (!rules.fixed && level == nullptr && n > 0) throw std::invalid_argument("window: levels are required");
  s.seen.assign((n_pixels + 63) / 64, 0);
  for (size_t i = 0; i < n; ++i) {
    if (pixel[i] >= n_pixels || rank[i] < 0 || static_cast<uint64_t>(rank[i]) >= n_pixels) {
      throw std::invalid_argument("window: pixel id or rank out of range");
    }
    uint64_t& word = s.seen[pixel[i] / 64];
    const uint64_t bit = uint64_t{1} << (pixel[i] % 64);
    if ((word & bit) != 0) throw std::invalid_argument("window: duplicate pixel id " + std::to_string(pixel[i]));
    word |= bit;
    if (level != nullptr && !rules.fixed && level[i] > time::kWindowLog2 - rules.shift) {
      throw std::invalid_argument("window: lattice level above 23 - log2(min_dt_ns)");
    }
  }
  // K4: the active pixels in rank order (P2-B B4a: no plan, so N_band for every pixel unless forced).
  s.runs_of = nullptr;
  band_planned = false;
  band_pixel = false;
  pv_plan_ok = pv_vhat_ok = false;  // P2-D D3a: no plan, so no P&V in this window (sample_and_sense)
  uint32_t* act_pixel = s.act_pixel.ensure(n);
  uint8_t* act_level = s.act_level.ensure(n);
  if (n > 0) {
    P::upload(s.in_pixel.ensure(n), pixel, n * sizeof(uint32_t), s.stream);
    if (level != nullptr) P::upload(s.in_level.ensure(n), level, n, s.stream);
    P::upload(s.in_rank.ensure(n), rank, n * sizeof(int64_t), s.stream);
    P::for_each(n, Iota{s.order.ensure(n)}, s.stream);
    const bool alt = P::sort_pairs(s.in_rank.data(), s.rank_alt.ensure(n), s.order.data(), s.order_alt.ensure(n), n,
                                   rank_bits, s.temp, s.stream);
    P::for_each(n,
                GatherActive{alt ? s.order_alt.data() : s.order.data(), s.in_pixel.data(),
                             level != nullptr ? s.in_level.data() : nullptr, act_pixel, act_level},
                s.stream);
  }
  const double k4_ms = ms_since(t0);
  sensor::DeviceEvents ev;
  WindowResult out = sample_and_sense(t_w, n, &ev);
  out.held = held_set;
  out.sample_ms += k4_ms;
  const Clock::time_point t1 = Clock::now();
  out.events = sensor_.download_events(ev);
  out.sense_ms += ms_since(t1);
  return out;
}

template <Space S>
StepResult Camera<S>::step(int64_t t_w) {
  Shared<S>& s = shared;
  const std::lock_guard<std::mutex> lock(s.mutex);
  check_time(t_w, "step");
  check_cancel(sample.cancel, "step");  // BL-0b
  StepResult out;
  const HeldSet held_set = s.hold(t_w);
  const size_t n = plan_window(t_w, out);
  sensor::DeviceEvents ev;
  out.window = sample_and_sense(t_w, n, &ev);
  out.window.held = held_set;
  if (out.kappa.pixels > 0) {  // plan_window's sums, downloaded while the window was sampled
    const unsigned long long* w = s.plan_words.data();
    out.kappa = {w[0], static_cast<double>(w[1]) / 16.0, static_cast<double>(w[2]) / 16.0,
                 static_cast<double>(w[3]) / 16.0};
  }
  const Clock::time_point t1 = Clock::now();
  out.ready = output(ev, t_w, &out.events);
  out.output_ms = ms_since(t1);
  out.limiter = limit(out.ready, output_cut(t_w));
  return out;
}

// H0-H1 and K4 of a window (the plan, then the active pixels in rank order with their capped levels in
// shared.act_pixel / act_level): their count; plan stats, level cap and plan_ms in out.
template <Space S>
size_t Camera<S>::plan_window(int64_t t_w, StepResult& out) {
  Shared<S>& s = shared;
  const Clock::time_point t0 = Clock::now();
  s.runs_of = nullptr;  // K4 below rewrites act_pixel
  band_planned = false;
  band_pixel = false;
  const size_t n_pixels = geo.n_pixels;
  const bool fixed = sample.rules.fixed;
  // H0, K1, K2, H1: the tile plan.
  const int8_t* levels = nullptr;
  int32_t cap = 127;
  if (!fixed) {
    Planner& pl = *planner;
    pl.begin(t_w);
    render::GtSample* gt_pass = nullptr;
    uint32_t* states = nullptr;
    float* normals = nullptr;
    // P2-D D1: the obliquity factors and P&V's plan bits read the plan pass's geometric normals (every window with
    // moving-shadow pairs or moving lights: PlanSchedule::bits).
    const bool pv_bits = pl.plan_bits();
    if (pl.plan_pass()) {  // K1, while the host prepares the pixel stage's terms
      render::GtQuery* q = s.gt_queries.ensure(n_pixels);
      gt_pass = s.gt_out.ensure(n_pixels);
      // I8's band and P2-B B4a's seeds: the shadow states of the centre rays' hits (the plan pass's shadow rays).
      const auto bits = static_cast<size_t>(s.renderer.num_shadow_lights());
      if ((pl.schedule().band || pl.schedule().seeds) && bits > 0) states = s.plan_states.ensure(2 * n_pixels * bits);
      if (pv_bits) normals = s.plan_normals.ensure(3 * n_pixels);
      P::for_each(n_pixels, PlanQueries{t_w, static_cast<uint32_t>(width), q}, s.stream);
      s.renderer.gt_into(cam, q, n_pixels, gt_pass, s.stream, states, normals);
    }
    const PixelStage& st = pl.terms();
    if (st.needed) {
      static_assert(kStateMover == render::kShadowMover && kStateMulti == render::kShadowMulti &&
                    kStateUnknown == render::kShadowUnknown && sizeof(render::GtSample) % sizeof(float) == 0);
      enqueue_pixel_stage<S>(st, geo, &gt_pass->depth, sizeof(render::GtSample) / sizeof(float), plan, s.stream,
                             &gt_pass->instance, states, carried_class.data(), normals);
    }
    // On the host while the GPU runs K1 and K2: the uploads above return without waiting for K1
    // (measured: 17 us on sfm, where touches end 1.1 ms before K1 + K2 do).
    pl.touches();
    TileTotals tiles;
    if (st.needed) {
      tiles = collect_pixel_stage<S>(geo, plan, s.stream, st.params.band != 0, st.params.seeds != 0,
                                     st.params.obliquity != 0, st.params.n_ao != 0);
    }
    // P2-D D1f: the factors' lattice budget, from the configuration alone (motion.h obliquity_budget; the device
    // budget only batches).
    if (st.params.obliquity != 0) {
      pl.set_obliquity_budget(obliquity_budget(sample.groups, s.config.max_samples_per_window));
    }
    const TilePlan tp = pl.finish(st.needed ? &tiles : nullptr);
    std::vector<int64_t> counts(static_cast<size_t>(planner->schedule().l_max) + 1, 0);
    size_t active = 0;
    for (size_t t = 0; t < tp.level.size(); ++t) {
      if (tp.level[t] < 0) continue;
      counts[static_cast<size_t>(tp.level[t])] += planner->crop_pixels_per_tile()[t];
      active += planner->crop_pixels_per_tile()[t];
    }
    if (s.config.max_samples_per_window) {
      out.level_cap = level_cap_counts(std::move(counts), active, *s.config.max_samples_per_window,
                                       planner->schedule().l_min);
    }
    if (out.level_cap) cap = *out.level_cap;
    levels = upload_vector(tile_level, tp.level, s.stream);
    if (!tp.band.empty()) upload_vector(tile_band, tp.band, s.stream);  // P2-B B4a
    band_planned = !tp.band.empty();
    out.plan = tp.stats;
    if (tp.stats.pixel) {  // P2-B B6c
      pixel_band_map(tp, out.plan);
      band_planned = band_pixel = true;
    }
    // P2-D D3a: P&V's D_bound and brackets read v_hat in every planned window (and the plan bits where computed).
    if (pl.plan_pass() || sample.pv.n != 0) {
      upload_vector(tile_vhat, std::vector<float>(tp.v_hat.begin(), tp.v_hat.end()), s.stream);
    }
    pv_vhat_ok = sample.pv.n != 0;
    pv_plan_ok = sample.pv.n != 0 && st.needed && st.params.bits != 0;
    if (st.needed && st.params.bits != 0) {  // P2-D D1: R4 into the plan bits, and their counts (read below)
      unsigned long long* c = plan.pv_counts.ensure(4);
      P::zero(c, 4 * sizeof(unsigned long long), s.stream);
      P::for_each(n_pixels,
                  PvPlanPass{plan.pv_plan.data(), st.params.pv != 0 ? plan.pv_width.data() : nullptr, pixel_tile.data(),
                             levels, tile_vhat.data(), cap, st.params.pitch, width, crop.x0, crop.y0, crop.x1, crop.y1,
                             c, st.params.obliquity},
                  s.stream);
      P::download_async(s.pv_words.data(), c, 4 * sizeof(unsigned long long), s.stream);
      out.plan.pv_bits = true;
    }
    out.planned = true;
    out.kappa.pixels = pl.plan_pass() ? 1 : 0;  // a mark: the sums below follow (step())
  }
  // K4: the active pixels in rank order, with their (capped) levels.
  uint8_t* flags = s.flags.ensure(n_pixels);
  uint32_t* select = s.select.ensure(n_pixels);
  unsigned long long* count = s.counter.ensure(1);
  P::for_each(n_pixels,
              ActiveFlags{pixel_by_rank.data(), pixel_tile.data(), levels, static_cast<uint32_t>(width), crop, flags},
              s.stream);
  P::select_flagged(flags, select, n_pixels, count, s.temp, s.stream);
  unsigned long long n_active = 0;
  P::download(&n_active, count, sizeof(n_active), s.stream);
  if (out.plan.pv_bits) {  // P2-D D1: PvPlanPass's counts, downloaded before this sync
    const unsigned long long* w = s.pv_words.data();
    out.plan.oblique_pixels = w[0];
    out.plan.beyond_pixels = w[1];
    out.plan.rough_pixels = w[2];
    out.plan.thin_pixels = w[3];
  }
  const auto n = static_cast<size_t>(n_active);
  P::for_each(n,
              GatherSelected{select, pixel_by_rank.data(), pixel_tile.data(), levels, cap, s.act_pixel.ensure(n),
                             s.act_level.ensure(n)},
              s.stream);
  if (out.kappa.pixels > 0) {  // the plan pass's GT is in gt_out until the output stage
    unsigned long long* c = s.plan_counts.ensure(4);
    P::zero(c, 4 * sizeof(unsigned long long), s.stream);
    P::for_each(n,
                PlanCounts{s.act_pixel.data(), s.act_level.data(), pixel_tile.data(), tile_vhat.data(), s.gt_out.data(),
                           planner->schedule().d_px / (static_cast<double>(time::kWindowNs) * 1e-9), c},
                s.stream);
    P::download_async(s.plan_words.data(), c, 4 * sizeof(unsigned long long), s.stream);
  }
  out.plan_ms = ms_since(t0);
  return n;
}

// P2-B B6c, after Planner::finish under band_growth pixel: the band pixels (pixel_band, by pixel id) of the window's
// seeds (the pixel stage's seed_m and seed_kind; band_map.h): each seed's radius (BandRadius, with the tiles' floors
// tp.band_floor) and their histogram, read back (one sync) for the cascade's levels, then the cascade on the crop grown
// by the largest radius (band_domain: exact on the crop, whose pixels alone read the map; the review measured about
// 0.85 ms per level over a 640 x 480 frame, 8 % of a 64 x 64 crop's window at 14 levels). Without a plan pass there is
// no seed and no dynamic tile: an empty band. stats: the largest radius and the levels.
template <Space S>
void Camera<S>::pixel_band_map(const TilePlan& tp, PlanStats& stats) {
  Shared<S>& s = shared;
  const size_t n = geo.n_pixels;
  uint8_t* band = pixel_band.ensure(n);
  if (tp.band_floor.empty()) {
    P::zero(band, n, s.stream);
    return;
  }
  const int32_t cap = std::max(width, height);
  const auto n_levels = static_cast<size_t>(cap) + 1;
  uint32_t* hist = plan.band_levels.ensure(n_levels);
  int32_t* radius = plan.band_radius.ensure(n);
  P::zero(hist, n_levels * sizeof(uint32_t), s.stream);
  const double* floor_t = upload_vector(plan.band_floor, tp.band_floor, s.stream);
  P::for_each(n,
              BandRadius{plan.seed_m.data(), plan.seed_kind.data(), floor_t, pixel_tile.data(), geo.support_px, cap,
                         radius, hist},
              s.stream);
  std::vector<uint32_t> host(n_levels);
  P::download(host.data(), hist, n_levels * sizeof(uint32_t), s.stream);
  const std::vector<int32_t> levels = band_levels(host);
  stats.radius_max = levels.empty() ? -1 : levels.front();
  stats.radius_levels = levels.size();
  const BandDomain dom = band_domain(crop.x0, crop.y0, crop.x1, crop.y1, stats.radius_max, width, height);
  band_cascade<S>(radius, levels, width, height, dom, band, plan.band_tmp.ensure(n), s.stream);
}

// K14-K15: the window's raw events through the crop, GT and MOVING into the held buffer, the output
// sort, and the events reported before t_{w+1} + clock offset back to the host.
template <Space S>
std::vector<OutEvent> Camera<S>::output(const sensor::DeviceEvents& ev, int64_t t_w, uint64_t* kept) {
  Shared<S>& s = shared;
  const bool with_gt = s.config.gt;
  const uint32_t* select = nullptr;
  size_t n = ev.count;
  {
    // The budget (ENGINE_CPP I3 open item (d)): what the stage's buffers grow by for the window's events (the crop's
    // count bounded by theirs), the held events' old allocation during their copy and the sort's temporary storage, with
    // everything else the engine holds; the WindowEngine's buffers go first when they do not fit beside it (the sensor
    // has read the window's runs), and a window that does not fit then raises before allocating.
    using detail::extra;
    const size_t m = ev.count, total = n_held + m, grown = detail::runs_grown(held.capacity(), total, true);
    size_t grow = extra(s.out_alt, total) + extra(s.keys, total) + extra(s.keys_alt, total) + extra(s.order, total) +
                  extra(s.order_alt, total);
    if (grown > held.capacity()) grow += grown * sizeof(OutEvent) - (n_held > 0 ? 0 : held.bytes());
    if (with_gt) grow += extra(s.gt_queries, m) + extra(s.gt_out, m);
    if (s.config.crop) grow += extra(s.flags, m) + extra(s.select, m);
    const size_t temp = std::max(P::sort_temp_bytes(total, 64), s.config.crop ? P::select_temp_bytes(m) : 0);
    grow += temp > s.temp.bytes() ? temp - s.temp.bytes() : 0;
    if (s.core.held_bytes() + s.outside_bytes() + grow > s.budget) {
      s.core.release();
      last_active = 0;
      s.runs_of = nullptr;
    }
    if (const size_t need = s.core.held_bytes() + s.outside_bytes() + grow; need > s.budget) {
      throw MemoryBudgetError("window at " + std::to_string(t_w) + " ns: the output stage needs " + detail::mib(need) +
                              " MiB for its " + std::to_string(total) + " events with the planner's buffers and the "
                              "held world, above the device budget of " + detail::mib(s.budget) +
                              " MiB (execution.device_budget_mb)");
    }
  }
  if (n > 0 && s.config.crop) {
    uint8_t* flags = s.flags.ensure(n);
    uint32_t* sel = s.select.ensure(n);
    unsigned long long* count = s.counter.ensure(1);
    P::for_each(n, CropFlags{ev.data, static_cast<uint32_t>(width), crop.x0, crop.y0, crop.x1, crop.y1, flags},
                s.stream);
    P::select_flagged(flags, sel, n, count, s.temp, s.stream);
    unsigned long long m = 0;
    P::download(&m, count, sizeof(m), s.stream);
    n = static_cast<size_t>(m);
    select = sel;
  }
  *kept = n;
  const size_t total = n_held + n;
  detail::grow_keep(held, n_held, detail::runs_grown(held.capacity(), total, true), s.stream);
  OutEvent* events = held.data();
  render::GtQuery* q = with_gt ? s.gt_queries.ensure(n) : nullptr;
  P::for_each(n, BuildEvents{ev.data, select, static_cast<uint32_t>(width), sensor_.readout(), events + n_held, q},
              s.stream);
  if (with_gt && n > 0) {
    render::GtSample* g = s.gt_out.ensure(n);
    s.renderer.gt_into(cam, q, n, g, s.stream);
    P::for_each(n, AttachGt{g, s.moving, events + n_held}, s.stream);
  }
  // Three stable passes: t_cross, (x, y), t_reported.
  std::vector<OutEvent> ready;
  if (total == 0) return ready;
  uint32_t* perm = s.order.ensure(total);
  P::for_each(total, Iota{perm}, s.stream);
  for (int pass = 0; pass < 3; ++pass) {
    P::for_each(total, OutKeys{events, s.order.data(), pass, y_bits, s.keys.ensure(total)}, s.stream);
    const int bits = sensor::order_key_bits(pass, y_bits);
    if (P::sort_pairs(s.keys.data(), s.keys_alt.ensure(total), s.order.data(), s.order_alt.ensure(total), total, bits,
                      s.temp, s.stream)) {
      s.order.swap(s.order_alt);
    }
  }
  unsigned long long* count = s.counter.ensure(1);
  P::zero(count, sizeof(*count), s.stream);
  const int64_t limit = output_cut(t_w);
  P::for_each(total, GatherEvents{events, s.order.data(), s.out_alt.ensure(total), count, limit}, s.stream);
  unsigned long long n_ready = 0;
  P::download(&n_ready, count, sizeof(n_ready), s.stream);
  ready.resize(static_cast<size_t>(n_ready));
  P::download(ready.data(), s.out_alt.data(), ready.size() * sizeof(OutEvent), s.stream);
  n_held = total - ready.size();
  P::copy(events, s.out_alt.data() + ready.size(), n_held * sizeof(OutEvent), s.stream);
  return ready;
}

// The output stage's cut after window t_w: events of later windows are reported at or after the window end (d0 >= q),
// so the cut is the end on the sensor clock: offset and drift, through the readout's own formula (a zero delay and no
// quantum); non-decreasing in t for |drift| < 1e6 ppm. With no drift it is t_w + W + offset. Every event reported
// before it has been returned once the window's output stage ran.
template <Space S>
int64_t Camera<S>::output_cut(int64_t t_w) const {
  return sensor::output_cut(sensor_.readout(), t_w + time::kWindowNs);
}

// P2-H H4b (plan p2h.md §6): the rate limiter on the output stage's final events, on the host after their download
// (sensor/limiter.h: a pure function of the time-sorted stream and its integer state). Events are complete up to
// complete_ns (the output cut; the end of the stream at flush); the limiter keeps a bin later events may still join
// (q = 0 only) for the next call, so the events it returns are final. Off, it leaves them untouched.
template <Space S>
LimiterCounters Camera<S>::limit(std::vector<OutEvent>& events, int64_t complete_ns) {
  LimiterCounters out;
  if (!limiter_.config().on) return out;
  const Clock::time_point t0 = Clock::now();
  const sensor::LimiterState before = limiter_.state();
  limiter_.apply(events, complete_ns);
  const sensor::LimiterState& after = limiter_.state();
  out.on = true;
  out.kept = after.kept - before.kept;
  out.dropped = after.dropped - before.dropped;
  out.capped_bins = after.capped_bins - before.capped_bins;
  out.held = limiter_.held().size();
  out.ms = ms_since(t0);
  limiter_ms_ += out.ms;
  return out;
}

template <Space S>
LimiterCounters Camera<S>::limiter_counters() const {
  const std::lock_guard<std::mutex> lock(shared.mutex);
  LimiterCounters out;
  if (!limiter_.config().on) return out;
  const sensor::LimiterState& st = limiter_.state();
  out.on = true;
  out.kept = st.kept;
  out.dropped = st.dropped;
  out.capped_bins = st.capped_bins;
  out.held = limiter_.held().size();
  out.ms = limiter_ms_;
  return out;
}

template <Space S>
std::vector<OutEvent> Camera<S>::flush() {
  Shared<S>& s = shared;
  const std::lock_guard<std::mutex> lock(s.mutex);
  std::vector<OutEvent> out(n_held);
  P::download(out.data(), held.data(), out.size() * sizeof(OutEvent), s.stream);
  n_held = 0;
  if (limiter_.config().on) {  // the end of the stream: every held bin is decided
    const Clock::time_point t0 = Clock::now();
    limiter_.flush(out);
    limiter_ms_ += ms_since(t0);
  }
  // The end of the camera's run: the output stage's buffers (O(events)) go.
  held.reset();
  s.out_alt.reset();
  s.keys.reset();
  s.keys_alt.reset();
  s.order.reset();
  s.order_alt.reset();
  s.gt_queries.reset();
  s.gt_out.reset();
  s.flags.reset();
  s.select.reset();
  return out;
}

template <Space S>
std::vector<render::GtSample> Camera<S>::gt(const float* x, const float* y, const int64_t* t_ns, size_t n) {
  Shared<S>& s = shared;
  const std::lock_guard<std::mutex> lock(s.mutex);
  s.renderer.check_gt(cam, x, y, t_ns, n);
  s.renderer.hold({}, 0);  // any times: no held set
  std::vector<render::GtQuery> queries(n);
  for (size_t i = 0; i < n; ++i) queries[i] = {t_ns[i], x[i], y[i]};
  std::vector<render::GtSample> out(n);
  if (n == 0) return out;
  P::upload(s.gt_queries.ensure(n), queries.data(), n * sizeof(render::GtQuery), s.stream);
  s.renderer.gt_into(cam, s.gt_queries.data(), n, s.gt_out.ensure(n), s.stream);
  P::download(out.data(), s.gt_out.data(), n * sizeof(render::GtSample), s.stream);
  return out;
}

}  // namespace

}  // namespace engine_detail

}  // namespace balla::engine
