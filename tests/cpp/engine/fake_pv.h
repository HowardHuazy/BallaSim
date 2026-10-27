// P2-D D3a: a WindowEngine renderer stand-in for P&V's shadow mode (engine/localize.h), in either memory space. Every
// pixel sees one surface (instance 1, material 0) under one shadow-casting light; its N sub-samples sit at the
// renderer's own positions (the warp table and Owen-Sobol patterns of render/ray.h), and a shadow boundary sweeps the
// pixel at a per-pixel direction, speed and time: a straight edge, a thin line of `width` px, or a wiggly edge (relief-
// like), by pixel id mod 3 under kind 3. Irradiance, signature and class word follow from the sub-samples' states as
// the shade kernels form them (shade.h), and so do the P&V outputs (render/shade/codes.h) and the probes, so that a
// window with P&V must give the runs of the window without it, and every flip time is known in closed form.
// P2-D D3b: kind 4, a slanted caster's edge (its arrival time a ratio of affine forms in u, F020: the projective
// stage is exact); kind 5, a double edge (shadow, a lit gap of `width` px, shadow again: three flips of a sub-sample
// when they fall in one interval); a probe hook that reports the state `lie` ns late (a bug the state check must see).
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "core/pmath/trig.h"
#include "core/rng/warp_table.h"
#include "engine/par.h"
#include "engine/window.h"
#include "fake.h"
#include "render/shade/codes.h"
#include "render/shade/shade.h"

namespace balla::engine::test {

struct PvFakeScene {
  PvParams pv;      // positions (pv.warp in the hook's space), n
  int64_t t_w;
  uint32_t kind;    // 0 straight, 1 thin, 2 wiggle, 3 by pixel id mod 3
  double width;     // the thin line's width (px)
  double speed;     // the boundary's speed scale (px/s)
  double curve;     // a smooth term curve (dt / W)^2 in group 0's irradiance (no flip: J leaves it, R5 sees it)
  double slant;     // kind 4: the arrival time's perspective factor 1 + slant m.u (m the edge's tangent, 1/px)
  double wiggle;    // kind 2's relief amplitude (px; at a 0.3 px wavelength, or wiggle_k rad/px)
  double wiggle_k;

  BALLA_HD uint32_t kind_of(uint32_t pixel) const { return kind == 3 ? pixel % 3 : kind; }
  // The boundary of pixel `pixel`: unit normal (nx, ny), speed v (px/s) and the time tc (s after t_w) at which it
  // crosses the pixel centre.
  BALLA_HD void edge(uint32_t pixel, double& nx, double& ny, double& v, double& tc) const {
    const uint32_t h = pixel * 2654435761u;
    const double th = 6.283185307179586 * static_cast<double>(h % 1000u) / 1000.0;
    nx = pmath::cos(th);  // pmath: the host and the device agree bit for bit
    ny = pmath::sin(th);
    v = speed * (0.5 + static_cast<double>((h >> 10) % 100u) / 66.0);
    tc = 8.388608e-3 * (0.15 + 0.7 * static_cast<double>((h >> 17) % 1000u) / 1000.0);
  }
  // The signed position of sub-sample offset (ux, uy) across the boundary at time t (> 0: the boundary has passed).
  BALLA_HD double across(uint32_t pixel, double ux, double uy, int64_t t) const {
    double nx, ny, v, tc;
    edge(pixel, nx, ny, v, tc);
    double d = nx * ux + ny * uy;
    if (kind_of(pixel) == 2) d += wiggle * pmath::sin(wiggle_k * (-ny * ux + nx * uy));  // relief: 0.06 px at 0.3 px
    // kind 4: crossing at t = tc + d / (v (1 + slant m.u)), m the tangent (F020's ratio of affine forms in u).
    const double scale = kind_of(pixel) == 4 ? 1.0 + slant * (-ny * ux + nx * uy) : 1.0;
    return v * (static_cast<double>(t - t_w) * 1e-9 - tc) * scale - d;
  }
  BALLA_HD bool shadowed(uint32_t pixel, uint32_t k, int64_t t) const {
    const Vec2<float> u = pv_position(pv, 0, pixel, k);
    const double a = across(pixel, static_cast<double>(u.x), static_cast<double>(u.y), t);
    const uint32_t kd = kind_of(pixel);
    return kd == 1 ? (a > 0.0 && a < width) : kd == 5 ? (a > 0.0 && !(a >= width && a < 2.0 * width)) : a > 0.0;
  }
};

// Request i at the call's N (out_index row): irradiance [0.5 + 0.5 R + curve (dt / W)^2, 0.25], R the lit share;
// signature, class word, and with pv the outputs.
struct PvFakeKernel {
  PvFakeScene sc;
  const SampleRequest* requests;
  const uint32_t* out_index;
  float* irr;
  uint64_t* sig;
  uint32_t* klass;
  uint32_t n;
  render::PvOut pv;
  bool with_pv;
  BALLA_HD void operator()(size_t i) const {
    const SampleRequest& q = requests[i];
    const size_t row = out_index != nullptr ? out_index[i] : i;
    const int64_t t = sc.t_w + q.t_off_ns;
    uint64_t s = 0;
    uint32_t hash = 0, shadowed = 0, occ_or = 0, occ_and = ~0u;
    uint32_t words[render::kMaxSubsamples * 4 / 32] = {};
    for (uint32_t k = 0; k < n; ++k) {
      const uint32_t m = sc.shadowed(q.pixel, k, t) ? 1u : 0u;
      const uint64_t g = render::geometry_term(k, 1, 0);
      s += g + render::shadow_term(k, m);
      hash += static_cast<uint32_t>(g >> 32);
      shadowed += m;
      occ_or |= m;
      occ_and &= m;
      words[k * 4 / 32] |= render::pv_code(0, m) << (k * 4 % 32);
    }
    const float r = static_cast<float>(n - shadowed) / static_cast<float>(n);
    const double s_w = static_cast<double>(q.t_off_ns) / static_cast<double>(kW);
    irr[2 * row] = 0.5f + 0.5f * r + static_cast<float>(sc.curve * s_w * s_w);
    irr[2 * row + 1] = 0.25f;
    sig[row] = s;
    const uint32_t word = render::sample_class((hash << 16) + shadowed, false, render::class_masks(occ_or, occ_and, 1),
                                               false);
    if (klass != nullptr) klass[row] = word;
    if (!with_pv) return;
    const uint32_t e = render::pv_claim(pv, word);
    pv.slot[row] = e;
    if (e >= render::kPvOverflow) return;
    const uint32_t cw = render::pv_code_words(n, 1), tw = render::pv_table_words(2, 1);
    for (uint32_t w = 0; w < cw; ++w) pv.codes[static_cast<size_t>(e) * cw + w] = words[w];
    uint32_t* tb = pv.table + static_cast<size_t>(e) * tw;
    tb[0] = 1;
    const uint32_t row_words[] = {1u, 0u, n, float_to_bits(irr[2 * row]), float_to_bits(0.25f), float_to_bits(0.5f),
                                  float_to_bits(static_cast<float>(n)), float_to_bits(r)};
    for (uint32_t w = 0; w < 8; ++w) tb[1 + w] = row_words[w];
  }
};

struct PvFakeProbe {
  PvFakeScene sc;
  const SampleRequest* requests;
  render::ProbeOut* out;
  int64_t lie;  // D3b: the state reported is the one `lie` ns later (0: the truth)
  BALLA_HD void operator()(size_t i) const {
    const SampleRequest& q = requests[i];
    out[i] = {1u, 0u, sc.shadowed(q.pixel, q.k_begin, sc.t_w + q.t_off_ns + lie) ? 1u : 0u, 1.0f};
  }
};

template <Space S>
class PvFakeHook final : public RenderHook {
 public:
  explicit PvFakeHook(PvFakeScene sc, int64_t lie = 0) : sc_(sc), lie_(lie) {}
  void render(SampleRequest* requests, size_t n, const uint32_t* out_index, float* irr, uint64_t* sig, uint32_t* klass,
              uint32_t subsamples) override {
    Par<S>::for_each(n, PvFakeKernel{sc_, requests, out_index, irr, sig, klass, subsamples, {}, false}, nullptr);
    rendered += n;
  }
  void render_pv(SampleRequest* requests, size_t n, const uint32_t* out_index, float* irr, uint64_t* sig,
                 uint32_t* klass, uint32_t subsamples, const render::PvOut& pv) override {
    Par<S>::for_each(n, PvFakeKernel{sc_, requests, out_index, irr, sig, klass, subsamples, pv, true}, nullptr);
    rendered += n;
    ++pv_calls;
  }
  void probe(SampleRequest* requests, size_t n, render::ProbeOut* out) override {
    Par<S>::for_each(n, PvFakeProbe{sc_, requests, out, lie_}, nullptr);
    probes += n;
  }
  size_t rendered = 0, pv_calls = 0, probes = 0;

 private:
  PvFakeScene sc_;
  int64_t lie_;
};

struct PvFakeSetup {
  FakeSetup base;  // pixels, levels, cfg (2 groups), carried irr / sig (replaced by the scene's at t_w)
  uint32_t kind = 0;
  double width = 0.1, speed = 1000.0, curve = 0.0;
  bool pv = true;
  bool carried_codes = true;  // the carried nodes' codes and tables (false: none, so edge-class ones are R0c)
  float vhat = 0;             // PvParams::vhat (px/s; 0: the scene's speed bound)
  double slant = 0.6;         // kind 4's perspective factor (1/px)
  double wiggle = 0.06, wiggle_k = 20.94395;  // kind 2's relief (px, rad/px)
  uint32_t stages = 0xFFu;    // PvParams::stages (the ladder)
  std::vector<uint8_t> plan;  // P&V's plan bits by pixel id (empty: none)
  int64_t lie = 0;            // the probe hook's lie (ns)
};

struct PvFakeRuns {
  FakeRuns runs;
  std::vector<PvFlip> flips;
  PvFakeScene scene;  // pv.warp in host memory
  size_t probes = 0, pv_calls = 0;
};

// A P&V window of n pixels at N 256 (levels 1-7, as fake_setup's) with the economy-free schedule, 2 groups, the eta_L
// test and min_dt 1024.
inline PvFakeSetup pv_fake_setup(uint32_t seed, size_t n, uint32_t kind) {
  PvFakeSetup s;
  s.base = fake_setup(seed, n, false);
  s.base.cfg.subsamples = 256;
  s.kind = kind;
  return s;
}

// One window in space S on `engine`: the carried nodes are the scene's at t_w (rendered here with their codes unless
// carried_codes is false), with the renderer's warp table (default settings) uploaded to S.
template <Space S>
PvFakeRuns pv_fake_window(WindowEngine<S>& engine, const PvFakeSetup& ps, size_t budget) {
  using P = Par<S>;
  const FakeSetup& s = ps.base;
  const size_t n = s.pixel.size(), g = 2;
  const rng::WarpTable<float> warp(rng::PixelResponse(1.0, 0.15, 3.0), 256);
  Buffer<float, S> cdf, slope;
  Buffer<uint16_t, S> guide;
  PvFakeScene sc{};
  sc.pv.n = 256;
  sc.pv.lights = 1;
  sc.pv.seed = 11;
  sc.pv.warp = {upload_vector(cdf, warp.cdf(), nullptr), upload_vector(slope, warp.slope(), nullptr),
                upload_vector(guide, warp.guide(), nullptr), warp.cells(), warp.view().dx};
  sc.pv.vhat = ps.vhat > 0 ? ps.vhat : static_cast<float>(1.6 * ps.speed);
  sc.t_w = kTw;
  sc.kind = ps.kind;
  sc.width = ps.width;
  sc.speed = ps.speed;
  sc.curve = ps.curve;
  sc.slant = ps.slant;
  sc.wiggle = ps.wiggle;
  sc.wiggle_k = ps.wiggle_k;
  sc.pv.stages = ps.stages;
  SampleConfig cfg = s.cfg;
  if (ps.pv) cfg.pv = sc.pv;
  Buffer<uint32_t, S> pixel, klass, codes, table;
  Buffer<uint8_t, S> level;
  Buffer<float, S> irr;
  Buffer<uint64_t, S> sig;
  Buffer<SampleRequest, S> req;
  P::upload(pixel.ensure(n), s.pixel.data(), n * 4, nullptr);
  P::upload(level.ensure(n), s.level.data(), n, nullptr);
  const uint32_t cw = render::pv_code_words(256, 1), tw = render::pv_table_words(2, 1);
  codes.ensure(n * cw);
  table.ensure(n * tw);
  // The carried nodes: the scene at t_w, rows a (the batch's rows without carried_at).
  std::vector<SampleRequest> host_req(n);
  for (size_t a = 0; a < n; ++a) host_req[a] = {0, 0, s.pixel[a], 0, 256, 0, 0};
  P::upload(req.ensure(n), host_req.data(), n * sizeof(SampleRequest), nullptr);
  Buffer<uint32_t, S> slot, count;
  P::zero(count.ensure(1), 4, nullptr);
  const render::PvOut carried{slot.ensure(n), codes.data(), table.data(), count.data(), static_cast<uint32_t>(n), 1};
  P::for_each(n, PvFakeKernel{sc, req.data(), nullptr, irr.ensure(n * g), sig.ensure(n), klass.ensure(n), 256, carried,
                              true},
              nullptr);
  // pv.all wrote every row's entry in request order (one call, entries claimed in row order on the host; on the
  // device the order is not fixed, so the rows are put in place from the slots).
  std::vector<uint32_t> hs(n), hc(n * cw), ht(n * tw), oc(n * cw), ot(n * tw);
  P::download(hs.data(), slot.data(), n * 4, nullptr);
  P::download(hc.data(), codes.data(), n * cw * 4, nullptr);
  P::download(ht.data(), table.data(), n * tw * 4, nullptr);
  for (size_t a = 0; a < n; ++a) {
    for (uint32_t w = 0; w < cw; ++w) oc[a * cw + w] = hc[hs[a] * cw + w];
    for (uint32_t w = 0; w < tw; ++w) ot[a * tw + w] = ht[hs[a] * tw + w];
    if (!ps.carried_codes) ot[a * tw] = render::kPvNone;
  }
  P::upload(codes.data(), oc.data(), n * cw * 4, nullptr);
  P::upload(table.data(), ot.data(), n * tw * 4, nullptr);
  PvFakeHook<S> hook(sc, ps.lie);
  typename WindowEngine<S>::Input in{kTw, pixel.data(), level.data(), n, irr.data(), sig.data(), nullptr,
                                     klass.data()};
  in.pv_codes = codes.data();
  in.pv_table = table.data();
  Buffer<uint8_t, S> plan;
  if (!ps.plan.empty()) {
    P::upload(plan.ensure(ps.plan.size()), ps.plan.data(), ps.plan.size(), nullptr);
    in.pv_plan = plan.data();
  }
  PvFakeRuns out;
  const size_t others = BufferMeter<S>::live.load() - engine.held_bytes();
  BufferMeter<S>::reset_peak();
  engine.keep_pv_flips(true);
  out.runs.stats = engine.run(in, cfg, hook, budget, nullptr);
  out.runs.peak = BufferMeter<S>::peak.load() - others;
  out.runs.rendered = hook.rendered;
  out.probes = hook.probes;
  out.pv_calls = hook.pv_calls;
  const size_t samples = engine.samples();
  out.runs.offsets.resize(n + 1);
  out.runs.t.resize(samples);
  out.runs.irr.resize(samples * g);
  out.runs.last_irr.resize(n * g);
  out.runs.last_sig.resize(n);
  P::download(out.runs.offsets.data(), engine.offsets(), (n + 1) * 8, nullptr);
  P::download(out.runs.t.data(), engine.t(), samples * 8, nullptr);
  P::download(out.runs.irr.data(), engine.irr(), samples * g * 4, nullptr);
  P::download(out.runs.last_irr.data(), irr.data(), n * g * 4, nullptr);
  P::download(out.runs.last_sig.data(), sig.data(), n * 8, nullptr);
  out.flips = engine.pv_flips();
  out.scene = sc;
  out.scene.pv.warp = {};
  return out;
}

template <Space S>
PvFakeRuns pv_fake_window(const PvFakeSetup& s, size_t budget) {
  WindowEngine<S> engine;
  return pv_fake_window(engine, s, budget);
}

}  // namespace balla::engine::test
