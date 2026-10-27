// WindowEngine<Space> member definitions; included by window.cpp (host instantiation) and engine.cu
// (device instantiation) only.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "engine/window.h"

namespace balla::engine {

namespace detail {

// Grows b to exactly n elements, keeping its first `keep`: the old allocation lives until they are
// copied (without contents to keep, it is freed first).
template <Space S, class T>
void grow_keep(Buffer<T, S>& b, size_t keep, size_t n, Stream stream) {
  if (n <= b.capacity()) return;
  if (keep == 0) {
    b.ensure(n);
    return;
  }
  Buffer<T, S> next;
  next.ensure(n);
  Par<S>::copy(next.data(), b.data(), keep * sizeof(T), stream);
  b.swap(next);  // freeing the old allocation waits for the copy on the device
}

// Bytes b holds once it has at least n elements, and its old allocation if it must grow for them.
template <class T, Space S>
size_t after(const Buffer<T, S>& b, size_t n) {
  return std::max(b.capacity(), n) * sizeof(T);
}
template <class T, Space S>
size_t extra(const Buffer<T, S>& b, size_t n) {
  return n > b.capacity() ? (n - b.capacity()) * sizeof(T) : 0;
}
template <class T, Space S>
size_t old_if_grown(const Buffer<T, S>& b, size_t n) {
  return n > b.capacity() ? b.bytes() : 0;
}

// Elements the runs grow to for n: n, or with headroom (n + n/4) when the budget allows it, so that a
// clip whose windows grow slowly does not reallocate them at every new maximum.
inline size_t runs_grown(size_t capacity, size_t n, bool headroom) {
  return n <= capacity ? capacity : headroom ? n + n / 4 : n;
}

// Key bits of the assembly's sort: (before << 24) | dt with before < s0.
inline int pool_key_bits(size_t s0) {
  return static_cast<int>(std::bit_width(s0 > 0 ? s0 - 1 : 0)) + time::kWindowLog2 + 1;
}

inline std::string mib(size_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3f", static_cast<double>(bytes) / 1048576.0);
  return buf;
}

// Adds a batch's P2-0b counters to the window's.
struct AddWords {
  const unsigned long long* from;
  unsigned long long* to;
  BALLA_HD void operator()(size_t i) const { to[i] += from[i]; }
};

}  // namespace detail

template <Space S>
size_t WindowEngine<S>::batch_bytes() const {
  return off0_.bytes() + dt_.bytes() + owner_.bytes() + before_.bytes() + out_index_.bytes() + index_alt_.bytes() +
         select_.bytes() + irr_.bytes() + sig_.bytes() + keys_.bytes() + keys_alt_.bytes() + candidates_.bytes() +
         list_.bytes() + requests_.bytes() + sel_.bytes() + counts_.bytes() + ex_.bytes() + counters_.bytes() +
         cls_.bytes() + outer_.bytes() + cand_outer_.bytes() + tag_.bytes() + temp_.bytes() + pv_slot_.bytes() +
         pv_codes_.bytes() + pv_table_.bytes() + pv_count_.bytes() + pv_route_.bytes() + pv_flipc_.bytes() +
         pv_stats_.bytes() + pv_off_.bytes() + pv_nreq_.bytes() + pv_reqoff_.bytes() + pv_j_.bytes() + pv_thin_.bytes() +
         pv_flips_.bytes() + pv_fits_.bytes() + pv_req_.bytes() + pv_probe_.bytes() + pv_brk_.bytes() +
         pv_mcodes_.bytes() + pv_mtable_.bytes() + pv_mcount_.bytes() + pv_nkey_.bytes() + pv_nval_.bytes();
}

// P2-D D3a: the P&V buffers at the sizes pv_sz_ (each at max(held, needed)). D3b: the groups' fits, the rounds'
// requests and results or (sort) the nodes' sort in their place, the brackets with the pixels' flip offsets, and the
// refinement round's midpoint outputs.
template <Space S>
size_t WindowEngine<S>::pv_bytes() const {
  using detail::after;
  const PvSizes& z = pv_sz_;
  const size_t stage = z.stage ? z.slots : 0;  // localize's own per-slot and per-pixel arrays
  const size_t rounds = z.sort ? 0 : 2 * z.flips, sort = z.sort ? 4 * z.flips : 0;
  return after(pv_slot_, z.rows) + after(pv_codes_, z.entries * z.code_words) +
         after(pv_table_, z.entries * z.table_words) + after(pv_count_, z.slots > 0 ? 1 : 0) +
         after(pv_route_, z.slots) + after(pv_flipc_, stage) + after(pv_stats_, 4 * z.slots) +
         after(pv_j_, stage * z.groups) + after(pv_thin_, stage) +
         after(pv_off_, (z.stage || z.brackets > 0) && z.pixels > 0 ? 2 * (z.pixels + 1) : 0) +
         after(pv_nreq_, z.flips > 0 ? z.flips + 1 : 0) +
         after(pv_reqoff_, z.flips > 0 ? z.flips + 1 : 0) + after(pv_flips_, z.flips) + after(pv_fits_, z.ng) +
         after(pv_req_, rounds) + after(pv_probe_, rounds) + after(pv_nkey_, sort) + after(pv_nval_, sort) +
         after(pv_brk_, z.brackets) + after(pv_mcodes_, z.mid * z.code_words) +
         after(pv_mtable_, z.mid * z.table_words) + after(pv_mcount_, z.mid > 0 ? 1 : 0);
}

template <Space S>
size_t WindowEngine<S>::held_bytes() const {
  return batch_bytes() + count64_.bytes() + off64_.bytes() + offsets_.bytes() + t_.bytes() + fin_irr_.bytes() +
         diag_.bytes() + jump_.bytes() + pv_counts_.bytes();
}

// Frees what the batches hold between them, and the runs' capacity while the window has neither
// runs nor a reservation for them (what is left is an earlier window's). A rare path (overflows).
template <Space S>
void WindowEngine<S>::release_batches() {
  for (auto* b : {&off0_, &dt_, &owner_, &before_, &out_index_, &index_alt_, &select_, &counts_, &ex_, &cls_}) {
    b->reset();
  }
  for (auto* b : {&sig_, &keys_, &keys_alt_, &candidates_, &list_, &outer_, &cand_outer_}) b->reset();
  for (auto* b : {&pv_slot_, &pv_codes_, &pv_table_, &pv_count_, &pv_route_, &pv_flipc_, &pv_stats_, &pv_off_,
                  &pv_nreq_, &pv_reqoff_, &pv_mcodes_, &pv_mtable_, &pv_mcount_, &pv_nval_}) {
    b->reset();
  }
  pv_j_.reset();
  pv_thin_.reset();
  pv_sz_ = PvSizes{};
  pv_flips_.reset();
  pv_fits_.reset();
  pv_req_.reset();
  pv_probe_.reset();
  pv_brk_.reset();
  pv_nkey_.reset();
  irr_.reset();
  tag_.reset();
  requests_.reset();
  sel_.reset();
  counters_.reset();
  temp_.reset();
  if (samples_ == 0 && !runs_reserved_) {
    t_.reset();
    fin_irr_.reset();
  }
}

template <Space S>
void WindowEngine<S>::release() {
  samples_ = 0;
  runs_reserved_ = false;
  release_batches();
  for (auto* b : {&count64_, &off64_}) b->reset();
  offsets_.reset();
  diag_.reset();
  jump_.reset();
  pv_counts_.reset();
}

// What this engine holds after a step of a batch (nb pixels, s0 lattice samples) that brings it to
// `samples` samples, keeping the first `kept` while its per-sample buffers grow, with up to
// `requests` requests in a round, `list` listed and `candidates` candidate intervals: every buffer at
// max(held, needed), counting the batch's assembly and its runs at that sample count (so that the
// assembly needs no check of its own), the old allocation of the largest buffer grown with its
// contents (held twice during the copy), and the primitives' temporary storage.
template <Space S>
size_t WindowEngine<S>::footprint(const SampleConfig& cfg, size_t nb, size_t s0, size_t samples, size_t kept,
                                  size_t requests, size_t list, size_t candidates, bool headroom) const {
  using detail::after;
  using detail::old_if_grown;
  const size_t g = static_cast<size_t>(cfg.groups);
  const bool refine = !cfg.rules.fixed;
  const bool c2 = refine && cfg.rules.counters;  // P2-C C2's outer nodes and tags
  const size_t pool = samples - s0, runs = samples_ + samples;
  const size_t flags = refine ? std::max(s0, candidates) : 0, counts = pool > 0 ? s0 + 1 : 0;
  const size_t part = mixed_ ? requests + 1 : 0;  // a round's N partition (select_) and its scan (index_alt_)
  size_t temp = std::max(P::scan_temp_bytes(s0), refine ? P::select_temp_bytes(flags) : 0);
  if (pool > 0) temp = std::max({temp, P::sort_temp_bytes(pool, detail::pool_key_bits(s0)), P::scan_temp_bytes(s0 + 1)});
  if (mixed_) temp = std::max(temp, P::scan_temp_bytes(part));
  if (pv_sz_.stage && pv_sz_.pixels > 0) temp = std::max(temp, P::scan_temp_bytes(pv_sz_.pixels + 1));  // D3a's scans
  if (pv_sz_.flips > 0) temp = std::max(temp, P::scan_temp_bytes(pv_sz_.flips + 1));
  if (pv_sz_.sort) temp = std::max(temp, P::sort_temp_bytes(2 * pv_sz_.flips, pv_node_bits(pv_sz_.slots)));  // D3b
  const size_t bytes =
      count64_.bytes() + off64_.bytes() + offsets_.bytes() + diag_.bytes() + jump_.bytes() + pv_counts_.bytes() +
      pv_bytes() +
      after(cls_, classes_ ? samples : 0) + after(outer_, c2 ? list : 0) + after(tag_, c2 ? list : 0) +
      after(cand_outer_, c2 ? candidates : 0) +
      after(off0_, nb + 1) + after(dt_, samples) +
      after(owner_, samples) + after(sig_, samples) + after(irr_, samples * g) + after(before_, refine ? pool : 0) +
      after(requests_, requests) + after(out_index_, std::max(requests, pool)) + after(sel_, flags) +
      after(select_, std::max(flags, part)) + after(list_, refine ? list : 0) +
      after(candidates_, refine ? candidates : 0) + after(counters_, refine ? 3 : 0) + after(counts_, counts) +
      after(ex_, counts) + after(keys_, pool) + after(keys_alt_, pool) + after(index_alt_, std::max(pool, part)) +
      detail::runs_grown(t_.capacity(), runs, headroom) * 8 +
      detail::runs_grown(fin_irr_.capacity(), runs * g, headroom) * 4 + std::max(temp_.bytes(), temp);
  size_t twice = 0;
  if (kept > 0) {
    twice = std::max({old_if_grown(dt_, samples), old_if_grown(owner_, samples), old_if_grown(sig_, samples),
                      old_if_grown(irr_, samples * g), old_if_grown(before_, pool),
                      old_if_grown(cls_, classes_ ? samples : 0), old_if_grown(pv_slot_, pv_sz_.rows)});
  }
  if (samples_ > 0) twice = std::max({twice, old_if_grown(t_, runs), old_if_grown(fin_irr_, runs * g)});
  return bytes + twice;
}

// The check before a step: within the budget with the runs' headroom, or else without it (the
// assembly grows the runs as the last check allowed); out.overflow otherwise.
template <Space S>
bool WindowEngine<S>::fits(const SampleConfig& cfg, size_t nb, size_t s0, size_t samples, size_t kept, size_t requests,
                           size_t list, size_t candidates, size_t budget, BatchResult& out) {
  for (const bool headroom : {true, false}) {
    out.need = footprint(cfg, nb, s0, samples, kept, requests, list, candidates, headroom);
    if (out.need <= budget) {
      runs_headroom_ = headroom;
      return true;
    }
  }
  out.overflow = true;
  return false;
}

// With per-pixel N (P2-B B2) the round's samples are partitioned stably by N class (the base class first, in
// select_), and each class is rendered in a call of its own; one sync reads the band's count.
template <Space S>
bool WindowEngine<S>::render_samples(const SampleConfig& cfg, const Input& in, size_t a0, size_t m, RenderHook& render,
                                     Stream stream, WindowStats& st, const PvRound0* pv) {
  const uint8_t* n_log2 = mixed_ ? in.n_log2 + a0 : nullptr;
  const uint32_t* order = out_index_.data();
  const auto round = static_cast<uint32_t>(st.samples_by_round.size());  // pushed after the round's render
  uint32_t* cls = classes_ ? cls_.data() : nullptr;
  size_t base = m;
  if (mixed_) {
    const auto band_log2 = static_cast<uint8_t>(std::countr_zero(cfg.band_subsamples));
    uint32_t* part = select_.ensure(m + 1);
    uint32_t* before = index_alt_.ensure(m + 1);
    P::for_each(m + 1, BandFlags{out_index_.data(), owner_.data(), n_log2, band_log2, m, part}, stream);
    P::exclusive_scan(part, before, m + 1, temp_, stream);
    uint32_t band = 0;
    P::download(&band, before + m, sizeof(band), stream);
    base = m - band;
    P::for_each(m,
                BandPartition{out_index_.data(), owner_.data(), n_log2, band_log2, before,
                              static_cast<uint32_t>(base), part},
                stream);
    order = part;
  }
  // P2-D D3a: round 0's call at pv.n writes P&V's outputs, their capacity its requests (checked before any call).
  const uint32_t pv_n = pv != nullptr ? cfg.pv.n : 0;
  if (pv_n != 0) {
    const size_t count = !mixed_ ? (cfg.subsamples == pv_n ? m : 0) : (cfg.band_subsamples == pv_n ? m - base : 0);
    pv_r0_ = count > 0;
    if (pv_r0_ && !pv_outputs(cfg, *pv, count, stream)) return false;
  }
  const auto call = [&](size_t first, size_t count, uint32_t n) {
    if (pv_r0_ && n == pv_n) {
      render.render_pv(requests_.data() + first, count, order + first, irr_.data(), sig_.data(), cls, n, pv_out_);
    } else if (pv_mid_ && n == cfg.pv.n) {  // P2-D D3b: a refinement round's midpoints, for the state check
      render.render_pv(requests_.data() + first, count, order + first, irr_.data(), sig_.data(), cls, n, pv_mid_out_);
    } else {
      render.render(requests_.data() + first, count, order + first, irr_.data(), sig_.data(), cls, n);
    }
    ++st.render_calls;
    st.call_round.push_back(round);
  };
  P::for_each(m,
              BuildRequests{order, dt_.data(), owner_.data(), off0_.data(), in.pixel + a0, cfg.cam, cfg.subsamples,
                            cfg.rules.fixed, cfg.rules.shift, requests_.data(), n_log2},
              stream);
  if (!mixed_) {
    call(0, m, cfg.subsamples);
    st.band_samples_by_round.push_back(0);
    return true;
  }
  if (base > 0) call(0, base, cfg.subsamples);
  if (m > base) call(base, m - base, cfg.band_subsamples);
  st.band_samples += m - base;
  st.band_samples_by_round.push_back(m - base);
  return true;
}

// P2-D D3a: round 0's P&V outputs of `count` requests at pv.n (render/shade/codes.h; capacity = count, so no entry
// overflows), within the batch's first footprint check with the P&V buffers' per-slot and per-pixel arrays.
template <Space S>
bool WindowEngine<S>::pv_outputs(const SampleConfig& cfg, const PvRound0& r, size_t count, Stream stream) {
  pv_sz_.slots = pv_sz_.rows = r.s0;
  pv_sz_.pixels = r.nb;
  pv_sz_.entries = count;
  pv_sz_.groups = static_cast<size_t>(cfg.groups);
  pv_sz_.code_words = render::pv_code_words(cfg.pv.n, cfg.pv.lights);
  pv_sz_.table_words = render::pv_table_words(cfg.groups, cfg.pv.lights);
  pv_sz_.stage = true;
  if (!fits(cfg, r.nb, r.s0, r.reserve, 0, r.s0 - r.nb, r.s0, 0, r.budget, *r.out)) return false;
  pv_slot_.ensure(r.s0);
  pv_codes_.ensure(count * pv_sz_.code_words);
  pv_table_.ensure(count * pv_sz_.table_words);
  P::zero(pv_count_.ensure(1), sizeof(uint32_t), stream);
  pv_out_ = render::PvOut{pv_slot_.data(), pv_codes_.data(), pv_table_.data(), pv_count_.data(),
                          static_cast<uint32_t>(count), 0};
  return true;
}

// P2-D D3b: what the P&V functors read of the batch (sampling.h's layout; stage: localize's J and thin nodes too).
template <Space S>
PvBatch WindowEngine<S>::pv_batch(const Input& in, size_t a0, const SampleConfig& cfg, bool stage) {
  const PvParams& p = cfg.pv;
  const size_t s0 = pv_sz_.slots, G = static_cast<size_t>(cfg.groups);
  const uint32_t* carried_at = in.carried_at != nullptr ? in.carried_at + a0 : nullptr;
  const size_t row0 = carried_at == nullptr ? a0 : 0;  // the batch's carried rows without carried_at
  const auto code_words = static_cast<uint32_t>(pv_sz_.code_words), table_words = static_cast<uint32_t>(pv_sz_.table_words);
  const PvNodes nodes{pv_slot_.data(),
                      pv_codes_.data(),
                      pv_table_.data(),
                      in.pv_codes + row0 * code_words,
                      in.pv_table + row0 * table_words,
                      carried_at,
                      code_words,
                      table_words,
                      render::pv_row_words(cfg.groups, p.lights),
                      render::pv_code_bits(p.lights),
                      cfg.groups,
                      p.lights};
  const PvPixelData px{in.pixel + a0, in.level + a0, in.pv_plan, in.pixel_tile, in.tile_vhat};
  return PvBatch{cfg.rules,      p,
                 nodes,          px,
                 cfg.cam,        off0_.data(),
                 owner_.data(),  in.n_log2 != nullptr ? in.n_log2 + a0 : nullptr,
                 cfg.subsamples, dt_.data(),
                 irr_.data(),    sig_.data(),
                 cls_.data(),    cfg.groups,
                 pv_route_.ensure(s0), stage ? pv_j_.ensure(s0 * G) : nullptr,
                 stage ? pv_thin_.ensure(s0) : nullptr};
}

// P2-D D3a: P&V in shadow mode on round 0's lattice (localize.h's sequence), its counters into the batch's; outputs
// untouched. One sync for the flip count, one for the groups' count (D3b), one per probe round.
template <Space S>
bool WindowEngine<S>::localize(const Input& in, size_t a0, const SampleConfig& cfg, RenderHook& render,
                               const PvRound0& r, Stream stream, WindowStats& st) {
  const size_t nb = r.nb, s0 = r.s0;
  unsigned long long* counts = pv_counts_.data() + kPvWords;
  const PvBatch b = pv_batch(in, a0, cfg, true);
  uint32_t* slot_flips = pv_flipc_.ensure(s0);
  uint32_t* slot_stats = pv_stats_.ensure(4 * s0);
  P::zero(slot_flips, s0 * sizeof(uint32_t), stream);
  P::zero(slot_stats, 4 * s0 * sizeof(uint32_t), stream);
  P::for_each(s0, PvSlots{b}, stream);
  P::for_each(s0, PvThin{b}, stream);
  uint32_t* pixel_flips = pv_off_.ensure(2 * (nb + 1));
  uint32_t* off = pixel_flips + nb + 1;
  P::for_each(nb + 1, PvPixel{b, pixel_flips, slot_flips, nb}, stream);
  P::exclusive_scan(pixel_flips, off, nb + 1, temp_, stream);
  uint32_t nf = 0;
  P::download(&nf, off + nb, sizeof(nf), stream);
  if (nf == 0) {
    pv_release_stage(stream);
    return true;
  }
  // The flips and their groups (D3b: the groups' indices, then their fits), each step within the budget.
  pv_sz_.flips = nf;
  if (!fits(cfg, nb, s0, r.reserve, 0, s0 - nb, s0, 0, r.budget, *r.out)) return false;
  PvFlip* flips = pv_flips_.ensure(nf);
  uint32_t* nreq = pv_nreq_.ensure(nf + 1);
  uint32_t* req_off = pv_reqoff_.ensure(nf + 1);
  P::for_each(nb, PvFill{b, off, slot_flips, flips}, stream);
  P::for_each(nf + 1, PvSeeds{flips, nreq, nf}, stream);
  P::exclusive_scan(nreq, req_off, nf + 1, temp_, stream);
  uint32_t ng = 0;
  P::download(&ng, req_off + nf, sizeof(ng), stream);
  P::for_each(nf, PvGroupIds{flips, req_off}, stream);
  pv_sz_.ng = ng;
  if (!fits(cfg, nb, s0, r.reserve, 0, s0 - nb, s0, 0, r.budget, *r.out)) return false;
  PvFit* fits_g = pv_fits_.ensure(ng);
  SampleRequest* req = pv_req_.ensure(2 * static_cast<size_t>(nf));
  render::ProbeOut* res = pv_probe_.ensure(2 * static_cast<size_t>(nf));
  const PvFitGroups refit{flips, b.route, fits_g, cfg.pv.stages};
  P::for_each(nf, refit, stream);
  uint64_t rounds = 0;
  for (; rounds < static_cast<uint64_t>(kPvRoundsMax); ++rounds) {
    P::for_each(nf + 1, PvEmit{flips, fits_g, b.route, nreq, nf}, stream);
    P::exclusive_scan(nreq, req_off, nf + 1, temp_, stream);
    uint32_t total = 0;
    P::download(&total, req_off + nf, sizeof(total), stream);
    if (total == 0) break;
    P::for_each(nf, PvRequests{flips, nreq, req_off, in.pixel + a0, cfg.cam, cfg.rules.shift, req}, stream);
    render.probe(req, total, res);
    P::for_each(nf, PvConsume{flips, nreq, req_off, res, b.route, counts}, stream);
    P::for_each(nf, refit, stream);
  }
  // D3b: the localized set as the rounds left it (loc), R3 and its propagation, the outcomes; then the brackets for
  // the state check and the would-be nodes, in place of the rounds' requests and results.
  P::for_each(nf, refit, stream);
  P::for_each(nf, PvR3{b, flips, off}, stream);
  P::for_each(nf, PvR3Prop{flips, off, b.off0, b.route}, stream);
  P::for_each(nf, PvTally{flips, slot_stats, counts}, stream);
  st.pv[kPvRounds] += rounds;
  pv_req_.reset();
  pv_probe_.reset();
  pv_fits_.reset();
  pv_sz_.ng = 0;
  pv_sz_.sort = true;
  pv_sz_.brackets = nf;
  if (!fits(cfg, nb, s0, r.reserve, 0, s0 - nb, s0, 0, r.budget, *r.out)) return false;
  P::for_each(nf, PvBrackets{flips, pv_brk_.ensure(nf)}, stream);
  uint64_t* keys = pv_nkey_.ensure(4 * static_cast<size_t>(nf));
  uint32_t* vals = pv_nval_.ensure(4 * static_cast<size_t>(nf));
  const size_t nk = 2 * static_cast<size_t>(nf);
  P::for_each(nf, PvNodeKeys{flips, b.dt, keys, vals}, stream);
  const bool alt = P::sort_pairs(keys, keys + nk, vals, vals + nk, nk, pv_node_bits(s0), temp_, stream);
  P::for_each(nk, PvNodeCount{alt ? keys + nk : keys, slot_stats}, stream);
  pv_release_stage(stream);
  return true;
}

// P2-D D3a: once localize ends, what only it reads is freed (per slot J, the thin nodes and flip counts, the flips
// with their fits, requests and results), so that the refinement rounds' checks no longer charge it (plan §3.10: flips
// and probes for the stage in flight only); the routes, the intervals' stats and round 0's outputs stay for PvCount
// and PvCarry at the assembly. With keep_pv_flips the flips are downloaded first. D3b: the brackets and the pixels'
// flip offsets stay for the state check (pv_release_check frees them after the rounds).
template <Space S>
void WindowEngine<S>::pv_release_stage(Stream stream) {
  if (pv_keep_flips_) {
    pv_flips_host_.resize(pv_sz_.flips);
    if (pv_sz_.flips > 0) P::download(pv_flips_host_.data(), pv_flips_.data(), pv_sz_.flips * sizeof(PvFlip), stream);
  }
  for (auto* b : {&pv_flipc_, &pv_nreq_, &pv_reqoff_, &pv_nval_}) b->reset();
  if (pv_sz_.brackets == 0) pv_off_.reset();
  pv_j_.reset();
  pv_thin_.reset();
  pv_flips_.reset();
  pv_fits_.reset();
  pv_req_.reset();
  pv_probe_.reset();
  pv_nkey_.reset();
  pv_sz_.stage = pv_sz_.sort = false;
  pv_sz_.flips = pv_sz_.ng = 0;
}

// P2-D D3b: the state check of a refinement round's m samples from `first`, which its band call rendered with their
// outputs (pv_mid_out_), against the brackets (PvCheck; counters into the batch's).
template <Space S>
void WindowEngine<S>::pv_check(const Input& in, size_t a0, const SampleConfig& cfg, uint32_t first, size_t m,
                               Stream stream) {
  const PvBatch b = pv_batch(in, a0, cfg, false);
  const size_t nb = pv_sz_.pixels;
  P::for_each(m,
              PvCheck{b, first, static_cast<uint32_t>(pv_sz_.slots), before_.data(), pv_brk_.data(),
                      pv_off_.data() + nb + 1, pv_slot_.data(), pv_mcodes_.data(), pv_mtable_.data(),
                      pv_counts_.data() + kPvWords},
              stream);
}

// P2-D D3b: after the refinement rounds, the state check's brackets, offsets and midpoint outputs are freed.
template <Space S>
void WindowEngine<S>::pv_release_check() {
  pv_brk_.reset();
  pv_off_.reset();
  pv_mcodes_.reset();
  pv_mtable_.reset();
  pv_mcount_.reset();
  pv_sz_.brackets = pv_sz_.mid = 0;
  pv_sz_.rows = pv_sz_.slots;
}

template <Space S>
typename WindowEngine<S>::BatchResult WindowEngine<S>::run_batch(const Input& in, size_t a0, size_t a1,
                                                                 const SampleConfig& cfg, RenderHook& render,
                                                                 size_t budget, Stream stream) {
  check_cancel(cfg.cancel, "window: batch");  // BL-0b: at each batch start
  BatchResult out;
  const SampleRules& rules = cfg.rules;
  const bool refine = !rules.fixed;
  const int32_t groups = cfg.groups;
  const size_t nb = a1 - a0;
  const uint64_t base0 = off64_host_.empty() ? 0 : off64_host_[a0];
  const uint64_t s0_64 = (off64_host_.empty() ? s0_total_ : off64_host_[a1]) - base0;
  const auto overflow = [&out](size_t need) {
    out.overflow = true;
    out.need = need;
  };
  // Indices are 32-bit within a batch. The lattice reserves room for as many refinement samples as
  // lattice samples (§1.2's factor 2); the fixed-rate grid is never refined.
  const uint64_t reserve = refine ? 2 * s0_64 : s0_64;
  pv_sz_ = PvSizes{};  // P2-D D3a: this batch's P&V sizes (none until round 0 renders)
  pv_r0_ = false;
  if (reserve >= (uint64_t{1} << 32)) {
    overflow(SIZE_MAX);
    return out;
  }
  if (!fits(cfg, nb, s0_64, reserve, 0, s0_64 - nb, s0_64, 0, budget, out)) return out;
  const auto s0 = static_cast<uint32_t>(s0_64);
  float* carried_irr = in.carried_irr;
  uint64_t* carried_sig = in.carried_sig;
  uint32_t* carried_class = in.carried_class;
  const uint32_t* carried_at = in.carried_at != nullptr ? in.carried_at + a0 : nullptr;
  if (carried_at == nullptr) {
    carried_irr += a0 * static_cast<size_t>(groups);
    carried_sig += a0;
    if (classes_) carried_class += a0;
  }
  // The P2-0b counters of this batch (sampling.h kDiag*), added to the window's once it succeeds.
  uint32_t* cls = nullptr;
  unsigned long long* diag = nullptr;
  uint8_t* edge = classes_ && in.edge != nullptr ? in.edge + a0 : nullptr;  // P2-B B4a
  if (classes_) {
    cls = cls_.ensure(reserve);
    diag = diag_.data() + kDiagWords;
    P::zero(diag, kDiagWords * sizeof(unsigned long long), stream);
  }
  // P2-C C2's counters of this batch (sampling.h kJump*, kSib*), added to the window's once it succeeds.
  unsigned long long* jump = nullptr;
  if (refine && rules.counters) {
    jump = jump_.data() + kJumpWords;
    P::zero(jump, kJumpWords * sizeof(unsigned long long), stream);
  }
  // P2-D D3a: P&V's shadow mode (SampleConfig::pv): its buffers' sizes for this batch, its counters (added to the
  // window's once it succeeds).
  const bool pv = refine && cfg.pv.n != 0;
  if (pv) P::zero(pv_counts_.data() + kPvWords, kPvWords * sizeof(unsigned long long), stream);

  // K5-K7, R0: the lattice (or grid) and its render.
  off0_.ensure(nb + 1);
  P::for_each(nb + 1, LocalOffsets{off64_.data() + a0, base0, off0_.data()}, stream);
  dt_.ensure(reserve);
  owner_.ensure(reserve);
  sig_.ensure(reserve);
  irr_.ensure(reserve * groups);
  if (refine) before_.ensure(reserve - s0);
  size_t requests = s0 - nb;
  out_index_.ensure(requests);
  requests_.ensure(requests);
  P::zero(dt_.data(), static_cast<size_t>(s0) * sizeof(uint32_t), stream);  // the marks, then the times
  P::for_each(nb - 1, MarkPixelEnds{off0_.data(), dt_.data()}, stream);
  P::exclusive_scan(dt_.data(), owner_.data(), s0, temp_, stream);
  P::for_each(s0,
              LatticeSlots{in.t_w, rules, off0_.data(), owner_.data(), in.level + (in.level ? a0 : 0), carried_irr,
                           carried_sig, carried_at, groups, dt_.data(), irr_.data(), sig_.data(), out_index_.data(),
                           carried_class, cls},
              stream);
  WindowStats& st = out.stats;
  const PvRound0 pv_r0{nb, s0, static_cast<size_t>(reserve), budget, &out};
  if (!render_samples(cfg, in, a0, requests, render, stream, st, pv ? &pv_r0 : nullptr)) return out;
  st.samples_by_round.push_back(requests);
  uint32_t samples = s0;

  // K8-K12: refinement rounds. Round k's flags (counted on arrival; the halves of round k lie k + 1 levels below
  // the lattice) select the intervals to split; after max_refine_levels rounds the remaining flags are cap hits.
  // One sync per round reads the select's count and the flag counters together.
  if (refine) {
    unsigned long long* words = words_.data();
    counters_.ensure(3);
    sel_.ensure(s0);
    select_.ensure(s0);
    P::zero(counters_.data(), 3 * sizeof(unsigned long long), stream);
    const Round0Flags flags0{rules,      off0_.data(),     owner_.data(), in.pixel + a0, dt_.data(), irr_.data(),
                             sig_.data(), groups,          sel_.data(),   counters_.data(), cls,      diag,
                             edge,        jump};
    if (jump != nullptr) {
      P::for_each(s0, Counting<Round0Flags>{flags0}, stream);  // P2-C C2's counters compiled in
    } else {
      P::for_each(s0, flags0, stream);
    }
    P::download_async(words + 1, counters_.data(), 3 * sizeof(unsigned long long), stream);
    P::select_flagged(sel_.data(), select_.data(), s0, words, temp_, stream);
    P::sync(stream);
    size_t listed = words[0];
    list_.ensure(listed);
    if (jump != nullptr) {
      P::for_each(listed,
                  Round0List{select_.data(), list_.data(), off0_.data(), owner_.data(), sel_.data(),
                             outer_.ensure(listed), tag_.ensure(listed)},
                  stream);
    } else {
      P::for_each(listed, Round0List{select_.data(), list_.data()}, stream);
    }
    // P2-D D3a: P&V beside bisection on round 0's intervals (counters only; the rounds below are as without it).
    if (pv_r0_ && !localize(in, a0, cfg, render, pv_r0, stream, st)) return out;
    for (int32_t k = 0;; ++k) {
      st.flagged_signature += words[1];
      st.flagged_midpoint += words[2];
      if (k == rules.max_refine_levels) {
        st.refine_cap_hits += words[3] + listed;
        break;
      }
      st.refine_cap_hits += words[3];
      if (listed == 0) break;
      check_cancel(cfg.cancel, "window: refinement round");  // BL-0b: at each refinement round
      const size_t m = listed, next = samples + m;
      if (next >= (uint64_t{1} << 32)) {
        overflow(SIZE_MAX);
        return out;
      }
      // P2-D D3b: where P&V found flips, the round's band call writes its midpoints' outputs for the state check.
      const bool check = pv_sz_.brackets > 0;
      if (check) {
        pv_sz_.rows = next;
        pv_sz_.mid = m;
      }
      if (!fits(cfg, nb, s0, next, samples, std::max(requests, m), 2 * m, 2 * m, budget, out)) return out;
      detail::grow_keep(dt_, samples, next, stream);
      detail::grow_keep(owner_, samples, next, stream);
      detail::grow_keep(sig_, samples, next, stream);
      detail::grow_keep(irr_, static_cast<size_t>(samples) * groups, next * groups, stream);
      detail::grow_keep(before_, samples - s0, next - s0, stream);
      if (classes_) {
        detail::grow_keep(cls_, samples, next, stream);
        cls = cls_.data();
      }
      requests = std::max(requests, m);
      out_index_.ensure(requests);
      requests_.ensure(requests);
      P::for_each(m, Midpoints{list_.data(), samples, s0, dt_.data(), owner_.data(), before_.data(), out_index_.data()},
                  stream);
      if (check) {
        detail::grow_keep(pv_slot_, s0, next, stream);  // the lattice's entries kept
        uint32_t* count = pv_mcount_.ensure(1);
        P::zero(count, sizeof(uint32_t), stream);
        pv_mid_out_ = render::PvOut{pv_slot_.data(), pv_mcodes_.ensure(m * pv_sz_.code_words),
                                    pv_mtable_.ensure(m * pv_sz_.table_words), count, static_cast<uint32_t>(m), 0};
        pv_mid_ = true;
      }
      render_samples(cfg, in, a0, m, render, stream, st);
      if (check) {
        pv_mid_ = false;
        pv_check(in, a0, cfg, samples, m, stream);
      }
      candidates_.ensure(2 * m);
      if (jump != nullptr) cand_outer_.ensure(2 * m);
      sel_.ensure(std::max<size_t>(s0, 2 * m));
      select_.ensure(std::max<size_t>(s0, 2 * m));
      P::zero(counters_.data(), 3 * sizeof(unsigned long long), stream);
      const HalfFlags halves{rules,       list_.data(), samples,           k + 1,        owner_.data(),
                             in.pixel + a0, dt_.data(),   irr_.data(),       sig_.data(),  groups,
                             candidates_.data(), sel_.data(), counters_.data(), cls,      diag,
                             edge,        jump,         outer_.data(),     tag_.data(),  cand_outer_.data()};
      if (jump != nullptr) {
        P::for_each(m, Counting<HalfFlags>{halves}, stream);
      } else {
        P::for_each(m, halves, stream);
      }
      P::download_async(words + 1, counters_.data(), 3 * sizeof(unsigned long long), stream);
      P::select_flagged(sel_.data(), select_.data(), 2 * m, words, temp_, stream);
      P::sync(stream);
      listed = words[0];
      list_.ensure(listed);
      if (jump != nullptr) {
        P::for_each(listed,
                    GatherList{select_.data(), candidates_.data(), list_.data(), sel_.data(), cand_outer_.data(),
                               outer_.ensure(listed), tag_.ensure(listed)},
                    stream);
      } else {
        P::for_each(listed, GatherList{select_.data(), candidates_.data(), list_.data()}, stream);
      }
      samples = static_cast<uint32_t>(next);
      st.samples_by_round.push_back(m);
    }
    if (pv_sz_.brackets > 0) pv_release_check();  // P2-D D3b
  }

  // K13: assembly into the window's runs (within the last check's footprint), and each pixel's last
  // sample carried over. Without refinement samples every slot keeps its place (ex null).
  const size_t pool = samples - s0;
  const size_t runs = samples_ + samples;
  detail::grow_keep(t_, samples_, detail::runs_grown(t_.capacity(), runs, runs_headroom_), stream);
  detail::grow_keep(fin_irr_, samples_ * groups, detail::runs_grown(fin_irr_.capacity(), runs * groups, runs_headroom_),
                    stream);
  const uint32_t* ex = nullptr;
  const uint32_t* sorted = nullptr;
  if (pool > 0) {
    counts_.ensure(s0 + 1);
    ex_.ensure(s0 + 1);
    keys_.ensure(pool);
    keys_alt_.ensure(pool);
    out_index_.ensure(pool);
    index_alt_.ensure(pool);
    P::zero(counts_.data(), (s0 + 1) * sizeof(uint32_t), stream);
    P::for_each(pool, PoolKeys{s0, dt_.data(), before_.data(), keys_.data(), out_index_.data(), counts_.data()}, stream);
    sorted = P::sort_pairs(keys_.data(), keys_alt_.data(), out_index_.data(), index_alt_.data(), pool,
                           detail::pool_key_bits(s0), temp_, stream)
                 ? index_alt_.data()
                 : out_index_.data();
    P::exclusive_scan(counts_.data(), ex_.data(), s0 + 1, temp_, stream);
    ex = ex_.data();
  }
  int64_t* t_out = t_.data() + samples_;
  float* irr_out = fin_irr_.data() + samples_ * groups;
  P::for_each(s0, ScatterSlots{in.t_w, ex, dt_.data(), irr_.data(), groups, t_out, irr_out}, stream);
  P::for_each(pool, ScatterPool{in.t_w, s0, sorted, before_.data(), dt_.data(), irr_.data(), groups, t_out, irr_out},
              stream);
  int64_t* const offsets = in.append_offsets != nullptr ? in.append_offsets : offsets_.data();
  P::for_each(nb,
              PixelRuns{static_cast<int64_t>(samples_), off0_.data(), ex, irr_.data(), sig_.data(), groups, carried_at,
                        carried_irr, carried_sig, offsets + a0, cls, carried_class},
              stream);
  if (classes_) P::for_each(kDiagWords, detail::AddWords{diag, diag_.data()}, stream);
  if (jump != nullptr) P::for_each(kJumpWords, detail::AddWords{jump, jump_.data()}, stream);
  if (pv) {  // P2-D D3a: the intervals' classes with their refinement cost, the carried codes, the batch's counters
    const uint8_t* n_log2 = in.n_log2 != nullptr ? in.n_log2 + a0 : nullptr;
    unsigned long long* counts = pv_counts_.data() + kPvWords;
    if (pv_r0_) {
      P::for_each(s0,
                  PvCount{pv_route_.data(), pv_stats_.data(), pool > 0 ? counts_.data() : nullptr, owner_.data(),
                          off0_.data(), n_log2, cfg.subsamples, cfg.pv.n, counts},
                  stream);
    }
    const auto cw = render::pv_code_words(cfg.pv.n, cfg.pv.lights), tw = render::pv_table_words(cfg.groups, cfg.pv.lights);
    const size_t row0 = carried_at == nullptr ? a0 : 0;
    P::for_each(nb,
                PvCarry{off0_.data(), carried_at, pv_r0_ ? pv_slot_.data() : nullptr, pv_codes_.data(),
                        pv_table_.data(), pv_r0_ ? n_log2 : nullptr, pv_r0_ ? cfg.subsamples : 0u, cfg.pv.n, cw, tw,
                        in.pv_codes + row0 * cw, in.pv_table + row0 * tw},
                stream);
    P::for_each(kPvWords, detail::AddWords{counts, pv_counts_.data()}, stream);
  }
  samples_ += samples;
  return out;
}

template <Space S>
WindowStats WindowEngine<S>::run(const Input& in, const SampleConfig& cfg, RenderHook& render, size_t budget,
                                 Stream stream) {
  using detail::extra;
  using detail::mib;
  WindowStats total;
  total.pixels = in.n;
  // An appended pass (P2-B B4b) keeps the window's runs: they are its to grow, as a batched window's reservation.
  const bool append = in.append_offsets != nullptr;
  if (!append) samples_ = 0;
  runs_reserved_ = append;
  off64_host_.clear();
  classes_ = in.carried_class != nullptr;
  mixed_ = in.n_log2 != nullptr;
  const auto pow2 = [](uint32_t v) { return v != 0 && (v & (v - 1)) == 0; };
  if (mixed_ && !(pow2(cfg.subsamples) && pow2(cfg.band_subsamples) && cfg.band_subsamples != cfg.subsamples)) {
    throw std::invalid_argument("window: per-pixel N needs two distinct powers of two, subsamples and band_subsamples");
  }
  if (classes_) {
    diag_.ensure(2 * kDiagWords);
    P::zero(diag_.data(), kDiagWords * sizeof(unsigned long long), stream);
  }
  total.jump_on = cfg.rules.counters && !cfg.rules.fixed;
  if (total.jump_on) {
    jump_.ensure(2 * kJumpWords);
    P::zero(jump_.data(), kJumpWords * sizeof(unsigned long long), stream);
  }
  total.pv_on = cfg.pv.n != 0 && !cfg.rules.fixed;  // P2-D D3a
  uint64_t pv_rounds = 0;
  if (total.pv_on) {
    if (!classes_ || in.pv_codes == nullptr || in.pv_table == nullptr || cfg.pv.lights < 0 ||
        cfg.pv.lights > render::kPvMaxLights) {
      throw std::invalid_argument("window: P&V (SampleConfig::pv) needs the carried class words, codes and tables, and "
                                  "at most " + std::to_string(render::kPvMaxLights) + " shadow-casting lights");
    }
    pv_counts_.ensure(2 * kPvWords);
    P::zero(pv_counts_.data(), kPvWords * sizeof(unsigned long long), stream);
  }
  // K4's offsets come first; held capacity is released when they would not fit next to it.
  const auto k4 = [&] {
    const size_t n1 = in.n + 1;
    size_t need = held_bytes() + (append ? 0 : extra(offsets_, n1));
    if (in.n > 0) {
      const size_t temp = P::scan_temp_bytes(n1);
      need += extra(count64_, n1) + extra(off64_, n1) + (temp > temp_.bytes() ? temp - temp_.bytes() : 0);
    }
    return need;
  };
  if (k4() > budget) release_batches();
  if (const size_t need = k4(); need > budget) {
    throw MemoryBudgetError("window at " + std::to_string(in.t_w) + " ns: the offsets of its " + std::to_string(in.n) +
                            " active pixels need " + mib(need) + " MiB, above the device budget of " + mib(budget) +
                            " MiB (execution.device_budget_mb)");
  }
  if (!append) offsets_.ensure(in.n + 1);
  if (in.n > 0) {
    // K4: lattice offsets of the whole window, 64-bit before batching.
    count64_.ensure(in.n + 1);
    off64_.ensure(in.n + 1);
    P::for_each(in.n + 1, LatticeCounts{in.level, cfg.rules.fixed, cfg.rules.m, in.n, count64_.data()}, stream);
    P::exclusive_scan(count64_.data(), off64_.data(), in.n + 1, temp_, stream);
    P::download(&s0_total_, off64_.data() + in.n, sizeof(uint64_t), stream);
    // Batches run in active-pixel order (the runs are appended). A batch that overflows the budget is
    // retried once without the capacity that earlier batches or windows left, then redone as two
    // halves split at its middle pixel.
    const std::string where = "window at " + std::to_string(in.t_w) + " ns: ";
    const std::string limit = ", above the device budget of " + mib(budget) + " MiB (execution.device_budget_mb)";
    std::vector<std::pair<size_t, size_t>> todo{{0, in.n}};
    while (!todo.empty()) {
      const auto [a0, a1] = todo.back();
      const bool fresh = batch_bytes() == 0 && (runs_reserved_ || t_.capacity() <= samples_);
      BatchResult r = run_batch(in, a0, a1, cfg, render, budget, stream);
      if (r.overflow) {
        // The calls it made are timed (render_ms_by_call) and redone: one kRedoneCall entry each.
        total.call_round.insert(total.call_round.end(), r.stats.call_round.size(), kRedoneCall);
        if (!fresh) {
          release_batches();
          continue;
        }
        if (a1 - a0 == 1) {
          throw MemoryBudgetError(
              where + "active pixel " + std::to_string(a0) + " alone needs " +
              (r.need == SIZE_MAX ? std::string("2^32 samples or more") : mib(r.need) + " MiB") +
              " with the window's runs (" + mib(t_.bytes() + fin_irr_.bytes()) + " MiB of it)" + limit);
        }
        if (off64_host_.empty()) {
          // Batching starts: sub-batch offsets come from a host copy, and the window's runs are
          // reserved once, at the lattice's projection (§1.2's factor 2) or else at the lattice, so
          // that the batches append to them without growth copies (an appended pass: beyond the runs
          // held, which are copied once if the runs grow).
          off64_host_.resize(in.n + 1);
          P::download(off64_host_.data(), off64_.data(), (in.n + 1) * sizeof(uint64_t), stream);
          const size_t g = static_cast<size_t>(cfg.groups), per = 8 + 4 * g;
          // What holding the runs at `want` more samples takes: the window's first pass counts them anew; an
          // appended one the runs grown beside the held ones (grow_keep copies them).
          const auto need = [&](size_t want) {
            if (samples_ == 0) return held_bytes() + want * per;
            const size_t runs = samples_ + want;
            return held_bytes() + (runs > t_.capacity() ? runs * 8 : 0) +
                   (runs * g > fin_irr_.capacity() ? runs * g * 4 : 0);
          };
          size_t want = (cfg.rules.fixed ? 1 : 2) * s0_total_;
          if (need(want) > budget) want = s0_total_;
          if (need(want) > budget) {
            throw MemoryBudgetError(where + "the runs of its " + std::to_string(s0_total_) + " lattice samples need " +
                                    mib(need(want)) + " MiB with the pixel offsets" + limit);
          }
          detail::grow_keep(t_, samples_, samples_ + want, stream);
          detail::grow_keep(fin_irr_, samples_ * g, (samples_ + want) * g, stream);
          runs_reserved_ = true;
        }
        const size_t m = a0 + (a1 - a0) / 2;
        todo.back() = {m, a1};
        todo.push_back({a0, m});
        continue;
      }
      todo.pop_back();
      ++total.batches;
      const WindowStats& b = r.stats;
      if (total.samples_by_round.size() < b.samples_by_round.size()) total.samples_by_round.resize(b.samples_by_round.size());
      for (size_t k = 0; k < b.samples_by_round.size(); ++k) total.samples_by_round[k] += b.samples_by_round[k];
      if (total.band_samples_by_round.size() < b.band_samples_by_round.size()) {
        total.band_samples_by_round.resize(b.band_samples_by_round.size());
      }
      for (size_t k = 0; k < b.band_samples_by_round.size(); ++k) {
        total.band_samples_by_round[k] += b.band_samples_by_round[k];
      }
      total.call_round.insert(total.call_round.end(), b.call_round.begin(), b.call_round.end());
      total.flagged_signature += b.flagged_signature;
      total.flagged_midpoint += b.flagged_midpoint;
      total.refine_cap_hits += b.refine_cap_hits;
      total.render_calls += b.render_calls;
      total.band_samples += b.band_samples;
      pv_rounds += b.pv[kPvRounds];
    }
  } else {
    total.samples_by_round = {0};
    total.band_samples_by_round = {0};
  }
  const auto end = static_cast<int64_t>(samples_);
  P::upload((append ? in.append_offsets : offsets_.data()) + in.n, &end, sizeof(end), stream);
  if (classes_) {
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t));
    P::download(total.diag.data(), diag_.data(), kDiagWords * sizeof(uint64_t), stream);
  }
  if (total.jump_on) P::download(total.jump.data(), jump_.data(), kJumpWords * sizeof(uint64_t), stream);
  if (total.pv_on) {
    P::download(total.pv.data(), pv_counts_.data(), kPvWords * sizeof(uint64_t), stream);
    total.pv[kPvRounds] = pv_rounds;  // host-side
  }
  P::sync(stream);
  return total;
}

}  // namespace balla::engine
