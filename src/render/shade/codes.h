// P2-D D2 (plan ballasim_out/p2_plans/p2d.md §3.1 Step A, §3.6, §3.10): what predict-and-verify (P&V) reads from a
// render without tracing again. render_into's optional PvOut receives, per pixel-sample, a cluster table and a compact
// code per sub-sample, written by the shade kernels from the sums they already hold (null keeps the shading path as
// it was); Renderer::probe_into's ProbeOut is the state of one sub-sample at any time (render/shade/probe.h).
//
// Which pixel-samples write: those whose class word (shade.h sample_class) has an edge bit (kClassMulti or
// kClassShadowEdge, engine/sampling.h kBandEdge), or every one under PvOut::all (tests). A sample without edge bits has
// one cluster and one shadow mask: its state follows from its class word and pixel value.
//
// The table (pv_table_words(G, L) words per entry): a header word (bits 0-2 the rows used, 0-4; kPvTableOverflow when
// a fifth non-environment cluster exists), then kPvRows rows of pv_row_words(G, L) words, one per non-environment
// cluster in the shading's cluster order (by first sub-sample), rows past the header's count unwritten: instance,
// surface (the cluster key, geometry_term's input: the material, or the patch under accuracy.shading_patches crease,
// P2-F F5), n_s (its sub-samples), L_s,g (G floats, cluster_radiance's radiance), then per shadow light l:
// a_s,l = (n_s / N) rho_s e_l / pi (e_l: lambert_irradiance at the cluster's mean point and normal, so a_s,l R_s,l is
// the light's share of the pixel value), F_s,l = sum_k f_l(k) and
// R_s,l = sum_k f_l(k) V_k / F_s,l (0 where F_s,l is 0). Floats are stored as their bits. The environment (misses and
// their background) has no row.
//
// The code of sub-sample k (b = pv_code_bits(L) bits at bit k b of the entry's little-endian words): bits 0-2 its row
// (0-3 a table row, kPvCodeEnv the environment, kPvCodeOverflow a fifth or later cluster), bits 3.. its shadow mask
// as the signature takes it (accuracy.unlit_mask applied), so that sum_k signature_term(k, key(row), mask) is the
// sample's signature. b is 4 for L <= 1, 8 for L <= 5, 16 for L <= kPvMaxLights (render_into refuses more lights).
// The mask field holds the shadow lights' bits only: under accuracy.mover_ao traced (P2-F F9b) the occlusion word's AO
// bit (render/shade/ao_traced.h kAoTracedBit, 31) enters the signature but has no field, so render and render_into
// refuse P&V outputs there, execution.pv_outputs writes none, and probe_into and probe refuse probes.
//
// Entries are claimed through an atomic counter: their order is not deterministic, their contents are (host = device
// bit for bit on identical hit records). The caller reads them through PvOut::slot, per output row.
#pragma once

#include <stdint.h>

#if !defined(__CUDA_ARCH__)
#include <atomic>
#endif

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "render/scene_view.h"
#include "render/shade/lights.h"
#include "render/shade/nir.h"
#include "render/types.h"

namespace balla::render {

inline constexpr int32_t kPvMaxLights = 8;  // P&V's eligibility (plan §3.1): the code's mask field
inline constexpr uint32_t kPvRows = 4;      // C, the table's rows: a fifth cluster routes its intervals (R0)
inline constexpr uint32_t kPvNone = 0xFFFFFFFFu, kPvOverflow = 0xFFFFFFFEu;  // PvOut::slot without an entry
inline constexpr uint32_t kPvCodeEnv = 6, kPvCodeOverflow = 7;               // a code's row field
inline constexpr uint32_t kPvTableOverflow = 1u << 3;                        // the table header's fifth-cluster bit
// A probe's shadow light (ProbeOut::f) in SampleRequest::flags above the request kind (backend/api.h).
inline constexpr uint32_t kProbeLightShift = 3;

BALLA_HD inline uint32_t pv_code_bits(int32_t lights) { return lights <= 1 ? 4u : lights <= 5 ? 8u : 16u; }
BALLA_HD inline uint32_t pv_code_words(uint32_t n, int32_t lights) { return (n * pv_code_bits(lights) + 31u) / 32u; }
BALLA_HD inline uint32_t pv_row_words(int32_t groups, int32_t lights) {
  return 3u + static_cast<uint32_t>(groups) + 3u * static_cast<uint32_t>(lights);
}
BALLA_HD inline uint32_t pv_table_words(int32_t groups, int32_t lights) {
  return 1u + kPvRows * pv_row_words(groups, lights);
}
BALLA_HD inline uint32_t pv_code(uint32_t row, uint32_t mask) { return row | (mask << 3); }
BALLA_HD inline uint32_t pv_get_code(const uint32_t* words, uint32_t k, uint32_t bits) {
  return (words[k * bits / 32u] >> (k * bits % 32u)) & ((1u << bits) - 1u);
}

// render_into's P&V outputs, in executor memory. slot (per output row, as irradiance's; may be null) receives the
// entry index, kPvNone (no edge bit) or kPvOverflow (the count passed capacity); entry e's codes and table are
// codes[e * pv_code_words(N, L)...] and table[e * pv_table_words(G, L)...]. *count (zeroed by the caller) counts the
// entries claimed, also beyond capacity, so calls may append to one output.
struct PvOut {
  uint32_t* slot = nullptr;
  uint32_t* codes = nullptr;
  uint32_t* table = nullptr;
  uint32_t* count = nullptr;
  uint32_t capacity = 0;
  uint32_t all = 0;  // nonzero: every pixel-sample writes, not only the edge-class ones
};

// The state of one sub-sample (Renderer::probe_into; render/shade/probe.h), 16 B.
struct ProbeOut {
  uint32_t instance;  // kInstanceNone on a miss
  uint32_t surface;   // the cluster key (material, or patch under crease), geometry_term's input (0 on a miss)
  uint32_t mask;      // the shadow mask as the signature takes it (accuracy.unlit_mask applied; baked proxies: U2's word)
  float f;            // f_l: lambert_irradiance of the request's shadow light at the hit (0 on a miss)
};
static_assert(sizeof(ProbeOut) == 16);

// The entry of a pixel-sample with class word `word`: kPvNone without an edge bit (unless pv.all), kPvOverflow past
// the capacity.
BALLA_HD inline uint32_t pv_claim(const PvOut& pv, uint32_t word) {
  if (pv.all == 0 && (word & (kClassMulti | kClassShadowEdge)) == 0) return kPvNone;
#if defined(__CUDA_ARCH__)
  const uint32_t e = atomicAdd(pv.count, 1u);
#else
  const uint32_t e = std::atomic_ref<uint32_t>(*pv.count).fetch_add(1u, std::memory_order_relaxed);
#endif
  return e < pv.capacity ? e : kPvOverflow;
}

// A table row (pv_row_words(G, L) words) of one cluster from the sums cluster_radiance shades it with (Sums: shade.h
// ClusterSums; total holds the per-member terms of term_count sub-samples, count its members) and its radiance rad.
// The same arithmetic as cluster_radiance's for the mean point, normal and albedo. a_s,l is the Lambert share rho_s e_l / pi:
// a cluster with a specular lobe (P2-F F6) has a lobe term and (1 - m)(1 - E_s) rho beside it that this row does not
// carry, so P&V is not eligible in a scene with a lobe (Renderer::has_specular; the row's L_s,g is cluster_radiance's).
template <class Sums>
BALLA_HD void pv_table_row(uint32_t* row, const SceneView& s, const PoseTableView& poses, const SampleRequest& req,
                           uint32_t instance, uint32_t surface, const MaterialView& m, bool textured, uint32_t count,
                           uint32_t term_count, float inv_n, const Sums& total, const float* rad) {
  constexpr int32_t kShadow = Sums::kLights;
  const float inv_c = 1.0f / static_cast<float>(term_count);
  const Vec3f x = inv_c * total.vec3(0);
  const Vec3f n_shade = inv_c * total.vec3(3);
  // A baked cluster (scans.md §4.1) takes no light: a_s,l = 0 and F_s,l = R_s,l = 0 (its U2 factor, §4.2, is not a
  // pooled light ratio: the table does not model it).
  const bool baked = m.baked >= 0;
  const float w = static_cast<float>(count) * inv_n;
  row[0] = instance;
  row[1] = surface;
  row[2] = count;
  for (int32_t g = 0; g < s.num_groups; ++g) row[3 + g] = float_to_bits(rad[g]);
  uint32_t* l = row + 3 + s.num_groups;
  BALLA_UNROLL
  for (int32_t b = 0; b < kShadow; ++b) {
    if (b >= s.num_shadow_lights) break;
    const int32_t j = s.shadow_light[b];
    const float e = lambert_irradiance(s.lights[j], light_row(poses, req.pose_slot, j), x, n_shade);
    // A baked cluster's slot pairs hold U2's factor sums (shade.h accumulate_baked), not f and f V.
    const float f = baked ? 0.0f : total.t[Sums::kTerms + 2 * b], fv = baked ? 0.0f : total.t[Sums::kTerms + 2 * b + 1];
    // The albedo of the light's group for this camera (P2-F F11a; cluster_albedo: the visible mean without NIR); a baked
    // cluster takes no light: a_s,l = 0.
    const float rho = baked ? 0.0f : cluster_albedo(s, req.cam, m, textured, inv_c, total, s.lights[j].group);
    l[3 * b] = float_to_bits(w * rho * e * 0.318309886183790672f);
    l[3 * b + 1] = float_to_bits(f);
    l[3 * b + 2] = float_to_bits(f > 0.0f ? fv / f : 0.0f);
  }
}

// The host's write (shade.h shade_pixel_sample; the device kernel writes the same words from its lanes): claims the
// entry of output row `row` from the class word and stores the codes of `stride` sub-samples (rows: the code_row of the
// n traced ones, kPvCodeEnv or a row; mask: their masks; zeros past n) and the header with `rows` (the first kPvRows
// tables rows of `clusters` non-environment clusters).
inline void pv_store(const PvOut& pv, size_t row, uint32_t stride, uint32_t word, uint32_t n, int32_t groups,
                     int32_t lights, const uint32_t* rows, uint32_t clusters, const uint8_t* code_row,
                     const uint32_t* mask) {
  const uint32_t e = pv_claim(pv, word);
  if (pv.slot != nullptr) pv.slot[row] = e;
  if (e >= kPvOverflow) return;
  const uint32_t bits = pv_code_bits(lights), words = pv_code_words(stride, lights);
  uint32_t* codes = pv.codes + static_cast<size_t>(e) * words;
  for (uint32_t w = 0; w < words; ++w) codes[w] = 0;
  for (uint32_t k = 0; k < n && k < stride; ++k) codes[k * bits / 32u] |= pv_code(code_row[k], mask[k]) << (k * bits % 32u);
  uint32_t* table = pv.table + static_cast<size_t>(e) * pv_table_words(groups, lights);
  const uint32_t used = clusters < kPvRows ? clusters : kPvRows;
  table[0] = used | (clusters > kPvRows ? kPvTableOverflow : 0u);
  for (uint32_t i = 0; i < used * pv_row_words(groups, lights); ++i) table[1 + i] = rows[i];
}

}  // namespace balla::render
