// Request and result records of the renderer (contract C2, DESIGN §4.3).
//
// SampleRequest (24 B) and HitRecord (16 B) are the fixed records of backend/api.h. This header
// completes the types that api.h only declared: ground-truth requests and hits, and the limits
// that size per-pixel-sample state.
#pragma once

#include <stdint.h>

#include "backend/api.h"

namespace balla::render {

using backend::HitRecord;
using backend::kInstanceNone;
using backend::RequestKind;
using backend::SampleRequest;

// Shadow visibility of one sub-sample is a bit mask over the shadow-casting point, spot and
// directional lights, so at most 32 of them per scene.
inline constexpr int kMaxShadowLights = 32;
// Light groups per scene (DESIGN §7.5: G <= 8 in v1).
inline constexpr int kMaxLightGroups = 8;
// Sub-samples per pixel-sample: the cluster loop tracks membership in a kMaxSubsamples-bit mask, the device kernel in
// one 32-bit slot mask per lane (32 slots x 32 lanes). Phase 1 uses N = 64; the nested Owen-Sobol prefixes 16, 64, 256
// and 1024 all fit (DESIGN §6.6; 1024 for reference-mode crops, PHASE2 P2-0a; 4096 needs wider slot masks).
inline constexpr int kMaxSubsamples = 1024;

// Area lights (P2-F F8a-1, plan p2f §4.6): the fixed point set Y_l of a light has kLightPoints = 1024 points, one in
// each cell of the 32 x 32 grid over the unit square (the first 1024 points of an Owen-scrambled 2-D Sobol sequence
// form a (0, 10, 2)-net), so min(N, M) = N for every N <= kMaxSubsamples. The pair of sub-sample k picks the point of
// its cell. Maths-bounded (a net needs a power of two, 32 x 32 its balanced cells), not a knob.
inline constexpr int kLightGrid = 32, kLightPoints = kLightGrid * kLightGrid;
static_assert(kLightPoints >= kMaxSubsamples);

// The class word of a pixel-sample (render_into's sample_class; shade.h sample_class), for the engine's P2-0b counters
// (ENGINE_CPP §0.1) and, with the band N, its carried seeds (P2-B B4a) and its escalations (B4b: an N_base pixel with
// edge bits re-renders its window at N_band; engine/sampling.h kBandEdge). Outputs depend on it only there, and T2
// makes the band-on outputs those of the uniform N_band run wherever its edge bits see the edges (over-escalation, as
// from the shadow bits' fold beyond 16 lights, costs time only): bits 31-16 a geometry hash (equal surface codes of the
// sub-samples give equal hashes; a change is missed with probability 2^-16); bit 15 two or more clusters (a silhouette
// or material boundary in the support); bit 14 a shadow light whose bit differs between hit sub-samples (a shadow
// boundary; shadow bits folded mod 16); bit 13 a textured hit; bit 12 a specular hit (P2-F F6, p2f.md §4.8: a hit
// material with a lobe and alpha <= 0.5; counters only, and the exclusion of relaxed caps); bits 11-0 the shadowed (hit
// sub-sample, light) pairs mod 2^12 (13 bits before F6: the counters' signed difference is now 12-bit, so a terminal
// interval's net change reads right up to 2047 pairs instead of 4095).
inline constexpr uint32_t kClassMulti = 1u << 15, kClassShadowEdge = 1u << 14, kClassTextured = 1u << 13,
                          kClassSpecular = 1u << 12, kClassShadowed = (1u << 12) - 1;

// The P2-B B0 census's counters (render/census.h; RenderStats::census), per class of the pixel-sample's class word: the
// band proxy (the shadow-edge or the multi-cluster bit), then the rest; each the sub-samples k >= 16, those B1's reuse
// rule would take from the probes, and those of them whose traced hit differs from the probes'.
inline constexpr int kCensusBand = 0, kCensusRest = 3, kCensusSubsamples = 0, kCensusReusable = 1,
                     kCensusMismatched = 2, kCensusWords = 6;

// Centre ray of a pixel position at one pose slot (DESIGN §9.1): ground truth and the dense API.
// x, y are distorted pixel coordinates with pixel centres at integers.
struct GtRequest {
  uint16_t cam;
  uint16_t reserved;
  float x, y;
  uint32_t pose_slot;
};
static_assert(sizeof(GtRequest) == 16);

// A ground-truth query of Renderer::gt_into: a pixel position (as GtRequest) at its own time.
struct GtQuery {
  int64_t t_ns;
  float x, y;
};
static_assert(sizeof(GtQuery) == 16);

// Closest hit of a centre ray; instance == kInstanceNone on a miss.
struct GtHit {
  uint32_t instance;
  uint32_t prim;
  float t_hit;
  uint32_t reserved;
};
static_assert(sizeof(GtHit) == 16);

// Ground truth at a pixel position (DESIGN §9.1, §9.3): instance and semantic of the centre-ray
// hit, camera-frame depth z (m) and the instantaneous motion field (px/s, raw distorted
// coordinates, M2 D8). Misses have instance 0, semantic 0 and NaN depth and flow.
struct GtSample {
  uint32_t instance;
  uint32_t semantic;
  float depth;
  float flow_x, flow_y;
};

}  // namespace balla::render
