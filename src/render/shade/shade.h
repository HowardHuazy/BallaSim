// Per-pixel-sample estimator: visibility per sub-sample, shading per cluster (DESIGN §7.3, Phase 1
// scope). The building blocks here are shared by the CPU executor (shade_pixel_sample below) and the
// warp-per-pixel-sample CUDA kernel (render/cuda/device_executor.cu); both follow the lane schedule
// of render/lanes.h, so they produce bitwise-equal results from identical hit records (DESIGN §7.9).
//
// Clusters are the sub-samples of one pixel-sample that hit the same (instance, material) (or patch: below); misses
// form the environment cluster, which contributes the scene's background radiance B_g per light group
// (D13 item 1; zero by default, the Phase 1 environment): (n_miss / N) B_g, added in cluster order like
// any other cluster, so a pixel that sees nothing returns B_g exactly. Materials are the Phase 1
// stand-in for shading patches, which DESIGN §5.2 cuts at material boundaries. Clusters are taken in
// order of their first sub-sample. Each sums its sub-samples' hit points, unnormalized shading
// normals and texture albedos, and per shadow-casting light the pooled-ratio sums of DESIGN §7.6.
// Then, once per cluster and light group g,
//
//   L_s,g = rho_s (sum_{l in g} max(0, n_s . F_l(x_s)) R_s,l / pi + A_g) + [g = 0] L_e,s
//   R_s,l = sum_k f_l(k) V_k,l / sum_k f_l(k),   f_l(k) = max(0, n_k . F_l(x_k))
//
// with x_s, n_s the cluster means, A_g the ambient radiance of group g, and rho_s the material albedo
// or, for textured materials, the mean of the sub-samples' own texture lookups (subsample_albedo).
// In fully lit or fully shadowed clusters R is exactly 1 or 0, so shadows add no sampling error away
// from shadow edges. The pixel-sample value is sum_s (n_s / N) L_s,g: the pixel-area mean over the N
// sub-samples (DESIGN §6.1).
//
// Why the texture is looked up per sub-sample and not once per cluster: a cluster's UVs are only an
// affine image of its hit points where the UV map is continuous over the footprint. Real assets
// break that: atlas charts meet at seams, and vertex colours baked into one texel cell per triangle
// make every face its own chart. A UV mean (or any single lookup point) taken across such a
// discontinuity lands in an unrelated chart, a cell of another face or the empty gap between
// charts. Each sub-sample instead reads its own chart at its own UV, so a seam or a face boundary
// inside the pixel weights each side by its coverage, and a sub-sample crossing it changes the value
// by at most 1/N of the contrast, the staircase of DESIGN §7.9 B2; the seam gutters that DESIGN §7.3
// asks the importer for shrink that step to the gutter error. DESIGN §7.3 [D-1] measured this
// estimator (per-sub-sample prefilter sigma = 0.5/sqrt(N) px) at N = 64 with a smaller error than
// the single cluster Gaussian (log error RMS 0.009-0.014 against 0.016-0.017), at the price of a
// larger deterministic ripple (0.003 against 0.0013 log-e).
//
// Pixel integral (accuracy.pixel_integral; PHASE2 D-P2-3, P2-B B3): phase1 is the above at every N. fixed64 keeps the
// per-member terms (hit point, shading normal, texture albedo, at the footprint 0.25 / 64) to the sub-samples k < 64,
// the fixed 64-position mean whatever the ray count, and takes the coverage n_s / N and the shadow sums sum f and
// sum f V from all N; a cluster with no member below 64 takes its terms from all its members. At N <= 64 the two are
// the same. T1: a pixel-sample with one cluster and no shadow edge (neither kClassMulti nor kClassShadowEdge) has
// shadow ratios of exactly 1 or 0, and its prefix sums keep N = 64's lanes and order; its coverage weight N * fl(1/N)
// is exactly 1 at every power of two N, so there (N >= 64) its value equals its value at N = 64 bit for bit. At 133
// other N in 64..1024 (82, 83, 97, ...) that weight is 1 - 2^-24 and the value moves by an ulp, so the band (B4a)
// takes only powers of two for N_base and N_band. The one gap: a lit light whose f is 0 on every prefix member but not
// on a later one (a shading-normal terminator inside the pixel), whose Lambert term at the prefix means N = 64 drops
// and N > 64 adds; that term is the clipped mean of non-positive terms (to first order in the light's variation).
//
// Shadows over texture (accuracy.texture_shadow; plan p2f §4.1, P2-F F1): pooled is the above. weighted takes the
// shadow ratio of a textured cluster with every member's own texture albedo rho_k as the weight of its f,
//   R_s,l = sum_k rho_k f_l(k) V_k,l / sum_k rho_k f_l(k)   (0 when the denominator is 0),
// rho_k the member's subsample_albedo at the term footprint 0.25 / member_count, for every member (fixed64's k >= 64,
// which carry no per-member terms, included). Where all members carry terms (rho_s = rho_bar, f_bar the members' mean
// f_l), rho_s f_bar R_s,l = (1/n) sum_k rho_k f_l(k) V_k,l / (1 + Cov(rho, f_l) / (rho_bar f_bar)), the per-sub-sample
// sum up to that smooth gain; under uniform f_l, rho_s R_s,l = (1/n) sum_k rho_k V_k,l exactly. The pooled ratio's
// bias over texture (G1 S2: 3-68 eps_L) goes. A cluster lit (or shadowed) for light l on every member has the same
// fp32 sums in both (or 0), so R is exactly 1 (0) as under pooled: lit and umbra values, and T1, are unchanged.
// Untextured clusters keep pooled's arithmetic. A baked cluster under U2 (scans.md §4.2 (i)) takes the same switch:
// its factor is the texel-weighted mean of the members' F_kg (cluster_radiance).
//
// Shading patches (accuracy.shading_patches; plan p2f §4.4, P2-F F5): material is the above, clusters and signature
// terms keyed by (instance, material). crease keys both by (instance, patch), the patches of render/patches.h (cut at
// material changes, between disconnected shells, at hard edges above 30 degrees or on specular bodies, and above 30
// degrees on meshes without normals): hit_key is the triangle's global patch id and key_material its material. A
// sub-sample crossing a patch boundary changes the signature (a J1 jump, localized like a silhouette) and a
// pixel-sample over two patches sets kClassMulti (the band's N). A body whose patches are its materials (a smooth
// single-shell mesh, flat facets below 30 degrees) shades as under material: the same clusters in the same order.
//
// Mover AO (accuracy.mover_ao; plan p2f §4.7, P2-F F9a; render/shade/ao.h): off is the above. proxy multiplies each
// cluster's ambient A_g by 1 - O(x_s, n_s / |n_s|), the cosine-weighted share of its hemisphere left open by the other
// movers' proxy spheres (ao_receiver, ao_visibility_host; the warp kernel's lane-parallel twin): L_s,g = rho_s (... +
// A_g (1 - O)). 1 - O is exactly 1 where no sphere's window reaches the cluster (and for a zero mean normal), so those
// clusters keep off's value bit for bit; O is C2 in the cluster's mean point and normal and enters no signature.
// traced (P2-F F9b; render/shade/ao_traced.h) takes 1 - O as the cluster's open share of its members' ambient rays,
// (n_s - b_s) / n_s over all its members (fixed64's k >= 64 included, as the shadow sums), the blocked bit
// kAoTracedBit of each member's occlusion word; that bit is in the signature like a shadow bit.
//
// Specular lobes (C1 feature "pbr"; plan p2f §4.3, P2-F F6; render/shade/bsdf.h): a material with s > 0 or m > 0
// (MaterialView::lobe) shades its clusters with the GGX + Lambert BSDF instead of Lambert; a material without one,
// and every scene without one, is the Lambert arithmetic above bit for bit (lobe_radiance is not reached). With
// n_u the unnormalized mean shading normal of the cluster's term members (r_s = |n_u|), n^ = n_u / r_s,e its unit
// normal, w_v = -x_s / |x_s| the view, alpha_eff^2 = alpha^2 widened by one resultant (bsdf::alpha_eff, Toksvig's vMF
// width: Lambert's mean normal stays unnormalized, the lobe sees the footprint's spread) and E_s the view-dependent
// specular albedo from the C2 tables,
//   L_s,g = rho_s (1 - m)(1 - E_s) (sum_{l in g} max(0, n_u . w_l) e_l R_s,l / pi + A_g (1 - O)) + A_g (1 - O) E_s
//           + sum_{l in g} e_l max(0, n^ . w_l) D V F(w_v, w_l) R^s_s,l + [g = 0] L_e,s.
// The ambient is uniform, so its specular share is exactly E_s (bsdf.h). R^s_s,l is the lobe's shadow ratio: the
// pooled one, except in a textured cluster under texture_shadow weighted, whose Fresnel reflectance F0 = (1 - m) s +
// m rho_k weights the members: R^s = ((1 - m) s sum f V + m sum rho f V) / ((1 - m) s sum f + m sum rho f), the
// unweighted pair beside the weighted one (ClusterSums::kUnw; the capacity rule: at most kUnweightedLights shadow
// lights in the scene, else R^s = R). Lambert-weighted pooling of a narrow lobe at a shadow edge is a residual of
// the model (O-ST's glossy case measures it).
//
// Near-infrared albedo (plan p2f §4.9, P2-F F11a; render/shade/nir.h): a scene whose cameras' nir_fraction f differs
// between (camera, light group) pairs shades with two albedo channels (SceneView::nir_two_channel). The members' lookups
// return (rho_k, rho_NIR,k) from one weight set (subsample_albedo2), the kV1 ClusterSums (which carry
// the NIR term, F11a folded into F6's kV1) add the sum of the second, and light group g's radiance takes rho_{c,g} = (1 - f_{c,g}) rho + f_{c,g} rho_NIR of the cluster's means;
// accuracy.texture_shadow weighted weights a light's f by the member's own mix for the light's group. Otherwise one
// albedo: f = 0 everywhere is Phase 1's, and an equal f is folded into the textures at level 0.
#pragma once

#include <stdint.h>

#include "core/camera/types.h"
#include "core/hd.h"
#include "core/math/vec.h"
#include "render/lanes.h"
#include "render/ray.h"
#include "render/scene_view.h"
#include "render/shade/ao.h"
#include "render/shade/ao_traced.h"
#include "render/shade/bsdf.h"
#include "render/shade/codes.h"
#include "render/shade/ewa.h"
#include "render/shade/lights.h"
#include "render/shade/nir.h"
#include "render/surface.h"

namespace balla::render {

BALLA_HD inline bool camera_jacobian(const CameraView& cam, Vec3f x, camera::Mat23<float>* j) {
  return cam.model == CameraModel::kRadtan ? camera::project_jacobian(cam.radtan, x, j)
                                           : camera::project_jacobian(cam.kb, x, j);
}

// UV covariance of an isotropic pixel-plane Gaussian of per-axis variance variance_px, centred where
// the surface point x projects, mapped onto the surface (Igehy-style ray differentials): the
// tangent-plane Jacobian d(point)/d(pixel) solves [J_proj(x); n^T] M = [I; 0], so M is the
// first two columns of the inverse, and d(uv)/d(pixel) = [grad u; grad v] M. Returns false where
// the projection Jacobian is undefined or the plane is seen exactly edge-on.
BALLA_HD inline bool footprint_uv_covariance(const CameraView& cam, Vec3f x, Vec3f n, Vec3f grad_u,
                                             Vec3f grad_v, float variance_px, Sym2f* cov) {
  camera::Mat23<float> jp;
  if (!camera_jacobian(cam, x, &jp)) return false;
  const Vec3f r0{jp.m[0][0], jp.m[0][1], jp.m[0][2]};
  const Vec3f r1{jp.m[1][0], jp.m[1][1], jp.m[1][2]};
  const Vec3f c12 = cross(r1, n), c20 = cross(n, r0);
  const float det = dot(r0, c12);
  if (!(abs(det) > 0.0f)) return false;
  const float inv = 1.0f / det;
  const Vec3f m0 = inv * c12, m1 = inv * c20;  // columns of the inverse for the two pixel axes
  const float j00 = dot(grad_u, m0), j01 = dot(grad_u, m1);
  const float j10 = dot(grad_v, m0), j11 = dot(grad_v, m1);
  *cov = {variance_px * (j00 * j00 + j01 * j01), variance_px * (j00 * j10 + j01 * j11),
          variance_px * (j10 * j10 + j11 * j11)};
  return true;
}

// SplitMix64 finalizer (Steele et al. 2014): a bijective 64-bit mixer.
BALLA_HD inline uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Contribution of sub-sample k to the visibility signature, which is the wrap-around sum of these
// terms over the pixel-sample's sub-samples: integer addition is associative, so the warp kernel
// and the CPU agree in any order. Any change of one sub-sample's surface code (instance, material), or
// (instance, patch) under accuracy.shading_patches crease, or of its shadow mask (DESIGN §6.1 J1/J2) changes
// the signature (up to 2^-64 collisions). The term is the sum of its geometry and shadow parts.
BALLA_HD inline uint64_t geometry_term(uint32_t k, uint32_t instance, uint32_t material) {
  const uint64_t salt = 0x9E3779B97F4A7C15ull * (static_cast<uint64_t>(k) + 1);
  return mix64((static_cast<uint64_t>(instance) | (static_cast<uint64_t>(material) << 32)) ^ salt);
}
BALLA_HD inline uint64_t shadow_term(uint32_t k, uint32_t occluded) {
  const uint64_t salt = 0x9E3779B97F4A7C15ull * (static_cast<uint64_t>(k) + 1);
  return mix64(static_cast<uint64_t>(occluded) ^ (salt >> 1) ^ 0xD1B54A32D192ED03ull);
}
BALLA_HD inline uint64_t signature_term(uint32_t k, uint32_t instance, uint32_t material, uint32_t occluded) {
  return geometry_term(k, instance, material) + shadow_term(k, occluded);
}
// The static word of a baked-proxy scene (scans.md §4.2; Tracer::sample's static_occluded: per shadow bit, a proxy
// whose ray the scan blocks in a group with an inserted blocker) enters the signature through this term, shadow_term's
// mixer with its own constant, and only where the word is not 0, so every signature without one is unchanged. A
// sub-sample's term is signature_term + (static word != 0 ? static_term : 0).
BALLA_HD inline uint64_t static_term(uint32_t k, uint32_t static_occluded) {
  const uint64_t salt = 0x9E3779B97F4A7C15ull * (static_cast<uint64_t>(k) + 1);
  return mix64(static_cast<uint64_t>(static_occluded) ^ (salt >> 1) ^ 0xE7037ED1A0B428DBull);
}

// The class word of a pixel-sample (types.h) from two sums, so that the warp kernel reduces them in two passes: `sums`
// holds the wrap-around sum of geometry_term's top words shifted up 16 bits plus the shadowed pairs (< 2^15, no
// carry); `masks` the union of the hit sub-samples' shadow masks in its low half and the union of their complements
// in its high half (class_masks, OR-reducible; both halves are masked to the scene's shadow lights, so a sample in a
// light's umbra everywhere is not an edge and accuracy.mover_ao traced's AO bit, kAoTracedBit = 31 above the at most
// 31 shadow lights, never is one; shadow bits folded mod 16, so with more than 16 shadow lights two lights that share
// a fold can set the edge bit). A lane with no hit (occ_and = ~0) contributes nothing.
BALLA_HD inline uint32_t class_masks(uint32_t occ_or, uint32_t occ_and, int32_t num_shadow_lights) {
  const uint32_t lights = num_shadow_lights >= 32 ? ~0u : (1u << num_shadow_lights) - 1u;
  const uint32_t shadowed = occ_or & lights, lit = ~occ_and & lights;
  return ((shadowed | (shadowed >> 16)) & 0xFFFFu) | ((lit | (lit >> 16)) << 16);
}
BALLA_HD inline uint32_t sample_class(uint32_t sums, bool two_clusters, uint32_t masks, bool textured,
                                      bool specular = false) {
  const bool edge = (masks & (masks >> 16) & 0xFFFFu) != 0;
  return (sums & 0xFFFF0000u) | (two_clusters ? kClassMulti : 0u) | (edge ? kClassShadowEdge : 0u) |
         (textured ? kClassTextured : 0u) | (specular ? kClassSpecular : 0u) | (sums & kClassShadowed);
}

// P2-F F6 (p2f §4.8): a hit material that sets the class word's specular bit, one with a lobe and alpha <= 0.5, i.e.
// alpha^2 <= 1/4. Conservative: alpha_eff >= alpha, and the first pass has no cluster sums to widen it with.
BALLA_HD inline bool class_specular(const MaterialView& m) { return m.lobe != 0 && m.alpha2 <= 0.25f; }

// accuracy.unlit_mask (P2-C C2; SceneView::unlit_mask): the occlusion word of a hit sub-sample (point p, shading normal
// n_shade) as the signature and the class word take it, the bits of the shadow lights whose direct term f there
// (accumulate's lambert_irradiance) is exactly 0 cleared (lit). Such a bit adds 0 to both pooled sums, sum f and
// sum f V, so the radiance does not depend on it: a self-shadow flip on a surface facing away from the light is then
// no signature change and no shadow edge. A pure per-sub-sample function of the hit (N-independent).
BALLA_HD inline uint32_t lit_occlusion(const SceneView& s, const PoseTableView& poses, uint32_t slot, Vec3f p,
                                       Vec3f n_shade, uint32_t occluded) {
  for (int32_t b = 0; b < s.num_shadow_lights; ++b) {
    if (((occluded >> b) & 1u) == 0) continue;
    const int32_t j = s.shadow_light[b];
    if (lambert_irradiance(s.lights[j], light_row(poses, slot, j), p, n_shade) == 0.0f) occluded &= ~(1u << b);
  }
  return occluded;
}

BALLA_HD inline uint32_t hit_material(const GeometryView& g, uint32_t prim) {
  return g.materials[g.single_material ? 0u : prim];
}

// accuracy.shading_patches (P2-F F5): the cluster key of a hit triangle, its global patch id under crease and its
// material under material (GeometryView::patches nullptr), and the material of a key.
BALLA_HD inline uint32_t hit_key(const GeometryView& g, uint32_t prim) {
  return g.patches != nullptr ? g.patches[prim] : hit_material(g, prim);
}
BALLA_HD inline uint32_t key_material(const SceneView& s, uint32_t key) {
  return s.patch_material != nullptr ? s.patch_material[key] : key;
}

// Per-axis variance (px^2) of one sub-sample's texture footprint: its share of the pixel,
// (0.5 / sqrt(N))^2 (DESIGN §7.3 [D-1]).
BALLA_HD inline float subsample_footprint_variance(uint32_t n) { return 0.25f / static_cast<float>(n); }

// fixed64's prefix (D-P2-3): the sub-samples k < kTexturePrefix carry the per-member terms. The sub-samples whose terms a
// pixel-sample of N = n sums: n under phase1 (fixed64 == 0) or at n <= 64, else 64; the footprint is that count's.
inline constexpr uint32_t kTexturePrefix = 64;
BALLA_HD inline uint32_t member_count(int32_t fixed64, uint32_t n) {
  return fixed64 != 0 && n > kTexturePrefix ? kTexturePrefix : n;
}

// Texture albedo seen by one sub-sample: the EWA lookup at its own UV, with its footprint mapped by
// analytic ray differentials onto its own triangle's plane, the plane in which that triangle's UV
// gradients live. An undefined footprint (projection Jacobian undefined, triangle seen exactly
// edge-on) filters to the texture mean.
BALLA_HD inline float subsample_albedo(const CameraView& cam, const TextureView& tex, const float* texels,
                                       const SurfaceSample& ss, float variance_px) {
  Sym2f cov{1e30f, 0.0f, 1e30f};
  footprint_uv_covariance(cam, ss.p, ss.n_geo, ss.grad_u, ss.grad_v, variance_px, &cov);
  return ewa_lookup(tex, texels, ss.uv, cov);
}

// The two-channel lookup of a two-channel texture (TextureView::channels 2): the visible and the near-infrared albedo
// of one sub-sample, from the same footprint and weights.
struct Albedo2 {
  float vis, nir;
};
BALLA_HD inline Albedo2 subsample_albedo2(const CameraView& cam, const TextureView& tex, const float* texels,
                                          const SurfaceSample& ss, float variance_px) {
  Sym2f cov{1e30f, 0.0f, 1e30f};
  footprint_uv_covariance(cam, ss.p, ss.n_geo, ss.grad_u, ss.grad_v, variance_px, &cov);
  float v[2];
  ewa_lookup_n<2>(tex, texels, ss.uv, cov, v);
  return {v[0], v[1]};
}

// Sums of one cluster's sub-sample terms: hit point (3), shading normal (3), texture albedo (1, zero
// for untextured materials), with kV1 the texture's near-infrared albedo (1, P2-F F11a), then per shadow bit b <
// kShadow the pooled-ratio sums sum f and sum f V (DESIGN §7.6). kShadow is a capacity: the CUDA kernel instantiates
// small ones so the sums stay in registers.
// kV1 (P2-F F6): the kernel's V1 path (a scene with a specular lobe, or accuracy.mover_ao): after the kShadow
// (sum f, sum f V) pairs it holds the unweighted pair (sum f, sum f V of the same members, f unscaled by rho) of the
// first kUnw = kUnwLights shadow lights, which a textured lobe cluster under texture_shadow weighted needs for R^s
// (lobe_radiance); the capacity rule is the scene's: at most kUnweightedLights shadow lights, above which R^s = R and
// the pair is never read. unweighted_lights is the pair's capacity: min(kShadow, kUnweightedLights) in a kV1 sum
// whose kernel can see such a scene, 0 otherwise (the plan's "kShadow <= 4 kernels only", p2f.md §4.10; the 32-slot
// kernel takes every scene above N = 256 whatever its lights and keeps the pair). The default is the full pair.
inline constexpr int32_t kUnweightedLights = 4;

BALLA_HD constexpr int32_t unweighted_lights(int32_t shadow, bool v1, bool reachable = true) {
  return v1 && reachable ? (shadow < kUnweightedLights ? shadow : kUnweightedLights) : 0;
}

template <int32_t kShadow, bool kV1 = false, int32_t kUnwLights = unweighted_lights(kShadow, kV1)>
struct ClusterSums {
  static constexpr bool kNir = kV1;  // P2-F F11a: the NIR albedo's sum t[7] rides with the V1 kernels (F6's kV1 fold)
  static constexpr int32_t kTerms = kV1 ? 8 : 7;
  static constexpr int32_t kLights = kShadow;
  static constexpr int32_t kUnw = kUnwLights;
  static constexpr int32_t kUnwBase = kTerms + 2 * kShadow;
  static constexpr int32_t kSize = kUnwBase + 2 * kUnw;
  float t[kSize];

  BALLA_HD void clear() {
    BALLA_UNROLL
    for (int32_t e = 0; e < kSize; ++e) t[e] = 0.0f;
  }
  BALLA_HD Vec3f vec3(int32_t e) const { return {t[e], t[e + 1], t[e + 2]}; }
};

// Adds one sub-sample's terms (p, n_shade: its SurfaceSample's; rho: its subsample_albedo, or 0 for
// untextured materials); each element's sum runs over sub-samples in a fixed order. member_terms false adds the
// shadow sums only (fixed64: a sub-sample k >= 64 of a cluster with a member below 64). rho_weighted (accuracy.
// texture_shadow weighted, textured clusters): rho f in both shadow sums instead of f. unweighted (kV1 sums, F6;
// wants_unweighted): also f in the unweighted pair of the first kUnw lights. light_terms false adds no shadow sums (a
// baked cluster, whose radiance takes none: cluster_radiance). With the NIR term (kV1 sums, F11a), rho_nir is the
// sub-sample's NIR lookup, summed as well, and nir_fraction (its camera's, per light group) mixes the weight of each
// light under rho_weighted: (1 - f) rho + f rho_nir for the light's group.
// Precondition: s.num_shadow_lights <= kShadow.
template <int32_t kShadow, bool kV1, int32_t kUnwLights>
BALLA_HD void accumulate(ClusterSums<kShadow, kV1, kUnwLights>& c, const SceneView& s, const PoseTableView& poses,
                         uint32_t slot, Vec3f p, Vec3f n_shade, float rho, uint32_t occluded, bool member_terms = true,
                         bool rho_weighted = false, bool unweighted = false, bool light_terms = true,
                         float rho_nir = 0.0f, const float* nir_fraction = nullptr) {
  using Sums = ClusterSums<kShadow, kV1, kUnwLights>;
  if (member_terms) {
    c.t[0] += p.x;
    c.t[1] += p.y;
    c.t[2] += p.z;
    c.t[3] += n_shade.x;
    c.t[4] += n_shade.y;
    c.t[5] += n_shade.z;
    c.t[6] += rho;
    if constexpr (Sums::kNir) c.t[7] += rho_nir;
  }
  BALLA_UNROLL
  for (int32_t b = 0; b < kShadow; ++b) {
    if (!light_terms || b >= s.num_shadow_lights) break;
    const int32_t j = s.shadow_light[b];
    float f = lambert_irradiance(s.lights[j], light_row(poses, slot, j), p, n_shade);
    const float f_plain = f;
    if (rho_weighted) {
      if constexpr (Sums::kNir) {
        f *= nir_mix(rho, rho_nir, nir_fraction[s.lights[j].group]);
      } else {
        f *= rho;
      }
    }
    const bool visible = ((occluded >> b) & 1u) == 0;
    c.t[Sums::kTerms + 2 * b] += f;
    if (visible) c.t[Sums::kTerms + 2 * b + 1] += f;
    if constexpr (Sums::kUnw > 0) {
      if (unweighted && b < Sums::kUnw) {
        c.t[Sums::kUnwBase + 2 * b] += f_plain;
        if (visible) c.t[Sums::kUnwBase + 2 * b + 1] += f_plain;
      }
    }
  }
}

// The cluster of material m keeps the unweighted pair (accumulate, lobe_radiance): a textured lobe cluster under
// texture_shadow weighted, in a scene within the capacity rule (at most kUnweightedLights shadow lights).
BALLA_HD inline bool wants_unweighted(const SceneView& s, const MaterialView& m, bool textured) {
  return textured && s.texture_shadow != 0 && m.lobe != 0 && s.num_shadow_lights <= kUnweightedLights;
}

// accuracy.mover_ao proxy: a cluster's receiver for ao_sphere_factor, its mean point (as cluster_radiance takes it)
// and unit mean shading normal; false when the normals cancel (no AO: 1 - O = 1).
template <int32_t kShadow, bool kV1, int32_t kUnwLights>
BALLA_HD bool ao_receiver(uint32_t count, const ClusterSums<kShadow, kV1, kUnwLights>& total, Vec3f* x, Vec3f* n) {
  const float inv_c = 1.0f / static_cast<float>(count);
  *x = inv_c * total.vec3(0);
  const Vec3f n_u = total.vec3(3);
  const float n2 = dot(n_u, n_u);
  if (!(n2 > 0.0f)) return false;
  *n = (1.0f / sqrt(n2)) * n_u;
  return true;
}

// A lobe cluster's radiance per light group (rad[0..num_groups)): the specular formula of the header (P2-F F6,
// plan p2f §4.3), for a material with a lobe (MaterialView::lobe) in a scene with tables (SceneView::bsdf). Same
// operands as cluster_radiance's Lambert path: x, n_u and rho from the reduced sums of the term members, the shadow
// ratios from the pairs (R from the first, R^s from the unweighted one where wants_unweighted), the lights in the same
// order (shadow bits, then the others). With s = m = 0 (E_s = 0, (1 - m)(1 - E_s) = 1) every term but Lambert's is
// exactly 0 and the value is the Lambert path's bit for bit (test_lobe.cpp). n^ = n_u / r_s is a unit normal (the
// visibility V is not scale-free in n^); a mean normal that cancels (r_s below delta_r) widens the lobe to
// alpha_eff^2 -> 1 through bsdf::alpha_eff's floored resultant, and for r_s = 0 exactly n^ faces the view. The view is
// -x_s / |x_s| (the camera is the origin of the camera frame).
template <int32_t kShadow, bool kV1, int32_t kUnwLights>
BALLA_HD void lobe_radiance(const SceneView& s, const PoseTableView& poses, const SampleRequest& req,
                            const MaterialView& m, bool textured, uint32_t count,
                            const ClusterSums<kShadow, kV1, kUnwLights>& total, float* rad, float ao) {
  using Sums = ClusterSums<kShadow, kV1, kUnwLights>;
  const float inv_c = 1.0f / static_cast<float>(count);
  const Vec3f x = inv_c * total.vec3(0);
  const Vec3f n_u = inv_c * total.vec3(3);
  const float rho = textured ? inv_c * total.t[6] : m.albedo;
  const float x2 = dot(x, x);
  const Vec3f w_v = x2 > 0.0f ? (-1.0f / sqrt(x2)) * x : Vec3f{0.0f, 0.0f, -1.0f};
  const float r_s = sqrt(dot(n_u, n_u));
  const Vec3f n_hat = r_s > 1e-20f ? (1.0f / r_s) * n_u : w_v;  // a unit normal (V is not scale-free in n^)
  const float alpha2 = bsdf::alpha_eff(m.alpha2, 0.0f, r_s);
  const bsdf::ViewTerms<float> vt = bsdf::view_terms(n_hat, w_v);
  const bsdf::Fresnel<float> fr = bsdf::fresnel_params(rho, m.metallic, m.specular);
  const float e_s = bsdf::spec_albedo(fr, bsdf::spec_ab(s.bsdf, vt.t, bsdf::table_r(alpha2)));
  const float dw = bsdf::diffuse_weight(m.metallic, e_s);
  // R^s of a textured cluster under texture_shadow weighted: (1 - m) s on the unweighted pair, m on the albedo-weighted
  // one (the header).
  [[maybe_unused]] const bool unw = wants_unweighted(s, m, textured);
  [[maybe_unused]] const float w_dielectric = (1.0f - m.metallic) * m.specular;
  float spec[kMaxLightGroups];
  for (int32_t g = 0; g < s.num_groups; ++g) rad[g] = spec[g] = 0.0f;
  BALLA_UNROLL
  for (int32_t b = 0; b < kShadow; ++b) {
    if (b >= s.num_shadow_lights) break;
    const int32_t j = s.shadow_light[b];
    const LightView& l = s.lights[j];
    const Incidence in = incidence(l, light_row(poses, req.pose_slot, j), x);
    const float f = total.t[Sums::kTerms + 2 * b];
    const float fv = total.t[Sums::kTerms + 2 * b + 1];
    const float ratio = f > 0.0f ? fv / f : 0.0f;  // R_s,l
    rad[l.group] += f > 0.0f ? max(0.0f, in.e * dot(n_u, in.omega)) * ratio : 0.0f;
    float ratio_spec = ratio;                      // R^s_s,l
    if constexpr (Sums::kUnw > 0) {
      if (unw && b < Sums::kUnw) {
        const float den = w_dielectric * total.t[Sums::kUnwBase + 2 * b] + m.metallic * f;
        const float num = w_dielectric * total.t[Sums::kUnwBase + 2 * b + 1] + m.metallic * fv;
        ratio_spec = den > 0.0f ? num / den : 0.0f;
      }
    }
    spec[l.group] +=
        in.e * max(0.0f, dot(n_hat, in.omega)) * bsdf::lobe(alpha2, fr, n_hat, w_v, vt, in.omega) * ratio_spec;
  }
  for (int32_t j = 0; j < s.num_lights; ++j) {
    const LightView& l = s.lights[j];
    if (l.shadow_bit >= 0) continue;
    const Incidence in = incidence(l, light_row(poses, req.pose_slot, j), x);
    rad[l.group] += max(0.0f, in.e * dot(n_u, in.omega));
    spec[l.group] += in.e * max(0.0f, dot(n_hat, in.omega)) * bsdf::lobe(alpha2, fr, n_hat, w_v, vt, in.omega);
  }
  for (int32_t g = 0; g < s.num_groups; ++g) {
    const float ambient = s.ambient[g] * ao;
    rad[g] = rho * dw * (rad[g] * 0.318309886183790672f + ambient) + ambient * e_s + spec[g];
    if (g == 0) rad[g] += m.emissive;
  }
}

// U2's shadow factor of a baked sub-sample (scans.md §4.2; p, n_shade: its SurfaceSample's; occluded, static_occluded:
// its occlusion and static words from the tracer, Tracer::sample): for each light group g with proxy lights,
//   F_kg = 1                              where no proxy of g is blocked by an inserted object (occluded & M_g == 0),
//   F_kg = n_kg / d_kg  (1 if d_kg = 0)   otherwise,
//   n_kg = (sum_{l in P_g} f_kl [s_kl = 0][b_kl = 0]) / pi + A^f_g,
//   d_kg = (sum_{l in P_g} f_kl [s_kl = 0]) / pi + A^f_g,
// f_kl = lambert_irradiance at the sub-sample, the l-sums in shadow-bit order, divisions IEEE on both backends. A
// proxy the scan blocks (s) leaves both sums: the texture holds its absence. accuracy.unlit_mask does not change F: a
// cleared bit has f = 0 and adds 0 to n and d.
// F_kg is added to the cluster's slot pair of g's first proxy bit b_g (SceneView::proxy_slot) as the pooled sum, F in
// the first slot (the second stays 0); with `weighted` (accuracy.texture_shadow weighted on a textured cluster; rho:
// the member's texel value T_k, for every member) the pair holds the sums of T_k and of T_k F_kg (cluster_radiance).
// Precondition: s.num_shadow_lights <= kShadow.
BALLA_HD BALLA_DEVICE_NOINLINE inline float baked_shadow_factor(const SceneView& s, const PoseTableView& poses,
                                                                uint32_t slot, Vec3f p, Vec3f n_shade, int32_t group,
                                                                uint32_t occluded, uint32_t static_occluded) {
  float num = 0.0f, den = 0.0f;
  const uint32_t mask = s.proxy_mask[group];
  for (int32_t b = s.proxy_slot[group]; b < s.num_shadow_lights; ++b) {
    if (((mask >> b) & 1u) == 0 || ((static_occluded >> b) & 1u) != 0) continue;
    const int32_t j = s.shadow_light[b];
    const float f = lambert_irradiance(s.lights[j], light_row(poses, slot, j), p, n_shade);
    den += f;
    if (((occluded >> b) & 1u) == 0) num += f;
  }
  const float n = num * 0.318309886183790672f + s.fill[group];
  const float d = den * 0.318309886183790672f + s.fill[group];
  return d > 0.0f ? n / d : 1.0f;
}

template <int32_t kShadow, bool kV1, int32_t kUnwLights>
BALLA_HD void accumulate_baked(ClusterSums<kShadow, kV1, kUnwLights>& c, const SceneView& s, const PoseTableView& poses,
                               uint32_t slot, Vec3f p, Vec3f n_shade, float rho, uint32_t occluded,
                               uint32_t static_occluded, bool weighted) {
  using Sums = ClusterSums<kShadow, kV1, kUnwLights>;
  BALLA_UNROLL
  for (int32_t b = 0; b < kShadow; ++b) {
    if (b >= s.num_shadow_lights) break;
    const LightView& l = s.lights[s.shadow_light[b]];
    if (l.proxy == 0 || s.proxy_slot[l.group] != b) continue;  // the first proxy bit of a group holds its sums
    const bool blocked = (occluded & s.proxy_mask[l.group]) != 0;
    const float f =
        blocked ? baked_shadow_factor(s, poses, slot, p, n_shade, l.group, occluded, static_occluded) : 1.0f;
    if (weighted) {
      c.t[Sums::kTerms + 2 * b] += rho;
      c.t[Sums::kTerms + 2 * b + 1] += rho * f;
    } else {
      c.t[Sums::kTerms + 2 * b] += f;
    }
  }
}

// Radiance per light group (rad[0..num_groups)) of a cluster whose reduced sums `total` hold the per-member terms of
// `term_count` sub-samples (and, for a baked cluster's shadow factor, the sums of `count` members, its c_s).
// Shadow-casting lights are added in shadow-bit order, then the others in light order. ao: the cluster's 1 - O under
// accuracy.mover_ao proxy (1 otherwise), which multiplies the ambient (a baked cluster's is U3, scans.md §4.2: it
// returns before ao is used). kV1 sums (P2-F F6): a material with a lobe is shaded by lobe_radiance; every other
// material, in every scene, is the Lambert path below.
//
// Baked materials (C1 feature "baked"; docs/design/phase3/scans.md §4.1, U0): the texture holds the radiance, so no
// light, ambient or emission term enters: L_g = T S_g, with T the cluster mean of the sub-samples' texel values (the
// albedo's sum and term_count: under fixed64 the prefix's) or the material's value without a texture, and S_g =
// fl32(K a_g) (SceneView::baked_scale). It equals the lit path's ambient-only value rho (0 / pi + A_g) + 0 at
// A_g = S_g bit for bit. A baked material has no lobe (bundle validation), so it returns first.
// U2 (§4.2; proxy_factor: the sums hold accumulate_baked's, SceneView::baked_proxies with static words): in a group g
// with proxy lights the radiance is (T S_g) F_g, where F_g is
//   pooled:   (sum of the members' F_kg) / count   (count = c_s: all the members that entered the sums, so under
//             fixed64 not the texture prefix, whose count sets T),
//   weighted: (sum_k T_k F_kg) / (sum_k T_k) over the same members, for a textured cluster under
//             accuracy.texture_shadow weighted (P2-F F1's switch, §4.2 (i)). With T the prefix mean, S_g T F_g is
//             S_g (1/c_t) sum_k T_k F_kg wherever every member carries a term; under fixed64 the sums run over all
//             N as the lit weighted ratio's do (F_g = 1 if the denominator is 0).
// With no proxy blocked anywhere in the cluster every F_kg is 1.0, both forms are exactly 1.0 (x / x with equal
// addends in equal order), and the value is U0's bit for bit.
// unshadowed (optional; the dense channel shadow_factor, scans.md §3.7): receives the radiance with every proxy factor
// set to 1, T S_g for a baked cluster (the value before F_g) and rad[g] for a lit one.
template <int32_t kShadow, bool kV1, int32_t kUnwLights>
BALLA_HD void cluster_radiance(const SceneView& s, const PoseTableView& poses, const SampleRequest& req,
                               const MaterialView& m, bool textured, uint32_t term_count, uint32_t count,
                               const ClusterSums<kShadow, kV1, kUnwLights>& total, float* rad, float ao = 1.0f,
                               bool proxy_factor = false, float* unshadowed = nullptr) {
  using Sums = ClusterSums<kShadow, kV1, kUnwLights>;
  const float inv_c = 1.0f / static_cast<float>(term_count);
  if (m.baked >= 0) {
    const float t = textured ? inv_c * total.t[6] : m.albedo;
    for (int32_t g = 0; g < s.num_groups; ++g) rad[g] = t * s.baked_scale[m.baked + g];
    if (unshadowed != nullptr) {
      for (int32_t g = 0; g < s.num_groups; ++g) unshadowed[g] = rad[g];
    }
    if (proxy_factor) {
      const bool weighted = textured && s.texture_shadow != 0;
      BALLA_UNROLL
      for (int32_t b = 0; b < kShadow; ++b) {
        if (b >= s.num_shadow_lights) break;
        const LightView& l = s.lights[s.shadow_light[b]];
        if (l.proxy == 0 || s.proxy_slot[l.group] != b) continue;
        const float first = total.t[Sums::kTerms + 2 * b];
        rad[l.group] *= weighted ? (first > 0.0f ? total.t[Sums::kTerms + 2 * b + 1] / first : 1.0f)
                                 : first / static_cast<float>(count);
      }
    }
    return;
  }
  if constexpr (kV1) {
    if (m.lobe != 0) {
      lobe_radiance(s, poses, req, m, textured, term_count, total, rad, ao);
      if (unshadowed != nullptr) {  // a lit receiver: no proxy factor
        for (int32_t g = 0; g < s.num_groups; ++g) unshadowed[g] = rad[g];
      }
      return;
    }
  }
  const Vec3f x = inv_c * total.vec3(0);
  // Unnormalized: Lambert is linear in n, so this is the footprint mean while F is uniform over the cluster
  // (a directional light) and no sub-sample faces away (S2 MAT-13).
  const Vec3f n_shade = inv_c * total.vec3(3);
  const float rho = textured ? inv_c * total.t[6] : m.albedo;
  [[maybe_unused]] float rho_nir = 0.0f;  // the NIR mean (P2-F F11a), mixed in per light group below
  if constexpr (Sums::kNir) rho_nir = textured ? inv_c * total.t[7] : m.nir_albedo;
  for (int32_t g = 0; g < s.num_groups; ++g) rad[g] = 0.0f;
  BALLA_UNROLL
  for (int32_t b = 0; b < kShadow; ++b) {
    if (b >= s.num_shadow_lights) break;
    const int32_t j = s.shadow_light[b];
    const LightView& l = s.lights[j];
    const float f = total.t[Sums::kTerms + 2 * b];
    const float e = lambert_irradiance(l, light_row(poses, req.pose_slot, j), x, n_shade);
    rad[l.group] += f > 0.0f ? e * (total.t[Sums::kTerms + 2 * b + 1] / f) : 0.0f;
  }
  for (int32_t j = 0; j < s.num_lights; ++j) {
    const LightView& l = s.lights[j];
    if (l.shadow_bit < 0) rad[l.group] += lambert_irradiance(l, light_row(poses, req.pose_slot, j), x, n_shade);
  }
  for (int32_t g = 0; g < s.num_groups; ++g) {
    float rho_g = rho;  // the albedo this camera sees light group g with: the visible mean alone without the NIR term
    if constexpr (Sums::kNir) rho_g = nir_mix(rho, rho_nir, s.cameras[req.cam].nir_fraction[g]);
    rad[g] = rho_g * (rad[g] * 0.318309886183790672f + s.ambient[g] * ao);
    if (g == 0) rad[g] += m.emissive;
  }
  if (unshadowed != nullptr) {
    for (int32_t g = 0; g < s.num_groups; ++g) unshadowed[g] = rad[g];
  }
}

// P2-D D2: where shade_pixel_sample writes its P&V outputs (render/shade/codes.h): output row `row` of `out`, codes
// of `stride` sub-samples (the call's N).
struct PvSample {
  const PvOut* out;
  size_t row;
  uint32_t stride;
};

// CPU form: shades one pixel-sample on the calling thread with the lane schedule of the warp
// kernel. hits and occluded hold req.k_count entries (occluded may be nullptr when the scene has no
// shadow-casting lights); irradiance receives num_groups values; sub_instance (optional) receives
// the k_count instance ids; klass (optional) its sample_class word; pv (optional) the P&V outputs; static_occluded
// (optional, k_count entries) the static words of a baked-proxy scene (Tracer::sample, scans.md §4.2): they enter the
// signature (static_term) and make a baked cluster's radiance take U2's factor (accumulate_baked); without them a baked
// cluster renders U0. irradiance_unshadowed (optional; the dense channel shadow_factor, scans.md §3.7): receives the
// irradiance with every proxy factor set to 1 (cluster_radiance's `unshadowed`; the background adds to both).
inline void shade_pixel_sample(const SceneView& s, const PoseTableView& poses, const SampleRequest& req,
                               const HitRecord* hits, const uint32_t* occluded, float* irradiance,
                               uint64_t* signature, uint32_t* sub_instance, uint32_t* klass = nullptr,
                               const PvSample* pv = nullptr, const uint32_t* static_occluded = nullptr,
                               float* irradiance_unshadowed = nullptr) {
  const CameraView& cam = s.cameras[req.cam];
  const uint32_t n = req.k_count;
  if (s.num_shadow_lights > kPvMaxLights) pv = nullptr;  // no code holds the mask (render_into refuses it)

  uint64_t sig = 0;
  uint32_t geometry_hash = 0, shadowed = 0, occ_or = 0, occ_and = ~0u, clusters = 0;
  bool any_textured = false, any_specular = false;
  // P2-D D2: each sub-sample's mask as the signature takes it and its row code (codes.h), the rows of the table.
  uint32_t pv_mask[kMaxSubsamples];
  uint8_t pv_code_row[kMaxSubsamples];
  uint32_t pv_rows[kPvRows * (3 + kMaxLightGroups + 3 * kPvMaxLights)];
  uint32_t pv_clusters = 0;
  {
    uint32_t cached_id = kInstanceNone;
    const GeometryView* cached = nullptr;
    int32_t cached_geometry = -1;
    for (uint32_t k = 0; k < n; ++k) {
      const HitRecord h = hits[k];
      uint32_t mat = 0;
      if (h.instance != kInstanceNone) {
        if (h.instance != cached_id) {
          const BodyView* b = find_body(s, h.instance);
          cached = b != nullptr ? &s.geometries[b->geometry] : nullptr;
          cached_geometry = b != nullptr ? b->geometry : -1;
          cached_id = h.instance;
        }
        if (cached != nullptr) mat = hit_key(*cached, h.prim);  // the cluster key (material or patch)
      }
      uint32_t occ = occluded != nullptr ? occluded[k] : 0u;
      if (s.unlit_mask != 0 && occ != 0 && h.instance != kInstanceNone && cached != nullptr) {
        // accuracy.unlit_mask: the cluster loop's surface of this sub-sample (the radiance takes occluded[k] as is).
        const bool tex = s.materials[key_material(s, mat)].texture >= 0 && cached->uvs != nullptr;
        const SurfaceSample ss = surface_sample(*cached, pose_row(poses, req.pose_slot, cached_geometry), h.prim,
                                                subsample_direction(s, cam, req.pixel, req.k_begin + k), tex);
        occ = lit_occlusion(s, poses, req.pose_slot, ss.p, ss.n_shade, occ);
      }
      if (pv != nullptr) pv_mask[k] = occ;
      const uint64_t g = geometry_term(k, h.instance, mat);
      sig += g + shadow_term(k, occ);
      if (static_occluded != nullptr && static_occluded[k] != 0) sig += static_term(k, static_occluded[k]);
      geometry_hash += static_cast<uint32_t>(g >> 32);
      if (h.instance != kInstanceNone && cached != nullptr) {
        shadowed += static_cast<uint32_t>(__builtin_popcount(occ));
        occ_or |= occ;
        occ_and &= occ;
        const MaterialView& hm = s.materials[key_material(s, mat)];
        any_textured = any_textured || (hm.texture >= 0 && cached->uvs != nullptr);
        any_specular = any_specular || class_specular(hm);  // P2-F F6
      }
      if (sub_instance != nullptr) sub_instance[k] = h.instance;
    }
  }
  *signature = sig;

  float irr[kMaxLightGroups], irr_u[kMaxLightGroups];
  for (int32_t g = 0; g < s.num_groups; ++g) irr[g] = irr_u[g] = 0.0f;
  const float inv_n = 1.0f / static_cast<float>(n);
  // kV1 sums: a cluster with a lobe takes lobe_radiance (F6); they carry the NIR term (F11a), zero and exact in the mix
  // (f = 0) when the scene is not two-channel.
  using Sums = ClusterSums<kMaxShadowLights, true>;
  const int32_t terms = Sums::kTerms + 2 * s.num_shadow_lights;
  // The unweighted pairs (F6): the first min(num_shadow_lights, kUnw) lights', only in a scene within the capacity
  // rule.
  const int32_t unw_terms = s.specular != 0 && s.texture_shadow != 0 && s.num_shadow_lights <= kUnweightedLights
                                ? 2 * s.num_shadow_lights
                                : 0;
  uint64_t done[kMaxSubsamples / 64] = {};
  auto is_done = [&](uint32_t k) { return ((done[k >> 6] >> (k & 63u)) & 1u) != 0; };
  auto set_done = [&](uint32_t k) { done[k >> 6] |= uint64_t{1} << (k & 63u); };
  Sums lane_sums[kLanes];
  const uint32_t n_terms = member_count(s.fixed64, n);  // fixed64 (D-P2-3): the prefix k < 64 above N = 64
  const float footprint = subsample_footprint_variance(n_terms);

  for (uint32_t k0 = 0; k0 < n; ++k0) {
    if (is_done(k0)) continue;
    ++clusters;
    const uint32_t inst = hits[k0].instance;
    const BodyView* body = inst != kInstanceNone ? find_body(s, inst) : nullptr;
    if (body == nullptr) {  // environment cluster (misses): the background radiance, if any
      uint32_t count = 0;
      for (uint32_t k = k0; k < n; ++k) {
        if (hits[k].instance == inst) {
          set_done(k);
          ++count;
          if (pv != nullptr) pv_code_row[k] = kPvCodeEnv;
        }
      }
      if (inst == kInstanceNone && s.has_background != 0) {
        const float w = static_cast<float>(count) * inv_n;
        for (int32_t g = 0; g < s.num_groups; ++g) irr[g] += w * s.background[g];
        if (irradiance_unshadowed != nullptr) {
          for (int32_t g = 0; g < s.num_groups; ++g) irr_u[g] += w * s.background[g];
        }
      }
      continue;
    }
    const GeometryView& geo = s.geometries[body->geometry];
    const PoseRow& row = pose_row(poses, req.pose_slot, body->geometry);
    const uint32_t mat = hit_key(geo, hits[k0].prim);  // the cluster key (material or patch)
    const MaterialView& m = s.materials[key_material(s, mat)];
    const bool textured = m.texture >= 0 && geo.uvs != nullptr;
    // accuracy.texture_shadow weighted: every member's own lookup weights its f (k >= 64 under fixed64 included; the
    // device makes these lookups for every slot). A baked cluster takes no light sums (light_terms); with static words
    // (u2) it takes U2's factor sums (accumulate_baked), the weighted form again with every member's own lookup.
    const bool light_terms = m.baked < 0;
    const bool u2 = !light_terms && static_occluded != nullptr;
    const bool weighted = light_terms && textured && s.texture_shadow != 0 && s.num_shadow_lights > 0;
    const bool unweighted = wants_unweighted(s, m, textured);  // P2-F F6
    const bool baked_weighted = u2 && textured && s.texture_shadow != 0;
    for (uint32_t l = 0; l < kLanes; ++l) {
      for (int32_t e = 0; e < terms; ++e) lane_sums[l].t[e] = 0.0f;
      for (int32_t e = 0; e < unw_terms; ++e) lane_sums[l].t[Sums::kUnwBase + e] = 0.0f;
    }
    // The leader k0 is the cluster's lowest member: a member below n_terms gives the prefix's terms (fixed64).
    const uint32_t terms_end = k0 < n_terms ? n_terms : n;
    uint32_t count = 0, term_count = 0, ao_blocked = 0;
    const uint32_t pv_row = pv_clusters < kPvRows ? pv_clusters : kPvCodeOverflow;
    for (uint32_t k = k0; k < n; ++k) {
      if (is_done(k) || hits[k].instance != inst || hit_key(geo, hits[k].prim) != mat) continue;
      set_done(k);
      ++count;
      if (pv != nullptr) pv_code_row[k] = static_cast<uint8_t>(pv_row);
      ao_blocked += occluded != nullptr && (occluded[k] & kAoTracedMask) != 0 ? 1u : 0u;  // traced's AO bit
      const bool member_terms = k < terms_end;
      term_count += member_terms ? 1u : 0u;
      const Vec3f d = subsample_direction(s, cam, req.pixel, req.k_begin + k);
      const SurfaceSample ss = surface_sample(geo, row, hits[k].prim, d, textured);
      float rho = 0.0f, rho_nir = 0.0f;
      if (textured && (member_terms || weighted || baked_weighted)) {
        if (s.textures[m.texture].channels == 2) {  // F11a: two-channel scenes' lit textures (a baked one is mono)
          const Albedo2 a = subsample_albedo2(cam, s.textures[m.texture], s.texels, ss, footprint);
          rho = a.vis;
          rho_nir = a.nir;
        } else {
          rho = subsample_albedo(cam, s.textures[m.texture], s.texels, ss, footprint);
        }
      }
      accumulate(lane_sums[k & (kLanes - 1)], s, poses, req.pose_slot, ss.p, ss.n_shade, rho,
                 occluded != nullptr ? occluded[k] : 0u, member_terms, weighted, unweighted, light_terms, rho_nir,
                 cam.nir_fraction);
      if (u2) {
        accumulate_baked(lane_sums[k & (kLanes - 1)], s, poses, req.pose_slot, ss.p, ss.n_shade, rho,
                         occluded != nullptr ? occluded[k] : 0u, static_occluded[k], baked_weighted);
      }
    }
    Sums total{};  // zeroed: GCC cannot see that terms >= kTerms covers what cluster_radiance reads
    for (int32_t e = 0; e < terms + unw_terms; ++e) {
      const int32_t i = e < terms ? e : Sums::kUnwBase + (e - terms);
      float v[kLanes];
      for (uint32_t l = 0; l < kLanes; ++l) v[l] = lane_sums[l].t[i];
      total.t[i] = butterfly_sum(v);
    }
    float ao = 1.0f;  // accuracy.mover_ao: 1 - O at the cluster (proxy: ao.h; traced: ao_traced.h)
    Vec3f ao_x, ao_n;
    if (s.mover_ao == kMoverAoTraced) {
      ao = ao_traced_visibility(count, ao_blocked);
    } else if (s.mover_ao != 0 && s.num_ao_spheres > 0 && ao_receiver(term_count, total, &ao_x, &ao_n)) {
      ao = ao_visibility_host(s, poses, req.pose_slot, inst, ao_x, ao_n);
    }
    float rad[kMaxLightGroups], rad_u[kMaxLightGroups];
    // rad: with 1 - O (P2-F F9a), as the P&V row stores it; u2: a baked cluster takes U2's factor (scans.md §4.2)
    cluster_radiance(s, poses, req, m, textured, term_count, count, total, rad, ao, u2,
                     irradiance_unshadowed != nullptr ? rad_u : nullptr);
    if (pv != nullptr && pv_row < kPvRows) {
      pv_table_row(pv_rows + pv_row * pv_row_words(s.num_groups, s.num_shadow_lights), s, poses, req, inst, mat, m,
                   textured, count, term_count, inv_n, total, rad);
    }
    pv_clusters += pv != nullptr ? 1u : 0u;
    const float w = static_cast<float>(count) * inv_n;
    for (int32_t g = 0; g < s.num_groups; ++g) irr[g] += w * rad[g];
    if (irradiance_unshadowed != nullptr) {
      for (int32_t g = 0; g < s.num_groups; ++g) irr_u[g] += w * rad_u[g];
    }
  }
  for (int32_t g = 0; g < s.num_groups; ++g) irradiance[g] = irr[g];
  if (irradiance_unshadowed != nullptr) {
    for (int32_t g = 0; g < s.num_groups; ++g) irradiance_unshadowed[g] = irr_u[g];
  }
  if (klass != nullptr || pv != nullptr) {
    const uint32_t masks = class_masks(occ_or, occ_and, s.num_shadow_lights);
    const uint32_t word =
        sample_class((geometry_hash << 16) + shadowed, clusters > 1, masks, any_textured, any_specular);
    if (klass != nullptr) *klass = word;
    if (pv != nullptr) {
      pv_store(*pv->out, pv->row, pv->stride, word, n, s.num_groups, s.num_shadow_lights, pv_rows, pv_clusters,
               pv_code_row, pv_mask);
    }
  }
}

}  // namespace balla::render
