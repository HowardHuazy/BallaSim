// The GGX + Lambert BSDF of radiance v1 (p2f.md §4.3, DESIGN §7.4): the microfacet lobe, the Fresnel model, the
// view-dependent specular albedo from C2 tables, the layered diffuse, the smooth floors and caps that keep every
// term C2 (DESIGN §7.10), and the lobe's LTC for area lights. Templated on T: float on the executors (bitwise host =
// device, as all BALLA_HD maths), double for the conformance gate (DESIGN §7.11) and brute-force references. No
// renderer code path uses it yet: F6 adds the per-cluster shading, F8b the LTC's polygon integral.
//
// Mono parameters, sensor-weighted like the albedo: rho base albedo, m metallic, alpha = r^2 from roughness r, and s
// the dielectric F0 (glTF 0.04 specularColor specularFactor at IOR 1.5; Blender ((eta - 1)/(eta + 1))^2 2 level).
//
// Conventions (p2f.md §3), as read in Cycles (Blender 5.2.2 kernel sources) and Mitsuba 3.9.1 (installed headers).
// pbrt-v4 (b4ce968) has no local copy: its entries are p2f.md §3 and Appendix A, and those marked "from memory" were
// not re-read against its source.
// - alpha = r^2: Cycles svm/closure.h (Principled: alpha_x = sqr(roughness)) and glTF. pbrt-v4 remaps with
//   RoughnessToAlpha = sqrt(r); Mitsuba takes alpha as given (its rough* default is Beckmann, alpha 0.1).
// - D: GGX / Trowbridge-Reitz in the tangent form alpha^2 / (pi (alpha^2 c_h^2 + s_h^2)^2), s_h^2 = |n x h|^2 by
//   the cross product: Mitsuba's render/microfacet.h eval() has this form (x^2 + y^2 from the local frame). Cycles'
//   bsdf_D (closure/bsdf_microfacet.h) uses 1 - c_h^2, which loses fp32 accuracy at small alpha (p2f.md §1.2).
//   pbrt-v4: TrowbridgeReitzDistribution::D (util/scattering.h), in tan^2 theta_h (from memory).
// - Shadowing-masking: height-correlated Smith, G2 = 1/(1 + Lambda_v + Lambda_l) (Heitz 2014; Cycles bsdf_G(alpha2,
//   cos_NI, cos_NO); pbrt-v4 TrowbridgeReitzDistribution::G). Mitsuba's G() is the separable G1 G1. Folded into
//   V = G2/(4 mu_l mu_v), with s_x^2 = |n x w_x|^2 from the cross product, never 1 - mu^2.
// - Fresnel: Schlick F0 + (F90 - F0)(1 - u)^5 with F0 = (1 - m) s + m rho and F90 = m + (1 - m) min(1, 50 s): the
//   UE4 F_Schlick / Filament f90 = saturate(50 f0) convention (Karis 2013; Lagarde and de Rousiers 2014; cited, not
//   re-read), continuous to s = 0. Cycles, pbrt-v4 (FrDielectric) and Mitsuba (fresnel()) use the exact dielectric
//   Fresnel, which needs eta-indexed 3-D albedo tables (Cycles' ggx_gen_schlick_ior_s). PROVISIONAL (p2f.md §13.2):
//   every Fresnel dependence goes through fresnel_params, schlick and spec_albedo, which an exact Fresnel replaces.
// - Diffuse under the lobe: (1 - m)(1 - E_s(mu_v)) rho/pi, Cycles' view-only layering (closure_layering_weight and
//   weight *= 1 - metallic in svm/closure.h; the albedo from bsdf_microfacet_estimate_albedo). Not reciprocal, as
//   in Cycles; the lobe alone is. pbrt-v4 has no principled layering (p2f.md §3; its CoatedDiffuse is a stochastic
//   LayeredBxDF, from memory). Mitsuba's roughplastic weights the base by the transmittance at both directions,
//   with internal scattering (p2f.md §3).
// - Single scattering, no multiscatter compensation: Cycles' default GGX (Multiscatter GGX, with
//   microfacet_ggx_preserve_energy, is its option); pbrt-v4's ConductorBxDF / DielectricBxDF and Mitsuba's rough*
//   are single scattering too (p2f.md §3; from memory).
// - Tables over (t, r) = (sqrt(1 - sqrt(mu_v)), sqrt(alpha)), C2 cubic B-splines (bspline_table.h). r is Heitz et al.'s
//   (2016) fitLTC axis. The view axis departs from their sqrt(1 - mu_v) (p2f.md §4.3): A and B behave like
//   mu_v log mu_v at grazing, where sqrt(1 - mu_v) left 6e-3 at the last cell centre against T-TAB's 1e-3; t puts
//   mu_v ~ 4 (1 - t)^2 there (max 5.9e-4 for r >= 0.2) and is, like sqrt(1 - mu_v), an odd analytic function of the
//   signed view angle (t ~ theta / 2), so the even and odd extensions at t = 0 stay exact. Near-mirror rows (r < 0.2,
//   only under scene.allow_missing_reflections) are not resolved at grazing (up to 5.5e-2 below r 0.085; tests).
//   Cycles reads its 32 x 32 ggx_E over (roughness, mu) bilinearly (bsdf_microfacet.h, util/lookup_table.h);
//   pbrt-v4 and Mitsuba tabulate no microfacet albedo (p2f.md §3).
// - E_s <= 1 - kDeltaE / 2 through a C2 cap (spec_albedo): the spline overshoots 1 by up to 3e-3 in the near-mirror
//   rows at grazing, and the cap keeps (1 - m)(1 - E_s) rho + E_s <= 1 at every r. Cycles clamps the same quantity,
//   but hard: closure_layering_weight (closure/bsdf_util.h) saturates 1 - albedo to [0, 1]. pbrt-v4 and Mitsuba have
//   no tabulated albedo to cap (p2f.md §3).
// - LTC (F3c): Heitz et al.'s (2016) linearly transformed cosine of the normalized lobe (F = 1; E_s carries Fresnel and
//   magnitude), its inverse matrix in their layout (middle entry 1) on the (t, r) splines, odd off-diagonals at t = 0,
//   and the frame t towards the view's projection. Fitted by fitLTC's procedure with the frame's angle free
//   (tools/bsdf_tables/ltc_ref.h; fitLTC fixes it at the lobe's mean direction). Cycles shades GGX by sampling, and its
//   one LTC is the sheen closure (Zeltner et al. 2022; closure/bsdf_sheen.h): two parameters in a 32 x 32 bilinear
//   table over (cos theta, roughness), and the same frame, make_orthonormals_safe_tangent(N, wi) (T towards the
//   view's projection; make_orthonormals of N at normal view, sample/mapping.h). Blender's Eevee shades area lights
//   with Heitz et al.'s fitted tables (from memory: its shaders are not in the release). pbrt-v4 and Mitsuba 3 sample
//   emitters and have no LTC (Mitsuba 3.9.1 headers checked; pbrt-v4 from memory).
// - alpha_eff^2 >= kAlpha2Min / 2 = 1e-8 (alpha >= 1e-4) through a C2 floor, identity above 2e-8: Mitsuba clamps
//   alpha to 1e-4 (microfacet.h configure()); pbrt-v4 raises alpha to 1e-4 when the lobe is not effectively smooth and
//   treats alpha < 1e-3 as a perfect mirror (TrowbridgeReitzDistribution, EffectivelySmooth; from memory); Cycles
//   treats alpha_x alpha_y <= 2e-10 as singular (roughness_is_almost_specular, bsdf_microfacet.h).
#pragma once

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "render/shade/bspline_table.h"

namespace balla::render::bsdf {

// Maths-bounded constants (p2f.md §4.3, §4.11), documented and not knobs.
inline constexpr double kDeltaMu = 1e-3;     // floor of mu_v: shading normals may face away from the view
inline constexpr double kDeltaR = 1e-3;      // floor of the normal resultant r_s: normals that cancel
inline constexpr double kDeltaA = 1e-2;      // cap of alpha_eff^2 at 1
inline constexpr double kDeltaE = 1e-3;      // cap of E_s below 1: never above 1 - 5e-4, identity below 0.999
inline constexpr double kAlpha2Min = 2e-8;   // floor of alpha_eff^2: never below 1e-8 (alpha >= 1e-4, Mitsuba's)
inline constexpr double kF90Scale = 50.0;    // F90 = min(1, 50 s) for dielectrics

template <class T>
inline constexpr T kInvPi = T(0.318309886183790671537767526745028724);

// --- Smooth floors and caps (p2f.md §4.3, §4.7) ---------------------------------------------------------------

// q(u) = 1/2 (u <= 0), 1/2 + u^3 - u^4/2 (0 < u < 1), u (u >= 1): monotone and C2 (value, slope and curvature match
// at both joins).
template <class T>
BALLA_HD T floor_q(T u) {
  if (!(u > T(0))) return T(0.5);
  if (u >= T(1)) return u;
  const T u3 = u * u * u;
  return T(0.5) + (u3 - T(0.5) * (u3 * u));
}

// delta q(x / delta): x itself (exactly) for x >= delta, never below delta / 2.
template <class T>
BALLA_HD T smooth_floor(T x, T delta) {
  return x >= delta ? x : delta * floor_q(x / delta);
}

// smin(a, 1) = 1 - delta q((1 - a) / delta): a itself (exactly) for a <= 1 - delta, never above 1 - delta / 2.
template <class T>
BALLA_HD T smooth_cap1(T a, T delta) {
  return a <= T(1) - delta ? a : T(1) - delta * floor_q((T(1) - a) / delta);
}

// --- The Fresnel model (the one place to replace for an exact dielectric Fresnel) -------------------------------

template <class T>
struct Fresnel {
  T f0, f90;
};

template <class T>
BALLA_HD Fresnel<T> fresnel_params(T rho, T metallic, T s) {
  const T dielectric = T(1) - metallic;
  return {dielectric * s + metallic * rho, metallic + dielectric * min(T(1), T(kF90Scale) * s)};
}

// F(u) at u = w . h in [0, 1].
template <class T>
BALLA_HD T schlick(const Fresnel<T>& fr, T u) {
  const T g = T(1) - clamp(u, T(0), T(1));
  const T g2 = g * g;
  return fr.f0 + (fr.f90 - fr.f0) * (g2 * g2 * g);
}

// The table split of the specular albedo, E_s = F0 A + F90 B: A = int D V (1 - (1 - u)^5) mu_l dw_l and
// B = int D V (1 - u)^5 mu_l dw_l over the hemisphere of w_l.
template <class T>
struct SpecAB {
  T a, b;
};

// E_s = smin(F0 A + F90 B, 1) with delta kDeltaE: F0 A + F90 B itself (exactly; so 0 for s = m = 0) up to 0.999, and
// never above 1 - kDeltaE / 2, where the near-mirror rows' spline overshoots 1 at grazing. It moves an E_s <= 1 by at
// most kDeltaE / 2, and none in the policy region r >= 0.2, where A + B <= 0.9987 at every view mu_v,e >= delta_mu / 2
// (test_bsdf.cpp).
template <class T>
BALLA_HD T spec_albedo(const Fresnel<T>& fr, const SpecAB<T>& ab) {
  return smooth_cap1(fr.f0 * ab.a + fr.f90 * ab.b, T(kDeltaE));
}

// --- Tables -----------------------------------------------------------------------------------------------------

// Coefficient tables (bspline_table.h layout) indexed by (t, r) = (sqrt(1 - sqrt(mu_v)), sqrt(alpha)): A and B (F3a),
// and the LTC's inverse matrix entries (F3c; ltc_inverse). Host views come from bsdf_tables.h; F6 uploads them.
struct BsdfTables {
  const float* spec_a;
  const float* spec_b;
  const float* ltc_m00;
  const float* ltc_m02;
  const float* ltc_m20;
  const float* ltc_m22;
};

// The table's roughness coordinate sqrt(alpha) = alpha2^(1/4).
template <class T>
BALLA_HD T table_r(T alpha2) {
  return sqrt(sqrt(max(alpha2, T(0))));
}

template <class T>
BALLA_HD SpecAB<T> spec_ab(const BsdfTables& tables, T t, T r) {
  const BsplineTap<T> tt = bspline_tap(t), tr = bspline_tap(r);
  return {bspline_eval(tables.spec_a, tt, tr), bspline_eval(tables.spec_b, tt, tr)};
}

// --- View terms -------------------------------------------------------------------------------------------------

// mu: the floored mu_v,e = delta_mu q(mu_v / delta_mu); s2 = |n x w_v|^2; t = sqrt(1 - sqrt(mu_v,e)), the tables'
// view coordinate. Near normal view 1 - sqrt(mu_v) is formed as s2 / ((1 + mu_v)(1 + sqrt(mu_v))) without
// cancellation, so t ~ |theta| / 2 stays accurate and, with the even/odd table extensions, smooth through theta = 0.
template <class T>
struct ViewTerms {
  T mu, s2, t;
};

template <class T>
BALLA_HD ViewTerms<T> view_terms(Vec3<T> n, Vec3<T> wv) {
  const T mu_v = dot(n, wv);
  const Vec3<T> c = cross(n, wv);
  const T s2 = dot(c, c);
  const T mu = smooth_floor(mu_v, T(kDeltaMu));
  const T sq = sqrt(mu);
  const T one_minus_sq = mu_v >= T(0.5) ? s2 / ((T(1) + mu_v) * (T(1) + sq)) : T(1) - sq;
  return {mu, s2, sqrt(clamp(one_minus_sq, T(0), T(1)))};
}

// --- The microfacet lobe ----------------------------------------------------------------------------------------

// GGX D for a half vector h, given c = n . h and s2 = |n x h|^2 with unit n: alpha^2 (c^2 + s2)^2 /
// (pi (alpha^2 c^2 + s2)^2), which is alpha^2 / (pi (alpha^2 c_h^2 + s_h^2)^2) for the unit half vector, so h need
// not be normalized. 0 for h = 0 (w_l = -w_v, reachable only through the floor of a back-facing mu_v).
template <class T>
BALLA_HD T ggx_d(T alpha2, T c, T s2) {
  const T c2 = c * c;
  const T den = alpha2 * c2 + s2;
  if (!(den > T(0))) return T(0);
  const T q = (c2 + s2) / den;
  return alpha2 * (q * q) * kInvPi<T>;
}

// The height-correlated visibility V = G2 / (4 mu_l mu_v) = 1 / (2 (mu_l g_v + mu_v g_l)), g_x = sqrt(mu_x^2 +
// alpha^2 s_x^2), for mu_v, mu_l > 0.
template <class T>
BALLA_HD T smith_v(T alpha2, T mu_v, T s2_v, T mu_l, T s2_l) {
  const T g_v = sqrt(mu_v * mu_v + alpha2 * s2_v);
  const T g_l = sqrt(mu_l * mu_l + alpha2 * s2_l);
  return T(0.5) / (mu_l * g_v + mu_v * g_l);
}

// D V F towards light direction wl (0 for mu_l <= 0), for the view terms vt of wv. Every factor is symmetric in
// (wv, wl): h = wv + wl, u = |h| / 2 (= wv . h_unit for unit vectors), and the sum in smith_v commutes, so with
// mu_v, mu_l >= delta_mu the lobe is bitwise reciprocal.
template <class T>
BALLA_HD T lobe(T alpha2, const Fresnel<T>& fr, Vec3<T> n, Vec3<T> wv, const ViewTerms<T>& vt, Vec3<T> wl) {
  const T mu_l = dot(n, wl);
  if (!(mu_l > T(0))) return T(0);
  const Vec3<T> cl = cross(n, wl);
  const Vec3<T> h = wv + wl;
  const Vec3<T> ch = cross(n, h);
  const T c = dot(n, h), s2_h = dot(ch, ch);
  const T d = ggx_d(alpha2, c, s2_h);
  const T v = smith_v(alpha2, vt.mu, vt.s2, mu_l, dot(cl, cl));
  const T u = T(0.5) * sqrt(dot(h, h));
  return d * v * schlick(fr, u);
}

// --- The LTC of the lobe (F3c; F8b integrates polygons with it) -------------------------------------------------

// The inverse of the linearly transformed cosine that approximates the normalized lobe D V mu_l / (A + B) (F = 1; the
// Fresnel factor and the magnitude enter as E_s), in the frame (t, b, n) of ltc_frame, normalized so that its middle
// entry is 1 (Heitz et al. 2016's layout): [[m00, 0, m02], [0, 1, 0], [m20, 0, m22]]. Fitted by
// tools/bsdf_tables/ltc_ref.h. At normal view (t = 0) the fit is isotropic, m00 = 1 and m02 = m20 = 0, and the
// splines extend m00 and m22 evenly and m02 and m20 oddly through it, so the matrix in a fixed frame is C2 in the
// signed view angle (the frame turns by pi there, which flips the off-diagonal entries).
template <class T>
struct LtcInverse {
  T m00, m02, m20, m22;
};

template <class T>
BALLA_HD LtcInverse<T> ltc_inverse(const BsdfTables& tables, T t, T r) {
  const BsplineTap<T> tt = bspline_tap(t), tr = bspline_tap(r);
  return {bspline_eval(tables.ltc_m00, tt, tr), bspline_eval(tables.ltc_m02, tt, tr),
          bspline_eval(tables.ltc_m20, tt, tr), bspline_eval(tables.ltc_m22, tt, tr)};
}

// The LTC's tangent frame: t = normalize(w_v - (n . w_v) n), towards the view's projection, and b = n x t, so (t, b,
// n) is right-handed. At normal view (|w_v - (n . w_v) n|^2 <= 1e-12) t comes from Duff et al.'s (2017) branchless
// orthonormal basis of n; the isotropic table node makes the result independent of t there.
template <class T>
struct LtcFrame {
  Vec3<T> t, b;
};

template <class T>
BALLA_HD LtcFrame<T> ltc_frame(Vec3<T> n, Vec3<T> wv) {
  const Vec3<T> d = wv - dot(n, wv) * n;
  const T d2 = dot(d, d);
  Vec3<T> t;
  if (d2 > T(1e-12)) {
    t = (T(1) / sqrt(d2)) * d;
  } else {
    const T sign = copysign(T(1), n.z);
    const T a = T(-1) / (sign + n.z);
    t = {T(1) + sign * (n.x * n.x) * a, sign * ((n.x * n.y) * a), -sign * n.x};
  }
  return {t, cross(n, t)};
}

// A vector d from the shading point (an emitter vertex y - x) in the LTC's cosine space: M^-1 (t . d, b . d, n . d).
// F8b clips the polygon of these at z = 0 and sums its edges there (p2f.md §4.6).
template <class T>
BALLA_HD Vec3<T> ltc_apply(const LtcInverse<T>& m, const LtcFrame<T>& f, Vec3<T> n, Vec3<T> d) {
  const T x = dot(f.t, d), z = dot(n, d);
  return {m.m00 * x + m.m02 * z, dot(f.b, d), m.m20 * x + m.m22 * z};
}

// --- Effective roughness ----------------------------------------------------------------------------------------

// alpha_eff^2 = smin(alpha^2 + mean alpha^2_add + kappa(r_s,e), 1), floored by smooth_floor at kAlpha2Min (so never
// below kAlpha2Min / 2 = 1e-8), with r_s,e = delta_r q(r_s / delta_r) and kappa(r) = 2 (1 - r^2) / (3 r - r^3), the
// GGX/Beckmann width of a vMF lobe whose mean resultant is r (Toksvig 2005; Han et al. 2007). r_s = 1 (all normals
// equal) adds nothing.
template <class T>
BALLA_HD T alpha_eff(T alpha2, T alpha2_add_mean, T r_s) {
  const T r = smooth_floor(r_s, T(kDeltaR));
  const T one_minus_r2 = max(T(0), (T(1) - r) * (T(1) + r));
  const T kappa = T(2) * one_minus_r2 / (r * (T(3) - r * r));
  return smooth_floor(smooth_cap1(alpha2 + alpha2_add_mean + kappa, T(kDeltaA)), T(kAlpha2Min));
}

// --- The BSDF ---------------------------------------------------------------------------------------------------

template <class T>
struct BsdfParams {
  T rho;       // base albedo
  T metallic;  // m in [0, 1]
  T alpha2;    // alpha^2 = r^4, after alpha_eff for a cluster
  T s;         // dielectric F0
};

// (1 - m)(1 - E_s), the weight of the Lambert part under the lobe.
template <class T>
BALLA_HD T diffuse_weight(T metallic, T e_s) {
  return (T(1) - metallic) * (T(1) - e_s);
}

// f(wv, wl) = (1 - m)(1 - E_s(mu_v)) rho/pi + D V F for unit n, wv (towards the camera) and wl (towards the light);
// 0 for mu_l <= 0. With s = m = 0, F0 = F90 = 0: E_s and the lobe are exactly 0 and f = rho/pi (Lambert).
template <class T>
BALLA_HD T bsdf_eval(const BsdfTables& tables, const BsdfParams<T>& p, Vec3<T> n, Vec3<T> wv, Vec3<T> wl) {
  if (!(dot(n, wl) > T(0))) return T(0);
  const ViewTerms<T> vt = view_terms(n, wv);
  const Fresnel<T> fr = fresnel_params(p.rho, p.metallic, p.s);
  const T e_s = spec_albedo(fr, spec_ab(tables, vt.t, table_r(p.alpha2)));
  return diffuse_weight(p.metallic, e_s) * p.rho * kInvPi<T> + lobe(p.alpha2, fr, n, wv, vt, wl);
}

}  // namespace balla::render::bsdf
