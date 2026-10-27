// The reflection policy of radiance v1 (DESIGN §7.7, DL-37; p2f.md §4.3, P2-F F4). v1 traces no reflection of scene
// geometry, so a material with a specular lobe (C1 feature "pbr": specular or metallic > 0) misses what its mirror
// direction would show. Per used material with a lobe, reflection_policy reports:
// - refused: roughness < 0.2 (DESIGN's r_s), or metallic > 0 and roughness < 0.5. ballasim.engine.engine.run refuses
//   such a material (ConfigError) unless scene.allow_missing_reflections is true.
// - bound = ln(1 + F_max L_max / L_diff,min), the largest missing log contrast, for run.json's reflection_bounds:
//   - F_max = 1, Schlick's supremum at grazing (v1 does not compute the trajectories' most grazing view);
//   - L_max = max(B, max over used materials of rho_max (E_max / pi + A) + L_e, and K rho_max for a baked one), the
//     brightest radiance a reflection can show: A and B the ambient and background sums, L_e the band emission, K the
//     baked scale (times sum_g a_g m_g,max for its group shares a_g), and E_max = sum over lights of I / d_min^2 (point, spot), I (directional) or pi L (rect, disk),
//     every light taken at its group's waveform peak m_g,max (A, B and E_max are sums of m_g,max times the light's
//     intensity: an upper bound at every instant, as the groups may peak at different times):
//     each light at its closest approach to any geometry, unshadowed and at normal incidence. d_min is exact (point to
//     triangle) between a fixed light and static geometry; a moving light or body is bounded by the ball of its
//     trajectory's control positions (a B-spline stays in their convex hull), widened by its mesh's or offset's radius;
//   - L_diff,min = min(1, (1 - m) rho_min) A_min, which bounds the rendered radiance A [(1 - m)(1 - E_s) rho + E_s]
//     under the ambient alone from below for every E_s in [0, 1]; A_min is the ambient at its groups' waveform
//     troughs m_g,min (a lower bound at every instant: 0 in the off phase of a pulse with floor 0). It is 0, and the
//     bound +inf, without ambient or for m = 1.
//   rho is the albedo a camera's light group sees (render/shade/nir.h): (1 - f) rho_vis + f rho_NIR with the cameras'
//   sensor.nir_fraction f (P2-F F11a; rho_vis alone at f = 0), so rho_max and rho_min take every f present.
//   Radiance is the renderer's band radiance (sensor RGB weights w, as scene_data.cpp weighs albedo, emission and
//   light colour), without mover AO (P2-F F9 lowers A by 1 - O). A light group's waveform m_g(t) is mean-normalised
//   and may peak above 1 (a pulse: floor + (1 - floor) / duty; a Fourier sum: 1 + sum |a_k|) and drop below it (floor;
//   1 - sum |a_k|), hence the peak and trough above. They come from the caller (`groups`; the C++ bundle holds group
//   names only): empty means every group is constant (m_g = 1).
//   Reflected highlights of other glossy surfaces are not in L_max (second order); O22 (F12b) measures the bound.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "render/bundle.h"

namespace balla::render {

// Maths-bounded constants of the policy (DESIGN §7.7), documented and not knobs.
inline constexpr float kReflectionRoughnessMin = 0.2f;       // r_s: below it reflections carry full-contrast edges
inline constexpr float kReflectionMetalRoughnessMin = 0.5f;  // metals below it are refused too
inline constexpr double kReflectionFMax = 1.0;               // Schlick's F at grazing

// A light group's waveform range over all time: [m_min, m_max] (engine's waveform_range; constant: [1, 1]).
struct GroupRange {
  double lo = 1.0, hi = 1.0;
};

struct ReflectionReport {
  uint32_t body = 0;      // index into bundle.bodies
  uint32_t material = 0;  // index into that body's materials
  float roughness = 1.0f, metallic = 0.0f, specular = 0.0f;
  bool refused = false;   // by the policy, unless scene.allow_missing_reflections
  double l_max = 0.0, l_diff_min = 0.0;
  double bound = 0.0;     // ln(1 + kReflectionFMax l_max / l_diff_min); +inf when l_diff_min is 0 or l_max is +inf
};

// Whether the policy refuses material m (it has a lobe and is too smooth for v1's missing reflections).
bool reflection_refused(const Material& m);

// One report per used material with a lobe, in body and material order; empty when no used material has one. `w`: the
// run's sensor RGB weights (sensor.rgb_weights). `groups`: the waveform range of every light group of the bundle, in
// order (empty: all constant). `nir_fraction`: the simulated cameras' sensor.nir_fraction rows (P2-F F11a; an empty row
// or no rows is f = 0): rho_hi and rho_lo are then the largest and smallest of (1 - f) rho_vis + f rho_NIR over the
// material's texels and every f present (a lit material; the visible albedo alone where every f is 0, bit for bit).
// Throws std::invalid_argument for an invalid bundle, for `groups` of another length or with a range outside
// 0 <= lo <= hi < inf, or for an f outside [0, 1].
std::vector<ReflectionReport> reflection_policy(const SceneBundle& bundle, const std::array<double, 3>& w,
                                                const std::vector<GroupRange>& groups = {},
                                                const std::vector<std::vector<double>>& nir_fraction = {});

}  // namespace balla::render
