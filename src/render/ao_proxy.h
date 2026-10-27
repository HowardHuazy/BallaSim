// Mover AO proxies (accuracy.mover_ao proxy; plan p2f §4.7, P2-F F9a; PHASE2 F108): each moving body with triangles is
// replaced, for the ambient occlusion it casts on every other surface, by a set of k <= 8 spheres in its body frame
// (render/shade/ao.h evaluates their form factors per cluster). The fit, from the body's mesh at SceneData
// construction (host code, deterministic, fp64):
// - kAoFitSamples area-weighted surface samples: the i-th takes the triangle where the cumulative area reaches
//   (i + 1/2) / kAoFitSamples of the total and barycentrics from the R2 sequence (Roberts 2018: the plastic number's
//   additive recurrence, folded into the triangle), so the samples depend only on the mesh;
// - k = clamp(ceil(e_max / e_min - kAoFitSlack), 1, kAoMaxSpheres), e the extents (max - min) of the samples along the
//   principal axes of their covariance (Jacobi). The slack (F0's prototype's, p2_f_ao_bias.py) keeps a tessellated
//   sphere, whose sampled extents differ by about 1 %, at k = 1 (O-AO's sphere case takes the proxy to be the sphere);
//   a flat body (e_min = 0) takes 8. Where the two minor axes are degenerate (a 4:1:1 box) the solver's basis between
//   them sets e_min (0.1 to 0.14 there: k = 3 or 4);
// - centres by k-means (Lloyd, at most kAoFitIterations rounds, until the assignment repeats; ties to the lowest
//   index), seeded by farthest points: the sample farthest from the centroid, then each next the sample farthest from
//   the chosen ones (k = 1: the centroid);
// - R_i the RMS distance of sphere i's samples to c_i (a sphere's surface samples give its radius, up to the
//   tessellation), spheres without samples or radius dropped; the body's R_b = sqrt(sum R_i^2), which sets the
//   window d0 = R_b / sqrt(eps_AO) (ao.h; eps_AO = kAoEps below).
// Maths-bounded constants (§4.11), documented, not parameters.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace balla::render {

inline constexpr int32_t kAoMaxSpheres = 8;        // k <= 8 spheres per body
inline constexpr uint32_t kAoFitSamples = 4096;    // area-weighted surface samples of the fit
inline constexpr int32_t kAoFitIterations = 64;    // Lloyd rounds at most
inline constexpr double kAoFitSlack = 0.1;         // ceil's slack on e_max / e_min (a sampled sphere stays k = 1)
// The window's omission per body, eps_AO (§4.7): a fixed 0.002, accuracy.eps_L's family prior (fidelity's eps_L at the
// family priors, phase1's), not the run's eps_L. A run whose eps_L resolves lower (sigma_v,min < 0.02, or an explicit
// value) keeps 0.002 per body, e.g. 2 eps_L at eps_L = 0.001. Reading the run's eps_L in the render stage would move
// accuracy.eps_L (a sensor-stage key) into the render digest; that run-dependent form is open for F9b / F13.
inline constexpr double kAoEps = 2e-3;
inline constexpr double kAoWindowScale = 22.360679774997896964;  // 1 / sqrt(kAoEps): d0 = R_b kAoWindowScale

struct AoProxySphere {
  std::array<double, 3> centre;  // body frame
  double radius;
};

struct AoProxy {
  std::vector<AoProxySphere> spheres;  // empty for a mesh without area
  double radius_b = 0.0;               // R_b = sqrt(sum_i R_i^2)
  double d0 = 0.0;                     // the window's start R_b / sqrt(kAoEps) (render/shade/ao.h)
};

// The proxy of one body's mesh (positions V x 3 in the body frame, indices T x 3).
AoProxy fit_ao_proxy(std::span<const float> positions, std::span<const uint32_t> indices);

}  // namespace balla::render
