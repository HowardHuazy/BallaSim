// P2-F F11a (plan p2f §4.9): the near-infrared albedo channel of the cluster shading. A camera c sees light group g
// through the albedo
//   rho_{c,g} = (1 - f_{c,g}) rho_vis + f_{c,g} rho_NIR,
// f the fraction of its photo-electrons from the near infrared (CameraView::nir_fraction; the fraction of a strobe at
// 850 nm is 1, a white LED's 0.03, DESIGN §7.2). Scenes whose f is the same for every (camera, group) fold the mix into
// the one albedo at level 0 (scene_data.cpp), at f = 0 leaving Phase 1's single channel untouched; the others run the
// two-channel path: (visible, NIR) texels, one more cluster sum (ClusterSums<kShadow, true>) and this mix.
#pragma once

#include "core/hd.h"
#include "render/scene_view.h"

namespace balla::render {

// (1 - f) vis + f nir: exactly vis at f = 0 and exactly nir at f = 1 (finite vis and nir).
BALLA_HD inline float nir_mix(float vis, float nir, float f) { return (1.0f - f) * vis + f * nir; }

// The mean albedo of a cluster for light group g: the mean of its members' visible lookups (total.t[6], or the
// material's albedo untextured), mixed with the NIR ones (total.t[7], or the material's nir_albedo) by the camera's
// f of the group. Sums is a ClusterSums (shade.h); without the NIR term it is the visible mean.
template <class Sums>
BALLA_HD inline float cluster_albedo(const SceneView& s, uint32_t cam, const MaterialView& m, bool textured,
                                     float inv_c, const Sums& total, int32_t g) {
  const float vis = textured ? inv_c * total.t[6] : m.albedo;
  if constexpr (Sums::kNir) {
    const float nir = textured ? inv_c * total.t[7] : m.nir_albedo;
    return nir_mix(vis, nir, s.cameras[cam].nir_fraction[g]);
  } else {
    return vis;
  }
}

}  // namespace balla::render
