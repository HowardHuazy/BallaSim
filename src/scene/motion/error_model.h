// The image-space error model of the Blender motion session (docs/design/phase3/blender.md §4.7.2): how many pixels a
// pose error of a body, a camera rig or a light carrier moves anything in any simulated camera, and the float32
// floor of that error at each sample.
//
//   bodies:           e_px = G (|dc| + rho |dtheta|) / d_min
//   cameras and rigs: e_px = G (|dtheta| + (|dp| + l |dtheta|) / d_scene)
//   light carriers:   e_px = G ((1 + lambda_max) (|dp| + l_L |dtheta|) + s D_max |dtheta|) / d_recv
//
// Each is w_pos |dp| + w_rot |dtheta| for the body at that sample; distances are floored per camera at
// max(5 cm, clip_start). A body that has a mesh and carries lights takes the larger of both weights. s = 1 for spot
// and directional lights on any carrier and for shadow-casting lights on a body with a mesh; a directional light's
// D_max is at least the scene box's diagonal (the longest occluder-to-receiver distance).
#pragma once

#include <cstdint>
#include <vector>

#include "render/bundle.h"
#include "scene/motion/poses.h"
#include "scene/motion/static_distance.h"

namespace balla::scene::motion {

inline constexpr double kDepthFloorM = 0.05;  // SCENE.md's 5 cm depth floor

// G_j, the camera's angular pixel gain (px per rad of ray rotation) maximized over its image (pixel centres +- 0.5 px):
// radtan f max|J_D| (1 + r_max^2), kb max(fx, fy) max over theta <= theta_max of max(theta_d'(theta),
// theta_d(theta) / sin theta) (the radial and tangential gains of core/camera/kannala_brandt.h). Throws
// std::domain_error for a camera that is not injective over its image.
double angular_gain(const render::Camera& cam);

class ErrorModel {
 public:
  // Keeps a reference to `scene`, which must outlive the model.
  explicit ErrorModel(const MotionScene& scene);
  double gain() const { return gain_; }  // G = max_j G_j
  const MotionScene& scene() const { return scene_; }
  const StaticDistance& statics() const { return statics_; }
  // Fills w_pos, w_rot, floor1 and light_near of every present (sample, body) of a decoded batch.
  void weigh(PoseBatch& batch) const;
  // floor_px of a prediction with stencil gain g = 1 + sum |w_k| from samples whose largest floor1 is f1 (§4.7.2).
  static double floor_px(double f1, double stencil_gain) { return stencil_gain * f1; }

 private:
  const MotionScene& scene_;
  StaticDistance statics_;
  double gain_ = 0.0;
};

}  // namespace balla::scene::motion
