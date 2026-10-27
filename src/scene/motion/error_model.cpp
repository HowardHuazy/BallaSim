#include "scene/motion/error_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "core/camera/injective_domain.h"
#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"

namespace balla::scene::motion {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kSupportPx = 0.5;

double dist(const rot::V& a, const rot::V& b) { return norm(a - b); }

bool box_empty(const std::array<double, 6>& b) { return !(b[0] <= b[3] && b[1] <= b[4] && b[2] <= b[5]); }

// Distance from x to the box (0 inside); +inf for an empty box.
double box_near(const std::array<double, 6>& b, const rot::V& x) {
  if (box_empty(b)) return kInf;
  const double dx = std::max({b[0] - x.x, 0.0, x.x - b[3]}), dy = std::max({b[1] - x.y, 0.0, x.y - b[4]}),
               dz = std::max({b[2] - x.z, 0.0, x.z - b[5]});
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// The largest distance from x to the box (its farthest corner); 0 for an empty box.
double box_far(const std::array<double, 6>& b, const rot::V& x) {
  if (box_empty(b)) return 0.0;
  const double dx = std::max(std::fabs(x.x - b[0]), std::fabs(x.x - b[3])),
               dy = std::max(std::fabs(x.y - b[1]), std::fabs(x.y - b[4])),
               dz = std::max(std::fabs(x.z - b[2]), std::fabs(x.z - b[5]));
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// The box's diagonal; 0 for an empty box.
double box_diagonal(const std::array<double, 6>& b) {
  if (box_empty(b)) return 0.0;
  const double dx = b[3] - b[0], dy = b[4] - b[1], dz = b[5] - b[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double coeff(const std::vector<double>& d, size_t i) { return i < d.size() ? d[i] : 0.0; }

}  // namespace

double angular_gain(const render::Camera& cam) {
  if (cam.width <= 0 || cam.height <= 0 || !(cam.fx > 0.0) || !(cam.fy > 0.0)) {
    throw std::invalid_argument("angular_gain: invalid camera");
  }
  if (cam.model == render::CameraModelKind::kRadtan) {
    camera::RadtanCamera<double> c{cam.fx, cam.fy, cam.cx, cam.cy, coeff(cam.dist, 0), coeff(cam.dist, 1),
                                   coeff(cam.dist, 2), coeff(cam.dist, 3), coeff(cam.dist, 4), 0.0, 0.0};
    const camera::RadtanDomain dom = camera::radtan_injective_domain(c, cam.width, cam.height, kSupportPx);
    c.r_lim = dom.r_lim;
    // max sigma(diag(fx, fy) J_D) over a 33 x 25 grid spanning the support rectangle
    double smax = 0.0;
    for (int a = 0; a <= 32; ++a) {
      for (int b = 0; b <= 24; ++b) {
        const double u = -kSupportPx + (cam.width - 1 + 2 * kSupportPx) * a / 32.0;
        const double v = -kSupportPx + (cam.height - 1 + 2 * kSupportPx) * b / 24.0;
        Vec3<double> ray;
        if (!camera::unproject(c, Vec2<double>{u, v}, &ray)) {
          throw std::domain_error("angular_gain: unprojection failed");
        }
        const camera::Sym2<double> j = camera::distort_jacobian(c, Vec2<double>{ray.x, ray.y});
        const double m00 = cam.fx * j.xx, m01 = cam.fx * j.xy, m10 = cam.fy * j.xy, m11 = cam.fy * j.yy;
        // sigma_max^2 = largest eigenvalue of M^T M
        const double a00 = m00 * m00 + m10 * m10, a01 = m00 * m01 + m10 * m11, a11 = m01 * m01 + m11 * m11;
        const double mean = 0.5 * (a00 + a11), half = 0.5 * (a00 - a11);
        smax = std::max(smax, std::sqrt(mean + std::sqrt(half * half + a01 * a01)));
      }
    }
    return smax * (1.0 + dom.r_img * dom.r_img);
  }
  const camera::KannalaBrandtCamera<double> c{cam.fx, cam.fy, cam.cx, cam.cy, coeff(cam.dist, 0), coeff(cam.dist, 1),
                                              coeff(cam.dist, 2), coeff(cam.dist, 3), 0.0};
  const double theta_max = camera::kannala_brandt_injective_domain(c, cam.width, cam.height, kSupportPx).theta_img;
  double best = 1.0;  // both gains tend to 1 at theta = 0
  constexpr int kSteps = 4096;
  for (int i = 1; i <= kSteps; ++i) {
    const double th = theta_max * i / kSteps;
    best = std::max({best, camera::theta_distorted_derivative(c, th), camera::theta_distorted(c, th) / std::sin(th)});
  }
  if (!std::isfinite(best)) throw std::domain_error("angular_gain: unbounded Kannala-Brandt gain");
  return std::max(cam.fx, cam.fy) * best;
}

ErrorModel::ErrorModel(const MotionScene& scene)
    : scene_(scene), statics_(scene.static_vertices, scene.static_indices) {
  check_scale_refs(scene);  // before any oracle call: decode would refuse the first answer
  for (const MotionCamera& c : scene.cameras) {
    if (c.rig_body < 0 || c.rig_body >= static_cast<int32_t>(scene.bodies.size()) ||
        scene.bodies[static_cast<size_t>(c.rig_body)].role != Role::kRig) {
      throw std::invalid_argument("ErrorModel: a camera's rig_body is not a rig column");
    }
    if (!(c.gain_px_per_rad > 0.0) || !std::isfinite(c.gain_px_per_rad)) {
      throw std::invalid_argument("ErrorModel: camera gains must be finite and > 0");
    }
    gain_ = std::max(gain_, c.gain_px_per_rad);
  }
  for (size_t b = 0; b < scene.bodies.size(); ++b) {
    const MotionBody& mb = scene.bodies[b];
    if (!(mb.rho_m >= 0.0)) throw std::invalid_argument("ErrorModel: rho must be >= 0");
    if (mb.role == Role::kRig && std::none_of(scene.cameras.begin(), scene.cameras.end(), [&](const MotionCamera& c) {
          return c.rig_body == static_cast<int32_t>(b);
        })) {
      throw std::invalid_argument("ErrorModel: a rig column carries no camera");
    }
  }
}

void ErrorModel::weigh(PoseBatch& batch) const {
  const size_t n = batch.size(), nb = static_cast<size_t>(batch.bodies), nc = scene_.cameras.size();
  if (nb != scene_.bodies.size()) throw std::invalid_argument("ErrorModel::weigh: body count differs");
  batch.w_pos.assign(n * nb, 0.0);
  batch.w_rot.assign(n * nb, 0.0);
  batch.floor1.assign(n * nb, 0.0);
  batch.light_near.assign(n * nb, 0);
  const double G = gain_;
  std::vector<rot::V> o(nc);
  std::vector<double> floor(nc);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < nc; ++j) {
      const MotionCamera& cam = scene_.cameras[j];
      const size_t k = batch.at(i, cam.rig_body);
      o[j] = batch.p[k] + rotate(batch.q[k], cam.t_rig_cam_m);
      floor[j] = std::max(kDepthFloorM, cam.clip_start_m);
    }
    // the nearest body surface bound to x (bodies with a mesh other than `skip`), +inf without one
    auto nearest_body = [&](const rot::V& x, size_t skip) {
      double d = kInf;
      for (size_t b = 0; b < nb; ++b) {
        if (b == skip || scene_.bodies[b].role != Role::kBody || scene_.bodies[b].still ||
            !batch.present[batch.at(i, static_cast<int32_t>(b))]) {
          continue;
        }
        d = std::min(d, dist(batch.p[batch.at(i, static_cast<int32_t>(b))], x) - scene_.bodies[b].rho_m);
      }
      return d;
    };
    double d_recv = kInf;
    for (size_t j = 0; j < nc; ++j) d_recv = std::min(d_recv, std::max(box_near(scene_.scene_box, o[j]), floor[j]));
    for (size_t b = 0; b < nb; ++b) {
      const size_t k = batch.at(i, static_cast<int32_t>(b));
      if (!batch.present[k]) continue;
      const MotionBody& mb = scene_.bodies[b];
      double wp = 0.0, wr = 0.0;
      if (mb.role == Role::kBody) {
        double d = kInf;
        for (size_t j = 0; j < nc; ++j) d = std::min(d, std::max(dist(batch.p[k], o[j]) - mb.rho_m, floor[j]));
        wp = G / d;
        wr = G * mb.rho_m / d;
      } else if (mb.role == Role::kRig) {
        double d = kInf, ell = 0.0;
        for (size_t j = 0; j < nc; ++j) {
          if (scene_.cameras[j].rig_body != static_cast<int32_t>(b)) continue;
          const double ds = std::min(statics_(o[j]), nearest_body(o[j], nb));
          d = std::min(d, std::max(ds, floor[j]));
          ell = std::max(ell, norm(scene_.cameras[j].t_rig_cam_m));
        }
        wp = G / d;
        wr = G * (1.0 + ell / d);
      }
      uint8_t near = 0;
      for (const MotionLight& l : mb.lights) {
        if (!std::isfinite(d_recv)) break;
        const rot::V x = batch.p[k] + rotate(batch.q[k], l.offset_m);
        // A directional light has no position: turning it by dtheta moves the shadow of an occluder point P on a
        // receiver X by up to |X - P| dtheta (ENGINE.md, moving lights), and both lie in the scene box, so its D_max
        // also covers the box's diagonal wherever the lamp object sits.
        const double d_max = l.directional ? std::max(box_far(scene_.scene_box, x), box_diagonal(scene_.scene_box))
                                           : box_far(scene_.scene_box, x);
        double lambda = 0.0;
        if (l.casts_shadows) {
          const double d_occ = std::min(statics_(x), nearest_body(x, b));
          if (d_occ < kDepthFloorM) near = 1;
          lambda = d_max / std::max(d_occ, kDepthFloorM);
        }
        // s = 1 for spots and directional lights (on any carrier: their direction is the carrier's rotation) and for
        // shadow-casting lights on a body with a mesh (its own shadow turns with it)
        const double s = (l.spot || l.directional || (l.casts_shadows && mb.role == Role::kBody)) ? 1.0 : 0.0;
        wp = std::max(wp, G * (1.0 + lambda) / d_recv);
        wr = std::max(wr, G * ((1.0 + lambda) * norm(l.offset_m) + s * d_max) / d_recv);
      }
      batch.w_pos[k] = wp;
      batch.w_rot[k] = wr;
      batch.floor1[k] = wp * batch.dp_round[k] + wr * batch.dtheta_round[k];
      batch.light_near[k] = near;
    }
  }
}

}  // namespace balla::scene::motion
