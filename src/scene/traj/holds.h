// Static-interval (hold) detection and hold constraints (ballasim.traj.holds; DESIGN §5.4 step 9, §5.6, M2 Lemma B1),
// both hold rules ('sphere' and 'image', F090), tie maps and hold groups, bitwise as the Python.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "scene/rotation.h"
#include "scene/sparse.h"
#include "scene/traj/basis.h"

namespace balla::scene::traj {

// hold_rule 'image': per camera the pixel Jacobian's largest singular value and corner r^2, per input sample a depth
// bound; cameras_body (the rig's camera centres in its frame) or empty for another body.
struct HoldView {
  std::vector<double> focal_px, corner_r2, depth_m;
  std::vector<rot::V> cameras_body;
  bool has_cameras = false;

  std::vector<double> speed(std::span<const rot::Q> q, std::span<const rot::V> v, std::span<const rot::V> omega,
                            rot::V center, double radius, std::span<const int64_t> idx) const;
  double excursion(std::span<const rot::Q> q, std::span<const rot::V> p, rot::V center, double radius,
                   std::span<const int64_t> idx) const;
};

struct StaticInterval {
  int64_t i0, i1, t_begin_ns, t_end_ns;
  double max_displacement;
};

struct RejectedCandidate {
  int64_t t_begin_ns, t_end_ns;
  double max_displacement;
  std::string reason;
};

struct HoldResult {
  std::vector<StaticInterval> accepted;
  std::vector<RejectedCandidate> rejected;
};

double sphere_displacement(rot::Q qa, rot::V pa, rot::Q qb, rot::V pb, rot::V center, double radius);
double sphere_speed(rot::V v, rot::V omega, rot::Q q, rot::V center, double radius);

struct StaticOptions {
  double max_disp = 1e-3;
  int64_t min_duration_ns = 30'000'000;
  double moving_speed = 0.02;
  double edge_speed_floor = 1e-3;
  double edge_noise_factor = 3.0;
  bool split = false;
  int64_t min_piece_ns = 0;
};

// detect_static with the default excursion (bounding-sphere displacement from the robust mean pose) or `excursion`.
HoldResult detect_static(std::span<const int64_t> t_ns, std::span<const rot::Q> q, std::span<const rot::V> p,
                         std::span<const double> speed, rot::V center, double radius, const StaticOptions& o,
                         const std::function<double(int64_t, int64_t)>& excursion = {});

// detect_static_image on the pre-fit's values (q, p, v, omega) at the kept samples, idx their input indices.
HoldResult detect_static_image(std::span<const int64_t> t_ns, std::span<const rot::Q> q, std::span<const rot::V> p,
                               std::span<const rot::V> v, std::span<const rot::V> omega, const HoldView& view,
                               std::span<const int64_t> idx, rot::V center, double radius, int64_t min_duration_ns,
                               double max_disp_px, int64_t min_piece_ns);

// TieMap: control c -> free parameter f(c); groups (c0, c1) inclusive share one.
struct TieMap {
  int64_t num_ctrl = 0, num_free = 0;
  std::vector<std::pair<int64_t, int64_t>> groups;
  std::vector<int64_t> free_index;

  TieMap() = default;
  TieMap(int64_t n, std::vector<std::pair<int64_t, int64_t>> g);
  // G (block N x block N_free) with x = G x_free (sp.csr_matrix from COO).
  sparse::Matrix matrix(int block) const;
};

// hold_groups: tie groups for closed hold intervals and the exact still intervals [t_begin, t_end).
std::pair<std::vector<std::pair<int64_t, int64_t>>, std::vector<std::pair<int64_t, int64_t>>> hold_groups(
    const KnotGrid& grid, std::vector<std::pair<int64_t, int64_t>> intervals);

}  // namespace balla::scene::traj
