// Cycles timing, motion steps and the sub-render rules (cycles_timing.h; docs/design/phase3/frames.md §4.2, §4.3).
#include "frames/cycles_timing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/camera/injective_domain.h"
#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "core/math/quat.h"
#include "core/math/vec.h"
#include "core/traj/spline.h"
#include "core/traj/spline_build.h"

namespace balla::frames {

namespace {

__extension__ typedef __int128 i128;

using Vec3d = Vec3<double>;
using Quatd = Quat<double>;
using Mat3d = Mat3<double>;
using scene::motion::TimeMap;

// round_half_even(n / d), d > 0.
i128 div_round_even(i128 n, i128 d) {
  i128 q = n / d, r = n % d;
  if (r < 0) {
    q -= 1;
    r += d;
  }
  if (2 * r > d || (2 * r == d && (q & 1) != 0)) q += 1;
  return q;
}

// Nearest int64 of a finite long double, ties to even (the default rounding mode).
int64_t round_even(long double x) {
  const long double r = std::nearbyint(x);
  if (!(r >= -0x1p63L && r < 0x1p63L)) throw std::range_error("frames: Cycles timing out of the int64 range");
  return static_cast<int64_t>(r);
}

int64_t to_i64(i128 v) {
  if (v > INT64_MAX || v < INT64_MIN) throw std::range_error("frames: Cycles timing out of the int64 range");
  return static_cast<int64_t>(v);
}

std::string num(double v) {
  char buf[48];
  std::snprintf(buf, sizeof buf, "%.6g", v);
  return buf;
}

// P_ns of the TimeMap as a long double: 1e9 B_num / (fps B_den).
long double period_ld(const TimeMap& t) {
  return 1e9L * static_cast<long double>(t.base_num()) /
         (static_cast<long double>(t.fps()) * static_cast<long double>(t.base_den()));
}

float to_f32(long double v, const char* what) {
  if (!(v >= 0.0L && v <= static_cast<long double>(std::numeric_limits<float>::max()))) {
    throw std::invalid_argument(std::string("frames: Cycles ") + what + " is out of the float32 range");
  }
  return static_cast<float>(v);
}

// ---- Poses ----

// A trajectory (world pose of its body) or a fixed world pose. A trajectory's spline nodes are built only for the
// controls of the segments that [lo, hi] meets, so the cost does not grow with the trajectory's length. The poses are
// bitwise those of the whole spline: node c depends on controls c - 1 and c only (build_spline_nodes), segment s reads
// node s's q and p and the increments of nodes s + 1 ... s + K - 1 (traj::evaluate_u), and a time in [lo, hi] lands
// on the segment the whole spline uses (the built segments end at t0 + (s1 + 1) h > hi, or at the spline's own end).
struct Track {
  std::vector<traj::SplineNode<double>> nodes;
  int32_t segments = 0, h_ns = 1, order = 6;
  int64_t t0 = 0, begin = 0, end = 0;  // t0 of the built segments; the trajectory's span (within its spline)
  bool fixed = true;
  Quatd q{1.0, 0.0, 0.0, 0.0};
  Vec3d p{0.0, 0.0, 0.0};

  void pose(int64_t t, Quatd* q_out, Vec3d* p_out) const {
    if (fixed) {
      *q_out = q;
      *p_out = p;
      return;
    }
    if (t < begin || t > end) {
      throw FramesError("FRM-SPAN", "frames motion steps: time " + std::to_string(t) +
                                        " ns is outside a trajectory's span [" + std::to_string(begin) + ", " +
                                        std::to_string(end) + "] ns");
    }
    const traj::SplineView<double> view{nodes.data(), segments, h_ns, 1e9 / static_cast<double>(h_ns)};
    traj::SegmentTime st{};
    if (!traj::locate(t, t0, h_ns, segments, &st)) {
      throw std::logic_error("frames motion steps: a pose time outside the built spline segments");
    }
    const traj::Kinematics<double> k = order == 4 ? traj::evaluate<4>(view, st) : traj::evaluate<6>(view, st);
    *q_out = k.q;
    *p_out = k.p;
  }
};

// The trajectory's spline over the segments that [lo, hi] meets (lo and hi clamped to the span).
Track track_of(const render::Trajectory& tr, int64_t lo, int64_t hi) {
  // One spline's segments: a piecewise trajectory (C1 "pieces", rendered since BL-5b) is refused rather than read as
  // one spline over every piece's controls.
  if (!tr.pieces.empty()) {
    throw FramesError("FRM-VALUE", "frames motion steps: piecewise trajectories (C1 'pieces') are not supported yet");
  }
  const auto n = static_cast<int64_t>(tr.rot.size() / 4);
  const int64_t all = n - tr.order + 1;
  if ((tr.order != 4 && tr.order != 6) || all < 1 || tr.h_ns < 1 || tr.h_ns > INT32_MAX ||
      tr.pos.size() / 3 != tr.rot.size() / 4 || tr.span_begin_ns > tr.span_end_ns) {
    throw std::invalid_argument("frames motion steps: invalid trajectory");
  }
  const auto h = static_cast<int64_t>(tr.h_ns);
  Track t;
  // The span within the spline's own domain [t0, t0 + all h]: a pose outside it is FRM-SPAN.
  t.begin = std::max(tr.span_begin_ns, tr.t0_ns);
  const i128 spline_end = static_cast<i128>(tr.t0_ns) + static_cast<i128>(all) * h;
  t.end = static_cast<int64_t>(std::min<i128>(tr.span_end_ns, spline_end));
  const auto segment = [&](int64_t x) {  // floor((clamp(x) - t0) / h), clamped to [0, all)
    if (t.begin <= t.end) x = std::clamp(x, t.begin, t.end);
    const i128 d = static_cast<i128>(x) - tr.t0_ns;
    i128 s = d / h;
    if (d < 0 && d % h != 0) s -= 1;
    return static_cast<int64_t>(std::clamp<i128>(s, 0, all - 1));
  };
  const int64_t s0 = segment(lo), s1 = segment(hi);
  const int64_t controls = s1 - s0 + tr.order;
  std::vector<Quatd> qs(static_cast<size_t>(controls));
  std::vector<Vec3d> ps(static_cast<size_t>(controls));
  for (int64_t c = 0; c < controls; ++c) {
    const auto i = static_cast<size_t>(s0 + c);
    qs[static_cast<size_t>(c)] = {tr.rot[4 * i], tr.rot[4 * i + 1], tr.rot[4 * i + 2], tr.rot[4 * i + 3]};
    ps[static_cast<size_t>(c)] = {tr.pos[3 * i], tr.pos[3 * i + 1], tr.pos[3 * i + 2]};
  }
  t.nodes = traj::build_spline_nodes(qs, ps, {0.0, 0.0, 0.0});
  t.segments = static_cast<int32_t>(s1 - s0 + 1);
  t.h_ns = static_cast<int32_t>(h);
  t.order = tr.order;
  t.t0 = to_i64(static_cast<i128>(tr.t0_ns) + static_cast<i128>(s0) * h);
  t.fixed = false;
  return t;
}

Mat3d rot_of(const std::array<double, 16>& m) {
  return {{{m[0], m[1], m[2]}, {m[4], m[5], m[6]}, {m[8], m[9], m[10]}}};
}

Track fixed_track(const std::array<double, 16>& m) {
  Track t;
  t.q = from_matrix(rot_of(m));
  t.p = {m[3], m[7], m[11]};
  return t;
}

// The camera model with its injective domain (support 0.5 px: the image boundary).
struct Projector {
  bool kb = false;
  double f_max = 0.0;  // max(fx, fy): the image scale near the centre, px per unit of x/z
  camera::RadtanCamera<double> rt{};
  camera::KannalaBrandtCamera<double> kbc{};

  explicit Projector(const render::Camera& c) {
    const auto dist = [&](size_t i) { return i < c.dist.size() ? c.dist[i] : 0.0; };
    kb = c.model == render::CameraModelKind::kKannalaBrandt;
    f_max = std::max(std::fabs(c.fx), std::fabs(c.fy));
    if (kb) {
      kbc = {c.fx, c.fy, c.cx, c.cy, dist(0), dist(1), dist(2), dist(3), 0.0};
      kbc.theta_lim = camera::kannala_brandt_injective_domain(kbc, c.width, c.height, 0.5).theta_lim;
    } else {
      rt = {c.fx, c.fy, c.cx, c.cy, dist(0), dist(1), dist(2), dist(3), dist(4), 0.0, 0.0};
      rt.r_lim = camera::radtan_injective_domain(rt, c.width, c.height, 0.5).r_lim;
    }
  }
  bool project(Vec3d x, Vec2<double>* px) const {
    return kb ? camera::project(kbc, x, px) : camera::project(rt, x, px);
  }
  bool unproject(Vec2<double> px, Vec3d* ray) const {
    return kb ? camera::unproject(kbc, px, ray) : camera::unproject(rt, px, ray);
  }
};

double dot(const Quatd& a, const Quatd& b) { return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z; }

Quatd nlerp(const Quatd& a, const Quatd& b, double u) {
  return normalized(Quatd{(1.0 - u) * a.w + u * b.w, (1.0 - u) * a.x + u * b.x, (1.0 - u) * a.y + u * b.y,
                          (1.0 - u) * a.z + u * b.z});
}

// Cycles' quat_interpolate (util/transform.h ~407-431), in fp64; q1 and q2 are sign-matched.
Quatd interpolate(const Quatd& q1, const Quatd& q2, double u, QuatInterp model) {
  const double cos_t = dot(q1, q2);
  if (model == QuatInterp::kGpuRt || cos_t > 0.9995) return nlerp(q1, q2, u);
  const double theta = std::acos(std::clamp(cos_t, -1.0, 1.0));
  const Quatd perp = normalized(Quatd{q2.w - q1.w * cos_t, q2.x - q1.x * cos_t, q2.y - q1.y * cos_t,
                                      q2.z - q1.z * cos_t});
  const double c = std::cos(theta * u), s = std::sin(theta * u);
  return {q1.w * c + perp.w * s, q1.x * c + perp.x * s, q1.y * c + perp.y * s, q1.z * c + perp.z * s};
}

// Everything motion_chord_px needs from a probe, with the splines built over [lo, hi] (every pose time of the call).
struct ProbeCtx {
  const MotionProbe& probe;
  Track rig, body;
  Quatd q_rc{1.0, 0.0, 0.0, 0.0};
  Vec3d p_rc{0.0, 0.0, 0.0};
  Projector proj;
  Vec3d pivot;

  ProbeCtx(const MotionProbe& pr, int64_t lo, int64_t hi)
      : probe(pr), proj(checked(pr).cameras[pr.cam]), pivot{pr.pivot[0], pr.pivot[1], pr.pivot[2]} {
    const render::SceneBundle& b = *pr.bundle;
    const render::Camera& c = b.cameras[pr.cam];
    if (c.rig_trajectory < 0 || static_cast<size_t>(c.rig_trajectory) >= b.trajectories.size()) {
      throw std::invalid_argument("frames motion steps: the camera's rig trajectory is out of range");
    }
    rig = track_of(b.trajectories[static_cast<size_t>(c.rig_trajectory)], lo, hi);
    q_rc = from_matrix(rot_of(c.T_rig_cam));
    p_rc = {c.T_rig_cam[3], c.T_rig_cam[7], c.T_rig_cam[11]};
    if (pr.body >= 0) {
      const render::Body& bd = b.bodies[static_cast<size_t>(pr.body)];
      if (bd.trajectory && (*bd.trajectory < 0 || static_cast<size_t>(*bd.trajectory) >= b.trajectories.size())) {
        throw std::invalid_argument("frames motion steps: the body's trajectory is out of range");
      }
      body = bd.trajectory ? track_of(b.trajectories[static_cast<size_t>(*bd.trajectory)], lo, hi)
                           : fixed_track(bd.T_world_body);
    }
  }
  static const render::SceneBundle& checked(const MotionProbe& pr) {
    if (pr.bundle == nullptr) throw std::invalid_argument("frames motion steps: the probe has no bundle");
    if (pr.cam >= pr.bundle->cameras.size()) throw std::invalid_argument("frames motion steps: camera out of range");
    if (pr.body < -1 || pr.body >= static_cast<int32_t>(pr.bundle->bodies.size())) {
      throw std::invalid_argument("frames motion steps: body out of range");
    }
    return *pr.bundle;
  }

  // World pose of the camera (T_world_rig T_rig_cam) and of the body's pivot frame at t.
  void cam_pose(int64_t t, Quatd* q, Vec3d* p) const {
    Quatd qr;
    Vec3d pr;
    rig.pose(t, &qr, &pr);
    *q = qr * q_rc;
    *p = rotate(qr, p_rc) + pr;
  }
  void body_pose(int64_t t, Quatd* q, Vec3d* p) const {  // rotation, and the pivot's world position
    if (probe.body < 0) {
      *q = {1.0, 0.0, 0.0, 0.0};
      *p = {0.0, 0.0, 0.0};
      return;
    }
    Vec3d o;
    body.pose(t, q, &o);
    *p = rotate(*q, pivot) + o;
  }
  // The world point of probe point X under a body pose (rotation q, pivot position p).
  Vec3d world(const std::array<double, 3>& x, const Quatd& q, const Vec3d& p) const {
    const Vec3d xv{x[0], x[1], x[2]};
    return probe.body < 0 ? xv : rotate(q, xv - pivot) + p;
  }
};

// The rig and body poses at the node times, sign-matched as Cycles' transform_motion_decompose matches them.
struct NodePoses {
  std::vector<int64_t> t;
  std::vector<Quatd> qc, qb;
  std::vector<Vec3d> pc, pb;
};

NodePoses node_poses(const ProbeCtx& ctx, const std::vector<float>& nodes, const TimeMap& time) {
  NodePoses np;
  const size_t n = nodes.size();
  np.t.resize(n);
  np.qc.resize(n);
  np.qb.resize(n);
  np.pc.resize(n);
  np.pb.resize(n);
  for (size_t i = 0; i < n; ++i) {
    np.t[i] = time.to_ns(nodes[i]);
    ctx.cam_pose(np.t[i], &np.qc[i], &np.pc[i]);
    ctx.body_pose(np.t[i], &np.qb[i], &np.pb[i]);
    if (i > 0) {
      if (dot(np.qc[i - 1], np.qc[i]) < 0.0) np.qc[i] = {-np.qc[i].w, -np.qc[i].x, -np.qc[i].y, -np.qc[i].z};
      if (dot(np.qb[i - 1], np.qb[i]) < 0.0) np.qb[i] = {-np.qb[i].w, -np.qb[i].x, -np.qb[i].y, -np.qb[i].z};
    }
  }
  return np;
}

// The camera-frame positions of the probe points at t; true when one of them lies in front of the camera.
bool cam_points(const ProbeCtx& ctx, int64_t t, std::vector<Vec3d>* out) {
  Quatd qc, qb;
  Vec3d pc, pb;
  ctx.cam_pose(t, &qc, &pc);
  ctx.body_pose(t, &qb, &pb);
  bool front = false;
  for (size_t i = 0; i < ctx.probe.points.size(); ++i) {
    (*out)[i] = rotate_inverse(qc, ctx.world(ctx.probe.points[i], qb, pb) - pc);
    front = front || (*out)[i].z > 0.0;
  }
  return front;
}

// The image distance (px) between camera-frame points x and y: the distance of their projections when both lie in
// the injective domain, else the first-order bound f_max |y - x| / max(min(x.z, y.z), z_near) (z_near when finite;
// 0 when that depth is <= 0). See motion_chord_px.
double image_distance(const ProbeCtx& ctx, const Vec3d& x, const Vec3d& y) {
  Vec2<double> a{}, b{};
  if (ctx.proj.project(x, &a) && ctx.proj.project(y, &b)) return std::hypot(a.x - b.x, a.y - b.y);
  double z = std::min(x.z, y.z);
  if (std::isfinite(ctx.probe.z_near_m)) z = std::max(z, ctx.probe.z_near_m);
  return z > 0.0 ? ctx.proj.f_max * norm(y - x) / z : 0.0;
}

// v_b: the largest finite-difference image speed (px/s) of the probe points over 256 equal steps of [t_a, t_b]
// (image_distance; steps where every point lies behind the camera add nothing).
double image_speed(const ProbeCtx& ctx, int64_t t_a, int64_t t_b) {
  if (t_b <= t_a) return 0.0;
  constexpr int kSteps = 256;
  const size_t n = ctx.probe.points.size();
  std::vector<Vec3d> prev(n), cur(n);
  bool prev_front = false;
  int64_t t_prev = 0;
  double v = 0.0;
  for (int k = 0; k <= kSteps; ++k) {
    const int64_t t = t_a + static_cast<int64_t>((static_cast<i128>(t_b) - t_a) * k / kSteps);
    const bool front = cam_points(ctx, t, &cur);
    if (k > 0 && t > t_prev && (front || prev_front)) {
      const double dt_s = 1e-9 * static_cast<double>(static_cast<i128>(t) - t_prev);
      for (size_t i = 0; i < n; ++i) v = std::max(v, image_distance(ctx, prev[i], cur[i]) / dt_s);
    }
    std::swap(prev, cur);
    prev_front = front;
    t_prev = t;
  }
  return v;
}

double ulp32(float x) {
  x = std::fabs(x);
  return static_cast<double>(std::nextafter(x, std::numeric_limits<float>::infinity())) - static_cast<double>(x);
}

}  // namespace

// ---- Rasters, bands and sub-renders ----

Raster blender_raster(const RowTiming& timing, RowBand band, int32_t s, int32_t margin_rows) {
  if (timing.height < 1) throw std::invalid_argument("frames Cycles timing: height must be >= 1");
  if (band.rows < 0) band.rows = timing.height - band.row0;
  if (band.row0 < 0 || band.rows < 1 || band.row0 > timing.height - band.rows) {
    throw std::invalid_argument("frames Cycles timing: the band must hold read rows inside [0, height)");
  }
  if (s < 1 || margin_rows < 0) throw std::invalid_argument("frames Cycles timing: s must be >= 1 and margin >= 0");
  const int64_t rows = static_cast<int64_t>(s) * (static_cast<int64_t>(band.rows) + 2 * int64_t{margin_rows});
  if (rows > std::numeric_limits<int32_t>::max()) throw std::invalid_argument("frames Cycles timing: raster too tall");
  const long double delta = timing.shutter == Shutter::kRolling ? static_cast<long double>(timing.row_dt_ns) : 0.0L;
  const long double offset =
      static_cast<long double>(band.row0) * delta -
      (static_cast<long double>(s) * margin_rows + (static_cast<long double>(s) - 1.0L) / 2.0L) * delta / s;
  return {static_cast<int32_t>(rows), static_cast<double>(delta / s), static_cast<double>(offset)};
}

std::vector<RowBand> row_bands(int32_t height, int32_t n_band) {
  if (height < 1 || n_band < 1 || n_band > height) {
    throw std::invalid_argument("frames Cycles timing: n_band must lie in [1, height]");
  }
  std::vector<RowBand> out;
  out.reserve(static_cast<size_t>(n_band));
  for (int32_t b = 0; b < n_band; ++b) {
    const auto r0 = static_cast<int32_t>(int64_t{b} * height / n_band);
    const auto r1 = static_cast<int32_t>(int64_t{b + 1} * height / n_band);
    out.push_back({r0, r1 - r0});
  }
  return out;
}

SubWindow sub_window(int64_t exposure_ns, int32_t n_sub, int32_t j) {
  if (exposure_ns < 0 || n_sub < 1 || j < 0 || j >= n_sub) {
    throw std::invalid_argument("frames Cycles timing: sub-render j must lie in [0, n_sub) and T >= 0");
  }
  const auto lo = static_cast<int64_t>(static_cast<i128>(j) * exposure_ns / n_sub);
  const auto hi = static_cast<int64_t>(static_cast<i128>(j + 1) * exposure_ns / n_sub);
  return {lo, hi - lo};
}

// ---- Cycles timing ----

CyclesTiming cycles_timing(const TimeMap& time, const RowTiming& timing, const Raster& raster, int64_t sub_first_soe_ns,
                           int64_t sub_exposure_ns) {
  if (sub_exposure_ns < 0) throw std::invalid_argument("frames Cycles timing: the exposure must be >= 0");
  if (raster.rows < 1 || !(raster.row_dt_ns >= 0.0) || !std::isfinite(raster.row_dt_ns) ||
      !std::isfinite(raster.first_row_offset_ns)) {
    throw std::invalid_argument("frames Cycles timing: invalid raster");
  }
  const long double p = period_ld(time);
  const long double t1 = static_cast<long double>(sub_exposure_ns);
  const long double readout = static_cast<long double>(raster.rows) * static_cast<long double>(raster.row_dt_ns);
  const long double t_b0 =
      static_cast<long double>(sub_first_soe_ns) + static_cast<long double>(raster.first_row_offset_ns);
  // delta_t = round(delta_c T' / (b - a)) = round(128 T' / 65025), exactly.
  const auto delta_t = static_cast<int64_t>(div_round_even(static_cast<i128>(sub_exposure_ns) * 128, 65025));
  const long double x = t1 * 65536.0L / 65025.0L;  // T' / (b - a), ns

  CyclesTiming ct;
  ct.rolling = timing.shutter == Shutter::kRolling && readout > 0.0L;
  long double t_c = 0.0L;
  float d32 = 1.0f, s32 = 0.0f;
  if (ct.rolling) {
    ct.motion_blur = true;  // also for T' = 0: rows are time-shifted only with motion blur on
    s32 = to_f32((x + readout) / p, "shutter");
    if (!(s32 > 0.0f)) throw std::invalid_argument("frames Cycles timing: the rolling shutter underflows float32");
    const long double d = sub_exposure_ns == 0 ? 0.0L : std::min(1.0L, x / (static_cast<long double>(s32) * p));
    d32 = static_cast<float>(d);
    t_c = t_b0 + t1 / 2.0L + static_cast<long double>(raster.row_dt_ns) * (raster.rows / 2.0L - 1.0L) +
          static_cast<long double>(delta_t);
  } else if (sub_exposure_ns > 0) {
    ct.motion_blur = true;
    s32 = to_f32(x / p, "shutter");
    if (!(s32 > 0.0f)) throw std::invalid_argument("frames Cycles timing: the shutter underflows float32");
    t_c = t_b0 + static_cast<long double>(sub_exposure_ns / 2) + static_cast<long double>(delta_t);
  } else {
    t_c = t_b0;  // motion blur off: Cycles renders the frame's own time
  }
  ct.shutter_frames = s32;
  ct.rolling_duration = d32;

  const int64_t t_req = round_even(t_c);
  int32_t frame = 0;
  float sub = 0.0f;
  time.request(t_req, &frame, &sub);
  ct.frame = frame;
  ct.subframe = sub;
  ct.expected_c = TimeMap::realized(frame, sub);
  ct.expected_ctime = time.reported(ct.expected_c);
  const long double ds = static_cast<long double>(d32) * static_cast<long double>(s32);  // exact: 24 + 24 bits
  const int64_t delta_r = ct.motion_blur ? round_even(kDeltaC * ds * p) : 0;
  ct.realized_center_ns = to_i64(static_cast<i128>(time.to_ns(ct.expected_c)) - delta_r);
  ct.snap_shift_ns = to_i64(static_cast<i128>(ct.realized_center_ns) - t_req + (ct.motion_blur ? delta_t : 0));
  ct.realized_row_exposure_ns = ct.motion_blur ? static_cast<double>(65025.0L / 65536.0L * ds * p) : 0.0;
  ct.q_ns = time.resolution_ns(frame);
  if (ct.q_ns > 1000.0 || (sub_exposure_ns > 0 && ct.q_ns > 0.01 * static_cast<double>(sub_exposure_ns))) {
    ct.warnings.push_back({"FRM-TIME-RES", "frames: Blender's time resolution q = " + num(ct.q_ns) +
                                               " ns at frame " + std::to_string(frame) +
                                               " exceeds 1 us or 1 % of the exposure " +
                                               std::to_string(sub_exposure_ns) + " ns"});
  }
  return ct;
}

std::vector<float> node_times(float c, float shutter_frames, int32_t steps) {
  if (steps < 1 || steps > 7) throw std::invalid_argument("frames: motion steps must lie in [1, 7]");
  if (!std::isfinite(c) || !(shutter_frames >= 0.0f) || !std::isfinite(shutter_frames)) {
    throw std::invalid_argument("frames: node times need a finite frame and shutter >= 0");
  }
  const int32_t n = (2 << (steps - 1)) + 1;
  const auto denom = static_cast<float>(n - 1);
  std::vector<float> out(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const float twice = 2.0f * static_cast<float>(i);
    const float rel = twice / denom - 1.0f;
    if (rel == 0.0f) {  // the centre: the scene's own frame, no frame_set
      out[static_cast<size_t>(i)] = c;
      continue;
    }
    const float half = (rel * shutter_frames) * 0.5f;
    const float t = c + half;
    const float fl = std::floor(t);
    if (!(std::fabs(fl) < 2147483520.0f)) throw std::range_error("frames: node time out of Blender's frame range");
    const auto frame = static_cast<int32_t>(fl);
    const float subframe = t - static_cast<float>(frame);
    out[static_cast<size_t>(i)] = TimeMap::realized(frame, subframe);
  }
  return out;
}

// ---- Motion steps ----

MotionProbe motion_probe(const render::SceneBundle& b, uint16_t cam, int32_t body, const std::vector<float>& depth,
                         int64_t t_mid_ns) {
  MotionProbe pr;
  pr.bundle = &b;
  pr.cam = cam;
  pr.body = body;
  const ProbeCtx ctx(pr, t_mid_ns, t_mid_ns);  // validates the indices
  double z = std::numeric_limits<double>::infinity();
  for (const float d : depth) {
    if (std::isfinite(d) && d > 0.0f) z = std::min(z, static_cast<double>(d));
  }
  pr.z_near_m = std::isfinite(z) ? z : std::numeric_limits<double>::quiet_NaN();
  if (body >= 0) {
    const std::vector<float>& v = b.bodies[static_cast<size_t>(body)].mesh.positions;
    if (v.size() < 3) return pr;
    std::array<double, 3> lo{}, hi{};
    for (int a = 0; a < 3; ++a) lo[a] = hi[a] = static_cast<double>(v[static_cast<size_t>(a)]);
    for (size_t i = 0; i + 2 < v.size(); i += 3) {
      for (size_t a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], static_cast<double>(v[i + a]));
        hi[a] = std::max(hi[a], static_cast<double>(v[i + a]));
      }
    }
    for (int k = 0; k < 8; ++k) {
      pr.points.push_back({(k & 1) ? hi[0] : lo[0], (k & 2) ? hi[1] : lo[1], (k & 4) ? hi[2] : lo[2]});
    }
    return pr;
  }
  if (!std::isfinite(z)) return pr;
  const render::Camera& c = b.cameras[cam];
  Quatd q;
  Vec3d p;
  ctx.cam_pose(t_mid_ns, &q, &p);
  const double w1 = c.width - 1, h1 = c.height - 1;
  for (const Vec2<double> px : {Vec2<double>{0.0, 0.0}, Vec2<double>{w1, 0.0}, Vec2<double>{0.0, h1},
                                Vec2<double>{w1, h1}, Vec2<double>{0.5 * w1, 0.5 * h1}}) {
    Vec3d ray{};
    if (!ctx.proj.unproject(px, &ray) || !(ray.z > 0.0)) continue;
    const Vec3d xw = rotate(q, (z / ray.z) * ray) + p;
    pr.points.push_back({xw.x, xw.y, xw.z});
  }
  return pr;
}

namespace {

// motion_chord_px on a built context (points present, S > 0).
double chord(const ProbeCtx& ctx, int64_t t_a, int64_t t_b, float c, float shutter_frames, const TimeMap& time,
             int32_t steps, QuatInterp model) {
  const MotionProbe& probe = ctx.probe;
  const NodePoses np = node_poses(ctx, node_times(c, shutter_frames, steps), time);
  const QuatInterp models[2] = {model == QuatInterp::kGpuRt ? QuatInterp::kGpuRt : QuatInterp::kCpu,
                                QuatInterp::kGpuRt};
  const int n_models = model == QuatInterp::kWorst ? 2 : 1;
  const size_t n_points = probe.points.size();
  std::vector<Vec3d> truth(n_points);
  double err = 0.0;
  for (size_t i = 0; i + 1 < np.t.size(); ++i) {
    const int64_t ta = np.t[i], tb = np.t[i + 1];
    if (tb <= ta) continue;  // nodes that float32 merged
    const i128 span = static_cast<i128>(tb) - ta;
    for (int k = 0; k < 16; ++k) {
      const int64_t t = ta + static_cast<int64_t>(span * k / 16);
      if (t < t_a || t > t_b) continue;
      if (!cam_points(ctx, t, &truth)) continue;  // the probe lies wholly behind the camera: nothing is visible
      const double u = static_cast<double>(static_cast<i128>(t) - ta) / static_cast<double>(span);
      for (int m = 0; m < n_models; ++m) {
        const Quatd qc_i = interpolate(np.qc[i], np.qc[i + 1], u, models[m]);
        const Quatd qb_i = interpolate(np.qb[i], np.qb[i + 1], u, models[m]);
        const Vec3d pc_i = (1.0 - u) * np.pc[i] + u * np.pc[i + 1];
        const Vec3d pb_i = (1.0 - u) * np.pb[i] + u * np.pb[i + 1];
        for (size_t j = 0; j < n_points; ++j) {
          const Vec3d interp = rotate_inverse(qc_i, ctx.world(probe.points[j], qb_i, pb_i) - pc_i);
          err = std::max(err, image_distance(ctx, truth[j], interp));
        }
      }
    }
  }
  return err;
}

// [lo, hi]: every time a call evaluates a pose at, the window and the outer node times (the same for every steps).
std::pair<int64_t, int64_t> pose_range(int64_t t_a, int64_t t_b, float c, float shutter_frames,
                                       const TimeMap& time) {
  const std::vector<float> outer = node_times(c, shutter_frames, 1);
  return {std::min(t_a, time.to_ns(outer.front())), std::max(t_b, time.to_ns(outer.back()))};
}

}  // namespace

double motion_chord_px(const MotionProbe& probe, int64_t t_a, int64_t t_b, float c, float shutter_frames,
                       const TimeMap& time, int32_t steps, QuatInterp model) {
  if (t_b < t_a) throw std::invalid_argument("frames motion steps: the window must have t_a <= t_b");
  ProbeCtx::checked(probe);
  if (probe.points.empty() || !(shutter_frames > 0.0f)) return 0.0;
  const auto [lo, hi] = pose_range(t_a, t_b, c, shutter_frames, time);
  const ProbeCtx ctx(probe, lo, hi);
  return chord(ctx, t_a, t_b, c, shutter_frames, time, steps, model);
}

int32_t motion_steps(const MotionProbe& probe, int64_t t_a, int64_t t_b, float c, float shutter_frames,
                     const TimeMap& time, double tol_px, QuatInterp model, MotionStepsInfo* info) {
  if (!(tol_px > 0.0)) throw std::invalid_argument("frames motion steps: motion_tol_px must be > 0");
  if (t_b < t_a) throw std::invalid_argument("frames motion steps: the window must have t_a <= t_b");
  MotionStepsInfo local;
  MotionStepsInfo& out = info != nullptr ? *info : local;
  out = MotionStepsInfo{};
  out.budget_px = tol_px;
  ProbeCtx::checked(probe);
  if (probe.points.empty() || !(shutter_frames > 0.0f)) return 1;
  const auto [lo, hi] = pose_range(t_a, t_b, c, shutter_frames, time);
  const ProbeCtx ctx(probe, lo, hi);
  out.speed_px_s = image_speed(ctx, t_a, t_b);
  const float largest = std::max(std::fabs(c) + 0.5f * shutter_frames, 1.0f);
  out.time_res_px = out.speed_px_s * 2.0 * ulp32(largest) * time.period_ns() * 1e-9;
  if (out.time_res_px > tol_px / 2) {
    out.warnings.push_back({"FRM-TIME-RES", "frames: image speed " + num(out.speed_px_s) +
                                                " px/s times Blender's node-time resolution near frame " + num(c) +
                                                " is " + num(out.time_res_px) +
                                                " px, above motion_tol_px / 2; neither motion steps nor sub-renders "
                                                "lower it, so the chord keeps motion_tol_px / 2"});
  }
  // The time-resolution term does not depend on n and sub-renders keep its ulp32, so it never asks for more
  // sub-renders: it comes out of the chord's budget up to tol / 2, and FRM-TIME-RES carries the rest.
  out.budget_px = std::max(tol_px - out.time_res_px, tol_px / 2);
  for (int32_t n = 1; n <= 7; ++n) {
    out.chord_px = chord(ctx, t_a, t_b, c, shutter_frames, time, n, model);
    if (out.chord_px <= out.budget_px) return n;
  }
  return 0;
}

// ---- Sub-renders and row bands ----

LightSplit light_split(double rho_per_s, const RowTiming& timing, double factor_tol) {
  if (!(rho_per_s >= 0.0) || !std::isfinite(rho_per_s)) throw std::invalid_argument("frames: rho must be >= 0");
  if (!(factor_tol > 0.0)) throw std::invalid_argument("frames: factor_tol must be > 0");
  if (timing.height < 1 || timing.exposure_ns < 0 || timing.row_dt_ns < 0) {
    throw std::invalid_argument("frames: invalid row timing");
  }
  const bool rolling = timing.shutter == Shutter::kRolling;
  const double t = static_cast<double>(timing.exposure_ns) * 1e-9;
  const double delta = rolling ? static_cast<double>(timing.row_dt_ns) : 0.0;
  const double r = static_cast<double>(timing.height - 1) * delta * 1e-9;
  const auto need = [&](double x) {  // ceil(x), at least 1, without overflow
    return x <= 1.0 ? 1.0 : std::ceil(std::min(x, 1e9));
  };
  const double sub = need((rolling ? 2.0 : 1.0) * rho_per_s * t / factor_tol);
  const double band = rolling ? need(2.0 * rho_per_s * r / factor_tol) : 1.0;
  LightSplit ls;
  const int32_t band_cap = std::min(timing.height, kMaxBands);
  ls.n_sub = static_cast<int32_t>(std::min(sub, static_cast<double>(kMaxSubRenders)));
  ls.n_band = static_cast<int32_t>(std::min(band, static_cast<double>(band_cap)));
  const double t_max = std::ceil(static_cast<double>(timing.exposure_ns) / ls.n_sub) * 1e-9;
  const double rows_max = std::ceil(static_cast<double>(timing.height) / ls.n_band);
  ls.bound = rho_per_s * (t_max + (rows_max - 1.0) * delta * 1e-9);
  if (sub > kMaxSubRenders || band > band_cap) {
    ls.warnings.push_back({"FRM-LIGHT-MOTION",
                           "frames: a moving light (rho = " + num(rho_per_s) + " 1/s) needs " + num(sub) +
                               " sub-renders and " + num(band) + " row bands; at the caps (" +
                               std::to_string(ls.n_sub) + ", " + std::to_string(ls.n_band) +
                               ") the frozen-light bound rho (T'_max + R_band) is " + num(ls.bound) +
                               " against factor_tol " + num(factor_tol) + " (T term " + num(rho_per_s * t_max) +
                               ", R term " + num(ls.bound - rho_per_s * t_max) + ")"});
  }
  return ls;
}

SubRenderChoice choose_n_sub(int32_t n_sub0, double factor_tol, const std::function<bool(int32_t)>& steps_ok,
                             const std::function<double(int32_t)>& fact_p99) {
  if (n_sub0 < 1 || n_sub0 > kMaxSubRenders) throw std::invalid_argument("frames: n_sub must lie in [1, 64]");
  if (!(factor_tol > 0.0)) throw std::invalid_argument("frames: factor_tol must be > 0");
  SubRenderChoice ch;
  for (int32_t n = n_sub0;; n = std::min(2 * n, kMaxSubRenders)) {
    const bool steps = !steps_ok || steps_ok(n);
    const double f = fact_p99 ? fact_p99(n) : 0.0;
    const bool fact = f <= factor_tol;
    ch.n_sub = n;
    ch.fact_p99 = f;
    if (steps && fact) return ch;
    if (n == kMaxSubRenders) {
      if (!steps) {
        ch.warnings.push_back({"FRM-MOTION", "frames: 7 motion steps are not enough at n_sub = 64"});
      }
      if (!fact) {
        ch.warnings.push_back({"FRM-FACTOR", "frames: the factorization error p99 " + num(f) +
                                                 " exceeds factor_tol " + num(factor_tol) + " at n_sub = 64"});
      }
      return ch;
    }
  }
}

}  // namespace balla::frames
