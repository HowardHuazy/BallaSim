// The exposure reference (reference.h; docs/design/phase3/frames.md §4.3, §4.4, §4.9.4).
#include "frames/reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "core/camera/injective_domain.h"
#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "core/time/time.h"
#include "core/traj/bounds.h"
#include "core/traj/spline_build.h"
#include "render/pose.h"
#include "sensor/params.h"

namespace balla::frames {

namespace {

__extension__ typedef __int128 i128;

using render::RigidPose;
using Vec3d = Vec3<double>;

// Level K's sample j sits at t_soe + floor((2j + 1) T / (2K)) and averages the waveforms over
// [t_soe + floor(j T / K), t_soe + floor((j + 1) T / K)] (T >= 0, so truncation is floor).
int64_t sample_offset(int64_t t, int32_t k, int32_t j) { return static_cast<int64_t>(i128{2 * j + 1} * t / (2 * k)); }
int64_t window_edge(int64_t t, int32_t k, int32_t j) { return static_cast<int64_t>(i128{j} * t / k); }

// The ceil(q n)-th smallest of the n finite values (nearest rank; a pixel without a lens preimage is NaN); 0 for none.
double quantile(std::vector<double> v, double q) {
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }), v.end());
  if (v.empty()) return 0.0;
  const auto n = static_cast<double>(v.size());
  size_t r = static_cast<size_t>(std::ceil(q * n));
  r = std::min(std::max<size_t>(r, 1), v.size()) - 1;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(r), v.end());
  return v[r];
}

RigidPose pose_of(const std::array<double, 16>& t) {
  return {{{{t[0], t[1], t[2]}, {t[4], t[5], t[6]}, {t[8], t[9], t[10]}}}, {t[3], t[7], t[11]}};
}

// One trajectory's spline and its per-segment speed bounds (traj/bounds.h).
struct TrajInfo {
  std::vector<traj::SplineNode<double>> nodes;
  render::TrajectoryView view{};
  std::vector<double> v, omega;

  // The largest bounds over the segments that [a, b] meets (clamped to the spline).
  std::pair<double, double> bounds(int64_t a, int64_t b) const {
    const int64_t segments = view.spline.num_segments, h = view.spline.h_ns;
    const auto seg = [&](int64_t t) {
      return std::clamp<int64_t>(time::floor_div(t - view.t0_ns, h), 0, segments - 1);
    };
    double vm = 0.0, wm = 0.0;
    for (int64_t s = seg(a); s <= seg(b); ++s) {
      vm = std::max(vm, v[static_cast<size_t>(s)]);
      wm = std::max(wm, omega[static_cast<size_t>(s)]);
    }
    return {vm, wm};
  }
  RigidPose pose(int64_t t) const {
    render::BodyState st{};
    if (!render::body_state(view, t, &st)) {
      throw std::invalid_argument("frames reference: time " + std::to_string(t) + " ns is outside a trajectory's span");
    }
    return st.pose;
  }
};

render::Mat3d mul(const render::Mat3d& a, const render::Mat3d& b) {
  render::Mat3d c{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) c.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
  }
  return c;
}

}  // namespace

struct ExposureReference::Impl {
  std::vector<TrajInfo> traj;
  std::vector<int32_t> watched;  // trajectories whose motion counts for any_motion
  bool flicker = false;          // some waveform is not constant
  // Camera model of the light-distance heuristic, built on first use.
  bool model_ready = false, kb = false;
  camera::RadtanCamera<double> radtan{};
  camera::KannalaBrandtCamera<double> kbc{};
  std::unique_ptr<render::Renderer> no_shadow;
};

ExposureReference::ExposureReference(render::Renderer& renderer, uint16_t cam,
                                     std::span<const sensor::Waveform> waveforms, Roi roi)
    : renderer_(renderer), cam_(cam), roi_(roi), groups_(renderer.num_groups()), impl_(std::make_unique<Impl>()) {
  const render::SceneBundle& b = renderer_.bundle();
  if (cam >= b.cameras.size()) throw std::invalid_argument("frames reference: camera index out of range");
  const render::Camera& c = b.cameras[cam];
  if (!(roi.x0 >= 0 && roi.y0 >= 0 && roi.x0 < roi.x1 && roi.y0 < roi.y1 && roi.x1 <= c.width &&
        roi.y1 <= c.height)) {
    throw FramesError("FRM-VALUE", "frames: the ROI must satisfy 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height");
  }
  if (waveforms.size() != b.light_groups.size()) {
    throw std::invalid_argument("frames reference: one waveform per light group is needed");
  }
  for (const sensor::Waveform& w : waveforms) {
    sensor::validate_waveform(w);
    impl_->flicker = impl_->flicker || w.type != sensor::WaveformType::kConstant;
  }
  waves_.assign(waveforms.begin(), waveforms.end());

  for (const render::Trajectory& t : b.trajectories) {
    // Its spline and segment bounds assume one piece; a piecewise trajectory (C1 "pieces", which the renderer
    // evaluates since BL-5b) is refused rather than read as one spline over every piece's controls.
    if (!t.pieces.empty()) {
      throw FramesError("FRM-VALUE", "frames reference: piecewise trajectories (C1 'pieces') are not supported yet");
    }
    TrajInfo info;
    const size_t n = t.rot.size() / 4;
    std::vector<Quat<double>> q(n);
    std::vector<Vec3d> p(n);
    for (size_t i = 0; i < n; ++i) {
      q[i] = {t.rot[4 * i], t.rot[4 * i + 1], t.rot[4 * i + 2], t.rot[4 * i + 3]};
      p[i] = {t.pos[3 * i], t.pos[3 * i + 1], t.pos[3 * i + 2]};
    }
    info.nodes = traj::build_spline_nodes(q, p, {0.0, 0.0, 0.0});
    const auto segments = static_cast<int32_t>(n) - t.order + 1;
    const auto h = static_cast<int32_t>(t.h_ns);
    info.view = {{info.nodes.data(), segments, h, 1e9 / static_cast<double>(h)}, t.t0_ns, t.order, t.span_begin_ns,
                 t.span_end_ns};
    for (int32_t s = 0; s < segments; ++s) {
      const auto mb = t.order == 4 ? traj::segment_bounds<4>(info.view.spline, s)
                                   : traj::segment_bounds<6>(info.view.spline, s);
      info.v.push_back(mb.v);
      info.omega.push_back(mb.omega);
    }
    impl_->traj.push_back(std::move(info));
  }
  // The node buffers stay where they were built (vector moves keep them), so the views stay valid.
  impl_->watched.push_back(c.rig_trajectory);
  for (const render::Body& body : b.bodies) {
    if (body.trajectory && !body.mesh.indices.empty()) impl_->watched.push_back(*body.trajectory);
  }
  for (const render::Light& l : b.lights) {
    if (l.body && b.bodies[static_cast<size_t>(*l.body)].trajectory) {
      impl_->watched.push_back(*b.bodies[static_cast<size_t>(*l.body)].trajectory);
    }
  }
}

ExposureReference::~ExposureReference() = default;

std::vector<render::GtSample> ExposureReference::centre_gt(int64_t t_ns) const {
  const auto w = static_cast<size_t>(roi_.x1 - roi_.x0), h = static_cast<size_t>(roi_.y1 - roi_.y0);
  std::vector<uint16_t> cams(w * h, cam_);
  std::vector<float> xs(w * h), ys(w * h);
  for (size_t p = 0; p < w * h; ++p) {
    xs[p] = static_cast<float>(roi_.x0 + static_cast<int32_t>(p % w));
    ys[p] = static_cast<float>(roi_.y0 + static_cast<int32_t>(p / w));
  }
  std::vector<int64_t> ts(w * h, t_ns);
  return renderer_.gt(cams, xs, ys, ts);
}

MotionFacts ExposureReference::motion(const Schedule& s, int32_t k) const { return motion_facts(s, k, nullptr); }

MotionFacts ExposureReference::motion_facts(const Schedule& s, int32_t k,
                                            const std::vector<render::GtSample>* gt_mid) const {
  if (k < 0 || k >= s.size()) throw std::invalid_argument("frames reference: frame index out of range");
  const render::SceneBundle& b = renderer_.bundle();
  const int64_t a = s.frame_soe(k), e = s.frame_eoe(k), mid = s.frame_mid(k), t = s.timing().exposure_ns;
  MotionFacts f;
  for (const int32_t i : impl_->watched) {
    const auto [v, w] = impl_->traj[static_cast<size_t>(i)].bounds(a, e);
    f.any_motion = f.any_motion || v > 0.0 || w > 0.0;
  }
  if (!f.any_motion) return f;

  // D_max = T max |flow| at the frame's ends, its middle and every piece boundary inside it.
  if (t > 0) {
    std::vector<int64_t> times{a, mid, e};
    times.insert(times.end(), f.breakpoints.begin(), f.breakpoints.end());
    double flow = 0.0;
    for (const int64_t ti : times) {
      std::vector<render::GtSample> own;
      if (ti != mid || gt_mid == nullptr) own = centre_gt(ti);
      for (const render::GtSample& g : own.empty() ? *gt_mid : own) {
        const double m = std::hypot(static_cast<double>(g.flow_x), static_cast<double>(g.flow_y));
        if (std::isfinite(m)) flow = std::max(flow, m);
      }
    }
    f.d_max_px = static_cast<double>(t) * 1e-9 * flow;
  }

  // Lights on moving bodies and their rate bound rho.
  std::vector<Vec3d> points;  // visible ROI points at frame_mid, world frame
  bool points_ready = false;
  for (const render::Light& l : b.lights) {
    if (!render::positional(l.kind) || !l.body) continue;
    const auto& body = b.bodies[static_cast<size_t>(*l.body)];
    if (!body.trajectory) continue;
    const TrajInfo& tr = impl_->traj[static_cast<size_t>(*body.trajectory)];
    const auto [v, w] = tr.bounds(a, e);
    if (v == 0.0 && w == 0.0) continue;
    f.light_motion = true;
    if (l.kind == render::BundleLightKind::kDirectional) {
      f.light_rate_per_s = std::max(f.light_rate_per_s, 3.0 * w);
      continue;
    }
    if (!points_ready) {
      points_ready = true;
      std::vector<render::GtSample> own;
      if (gt_mid == nullptr) own = centre_gt(mid);
      const std::vector<render::GtSample>& gt = gt_mid != nullptr ? *gt_mid : own;
      const render::Camera& c = b.cameras[cam_];
      if (!impl_->model_ready) {
        const double support = render::support_px(renderer_.options().scene);
        const auto dist = [&](size_t i) { return i < c.dist.size() ? c.dist[i] : 0.0; };
        impl_->kb = c.model == render::CameraModelKind::kKannalaBrandt;
        if (impl_->kb) {
          impl_->kbc = {c.fx, c.fy, c.cx, c.cy, dist(0), dist(1), dist(2), dist(3), 0.0};
          impl_->kbc.theta_lim = camera::kannala_brandt_injective_domain(impl_->kbc, c.width, c.height, support).theta_lim;
        } else {
          impl_->radtan = {c.fx, c.fy, c.cx, c.cy, dist(0), dist(1), dist(2), dist(3), dist(4), 0.0, 0.0};
          impl_->radtan.r_lim = camera::radtan_injective_domain(impl_->radtan, c.width, c.height, support).r_lim;
        }
        impl_->model_ready = true;
      }
      const RigidPose rig = impl_->traj[static_cast<size_t>(c.rig_trajectory)].pose(mid);
      const RigidPose rc = pose_of(c.T_rig_cam);
      const render::Mat3d r_wc = mul(rig.r, rc.r);
      const Vec3d p_wc = rig.r * rc.p + rig.p;
      const auto w_roi = static_cast<size_t>(roi_.x1 - roi_.x0);
      for (size_t p = 0; p < gt.size(); ++p) {
        const double z = gt[p].depth;
        if (!(std::isfinite(z) && z > 0.0)) continue;
        const Vec2<double> px{static_cast<double>(roi_.x0) + static_cast<double>(p % w_roi),
                              static_cast<double>(roi_.y0) + static_cast<double>(p / w_roi)};
        Vec3d ray{};
        const bool ok = impl_->kb ? camera::unproject(impl_->kbc, px, &ray) : camera::unproject(impl_->radtan, px, &ray);
        if (!ok || !(ray.z > 0.0)) continue;
        points.push_back(r_wc * ((z / ray.z) * ray) + p_wc);
      }
    }
    const RigidPose bp = tr.pose(mid);
    const Vec3d pos{l.position[0], l.position[1], l.position[2]};
    const Vec3d lw = bp.r * pos + bp.p;
    double d = std::numeric_limits<double>::infinity();
    for (const Vec3d& x : points) d = std::min(d, norm(x - lw));
    if (!(d > 0.0) || !std::isfinite(d)) continue;  // nothing visible, or the light on a visible surface
    const double v_l = v + w * norm(pos);
    f.light_rate_per_s = std::max(f.light_rate_per_s, v_l / d * 5.0);
  }
  return f;
}

namespace {

// One level K over a list of ROI pixels i, from each pixel's K samples I_{g,j} (summarize): F_g in fg[i G + g]; the
// extremes over j of the probe mask's quantity sum_g wbar_g I_{g,j} in lo[i], hi[i]; e_fact[i] with the factorization
// test (else empty); and, when the level is kept for a 3K level's reuse, the samples raw[(i K + j) G + g] (else empty).
struct Level {
  int32_t k = 0;
  std::vector<double> fg, lo, hi, e_fact;
  std::vector<float> raw;
};

// Per frame: read-row starts, waveform prefixes and row means.
struct FrameCtx {
  int64_t t = 0;
  int32_t groups = 0, w = 0, h = 0, x0 = 0, y0 = 0, width = 0;
  std::vector<int64_t> row_soe;               // ROI rows
  std::vector<sensor::WaveformPrefix> prefix;  // one per group over [frame_soe, frame_eoe]
  std::vector<double> wbar;                   // ROI H x G

  // Level K's sample means mbar_{g,j} of every ROI row: H x K x G.
  std::vector<double> means(int32_t k) const {
    std::vector<double> m(static_cast<size_t>(h) * static_cast<size_t>(k) * static_cast<size_t>(groups));
    for (int32_t r = 0; r < h; ++r) {
      for (int32_t j = 0; j < k; ++j) {
        const int64_t lo = row_soe[static_cast<size_t>(r)] + window_edge(t, k, j);
        const int64_t hi = row_soe[static_cast<size_t>(r)] + window_edge(t, k, j + 1);
        for (int32_t g = 0; g < groups; ++g) {
          m[(static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(j)) * static_cast<size_t>(groups) +
            static_cast<size_t>(g)] = prefix[static_cast<size_t>(g)].mean(lo, hi);
        }
      }
    }
    return m;
  }
};

using Groups = std::array<double, sensor::kMaxLightGroups>;

// Pixel i (read-row r) of level lv from its K samples blk[j G + g] and the level's means m:
//   F_g = (sum_j mbar_{g,j} I_{g,j}) / K, accumulated in sample order;
//   lo, hi: the extremes over j of sum_g wbar_g I_{g,j};
//   with n_sub > 0, e_fact = max over sub-windows s of |sum_g (mean_s(mbar_g I_g) - mean_s(mbar_g) mean_s(I_g))| / F
//   (0 when F <= 0), where sample j belongs to sub-window floor((2j + 1) n_sub / (2K)), the one that holds its
//   midpoint (with K a multiple of n_sub, floor(j n_sub / K): the samples' windows then tile the sub-windows).
void summarize(const FrameCtx& c, const std::vector<double>& m, int32_t r, const float* blk, int32_t n_sub, size_t i,
               Level& lv) {
  const auto g_n = static_cast<size_t>(c.groups);
  const auto k = static_cast<size_t>(lv.k);
  const double* mr = m.data() + static_cast<size_t>(r) * k * g_n;
  const double* wb = c.wbar.data() + static_cast<size_t>(r) * g_n;
  Groups fg{};
  double lo = std::numeric_limits<double>::infinity(), hi = -lo;
  for (size_t j = 0; j < k; ++j) {
    double sj = 0.0;
    for (size_t g = 0; g < g_n; ++g) {
      const auto ii = static_cast<double>(blk[j * g_n + g]);
      fg[g] += mr[j * g_n + g] * ii;
      sj += wb[g] * ii;
    }
    lo = std::min(lo, sj);
    hi = std::max(hi, sj);
  }
  double f = 0.0;
  for (size_t g = 0; g < g_n; ++g) {
    fg[g] /= static_cast<double>(k);
    lv.fg[i * g_n + g] = fg[g];
    f += fg[g];
  }
  lv.lo[i] = lo;
  lv.hi[i] = hi;
  if (n_sub == 0 || !(f > 0.0)) return;
  double worst = 0.0;
  size_t j = 0;
  for (int64_t sw = 0; sw < n_sub; ++sw) {
    Groups smi{}, sm{}, si{};
    int32_t cnt = 0;
    for (; j < k && static_cast<int64_t>(2 * j + 1) * n_sub / static_cast<int64_t>(2 * k) == sw; ++j, ++cnt) {
      for (size_t g = 0; g < g_n; ++g) {
        const double mi = mr[j * g_n + g], ii = blk[j * g_n + g];
        smi[g] += mi * ii;
        sm[g] += mi;
        si[g] += ii;
      }
    }
    if (cnt == 0) continue;
    double sum = 0.0;
    for (size_t g = 0; g < g_n; ++g) sum += smi[g] / cnt - (sm[g] / cnt) * (si[g] / cnt);
    worst = std::max(worst, std::abs(sum) / f);
  }
  lv.e_fact[i] = worst;
}

// Pixel `from` of level `src` copied to pixel `to` of `dst` (the same K, so the same samples); raw too when dst keeps
// it (src then does).
void copy_pixel(const Level& src, size_t from, Level& dst, size_t to, size_t g_n) {
  std::copy_n(src.fg.begin() + static_cast<std::ptrdiff_t>(from * g_n), g_n,
              dst.fg.begin() + static_cast<std::ptrdiff_t>(to * g_n));
  dst.lo[to] = src.lo[from];
  dst.hi[to] = src.hi[from];
  if (!dst.e_fact.empty()) dst.e_fact[to] = src.e_fact[from];
  if (!dst.raw.empty()) {
    const size_t block = static_cast<size_t>(dst.k) * g_n;
    std::copy_n(src.raw.begin() + static_cast<std::ptrdiff_t>(from * block), block,
                dst.raw.begin() + static_cast<std::ptrdiff_t>(to * block));
  }
}

}  // namespace

RefFrame ExposureReference::frame(const Schedule& s, int32_t k, const RefOptions& o) {
  if (k < 0 || k >= s.size()) throw std::invalid_argument("frames reference: frame index out of range");
  if (!(o.h_px > 0.0) || !(o.probe_tol > 0.0) || o.k_max < 2 || o.k_max > 4096 || o.k_fixed < 0 ||
      o.k_fixed > kMaxSamples || o.n_sub < 1 || o.n_sub > 64 || !(o.reuse_budget_mb >= 0.0)) {
    throw std::invalid_argument("frames reference: need h_px > 0, probe_tol > 0, k_max in [2, 4096], k_fixed in "
                                "[0, " + std::to_string(kMaxSamples) + "], n_sub in [1, 64] and reuse_budget_mb >= 0");
  }
  const render::Camera& cam = renderer_.bundle().cameras[cam_];
  if (s.timing().height != cam.height) {
    throw std::invalid_argument("frames reference: the schedule's height differs from the camera's");
  }
  FrameCtx c;
  c.t = s.timing().exposure_ns;
  c.groups = groups_;
  c.w = roi_.x1 - roi_.x0;
  c.h = roi_.y1 - roi_.y0;
  c.x0 = roi_.x0;
  c.y0 = roi_.y0;
  c.width = cam.width;
  const auto g_n = static_cast<size_t>(groups_);
  const size_t npix = static_cast<size_t>(c.w) * static_cast<size_t>(c.h);
  const auto row_of = [&](size_t p) { return static_cast<int32_t>(p / static_cast<size_t>(c.w)); };
  for (int32_t r = 0; r < c.h; ++r) c.row_soe.push_back(s.soe(k, roi_.y0 + r));
  for (const sensor::Waveform& w : waves_) c.prefix.emplace_back(w, s.frame_soe(k), s.frame_eoe(k));
  c.wbar.resize(static_cast<size_t>(c.h) * g_n);
  for (int32_t r = 0; r < c.h; ++r) {
    for (size_t g = 0; g < g_n; ++g) {
      c.wbar[static_cast<size_t>(r) * g_n + g] = c.prefix[g].mean(c.row_soe[static_cast<size_t>(r)],
                                                                  c.row_soe[static_cast<size_t>(r)] + c.t);
    }
  }

  RefFrame out;
  out.m_mean = c.wbar;
  renderer_.set_window(s.frame_soe(k));
  const std::vector<render::GtSample> gt_mid = centre_gt(s.frame_mid(k));
  out.depth.resize(npix);
  out.instance.resize(npix);
  for (size_t p = 0; p < npix; ++p) {
    out.depth[p] = gt_mid[p].depth;
    out.instance[p] = gt_mid[p].instance;
  }
  const MotionFacts mf = motion_facts(s, k, &gt_mid);
  out.d_max_px = mf.d_max_px;

  // ---- K0 (§4.4) and the factorization's K (§4.3) ----
  const int32_t t_cap = c.t <= 0 ? 1 : static_cast<int32_t>(std::min<int64_t>(c.t, kMaxSamples));
  const int32_t k_cap = std::min(t_cap, o.k_max);
  int32_t k0 = 1;
  if (o.k_fixed > 0) {
    k0 = std::min(o.k_fixed, t_cap);
  } else if (mf.any_motion && c.t > 0) {
    const double d = mf.d_max_px;
    const double need = d < 4.0 ? 2.5 * std::sqrt(d) / o.h_px : 1.25 * d / o.h_px;  // 1.25 D / h_eff
    k0 = static_cast<int32_t>(std::min<double>(k_cap, std::max(2.0, std::ceil(need))));
  }
  const bool factor = mf.any_motion && impl_->flicker && c.t > 0;
  // The probe's all-pixel levels are capped at k_top: k_cap, or with the factorization test k_cap rounded down to a
  // multiple of n_sub when that exceeds K0 (K0 is then one too, and so is every level: their samples' windows tile
  // the sub-windows).
  int32_t k_top = k_cap;
  if (factor && o.k_fixed == 0) {
    int32_t want = std::max(k0, 4 * o.n_sub);
    want = (want + o.n_sub - 1) / o.n_sub * o.n_sub;
    if (want > k_cap) want = std::max(k_cap / o.n_sub * o.n_sub, std::min(k_cap, k0));
    if (want < 4 * o.n_sub) {
      out.warnings.push_back({"FRM-FACTOR", "frames: the K cap " + std::to_string(k_cap) + " leaves fewer than 4 "
                                            "reference samples in each of the " + std::to_string(o.n_sub) +
                                            " sub-windows"});
    }
    k0 = std::max(want, 1);
    if (k_cap / o.n_sub * o.n_sub > k0) k_top = k_cap / o.n_sub * o.n_sub;
  }
  // The mask level above K: 3K (its samples 3j + 1 are K's) or T; with the factorization test, a level capped by T is
  // rounded down to a multiple of n_sub when that still exceeds K.
  const auto mask_level = [&](int32_t kk) {
    const auto kn = static_cast<int32_t>(std::min<int64_t>(int64_t{3} * kk, t_cap));
    const int32_t whole = kn / o.n_sub * o.n_sub;
    return factor && kn < 3 * kk && whole > kk ? whole : kn;
  };

  // ---- Rendering levels ----
  std::vector<uint16_t> cams;
  std::vector<uint32_t> pixels;
  std::vector<int64_t> times;
  std::vector<size_t> dest;
  std::vector<float> buf;  // the pending pixels' sample blocks
  // Fills level kk for the ROI pixels `pix`: a pixel with whole(i) >= 0 takes the results of donor's pixel whole(i),
  // which has the same samples; third(i) is its level-kk/3 block when that is kept (sample j there is sample 3j + 1
  // here); every other sample is rendered, at most kRequestChunk samples' blocks at a time, and each pixel is
  // summarized as soon as its block is complete. With keep, the level keeps its samples (raw).
  const auto fill = [&](Level& lv, int32_t kk, const std::vector<int32_t>& pix, bool keep, const Level* donor,
                        const auto& whole, const auto& third) {
    const std::vector<double> m = c.means(kk);
    const size_t block = static_cast<size_t>(kk) * g_n;
    const size_t batch = std::min(pix.size(), std::max<size_t>(1, kRequestChunk / static_cast<size_t>(kk)));
    lv = Level{};  // frees a reused level's samples first
    lv.k = kk;
    lv.fg.assign(pix.size() * g_n, 0.0);
    lv.lo.assign(pix.size(), 0.0);
    lv.hi.assign(pix.size(), 0.0);
    lv.e_fact.assign(factor ? pix.size() : 0, 0.0);
    if (keep) lv.raw.resize(pix.size() * block);
    buf.assign(batch * block, 0.0f);
    std::vector<size_t> pend;
    const auto flush = [&] {
      if (!times.empty()) {
        const render::RenderResult res = renderer_.render(cams, pixels, times);
        for (size_t q = 0; q < times.size(); ++q) {
          std::memcpy(buf.data() + dest[q], res.irradiance.data() + q * g_n, g_n * sizeof(float));
        }
        out.rendered += static_cast<int64_t>(times.size());
        cams.clear();
        pixels.clear();
        times.clear();
        dest.clear();
      }
      for (size_t n = 0; n < pend.size(); ++n) {
        const size_t i = pend[n];
        const float* blk = buf.data() + n * block;
        summarize(c, m, row_of(static_cast<size_t>(pix[i])), blk, factor ? o.n_sub : 0, i, lv);
        if (keep) std::memcpy(lv.raw.data() + i * block, blk, block * sizeof(float));
      }
      pend.clear();
    };
    for (size_t i = 0; i < pix.size(); ++i) {
      if (const int32_t d = whole(i); d >= 0) {
        copy_pixel(*donor, static_cast<size_t>(d), lv, i, g_n);
        continue;
      }
      if (pend.size() == batch) flush();
      float* blk = buf.data() + pend.size() * block;
      const float* prev = third(i);
      const auto p = static_cast<size_t>(pix[i]);
      const int32_t r = row_of(p);
      const auto x = static_cast<uint32_t>(c.x0 + static_cast<int32_t>(p % static_cast<size_t>(c.w)));
      const auto y = static_cast<uint32_t>(c.y0 + r);
      for (int32_t j = 0; j < kk; ++j) {
        if (prev != nullptr && j % 3 == 1) {
          std::memcpy(blk + static_cast<size_t>(j) * g_n, prev + static_cast<size_t>(j / 3) * g_n, g_n * sizeof(float));
          continue;
        }
        cams.push_back(cam_);
        pixels.push_back(y * static_cast<uint32_t>(c.width) + x);
        times.push_back(c.row_soe[static_cast<size_t>(r)] + sample_offset(c.t, kk, j));
        dest.push_back(pend.size() * block + static_cast<size_t>(j) * g_n);
      }
      pend.push_back(i);
    }
    flush();
  };
  const auto no_donor = [](size_t) { return int32_t{-1}; };
  const auto no_third = [](size_t) -> const float* { return nullptr; };
  // The samples of a level are kept only while a 3K level may reuse them and every kept level fits in
  // reuse_budget_mb; the output does not depend on it (a reused sample is the same render).
  const double budget = o.reuse_budget_mb * 1048576.0;
  const auto raw_bytes = [&](size_t n, int32_t kk) {
    return static_cast<double>(n) * static_cast<double>(kk) * static_cast<double>(g_n * sizeof(float));
  };
  const bool probe = o.k_fixed == 0 && c.t > 0 && (k0 >= 2 || mf.light_motion);
  const auto reusable = [&](int32_t kk) { return probe && int64_t{3} * kk <= t_cap; };

  std::vector<int32_t> all_pix(npix);
  for (size_t p = 0; p < npix; ++p) all_pix[p] = static_cast<int32_t>(p);
  Level all;
  fill(all, k0, all_pix, reusable(k0) && raw_bytes(npix, k0) <= budget, nullptr, no_donor, no_third);

  // ---- Edge-mask probe (§4.4) ----
  Level mlev;
  std::vector<int32_t> mask;
  std::vector<int32_t> in_mask(npix, -1);  // position in mask, or -1
  int32_t km = all.k;
  if (probe) {
    while (true) {
      const int32_t kk = all.k;
      std::vector<double> f_all(npix);
      for (size_t p = 0; p < npix; ++p) {
        double f = 0.0;
        for (size_t g = 0; g < g_n; ++g) f += all.fg[p * g_n + g];
        f_all[p] = f;
      }
      double eps = 1e-3 * quantile(f_all, 0.99);
      if (!(eps > 0.0)) eps = 1e-30;
      mask.clear();
      std::fill(in_mask.begin(), in_mask.end(), -1);
      for (size_t p = 0; p < npix; ++p) {
        const int32_t x = c.x0 + static_cast<int32_t>(p % static_cast<size_t>(c.w)), y = c.y0 + row_of(p);
        if ((x % 4 == 0 && y % 4 == 0) || std::log(all.hi[p] + eps) - std::log(all.lo[p] + eps) > o.probe_tol) {
          in_mask[p] = static_cast<int32_t>(mask.size());
          mask.push_back(static_cast<int32_t>(p));
        }
      }
      km = mask_level(kk);
      out.probe_p999 = out.probe_max = 0.0f;
      if (km == kk || mask.empty()) {
        km = kk;
        break;
      }
      // The next level should the probe fail; it takes the mask level's pixels whole when the K agree, so the mask
      // level keeps its samples when the next level does.
      const int32_t knew = static_cast<int32_t>(std::min<int64_t>(int64_t{3} * kk, k_top));
      const bool next_keep = kk < k_top && reusable(knew) &&
                             static_cast<double>(all.raw.size() * sizeof(float)) +
                                     (knew == km ? raw_bytes(mask.size(), km) : 0.0) + raw_bytes(npix, knew) <=
                                 budget;
      const size_t kblock = static_cast<size_t>(kk) * g_n;
      fill(mlev, km, mask, next_keep && knew == km, nullptr, no_donor, [&](size_t i) -> const float* {
        return km == 3 * kk && !all.raw.empty() ? all.raw.data() + static_cast<size_t>(mask[i]) * kblock : nullptr;
      });
      std::vector<double> e(mask.size());
      const double ratio = static_cast<double>(km) / static_cast<double>(km - kk);
      for (size_t i = 0; i < mask.size(); ++i) {
        double fm = 0.0;
        for (size_t g = 0; g < g_n; ++g) fm += mlev.fg[i * g_n + g];
        e[i] = ratio * std::abs(std::log(fm + eps) - std::log(f_all[static_cast<size_t>(mask[i])] + eps));
      }
      const double p999 = quantile(e, 0.999);
      out.probe_max = static_cast<float>(quantile(e, 1.0));
      out.probe_p999 = static_cast<float>(p999);
      if (p999 <= o.probe_tol) break;
      if (kk >= k_top) {
        out.warnings.push_back({"FRM-REF-K", "frames: the reference probe's p99.9 error " + std::to_string(p999) +
                                                 " exceeds frames.ref.probe_tol at "
                                                 "the cap K = " + std::to_string(kk)});
        break;
      }
      Level next;
      fill(
          next, knew, all_pix, next_keep, &mlev,
          [&](size_t p) { return knew == km ? in_mask[p] : int32_t{-1}; },
          [&](size_t p) -> const float* {
            return knew == 3 * kk && !all.raw.empty() ? all.raw.data() + p * kblock : nullptr;
          });
      all = std::move(next);
    }
  }
  out.k_used = all.k;
  out.k_mask = km;
  out.mask_px = probe && km != all.k ? static_cast<int64_t>(mask.size()) : 0;
  const bool use_mask = out.mask_px > 0;

  // ---- The frame: each pixel from its own level (the mask's on the mask) ----
  out.per_group.resize(npix * g_n);
  out.mono.resize(npix);
  std::vector<double> e_fact(factor ? npix : 0);
  for (size_t p = 0; p < npix; ++p) {
    const bool on_mask = use_mask && in_mask[p] >= 0;
    const Level& lv = on_mask ? mlev : all;
    const size_t i = on_mask ? static_cast<size_t>(in_mask[p]) : p;
    double f = 0.0;
    for (size_t g = 0; g < g_n; ++g) {
      out.per_group[p * g_n + g] = static_cast<float>(lv.fg[i * g_n + g]);
      f += lv.fg[i * g_n + g];
    }
    out.mono[p] = static_cast<float>(f);
    if (factor) e_fact[p] = lv.e_fact[i];
  }
  if (factor) out.fact_p99 = static_cast<float>(quantile(e_fact, 0.99));
  return out;
}

std::vector<float> ExposureReference::visibility(int64_t t_ns) {
  if (!impl_->no_shadow) {
    render::SceneBundle b = renderer_.bundle();
    for (render::Light& l : b.lights) l.casts_shadows = false;
    impl_->no_shadow = std::make_unique<render::Renderer>(std::move(b), renderer_.options());
  }
  const auto w = static_cast<size_t>(roi_.x1 - roi_.x0), h = static_cast<size_t>(roi_.y1 - roi_.y0);
  std::vector<uint16_t> cams(w * h, cam_);
  std::vector<uint32_t> pixels(w * h);
  for (size_t p = 0; p < w * h; ++p) {
    pixels[p] = static_cast<uint32_t>(roi_.y0 + static_cast<int32_t>(p / w)) *
                    static_cast<uint32_t>(renderer_.bundle().cameras[cam_].width) +
                static_cast<uint32_t>(roi_.x0 + static_cast<int32_t>(p % w));
  }
  std::vector<int64_t> ts(w * h, t_ns);
  const render::RenderResult lit = renderer_.render(cams, pixels, ts);
  const render::RenderResult open = impl_->no_shadow->render(cams, pixels, ts);
  std::vector<float> v(lit.irradiance.size());
  for (size_t i = 0; i < v.size(); ++i) v[i] = lit.irradiance[i] / std::max(open.irradiance[i], 1e-12f);
  return v;
}

}  // namespace balla::frames
