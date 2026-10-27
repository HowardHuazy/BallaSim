// Contains C++ translations of NumPy (einsum's contig_contig_outstride0_two kernel order), under its BSD licence:
// see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/evimo2/build.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "render/thread_pool.h"
#include "scene/blas.h"
#include "scene/errors.h"
#include "scene/kdtree.h"
#include "scene/npla.h"
#include "scene/npo.h"
#include "scene/pyfmt.h"
#include "scene/traj/gating.h"

namespace balla::scene::evimo2 {

namespace {

using geo::M4;
using rot::M;

constexpr int64_t kTabletopId = 22;
constexpr double kNearFloorM = 0.05;
const char* const kRelightFormat = "ballasim.relight/1";

struct SensorPreset {
  const char* sensor;
  double pitch;
  double drift_ppm;
};

SensorPreset sensor_of(const std::string& camera) {
  if (camera == "left_camera") return {"prophesee_gen3", 15e-6, 10.8};
  if (camera == "right_camera") return {"prophesee_gen3", 15e-6, 18.0};
  if (camera == "samsung_mono") return {"samsung_gen3", 9e-6, -2.0};
  throw std::invalid_argument("unknown EVIMO2 camera folder " + camera);
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

V scaled(V v, double s) { return {v.x * s, v.y * s, v.z * s}; }

// np.minimum (NaN-propagating).
double npmin(double a, double b) { return (a <= b || std::isnan(a)) ? a : b; }

// einsum("nij,kj->nki")'s kernel for contiguous operands (sum_of_products_contig_contig_outstride0_two on SSE2, two
// lanes): (a0 b0 + a2 b2) + a1 b1, added to the zeroed output.
double lanes3(const double a[3], V b) {
  const double l0 = a[2] * b.z + (a[0] * b.x + 0.0);
  const double l1 = 0.0 * 0.0 + (a[1] * b.y + 0.0);
  return 0.0 + (l0 + l1);
}

// PoseSamples of T_world_x @ t_x_b at the given transforms (adapter._samples_to_world): q normalized once, as
// PoseSamples.__post_init__ does.
void samples_to_world(const std::vector<M4>& t_world_x, const M4& t_x_b, std::vector<Q>& q, std::vector<V>& p) {
  q.resize(t_world_x.size());
  p.resize(t_world_x.size());
  for (size_t i = 0; i < t_world_x.size(); ++i) std::tie(q[i], p[i]) = geo::transform_to_pose(geo::matmul(t_world_x[i], t_x_b));
  q = traj::pose_samples_q(q);
}

Tree pairs_tree(const std::vector<std::pair<int64_t, int64_t>>& v) {
  Tree t = Tree::list();
  for (const auto& [a, b] : v) t.push_back(Tree::ints({a, b}));
  return t;
}

// adapter._fit_summary.
Tree fit_summary(const traj::FitResult& f) {
  const traj::FitDiagnostics& d = f.diag;
  Tree t = Tree::dict();
  t["lambda_pos_u"] = Tree::real(d.lambda_pos_u);
  t["lambda_rot_u"] = Tree::real(d.lambda_rot_u);
  t["cutoff_hz_pos"] = Tree::real(d.cutoff_hz_pos);
  t["cutoff_hz_rot"] = Tree::real(d.cutoff_hz_rot);
  t["sigma_pos_m"] = Tree::real(d.sigma_pos_m);
  t["sigma_rot_rad"] = Tree::real(d.sigma_rot_rad);
  t["rms_pos_m"] = Tree::real(d.rms_pos_m);
  t["rms_rot_rad"] = Tree::real(d.rms_rot_rad);
  t["num_samples"] = Tree::integer(d.num_samples);
  t["num_kept"] = Tree::integer(d.num_kept);
  t["holds_ns"] = pairs_tree(f.holds);
  t["presence_holds_ns"] = pairs_tree(d.presence_holds_ns);
  t["static"] = Tree::boolean(d.is_static);
  t["max_increment_rad"] = Tree::real(d.max_increment_rad);
  t["still_max_displacement_m"] = Tree::real(d.still_max_displacement_m);
  int64_t rejected = 0;
  if (d.gated) {
    for (const uint8_t k : d.gate.keep) rejected += k ? 0 : 1;
  }
  t["gate_rejected"] = Tree::integer(rejected);
  t["gb_rejected"] = Tree::integer(static_cast<int64_t>(d.gb_rejected_t_ns.size()));
  t["gcv_at_grid_edge"] = Tree::boolean(d.gcv_pos.at_grid_edge || d.gcv_rot.at_grid_edge);
  if (d.presence_fallback) {
    Tree fb = Tree::dict();
    fb["error"] = Tree::string(d.presence_fallback_error);
    fb["groups"] = pairs_tree(d.presence_fallback_groups);
    t["presence_fallback"] = std::move(fb);
  }
  return t;
}

// The warning of a fit that took the presence fallback (traj::fit_smoothing): the error and its data-free ends.
std::string presence_fallback_warning(const std::string& who, const traj::FitResult& f) {
  std::string ends;
  for (const auto& [a, b] : f.diag.presence_holds_ns) {
    ends += (ends.empty() ? "" : " and ") + pyfmt::fixed(static_cast<double>(b - a) * 1e-9, 2) + " s";
  }
  return who + ": " + f.diag.presence_fallback_error + " with " + (ends.empty() ? std::string("its ends") : ends) +
         " outside its kept Vicon samples free; refitted with its presence holds tied from the start";
}

// A failing fit's exception with the fit's name in front ("rig", "drone_01 (15)"), its type kept so the binding maps it
// as before (LinAlgError, TrajectoryError and ZeroDivisionError to their Python types, other invalid_argument to
// ValueError). Without the name a bare "<N>-th leading minor not positive definite" names no body or sequence.
std::exception_ptr named_error(const std::string& who, const std::exception_ptr& e) {
  try {
    std::rethrow_exception(e);
  } catch (const LinAlgError& x) {
    return std::make_exception_ptr(LinAlgError(who + ": " + x.what()));
  } catch (const TrajectoryError& x) {
    return std::make_exception_ptr(TrajectoryError(who + ": " + x.what()));
  } catch (const ZeroDivisionError& x) {
    return std::make_exception_ptr(ZeroDivisionError(who + ": " + x.what()));
  } catch (const AssetError&) {
    return e;
  } catch (const std::invalid_argument& x) {
    return std::make_exception_ptr(std::invalid_argument(who + ": " + x.what()));
  } catch (...) {
    return e;
  }
}

// The object mesh of adapter._object_mesh (with obj.registration()), in the body frame.
struct ObjectMesh {
  geo::Mesh mesh;
  M4 t_body_subject{};
  V centre{};
  double radius = 0;
  Tree info;
};

ObjectMesh object_mesh(const CatalogueObject& obj, const Options& opts) {
  const size_t n = obj.config_markers.size();
  if (n < 3 || obj.vsk_markers_mm.size() < n) {
    throw std::invalid_argument(obj.name + ": " + std::to_string(n) + " config markers but " +
                                std::to_string(obj.vsk_markers_mm.size()) + " VSK markers");
  }
  std::vector<V> a(n), b(n);
  for (size_t i = 0; i < n; ++i) {
    a[i] = scaled(obj.config_markers[i], obj.unit_scale);
    b[i] = scaled(obj.vsk_markers_mm[i], 1e-3);
  }
  const geo::Kabsch k = geo::kabsch(a, b);
  std::vector<double> sq(n);
  for (size_t i = 0; i < n; ++i) sq[i] = k.residuals[i] * k.residuals[i];
  const double rms = std::sqrt(npo::sum(sq) / static_cast<double>(n));
  const double gate = obj.name == "tabletop" ? 7e-3 : 1.5e-3;
  if (rms > gate) {
    std::vector<double> per(n);
    for (size_t i = 0; i < n; ++i) per[i] = npo::round_decimals(k.residuals[i] * 1e3, 2);
    throw std::invalid_argument("AST-E21 " + obj.name + ": Kabsch residual " + pyfmt::fixed(rms * 1e3, 2) +
                                " mm exceeds " + pyfmt::fixed(gate * 1e3, 1) + " mm (per marker " +
                                pyfmt::repr_list(per) + " mm)");
  }
  if (!obj.has_asset) throw std::logic_error(obj.name + ": the build needs its asset (none was loaded)");
  const M4 t_sm = geo::make_transform(k.r, k.t);
  geo::Mesh mesh = geo::transform_mesh(obj.mesh, t_sm);
  const V c_b = geo::area_centroid(mesh);
  const M eye{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  ObjectMesh out;
  out.t_body_subject = geo::make_transform(eye, -c_b);
  mesh = geo::transform_mesh(mesh, out.t_body_subject);
  std::vector<std::pair<const geo::Mesh*, int>> parts = {{&mesh, obj.placeholder ? 1 : obj.num_materials}};
  geo::Mesh markers;
  if (opts.markers) {
    const M4 tb = geo::matmul(out.t_body_subject, t_sm);
    std::vector<double> cm(3 * n);
    for (size_t i = 0; i < n; ++i) {
      cm[3 * i] = a[i].x, cm[3 * i + 1] = a[i].y, cm[3 * i + 2] = a[i].z;
    }
    const auto ni = static_cast<int64_t>(n);
    const std::vector<double> c3 = npo::matmul(npo::View2{tb.data(), 3, 3, 4, 1}, npo::c_view(cm.data(), ni, 3).t());
    const double radius = opts.marker_radius_m ? *opts.marker_radius_m : obj.marker_radius_mm * 1e-3;
    const geo::Mesh sphere = geo::icosphere(opts.marker_subdivisions, radius);
    std::vector<geo::Mesh> spheres;
    spheres.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const V c{c3[i] + tb[3], c3[n + i] + tb[7], c3[2 * n + i] + tb[11]};
      spheres.push_back(geo::transform_mesh(sphere, geo::make_transform(eye, c)));
    }
    std::vector<std::pair<const geo::Mesh*, int>> sp;
    for (const geo::Mesh& s : spheres) sp.emplace_back(&s, 0);
    markers = geo::merge_meshes(sp);
    markers.material_ids.clear();
    markers.has_material_ids = false;
    parts.emplace_back(&markers, 1);
  }
  out.mesh = geo::merge_meshes(parts);
  std::tie(out.centre, out.radius) = geo::bounding_sphere(out.mesh);
  std::vector<double> res(n);
  for (size_t i = 0; i < n; ++i) res[i] = npo::round_decimals(k.residuals[i] * 1e3, 3);
  out.info = Tree::dict();
  out.info["name"] = Tree::string(obj.name);
  out.info["kabsch_residual_mm"] = Tree::reals(res);
  out.info["centroid_subject_m"] = Tree::reals({c_b.x, c_b.y, c_b.z});
  out.info["asset"] = Tree::null();   // the asset cache's info, filled in by Python
  out.info["albedo"] = Tree::string(obj.placeholder ? "placeholder " + pyfmt::repr(opts.untextured_albedo) +
                                                          " (untextured, AST-23)"
                                                    : "texture");
  return out;
}

// The RMS distance of a mesh's surface (area-weighted triangle centroids) from c: M10-PIN's lever arm for rotation.
double surface_rms_radius(const geo::Mesh& m, V c) {
  const std::vector<double> area = geo::triangle_areas(m);
  double num = 0, den = 0;
  for (size_t t = 0; t < m.num_triangles(); ++t) {
    double g[3] = {0, 0, 0};
    for (size_t j = 0; j < 3; ++j) {
      const size_t vi = m.indices[3 * t + j];
      for (size_t a = 0; a < 3; ++a) g[a] += static_cast<double>(m.positions[3 * vi + a]) / 3.0;
    }
    const double dx = g[0] - c.x, dy = g[1] - c.y, dz = g[2] - c.z;
    num += area[t] * (dx * dx + dy * dy + dz * dz);
    den += area[t];
  }
  return den > 0 ? std::sqrt(num / den) : 0.0;
}

struct Prepared {
  int64_t id;
  const CatalogueObject* obj;
  ObjectMesh om;
  const Track* track;
  std::vector<int64_t> t;
  std::vector<Q> q;
  std::vector<V> p;
  traj::SmoothingOptions fopts;
};

}  // namespace

ClockOffset clock_offset(const CameraInput& s, const std::string& reference) {
  if (reference != "npz" && reference != "raw") throw std::invalid_argument("reference must be 'npz' or 'raw'");
  ClockOffset out;
  if (!s.g1) {
    out.warnings.push_back("no G1 delta for " + s.camera + "/" + s.category + "/" + s.name +
                           ": residual event offset taken as 0 (G1 measured -47..+9 ms across sequences)");
  } else if (!s.g1_se_ms) {
    out.warnings.push_back("G1 delta " + pyfmt::signed_fixed(s.g1_delta_ms, 2) + " ms for " + s.camera + "/" +
                           s.category + "/" + s.name +
                           " has no SE: it is not identified (g1_offsets.json notes why); the clock offset may be tens "
                           "of ms off");
  }
  const int64_t d_ns = s.g1 ? static_cast<int64_t>(std::nearbyint(s.g1_delta_ms * 1e6)) : 0;
  std::string source;
  if (reference == "npz") {
    out.offset_ns = -d_ns;
    source = s.g1 ? "G1 delta" : "none (0)";
  } else {
    if (!s.raw_line3_s) {
      throw std::invalid_argument("reference='raw' needs extrinsics.txt line 3, absent for " + s.directory);
    }
    out.offset_ns = static_cast<int64_t>(std::nearbyint(*s.raw_line3_s * 1e9)) - d_ns;
    source = std::string("raw L3") + (s.g1 ? " - G1 delta" : "");
  }
  out.meta = Tree::dict();
  out.meta["offset_ns"] = Tree::integer(out.offset_ns);
  out.meta["reference"] = Tree::string(reference);
  out.meta["source"] = Tree::string(source);
  out.meta["delta_ms"] = s.g1 ? Tree::real(s.g1_delta_ms) : Tree::null();
  out.meta["delta_se_ms"] = s.g1 && s.g1_se_ms ? Tree::real(*s.g1_se_ms) : Tree::null();
  out.meta["raw_line3_s"] = s.raw_line3_s ? Tree::real(*s.raw_line3_s) : Tree::null();
  out.meta["drift_ppm"] = Tree::real(sensor_of(s.camera).drift_ppm);
  Tree w = Tree::list();
  for (const std::string& x : out.warnings) w.push_back(Tree::string(x));
  out.meta["warnings"] = std::move(w);
  return out;
}

TabletopStatic tabletop_static(std::span<const Q> q, std::span<const V> p, std::span<const uint8_t> keep, V centre,
                               double radius, double max_disp_m, double fraction) {
  std::vector<Q> qk;
  std::vector<V> pk;
  for (size_t k = 0; k < q.size(); ++k) {
    if (keep[k]) {
      qk.push_back(q[k]);
      pk.push_back(p[k]);
    }
  }
  const auto [qm, pm] = rot::robust_mean_pose(qk, pk);
  std::vector<double> disp(qk.size());
  int64_t inside = 0;
  for (size_t k = 0; k < qk.size(); ++k) {
    disp[k] = traj::sphere_displacement(qm, pm, qk[k], pk[k], centre, radius);
    inside += disp[k] <= max_disp_m ? 1 : 0;
  }
  TabletopStatic out;
  out.inlier_fraction = static_cast<double>(inside) / static_cast<double>(qk.size());
  out.max_displacement_m = npo::max(disp);
  if (out.inlier_fraction >= fraction) out.pose = geo::pose_to_transform(qm, pm);
  return out;
}

Anchor light_anchor(const std::vector<Body>& bodies, const std::vector<const traj::FitResult*>& trajectories,
                    std::span<const V> cam_positions, int64_t t_ns) {
  Anchor a{{0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, "world +z (no bodies; gravity unknown)"};
  const Body* ref = nullptr;
  bool table = false;
  for (const Body& b : bodies) {
    if (b.id == kTabletopId) {
      ref = &b;
      table = true;
      break;
    }
  }
  if (!ref) {
    double best = 0;
    for (const Body& b : bodies) {
      const double area = npo::sum(geo::triangle_areas(b.mesh));
      if (!ref || area > best) {
        ref = &b;
        best = area;
      }
    }
  }
  if (!ref) return a;
  M4 t_wb;
  if (!ref->trajectory) {
    t_wb = ref->T_world_body;
  } else {
    const auto [q, p] = traj::evaluate_pose(*trajectories.at(static_cast<size_t>(*ref->trajectory)), t_ns);
    t_wb = geo::pose_to_transform(q, p);
  }
  const V dn = geo::dominant_normal(ref->mesh);
  const double dnv[3] = {dn.x, dn.y, dn.z};
  const std::vector<double> u = npo::matmul(npo::View2{t_wb.data(), 3, 3, 4, 1}, npo::View2{dnv, 3, 1, 1, 1});
  a.up = {u[0], u[1], u[2]};
  a.centre = geo::translation_of(t_wb);
  V s{0.0, 0.0, 0.0};
  for (const V& c : cam_positions) s = {s.x + c.x, s.y + c.y, s.z + c.z};
  const auto np = static_cast<double>(cam_positions.size());
  const V d = V{s.x / np, s.y / np, s.z / np} - a.centre;
  const double dv[3] = {d.x, d.y, d.z}, uv[3] = {a.up.x, a.up.y, a.up.z};
  if (blas::dot(3, dv, uv) < 0) a.up = -a.up;
  a.source = table ? "tabletop top" : "largest body " + ref->name + " (no tabletop; gravity unknown)";
  return a;
}

std::pair<double, double> radtan_field(const CameraInput& c) {
  const double fx = c.fx, fy = c.fy, cx = c.cx, cy = c.cy;
  const double k1 = c.dist[0], k2 = c.dist[1], p1 = c.dist[2], p2 = c.dist[3], k3 = c.dist[4];
  auto pix = [&](double x, double y, double& u, double& v) {
    const double r2 = x * x + y * y;
    const double rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
    u = fx * ((x * rad + 2.0 * p1 * x * y) + p2 * (r2 + 2.0 * x * x)) + cx;
    v = fy * ((y * rad + p1 * (r2 + 2.0 * y * y)) + 2.0 * p2 * x * y) + cy;
  };
  const std::vector<double> xs = npo::linspace(0.0, static_cast<double>(c.width), 33);
  const std::vector<double> ys = npo::linspace(0.0, static_cast<double>(c.height), 25);
  const size_t ng = xs.size() * ys.size();
  std::vector<double> gu(ng), gv(ng), x(ng), y(ng);
  for (size_t r = 0; r < ys.size(); ++r) {
    for (size_t k = 0; k < xs.size(); ++k) {
      const size_t i = r * xs.size() + k;
      gu[i] = xs[k];
      gv[i] = ys[r];
      x[i] = (gu[i] - cx) / fx;
      y[i] = (gv[i] - cy) / fy;
    }
  }
  for (int it = 0; it < 100; ++it) {
    for (size_t i = 0; i < ng; ++i) {
      double u, v;
      pix(x[i], y[i], u, v);
      x[i] = x[i] + (gu[i] - u) / fx;
      y[i] = y[i] + (gv[i] - v) / fy;
    }
  }
  std::vector<double> err(2 * ng);
  for (size_t i = 0; i < ng; ++i) {
    double u, v;
    pix(x[i], y[i], u, v);
    err[2 * i] = std::fabs(gu[i] - u);
    err[2 * i + 1] = std::fabs(gv[i] - v);
  }
  if (npo::max(err) > 1e-3) throw std::invalid_argument("radtan distortion not invertible over the field (hold_rule image)");
  const double e = 1e-6, two_e = 2.0 * e;
  std::vector<double> sv(ng), r2(ng);
  for (size_t i = 0; i < ng; ++i) {
    double ua, va, ub, vb, uc, vc, ud, vd;
    pix(x[i] + e, y[i], ua, va);
    pix(x[i] - e, y[i], ub, vb);
    pix(x[i], y[i] + e, uc, vc);
    pix(x[i], y[i] - e, ud, vd);
    // np.stack([d/dx, d/dy], -1): row (u, v), column (x, y).
    const double jac[4] = {(ua - ub) / two_e, (uc - ud) / two_e, (va - vb) / two_e, (vc - vd) / two_e};
    double s[2];
    npo::svd_values(jac, 2, 2, s);
    sv[i] = npo::max(s);
    r2[i] = (0.0 + x[i] * x[i]) + y[i] * y[i];
  }
  return {npo::max(sv), npo::max(r2)};
}

std::pair<traj::HoldView, std::vector<traj::HoldView>> hold_views(const std::vector<CameraInput>& cameras,
                                                                  const std::vector<M4>& t_w_rig,
                                                                  const std::vector<HoldBody>& bodies) {
  const size_t nc = cameras.size(), nr = t_w_rig.size();
  std::vector<double> focal(nc), corner(nc);
  std::vector<V> c_rig(nc);
  for (size_t k = 0; k < nc; ++k) {
    std::tie(focal[k], corner[k]) = radtan_field(cameras[k]);
    c_rig[k] = geo::translation_of(cameras[k].T_rig_cam);
  }
  // cams_w = einsum("nij,kj->nki", t_w_rig[:, :3, :3], c_rig) + t_w_rig[:, None, :3, 3].
  std::vector<V> cams_w(nr * nc);
  for (size_t n = 0; n < nr; ++n) {
    const M4& t = t_w_rig[n];
    for (size_t k = 0; k < nc; ++k) {
      double c[3];
      for (int i = 0; i < 3; ++i) c[i] = lanes3(&t[static_cast<size_t>(4 * i)], c_rig[k]) + t[static_cast<size_t>(4 * i + 3)];
      cams_w[n * nc + k] = {c[0], c[1], c[2]};
    }
  }
  std::vector<double> near(nr, std::numeric_limits<double>::infinity());
  std::vector<traj::HoldView> views;
  for (const HoldBody& b : bodies) {
    const geo::Mesh& mesh = *b.mesh;
    const std::vector<int64_t>& rows = *b.rows;
    // The longest edge (norm of tri - np.roll(tri, 1, axis=1) over every triangle corner).
    std::vector<double> edges(mesh.indices.size());
    for (size_t t = 0; t < mesh.num_triangles(); ++t) {
      V c[3];
      for (size_t j = 0; j < 3; ++j) {
        const size_t vi = mesh.indices[3 * t + j];
        c[j] = {mesh.positions[3 * vi], mesh.positions[3 * vi + 1], mesh.positions[3 * vi + 2]};
      }
      for (size_t j = 0; j < 3; ++j) edges[3 * t + j] = rot::norm_axis(c[j] - c[(j + 2) % 3]);
    }
    const double edge = npo::max(edges);
    const KdTree tree(std::vector<double>(mesh.positions.begin(), mesh.positions.end()));
    // Distance bound from the cameras at rig rows r to the body at its samples j.
    auto depth = [&](size_t j, size_t r) {
      const M rm = rot::quat_to_matrix((*b.q)[j]);
      const V pj = (*b.p)[j];
      double best = 0.0;
      for (size_t k = 0; k < nc; ++k) {
        const V d = cams_w[r * nc + k] - pj;
        // einsum("nji,nkj->nki"): a sequential sum over j from 0, added to the zeroed output.
        double l[3];
        for (int i = 0; i < 3; ++i) {
          double acc = 0.0;
          acc += rm.m[0][i] * d.x;
          acc += rm.m[1][i] * d.y;
          acc += rm.m[2][i] * d.z;
          l[i] = acc + 0.0;
        }
        const double dist = tree.nearest({l[0], l[1], l[2]});
        best = k == 0 ? dist : npmin(best, dist);   // .min(1)
      }
      return npo::maximum(best - edge, kNearFloorM);
    };
    const auto nrows = static_cast<int64_t>(rows.size());
    std::vector<double> d(nr);
    std::vector<int64_t> lo(nr), hi(nr);
    for (size_t r = 0; r < nr; ++r) {
      const auto ri = static_cast<int64_t>(r);
      lo[r] = std::clamp<int64_t>(std::upper_bound(rows.begin(), rows.end(), ri) - rows.begin() - 1, 0, nrows - 1);
      hi[r] = std::clamp<int64_t>(std::lower_bound(rows.begin(), rows.end(), ri) - rows.begin(), 0, nrows - 1);
      d[r] = depth(static_cast<size_t>(hi[r]), r);
    }
    traj::HoldView v;
    v.focal_px = focal;
    v.corner_r2 = corner;
    v.depth_m.resize(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) v.depth_m[i] = d[static_cast<size_t>(rows[i])];
    views.push_back(std::move(v));
    for (size_t r = 0; r < nr; ++r) {
      if (lo[r] != hi[r]) {
        const double other = depth(static_cast<size_t>(lo[r]), r);
        d[r] = npmin(d[r], other);
      }
    }
    for (size_t r = 0; r < nr; ++r) near[r] = npmin(near[r], d[r]);
  }
  traj::HoldView rig;
  rig.focal_px = focal;
  rig.corner_r2 = corner;
  rig.depth_m = near;
  rig.cameras_body = c_rig;
  rig.has_cameras = true;
  return {std::move(rig), std::move(views)};
}

Build build(const Inputs& in, const Options& opts) {
  const blas::OneThread one;
  if (in.cameras.empty()) throw std::invalid_argument("EVIMO2 build: no camera");
  const CameraInput& seq = in.cameras[0];
  const traj::SmoothingOptions& smooth = opts.smoothing;
  Build out;
  std::vector<std::string>& warnings = out.warnings;
  out.retiming = traj::retime_stamps(in.t_raw_ns);
  const traj::Retiming& rt = out.retiming;
  const auto h_ns = static_cast<int64_t>(std::nearbyint(rt.b_ns));
  const std::vector<int64_t>& t_hat = rt.t_ns;
  if (in.cam.rows.size() != in.t_raw_ns.size()) {
    throw std::invalid_argument(seq.directory + ": full_trajectory rows without a camera pose");
  }
  const size_t nrows = t_hat.size();
  std::vector<M4> t_c0_c(nrows);
  for (size_t i = 0; i < nrows; ++i) t_c0_c[i] = geo::pose_to_transform(in.cam.q[i], in.cam.p[i]);

  // World frame (adapter._world_frame).
  M4 t_c0_w;
  Tree world = Tree::dict();
  if (std::find(in.object_ids.begin(), in.object_ids.end(), kTabletopId) == in.object_ids.end()) {
    t_c0_w = geo::invert_transform(seq.T_rig_cam);
    world["frame"] = Tree::string("rig_at_C0");
    world["reason"] = Tree::string("no tabletop in this sequence");
    world["note"] = Tree::string("C0 = this camera at the first Vicon sample of the recording, not of this part");
  } else {
    const Track& tr = in.tracks.at(kTabletopId);
    const size_t m = tr.rows.size();
    std::vector<Q> q(m);
    std::vector<V> p(m);
    std::vector<int64_t> t(m);
    for (size_t k = 0; k < m; ++k) {
      const auto row = static_cast<size_t>(tr.rows[k]);
      std::tie(q[k], p[k]) = geo::transform_to_pose(geo::matmul(t_c0_c[row], geo::pose_to_transform(tr.q[k], tr.p[k])));
      t[k] = in.t_raw_ns[row];
    }
    const traj::GateResult gate = traj::gate_poses(t, q, p,
                                                   {rt.b_ns, smooth.gate_sigma_k, smooth.gate_rot_floor_rad,
                                                    smooth.gate_pos_floor_m, smooth.branch_jump_rad,
                                                    smooth.branch_rate_factor});
    std::vector<Q> qk;
    std::vector<V> pk;
    for (size_t k = 0; k < m; ++k) {
      if (gate.keep[k]) {
        qk.push_back(q[k]);
        pk.push_back(p[k]);
      }
    }
    const auto [qm, pm] = rot::robust_mean_pose(qk, pk);
    t_c0_w = geo::pose_to_transform(qm, pm);
    world["frame"] = Tree::string("tabletop_subject_mean");
    world["samples"] = Tree::integer(static_cast<int64_t>(qk.size()));
  }
  const M4 t_w_c0 = geo::invert_transform(t_c0_w);
  out.T_world_c0 = t_w_c0;
  std::vector<M4> t_w_c(nrows);
  for (size_t i = 0; i < nrows; ++i) t_w_c[i] = geo::matmul(t_w_c0, t_c0_c[i]);
  const std::pair<int64_t, int64_t> span{t_hat.front(), t_hat.back()};

  double rig_radius = 0;
  for (size_t k = 0; k < in.cameras.size(); ++k) {
    const double r = rot::norm_dot(geo::translation_of(in.cameras[k].T_rig_cam));
    if (k == 0 || r > rig_radius) rig_radius = r;
  }
  rig_radius = rig_radius + 0.05;
  traj::SmoothingOptions rig_opts = smooth;
  rig_opts.h_ns = h_ns;
  rig_opts.span_ns = span;
  rig_opts.radius_m = rig_radius;
  rig_opts.center_body_m = {0.0, 0.0, 0.0};
  std::vector<Q> rig_q;
  std::vector<V> rig_p;
  samples_to_world(t_w_c, geo::invert_transform(seq.T_rig_cam), rig_q, rig_p);

  // Objects.
  std::vector<Prepared> prepared;
  for (const int64_t oid : in.object_ids) {
    if (opts.objects && std::find(opts.objects->begin(), opts.objects->end(), oid) == opts.objects->end()) continue;
    const auto cat = in.catalogue.find(oid);
    if (cat == in.catalogue.end()) {
      warnings.push_back("object id " + std::to_string(oid) + " has no catalogue entry; skipped");
      continue;
    }
    const CatalogueObject& obj = cat->second;
    const Track& tr = in.tracks.at(oid);
    if (static_cast<int64_t>(tr.rows.size()) < opts.min_samples) {
      warnings.push_back(obj.name + " (" + std::to_string(oid) + "): only " + std::to_string(tr.rows.size()) +
                         " Vicon samples; skipped");
      continue;
    }
    Prepared pr{oid, &obj, object_mesh(obj, opts), &tr, {}, {}, {}, smooth};
    std::vector<M4> t_w_o(tr.rows.size());
    pr.t.resize(tr.rows.size());
    for (size_t k = 0; k < tr.rows.size(); ++k) {
      const auto row = static_cast<size_t>(tr.rows[k]);
      t_w_o[k] = geo::matmul(t_w_c[row], geo::pose_to_transform(tr.q[k], tr.p[k]));
      pr.t[k] = t_hat[row];
    }
    samples_to_world(t_w_o, geo::invert_transform(pr.om.t_body_subject), pr.q, pr.p);
    pr.fopts.h_ns = h_ns;
    pr.fopts.span_ns = span;
    pr.fopts.radius_m = pr.om.radius;
    pr.fopts.center_body_m = pr.om.centre;
    prepared.push_back(std::move(pr));
  }
  if (smooth.hold_rule == "image") {
    std::vector<M4> t_w_rig(nrows);
    const M4 inv_rig = geo::invert_transform(seq.T_rig_cam);
    for (size_t i = 0; i < nrows; ++i) t_w_rig[i] = geo::matmul(t_w_c[i], inv_rig);
    std::vector<HoldBody> hb;
    for (const Prepared& pr : prepared) hb.push_back({&pr.om.mesh, &pr.q, &pr.p, &pr.track->rows});
    auto [rig_view, views] = hold_views(in.cameras, t_w_rig, hb);
    rig_opts.hold_view = std::move(rig_view);
    for (size_t i = 0; i < prepared.size(); ++i) prepared[i].fopts.hold_view = std::move(views[i]);
  }

  // Fits: independent and deterministic, so the pool changes nothing but the wall time.
  // A failing fit's exception is kept by its index and the lowest is rethrown, as the Python's ordered map raised it:
  // which error a build reports never depends on the worker count or the threads' timing.
  out.fits.resize(prepared.size() + 1);
  std::vector<std::exception_ptr> errors(prepared.size() + 1);
  {
    render::ThreadPool pool(std::max(opts.workers, 1));
    pool.parallel_for(prepared.size() + 1, 1, [&](size_t begin, size_t end) {
      for (size_t j = begin; j < end; ++j) {
        try {
          out.fits[j] = j == 0 ? traj::fit_smoothing(t_hat, rig_q, rig_p, rig_opts)
                               : traj::fit_smoothing(prepared[j - 1].t, prepared[j - 1].q, prepared[j - 1].p,
                                                     prepared[j - 1].fopts);
        } catch (...) {
          const std::string who = j == 0 ? std::string("rig")
                                         : prepared[j - 1].obj->name + " (" + std::to_string(prepared[j - 1].id) + ")";
          errors[j] = named_error(who, std::current_exception());
        }
      }
    });
  }
  for (const std::exception_ptr& e : errors) {
    if (e) std::rethrow_exception(e);
  }
  const traj::FitResult& rig_fit = out.fits[0];
  if (rig_fit.diag.presence_fallback) warnings.push_back(presence_fallback_warning("rig", rig_fit));
  out.fit_names.push_back("rig");
  out.trajectories.push_back(0);
  Tree body_meta = Tree::dict();
  std::string pinned_names;
  for (size_t i = 0; i < prepared.size(); ++i) {
    Prepared& pr = prepared[i];
    traj::FitResult& fit = out.fits[i + 1];
    const CatalogueObject& obj = *pr.obj;
    out.T_body_subject.emplace_back(pr.id, pr.om.t_body_subject);
    out.fit_names.push_back(obj.name);
    const int64_t flipped = fit.diag.gated ? fit.diag.gate.num_branch_rejected : 0;
    if (flipped) {
      warnings.push_back(obj.name + " (" + std::to_string(pr.id) + "): " + std::to_string(flipped) +
                         " Vicon samples of a minority orientation branch rejected (persistent flips of a symmetric "
                         "marker set; the fit bridges them)");
    }
    if (fit.diag.presence_fallback) {
      warnings.push_back(presence_fallback_warning(obj.name + " (" + std::to_string(pr.id) + ")", fit));
    }
    Tree pin_info = Tree::null();
    if (opts.pin_static_px_s && !fit.diag.is_static) {
      // M10-PIN: freeze the fit where the body's Vicon track is scene-static (traj/pin.h); one interval over the whole
      // run makes it static, the decision below.
      std::vector<int64_t> t;
      std::vector<Q> q;
      std::vector<V> p;
      std::vector<double> z;
      for (size_t k = 0; k < fit.keep.size(); ++k) {
        if (!fit.keep[k]) continue;
        t.push_back(pr.t[k]);
        q.push_back(pr.q[k]);
        p.push_back(pr.p[k]);
        // The depth of the body centre along the camera at the sample's row: the third column of T_world_camera.
        const M4& tc = t_w_c[static_cast<size_t>(pr.track->rows[k])];
        const V c = rot::quat_rotate(pr.q[k], pr.om.centre) + pr.p[k];
        z.push_back(tc[2] * (c.x - tc[3]) + tc[6] * (c.y - tc[7]) + tc[10] * (c.z - tc[11]));
      }
      traj::PinOptions po;
      po.limit_px_s = *opts.pin_static_px_s;
      po.focal_px = std::max(seq.fx, seq.fy);
      po.bridge_ns = static_cast<int64_t>(fit.order - 1) * fit.h_ns;
      const traj::PinSamples ps{t, q, p, z, pr.om.centre, surface_rms_radius(pr.om.mesh, pr.om.centre)};
      const traj::PinResult r = traj::pin_intervals(ps, po);
      const double excursion = r.intervals_ns.empty() ? 0.0 : traj::pin_apply(fit, ps, po, r.intervals_ns, r.whole);
      pin_info = Tree::dict();
      pin_info["speed_p99_px_s"] = r.pairs ? Tree::real(r.speed_p99) : Tree::null();
      pin_info["speed_max_px_s"] = r.pairs ? Tree::real(r.speed_max) : Tree::null();
      pin_info["intervals_ns"] = pairs_tree(r.intervals_ns);
      pin_info["whole"] = Tree::boolean(r.whole);
      pin_info["excursion_px"] = Tree::real(excursion);
      if (!r.intervals_ns.empty()) pinned_names += (pinned_names.empty() ? "" : ", ") + obj.name;
    }
    Body body;
    body.id = pr.id;
    body.name = obj.name;
    body.semantic = pr.id == kTabletopId ? 1 : 2;
    std::optional<M4> pose;
    if (fit.diag.is_static) {
      pose = geo::pose_to_transform(fit.rot[0], fit.pos[0]);
    } else if (pr.id == kTabletopId && opts.static_tabletop) {
      const TabletopStatic ts = tabletop_static(pr.q, pr.p, fit.keep, pr.om.centre, pr.om.radius,
                                                pr.fopts.hold_max_disp_m, opts.static_tabletop_fraction);
      Tree tinfo = Tree::dict();
      tinfo["inlier_fraction"] = Tree::real(ts.inlier_fraction);
      tinfo["max_displacement_m"] = Tree::real(ts.max_displacement_m);
      pr.om.info["tabletop_static"] = std::move(tinfo);
      if (ts.pose) {
        pose = ts.pose;
        warnings.push_back("tabletop held static at its robust mean pose although " +
                           pyfmt::percent(1 - ts.inlier_fraction, 1) + " of its gated Vicon samples deviate by up to " +
                           pyfmt::fixed(ts.max_displacement_m * 1e3, 1) + " mm (bistable Vicon solution)");
      }
    }
    if (pose) {
      body.T_world_body = *pose;
    } else {
      out.trajectories.push_back(i + 1);
      body.trajectory = static_cast<int64_t>(out.trajectories.size()) - 1;
      body.T_world_body = geo::identity4();
    }
    body.mesh = std::move(pr.om.mesh);
    body.placeholder = obj.placeholder;
    body.asset_materials = obj.num_materials;
    body.markers = opts.markers;
    Tree info = std::move(pr.om.info);
    info["static"] = Tree::boolean(pose.has_value());
    info["fit"] = fit_summary(fit);
    if (pin_info.kind() == Tree::Kind::kDict) info["pin"] = std::move(pin_info);
    body_meta[obj.name] = std::move(info);
    out.bodies.push_back(std::move(body));
  }
  if (!pinned_names.empty()) {
    warnings.push_back("scene.evimo2.pin_static_px_s " + pyfmt::repr(*opts.pin_static_px_s) +
                       ": frozen where their Vicon image speed stays below it (static over the whole run when one "
                       "interval covers it): " + pinned_names);
  }

  // Cameras and clocks.
  Tree clocks = Tree::dict();
  for (const CameraInput& s : in.cameras) {
    if (s.recording != seq.recording || s.name != seq.name) {
      throw std::invalid_argument("extra camera " + s.directory + " is not the same sequence as " + seq.directory);
    }
    ClockOffset clk = clock_offset(s, opts.clock_reference);
    const SensorPreset preset = sensor_of(s.camera);
    const int64_t offset = clk.offset_ns;
    warnings.insert(warnings.end(), clk.warnings.begin(), clk.warnings.end());
    clocks[s.camera] = std::move(clk.meta);
    Camera cam;
    cam.name = s.camera;
    cam.width = s.width;
    cam.height = s.height;
    cam.fx = s.fx, cam.fy = s.fy, cam.cx = s.cx, cam.cy = s.cy;
    cam.dist = s.dist;
    cam.T_rig_cam = s.T_rig_cam;
    cam.clock_offset_ns = offset;
    cam.sensor = preset.sensor;
    cam.pixel_pitch_m = preset.pitch;
    out.cameras.push_back(std::move(cam));
  }

  out.t_start_ns = span.first + opts.warmup_ns;
  out.t_end_ns = span.second;
  if (out.t_start_ns >= span.second) throw std::invalid_argument(seq.directory + ": sequence shorter than the warm-up");

  // The light anchor (adapter._light_anchor).
  Tree anchor = Tree::null();
  if (opts.lights) {
    std::vector<const traj::FitResult*> trajs;
    for (const size_t f : out.trajectories) trajs.push_back(&out.fits[f]);
    const Anchor a = light_anchor(out.bodies, trajs, rig_fit.pos, out.t_start_ns);
    const V up = a.up, centre = a.centre;
    const std::string& source = a.source;
    if (source.find("gravity unknown") != std::string::npos) warnings.push_back("default point light placed along the " + source);
    anchor = Tree::dict();
    Tree dir = Tree::list(), cen = Tree::list();
    for (const double x : {up.x, up.y, up.z}) dir.push_back(Tree::string(pyfmt::hex(x)));
    for (const double x : {centre.x, centre.y, centre.z}) cen.push_back(Tree::string(pyfmt::hex(x)));
    anchor["direction"] = std::move(dir);
    anchor["centre"] = std::move(cen);
    anchor["source"] = Tree::string(source);
  }
  out.sidecar = Tree::dict();
  out.sidecar["format"] = Tree::string(kRelightFormat);
  out.sidecar["low_light"] = Tree::boolean(ends_with(seq.category, "_ll"));
  Tree rts = Tree::dict();
  rts["a_ns"] = Tree::string(pyfmt::hex(rt.a_ns));
  rts["b_ns"] = Tree::string(pyfmt::hex(rt.b_ns));
  out.sidecar["retiming"] = std::move(rts);
  out.sidecar["anchor"] = std::move(anchor);

  // The meta (lights: null, set by Python's light_rig; each body's asset: null, set from the asset cache).
  Tree& meta = out.meta = Tree::dict();
  Tree src = Tree::dict();
  src["dataset"] = Tree::string("EVIMO2 v2");
  src["sequence"] = Tree::string(seq.directory);
  src["camera"] = Tree::string(seq.camera);
  src["category"] = Tree::string(seq.category);
  src["recording"] = Tree::string(seq.recording);
  src["name"] = Tree::string(seq.name);
  meta["source"] = std::move(src);
  Tree lic = Tree::dict();
  lic["dataset"] = Tree::string("EVIMO2 v2, CC BY-SA 4.0");
  lic["meshes"] = Tree::string("EVIMO tool repository objects (GPL-3.0 repo)");
  lic["redistributable"] = Tree::boolean(false);
  meta["license"] = std::move(lic);
  Tree conv = Tree::dict();
  conv["time"] = Tree::string("re-timed Vicon clock, ns since bag start");
  conv["world"] = std::move(world);
  conv["gt_frame_keys"] = Tree::string("id - first id (G3 t13)");
  conv["instance_ids"] = Tree::string("EVIMO object ids (AST-25)");
  Tree sem = Tree::dict();
  sem["table"] = Tree::integer(1);
  sem["object"] = Tree::integer(2);
  conv["semantic_classes"] = std::move(sem);
  meta["conventions"] = std::move(conv);
  Tree rsum = Tree::dict();
  rsum["a_ns"] = Tree::real(rt.a_ns);
  rsum["b_ns"] = Tree::real(rt.b_ns);
  rsum["residual_std_ns"] = Tree::real(rt.residual_std_ns);
  rsum["num_gaps"] = Tree::integer(rt.num_gaps);
  rsum["num_samples"] = Tree::integer(static_cast<int64_t>(rt.t_ns.size()));
  meta["retiming"] = std::move(rsum);
  meta["clocks"] = std::move(clocks);
  meta["warmup_ns"] = Tree::integer(opts.warmup_ns);
  meta["bodies"] = std::move(body_meta);
  Tree rig = Tree::dict();
  rig["radius_m"] = Tree::real(rig_radius);
  rig["fit"] = fit_summary(rig_fit);
  meta["rig"] = std::move(rig);
  meta["lights"] = Tree::null();
  Tree wl = Tree::list();
  for (const std::string& w : warnings) wl.push_back(Tree::string(w));
  meta["warnings"] = std::move(wl);
  return out;
}

}  // namespace balla::scene::evimo2
