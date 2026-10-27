// Blender cameras in C1 terms (docs/design/phase3/blender.md §4.1, §4.3, §6.1): the render size, the perspective and
// fisheye intrinsics, the calibrations (EVIMO2 presets, Kalibr, OpenCV) with CAM-CALIB, the inverse settings and the
// camera pose.
//
// Provenance (PHASE3.md §0 rule 4): re-derived from the Blender manual (Camera: lens, sensor fit and size, shift,
// panorama types; Output Properties: resolution percentage, pixel aspect) and from headless oracles, not from
// Blender's sources: CV E3 (the perspective formula against Object.calc_matrix_camera), DS d6 (the render size by
// integer division: 641 x 481 at 50 % renders 320 x 240) and the BL-3a probe `ballasim_out/phase3_research/blender/
// bl3a/probes/m3_lights_cameras.py` (4.2.1 and 4.5.3, identical; check analysis/a1_check_m3.py): CV E3's 27
// configurations again (<= 1.8e-4 px), and Cycles renders of small spheres through panorama cameras (<= 0.03 px for
// every sphere the image border does not clip): FISHEYE_EQUISOLID r = 2 f sin(theta / 2) with f = fisheye_lens
// viewfac / sensor, AUTO horizontal only when pax W > pay H and then with sensor_width in both branches, an explicit
// VERTICAL fit with sensor_height, fy = fx / ycor; a panorama's shift moves its centre by shift_x W and shift_y H (not
// by viewfac, as a perspective camera's does); FISHEYE_LENS_POLYNOMIAL theta = -(k0 + k1 r + ... + k4 r^4) with r in
// mm on the sensor. The presets' corner displacement is RI r1_calib_corners' (1.37, 5.39 and 6.05 px). Probe m3b
// (`.../bl3a/probes/m3b_world.py`): resolution_x and resolution_y are at most 65536, resolution_percentage 32767.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/camera/injective_domain.h"
#include "scene/blender/convert.h"
#include "scene/blender/values.h"

namespace balla::scene::blender {

namespace {

using detail::boolean;
using detail::number;
using detail::numbers;
using detail::text;

constexpr double kPi = std::numbers::pi;
constexpr double kMaxFisheyeFov = 220.0 * kPi / 180.0;
// Blender's hard maxima of resolution_x / resolution_y and resolution_percentage (probe m3b, 4.2.1 and 4.5.3).
constexpr double kMaxResolution = 65536.0, kMaxPercentage = 32767.0;
// The largest image side C1 simulates: the sensor model's width and height (src/sensor/sensor_model.cpp).
constexpr int32_t kMaxImageSide = 65535;
// CAM-FOV visits every pixel centre up to this many pixels, else a regular grid of about as many (bound_kind sampled).
constexpr double kFovSamples = 1 << 20;
// Kannala-Brandt coefficients of 2 sin(theta / 2) = theta - theta^3 / 24 + ... (its Taylor series; DS d10: 6.5e-7 px
// at 180 deg, 5.9e-6 px at 220 deg for f = 186.67 px).
constexpr std::array<double, 4> kEquisolid{-1.0 / 24.0, 1.0 / 1920.0, -1.0 / 322560.0, 1.0 / 92897280.0};

std::string fmt(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.4g", v);
  return buf;
}

// The axis the sensor fits, its film size (mm) and viewfac (px), §4.3. A perspective AUTO fit is horizontal when
// pax W >= pay H, a panorama's only when pax W > pay H (probe m3); both use sensor_width; at equality both branches
// give the same numbers.
struct Fit {
  double size_mm, viewfac, ycor;
};
Fit sensor_fit(const std::string& fit, const RenderSize& rs, double sw, double sh, bool panorama,
               const std::string& where) {
  const double ycor = rs.pay / rs.pax;
  bool horizontal = true;
  double size = sw;
  if (fit == "AUTO") {
    horizontal = panorama ? rs.pax * rs.W > rs.pay * rs.H : rs.pax * rs.W >= rs.pay * rs.H;
  } else if (fit == "VERTICAL") {
    horizontal = false;
    size = sh;
  } else if (fit != "HORIZONTAL") {
    throw std::invalid_argument("BRG-PROTO: " + where + ".sensor_fit " + fit + " is not AUTO, HORIZONTAL or VERTICAL");
  }
  if (!(size > 0.0)) throw std::invalid_argument("BRG-PROTO: " + where + ": the sensor size must be > 0");
  return {size, horizontal ? static_cast<double>(rs.W) : ycor * rs.H, ycor};
}

// The outer corners of the image (pixel centres +- 0.5 px).
std::array<std::array<double, 2>, 4> outer_corners(int32_t W, int32_t H) {
  return {{{-0.5, -0.5}, {W - 0.5, -0.5}, {-0.5, H - 0.5}, {W - 0.5, H - 0.5}}};
}

camera::RadtanCamera<double> radtan_of(const render::Camera& c) {
  return {c.fx, c.fy, c.cx, c.cy, c.dist[0], c.dist[1], c.dist[2], c.dist[3], c.dist.size() > 4 ? c.dist[4] : 0.0,
          0.0, 0.0};
}
camera::KannalaBrandtCamera<double> kb_of(const render::Camera& c) {
  return {c.fx, c.fy, c.cx, c.cy, c.dist[0], c.dist[1], c.dist[2], c.dist[3], 0.0};
}

// The largest field angle over the image (pixel centres +- 0.5 px) of a C1 camera; std::domain_error when the model is
// not injective there (core/camera/injective_domain.h, as the renderer requires).
double theta_max(const render::Camera& c) {
  if (c.model == render::CameraModelKind::kKannalaBrandt) {
    return camera::kannala_brandt_injective_domain(kb_of(c), c.width, c.height, 0.5).theta_img;
  }
  if (std::all_of(c.dist.begin(), c.dist.end(), [](double d) { return d == 0.0; })) {
    double r = 0.0;   // a pinhole: the outer corners
    for (const auto& [u, v] : outer_corners(c.width, c.height)) {
      r = std::max(r, std::hypot((u - c.cx) / c.fx, (v - c.cy) / c.fy));
    }
    return std::atan(r);
  }
  return std::atan(camera::radtan_injective_domain(radtan_of(c), c.width, c.height, 0.5).r_img);
}

// The fraction of the pixel centres (u, v) whose field angle theta_of(u, v) exceeds `half_fov` (or is NaN): every
// centre up to kFovSamples pixels, else the centres of a regular grid with stride s (returned in *stride).
template <class F>
double beyond_fraction(int32_t W, int32_t H, double half_fov, F&& theta_of, int64_t* stride) {
  const double px = static_cast<double>(W) * H;
  const auto s = static_cast<int64_t>(std::max(1.0, std::ceil(std::sqrt(px / kFovSamples))));
  int64_t out = 0, all = 0;
  for (int64_t v = s / 2; v < H; v += s) {
    for (int64_t u = s / 2; u < W; u += s) {
      ++all;
      if (!(theta_of(static_cast<double>(u), static_cast<double>(v)) <= half_fov)) ++out;
    }
  }
  *stride = s;
  return all > 0 ? static_cast<double>(out) / static_cast<double>(all) : 0.0;
}

// The pixels by which a calibration's distortion moves the image corners (pixel centres) off its own pinhole part
// (RI r1_calib_corners); corners beyond 90 deg (kb) are skipped.
double distortion_corner_px(const render::Camera& c) {
  double worst = 0.0;
  const bool kb = c.model == render::CameraModelKind::kKannalaBrandt;
  double r_lim = 0.0, theta_lim = 0.0;
  if (kb) {
    theta_lim = camera::kannala_brandt_injective_domain(kb_of(c), c.width, c.height, 0.5).theta_lim;
  } else {
    r_lim = camera::radtan_injective_domain(radtan_of(c), c.width, c.height, 0.5).r_lim;
  }
  for (const auto& [u, v] : std::array<std::array<double, 2>, 4>{
           {{0.0, 0.0}, {c.width - 1.0, 0.0}, {0.0, c.height - 1.0}, {c.width - 1.0, c.height - 1.0}}}) {
    Vec3<double> ray{};
    bool ok = false;
    if (kb) {
      camera::KannalaBrandtCamera<double> m = kb_of(c);
      m.theta_lim = theta_lim;
      ok = camera::unproject(m, Vec2<double>{u, v}, &ray) && ray.z > 0.0;
    } else {
      camera::RadtanCamera<double> m = radtan_of(c);
      m.r_lim = r_lim;
      ok = camera::unproject(m, Vec2<double>{u, v}, &ray);
    }
    if (ok) worst = std::max(worst, std::hypot(c.fx * ray.x / ray.z + c.cx - u, c.fy * ray.y / ray.z + c.cy - v));
  }
  return worst;
}

bool intrinsic_path(const std::string& p) {
  static const char* const kPaths[] = {"lens", "angle", "angle_x", "angle_y", "sensor_width", "sensor_height",
                                       "sensor_fit", "shift_x", "shift_y", "type", "panorama_type", "fisheye_fov",
                                       "fisheye_lens", "ortho_scale", "NLA"};
  for (const char* k : kPaths) {
    if (p == k) return true;
  }
  return p.rfind("fisheye_polynomial_k", 0) == 0;
}

}  // namespace

RenderSize render_size(const ExchangeIn& ex) {
  const std::string where = "render";
  RenderSize rs;
  // Blender's own bounds (a value beyond them is no Blender value); W, H <= 65536 x 32767 / 100 fit int32.
  const auto integer = [&](const char* key, double fallback, double max, const char* what) {
    const double v = number(ex.render, key, fallback, where);
    if (v != std::floor(v) || !(v >= 1.0) || v > max) detail::wrong(where, key, what);
    return static_cast<int64_t>(v);
  };
  const int64_t pct = integer("resolution_percentage", 100.0, kMaxPercentage, "an integer in [1, 32767]");
  rs.W = static_cast<int32_t>(integer("resolution_x", -1.0, kMaxResolution, "an integer in [1, 65536]") * pct / 100);
  rs.H = static_cast<int32_t>(integer("resolution_y", -1.0, kMaxResolution, "an integer in [1, 65536]") * pct / 100);
  rs.pax = number(ex.render, "pixel_aspect_x", 1.0, where);
  rs.pay = number(ex.render, "pixel_aspect_y", 1.0, where);
  if (!(rs.pax > 0.0) || !(rs.pay > 0.0)) throw std::invalid_argument("BRG-PROTO: render.pixel_aspect must be > 0");
  rs.border = boolean(ex.render, "use_border", false, where);
  return rs;
}

BlenderCamera camera_settings(const render::Camera& c, double sensor_width_mm) {
  if (c.width < 1 || c.height < 1 || !(c.fx > 0.0) || !(c.fy > 0.0) || !(sensor_width_mm > 0.0)) {
    throw std::invalid_argument("camera_settings: the camera needs a size, fx, fy > 0 and a sensor width > 0");
  }
  BlenderCamera b;
  const double W = c.width, H = c.height, ratio = c.fx / c.fy;
  b.sensor_width_mm = sensor_width_mm;
  b.lens_mm = c.fx * sensor_width_mm / W;
  b.pixel_aspect_x = ratio >= 1.0 ? 1.0 : 1.0 / ratio;
  b.pixel_aspect_y = ratio >= 1.0 ? ratio : 1.0;
  b.shift_x = (W / 2.0 - 0.5 - c.cx) / W;
  b.shift_y = (c.cy + 0.5 - H / 2.0) * ratio / W;
  b.res_x = c.width;
  b.res_y = c.height;
  return b;
}

Calibration calibration_camera(const Tree& rec) {
  const std::string where = "calibration";
  const std::string source = text(rec, "source", "", where), label = text(rec, "name", "", where);
  const std::string what = source + (label.empty() ? "" : " " + label);
  if (const std::string err = text(rec, "error", "", where); !err.empty()) {
    throw std::invalid_argument("CAM-CALIB: the " + what + " calibration cannot be read: " + err);
  }
  const std::string model = text(rec, "model", "pinhole", where), dist = text(rec, "distortion", "none", where);
  if (model != "pinhole") {
    throw std::invalid_argument("CAM-MODEL: the " + what + " calibration's " + model + " camera model is refused "
                                "(unified, double sphere, extended unified; §4.3)");
  }
  Calibration out;
  render::Camera& cam = out.cam;
  const double w = number(rec, "width", 0.0, where), h = number(rec, "height", 0.0, where);
  const std::vector<double> in = numbers(rec, "intrinsics", -1, where);
  std::vector<double> k = numbers(rec, "coefficients", -1, where);
  if (!(w >= 1.0) || !(h >= 1.0) || w != std::floor(w) || h != std::floor(h) || in.size() != 4 || !(in[0] > 0.0) ||
      !(in[1] > 0.0)) {
    throw std::invalid_argument("CAM-CALIB: the " + what + " calibration needs its resolution and intrinsics "
                                "[fx, fy, cx, cy] with fx, fy > 0");
  }
  cam.width = static_cast<int32_t>(w);
  cam.height = static_cast<int32_t>(h);
  cam.fx = in[0];
  cam.fy = in[1];
  cam.cx = in[2];
  cam.cy = in[3];
  const auto extra_zero = [&k](size_t from) {
    return std::all_of(k.begin() + static_cast<std::ptrdiff_t>(std::min(from, k.size())), k.end(),
                       [](double x) { return x == 0.0; });
  };
  // The radtan family: OpenCV's rational, thin-prism and tilted models are radtan when their terms beyond k3 are
  // zero; non-zero terms there are refused for every name (plumb_bob with 8 coefficients is OpenCV's rational).
  if (dist == "radtan" || dist == "plumb_bob" || dist == "none" || dist == "rational" || dist == "thin_prism" ||
      dist == "tilted") {
    if (dist == "none" && !extra_zero(0)) {
      throw std::invalid_argument("CAM-CALIB: the " + what + " calibration has coefficients but no distortion model");
    }
    if (!extra_zero(5)) {
      throw std::invalid_argument("CAM-MODEL: the " + what + " calibration's " + dist + " distortion has non-zero "
                                  "terms beyond k3 (OpenCV rational, thin-prism and tilted terms are refused; §4.3)");
    }
    if (dist != "none" && k.size() < 4) {
      throw std::invalid_argument("CAM-CALIB: the " + what + " calibration's radtan needs k1, k2, p1, p2[, k3]");
    }
    k.resize(5, 0.0);
    cam.model = render::CameraModelKind::kRadtan;
  } else if (dist == "equidistant" || dist == "equi" || dist == "fisheye") {
    if (k.size() != 4) {
      throw std::invalid_argument("CAM-CALIB: the " + what + " calibration's equidistant model needs k1..k4");
    }
    cam.model = render::CameraModelKind::kKannalaBrandt;
  } else {
    throw std::invalid_argument("CAM-MODEL: the " + what + " calibration's " + dist + " distortion is refused (OpenCV "
                                "rational, thin-prism and tilted terms, Kalibr fov; §4.3)");
  }
  cam.dist = k;
  out.sensor = text(rec, "sensor", "", where);
  if (const double p = number(rec, "pixel_pitch_m", 0.0, where); p > 0.0) out.pixel_pitch_m = p;
  if (const double f = number(rec, "f_number", 0.0, where); f > 0.0) out.f_number = f;
  return out;
}

PolyFit fit_lens_polynomial(const std::array<double, 5>& k, double r_max_mm, double theta_cap) {
  constexpr int kSamples = 2000;
  std::vector<double> th, r;
  for (int i = 0; i < kSamples; ++i) {
    const double rr = r_max_mm * i / (kSamples - 1);
    const double t = -(k[0] + rr * (k[1] + rr * (k[2] + rr * (k[3] + rr * k[4]))));
    if (!(t >= 0.0) || t > theta_cap) continue;
    if (!th.empty() && !(t > th.back())) {
      throw std::invalid_argument("the lens polynomial's angle does not increase with the radius over the image");
    }
    th.push_back(t);
    r.push_back(rr);
  }
  const size_t m = th.size();
  if (m < 16) throw std::invalid_argument("the lens polynomial covers no field angle in [0, fisheye_fov / 2]");
  // Least squares r = sum_j c_j theta^(2j+1) by Householder QR on column-scaled columns.
  constexpr size_t n = 5;
  std::vector<double> A(m * n), b = r;
  std::array<double, n> scale{};
  for (size_t i = 0; i < m; ++i) {
    double p = th[i];
    for (size_t j = 0; j < n; ++j, p *= th[i] * th[i]) {
      A[i * n + j] = p;
      scale[j] = std::max(scale[j], std::abs(p));
    }
  }
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) A[i * n + j] /= scale[j];
  }
  for (size_t j = 0; j < n; ++j) {
    double norm = 0.0;
    for (size_t i = j; i < m; ++i) norm += A[i * n + j] * A[i * n + j];
    norm = std::sqrt(norm);
    if (!(norm > 0.0)) throw std::invalid_argument("the lens polynomial fit is singular");
    const double alpha = A[j * n + j] > 0.0 ? -norm : norm;
    std::vector<double> v(m - j);
    for (size_t i = j; i < m; ++i) v[i - j] = A[i * n + j];
    v[0] -= alpha;
    double vv = 0.0;
    for (const double x : v) vv += x * x;
    if (!(vv > 0.0)) continue;
    for (size_t c = j; c < n; ++c) {
      double s = 0.0;
      for (size_t i = j; i < m; ++i) s += v[i - j] * A[i * n + c];
      for (size_t i = j; i < m; ++i) A[i * n + c] -= 2.0 * s / vv * v[i - j];
    }
    double s = 0.0;
    for (size_t i = j; i < m; ++i) s += v[i - j] * b[i];
    for (size_t i = j; i < m; ++i) b[i] -= 2.0 * s / vv * v[i - j];
  }
  std::array<double, n> c{};
  for (size_t j = n; j-- > 0;) {
    double s = b[j];
    for (size_t q = j + 1; q < n; ++q) s -= A[j * n + q] * c[q];
    c[j] = s / A[j * n + j];
  }
  for (size_t j = 0; j < n; ++j) c[j] /= scale[j];
  PolyFit out;
  out.f_mm = c[0];
  if (!(out.f_mm > 0.0)) throw std::invalid_argument("the lens polynomial fits no positive focal length");
  for (size_t j = 0; j < 4; ++j) out.k[j] = c[j + 1] / c[0];
  for (size_t i = 0; i < m; ++i) {
    const double t2 = th[i] * th[i];
    const double fit = out.f_mm * th[i] * (1.0 + t2 * (out.k[0] + t2 * (out.k[1] + t2 * (out.k[2] + t2 * out.k[3]))));
    out.max_err_mm = std::max(out.max_err_mm, std::abs(fit - r[i]));
  }
  out.samples = static_cast<int>(m);
  out.theta_max = th.back();
  return out;
}

BlenderModel blender_camera(const Tree& p, const RenderSize& rs, double fisheye_fit_tol_px, const std::string& where) {
  BlenderModel bm;
  render::Camera& cam = bm.cam;
  cam.width = rs.W;
  cam.height = rs.H;
  const auto refuse = [&bm](const std::string& msg, double bound = 0.0, BoundKind bk = BoundKind::kNone) {
    bm.refusal = ReportItem{"CAM-MODEL", "refuse", "", msg, bound, bk};
    return bm;
  };
  const std::string type = text(p, "type", "PERSP", where);
  const std::string fit = text(p, "sensor_fit", "AUTO", where);
  const double sw = number(p, "sensor_width", 36.0, where), sh = number(p, "sensor_height", 24.0, where);
  const double shift_x = number(p, "shift_x", 0.0, where), shift_y = number(p, "shift_y", 0.0, where);
  if (type == "PERSP") {
    const Fit f = sensor_fit(fit, rs, sw, sh, false, where);
    const double lens = number(p, "lens", 50.0, where);
    if (!(lens > 0.0)) throw std::invalid_argument("BRG-PROTO: " + where + ".props.lens must be > 0");
    cam.model = render::CameraModelKind::kRadtan;
    cam.fx = lens * f.viewfac / f.size_mm;
    cam.fy = cam.fx / f.ycor;
    cam.cx = rs.W / 2.0 - 0.5 - shift_x * f.viewfac;
    cam.cy = rs.H / 2.0 - 0.5 + shift_y * f.viewfac / f.ycor;
    cam.dist.assign(5, 0.0);
    bm.pitch_m = lens * 1e-3 / std::sqrt(cam.fx * cam.fy);
    return bm;
  }
  if (type != "PANO") return refuse("the " + type + " camera type is refused (orthographic, custom; §4.3)");
  const std::string fisheye = text(p, "panorama_type", "", where);
  const Fit f = sensor_fit(fit, rs, sw, sh, true, where);
  const double ppm_x = f.viewfac / f.size_mm, ppm_y = ppm_x / f.ycor;   // px per mm on the sensor
  const double fov = number(p, "fisheye_fov", kPi, where);
  cam.model = render::CameraModelKind::kKannalaBrandt;
  cam.cx = rs.W / 2.0 - 0.5 - shift_x * rs.W;   // probe m3: a panorama's shift scales with W and H
  cam.cy = rs.H / 2.0 - 0.5 + shift_y * rs.H;
  bm.pitch_m = 1e-3 / std::sqrt(ppm_x * ppm_y);
  double r_max = 0.0;   // the outer corners' distance from the centre on the sensor (mm)
  for (const auto& [u, v] : outer_corners(rs.W, rs.H)) {
    r_max = std::max(r_max, std::hypot((u - cam.cx) / ppm_x, (v - cam.cy) / ppm_y));
  }
  // CAM-FOV: the pixel centres beyond fisheye_fov / 2, which Cycles renders black and we render.
  int64_t stride = 1;
  const auto uncovered = [&](auto&& theta_of_mm) {
    return beyond_fraction(rs.W, rs.H, fov / 2.0, [&](double u, double v) {
      return theta_of_mm(std::hypot((u - cam.cx) / ppm_x, (v - cam.cy) / ppm_y));
    }, &stride);
  };
  double frac = 0.0;
  if (fisheye == "FISHEYE_EQUISOLID") {
    const double lens = number(p, "fisheye_lens", 10.5, where);
    if (!(lens > 0.0)) throw std::invalid_argument("BRG-PROTO: " + where + ".props.fisheye_lens must be > 0");
    if (fov > kMaxFisheyeFov + 1e-6) return refuse("fisheye_fov " + fmt(fov * 180.0 / kPi) + " deg > 220 deg (§4.3)");
    cam.fx = lens * ppm_x;
    cam.fy = lens * ppm_y;
    cam.dist.assign(kEquisolid.begin(), kEquisolid.end());
    // r = 2 lens sin(theta / 2) ends at 2 lens (theta = 180 deg): pixels beyond have no ray. The kb model keeps the
    // monotone margin of core/camera/injective_domain.h (theta <= 0.99 pi).
    if (const double r_lim = 2.0 * lens * std::sin(0.99 * kPi / 2.0); r_max > r_lim) {
      return refuse("the image corners lie " + fmt(r_max) + " mm from the centre, beyond the equisolid circle " +
                    fmt(r_lim) + " mm (2 fisheye_lens): those pixels see no ray (a longer fisheye_lens or a smaller "
                    "image)", r_max, BoundKind::kMeasured);
    }
    frac = uncovered([lens](double r) { return 2.0 * std::asin(std::min(1.0, r / (2.0 * lens))); });
  } else if (fisheye == "FISHEYE_LENS_POLYNOMIAL") {
    std::array<double, 5> kp{};
    for (size_t q = 0; q < 5; ++q) {
      const std::string key = "fisheye_polynomial_k" + std::to_string(q);
      if (detail::at(p, key.c_str()) == nullptr) {
        throw std::invalid_argument("BRG-PROTO: " + where + ".props." + key + " is missing");
      }
      kp[q] = number(p, key.c_str(), 0.0, where);
    }
    PolyFit pf;
    try {
      pf = fit_lens_polynomial(kp, r_max, std::min(fov / 2.0, kPi));
    } catch (const std::invalid_argument& e) {
      return refuse(e.what());
    }
    const double err_px = pf.max_err_mm * std::max(ppm_x, ppm_y);
    if (err_px > fisheye_fit_tol_px) {
      return refuse("the lens polynomial fits Kannala-Brandt within " + fmt(err_px) +
                        " px > scene.blender.fisheye_fit_tol_px " + fmt(fisheye_fit_tol_px),
                    err_px, BoundKind::kMeasured);
    }
    bm.notes.push_back({"CAM-MODEL", "approx", "",
                        "the lens polynomial as Kannala-Brandt within " + fmt(err_px) + " px (" +
                            std::to_string(pf.samples) + " radii up to " + fmt(pf.theta_max * 180.0 / kPi) + " deg)",
                        err_px, BoundKind::kMeasured});
    cam.fx = pf.f_mm * ppm_x;
    cam.fy = pf.f_mm * ppm_y;
    cam.dist.assign(pf.k.begin(), pf.k.end());
    frac = uncovered([&kp](double r) { return -(kp[0] + r * (kp[1] + r * (kp[2] + r * (kp[3] + r * kp[4])))); });
  } else {
    return refuse("the " + fisheye + " panorama is refused (equidistant, equirectangular, mirror ball, central "
                  "cylindrical; §4.3)");
  }
  if (frac > 0.0) {
    bm.notes.push_back({"CAM-FOV", "info", "",
                        fmt(100.0 * frac) + " % of the pixels lie beyond fisheye_fov / 2: black in Cycles, rendered "
                        "here" + (stride > 1 ? " (pixel centres on a grid every " + std::to_string(stride) + " px)" :
                                               std::string()),
                        frac, stride > 1 ? BoundKind::kSampled : BoundKind::kMeasured});
  }
  return bm;
}

void convert_cameras(const ExchangeIn& ex, const ConvertOptions& o, const std::vector<bool>& is_static, Statics& st) {
  if (ex.cameras.empty()) return;
  const double k = ex.scale_length();
  const RenderSize rs = render_size(ex);
  for (size_t j = 0; j < ex.cameras.size(); ++j) {
    const CameraIn& c = ex.cameras[j];
    const ObjectIn& ob = ex.objects[static_cast<size_t>(c.object)];
    const std::string where = "cameras[" + std::to_string(j) + "]";
    if (ob.type != "CAMERA") {
      throw std::invalid_argument("BRG-PROTO: " + where + ".object is a " + ob.type + " object, not a camera");
    }
    const Tree& p = c.props;
    std::vector<ReportItem> items;
    bool refused = false;
    const auto refuse = [&](const char* code, const std::string& msg, double bound = 0.0,
                            BoundKind bk = BoundKind::kNone) {
      items.push_back({code, "refuse", ob.name_full, msg, bound, bk});
      refused = true;
    };
    CameraStatic cs;
    cs.index = static_cast<int32_t>(j);
    cs.object = c.object;
    cs.simulate = c.simulate;
    cs.rig_object = c.object;
    cs.T_rig_cam = {1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1};
    cs.clip_start_m = k * number(p, "clip_start", 0.1, where);
    render::Camera& cam = cs.cam;
    cam.name = st.camera_names[j];
    cam.width = rs.W;
    cam.height = rs.H;

    // CAM-ANIM: intrinsics that change over the span (and the f-number, when it comes from the DoF stop).
    const Tree* dof = detail::at(p, "dof");
    const bool use_dof = dof != nullptr && boolean(*dof, "use_dof", false, where + ".props.dof");
    std::string anim;
    for (const std::string& path : detail::animated_paths(ex, c.animation)) {
      if (intrinsic_path(path) || (use_dof && path.rfind("dof.", 0) == 0)) {
        anim += (anim.empty() ? "" : ", ") + path;
      }
    }
    if (!anim.empty()) refuse("CAM-ANIM", "animated intrinsics (" + anim + ")");
    if (rs.border) refuse("CAM-MODEL", "a render border (crop) is not a camera model");
    if (rs.W < 1 || rs.H < 1) refuse("CAM-MODEL", "the render size is zero at this resolution percentage");
    if (rs.W > kMaxImageSide || rs.H > kMaxImageSide) {
      refuse("CAM-MODEL", "the render size " + std::to_string(rs.W) + " x " + std::to_string(rs.H) + " exceeds " +
                              std::to_string(kMaxImageSide) + " pixels per side (the sensor model's limit)");
    }

    // The rig pose at the export frame: the polar factor of matrix_world (a camera's scale moves no ray).
    std::array<double, 9> A{};
    for (size_t r = 0; r < 3; ++r) {
      for (size_t q = 0; q < 3; ++q) A[3 * r + q] = ob.matrix_world[4 * r + q];
    }
    const std::array<double, 3> t{k * ob.matrix_world[3], k * ob.matrix_world[7], k * ob.matrix_world[11]};
    try {
      const Polar pol = polar(A);
      if (pol.mirror) refuse("CAM-MODEL", "a mirrored camera transform (negative scale)");
      const std::array<double, 9>& R = pol.R;
      cs.T_world_export = {R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1};
      if (is_static[static_cast<size_t>(c.object)]) cs.T_world_static = cs.T_world_export;
    } catch (const std::invalid_argument&) {
      refuse("CAM-MODEL", "a singular camera transform (zero scale)");
    }

    // Blender's own camera (§4.3).
    double pitch = 0.0;
    if (!refused) {
      BlenderModel bm = blender_camera(p, rs, o.fisheye_fit_tol_px, where);
      for (ReportItem& it : bm.notes) {
        it.object = ob.name_full;
        items.push_back(std::move(it));
      }
      if (bm.refusal) {
        bm.refusal->object = ob.name_full;
        items.push_back(*bm.refusal);
        refused = true;
      } else {
        bm.cam.name = cam.name;
        cam = std::move(bm.cam);
        pitch = bm.pitch_m;
      }
    }
    if (!refused) cs.blender = {cam.fx, cam.fy, cam.cx, cam.cy};

    // The calibration is authoritative (§4.3): it replaces Blender's model; CAM-CALIB compares the pinholes.
    std::optional<double> preset_pitch, preset_f;
    if (!refused && c.calibration.kind() == Tree::Kind::kDict) {
      try {
        Calibration cal = calibration_camera(c.calibration);
        if (cal.cam.width != rs.W || cal.cam.height != rs.H) {
          refuse("CAM-CALIB", "the calibration is for " + std::to_string(cal.cam.width) + " x " +
                                  std::to_string(cal.cam.height) + " pixels, the render has " +
                                  std::to_string(rs.W) + " x " + std::to_string(rs.H));
        } else {
          double moved = 0.0;
          for (const auto& [u, v] : std::array<std::array<double, 2>, 4>{
                   {{0.0, 0.0}, {rs.W - 1.0, 0.0}, {0.0, rs.H - 1.0}, {rs.W - 1.0, rs.H - 1.0}}}) {
            const double x = (u - cs.blender[2]) / cs.blender[0], y = (v - cs.blender[3]) / cs.blender[1];
            moved = std::max(moved, std::hypot(cal.cam.fx * x + cal.cam.cx - u, cal.cam.fy * y + cal.cam.cy - v));
          }
          if (moved > o.calib_tol_px) {
            items.push_back({"CAM-CALIB", "warn", ob.name_full,
                             "Blender's pinhole differs from the calibration's by " + fmt(moved) +
                                 " px at a corner (> scene.blender.calib_tol_px " + fmt(o.calib_tol_px) +
                                 "): matched frames need Apply calibration or a warp",
                             moved, BoundKind::kMeasured});
          }
          cal.cam.name = cam.name;
          if (std::any_of(cal.cam.dist.begin(), cal.cam.dist.end(), [](double d) { return d != 0.0; })) {
            const double corner = distortion_corner_px(cal.cam);
            items.push_back({"CAM-CALIB", "info", ob.name_full,
                             "the calibration's distortion moves the image corners by up to " + fmt(corner) +
                                 " px off its pinhole (the frames area's warp)",
                             corner, BoundKind::kMeasured});
          }
          cam = cal.cam;
          cs.sensor = cal.sensor;
          preset_pitch = cal.pixel_pitch_m;
          preset_f = cal.f_number;
          cs.calibrated = true;
        }
      } catch (const std::domain_error& e) {
        refuse("CAM-MODEL", std::string("the calibration: ") + e.what());
      } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        if (msg.rfind("BRG-PROTO", 0) == 0) throw;
        const size_t colon = msg.find(": ");
        const std::string code = colon == std::string::npos ? "CAM-CALIB" : msg.substr(0, colon);
        refuse(code == "CAM-MODEL" ? "CAM-MODEL" : "CAM-CALIB",
               colon == std::string::npos ? msg : msg.substr(colon + 2));
      }
    }
    if (!refused) {
      try {
        cs.theta_max = theta_max(cam);
      } catch (const std::domain_error& e) {
        refuse("CAM-MODEL", std::string("the camera model is not injective over the image: ") + e.what());
      }
    }

    // Sensor, pixel pitch and f-number (§4.3): the preset's, else the camera's family, Blender's film and the DoF stop.
    if (cs.sensor.empty()) cs.sensor = c.sensor_family;
    if (use_dof) {
      Tree set = Tree::dict();
      set["dof.use_dof"] = Tree::boolean(false);
      Tree m = Tree::dict();
      m["kind"] = Tree::string("camera");
      m["object"] = Tree::string(ob.name_full);
      m["set"] = std::move(set);
      st.matched.push_back(std::move(m));
      // Warned under both policies (PHASE3.md §3: Blender depth of field is warned until G2a's thin lens; §4.3),
      // the one warning without a bound under strict besides MAT-SPEC's sampled one (§1 guarantee 3).
      items.push_back({"CAM-DOF", "warn", ob.name_full,
                       "depth of field is not modelled (events are pinhole until G2a's thin lens): f_number from "
                       "aperture_fstop; matched frames turn it off",
                       0.0, BoundKind::kNone});
    }
    if (!refused) {
      if (preset_pitch) {
        cs.pixel_pitch_m = *preset_pitch;
      } else {
        cs.pixel_pitch_m = pitch;
        items.push_back({"CAM-PITCH", "info", ob.name_full,
                         "pixel_pitch_m " + fmt(pitch * 1e6) + " um: Blender's film over its pixels along the fitted "
                         "axis (lens / sqrt(fx fy))",
                         pitch, BoundKind::kMeasured});
      }
      cs.f_number = preset_f ? *preset_f : use_dof ? number(*dof, "aperture_fstop", 2.8, where + ".props.dof") : 2.0;
      if (!(cs.f_number > 0.0)) refuse("CAM-MODEL", "the f-number must be > 0");
    }
    if (!c.simulate) {   // a camera that is not simulated never refuses the scene: it is left out
      for (ReportItem& it : items) {
        if (it.severity == "refuse") {
          it.severity = "info";
          it.message += " (not simulated: the camera is left out)";
        }
      }
    }
    st.report.insert(st.report.end(), items.begin(), items.end());
    if (!refused) st.cameras.push_back(std::move(cs));
  }
}

}  // namespace balla::scene::blender
