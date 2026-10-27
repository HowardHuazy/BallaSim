// APS frames: photometry inputs, the frame loop and the digests (docs/design/phase3/frames.md §4.10, §3.6).
#include "frames/aps_frames.h"

#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#include "config/sha256.h"
#include "core/camera/injective_domain.h"
#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "core/camera/relative_illumination.h"
#include "core/camera/solid_angle.h"
#include "render/thread_pool.h"

namespace balla::frames {

namespace {

constexpr double kElementaryCharge = 1.602176634e-19;  // C (SI, exact)

__extension__ typedef __int128 i128;

std::string num(double v) {
  char buf[32];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, r.ptr);
}

[[noreturn]] void bad_value(const std::string& key, const std::string& value, const std::string& rule) {
  throw FramesError("FRM-VALUE", "frames.aps." + key + " = " + value + ": must be " + rule);
}
void at_least(const std::string& key, double v, double lo, bool open) {
  if (!std::isfinite(v) || (open ? !(v > lo) : !(v >= lo))) bad_value(key, num(v), (open ? "> " : ">= ") + num(lo));
}

// The engine's unit ray of each pixel centre (CameraGeometry: unproject, then divide by sqrt((x^2 + y^2) + z^2)) and
// its gain (Camera::init_photometry): optics.vignetting_law cos4, (c^2)(c^2) with c the ray's z, or jacobian,
// cos(theta) dOmega/dA of `open`, the model without a domain limit (the rays lie inside the limited `model`'s).
template <class Cam>
std::vector<float> law_gain(const Cam& model, const Cam& open, bool jacobian, Roi roi) {
  std::vector<float> out;
  out.reserve(static_cast<size_t>(roi.x1 - roi.x0) * static_cast<size_t>(roi.y1 - roi.y0));
  for (int32_t y = roi.y0; y < roi.y1; ++y) {
    for (int32_t x = roi.x0; x < roi.x1; ++x) {
      Vec3<double> r{};
      if (!camera::unproject(model, Vec2<double>{static_cast<double>(x), static_cast<double>(y)}, &r)) {
        throw std::invalid_argument("frames aps: pixel positions outside the camera's injective domain");
      }
      const double n = std::sqrt((r.x * r.x + r.y * r.y) + r.z * r.z);
      const double c = r.z / n;
      if (!jacobian) {
        if (!(c > 0)) {
          throw std::invalid_argument("frames aps: a pixel looks 90 degrees or more off axis, where cos^4 vignetting "
                                      "does not apply (optics.vignetting: false)");
        }
        out.push_back(static_cast<float>((c * c) * (c * c)));
        continue;
      }
      double g = 0.0;
      if (!camera::vignetting_gain(open, Vec3<double>{r.x / n, r.y / n, c}, &g)) {
        throw std::invalid_argument("frames aps: a pixel looks 90 degrees or more off axis, where cos(theta) dOmega/dA "
                                    "vignetting does not apply (optics.vignetting: false)");
      }
      out.push_back(static_cast<float>(g));
    }
  }
  return out;
}

}  // namespace

sensor::ApsParams aps_params(const ApsSettings& s, uint64_t run_sensor_seed) {
  if (s.bits < 8 || s.bits > 16) bad_value("bits", std::to_string(s.bits), "in [8, 16]");
  at_least("full_well_e", s.full_well_e, 0, true);
  at_least("read_noise_e", s.read_noise_e, 0, false);
  at_least("dark_current_e_s", s.dark_current_e_s, 0, false);
  at_least("prnu", s.prnu, 0, false);
  at_least("dsnu_e", s.dsnu_e, 0, false);
  if (s.gain_dn_per_e) at_least("gain_dn_per_e", *s.gain_dn_per_e, 0, true);
  at_least("offset_dn", s.offset_dn, 0, false);
  if (s.seed && *s.seed < 0) bad_value("seed", std::to_string(*s.seed), ">= 0");
  sensor::ApsParams p;
  p.bits = s.bits;
  p.full_well_e = s.full_well_e;
  p.read_noise_e = s.read_noise_e;
  p.dark_e_s = s.dark_current_e_s;
  p.prnu = s.prnu;
  p.dsnu_e = s.dsnu_e;
  p.gain_dn_e = s.gain_dn_per_e ? *s.gain_dn_per_e : (std::ldexp(1.0, s.bits) - 1.0) / s.full_well_e;
  if (!(std::isfinite(p.gain_dn_e) && p.gain_dn_e > 0)) bad_value("gain_dn_per_e", num(p.gain_dn_e), "> 0");
  p.offset_dn = s.offset_dn;
  p.seed = s.seed ? static_cast<uint64_t>(*s.seed) : run_sensor_seed;
  return p;
}

bool vignetting_on(std::optional<bool> vignetting, std::string_view photometry_mode) {
  return vignetting.value_or(photometry_mode != "relative");
}

std::vector<float> vignetting_gain(const render::Camera& cam, double support_px, Roi roi, bool jacobian,
                                   const std::vector<double>& relative_illumination) {
  if (roi.x0 < 0 || roi.y0 < 0 || roi.x1 > cam.width || roi.y1 > cam.height || roi.x0 >= roi.x1 || roi.y0 >= roi.y1) {
    throw FramesError("FRM-VALUE", "frames aps: the ROI must be non-empty and inside the image");
  }
  if (const char* why = camera::relative_illumination_error(relative_illumination)) {
    throw std::invalid_argument(std::string("frames aps: optics.relative_illumination: ") + why);
  }
  constexpr double kOpen = std::numeric_limits<double>::infinity();
  std::vector<double> k = cam.dist;
  std::vector<float> gain;
  if (cam.model == render::CameraModelKind::kRadtan) {
    k.resize(5, 0.0);
    camera::RadtanCamera<double> c{cam.fx, cam.fy, cam.cx, cam.cy, k[0], k[1], k[2], k[3], k[4], 0.0, 0.0};
    c.r_lim = camera::radtan_injective_domain(c, cam.width, cam.height, support_px).r_lim;
    camera::RadtanCamera<double> open = c;
    open.r_lim = kOpen;
    gain = law_gain(c, open, jacobian, roi);
  } else {
    k.resize(4, 0.0);
    camera::KannalaBrandtCamera<double> c{cam.fx, cam.fy, cam.cx, cam.cy, k[0], k[1], k[2], k[3], 0.0};
    c.theta_lim = camera::kannala_brandt_injective_domain(c, cam.width, cam.height, support_px).theta_lim;
    camera::KannalaBrandtCamera<double> open = c;
    open.theta_lim = kOpen;
    gain = law_gain(c, open, jacobian, roi);
  }
  if (!relative_illumination.empty()) {  // Camera::init_photometry's G(r~) on the law's gain, in full-frame pixels
    const double r2c = camera::corner_radius_sq(cam.width, cam.height, cam.cx, cam.cy);
    size_t i = 0;
    for (int32_t y = roi.y0; y < roi.y1; ++y) {
      for (int32_t x = roi.x0; x < roi.x1; ++x, ++i) {
        const double dx = static_cast<double>(x) - cam.cx, dy = static_cast<double>(y) - cam.cy;
        const double u = r2c > 0.0 ? (dx * dx + dy * dy) / r2c : 0.0;
        gain[i] = static_cast<float>(static_cast<double>(gain[i]) *
                                     camera::relative_illumination(relative_illumination[0],
                                                                   relative_illumination[1], u));
      }
    }
  }
  return gain;
}

ApsFrames aps_frames(const sensor::ApsParams& params, const ApsPhotometry& ph, uint32_t camera, int32_t full_width,
                     Roi roi, const RowTiming& timing, std::span<const int64_t> frame_soe_ns,
                     std::span<const float> mono, int threads) {
  params.validate();
  if (!(std::isfinite(ph.illuminance_scale_fA) && ph.illuminance_scale_fA > 0)) {
    throw FramesError("FRM-APS-SCALE", "frames aps: the camera has no photocurrent scale (run.json "
                                       "cameras[].illuminance_scale; a frames-only camera has none)");
  }
  if (!(std::isfinite(ph.veiling_glare) && ph.veiling_glare >= 0)) {
    throw std::invalid_argument("frames aps: veiling_glare must be finite and >= 0");
  }
  if (roi.x0 < 0 || roi.y0 < 0 || roi.x1 > full_width || roi.y1 > timing.height || roi.x0 >= roi.x1 ||
      roi.y0 >= roi.y1) {
    throw FramesError("FRM-VALUE", "frames aps: the ROI must be non-empty and inside the image");
  }
  sensor::check_layout(camera, full_width, timing.height);  // before the workers (aps_frame checks each row too)
  if (timing.exposure_ns < 0 || timing.row_dt_ns < 0) throw std::invalid_argument("frames aps: negative timing");
  const auto h = static_cast<size_t>(roi.y1 - roi.y0), w = static_cast<size_t>(roi.x1 - roi.x0);
  const size_t n = frame_soe_ns.size(), npix = h * w;
  if (mono.size() != n * npix) throw std::invalid_argument("frames aps: mono must be n x H x W of the ROI");
  if (!ph.gain.empty() && ph.gain.size() != npix) throw std::invalid_argument("frames aps: gain must be H x W");
  // Row r of frame k starts at frame_soe[k] + rho(r) Delta, rho(r) = r (top-down) or H - 1 - r (bottom-up). Exact in
  // __int128 (|rho Delta| < 2^94) and checked against the APS counter range [-2^56, 3 2^56) before the loop, as
  // timing.cpp keeps R + T inside int64: unbounded file values (row_dt_ns, t_true_soe_ns) never overflow.
  const i128 delta = timing.shutter == Shutter::kRolling ? timing.row_dt_ns : 0;
  const auto order = [&](int32_t row) -> i128 {
    return timing.readout == Readout::kTopDown ? row : timing.height - 1 - row;
  };
  const auto row_soe = [&](size_t k, int32_t row) { return static_cast<i128>(frame_soe_ns[k]) + order(row) * delta; };
  if (static_cast<i128>(timing.height - 1) * delta + timing.exposure_ns > std::numeric_limits<int64_t>::max()) {
    throw std::invalid_argument("frames aps: the readout span plus the exposure exceed 2^63 ns");
  }
  for (size_t k = 0; k < n; ++k) {
    for (const int32_t row : {roi.y0, roi.y1 - 1}) {  // rho is monotonic in the row, so the ROI's ends bound it
      const i128 t = row_soe(k, row);
      if (t < -sensor::kApsTimeBias || t >= 3 * static_cast<i128>(sensor::kApsTimeBias)) {
        throw std::invalid_argument("frames aps: frame " + std::to_string(k) + " row " + std::to_string(row) +
                                    " starts its exposure outside [-2^56, 3 2^56) ns, the APS counter range");
      }
    }
  }

  ApsFrames out;
  out.dn.assign(n * npix, 0);
  out.veil_mean.assign(n, 0.0);
  const unsigned hw = std::thread::hardware_concurrency();
  render::ThreadPool pool(threads > 0 ? threads : static_cast<int>(hw > 0 ? hw : 1));
  const auto gain = [&](size_t i) { return ph.gain.empty() ? 1.0 : static_cast<double>(ph.gain[i]); };
  // E_k: Neumaier sums in row-major order, one frame per task.
  pool.parallel_for(n, 1, [&](size_t b, size_t e) {
    for (size_t k = b; k < e; ++k) {
      double sum = 0.0, comp = 0.0;
      int64_t count = 0;
      for (size_t i = 0; i < npix; ++i) {
        const double v = gain(i) * static_cast<double>(mono[k * npix + i]);
        if (!std::isfinite(v)) continue;
        const double t = sum + v;
        comp += std::fabs(sum) >= std::fabs(v) ? (sum - t) + v : (v - t) + sum;
        sum = t;
        ++count;
      }
      out.veil_mean[k] = count > 0 ? (sum + comp) / static_cast<double>(count) : 0.0;
    }
  });
  const double exposure_s = static_cast<double>(timing.exposure_ns) * 1e-9;
  const double scale = ph.illuminance_scale_fA * 1e-15 * exposure_s / kElementaryCharge;
  std::vector<int64_t> fallbacks(n * h, 0);
  pool.parallel_for(n * h, 8, [&](size_t b, size_t e) {
    std::vector<double> sig(w);
    for (size_t item = b; item < e; ++item) {
      const size_t k = item / h, i = item % h;
      const int32_t row = roi.y0 + static_cast<int32_t>(i);
      const auto t_soe = static_cast<int64_t>(row_soe(k, row));  // inside the counter range (checked above)
      const double veil = ph.veiling_glare * out.veil_mean[k];
      for (size_t x = 0; x < w; ++x) {
        const size_t p = i * w + x;
        sig[x] = scale * (gain(p) * static_cast<double>(mono[k * npix + p]) + veil);
      }
      const sensor::ApsRegion region{full_width, roi.x0, row, static_cast<int32_t>(w), 1};
      fallbacks[item] = sensor::aps_frame(params, camera, region, sig.data(), &t_soe, exposure_s,
                                          out.dn.data() + k * npix + i * w);
    }
  });
  for (int64_t f : fallbacks) out.normal_fallbacks += f;
  if (timing.exposure_ns % 1000 != 0 || timing.exposure_ns > kApsMaxExposureNs) {
    out.warnings.push_back({"FRM-APS-EXPOSURE", "frames aps: exposure " + std::to_string(timing.exposure_ns) +
                                                    " ns is off the 1 us grid or above 4.194303 s, which a DAVIS "
                                                    "sensor cannot set"});
  }
  if (out.normal_fallbacks > 0) {
    out.warnings.push_back({"FRM-APS-RNG", "frames aps: " + std::to_string(out.normal_fallbacks) +
                                               " Poisson draws used the normal approximation after 64 rejections"});
  }
  return out;
}

std::string frames_digest_base(const DigestParts& d) {
  return config::sha256_hex("ballasim.frames-digest-base/1\nformat=" + d.format + "\nbundle=" + d.bundle_digest +
                            "\ncamera=" + d.camera_digest + "\nvalues=\n" + d.values + "plan=" + d.plan_digest +
                            "\nmanifest=" + d.manifest_sha256 + "\n");
}

std::string aps_canonical(const sensor::ApsParams& p) {
  return "bits=" + std::to_string(p.bits) + "\ndark_current_e_s=" + num(p.dark_e_s) + "\ndsnu_e=" + num(p.dsnu_e) +
         "\nfull_well_e=" + num(p.full_well_e) + "\ngain_dn_per_e=" + num(p.gain_dn_e) + "\noffset_dn=" +
         num(p.offset_dn) + "\nprnu=" + num(p.prnu) + "\nread_noise_e=" + num(p.read_noise_e) + "\nseed=" +
         std::to_string(p.seed) + "\n";
}

std::string frames_digest(std::string_view base, const sensor::ApsParams* aps) {
  return config::sha256_hex("ballasim.frames-digest/1\nbase=" + std::string(base) + "\naps=\n" +
                            (aps != nullptr ? aps_canonical(*aps) : std::string("none\n")));
}

}  // namespace balla::frames
