// Cases of the CPU/GPU parity tests (DESIGN §12.1 L1, PHASE1 principle 3).
//
// Each evaluate() is BALLA_HD and runs unchanged on the host (test_parity.cpp) and on the device
// (parity.cu); the tests require bitwise-equal results. This compiles every device-only branch of
// the shared maths (__umulhi in Philox, __brev in Sobol) and checks it against the host branch.
// Result types hold scalars of one type only, so they have no padding and compare with memcmp.
#pragma once

#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "core/camera/ray_lut.h"
#include "core/hd.h"
#include "core/math/vec.h"
#include "core/pmath/explog.h"
#include "core/pmath/smootherstep.h"
#include "core/pmath/special.h"
#include "core/pmath/trig.h"
#include "core/rng/philox.h"
#include "core/rng/purpose.h"
#include "core/rng/sobol.h"
#include "core/rng/subsample.h"
#include "core/rng/warp.h"
#include "core/rng/warp_table.h"
#include "render/shade/bsdf.h"

namespace balla::gpu_test {

// Philox on a raw (counter, key), the purpose-tagged pattern seed of (run_seed, camera, pixel), and
// the eight Owen-Sobol dimensions of sample `index` under that seed.
struct RngCase {
  uint32_t counter[4];
  uint32_t key[2];
  uint32_t camera, pixel;
  uint64_t run_seed;
  uint32_t index;
};

struct RngResult {
  uint32_t philox[4];
  uint32_t pattern_seed;
  uint32_t owen[rng::kSobolDimensions];
};

BALLA_HD inline RngResult evaluate(const RngCase& c) {
  RngResult r{};
  const rng::U32x4 counter{{c.counter[0], c.counter[1], c.counter[2], c.counter[3]}};
  const rng::U32x4 p = rng::philox4x32_10(counter, {c.key[0], c.key[1]});
  for (int i = 0; i < 4; ++i) r.philox[i] = p.v[i];
  r.pattern_seed = rng::subsample_pattern_seed(c.run_seed, c.camera, c.pixel);
  for (uint32_t d = 0; d < rng::kSobolDimensions; ++d) {
    r.owen[d] = rng::owen_sobol(c.index, d, r.pattern_seed);
  }
  return r;
}

// Sub-sample k of a pattern through the fp32 warp table, the device path of ray generation.
struct OffsetCase {
  uint32_t pattern_seed;
  uint32_t k;
};

BALLA_HD inline Vec2<float> evaluate(const rng::WarpTableView<float>& table, const OffsetCase& c) {
  return rng::subsample_offset(table, c.pattern_seed, c.k);
}

template <class T>
struct TrigCase {
  T x, y;
};

template <class T>
struct TrigResult {
  T sincos_s, sincos_c, sin, cos, atan2, asin, acos;
};

// asin and acos take y / 4, which covers [-1, 1] exactly at y = +-4.
template <class T>
BALLA_HD TrigResult<T> evaluate(const TrigCase<T>& c) {
  TrigResult<T> r{};
  pmath::sincos(c.x, &r.sincos_s, &r.sincos_c);
  r.sin = pmath::sin(c.x);
  r.cos = pmath::cos(c.x);
  r.atan2 = pmath::atan2(c.y, c.x);
  r.asin = pmath::asin(T(0.25) * c.y);
  r.acos = pmath::acos(T(0.25) * c.y);
  return r;
}

// Exponentials, logarithms and special functions. x drives the functions of any real argument and,
// through |x|, the residual exponentials (whose direct form overflows for huge arguments); y > 0
// drives the logarithms and erfcx. No input produces a NaN: the payloads of invalid operations
// differ between x86 and the GPU, and the special values returned as constants are covered by the
// host tests.
template <class T>
struct ExpLogCase {
  T x, y;
};

template <class T>
struct ExpLogResult {
  T exp, exp2, expm1, phi1, dawson, smootherstep, log, log2, log1p, erfcx;  // exp2: fp32 only (else 0)
  T resid_exp[5];
};

template <class T>
BALLA_HD ExpLogResult<T> evaluate(const ExpLogCase<T>& c) {
  ExpLogResult<T> r{};
  r.exp = pmath::exp(c.x);
  if constexpr (std::is_same_v<T, float>) r.exp2 = pmath::exp2(c.x);
  r.expm1 = pmath::expm1(c.x);
  r.phi1 = pmath::phi1(c.x);
  r.dawson = pmath::dawson(c.x);
  r.smootherstep = pmath::smootherstep(c.x);
  r.log = pmath::log(c.y);
  r.log2 = pmath::log2(c.y);
  r.log1p = pmath::log1p(c.y);
  r.erfcx = pmath::erfcx(c.y);
  for (int j = 0; j < 5; ++j) r.resid_exp[j] = pmath::resid_exp(j, abs(c.x));
  return r;
}

// Forward projection and its Jacobian of a camera-frame point, and unprojection of a pixel.
// Outputs of a failed call stay 0, and the flags are 1 or 0.
template <class T>
struct CameraCase {
  Vec3<T> point;
  Vec2<T> pixel;
};

template <class T>
struct CameraResult {
  T pixel[2];
  T jacobian[6];
  T ray[3];
  T projected, jacobian_valid, unprojected;
};

template <class T, class Camera>
BALLA_HD CameraResult<T> evaluate_camera(const Camera& cam, const CameraCase<T>& c) {
  CameraResult<T> r{};
  Vec2<T> pix{T(0), T(0)};
  if (camera::project(cam, c.point, &pix)) {
    r.pixel[0] = pix.x;
    r.pixel[1] = pix.y;
    r.projected = T(1);
  }
  camera::Mat23<T> jac{};
  if (camera::project_jacobian(cam, c.point, &jac)) {
    for (int i = 0; i < 6; ++i) r.jacobian[i] = jac.m[i / 3][i % 3];
    r.jacobian_valid = T(1);
  }
  Vec3<T> ray{T(0), T(0), T(0)};
  if (camera::unproject(cam, c.pixel, &ray)) {
    r.ray[0] = ray.x;
    r.ray[1] = ray.y;
    r.ray[2] = ray.z;
    r.unprojected = T(1);
  }
  return r;
}

template <class T>
BALLA_HD CameraResult<T> evaluate(const camera::RadtanCamera<T>& cam, const CameraCase<T>& c) {
  return evaluate_camera(cam, c);
}

template <class T>
BALLA_HD CameraResult<T> evaluate(const camera::KannalaBrandtCamera<T>& cam,
                                  const CameraCase<T>& c) {
  return evaluate_camera(cam, c);
}

BALLA_HD inline Vec3<float> evaluate(const camera::RayLutView<float>& lut,
                                     const Vec2<float>& pixel) {
  return camera::lut_ray(lut, pixel);
}

// The BSDF of radiance v1 (render/shade/bsdf.h, p2f.md §4.3): every table-, floor-, lobe- and LTC-dependent output of
// one configuration, through tables in device memory on the device.
template <class T>
struct BsdfCase {
  Vec3<T> n, v, l;
  T rho, metallic, alpha2, s, alpha2_add, r_s;
};

template <class T>
struct BsdfResult {
  T f, lobe, e_s, a, b, t, mu, alpha_eff;
  T ltc_m00, ltc_m02, ltc_m20, ltc_m22;  // the LTC's inverse matrix (F3c)
  Vec3<T> frame_t, frame_b, ltc_l;       // its frame, and l in its cosine space
};

template <class T>
BALLA_HD BsdfResult<T> evaluate(const render::bsdf::BsdfTables& tables, const BsdfCase<T>& c) {
  namespace bs = render::bsdf;
  BsdfResult<T> r{};
  const bs::BsdfParams<T> p{c.rho, c.metallic, c.alpha2, c.s};
  const bs::ViewTerms<T> vt = bs::view_terms(c.n, c.v);
  const bs::Fresnel<T> fr = bs::fresnel_params(c.rho, c.metallic, c.s);
  const bs::SpecAB<T> ab = bs::spec_ab(tables, vt.t, bs::table_r(c.alpha2));
  r.f = bs::bsdf_eval(tables, p, c.n, c.v, c.l);
  r.lobe = bs::lobe(c.alpha2, fr, c.n, c.v, vt, c.l);
  r.e_s = bs::spec_albedo(fr, ab);
  r.a = ab.a;
  r.b = ab.b;
  r.t = vt.t;
  r.mu = vt.mu;
  r.alpha_eff = bs::alpha_eff(c.alpha2, c.alpha2_add, c.r_s);
  const bs::LtcInverse<T> m = bs::ltc_inverse(tables, vt.t, bs::table_r(c.alpha2));
  const bs::LtcFrame<T> frame = bs::ltc_frame(c.n, c.v);
  r.ltc_m00 = m.m00;
  r.ltc_m02 = m.m02;
  r.ltc_m20 = m.m20;
  r.ltc_m22 = m.m22;
  r.frame_t = frame.t;
  r.frame_b = frame.b;
  r.ltc_l = bs::ltc_apply(m, frame, c.n, c.l);
  return r;
}

// Device evaluation of the same cases (parity.cu). Throw std::runtime_error on CUDA errors.
std::vector<RngResult> evaluate_on_device(std::span<const RngCase> cases);
std::vector<Vec2<float>> evaluate_on_device(const rng::WarpTable<float>& table,
                                            std::span<const OffsetCase> cases);
template <class T>
std::vector<TrigResult<T>> evaluate_on_device(std::span<const TrigCase<T>> cases);
template <class T>
std::vector<ExpLogResult<T>> evaluate_on_device(std::span<const ExpLogCase<T>> cases);
template <class T>
std::vector<CameraResult<T>> evaluate_on_device(const camera::RadtanCamera<T>& cam,
                                                std::span<const CameraCase<T>> cases);
template <class T>
std::vector<CameraResult<T>> evaluate_on_device(const camera::KannalaBrandtCamera<T>& cam,
                                                std::span<const CameraCase<T>> cases);
std::vector<Vec3<float>> evaluate_on_device(const camera::RayLut<float>& lut,
                                            std::span<const Vec2<float>> pixels);
// host_tables: host views of the coefficient tables, copied to device memory for the call.
template <class T>
std::vector<BsdfResult<T>> evaluate_on_device(const render::bsdf::BsdfTables& host_tables,
                                              std::span<const BsdfCase<T>> cases);

}  // namespace balla::gpu_test
