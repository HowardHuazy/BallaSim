// The motion session's clock (time_map.h; blender.md §4.2). Every conversion is exact rational arithmetic in 128-bit
// integers, overflow-checked (a time outside the representable range throws std::range_error).
#include "scene/motion/time_map.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

namespace balla::scene::motion {

namespace {

__extension__ typedef __int128 i128;
__extension__ typedef unsigned __int128 u128;

i128 mul(i128 a, i128 b) {
  i128 r;
  if (__builtin_mul_overflow(a, b, &r)) throw std::range_error("TimeMap: time out of range");
  return r;
}

i128 add(i128 a, i128 b) {
  i128 r;
  if (__builtin_add_overflow(a, b, &r)) throw std::range_error("TimeMap: time out of range");
  return r;
}

i128 pow2(int k) {
  if (k < 0 || k > 120) throw std::range_error("TimeMap: time out of range");
  return i128{1} << k;
}

// round_half_even(n / d), d > 0.
i128 div_round_even(i128 n, i128 d) {
  i128 q = n / d, r = n % d;
  if (r < 0) {
    q -= 1;
    r += d;
  }
  const i128 twice = r + r;
  if (twice > d || (twice == d && (q & 1) != 0)) q += 1;
  return q;
}

int64_t to_i64(i128 v) {
  if (v > INT64_MAX || v < INT64_MIN) throw std::range_error("TimeMap: time out of range");
  return static_cast<int64_t>(v);
}

// c = m 2^e exactly, |m| < 2^24.
void decompose(float c, int64_t* m, int* e) {
  if (!std::isfinite(c)) throw std::invalid_argument("TimeMap: frame value is not finite");
  int ex = 0;
  const float fr = std::frexp(c, &ex);
  *m = static_cast<int64_t>(std::ldexp(fr, 24));
  *e = ex - 24;
}

// The float32 nearest n / d (d > 0), ties to even; 0 below about 2^-32 in magnitude (requests round s there to 0).
float nearest_f32(i128 n, i128 d) {
  if (n == 0) return 0.0f;
  const bool neg = n < 0;
  const u128 a = neg ? static_cast<u128>(-n) : static_cast<u128>(n), den = static_cast<u128>(d);
  u128 q = a / den, r = a % den, rd = den;
  int e = 0;
  if (q >= (u128{1} << 24)) {  // drop the low bits of the integer part into the remainder
    int sh = 0;
    while ((q >> sh) >= (u128{1} << 24)) ++sh;
    if (sh > 8) throw std::range_error("TimeMap: frame out of range");  // |c| >= 2^32
    r = (q & ((u128{1} << sh) - 1)) * den + r;
    rd = den << sh;
    q >>= sh;
    e = sh;
  } else {
    while (q < (u128{1} << 23)) {  // long division, one bit at a time (r < den, so 2r cannot overflow)
      r <<= 1;
      q <<= 1;
      if (r >= den) {
        r -= den;
        q |= 1;
      }
      if (--e < -55) return 0.0f;
    }
  }
  if (r + r > rd || (r + r == rd && (q & 1) != 0)) ++q;
  if (q == (u128{1} << 24)) {
    q >>= 1;
    ++e;
  }
  const float c = std::ldexp(static_cast<float>(static_cast<uint32_t>(q)), e);
  return neg ? -c : c;
}

double half_ulp32(float x) {
  x = std::fabs(x);
  return 0.5 * (static_cast<double>(std::nextafter(x, INFINITY)) - static_cast<double>(x));
}

std::string hex32(uint32_t v) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "0x%08x", v);
  return buf;
}

const Tree& field(const Tree& t, const char* key) {
  const Tree* v = t.find(key);
  if (!v) throw std::invalid_argument(std::string("timeline record: missing ") + key);
  return *v;
}

[[noreturn]] void not_a(const char* key, const char* what) {
  throw std::invalid_argument(std::string("timeline record: ") + key + " is not " + what);
}

int64_t int_of(const Tree& v, const char* key) {
  if (v.kind() != Tree::Kind::kInt) not_a(key, "an int");
  return v.as_int();
}

int64_t int_field(const Tree& t, const char* key) { return int_of(field(t, key), key); }

int32_t int32_field(const Tree& t, const char* key) {
  const int64_t v = int_field(t, key);
  if (v < INT32_MIN || v > INT32_MAX) not_a(key, "an int32");
  return static_cast<int32_t>(v);
}

bool bool_field(const Tree& t, const char* key) {
  const Tree& v = field(t, key);
  if (v.kind() != Tree::Kind::kBool) not_a(key, "a bool");
  return v.as_bool();
}

// A float, or an int (JSON writes 1.0 as 1 in some tools), as its exact double.
double float_field(const Tree& t, const char* key) {
  const Tree& v = field(t, key);
  if (v.kind() == Tree::Kind::kFloat) return v.as_float();
  if (v.kind() == Tree::Kind::kInt) return static_cast<double>(v.as_int());
  not_a(key, "a number");
}

const std::string& string_field(const Tree& t, const char* key) {
  const Tree& v = field(t, key);
  if (v.kind() != Tree::Kind::kString) not_a(key, "a string");
  return v.as_string();
}

// "0x" and 1 to 8 hex digits (record() writes 8), the whole string.
uint32_t hex_field(const Tree& t, const char* key) {
  const std::string& s = string_field(t, key);
  const bool ok = s.size() > 2 && s.size() <= 10 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') &&
                  s.find_first_not_of("0123456789abcdefABCDEF", 2) == std::string::npos;
  if (!ok) not_a(key, "a 32-bit hex string");
  return static_cast<uint32_t>(std::stoul(s.substr(2), nullptr, 16));
}

constexpr i128 kGiga = 1000000000;

}  // namespace

TimeMap::TimeMap(const TimeSettings& s) {
  fps_ = s.fps;
  fps_base_bits_ = s.fps_base_bits;
  const float b = std::bit_cast<float>(s.fps_base_bits);
  if (!(std::isfinite(b) && b > 0.0f)) throw std::invalid_argument("TimeMap: fps_base must be finite and > 0");
  int64_t m = 0;
  int e = 0;
  decompose(b, &m, &e);
  if (e >= 0) {
    if (e > 38) throw std::invalid_argument("TimeMap: fps_base out of range");
    base_num_ = m << e;
    base_den_ = 1;
  } else {
    if (e < -62) throw std::invalid_argument("TimeMap: fps_base out of range");
    base_num_ = m;
    base_den_ = int64_t{1} << -e;
    while (base_den_ > 1 && base_num_ % 2 == 0) {
      base_num_ /= 2;
      base_den_ /= 2;
    }
  }
  if (s.fps_base_snap) {  // p / q within 1/2 ulp32 of fps_base, q in {1, 1000, 1001} in that order
    for (const int64_t q : {int64_t{1}, int64_t{1000}, int64_t{1001}}) {
      const double qb = static_cast<double>(q) * static_cast<double>(b);  // exact: 24 + 10 bits
      const double p = std::nearbyint(qb);
      if (p >= 1.0 && std::fabs(p - qb) <= static_cast<double>(q) * half_ulp32(b)) {
        const auto pi = static_cast<int64_t>(p);
        const int64_t g = std::gcd(pi, q);
        base_num_ = pi / g;
        base_den_ = q / g;
        break;
      }
    }
  }
  if (s.frame_map_old < 1 || s.frame_map_new < 1) throw std::invalid_argument("TimeMap: frame_map values must be >= 1");
  framelen_ = static_cast<float>(s.frame_map_old) / static_cast<float>(s.frame_map_new);
  frame_ref_ = s.frame_start;
  frame_start_ = s.frame_start;
  frame_end_ = s.frame_end;
  motion_blur_ = s.motion_blur;
  shutter_ = s.motion_blur_shutter;
  position_ = s.motion_blur_position;
  check();
}

void TimeMap::check() const {
  if (fps_ < 1 || fps_ > 32767) throw std::invalid_argument("TimeMap: fps must be in [1, 32767]");
  if (base_num_ < 1 || base_den_ < 1) throw std::invalid_argument("TimeMap: fps_base must be > 0");
  if (!(std::isfinite(framelen_) && framelen_ > 0.0f)) throw std::invalid_argument("TimeMap: invalid framelen");
  if (frame_end_ < frame_start_) throw std::invalid_argument("TimeMap: frame_end precedes frame_start");
  if (frame_start_ < -(1 << 21) || frame_end_ > (1 << 21)) throw std::invalid_argument("TimeMap: frames out of range");
  if (!(std::isfinite(shutter_) && shutter_ >= 0.0f)) throw std::invalid_argument("TimeMap: invalid shutter");
  if (position_ != "START" && position_ != "CENTER" && position_ != "END") {
    throw std::invalid_argument("TimeMap: motion_blur_position must be START, CENTER or END");
  }
}

float TimeMap::realized(int32_t F, float s) { return static_cast<float>(F) + s; }

float TimeMap::reported(float c) const { return c * framelen_; }

int64_t TimeMap::to_ns(float c) const {
  int64_t m = 0;
  int e = 0;
  decompose(c, &m, &e);
  i128 num, scale;  // c - F_ref = num / scale
  if (e >= 0) {
    num = add(mul(m, pow2(e)), -i128{frame_ref_});
    scale = 1;
  } else {
    scale = pow2(-e);
    num = add(m, -mul(frame_ref_, scale));
  }
  const i128 n = mul(mul(num, kGiga), base_num_), d = mul(mul(scale, fps_), base_den_);
  return to_i64(add(t_ref_ns_, div_round_even(n, d)));
}

int64_t TimeMap::frames_to_ns(int64_t num, int64_t den) const {
  if (den <= 0) throw std::invalid_argument("TimeMap: frames_to_ns needs den > 0");
  const i128 n = mul(mul(num, kGiga), base_num_), d = mul(mul(den, fps_), base_den_);
  return to_i64(add(t_ref_ns_, div_round_even(n, d)));
}

int64_t TimeMap::frame_to_ns(double frame) const {
  if (!std::isfinite(frame)) throw std::invalid_argument("TimeMap: frame value is not finite");
  const double d = frame - static_cast<double>(frame_ref_);
  if (d == 0.0) return t_ref_ns_;
  int ex = 0;
  const double fr = std::frexp(d, &ex);
  auto m = static_cast<int64_t>(std::ldexp(fr, 53));  // d = m 2^e exactly, |m| < 2^53
  int e = ex - 53;
  while (e < 0 && m % 2 == 0) {
    m /= 2;
    ++e;
  }
  if (e < -85) {  // keep the denominator within 128 bits: round m to the 2^-85 grid (< 1e-25 frames)
    const int sh = -85 - e;
    m = sh >= 63 ? 0 : static_cast<int64_t>(std::nearbyint(std::ldexp(static_cast<double>(m), -sh)));
    e = -85;
  }
  i128 num = mul(mul(m, kGiga), base_num_), den = mul(fps_, base_den_);
  if (e >= 0) {
    num = mul(num, pow2(e));
  } else {
    den = mul(den, pow2(-e));
  }
  return to_i64(add(t_ref_ns_, div_round_even(num, den)));
}

void TimeMap::request(int64_t t_ns, int32_t* F, float* s) const {
  const i128 d = mul(kGiga, base_num_);
  const i128 n = add(mul(frame_ref_, d), mul(add(t_ns, -i128{t_ref_ns_}), mul(fps_, base_den_)));
  const float c = nearest_f32(n, d);
  if (!(std::fabs(c) < 2147483520.0f)) throw std::range_error("TimeMap: frame out of range");
  int32_t f = static_cast<int32_t>(std::floor(c));
  float sub = c - static_cast<float>(f);  // exact for |c| >= 1 and F = 0 (Sterbenz); rounded for c in (-1, 0)
  if (sub >= 1.0f) {  // c in (-2^-25, 0): c + 1 rounds to 1
    f += 1;
    sub = 0.0f;
  }
  if (sub > 0.0f && sub < 0x1p-30f) sub = 0.0f;
  *F = f;
  *s = sub;
}

double TimeMap::period_ns() const {
  return 1e9 * static_cast<double>(base_num_) / (static_cast<double>(fps_) * static_cast<double>(base_den_));
}

double TimeMap::resolution_ns(int32_t F) const {
  const auto a = static_cast<float>(std::max<int64_t>(std::llabs(static_cast<int64_t>(F)), 1));
  return half_ulp32(a) * period_ns();
}

int64_t TimeMap::t_end_ns() const {
  const int64_t tf = to_ns(static_cast<float>(frame_end_));
  return to_i64(mul(i128{time::window_index(tf)} + 1, time::kWindowNs));
}

uint32_t TimeMap::framelen_bits() const { return std::bit_cast<uint32_t>(framelen_); }

Tree TimeMap::record() const {
  const int64_t tf = to_ns(static_cast<float>(frame_end_)), te = t_end_ns();
  int32_t f_lo = 0, f_hi = 0;
  float s_unused = 0.0f;
  request(-2 * time::kWindowNs, &f_lo, &s_unused);
  request(te, &f_hi, &s_unused);
  Tree r = Tree::dict();
  r["fps"] = Tree::integer(fps_);
  r["fps_base_bits"] = Tree::string(hex32(fps_base_bits_));
  r["base_num"] = Tree::integer(base_num_);
  r["base_den"] = Tree::integer(base_den_);
  r["frame_ref"] = Tree::integer(frame_ref_);
  r["t_ref_ns"] = Tree::integer(t_ref_ns_);
  r["framelen_bits"] = Tree::string(hex32(framelen_bits()));
  r["frame_start"] = Tree::integer(frame_start_);
  r["frame_end"] = Tree::integer(frame_end_);
  r["t_frame_end_ns"] = Tree::integer(tf);
  r["bundle_span"] = Tree::ints({-2 * time::kWindowNs, te});
  r["run_span"] = Tree::ints({0, te});
  r["run_span_extends_ns"] = Tree::integer(te - tf);
  r["q_max_ns"] = Tree::real(std::max(resolution_ns(f_lo), resolution_ns(f_hi)));
  Tree mb = Tree::dict();
  mb["use"] = Tree::boolean(motion_blur_);
  mb["shutter"] = Tree::real(static_cast<double>(shutter_));
  mb["position"] = Tree::string(position_);
  r["motion_blur"] = std::move(mb);
  return r;
}

TimeMap TimeMap::from_record(const Tree& rec) {
  if (rec.kind() != Tree::Kind::kDict) throw std::invalid_argument("timeline record: not a dict");
  TimeMap t;
  t.fps_ = int32_field(rec, "fps");
  t.fps_base_bits_ = hex_field(rec, "fps_base_bits");
  t.base_num_ = int_field(rec, "base_num");
  t.base_den_ = int_field(rec, "base_den");
  t.frame_ref_ = int32_field(rec, "frame_ref");
  t.t_ref_ns_ = int_field(rec, "t_ref_ns");
  t.framelen_ = std::bit_cast<float>(hex_field(rec, "framelen_bits"));
  t.frame_start_ = int32_field(rec, "frame_start");
  t.frame_end_ = int32_field(rec, "frame_end");
  const Tree& mb = field(rec, "motion_blur");
  if (mb.kind() != Tree::Kind::kDict) not_a("motion_blur", "a dict");
  t.motion_blur_ = bool_field(mb, "use");
  const double shutter = float_field(mb, "shutter");
  t.shutter_ = static_cast<float>(shutter);
  if (!(static_cast<double>(t.shutter_) == shutter)) not_a("motion_blur.shutter", "a float32");
  t.position_ = string_field(mb, "position");
  t.check();
  // the inputs among themselves (§4.2): F_ref is frame_start and t_ref = 0; B is fps_base's exact value, or its p / q
  // with fps_base_snap (the record does not say which, so either is accepted), reduced as the constructor reduces it
  if (t.frame_ref_ != t.frame_start_) throw std::invalid_argument("timeline record: frame_ref is not frame_start");
  if (t.t_ref_ns_ != 0) throw std::invalid_argument("timeline record: t_ref_ns is not 0");
  {
    TimeSettings s;
    s.fps = t.fps_;
    s.fps_base_bits = t.fps_base_bits_;
    s.frame_start = t.frame_start_;
    s.frame_end = t.frame_end_;
    bool base_ok = false;
    for (const bool snap : {false, true}) {
      s.fps_base_snap = snap;
      const TimeMap m(s);
      base_ok = base_ok || (m.base_num_ == t.base_num_ && m.base_den_ == t.base_den_);
    }
    if (!base_ok) throw std::invalid_argument("timeline record: inconsistent base_num / base_den for fps_base_bits");
  }
  const Tree again = t.record();
  if (!(float_field(rec, "q_max_ns") == again.find("q_max_ns")->as_float())) {
    throw std::invalid_argument("timeline record: inconsistent q_max_ns");
  }
  for (const char* key : {"t_frame_end_ns", "run_span_extends_ns"}) {
    if (int_field(again, key) != int_field(rec, key)) {
      throw std::invalid_argument(std::string("timeline record: inconsistent ") + key);
    }
  }
  for (const char* key : {"bundle_span", "run_span"}) {
    const auto& a = field(again, key).items();
    const Tree& span = field(rec, key);
    if (span.kind() != Tree::Kind::kList || span.items().size() != 2) not_a(key, "a list of two ints");
    const auto& b = span.items();
    if (a[0].as_int() != int_of(b[0], key) || a[1].as_int() != int_of(b[1], key)) {
      throw std::invalid_argument(std::string("timeline record: inconsistent ") + key);
    }
  }
  return t;
}

}  // namespace balla::scene::motion
