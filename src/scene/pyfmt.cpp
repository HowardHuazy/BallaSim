// Contains C++ translations of CPython (format_float_short's repr and float.hex), under the PSF License:
// see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/pyfmt.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace balla::scene::pyfmt {

std::string repr(double x) {
  if (std::isnan(x)) return "nan";
  if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
  // Shortest round-trip digits (what CPython's dtoa mode 0 gives) and the decimal point position decpt:
  // x = 0.d1 d2 ... x 10^decpt.
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), std::fabs(x), std::chars_format::scientific);
  const std::string sci(buf, res.ptr);
  const size_t e = sci.find('e');
  std::string digits;
  for (size_t i = 0; i < e; ++i) {
    if (sci[i] != '.') digits += sci[i];
  }
  const int decpt0 = std::atoi(sci.c_str() + e + 1) + 1;
  // format_float_short, 'r' with Py_DTSF_ADD_DOT_0.
  const bool use_exp = decpt0 <= -4 || decpt0 > 16;
  const int exp = decpt0 - 1;
  const int decpt = use_exp ? 1 : decpt0;
  const auto len = static_cast<int>(digits.size());
  const int vstart = decpt <= 0 ? decpt - 1 : 0;
  int vend = len;
  vend = !use_exp ? (vend > decpt ? vend : decpt + 1) : (vend > decpt ? vend : decpt);
  std::string out = std::signbit(x) ? "-" : "";
  if (decpt <= 0) {
    out.append(static_cast<size_t>(decpt - vstart), '0');
    out += '.';
    out.append(static_cast<size_t>(-decpt), '0');
  } else {
    out.append(static_cast<size_t>(-vstart), '0');
  }
  if (0 < decpt && decpt <= len) {
    out += digits.substr(0, static_cast<size_t>(decpt));
    out += '.';
    out += digits.substr(static_cast<size_t>(decpt));
  } else {
    out += digits;
  }
  if (len < decpt) {
    out.append(static_cast<size_t>(decpt - len), '0');
    out += '.';
    out.append(static_cast<size_t>(vend - decpt), '0');
  } else {
    out.append(static_cast<size_t>(vend - len), '0');
  }
  if (out.back() == '.') out.pop_back();
  if (use_exp) {
    char eb[16];
    std::snprintf(eb, sizeof(eb), "e%+.02d", exp);
    out += eb;
  }
  return out;
}

std::string repr_list(const std::vector<double>& v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out += ", ";
    out += repr(v[i]);
  }
  return out + "]";
}

std::string fixed(double x, int prec) {
  if (std::isnan(x)) return "nan";
  if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
  // As printf's %.*f (correctly rounded, ties to even on the exact binary value), which CPython's format matches.
  std::vector<char> buf(static_cast<size_t>(prec) + 400);
  const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), x, std::chars_format::fixed, prec);
  return std::string(buf.data(), res.ptr);
}

std::string signed_fixed(double x, int prec) {
  const std::string s = fixed(x, prec);
  return s[0] == '-' ? s : "+" + s;
}

std::string percent(double x, int prec) { return fixed(x * 100.0, prec) + "%"; }

std::string hex(double x) {
  if (std::isnan(x)) return "nan";
  if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
  if (x == 0.0) return std::signbit(x) ? "-0x0.0p+0" : "0x0.0p+0";
  static const char kHex[] = "0123456789abcdef";
  int e = 0;
  double m = std::frexp(std::fabs(x), &e);
  const int shift = 1 - std::max(-1021 - e, 0);   // DBL_MIN_EXP = -1021
  m = std::ldexp(m, shift);
  e -= shift;
  std::string s;
  s += kHex[static_cast<int>(m)];
  m -= static_cast<int>(m);
  s += '.';
  for (int i = 0; i < 13; ++i) {
    m *= 16.0;
    s += kHex[static_cast<int>(m)];
    m -= static_cast<int>(m);
  }
  const char esign = e < 0 ? '-' : '+';
  return std::string(x < 0 ? "-0x" : "0x") + s + "p" + esign + std::to_string(e < 0 ? -e : e);
}

}  // namespace balla::scene::pyfmt
