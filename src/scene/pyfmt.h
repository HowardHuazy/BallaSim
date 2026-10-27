// Python's float formatting, for the strings the adapter writes into the bundle meta, its warnings and errors:
// repr(x), format(x, '.Nf'), format(x, '.N%') and float.hex(x), character for character.
#pragma once

#include <string>
#include <vector>

namespace balla::scene::pyfmt {

// repr(float): the shortest round-trip digits; fixed notation for 1e-4 <= |x| < 1e16 (with ".0" when integral),
// else d.ddde+XX.
std::string repr(double x);
// repr of a list of floats ([a, b]).
std::string repr_list(const std::vector<double>& v);
// format(x, f".{prec}f") and format(x, f".{prec}%").
std::string fixed(double x, int prec);
std::string signed_fixed(double x, int prec);   // format(x, f"+.{prec}f")
std::string percent(double x, int prec);
// float.hex(x): 0x1.<13 hex digits>p+E.
std::string hex(double x);

}  // namespace balla::scene::pyfmt
