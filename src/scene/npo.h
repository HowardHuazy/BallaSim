// NumPy's order of operations (docs/dev/CPP_PORT.md item 5, E2): the reductions and helpers the Python fitter uses,
// replicated loop by loop so the C++ scene side is bitwise equal to it.
//
// - np.sum of a contiguous float64 array: identity 0.0 plus NumPy's pairwise sum (8 accumulators, blocks of 128)
//   over the whole array; along a short axis (<= 7 elements) or across rows, a sequential sum from 0.0.
// - np.mean = that sum / n; np.median = the mean of the middle one or two values; np.percentile (method linear) and
//   np.std (ddof 0) as NumPy computes them.
// - Python's float ** and np.power go to glibc's pow (py_pow, never folded into x * x); x ** 2 on arrays is x * x.
// - np.linalg.solve / lstsq / eigh and np.polyfit through the LAPACK calls NumPy makes (blas.h).
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace balla::scene::npo {

// NumPy's DOUBLE_pairwise_sum (loops_utils.h): n < 8 sequential, n <= 128 eight accumulators, else halves at a
// multiple of 8.
double pairwise_sum(const double* a, int64_t n, int64_t stride = 1);

// np.sum / np.add.reduce of a contiguous array (or all of a C-contiguous one): 0.0 + pairwise_sum.
inline double sum(std::span<const double> a) { return 0.0 + pairwise_sum(a.data(), static_cast<int64_t>(a.size())); }

// np.mean of a contiguous float64 array.
inline double mean(std::span<const double> a) { return sum(a) / static_cast<double>(a.size()); }

// np.median of float64 or int64 values (the int64 values convert to float64 before the mean, as np.mean does).
double median(std::vector<double> a);
double median(std::span<const int64_t> a);

// np.percentile(a, q) with the default method (linear): virtual index (n - 1) q / 100, then _lerp.
double percentile(std::vector<double> a, double q);

// np.std(a) (ddof 0).
double std0(std::span<const double> a);

// glibc pow, called through a pointer so the compiler never folds pow(x, 2.0) into x * x.
double py_pow(double x, double y);

// np.arange(start, stop, step) for floats: buffer[0] = start, delta = (start + step) - start, x_i = start + i delta.
std::vector<double> arange(double start, double stop, double step);

// np.round(x, decimals) for decimals > 0: rint(x * 10^d) / 10^d.
double round_decimals(double x, int decimals);

// The index of the first minimum (np.argmin; the first NaN wins); nanargmin: NaN counts as +inf, all NaN throws.
// Both throw on an empty input, as NumPy does.
int64_t argmin(std::span<const double> a);
int64_t nanargmin(std::span<const double> a);
// np.argmax: the index of the first maximum (the first NaN wins).
int64_t argmax(std::span<const double> a);

// ndarray.max() / min() (NaN-propagating; empty input throws) and np.maximum (NaN-propagating, the first on ties).
double max(std::span<const double> a);
double min(std::span<const double> a);
inline double maximum(double a, double b) { return (a >= b || a != a) ? a : b; }

// np.linalg.lstsq(a, b, rcond)[0] for a (m, n) row-major, b (m,), m >= n (dgelsd after its workspace query).
std::vector<double> lstsq(std::span<const double> a, int m, int n, std::span<const double> b, double rcond);

// np.polyfit(x, y, 2) for three points: scaled Vandermonde, lstsq with rcond 3 eps, unscaled coefficients.
std::vector<double> polyfit2(const double x[3], const double y[3]);

// np.linalg.solve(a, b) for one 3 x 3 system with three right-hand sides (row-major a and b; dgesv). The result
// overwrites b. Throws when singular (NumPy's LinAlgError).
void solve3(const double a[9], double b[9]);

// np.linalg.eigh(a) for a symmetric 4 x 4 row-major matrix (UPLO 'L', dsyevd): eigenvalues ascending and the
// eigenvectors as columns of v (row-major: v[4 * i + j] is component i of vector j).
void eigh4(const double a[16], double w[4], double v[16]);

}  // namespace balla::scene::npo
