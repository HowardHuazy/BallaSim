// Contains C++ translations of OpenBLAS (the Haswell dgemm kernel's order for 3x3 products), under their BSD licences:
// see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/blas.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>

namespace balla::scene::blas {

namespace {

std::mutex g_pin_mutex;
int g_pin_depth = 0;
int g_pin_saved = 1;

}  // namespace

OneThread::OneThread() {
  const std::lock_guard<std::mutex> lock(g_pin_mutex);
  if (g_pin_depth++ == 0) {
    g_pin_saved = openblas_get_num_threads();
    openblas_set_num_threads(1);
  }
}

OneThread::~OneThread() {
  const std::lock_guard<std::mutex> lock(g_pin_mutex);
  if (--g_pin_depth == 0 && openblas_get_num_threads() == 1) openblas_set_num_threads(g_pin_saved);
}

void OneThread::verify() {
  if (openblas_get_num_threads() != 1) {
    throw std::runtime_error("OpenBLAS's thread count changed during a scene fit (it must stay 1, the reference split)");
  }
}

std::string config() {
  const char* c = openblas_get_config();
  return c ? std::string(c) : std::string();
}

double dot(int n, const double* x, const double* y) {
  double sum = 0.0;
  sum += cblas_ddot(n, x, 1, y, 1);
  return sum;
}

namespace {

void matmul3_blas(const double* a, bool trans_a, const double* b, bool trans_b, double* c) {
  cblas_dgemm(kRowMajor, trans_a ? kTrans : kNoTrans, trans_b ? kTrans : kNoTrans, 3, 3, 3, 1.0, a, 3, b, 3, 0.0, c, 3);
}

// OpenBLAS's 3 x 3 dgemm on its Haswell kernel: an fma chain over k = 0, 1, 2 added to the zeroed C (beta = 0), so a
// -0 sum becomes +0.
void matmul3_inline(const double* a, bool trans_a, const double* b, bool trans_b, double* c) {
  for (int i = 0; i < 3; ++i) {
    double x[3];
    for (int k = 0; k < 3; ++k) x[k] = trans_a ? a[3 * k + i] : a[3 * i + k];
    for (int j = 0; j < 3; ++j) {
      const double y0 = trans_b ? b[3 * j] : b[j], y1 = trans_b ? b[3 * j + 1] : b[3 + j],
                   y2 = trans_b ? b[3 * j + 2] : b[6 + j];
      c[3 * i + j] = std::fma(x[2], y2, std::fma(x[1], y1, x[0] * y0)) + 0.0;
    }
  }
}

// Whether the inline product is bitwise the library's on this machine: 20000 random products with every transpose
// pair, magnitudes over 2^-40 .. 2^40, signed zeros and exact cancellations. Otherwise every product calls the library.
bool inline_matches_library() {
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> u(-0.5, 0.5), e(-28.0, 28.0);
  for (int trial = 0; trial < 20000; ++trial) {
    double a[9], b[9], c1[9], c2[9];
    for (int i = 0; i < 9; ++i) {
      a[i] = u(rng) * std::exp(e(rng));
      b[i] = u(rng) * std::exp(e(rng));
    }
    if (trial % 7 == 0) a[trial % 9] = trial % 2 ? 0.0 : -0.0;
    if (trial % 11 == 0) b[(trial / 11) % 9] = -0.0;
    if (trial % 13 == 0) {
      for (int i = 0; i < 9; ++i) b[i] = i % 3 == 0 ? 0.0 : -0.0;
    }
    const bool ta = trial % 2, tb = (trial / 2) % 2;
    matmul3_blas(a, ta, b, tb, c1);
    matmul3_inline(a, ta, b, tb, c2);
    if (std::memcmp(c1, c2, sizeof c1) != 0) return false;
  }
  return true;
}

}  // namespace

bool matmul3_is_inline() {
  static const bool ok = inline_matches_library();
  return ok;
}

void matmul3(const double* a, bool trans_a, const double* b, bool trans_b, double* c) {
  if (matmul3_is_inline()) {
    matmul3_inline(a, trans_a, b, trans_b, c);
  } else {
    matmul3_blas(a, trans_a, b, trans_b, c);
  }
}

void lapack_error(const char* routine, int info) {
  throw std::runtime_error(std::string(routine) + ": LAPACK info " + std::to_string(info));
}

}  // namespace balla::scene::blas
