// OpenBLAS / LAPACK as NumPy and SciPy call them (docs/dev/CPP_PORT.md item 5, E1).
//
// The scene side reproduces the Python fitter bit for bit. Every float64 operation NumPy or SciPy routes to BLAS or
// LAPACK goes to the same symbol of the same library (the environment's libopenblas, LP64), with the arguments and
// workspace queries NumPy and SciPy pass, on one OpenBLAS thread (the bench's split, E1): a large ddot or gemv splits
// its sum by the thread count. Own prototypes: no cblas.h or lapacke.h is needed.
#pragma once

#include <cstddef>
#include <string>

extern "C" {
double cblas_ddot(int n, const double* x, int incx, const double* y, int incy);
void cblas_dgemv(int order, int trans, int m, int n, double alpha, const double* a, int lda, const double* x, int incx,
                 double beta, double* y, int incy);
void cblas_dgemm(int order, int transa, int transb, int m, int n, int k, double alpha, const double* a, int lda,
                 const double* b, int ldb, double beta, double* c, int ldc);
void dpbtrf_(const char* uplo, const int* n, const int* kd, double* ab, const int* ldab, int* info, size_t uplo_len);
void dpbtrs_(const char* uplo, const int* n, const int* kd, const int* nrhs, const double* ab, const int* ldab,
             double* b, const int* ldb, int* info, size_t uplo_len);
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b, const int* ldb, int* info);
void dgelsd_(const int* m, const int* n, const int* nrhs, double* a, const int* lda, double* b, const int* ldb,
             double* s, const double* rcond, int* rank, double* work, const int* lwork, int* iwork, int* info);
void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w, double* work,
             const int* lwork, int* iwork, const int* liwork, int* info, size_t jobz_len, size_t uplo_len);
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgesdd_(const char* jobz, const int* m, const int* n, double* a, const int* lda, double* s, double* u,
             const int* ldu, double* vt, const int* ldvt, double* work, const int* lwork, int* iwork, int* info,
             size_t jobz_len);
int openblas_set_num_threads_local(int num_threads);
void openblas_set_num_threads(int num_threads);
int openblas_get_num_threads(void);
char* openblas_get_config(void);
}

namespace balla::scene::blas {

enum : int { kRowMajor = 101, kColMajor = 102, kNoTrans = 111, kTrans = 112 };

// OpenBLAS runs on one thread while any OneThread lives, process-wide, and returns to the previous count when the
// last one ends (reference-counted, so concurrent fits on pool threads share one pin), unless something else set a
// count meanwhile (that count is kept). This build's openblas_set_num_threads_local is not thread-local (it sets the
// process-wide count: tests/cpp/scene/test_scene.cpp), so the pin is openblas_set_num_threads(1), the plan's fallback:
// NumPy calls made by other Python threads during a C++ build run on one thread too (the bench's split).
class OneThread {
 public:
  OneThread();
  ~OneThread();
  OneThread(const OneThread&) = delete;
  OneThread& operator=(const OneThread&) = delete;
  // Throws std::runtime_error when OpenBLAS no longer runs on one thread (another caller changed the count during
  // the pin, so a large dot product may have been split): the caller's result is not the reference's.
  static void verify();
};

// openblas_get_config(): the version and the DYNAMIC_ARCH kernel (part of the bundle cache key, S4).
std::string config();

// NumPy's DOUBLE_dot (x.dot(y), 1-D @ 1-D, np.linalg.norm without an axis): 0.0 + cblas_ddot.
double dot(int n, const double* x, const double* y);

// One matrix of NumPy's matmul on row-major 3x3 blocks (DOUBLE_matmul_matrixmatrix): C = op(A) op(B), op the
// transpose for a swapaxes view (trans_a / trans_b), bitwise as cblas_dgemm as NumPy calls it. Inline (an fma chain
// over k, then + 0.0: OpenBLAS's Haswell kernel) when a self-check against the library passes on this machine (the
// first call), else the library itself.
void matmul3(const double* a, bool trans_a, const double* b, bool trans_b, double* c);
bool matmul3_is_inline();

// LAPACK error (info != 0) as std::runtime_error naming the routine.
[[noreturn]] void lapack_error(const char* routine, int info);

}  // namespace balla::scene::blas
