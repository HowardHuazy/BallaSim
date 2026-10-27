// Deterministic cascade trace on a direct piecewise-linear photocurrent: the deterministic pass of
// the kernel (substep.h) without waveforms, dark current, noise or comparator, evaluated at query
// times. It is the hook through which tests compare the integrator with high-accuracy ODE solvers
// (M3 V1 T3) and fp32 with fp64 (M3 O7).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "sensor/substep.h"

namespace balla::sensor {

struct TraceResult {
  std::vector<double> s3;  // deterministic stage-3 output at the query times
  uint64_t substeps = 0;
};

// t: m >= 2 strictly increasing sample times (ns); I: photocurrent (fA) at them, > 0. The cascade
// starts in steady state at I[0]. Query times must lie in [t[0], t[m-1]].
template <class T>
TraceResult cascade_trace(const CascadeConstants<T>& cc, const int64_t* t, const double* I, size_t m,
                          const int64_t* q, size_t nq) {
  if (m < 2) throw std::invalid_argument("cascade_trace: need at least two samples");
  for (size_t i = 0; i < m; ++i) {
    if (!(I[i] > 0)) throw std::invalid_argument("cascade_trace: photocurrent must be positive");
    if (i > 0 && t[i] <= t[i - 1]) throw std::invalid_argument("cascade_trace: times must increase");
  }
  CascadeState<T> st{};
  const T i_scale = static_cast<T>(I[0]);  // log offset: the states start at 0
  st.z = T(1) / i_scale;
  T x1 = T(0);
  st.s2 = st.s3 = x1;
  std::vector<SubStep<T>> subs;
  for (size_t i = 0; i + 1 < m; ++i) {
    const LinearPiece<T> pc{t[i], t[i + 1], static_cast<T>(I[i]), static_cast<T>(I[i + 1])};
    int64_t ta = pc.t_a, h_hint = 0;
    while (ta < pc.t_b) {
      SubStep<T> sub{};
      CascadeState<T> next{};
      SubStepInfo info{};
      ta = next_substep(cc, st, i_scale, x1, pc, ta, pc.at(ta), h_hint, sub, next, info);
      st = next;
      x1 = sub.x1_b;
      subs.push_back(sub);
    }
  }
  TraceResult r;
  r.substeps = subs.size();
  r.s3.resize(nq);
  for (size_t j = 0; j < nq; ++j) {
    if (q[j] < t[0] || q[j] > t[m - 1]) throw std::invalid_argument("cascade_trace: query outside the samples");
    auto it = std::upper_bound(subs.begin(), subs.end(), q[j],
                               [](int64_t v, const SubStep<T>& s) { return v < s.t_a; });
    const SubStep<T>& s = *(it == subs.begin() ? it : it - 1);
    const double offset = std::log(static_cast<double>(i_scale));
    r.s3[j] = static_cast<double>(s.s3(static_cast<T>(q[j] - s.t_a) * T(1e-9))) + offset;
  }
  return r;
}

}  // namespace balla::sensor
