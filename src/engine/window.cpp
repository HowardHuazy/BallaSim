// Host instantiation of WindowEngine (ENGINE_CPP §1.7): serial primitives; it serves sampling.py's
// functions through the Python binding.
#include "engine/window_impl.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace balla::engine {

template class WindowEngine<Space::kHost>;

std::pair<double, double> waveform_range(const sensor::Waveform& w) {
  if (w.type == sensor::WaveformType::kFourier) {
    double swing = 0.0;
    for (int k = 0; k < w.n_harmonics; ++k) swing += std::fabs(w.a[k]);
    return {1.0 - swing, 1.0 + swing};
  }
  if (w.type == sensor::WaveformType::kPulse) return {w.floor_level, w.peak_level};
  return {1.0, 1.0};
}

std::optional<int> level_cap(const int64_t* levels, size_t n, int64_t budget, int l_min) {
  std::vector<int64_t> counts;
  for (size_t i = 0; i < n; ++i) {
    if (levels[i] < 0 || levels[i] > time::kWindowLog2) throw std::invalid_argument("level_cap: levels must lie in [0, 23]");
    if (static_cast<size_t>(levels[i]) >= counts.size()) counts.resize(static_cast<size_t>(levels[i]) + 1, 0);
    ++counts[static_cast<size_t>(levels[i])];
  }
  return level_cap_counts(counts, n, budget, l_min);
}

std::optional<int> level_cap_counts(std::vector<int64_t> counts, size_t n, int64_t budget, int l_min) {
  while (!counts.empty() && counts.back() == 0) counts.pop_back();
  const auto total = [&](int64_t cap) {
    int64_t sum = 0;
    for (size_t l = 0; l < counts.size(); ++l) sum += counts[l] * ((int64_t{1} << std::min<int64_t>(l, cap)) + 1);
    return sum;
  };
  const auto top = static_cast<int64_t>(counts.size()) - 1;
  if (counts.empty() || total(top) <= budget) return std::nullopt;
  for (int64_t cap = top - 1; cap >= l_min; --cap) {
    if (total(cap) <= budget) return static_cast<int>(cap);
  }
  throw std::invalid_argument("accuracy.max_samples_per_window = " + std::to_string(budget) +
                              " cannot hold one window of " + std::to_string(n) + " active pixels at level " +
                              std::to_string(l_min) + " (" + std::to_string(total(l_min)) + " samples); raise it");
}

uint32_t fixed_rate_samples(int64_t t_w, int64_t period_ns) {
  const int64_t q = time::floor_div(t_w, period_ns);
  const int64_t inner = -time::floor_div(-(t_w + time::kWindowNs), period_ns) - q - 1;
  return static_cast<uint32_t>(std::max<int64_t>(inner, 0) + 1);
}

MidpointParams midpoint_params(double tol, double dark, const std::vector<std::pair<double, double>>& ranges) {
  if (ranges.empty() || ranges.size() > static_cast<size_t>(kMaxGroups)) {
    throw std::invalid_argument("midpoint test: need 1 to 8 light groups");
  }
  MidpointParams p{};
  p.tol = static_cast<float>(tol);
  p.dark = static_cast<float>(dark);
  p.groups = static_cast<int32_t>(ranges.size());
  for (size_t g = 0; g < ranges.size(); ++g) {
    p.m_min[g] = static_cast<float>(ranges[g].first);
    p.m_max[g] = static_cast<float>(ranges[g].second);
    p.mediant = p.mediant || ranges[g].first != ranges[g].second;
  }
  return p;
}

}  // namespace balla::engine
