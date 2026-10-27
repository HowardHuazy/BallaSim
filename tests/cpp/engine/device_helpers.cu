#include <cuda_runtime.h>

#include "device_helpers.h"
#include "engine/band_map.h"

namespace balla::engine::test {
namespace {

using D = Par<Space::kDevice>;

template <class T>
T* put(Buffer<T, Space::kDevice>& b, const std::vector<T>& v) {
  T* p = b.ensure(v.size());
  D::upload(p, v.data(), v.size() * sizeof(T), nullptr);
  return p;
}

template <class T>
std::vector<T> get(const T* p, size_t n) {
  std::vector<T> v(n);
  D::download(v.data(), p, n * sizeof(T), nullptr);
  return v;
}

template <class T>
std::vector<T> scan(const std::vector<T>& in) {
  Buffer<T, Space::kDevice> a, b;
  Temp<Space::kDevice> temp;
  D::exclusive_scan(put(a, in), b.ensure(in.size()), in.size(), temp, nullptr);
  return get(b.data(), in.size());
}

struct Count {
  const uint8_t* pred;
  unsigned long long* counter;
  BALLA_HD void operator()(size_t i) const { count_if(counter, pred[i] != 0); }
};

struct Eta {
  MidpointParams p;
  const float *a, *m, *b;
  float* out;
  BALLA_HD void operator()(size_t i) const {
    const size_t g = static_cast<size_t>(p.groups);
    out[i] = midpoint_eta(p, a + i * g, m + i * g, b + i * g);
  }
};

}  // namespace

bool device_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::vector<uint32_t> device_exclusive_scan(const std::vector<uint32_t>& in) { return scan(in); }
std::vector<uint64_t> device_exclusive_scan(const std::vector<uint64_t>& in) { return scan(in); }

std::pair<std::vector<uint64_t>, std::vector<uint32_t>> device_sort_pairs(const std::vector<uint64_t>& keys,
                                                                          const std::vector<uint32_t>& values,
                                                                          int end_bit) {
  Buffer<uint64_t, Space::kDevice> k, k2;
  Buffer<uint32_t, Space::kDevice> v, v2;
  Temp<Space::kDevice> temp;
  const size_t n = keys.size();
  const bool alt = D::sort_pairs(put(k, keys), k2.ensure(n), put(v, values), v2.ensure(n), n, end_bit, temp, nullptr);
  return {get(alt ? k2.data() : k.data(), n), get(alt ? v2.data() : v.data(), n)};
}

std::vector<uint32_t> device_select(const std::vector<uint8_t>& flags) {
  Buffer<uint8_t, Space::kDevice> f;
  Buffer<uint32_t, Space::kDevice> out;
  Temp<Space::kDevice> temp;
  HostWords<Space::kDevice> count;
  D::select_flagged(put(f, flags), out.ensure(flags.size()), flags.size(), count.data(), temp, nullptr);
  D::sync(nullptr);
  return get(out.data(), count.data()[0]);
}

unsigned long long device_count(const std::vector<uint8_t>& pred) {
  Buffer<uint8_t, Space::kDevice> f;
  Buffer<unsigned long long, Space::kDevice> c;
  D::zero(c.ensure(1), 8, nullptr);
  D::for_each(pred.size(), Count{put(f, pred), c.data()}, nullptr);
  return get(c.data(), 1)[0];
}

unsigned long long host_count(const std::vector<uint8_t>& pred) {
  unsigned long long c = 0;
  Par<Space::kHost>::for_each(pred.size(), Count{pred.data(), &c}, nullptr);
  return c;
}

std::vector<float> device_eta(const MidpointParams& p, const std::vector<float>& a, const std::vector<float>& m,
                              const std::vector<float>& b) {
  Buffer<float, Space::kDevice> da, dm, db, out;
  const size_t n = a.size() / static_cast<size_t>(p.groups);
  D::for_each(n, Eta{p, put(da, a), put(dm, m), put(db, b), out.ensure(n)}, nullptr);
  return get(out.data(), n);
}

std::pair<std::vector<int32_t>, std::vector<uint32_t>> device_band_radius(const std::vector<double>& seed_m,
                                                                          const std::vector<uint8_t>& kind,
                                                                          const std::vector<double>& tile_floor,
                                                                          const std::vector<uint32_t>& pixel_tile,
                                                                          double support, int32_t cap) {
  Buffer<double, Space::kDevice> m, f;
  Buffer<uint8_t, Space::kDevice> k;
  Buffer<uint32_t, Space::kDevice> tiles, hist;
  Buffer<int32_t, Space::kDevice> radius;
  const size_t n = seed_m.size(), levels = static_cast<size_t>(cap) + 1;
  D::zero(hist.ensure(levels), levels * sizeof(uint32_t), nullptr);
  D::for_each(n,
              BandRadius{put(m, seed_m), put(k, kind), put(f, tile_floor), put(tiles, pixel_tile), support, cap,
                         radius.ensure(n), hist.data()},
              nullptr);
  return {get(radius.data(), n), get(hist.data(), levels)};
}

std::vector<uint8_t> device_band_cascade(const std::vector<int32_t>& radius, const std::vector<int32_t>& levels,
                                         int32_t width, int32_t height, const BandDomain& dom) {
  Buffer<int32_t, Space::kDevice> r;
  Buffer<uint8_t, Space::kDevice> band, tmp;
  const size_t n = radius.size();
  band_cascade<Space::kDevice>(put(r, radius), levels, width, height, dom, band.ensure(n), tmp.ensure(n), nullptr);
  return get(band.data(), n);
}

PixelStageRun device_pixel_stage(const PixelStageInput& in) { return pixel_stage_run<Space::kDevice>(in); }

FakeRuns device_fake_window(const FakeSetup& s, size_t budget) { return fake_window<Space::kDevice>(s, budget); }

PvFakeRuns device_pv_fake_window(const PvFakeSetup& s, size_t budget) {
  return pv_fake_window<Space::kDevice>(s, budget);
}

std::vector<FakeRuns> device_fake_windows(const std::vector<FakeSetup>& setups, size_t budget) {
  WindowEngine<Space::kDevice> engine;
  std::vector<FakeRuns> out;
  for (const FakeSetup& s : setups) out.push_back(fake_window(engine, s, budget));
  return out;
}

}  // namespace balla::engine::test
