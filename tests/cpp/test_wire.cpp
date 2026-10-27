// The browser UI's event wire (core/wire/wire.h): docs/design/phase3/ui.md §8.1 W1-W7.
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/wire/wire.h"

using namespace balla::wire;

namespace {

struct Events {
  std::vector<int64_t> t;
  std::vector<uint16_t> x, y;
  std::vector<uint8_t> p;
  size_t size() const { return t.size(); }
};

// n events on a W x H sensor, time-sorted with gaps of 0..max_gap ns (0: a run of equal times)
Events random_events(size_t n, uint32_t w, uint32_t h, int64_t max_gap, uint64_t seed, int64_t t0 = 0) {
  std::mt19937_64 rng(seed);
  Events e;
  int64_t t = t0;
  for (size_t i = 0; i < n; ++i) {
    t += static_cast<int64_t>(rng() % static_cast<uint64_t>(max_gap + 1));
    e.t.push_back(t);
    e.x.push_back(static_cast<uint16_t>(rng() % w));
    e.y.push_back(static_cast<uint16_t>(rng() % h));
    e.p.push_back(static_cast<uint8_t>(rng() & 1));
  }
  return e;
}

BlockOrder order_of(const Events& e) { return lod_order(e.t, e.x, e.y, e.p); }

// the reference occurrence index: equal (x, y, p, t) earlier in file order
std::vector<uint32_t> reference_k(const Events& e) {
  std::vector<uint32_t> k(e.size(), 0);
  for (size_t i = 0; i < e.size(); ++i) {
    for (size_t j = i; j-- > 0 && e.t[j] == e.t[i];) {
      k[i] += e.x[j] == e.x[i] && e.y[j] == e.y[i] && (e.p[j] != 0) == (e.p[i] != 0);
    }
  }
  return k;
}

}  // namespace

TEST_CASE("W1 event_key and lod_level: golden values") {
  struct Golden {
    int64_t t;
    uint32_t x, y, p, k;
    uint64_t key;
    uint32_t level;
  };
  // computed independently with Python integers from §4.1's definition (SplitMix64 with its gamma step)
  const Golden goldens[] = {
      {0, 0, 0, 0, 0, 0xa706dd2f4d197e6full, 0},
      {1, 0, 0, 1, 0, 0x210aee97dce61845ull, 2},
      {-1, 5, 7, 0, 0, 0x119fda628959f022ull, 3},
      {1000000123, 639, 479, 1, 0, 0xa3cf171805dc64c6ull, 0},
      {1000000123, 639, 479, 1, 1, 0x39398a992534c6e3ull, 2},
      {7994614419, 16383, 16383, 0, 0, 0xbdafef9550661669ull, 0},
      {-(int64_t{1} << 62), 1, 2, 1, 3, 0x842a0dd1c263cf97ull, 0},
      {INT64_MAX, 100, 200, 0, 7, 0x5019041bc688a5b6ull, 1},
      {1032, 320, 240, 1, 0, 0x00490745160a9baeull, 8},
      {1068, 3, 4, 0, 0, 0x0612cc0c789647edull, 5},
  };
  for (const Golden& g : goldens) {
    CAPTURE(g.t);
    CHECK(event_key(g.t, g.x, g.y, g.p, g.k) == g.key);
    CHECK(lod_level(g.key) == g.level);
  }
  CHECK(lod_level(0) == 8);
  CHECK(lod_level(0xFFFFFFFFull << 32) == 0);
  CHECK(lod_level(uint64_t{1} << 63) == 0);
  CHECK(lod_level(uint64_t{1} << 56) == 7);
  CHECK(lod_level(uint64_t{1} << 55) == 8);
  // p enters as 0/1: any non-zero polarity is ON
  CHECK(event_key(5, 1, 1, 7, 0) == event_key(5, 1, 1, 1, 0));
  CHECK(kLodKeyVersion == 1);
}

TEST_CASE("W1 level fractions are 2^-j within 4 sigma, and duplicates get independent levels") {
  const Events e = random_events(1'000'000, 640, 480, 3000, 11);
  std::array<uint64_t, kLevels + 1> ge{};
  for (size_t i = 0; i < e.size(); ++i) {
    const uint32_t l = lod_level(event_key(e.t[i], e.x[i], e.y[i], e.p[i], 0));
    for (uint32_t j = 0; j <= l; ++j) ++ge[j];
  }
  const double n = static_cast<double>(e.size());
  for (int j = 1; j <= kLevels; ++j) {
    const double q = std::ldexp(1.0, -j), sigma = std::sqrt(q * (1 - q) / n);
    CAPTURE(j);
    CHECK(std::abs(static_cast<double>(ge[static_cast<size_t>(j)]) / n - q) < 4 * sigma);
  }
  // pairs of equal (x, y, p, t): k = 0 and 1; their levels agree with the probability of two independent draws,
  // sum_l P(l)^2 = 1/3 + (2/3) 2^-16
  const size_t pairs = 100'000;
  Events d;
  std::mt19937_64 rng(5);
  for (size_t i = 0; i < pairs; ++i) {
    const int64_t t = static_cast<int64_t>(i) * 1000 + static_cast<int64_t>(rng() % 1000);
    const auto x = static_cast<uint16_t>(rng() % 640), y = static_cast<uint16_t>(rng() % 480);
    const auto p = static_cast<uint8_t>(rng() & 1);
    for (int r = 0; r < 2; ++r) {
      d.t.push_back(t);
      d.x.push_back(x);
      d.y.push_back(y);
      d.p.push_back(p);
    }
  }
  const BlockOrder o = order_of(d);
  CHECK(o.duplicates == pairs);
  size_t agree = 0;
  for (size_t i = 0; i < pairs; ++i) {
    const size_t a = 2 * i;
    agree += lod_level(event_key(d.t[a], d.x[a], d.y[a], d.p[a], 0)) ==
             lod_level(event_key(d.t[a], d.x[a], d.y[a], d.p[a], 1));
  }
  const double q = 1.0 / 3.0 + (2.0 / 3.0) * std::ldexp(1.0, -16), np = static_cast<double>(pairs);
  CHECK(std::abs(static_cast<double>(agree) - np * q) < 4 * std::sqrt(np * q * (1 - q)));
}

TEST_CASE("W2 lod_order: a permutation, levels descending, file order within a level, n_ge and duplicates") {
  // short runs (gaps of 0..3 ns), a long run of one time with duplicates (the hash path) and a duplicate-rich
  // short run (the scan path)
  Events e = random_events(200'000, 64, 48, 3, 21, -5'000'000);
  const int64_t t_long = e.t.back() + 10;
  std::mt19937_64 rng(3);
  for (int i = 0; i < 5000; ++i) {
    e.t.push_back(t_long);
    e.x.push_back(static_cast<uint16_t>(rng() % 8));
    e.y.push_back(static_cast<uint16_t>(rng() % 8));
    e.p.push_back(static_cast<uint8_t>(rng() % 3));  // 2 counts as ON
  }
  for (int i = 0; i < 20; ++i) {
    e.t.push_back(t_long + 1);
    e.x.push_back(static_cast<uint16_t>(i % 3));
    e.y.push_back(0);
    e.p.push_back(1);
  }
  const BlockOrder o = order_of(e);
  const size_t n = e.size();
  REQUIRE(o.perm.size() == n);
  std::vector<uint8_t> seen(n, 0);
  for (const uint32_t i : o.perm) {
    REQUIRE(i < n);
    seen[i]++;
  }
  CHECK(std::all_of(seen.begin(), seen.end(), [](uint8_t s) { return s == 1; }));
  const std::vector<uint32_t> k = reference_k(e);
  std::vector<uint32_t> level(n);
  uint32_t dups = 0;
  for (size_t i = 0; i < n; ++i) {
    level[i] = lod_level(event_key(e.t[i], e.x[i], e.y[i], e.p[i], k[i]));
    dups += k[i] > 0;
  }
  CHECK(o.duplicates == dups);
  CHECK(dups > 4000);
  bool descending = true, file_order = true;
  for (size_t r = 1; r < n; ++r) {
    const uint32_t a = o.perm[r - 1], b = o.perm[r];
    descending = descending && level[a] >= level[b];
    file_order = file_order && (level[a] != level[b] || a < b);
  }
  CHECK(descending);
  CHECK(file_order);
  for (int j = 0; j <= kLevels; ++j) {
    const auto want = static_cast<uint32_t>(std::count_if(level.begin(), level.end(), [&](uint32_t l) {
      return l >= static_cast<uint32_t>(j);
    }));
    CHECK(o.n_ge[static_cast<size_t>(j)] == want);
  }
  CHECK(o.n_ge[0] == n);
  // prefix property: [0, n_ge[j]) is exactly the events of level >= j
  for (int j = 1; j <= kLevels; ++j) {
    const uint32_t m = o.n_ge[static_cast<size_t>(j)];
    for (uint32_t r = 0; r < m; ++r) REQUIRE(level[o.perm[r]] >= static_cast<uint32_t>(j));
  }

  const BlockOrder empty = lod_order({}, {}, {}, {});
  CHECK(empty.perm.empty());
  CHECK(empty.n_ge[0] == 0);
  CHECK(empty.duplicates == 0);
  Events bad = random_events(10, 4, 4, 5, 1);
  std::swap(bad.t[3], bad.t[7]);
  if (bad.t[3] != bad.t[7]) CHECK_THROWS_AS(order_of(bad), std::invalid_argument);
  const std::vector<int64_t> t2{1, 2};
  const std::vector<uint16_t> c1{1};
  const std::vector<uint8_t> p2{0, 1};
  CHECK_THROWS_AS(lod_order(t2, c1, c1, p2), std::invalid_argument);
}

TEST_CASE("W3 sampled count images are unbiased per pixel") {
  // a Poisson image: pixels drawn from a smooth intensity on VGA, 10^6 events
  const uint32_t w = 640, h = 480;
  std::mt19937_64 rng(7);
  std::vector<double> cdf(size_t{w} * h);
  double acc = 0;
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      acc += 0.2 + std::exp(-((x - 320.0) * (x - 320.0) + (y - 200.0) * (y - 200.0)) / (2 * 90.0 * 90.0)) * 8;
      cdf[size_t{y} * w + x] = acc;
    }
  }
  Events e;
  int64_t t = 0;
  std::uniform_real_distribution<double> u(0.0, acc);
  for (size_t i = 0; i < 1'000'000; ++i) {
    t += static_cast<int64_t>(rng() % 2000);
    const size_t pix = static_cast<size_t>(std::upper_bound(cdf.begin(), cdf.end(), u(rng)) - cdf.begin());
    e.t.push_back(t);
    e.x.push_back(static_cast<uint16_t>(pix % w));
    e.y.push_back(static_cast<uint16_t>(pix / w));
    e.p.push_back(static_cast<uint8_t>(rng() & 1));
  }
  const BlockOrder o = order_of(e);
  std::vector<double> full(size_t{w} * h, 0);
  for (size_t i = 0; i < e.size(); ++i) full[size_t{e.y[i]} * w + e.x[i]] += 1;
  for (const int j : {2, 4, 6}) {
    std::vector<double> est(full.size(), 0);
    for (uint32_t r = 0; r < o.n_ge[static_cast<size_t>(j)]; ++r) {
      const uint32_t i = o.perm[r];
      est[size_t{e.y[i]} * w + e.x[i]] += std::ldexp(1.0, j);
    }
    double s = 0, s2 = 0, m = 0;
    for (size_t pix = 0; pix < full.size(); ++pix) {
      if (full[pix] == 0) continue;
      const double z = (est[pix] - full[pix]) / std::sqrt(full[pix] * (std::ldexp(1.0, j) - 1));
      s += z;
      s2 += z * z;
      m += 1;
    }
    const double mean = s / m, sd = std::sqrt(s2 / m - mean * mean);
    CAPTURE(j);
    CHECK(std::abs(mean) < 0.02);
    CHECK(sd > 0.97);
    CHECK(sd < 1.03);
  }
}

TEST_CASE("W4 pack_ev: bit layout, exact times, time order without perm, refusals") {
  const std::vector<int64_t> t{1000, 1000 + (int64_t{1} << 28) - 1, 1000, 5000};
  const std::vector<uint16_t> x{16383, 0, 7, 640}, y{16383, 1, 9, 480};
  const std::vector<uint8_t> p{1, 0, 2, 0}, label{4, 0, 2, 1};
  const std::vector<uint32_t> perm{3, 1, 0, 2};
  std::vector<uint32_t> out(8);
  pack_ev(t, x, y, p, label.data(), perm, 0, 4, 1000, out.data());
  for (uint32_t r = 0; r < 4; ++r) {
    const uint32_t s = perm[r], w0 = out[2 * r], w1 = out[2 * r + 1];
    CAPTURE(r);
    CHECK((w0 & 0x3FFFu) == x[s]);
    CHECK(((w0 >> 14) & 0x3FFFu) == y[s]);
    CHECK(((w0 >> 28) & 7u) == 1u + label[s]);
    CHECK((w0 >> 31) == (p[s] != 0 ? 1u : 0u));
    CHECK(w1 == static_cast<uint32_t>(t[s] - 1000));
  }
  CHECK(out[2 * 2] == (16383u | (16383u << 14) | (5u << 28) | (1u << 31)));  // x = y = 16383, HOT, ON
  CHECK(out[2 * 2 + 1] == 0u);                                                // dt = 0
  CHECK(out[2 * 1 + 1] == (1u << 28) - 1);                                    // dt = 2^28 - 1
  // no perm: time order; no labels: tag 0; a sub-range
  std::vector<uint32_t> plain(4);
  pack_ev(t, x, y, p, nullptr, {}, 1, 3, 1000, plain.data());
  CHECK(plain[0] == (0u | (1u << 14)));
  CHECK(plain[1] == (1u << 28) - 1);
  CHECK(plain[2] == (7u | (9u << 14) | (1u << 31)));
  CHECK(plain[3] == 0u);
  // worker frames: dt up to 2^32 - 1
  const std::vector<int64_t> tw{(int64_t{1} << 32) - 1 + 77};
  const std::vector<uint16_t> one{1};
  const std::vector<uint8_t> on{1};
  std::vector<uint32_t> w(2);
  pack_ev(tw, one, one, on, nullptr, {}, 0, 1, 77, w.data());
  CHECK(w[1] == 0xFFFFFFFFu);
  // refusals
  const std::vector<uint16_t> wide{16384};
  CHECK_THROWS_AS(pack_ev(tw, wide, one, on, nullptr, {}, 0, 1, 77, w.data()), std::invalid_argument);
  CHECK_THROWS_AS(pack_ev(tw, one, wide, on, nullptr, {}, 0, 1, 77, w.data()), std::invalid_argument);
  const std::vector<uint8_t> label5{5};
  CHECK_THROWS_AS(pack_ev(tw, one, one, on, label5.data(), {}, 0, 1, 77, w.data()), std::invalid_argument);
  CHECK_THROWS_AS(pack_ev(tw, one, one, on, nullptr, {}, 0, 1, 76, w.data()), std::invalid_argument);  // 2^32
  CHECK_THROWS_AS(pack_ev(tw, one, one, on, nullptr, {}, 0, 1, tw[0] + 1, w.data()), std::invalid_argument);  // < 0
  const std::vector<uint32_t> outside{1};
  CHECK_THROWS_AS(pack_ev(tw, one, one, on, nullptr, outside, 0, 1, 77, w.data()), std::invalid_argument);
  CHECK_THROWS_AS(pack_ev(tw, one, one, on, nullptr, {}, 0, 2, 77, w.data()), std::invalid_argument);
  CHECK_THROWS_AS(pack_ev(tw, one, one, on, nullptr, {}, 1, 0, 77, w.data()), std::invalid_argument);
}

TEST_CASE("W4 pack_block equals pack_ev over lod_order's perm") {
  Events e = random_events(300'000, 640, 480, 2, 29, 3 * kBlockNs);
  std::vector<uint8_t> label(e.size());
  for (size_t i = 0; i < e.size(); ++i) label[i] = static_cast<uint8_t>(i % 5);
  for (int r = 0; r < 40; ++r) {  // duplicates
    e.t.push_back(e.t.back());
    e.x.push_back(e.x.back());
    e.y.push_back(e.y.back());
    e.p.push_back(e.p.back());
    label.push_back(0);
  }
  const auto n = static_cast<uint32_t>(e.size());
  const BlockOrder o = order_of(e);
  std::vector<uint32_t> a(2 * size_t{n}), b(2 * size_t{n});
  pack_ev(e.t, e.x, e.y, e.p, label.data(), o.perm, 0, n, 3 * kBlockNs, a.data());
  const BlockOrder q = pack_block(e.t, e.x, e.y, e.p, label.data(), 3 * kBlockNs, b.data());
  CHECK(a == b);
  CHECK(q.perm.empty());
  CHECK(q.n_ge == o.n_ge);
  CHECK(q.duplicates == o.duplicates);
  CHECK(q.duplicates >= 40);
  label[7] = 5;
  CHECK_THROWS_AS(pack_block(e.t, e.x, e.y, e.p, label.data(), 3 * kBlockNs, b.data()), std::invalid_argument);
  CHECK_THROWS_AS(pack_block(e.t, e.x, e.y, e.p, nullptr, e.t[5] + 1, b.data()), std::invalid_argument);
}

TEST_CASE("W5 rate_counts equals a reference loop, bin edges included") {
  std::mt19937_64 rng(13);
  const int64_t t0 = -1'234'567, bin = 1'000'000;
  const size_t n_bins = 50;
  std::vector<int64_t> t;
  std::vector<uint8_t> p;
  for (int i = 0; i < 100'000; ++i) {
    t.push_back(t0 - 3 * bin + static_cast<int64_t>(rng() % static_cast<uint64_t>(56 * bin)));
    p.push_back(static_cast<uint8_t>(rng() % 3));
  }
  for (int64_t b = -1; b <= static_cast<int64_t>(n_bins); ++b) {  // every edge, and one ns before it
    t.push_back(t0 + b * bin);
    t.push_back(t0 + b * bin - 1);
    p.push_back(1);
    p.push_back(0);
  }
  std::vector<uint32_t> total(n_bins, 0), on(n_bins, 0), rt(n_bins, 0), ro(n_bins, 0);
  rate_counts(t, p, t0, bin, total, on);
  for (size_t i = 0; i < t.size(); ++i) {
    for (size_t b = 0; b < n_bins; ++b) {
      const int64_t lo = t0 + static_cast<int64_t>(b) * bin;
      if (t[i] >= lo && t[i] < lo + bin) {
        ++rt[b];
        ro[b] += p[i] != 0;
      }
    }
  }
  CHECK(total == rt);
  CHECK(on == ro);
  // sorted input (the running-bin path) equals the unsorted result
  std::vector<size_t> idx(t.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return t[a] < t[b]; });
  std::vector<int64_t> ts(t.size());
  std::vector<uint8_t> ps(t.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    ts[i] = t[idx[i]];
    ps[i] = p[idx[i]];
  }
  std::vector<uint32_t> st(n_bins, 0), so(n_bins, 0);
  rate_counts(ts, ps, t0, bin, st, so);
  CHECK(st == rt);
  CHECK(so == ro);
  // counts are added; `on` may be left out
  std::vector<uint32_t> twice = total;
  rate_counts(t, {}, t0, bin, twice, {});
  for (size_t b = 0; b < n_bins; ++b) CHECK(twice[b] == 2 * total[b]);
  CHECK_THROWS_AS(rate_counts(t, p, t0, 0, total, on), std::invalid_argument);
  std::vector<uint32_t> short_on(3);
  CHECK_THROWS_AS(rate_counts(t, p, t0, bin, total, short_on), std::invalid_argument);
  // extreme times do not overflow
  const std::vector<int64_t> ext{INT64_MIN, INT64_MAX, 0};
  const std::vector<uint8_t> pe{1, 1, 1};
  std::vector<uint32_t> te(2, 0), oe(2, 0);
  rate_counts(ext, pe, INT64_MIN, int64_t{1} << 62, te, oe);
  CHECK(te[0] == 1);
  CHECK(te[1] == 0);
  const std::vector<int64_t> top{INT64_MAX - 2, INT64_MAX - 1, INT64_MAX, INT64_MAX};
  const std::vector<uint8_t> pt{1, 0, 1, 1};
  std::vector<uint32_t> tt(1, 0), ot(1, 0);
  rate_counts(top, pt, INT64_MAX - 2, INT64_MAX, tt, ot);  // one bin reaching past INT64_MAX
  CHECK(tt[0] == 4);
  CHECK(ot[0] == 3);
}

TEST_CASE("W6 region_bits equals a reference loop on a synthetic mask set with i64 interval edges") {
  std::mt19937_64 rng(19);
  const uint16_t w = 37, h = 11;                     // W H = 407: the last word of a mask is partly used
  const size_t words = mask_words(w, h), masks = 6;
  CHECK(words == 13);
  // ordered, disjoint intervals with gaps, an empty one and edges near the int64 limits
  const std::vector<int64_t> lo{INT64_MIN, -5'000'000'000'000, -7, 1'000, 1'000, 4'000'000'000'000'000};
  const std::vector<int64_t> hi{INT64_MIN + 3, -7, 1'000, 1'000, 9'000, INT64_MAX};
  std::vector<uint32_t> bits(masks * words);
  for (auto& b : bits) b = static_cast<uint32_t>(rng());
  auto set = [&](size_t m, uint32_t x, uint32_t y) {
    const size_t i = size_t{y} * w + x;
    return (bits[m * words + (i >> 5)] >> (i & 31)) & 1u;
  };
  // events at every edge and one ns either side, at random times, with coordinates in and outside the image
  std::vector<int64_t> t;
  for (size_t m = 0; m < masks; ++m) {
    for (const int64_t e : {lo[m], hi[m]}) {
      for (const int64_t d : {-1, 0, 1}) {
        if ((d < 0 && e == INT64_MIN) || (d > 0 && e == INT64_MAX)) continue;
        t.push_back(e + d);
      }
    }
  }
  for (int i = 0; i < 20'000; ++i) {
    const uint64_t pick = rng() % 4;
    t.push_back(pick == 0 ? static_cast<int64_t>(rng())
                          : pick == 1 ? -5'000'000'000'000 + static_cast<int64_t>(rng() % 10'000'000'000'000)
                                      : pick == 2 ? static_cast<int64_t>(rng() % 12'000) - 1'000
                                                  : 4'000'000'000'000'000 + static_cast<int64_t>(rng() % 1'000'000));
  }
  std::vector<uint16_t> x(t.size()), y(t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    x[i] = static_cast<uint16_t>(rng() % (w + 3));     // some beyond the image
    y[i] = static_cast<uint16_t>(rng() % (h + 2));
  }
  using I64 = std::vector<int64_t>;
  using U16 = std::vector<uint16_t>;
  auto reference = [&](const I64& tt, const U16& xx, const U16& yy) {
    std::vector<uint8_t> out(tt.size(), 0);
    for (size_t i = 0; i < tt.size(); ++i) {
      for (size_t m = 0; m < masks; ++m) {
        if (tt[i] >= lo[m] && tt[i] < hi[m] && xx[i] < w && yy[i] < h) {
          out[i] = static_cast<uint8_t>(set(m, xx[i], yy[i]));
        }
      }
    }
    return out;
  };
  const RegionMasks r{w, h, lo, hi, bits};
  std::vector<uint8_t> got(t.size());
  region_bits(x, y, t, r, got.data());              // unsorted times
  const std::vector<uint8_t> want = reference(t, x, y);
  CHECK(got == want);
  size_t inside = 0;
  for (const uint8_t v : want) inside += v;
  CHECK(inside > 1000);
  // time-sorted events (the path that reuses the previous interval) give the same answers
  std::vector<size_t> idx(t.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return t[a] < t[b]; });
  std::vector<int64_t> ts(t.size());
  std::vector<uint16_t> xs(t.size()), ys(t.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    ts[i] = t[idx[i]];
    xs[i] = x[idx[i]];
    ys[i] = y[idx[i]];
  }
  std::vector<uint8_t> sorted_got(t.size());
  region_bits(xs, ys, ts, r, sorted_got.data());
  CHECK(sorted_got == reference(ts, xs, ys));
  // no masks: nothing is inside
  std::vector<uint8_t> none(t.size(), 7);
  region_bits(x, y, t, RegionMasks{w, h, {}, {}, {}}, none.data());
  CHECK(std::all_of(none.begin(), none.end(), [](uint8_t v) { return v == 0; }));
  // refusals: overlapping or reversed intervals, a wrong word count, mismatched columns, an empty image
  std::vector<uint8_t> out(t.size());
  const std::vector<int64_t> overlap_hi{INT64_MIN + 3, -6, 1'000, 1'000, 9'000, INT64_MAX};
  CHECK_THROWS_AS(region_bits(x, y, t, RegionMasks{w, h, lo, overlap_hi, bits}, out.data()), std::invalid_argument);
  const std::vector<int64_t> reversed_hi{INT64_MIN + 3, -7, 1'000, 999, 9'000, INT64_MAX};
  CHECK_THROWS_AS(region_bits(x, y, t, RegionMasks{w, h, lo, reversed_hi, bits}, out.data()), std::invalid_argument);
  const std::vector<uint32_t> short_bits(masks * words - 1);
  CHECK_THROWS_AS(region_bits(x, y, t, RegionMasks{w, h, lo, hi, short_bits}, out.data()), std::invalid_argument);
  const std::vector<uint16_t> short_x(t.size() - 1);
  CHECK_THROWS_AS(region_bits(short_x, y, t, r, out.data()), std::invalid_argument);
  CHECK_THROWS_AS(region_bits(x, y, t, RegionMasks{0, h, lo, hi, bits}, out.data()), std::invalid_argument);
}

TEST_CASE("W7 gather equals indexing by perm for 1-, 2-, 4- and 8-byte columns, packed or interleaved") {
  std::mt19937_64 rng(17);
  const size_t n = 1000;
  std::vector<uint32_t> perm(n);
  for (size_t i = 0; i < n; ++i) perm[i] = static_cast<uint32_t>(i);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<uint8_t> c1(n);
  std::vector<uint16_t> c2(n);
  std::vector<float> c4(n);
  std::vector<int64_t> c8(n);
  for (size_t i = 0; i < n; ++i) {
    c1[i] = static_cast<uint8_t>(rng());
    c2[i] = static_cast<uint16_t>(rng());
    c4[i] = static_cast<float>(rng() % 100000) * 0.25f;
    c8[i] = static_cast<int64_t>(rng());
  }
  std::vector<uint8_t> g1(n);
  std::vector<uint16_t> g2(n);
  std::vector<float> g4(n);
  std::vector<int64_t> g8(n);
  gather(perm, c1.data(), 1, g1.data());
  gather(perm, c2.data(), 2, g2.data());
  gather(perm, c4.data(), 4, g4.data());
  gather(perm, c8.data(), 8, g8.data());
  bool ok = true;
  for (size_t i = 0; i < n; ++i) {
    ok = ok && g1[i] == c1[perm[i]] && g2[i] == c2[perm[i]] && g4[i] == c4[perm[i]] && g8[i] == c8[perm[i]];
  }
  CHECK(ok);
  // two f32 columns interleaved (flow_x, flow_y) and a 3-byte element (the generic path)
  std::vector<float> fy(n), inter(2 * n);
  for (size_t i = 0; i < n; ++i) fy[i] = -c4[i];
  gather(perm, c4.data(), 4, inter.data(), 8);
  gather(perm, fy.data(), 4, inter.data() + 1, 8);
  std::vector<unsigned char> raw3(3 * n), g3(3 * n);
  for (size_t i = 0; i < 3 * n; ++i) raw3[i] = static_cast<unsigned char>(rng());
  gather(perm, raw3.data(), 3, g3.data());
  for (size_t i = 0; i < n; ++i) {
    ok = ok && inter[2 * i] == c4[perm[i]] && inter[2 * i + 1] == -c4[perm[i]] &&
         std::memcmp(&g3[3 * i], &raw3[3 * size_t{perm[i]}], 3) == 0;
  }
  CHECK(ok);
  CHECK_THROWS_AS(gather(perm, c4.data(), 4, inter.data(), 2), std::invalid_argument);
}

TEST_CASE("C4 times and polarity") {
  const std::vector<int64_t> t_us{0, 1, 999, -5};
  const std::vector<uint16_t> rem{0, 999, 1, 500};
  std::vector<int64_t> out(4);
  c4_times(t_us, rem, 1'700'000'000'000'000, out);
  CHECK(out[0] == 1'700'000'000'000'000'000);
  CHECK(out[1] == 1'700'000'000'000'001'999);
  CHECK(out[2] == 1'700'000'000'000'999'001);
  CHECK(out[3] == 1'699'999'999'999'995'500);
  c4_times(t_us, {}, 0, out);
  CHECK(out[1] == 1000);
  std::vector<int64_t> short_out(3);
  CHECK_THROWS_AS(c4_times(t_us, rem, 0, short_out), std::invalid_argument);
  const std::vector<uint8_t> p{0, 1, 2, 255};
  std::vector<uint8_t> q(4);
  polarity01(p, false, q);
  CHECK(q == std::vector<uint8_t>{0, 1, 1, 1});
  polarity01(p, true, q);
  CHECK(q == std::vector<uint8_t>{1, 0, 0, 0});
}
