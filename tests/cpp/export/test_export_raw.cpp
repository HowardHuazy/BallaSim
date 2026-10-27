// The RAW, DAT and IMU CSV writers (docs/design/phase3/outputs.md §8.2): F1 golden words of R-FMT's seven events
// (out/t3_vectors.json; EVT 3.0 without vector words and EVT 2.1 under the order rule are R-FMT's words for this
// input, whose stored order is already (t, y, p, x)) and the regenerated headers; F4's order and duplicate cases; F8
// one append against many; the refusals of §6; the T1a rule, the header date and the export window. The round trips
// through independent decoders are tests/python/test_export.py's (export_oracles).
#include <doctest/doctest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include "config/sha256.h"
#include "export/errors.h"
#include "export/imu_csv.h"
#include "export/raw_writer.h"
#include "export/time_map.h"
#include "sensor/readout.h"

using namespace balla;
using exporter::RawFormat;

namespace {

// The suite's files, in one directory removed at exit.
struct TempDir {
  std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("balla_export_raw_tests_" + std::to_string(::getpid()));
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};
const TempDir kTemp;

std::string temp_path(const std::string& name) {
  std::filesystem::create_directories(kTemp.dir);
  return (kTemp.dir / name).string();
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

struct Columns {
  std::vector<uint16_t> x, y;
  std::vector<uint8_t> p;
  std::vector<int64_t> t;
  void add(int64_t t_ns, uint16_t xi, uint16_t yi, uint8_t pi) {
    t.push_back(t_ns);
    x.push_back(xi);
    y.push_back(yi);
    p.push_back(pi);
  }
  exporter::Events view(size_t a, size_t b) const {
    return {x.data() + a, y.data() + a, p.data() + a, t.data() + a, b - a};
  }
  size_t size() const { return t.size(); }
};

// R-FMT's seven test events (t3_vectors.json "events"): u in us, each at t_ns = 1000 u + 419.
Columns golden_events(int64_t origin_us = 0) {
  const int64_t u[] = {0, 0, 0, 1, 64, 4096, 4097};
  const uint16_t x[] = {10, 11, 12, 639, 3, 3, 1279}, y[] = {5, 5, 5, 479, 0, 0, 719};
  const uint8_t p[] = {1, 1, 1, 0, 1, 0, 1};
  Columns c;
  for (int i = 0; i < 7; ++i) c.add(1000 * (origin_us + u[i]) + 419, x[i], y[i], p[i]);
  return c;
}

exporter::RawOptions options(int32_t rem_min, int32_t rem_max,
                             exporter::Evt2TimeHigh th = exporter::Evt2TimeHigh::kOpenEb) {
  exporter::RawOptions o;
  o.evt2_time_high = th;
  o.t_ns_rem_min = rem_min;
  o.t_ns_rem_max = rem_max;
  o.version = "0.0.0";
  return o;
}

// The file written from `c` in batches of `batch` events (0: one append).
std::string write_raw(RawFormat f, const Columns& c, const exporter::RawOptions& o, size_t batch = 0,
                      exporter::TimeMap m = {}, int32_t w = 1280, int32_t h = 720) {
  const std::string path = temp_path("raw.raw");
  exporter::RawWriter writer(path, f, w, h, m, o);
  const size_t step = batch == 0 ? c.size() : batch;
  for (size_t a = 0; a < c.size(); a += step) writer.append(c.view(a, std::min(c.size(), a + step)));
  const exporter::FileInfo info = writer.close();
  const std::string bytes = read_file(path);
  CHECK(info.bytes == bytes.size());
  CHECK(info.sha256 == config::sha256_hex(bytes));
  REQUIRE(info.streams.size() == 1);
  CHECK(info.streams[0].count == static_cast<int64_t>(c.size()));
  return bytes;
}

std::string write_dat(const Columns& c, const exporter::RawOptions& o, size_t batch = 0, exporter::TimeMap m = {}) {
  const std::string path = temp_path("raw.dat");
  exporter::DatWriter writer(path, 1280, 720, m, o);
  const size_t step = batch == 0 ? c.size() : batch;
  for (size_t a = 0; a < c.size(); a += step) writer.append(c.view(a, std::min(c.size(), a + step)));
  writer.close();
  return read_file(path);
}

// The words after a RAW header ('% end\n'), as hex of the given width (little-endian words).
template <class W>
std::vector<std::string> words(const std::string& file) {
  const size_t at = file.find("% end\n");
  REQUIRE(at != std::string::npos);
  const size_t start = at + 6;
  REQUIRE((file.size() - start) % sizeof(W) == 0);
  std::vector<std::string> out;
  for (size_t i = start; i < file.size(); i += sizeof(W)) {
    W v = 0;
    std::memcpy(&v, file.data() + i, sizeof(W));
    char buf[24];
    std::snprintf(buf, sizeof buf, "%0*llx", static_cast<int>(2 * sizeof(W)), static_cast<unsigned long long>(v));
    out.emplace_back(buf);
  }
  return out;
}

std::string th32(int64_t step64) {
  char buf[16];
  const uint32_t word = 0x80000000u | static_cast<uint32_t>(step64 & 0x0FFFFFFF);
  std::snprintf(buf, sizeof buf, "%08x", static_cast<unsigned>(word));
  return buf;
}

// EVT 2.1 TIME_HIGH (type 0x8 in bits 63..60, the 28-bit step at 59..32) as the 16 hex digits of words<uint64_t>.
std::string th64(int64_t step64) {
  char buf[24];
  const uint64_t word = (uint64_t{0x8} << 60) | (static_cast<uint64_t>(step64 & 0x0FFFFFFF) << 32);
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(word));
  return buf;
}

// The TIME_HIGH words of an EVT 2.0 (4 hex digits per 16 bits: 8) or 2.1 (16) word list, in order.
std::vector<std::string> time_highs(const std::vector<std::string>& w) {
  std::vector<std::string> out;
  for (const auto& s : w) {
    if (s[0] == '8') out.push_back(s);
  }
  return out;
}

std::string hex(const std::string& bytes) {
  std::string out;
  char buf[3];
  for (const char c : bytes) {
    std::snprintf(buf, sizeof buf, "%02x", static_cast<unsigned>(static_cast<uint8_t>(c)));
    out += buf;
  }
  return out;
}

}  // namespace

TEST_CASE("F1 EVT 2.0, 2.1, 3.0 and DAT words of R-FMT's golden events") {
  const Columns c = golden_events();
  const auto ob = words<uint32_t>(write_raw(RawFormat::kEvt2, c, options(419, 419)));
  // R-FMT evt2_openeb_words: 264 words; a TIME_HIGH every 16 us from u = 0 to 4096, before the events of its step.
  REQUIRE(ob.size() == 264);
  const std::vector<std::string> head = {"80000000", "10005005", "10005805", "10006005", "0053f9df",
                                         "80000000", "80000000", "80000000", "80000001", "10001800"};
  CHECK(std::vector<std::string>(ob.begin(), ob.begin() + 10) == head);
  const std::vector<std::string> tail = {"80000040", "00001800", "1067facf"};
  CHECK(std::vector<std::string>(ob.end() - 3, ob.end()) == tail);
  for (size_t i = 10; i < 261; ++i) {  // u = 80..4080: TIME_HIGH 1 three more times, then 2..63 four times each
    CHECK(ob[i] == th32(i < 13 ? 1 : 2 + static_cast<int64_t>(i - 13) / 4));
  }
  CHECK(words<uint32_t>(write_raw(RawFormat::kEvt2, c, options(419, 419, exporter::Evt2TimeHigh::kOnChange))) ==
        std::vector<std::string>{"80000000", "10005005", "10005805", "10006005", "0053f9df", "80000001", "10001800",
                                 "80000040", "00001800", "1067facf"});  // R-FMT evt2_min_words
  CHECK(words<uint64_t>(write_raw(RawFormat::kEvt21, c, options(419, 419))) ==
        std::vector<std::string>{"8000000000000000", "1000000500001c00", "005301df80000000", "8000000100000000",
                                 "1000000000000008", "8000004000000000", "0000000000000008", "106702cf80000000"});
  CHECK(words<uint16_t>(write_raw(RawFormat::kEvt3, c, options(419, 419))) ==
        std::vector<std::string>{"8000", "6000", "0005", "280a", "280b", "280c", "6001", "01df", "227f", "6040", "0000",
                                 "2803", "8001", "6000", "2003", "6001", "02cf", "2cff"});  // R-FMT evt3_basic_words
  const std::string dat = write_dat(c, options(419, 419));
  const size_t at = dat.find(std::string("\x0c\x08", 2));
  REQUIRE(at != std::string::npos);
  CHECK(hex(dat.substr(at + 2)) ==
        "000000000a400110000000000b400110000000000c400110010000007fc277004000000003000010001000000300000001100000"
        "ffc4b310");  // R-FMT dat_records_hex
}

TEST_CASE("F1 the regenerated RAW and DAT headers: epoch-derived date, origin, epoch and t_ns_rem keys") {
  const exporter::TimeMap m{0, 1760000000000000000};
  const std::string raw = write_raw(RawFormat::kEvt3, golden_events(), options(419, 419), 0, m);
  CHECK(raw.substr(0, raw.find("% end\n") + 6) ==
        "% format EVT3;height=720;width=1280\n% camera_integrator_name BallaSim\n% plugin_integrator_name BallaSim\n"
        "% plugin_version BallaSim 0.0.0\n% date 2025-10-09 08:53:20\n% ballasim_t_offset_us 0\n"
        "% ballasim_epoch_ns 1760000000000000000\n% ballasim_t_ns_rem 419\n% end\n");
  const std::string dat = write_dat(golden_events(), options(419, 419), 0, m);
  CHECK(dat.substr(0, dat.find('\x0c') + 2) ==
        std::string("% Data file containing CD events.\n% Version 2\n% Date 2025-10-09 08:53:20\n% Height 720\n"
                    "% Width 1280\n% ballasim_t_offset_us 0\n% ballasim_epoch_ns 1760000000000000000\n"
                    "% ballasim_t_ns_rem 419\n\x0c\x08"));
  // The date of the origin: epoch + 1000 origin_us, floored to the second; format names; the t_ns_rem states.
  const Columns c = golden_events(5'000'000);
  const std::string later = write_raw(RawFormat::kEvt2, c, options(0, 999), 0, {5'000'000, 0});
  CHECK(later.find("% format EVT2;height=720;width=1280\n") == 0);
  CHECK(later.find("% date 1970-01-01 00:00:05\n% ballasim_t_offset_us 5000000\n% ballasim_epoch_ns 0\n"
                   "% ballasim_t_ns_rem varies\n") != std::string::npos);
  CHECK(write_raw(RawFormat::kEvt21, Columns{}, options(-1, -1)).find("% format EVT21;") == 0);
  CHECK(write_raw(RawFormat::kEvt21, Columns{}, options(-1, -1)).find("% ballasim_t_ns_rem none\n% end\n") !=
        std::string::npos);
  CHECK(words<uint16_t>(write_raw(RawFormat::kEvt3, Columns{}, options(-1, -1))).empty());
  // The origin moves u, not the words: the same events 5 s later with origin 5 s give the same words.
  CHECK(words<uint16_t>(write_raw(RawFormat::kEvt3, c, options(419, 419), 0, {5'000'000, 0})) ==
        words<uint16_t>(write_raw(RawFormat::kEvt3, golden_events(), options(419, 419))));
}

TEST_CASE("F4 order and duplicates: every encoder keeps the input order (checked by hand)") {
  // One microsecond (u = 10) in the stored (t, x, y) order, rows not sorted, a duplicate pixel, x decreasing in a
  // row: (1, 9, ON), (2, 3, ON), (3, 9, ON), (3, 9, ON), (2, 9, ON), (40, 9, OFF).
  Columns c;
  const uint16_t xs[] = {1, 2, 3, 3, 2, 40}, ys[] = {9, 3, 9, 9, 9, 9};
  const uint8_t ps[] = {1, 1, 1, 1, 1, 0};
  for (int i = 0; i < 6; ++i) c.add(10'000 + 7, xs[i], ys[i], ps[i]);
  // EVT 3.0: TIME_HIGH 0, TIME_LOW 10, then ADDR_Y on every row change and one ADDR_X per event (system bit 0).
  CHECK(words<uint16_t>(write_raw(RawFormat::kEvt3, c, options(7, 7))) ==
        std::vector<std::string>{"8000", "600a", "0009", "2801", "0003", "2802", "0009", "2803", "2803", "2802",
                                 "2028"});
  // EVT 2.1: (1, 9) and (3, 9) share a word only when nothing else comes between; the duplicate (3, 9) and the
  // smaller x (2, 9) open new words; (40, 9, OFF) has another base and polarity. 6 events in, 6 out, in order.
  CHECK(words<uint64_t>(write_raw(RawFormat::kEvt21, c, options(7, 7))) ==
        std::vector<std::string>{"8000000000000000", "1280000900000002", "1280000300000004", "1280000900000008",
                                 "1280000900000008", "1280000900000004", "0281000900000100"});
  // A row's increasing run shares one word: (3, 1), (4, 1), (5, 1) in, one mask 0x38 out.
  Columns run;
  for (uint16_t x = 3; x <= 5; ++x) run.add(10'000, x, 1, 1);
  CHECK(words<uint64_t>(write_raw(RawFormat::kEvt21, run, options(0, 0))) ==
        std::vector<std::string>{"8000000000000000", "1280000100000038"});
  // EVT 2.0 keeps every event as its own word, in order.
  const auto w2 = words<uint32_t>(write_raw(RawFormat::kEvt2, c, options(7, 7), 0, {}, 1280, 720));
  CHECK(w2.size() == 1 + 6);
  CHECK(w2.back() == "02814009");
}

TEST_CASE("F4 EVT 3.0 TIME_HIGH is dense from u = 0; EVT 2.0 and 2.1 on change keep alive every 8.192 ms") {
  Columns c;
  c.add(20'000'007'000, 5, 7, 1);  // the first event 20 s after the origin (u > 2^24)
  c.add(40'000'123'000, 6, 7, 0);  // 20 s later (a 2^24 us wrap in between)
  const auto w3 = words<uint16_t>(write_raw(RawFormat::kEvt3, c, options(0, 0)));
  const int64_t hi1 = 20'000'007 >> 12, hi2 = 40'000'123 >> 12;
  REQUIRE(w3.size() == static_cast<size_t>(hi1 + 1 + 3 + (hi2 - hi1) + 2));
  for (int64_t h = 0; h <= hi1; ++h) {
    char buf[8];
    std::snprintf(buf, sizeof buf, "8%03llx", static_cast<unsigned long long>(h & 0xFFF));
    CHECK(w3[static_cast<size_t>(h)] == std::string(buf));
  }
  // On change (EVT 2.0 change, EVT 2.1): the first step's TIME_HIGH, the keepalives at the multiples of 128 steps of
  // 64 us through the gap, the second step's.
  const int64_t k1 = 20'000'007 >> 6, k2 = 40'000'123 >> 6;
  const int64_t keepalives = (k2 - 1) / 128 - k1 / 128;
  CHECK(keepalives == 2441);
  const auto wc = words<uint32_t>(write_raw(RawFormat::kEvt2, c, options(0, 0, exporter::Evt2TimeHigh::kOnChange)));
  REQUIRE(wc.size() == static_cast<size_t>(1 + 1 + keepalives + 1 + 1));
  CHECK(wc[0] == th32(k1));
  for (int64_t i = 0; i < keepalives; ++i) CHECK(wc[static_cast<size_t>(2 + i)] == th32((k1 / 128 + 1 + i) * 128));
  CHECK(wc[static_cast<size_t>(2 + keepalives)] == th32(k2));
  CHECK(words<uint64_t>(write_raw(RawFormat::kEvt21, c, options(0, 0))).size() == static_cast<size_t>(4 + keepalives));
  // EVT 2.0 and 2.1 TIME_HIGH carry t >> 6 in 28 bits: across the 2^34 us wrap the value falls to 0 (the decoders'
  // loop rule; tests/python/test_export.py decodes it).
  Columns wrap;
  const int64_t T = int64_t{1} << 34;
  for (const int64_t u : {T - 100, T - 1, T, T + 70}) wrap.add(1000 * u, 1, 2, 1);
  const auto ww = words<uint32_t>(write_raw(RawFormat::kEvt2, wrap, options(0, 0, exporter::Evt2TimeHigh::kOnChange)));
  CHECK(ww == std::vector<std::string>{"8ffffffe", "17000802", "8fffffff", "1fc00802", "80000000", "10000802",
                                       "80000001", "11800802"});
  // Across the wrap with no event in its last 64 us step: OpenEB's EVT 2.1 decoder counts a loop only on a fall of
  // at least (2^28 - 1) steps, so EVT 2.1 writes the block's last step (0x0FFFFFFF) before the keepalive at 2^28.
  // EVT 2.0's rule allows 10 ms of slack, so its on-change words go from the keepalive at 2^28 - 128 straight to 0.
  Columns gap;
  const int64_t W = int64_t{1} << 28;
  for (const int64_t u : {T - 10'000, T + 70, T + 5'000}) gap.add(1000 * u, 1, 2, 1);
  const int64_t g1 = (T - 10'000) >> 6, g3 = (T + 5'000) >> 6;
  CHECK(g1 == W - 157);
  CHECK(time_highs(words<uint64_t>(write_raw(RawFormat::kEvt21, gap, options(0, 0)))) ==
        std::vector<std::string>{th64(g1), th64(W - 128), th64(W - 1), th64(W), th64(W + 1), th64(g3)});
  CHECK(time_highs(words<uint32_t>(write_raw(RawFormat::kEvt2, gap, options(0, 0, exporter::Evt2TimeHigh::kOnChange))))
        == std::vector<std::string>{th32(g1), th32(W - 128), th32(W), th32(W + 1), th32(g3)});
  // An event in the block's last step already ends it: no second 0x0FFFFFFF.
  CHECK(time_highs(words<uint64_t>(write_raw(RawFormat::kEvt21, wrap, options(0, 0)))) ==
        std::vector<std::string>{th64(W - 2), th64(W - 1), th64(W), th64(W + 1)});
  // One gap across two wraps (2^21 keepalives): each wrap gets its guard.
  Columns two;
  for (const int64_t u : {T - 10'000, 2 * T + 70}) two.add(1000 * u, 1, 2, 1);
  const std::string file = write_raw(RawFormat::kEvt21, two, options(0, 0));
  const size_t start = file.find("% end\n") + 6;
  std::vector<uint64_t> th;
  for (size_t i = start; i + 8 <= file.size(); i += 8) {
    uint64_t v = 0;
    std::memcpy(&v, file.data() + i, 8);
    if ((v >> 60) == 0x8) th.push_back((v >> 32) & 0x0FFFFFFF);
  }
  // g1, the keepalives W - 128 .. 2W, the guards W - 1 and 2W - 1, then 2W + 1.
  REQUIRE(th.size() == static_cast<size_t>(1 + (W / 128 + 2) + 2 + 1));
  CHECK(std::count(th.begin(), th.end(), uint64_t{0x0FFFFFFF}) == 2);
  CHECK(th[1] == static_cast<uint64_t>(W - 128));
  CHECK(th[2] == 0x0FFFFFFFu);
  CHECK(th[3] == 0u);
  CHECK(th[th.size() - 3] == 0x0FFFFFFFu);
  CHECK(th[th.size() - 2] == 0u);
  CHECK(th.back() == 1u);
}

TEST_CASE("EVT 2.0 and 2.1 refuse a first event 2^34 us or more after the origin; EVT 3.0 takes it") {
  // Their first TIME_HIGH is read as absolute (28 bits of 64 us steps), so u >= 2^34 would decode modulo 2^34.
  const int64_t T = int64_t{1} << 34;
  Columns late, ok;
  late.add(1000 * (T + 5), 1, 2, 1);
  ok.add(1000 * (T - 1), 1, 2, 1);
  ok.add(1000 * (T + 5), 3, 2, 1);  // later events may cross the wrap
  const std::string path = temp_path("late.raw");
  for (const RawFormat f : {RawFormat::kEvt2, RawFormat::kEvt21}) {
    for (const auto th : {exporter::Evt2TimeHigh::kOpenEb, exporter::Evt2TimeHigh::kOnChange}) {
      exporter::RawWriter w(path, f, 64, 64, {}, options(0, 0, th));
      CHECK_THROWS_WITH_AS(w.append(late.view(0, 1)), doctest::Contains("2^34 us TIME_HIGH loop"), exporter::Refused);
      w.append(ok.view(0, 2));
      CHECK(w.close().streams.at(0).t_last == T + 5);
    }
  }
  exporter::RawWriter w3(path, RawFormat::kEvt3, 64, 64, {}, options(0, 0));
  w3.append(late.view(0, 1));
  CHECK(w3.close().streams.at(0).t_first == T + 5);
}

TEST_CASE("F8 streaming: one append and appends of 1, 7, 997 or 2^17 events give the same bytes") {
  // A dense stream with many same-microsecond groups, duplicates, row runs and idle gaps (EVT 2.1 holds its open
  // word across appends, EVT 3.0 its TIME_HIGH, TIME_LOW and ADDR_Y state, EVT 2.0 its TIME_HIGH steps).
  std::mt19937_64 rng(7);
  Columns c;
  int64_t t = 3'000'000'123;
  for (int i = 0; i < 300'000; ++i) {
    const uint64_t r = rng();
    if (r % 50 == 0) t += static_cast<int64_t>(r % 9'000'000);  // gaps of up to 9 ms
    else if (r % 3 != 0) t += static_cast<int64_t>(r % 2000);
    const uint16_t y = static_cast<uint16_t>((r >> 20) % 4 == 0 ? (r >> 24) % 720 : 100);
    c.add(t, static_cast<uint16_t>((r >> 32) % 1280), y, static_cast<uint8_t>((r >> 40) & 1));
  }
  const exporter::TimeMap m{3'000'000, 0};
  for (const RawFormat f : {RawFormat::kEvt2, RawFormat::kEvt21, RawFormat::kEvt3}) {
    for (const auto th : {exporter::Evt2TimeHigh::kOpenEb, exporter::Evt2TimeHigh::kOnChange}) {
      const exporter::RawOptions o = options(0, 999, th);
      const std::string one = write_raw(f, c, o, 0, m);
      for (const size_t b : {size_t{1}, size_t{7}, size_t{997}, size_t{1} << 17}) {
        CHECK(write_raw(f, c, o, b, m) == one);
      }
    }
  }
  const std::string one = write_dat(c, options(0, 999), 0, m);
  for (const size_t b : {size_t{1}, size_t{997}, size_t{1} << 17}) CHECK(write_dat(c, options(0, 999), b, m) == one);
}

TEST_CASE("Refusals: arguments, order, range, origin, t_ns_rem, DAT's u32; a refused batch writes nothing") {
  const std::string path = temp_path("refused.raw");
  using exporter::Refused;
  CHECK_THROWS_AS(exporter::RawWriter(path, RawFormat::kEvt3, 2049, 720, {}, options(0, 0)), Refused);
  CHECK_THROWS_AS(exporter::RawWriter(path, RawFormat::kEvt3, 0, 720, {}, options(0, 0)), Refused);
  CHECK_THROWS_AS(exporter::DatWriter(path, 16385, 720, {}, options(0, 0)), Refused);
  CHECK_NOTHROW(exporter::DatWriter(path, 16384, 16384, {}, options(0, 0)));
  CHECK_THROWS_WITH_AS(exporter::RawWriter(path, RawFormat::kEvt3, 64, 64, {0, 999}, options(0, 0)),
                       doctest::Contains("multiple of 1000"), Refused);
  CHECK_THROWS_AS(exporter::DatWriter(path, 64, 64, {0, 1}, options(0, 0)), Refused);
  CHECK_THROWS_AS(exporter::RawWriter(path, RawFormat::kEvt3, 64, 64, {}, options(5, 4)), Refused);
  CHECK_THROWS_AS(exporter::RawWriter(path, RawFormat::kEvt3, 64, 64, {}, options(0, 1000)), Refused);
  CHECK_THROWS_AS(exporter::RawWriter(path, RawFormat::kEvt3, 64, 64, {}, options(-1, 3)), Refused);
  exporter::RawOptions bad = options(0, 0);
  bad.version = "1\n% end";
  CHECK_THROWS_AS(exporter::RawWriter(path, RawFormat::kEvt3, 64, 64, {}, bad), Refused);
  bad = options(0, 0);
  bad.evt21_legacy_endianness = true;
  CHECK_THROWS_WITH_AS(exporter::RawWriter(path, RawFormat::kEvt21, 64, 64, {}, bad),
                       doctest::Contains("legacy endianness is not implemented"), Refused);

  // Each bad batch is refused whole, so the file equals the good batches alone.
  Columns good1, good2;
  good1.add(5'000'100, 1, 2, 1);
  good1.add(5'001'100, 3, 4, 0);
  good2.add(5'002'100, 5, 6, 1);
  const exporter::TimeMap m{5'000, 0};
  const std::string expected = [&] {
    Columns both = good1;
    both.add(5'002'100, 5, 6, 1);
    return write_raw(RawFormat::kEvt21, both, options(100, 100), 0, m, 64, 64);
  }();
  for (const RawFormat f : {RawFormat::kEvt2, RawFormat::kEvt21, RawFormat::kEvt3}) {
    exporter::RawWriter w(path, f, 64, 64, m, options(100, 100));
    w.append(good1.view(0, 2));
    Columns decreasing;  // within a batch, then across batches
    decreasing.add(5'003'100, 1, 1, 1);
    decreasing.add(5'002'100, 1, 1, 1);
    CHECK_THROWS_WITH_AS(w.append(decreasing.view(0, 2)), doctest::Contains("t_reported order"), Refused);
    Columns earlier;
    earlier.add(5'000'100, 1, 1, 1);
    CHECK_THROWS_WITH_AS(w.append(earlier.view(0, 1)), doctest::Contains("t_reported order"), Refused);
    Columns outside, polarity, rem;
    outside.add(5'002'100, 64, 1, 1);
    polarity.add(5'002'100, 1, 1, 2);
    rem.add(5'002'101, 1, 1, 1);
    CHECK_THROWS_WITH_AS(w.append(outside.view(0, 1)), doctest::Contains("outside the 64 x 64 sensor"), Refused);
    CHECK_THROWS_WITH_AS(w.append(polarity.view(0, 1)), doctest::Contains("polarity"), Refused);
    CHECK_THROWS_WITH_AS(w.append(rem.view(0, 1)), doctest::Contains("declared range [100, 100]"), Refused);
    w.append(good2.view(0, 1));
    w.close();
    CHECK_THROWS_AS(w.append(good2.view(0, 1)), Refused);
    CHECK_THROWS_AS(w.close(), Refused);
    if (f == RawFormat::kEvt21) CHECK(read_file(path) == expected);
  }
  {
    exporter::RawWriter w(path, RawFormat::kEvt3, 64, 64, m, options(100, 100));
    Columns before;
    before.add(4'999'100, 1, 1, 1);  // u = -1
    CHECK_THROWS_WITH_AS(w.append(before.view(0, 1)), doctest::Contains("before the export origin"), Refused);
  }
  // DAT: u32 microseconds; u = 2^32 - 1 is the last one written.
  exporter::DatWriter d(path, 64, 64, {0, 0}, options(0, 0));
  Columns last, over;
  last.add(((int64_t{1} << 32) - 1) * 1000, 1, 1, 1);
  over.add((int64_t{1} << 32) * 1000, 1, 1, 1);
  d.append(last.view(0, 1));
  CHECK_THROWS_WITH_AS(d.append(over.view(0, 1)), doctest::Contains("DAT stores u32 microseconds"), Refused);
  const exporter::FileInfo info = d.close();
  CHECK(info.streams.at(0).count == 1);
  CHECK(info.streams.at(0).t_last == (int64_t{1} << 32) - 1);
  const std::string bytes = read_file(path);
  CHECK(hex(bytes.substr(bytes.size() - 8)) == "ffffffff01400010");
}

TEST_CASE("IMU CSV: the header, T1a t_us, the counts, %.17g values that round-trip, the refusals") {
  const std::string path = temp_path("imu.csv");
  const exporter::TimeMap m{1'000, 0};
  const int64_t t[] = {1'000'999, 1'001'000, 1'002'500};
  const double gyro[] = {0.1, -0.0, 1e-300, 3.0, 2.5e-5, -7.0 / 3.0, 0, 0, 0};
  const double accel[] = {9.81, 0.30000000000000004, -1e300, 1, 2, 3, 0, 0, 0};
  const int32_t cg[] = {1, -2, 3, 32767, -32768, 0, 0, 0, 0};
  {
    exporter::ImuCsvWriter w(path, m, true, false);
    w.append({t, gyro, accel, cg, nullptr, 2});
    w.append({t + 2, gyro + 6, accel + 6, cg + 6, nullptr, 1});
    const exporter::FileInfo info = w.close();
    CHECK(info.streams.at(0).kind == "imu");
    CHECK(info.streams.at(0).count == 3);
    CHECK(info.streams.at(0).t_first == 0);
    CHECK(info.streams.at(0).t_last == 2);
  }
  CHECK(read_file(path) ==
        "t_us,cgx,cgy,cgz,gx,gy,gz,ax,ay,az\n"
        "0,1,-2,3,0.10000000000000001,-0,1e-300,9.8100000000000005,0.30000000000000004,-1.0000000000000001e+300\n"
        "1,32767,-32768,0,3,2.5000000000000001e-05,-2.3333333333333335,1,2,3\n"
        "2,0,0,0,0,0,0,0,0,0\n");
  for (const double v : gyro) {  // %.17g round-trips
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    CHECK(std::strtod(buf, nullptr) == v);
  }
  exporter::ImuCsvWriter w(path, m, false, true);
  CHECK_THROWS_WITH_AS(w.append({t, gyro, accel, cg, nullptr, 1}), doctest::Contains("counts given"),
                       exporter::Refused);
  const int64_t back[] = {1'002'000, 1'001'000};
  CHECK_THROWS_WITH_AS(w.append({back, gyro, accel, nullptr, cg, 2}), doctest::Contains("stamp order"),
                       exporter::Refused);
  const int64_t early[] = {999'999};
  CHECK_THROWS_WITH_AS(w.append({early, gyro, accel, nullptr, cg, 1}), doctest::Contains("before the export origin"),
                       exporter::Refused);
  w.append({t, gyro, accel, nullptr, cg, 1});
  w.append({t + 1, gyro, accel, nullptr, cg, 1});  // equal stamps are in order
  w.close();
  CHECK(read_file(path) == "t_us,cax,cay,caz,gx,gy,gz,ax,ay,az\n"
                           "0,1,-2,3,0.10000000000000001,-0,1e-300,9.8100000000000005,0.30000000000000004,"
                           "-1.0000000000000001e+300\n1,1,-2,3,0.10000000000000001,-0,1e-300,9.8100000000000005,"
                           "0.30000000000000004,-1.0000000000000001e+300\n");
  CHECK_THROWS_AS(exporter::ImuCsvWriter(path, {0, 5}, false, false), exporter::Refused);
}

TEST_CASE("Time rules: T1a floors, the header date, the sensor clock and the camera window") {
  CHECK(exporter::t1a_us({0, 0}, 999) == 0);
  CHECK(exporter::t1a_us({0, 0}, -1) == -1);
  CHECK(exporter::t1a_us({-5, 0}, -1) == 4);
  CHECK(exporter::utc_date(0) == "1970-01-01 00:00:00");
  CHECK(exporter::utc_date(-1) == "1969-12-31 23:59:59");
  CHECK(exporter::utc_date(1760000000999999999) == "2025-10-09 08:53:20");
  CHECK(exporter::utc_date(951782400000000000) == "2000-02-29 00:00:00");  // a leap day
  CHECK(exporter::header_date({-1'000'000, 2'000'000'000}) == "1970-01-01 00:00:01");
  // clock_c is readout.h's formula with no delay and no quantum (the engine's sensor_clock_ns).
  for (const double drift : {0.0, 200.0, -15.3}) {
    for (const int64_t t : {int64_t{0}, int64_t{1'234'567'891}, int64_t{-50'000'001}, int64_t{987'654'321'012}}) {
      const sensor::Readout r{0, 0, 0, -388'200, drift};
      CHECK(exporter::sensor_clock_ns(-388'200, drift, t) == sensor::reported_time(r, t));
    }
  }
  CHECK_THROWS_AS(exporter::sensor_clock_ns(1, 0.0, INT64_MAX), exporter::Refused);
  CHECK_THROWS_AS(exporter::sensor_clock_ns(0, 1e4, 5), exporter::Refused);
  // The window and the close bound clock_c(t_e + d0 + q), d0 and q rounded as the sensor rounds them.
  const exporter::CameraTimes c = exporter::camera_times(-388'200, 200.0, 3e-4, 1e-6, 2'000'000'000, 1'500'000'000,
                                                         std::nullopt);
  const sensor::Readout r{0, 0, 0, -388'200, 200.0};
  CHECK(c.lo_ns.value() == sensor::reported_time(r, 1'500'000'000));
  CHECK_FALSE(c.hi_ns.has_value());
  CHECK(c.file_end_bound_ns == sensor::reported_time(r, 2'000'000'000 + 300'000 + 1'000));
  CHECK_THROWS_AS(exporter::camera_times(0, 0, 1.0, 0, 0, std::nullopt, std::nullopt), exporter::Refused);
}
