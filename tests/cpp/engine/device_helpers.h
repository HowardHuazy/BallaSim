// Device runs of the engine's primitives and of the fake WindowEngine (device_helpers.cu), for the
// host = device tests of test_device.cpp.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "engine/band_map.h"
#include "fake.h"
#include "fake_pv.h"
#include "plan_cases.h"

namespace balla::engine::test {

bool device_available();
std::vector<uint32_t> device_exclusive_scan(const std::vector<uint32_t>& in);
std::vector<uint64_t> device_exclusive_scan(const std::vector<uint64_t>& in);
std::pair<std::vector<uint64_t>, std::vector<uint32_t>> device_sort_pairs(const std::vector<uint64_t>& keys,
                                                                          const std::vector<uint32_t>& values,
                                                                          int end_bit);
std::vector<uint32_t> device_select(const std::vector<uint8_t>& flags);
unsigned long long device_count(const std::vector<uint8_t>& pred);
std::vector<float> device_eta(const MidpointParams& p, const std::vector<float>& a, const std::vector<float>& m,
                              const std::vector<float>& b);
FakeRuns device_fake_window(const FakeSetup& s, size_t budget);
// Several windows on one device engine.
std::vector<FakeRuns> device_fake_windows(const std::vector<FakeSetup>& setups, size_t budget);

// P2-B B6c: the per-pixel band's radii (BandRadius: per pixel its seed motion and kind, per tile the floor; returns
// the radii and the histogram) and the cascade (band_map.h) on a width x height radius map over the domain dom, on
// the device.
std::pair<std::vector<int32_t>, std::vector<uint32_t>> device_band_radius(const std::vector<double>& seed_m,
                                                                          const std::vector<uint8_t>& kind,
                                                                          const std::vector<double>& tile_floor,
                                                                          const std::vector<uint32_t>& pixel_tile,
                                                                          double support, int32_t cap);
std::vector<uint8_t> device_band_cascade(const std::vector<int32_t>& radius, const std::vector<int32_t>& levels,
                                         int32_t width, int32_t height, const BandDomain& dom);

// P2-D D1: plan_cases.h's pixel stage on the device.
PixelStageRun device_pixel_stage(const PixelStageInput& in);
// P2-D D3a: fake_pv.h's P&V window on the device.
PvFakeRuns device_pv_fake_window(const PvFakeSetup& s, size_t budget);

// The host versions, through the same entry points (serial primitives).
unsigned long long host_count(const std::vector<uint8_t>& pred);

}  // namespace balla::engine::test
