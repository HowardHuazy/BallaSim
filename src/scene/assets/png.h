// PNG decoding with the environment's libpng (docs/dev/CPP_PORT.md item 5, S5): the RGB pixels PIL's
// Image.open(path).convert("RGB") gives for an 8-bit PNG (no gamma or colour-space transform): RGB as stored, RGBA and
// grey + alpha with the alpha dropped, grey replicated, palettes expanded (a tRNS chunk ignored). 16-bit and sub-8-bit
// grey files are refused (AssetError): PIL's conversions of those are not reproduced. Every EVIMO2 atlas is 8-bit RGB.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace balla::scene::assets {

struct Image {
  int64_t height = 0, width = 0;
  std::vector<uint8_t> rgb;   // (height, width, 3), row-major
};

// PIL's decompression-bomb limit (Image.MAX_IMAGE_PIXELS * 2): larger images are refused before any allocation.
inline constexpr uint64_t kMaxPngPixels = 2ull * 89478485ull;

// Throws AssetError for a missing, unreadable, oversized or unsupported file.
Image read_png_rgb(const std::string& path);

}  // namespace balla::scene::assets
