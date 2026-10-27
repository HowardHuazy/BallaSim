// PLY import (ballasim.scene.assets.ply.load_ply and assets._import for a PLY; docs/dev/CPP_PORT.md item 5, S5),
// bitwise as the Python: ascii and binary PLY, vertex normals, UVs and uniform vertex colours, the Artec
// multi-texture layout (per-corner UVs, wedges split after angle-weighted normals on the welded mesh, the atlas
// decoded by libpng), then the units/axes transform, validation and texture size limiting. With `baked` (scans.md
// §3.5, the C1 feature "baked") every material is baked radiance, and a texture above max_texture_size is halved in
// linear light and rounded once to binary16 (`texture_linear`) instead of being re-quantized to 8 bits.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "scene/assets/png.h"
#include "scene/geometry.h"
#include "scene/tree.h"

namespace balla::scene::assets {

// Linear RGB as IEEE binary16 bits, (height, width, 3) row-major: C1 Material.texture_linear (scans.md §3.1).
struct LinearImage {
  int64_t height = 0, width = 0;
  std::vector<uint16_t> bits;
};

struct Material {
  std::array<float, 3> base_color{};
  std::optional<Image> texture;
  std::array<float, 3> emissive{};
  bool baked = false;                         // C1 shading "baked": the texture times base_color is radiance
  std::optional<LinearImage> texture_linear;  // baked only, in place of `texture` (which is then empty)
  int32_t texture_tile = 0;                   // C1 texture_tile: 0, or the pitch P of a per-face tile atlas (replica.h)
};

struct Asset {
  geo::Mesh mesh;
  std::vector<Material> materials;
  Tree info;
};

// --- the PLY element reader, shared with the Replica importer (replica.cpp; scans.md §4.5) ---

enum class PlyType { kI1, kU1, kI2, kU2, kI4, kU4, kF4, kF8 };

// One property's values (every value as a double: the PLY types convert exactly) and, for a list, each row's range.
// stacked: the Python held the list as a 2-D array (all rows equally long and read as one block), else a list.
struct PlyColumn {
  PlyType type = PlyType::kF4;
  bool list = false, stacked = false;
  std::vector<double> values;
  std::vector<size_t> offsets;   // list: rows + 1 offsets into values
  size_t rows() const { return list ? offsets.size() - 1 : values.size(); }
  size_t len(size_t r) const { return offsets[r + 1] - offsets[r]; }
};

using PlyElement = std::map<std::string, PlyColumn>;     // property name -> column
using PlyElements = std::map<std::string, PlyElement>;   // element name -> properties

// Reads an ascii or binary PLY into its elements (AssetError for a malformed header or body).
PlyElements read_ply(const std::string& path);

// ply.load_ply(path, scale=scale, uv_origin=uv_origin).
Asset load_ply(const std::string& path, double scale, const std::string& uv_origin);

// assets._import of a PLY: load_ply at scale 1, the 4 x 4 units/axes transform (unless np.allclose to the identity),
// validate_mesh, and textures halved in linear light until no side exceeds max_texture_size. `baked` marks every
// material baked (info "baked_materials") and halves its texture with downsample_linear.
Asset import_ply(const std::string& path, const geo::M4& transform, const std::string& uv_origin,
                 int64_t max_texture_size, bool baked = false);

// assets._downsample: (texture, halvings).
std::pair<Image, int> downsample(Image tex, int64_t max_size);

// assets._downsample_linear (baked textures, scans.md §3.5): the halvings of `downsample` in fp64 linear light, the
// result rounded once to binary16 instead of re-quantized at each level: (texture_linear, halvings). With no halving
// (no side above max_size) it returns {nullopt, 0}: the 8-bit texture stays as it is.
std::pair<std::optional<LinearImage>, int> downsample_linear(const Image& tex, int64_t max_size);

// The IEEE binary16 bits of x, rounded to nearest even directly from fp64 (NumPy's float64 -> float16 cast):
// subnormals kept, |x| >= 65520 gives infinity, NaN a quiet NaN.
uint16_t binary16_bits(double x);

}  // namespace balla::scene::assets
