#include "scene/assets/replica.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "scene/errors.h"
#include "render/thread_pool.h"

namespace balla::scene::assets {

namespace fs = std::filesystem;

double binary16_value(uint16_t bits) {
  const int e = (bits >> 10) & 31, m = bits & 1023;
  const double sign = (bits & 0x8000) ? -1.0 : 1.0;
  if (e == 0) return sign * std::ldexp(static_cast<double>(m), -24);
  if (e == 31) return m ? std::numeric_limits<double>::quiet_NaN() : sign * std::numeric_limits<double>::infinity();
  return sign * std::ldexp(static_cast<double>(1024 + m), e - 25);
}

namespace {

constexpr int kSourceTile = 16;                        // Replica's tileSize (textures/parameters.json)
constexpr int64_t kAutoQuadLimit = 2'500'000;          // auto T is 8 up to this many quads, else 4 (scans.md §4.5.2)
constexpr double kMaxBinary16 = 65504.0;

// Quad corner in[k]'s uv in its tile (mesh-ptex.geom): (0,0), (1,0), (1,1), (0,1).
constexpr std::array<std::array<int, 2>, 4> kCornerUv = {{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};

const std::vector<double>& binary16_table() {
  static const std::vector<double> table = [] {
    std::vector<double> t(65536);
    for (uint32_t i = 0; i < t.size(); ++i) t[i] = binary16_value(static_cast<uint16_t>(i));
    return t;
  }();
  return table;
}

std::vector<char> read_bytes(const fs::path& path, const char* what) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw AssetError(path.string() + ": missing " + what);
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> out(static_cast<size_t>(n));
  if (n > 0 && !f.read(out.data(), n)) throw AssetError(path.string() + ": cannot read " + what);
  return out;
}

template <class U>
U take(const std::vector<char>& b, size_t& off, const fs::path& path) {
  if (off + sizeof(U) > b.size()) throw AssetError(path.string() + ": truncated");
  U v;
  std::memcpy(&v, b.data() + off, sizeof(U));
  off += sizeof(U);
  return v;
}

// habitat/sorted_faces.bin: u64 n_chunks, then per chunk u64 n and n u32 face ids (PTexLib.cpp SplitMesh).
std::vector<std::vector<uint32_t>> read_sorted_faces(const fs::path& path, int64_t quads) {
  const std::vector<char> b = read_bytes(path, "habitat/sorted_faces.bin (the chunks' face order; a Replica copy "
                                               "without it cannot be mapped to its atlases)");
  size_t off = 0;
  const auto n_chunks = take<uint64_t>(b, off, path);
  if (n_chunks == 0 || n_chunks > b.size() / 8) throw AssetError(path.string() + ": implausible chunk count");
  std::vector<std::vector<uint32_t>> chunks(n_chunks);
  std::vector<uint8_t> seen(static_cast<size_t>(quads), 0);
  int64_t total = 0;
  for (auto& chunk : chunks) {
    const auto n = take<uint64_t>(b, off, path);
    if (n > (b.size() - off) / 4) throw AssetError(path.string() + ": truncated");
    chunk.resize(static_cast<size_t>(n));
    std::memcpy(chunk.data(), b.data() + off, static_cast<size_t>(n) * 4);
    off += static_cast<size_t>(n) * 4;
    for (const uint32_t f : chunk) {
      if (f >= static_cast<uint64_t>(quads) || seen[f]) {
        throw AssetError(path.string() + ": not a permutation of the " + std::to_string(quads) + " faces (face " +
                         std::to_string(f) +
                         (f >= static_cast<uint64_t>(quads) ? " is out of range)" : " listed twice)"));
      }
      seen[f] = 1;
    }
    total += static_cast<int64_t>(n);
  }
  if (off != b.size()) {
    throw AssetError(path.string() + ": " + std::to_string(b.size() - off) + " bytes after the last chunk");
  }
  if (total != quads) {
    throw AssetError(path.string() + ": not a permutation of the " + std::to_string(quads) + " faces (it lists " +
                     std::to_string(total) + ")");
  }
  return chunks;
}

struct QuadEdge {
  uint32_t lo, hi, face;
  uint8_t edge;
};
struct Neighbour {
  int32_t face = -1;
  uint8_t edge = 0;
  bool same = false;
};
struct SharedEdge {
  uint32_t f, g;
  uint8_t e, m;
  bool same;
  uint64_t rank;
};
struct Adjacency {
  std::vector<std::array<Neighbour, 4>> neighbours;
  std::vector<SharedEdge> shared;
  int64_t boundary = 0, nonmanifold = 0;
};

uint64_t edge_rank(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

Adjacency adjacency(const PlyColumn& faces) {
  const size_t nq = faces.rows();
  std::vector<QuadEdge> edges;
  edges.reserve(4 * nq);
  const auto vertex = [&](size_t f, int e) {
    return static_cast<uint32_t>(faces.values[faces.offsets[f] + static_cast<size_t>(e)]);
  };
  for (size_t f = 0; f < nq; ++f) {
    for (int e = 0; e < 4; ++e) {
      const uint32_t a = vertex(f, e), b = vertex(f, (e + 1) % 4);
      edges.push_back({std::min(a, b), std::max(a, b), static_cast<uint32_t>(f), static_cast<uint8_t>(e)});
    }
  }
  std::sort(edges.begin(), edges.end(), [](const QuadEdge& a, const QuadEdge& b) {
    if (a.lo != b.lo) return a.lo < b.lo;
    if (a.hi != b.hi) return a.hi < b.hi;
    if (a.face != b.face) return a.face < b.face;
    return a.edge < b.edge;
  });
  Adjacency out;
  out.neighbours.resize(nq);
  for (size_t i = 0; i < edges.size();) {
    size_t j = i + 1;
    while (j < edges.size() && edges[j].lo == edges[i].lo && edges[j].hi == edges[i].hi) ++j;
    if (j - i == 2 && edges[i].face != edges[i + 1].face && edges[i].lo != edges[i].hi) {
      const QuadEdge& a = edges[i];
      const QuadEdge& b = edges[i + 1];
      const bool same = vertex(a.face, a.edge) == vertex(b.face, b.edge);
      out.neighbours[a.face][a.edge] = {static_cast<int32_t>(b.face), b.edge, same};
      out.neighbours[b.face][b.edge] = {static_cast<int32_t>(a.face), a.edge, same};
      out.shared.push_back({a.face, b.face, a.edge, b.edge, same,
                            edge_rank((uint64_t{a.lo} << 32) | b.hi)});
    } else if (j - i == 1) {
      ++out.boundary;
    } else {
      ++out.nonmanifold;
    }
    i = j;
  }
  return out;
}

std::array<int, 2> edge_inside(int e, int s, int d, int T) {
  switch (e) {
    case 0: return {s, d};
    case 1: return {T - 1 - d, s};
    case 2: return {T - 1 - s, T - 1 - d};
    default: return {d, T - 1 - s};
  }
}

void copy_neighbours(std::vector<uint16_t>& bits, const std::vector<uint32_t>& tile_of, const Adjacency& adj,
                     int T, int64_t S, int64_t side, int workers) {
  const int G = T / 2, P = 2 * T;
  const auto offset = [&](size_t f, int x, int y) {
    const int64_t k = tile_of[f];
    return static_cast<size_t>(((k / S * P + G + y) * side + k % S * P + G + x) * 3);
  };
  render::ThreadPool pool(workers);
  pool.parallel_for(tile_of.size(), 256, [&](size_t first, size_t last) {
    for (size_t f = first; f < last; ++f) {
      for (int e = 0; e < 4; ++e) {
        const Neighbour& n = adj.neighbours[f][static_cast<size_t>(e)];
        if (n.face < 0) continue;  // initial packing already clamps boundary and non-manifold edges
        for (int s = 0; s < T; ++s) {
          for (int d = 1; d <= G; ++d) {
            const auto dst = edge_inside(e, s, -d, T);
            const auto src = edge_inside(n.edge, n.same ? s : T - 1 - s, d - 1, T);
            std::memcpy(&bits[offset(f, dst[0], dst[1])],
                        &bits[offset(static_cast<size_t>(n.face), src[0], src[1])], 3 * sizeof(uint16_t));
          }
        }
      }
      // The top/bottom gutter rows take precedence in the corners, extending their own x endpoints.
      for (int y = -G; y < T + G; ++y) {
        if (y >= 0 && y < T) continue;
        for (int x = -G; x < T + G; ++x) {
          if (x >= 0 && x < T) continue;
          std::memcpy(&bits[offset(f, x, y)], &bits[offset(f, std::clamp(x, 0, T - 1), y)],
                      3 * sizeof(uint16_t));
        }
      }
    }
  });
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const size_t m = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(m), values.end());
  const double hi = values[m];
  return values.size() % 2 ? hi : 0.5 * (hi + *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(m)));
}

Tree self_check(const std::vector<uint16_t>& bits, const std::vector<uint32_t>& tile_of, Adjacency& adj,
                int T, int64_t S, int64_t side, double limit) {
  // Sample without replacement by a fixed hash of the undirected edge: deterministic across platforms and workers.
  const size_t count = std::min<size_t>(20000, adj.shared.size());
  std::partial_sort(adj.shared.begin(), adj.shared.begin() + static_cast<std::ptrdiff_t>(count), adj.shared.end(),
                    [](const SharedEdge& a, const SharedEdge& b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.f < b.f || (a.f == b.f && a.e < b.e);
  });
  const int G = T / 2, P = 2 * T;
  const auto luma = [&](size_t f, int x, int y) {
    const int64_t k = tile_of[f];
    const size_t off = static_cast<size_t>(((k / S * P + G + y) * side + k % S * P + G + x) * 3);
    const auto& v = binary16_table();
    return 0.2126 * v[bits[off]] + 0.7152 * v[bits[off + 1]] + 0.0722 * v[bits[off + 2]];
  };
  const auto dlog = [](double a, double b) { return std::fabs(std::log(std::max(a, 1e-3)) - std::log(std::max(b, 1e-3))); };
  std::vector<double> cross, interior;
  for (size_t i = 0; i < count; ++i) {
    const auto& e = adj.shared[i];
    std::vector<double> c, in;
    for (int s = 0; s < T; ++s) {
      const auto a = edge_inside(e.e, s, 0, T);
      const auto b = edge_inside(e.m, e.same ? s : T - 1 - s, 0, T);
      c.push_back(dlog(luma(e.f, a[0], a[1]), luma(e.g, b[0], b[1])));
      in.push_back(dlog(luma(e.f, s, T / 2 - 1), luma(e.f, s, T / 2)));
    }
    cross.push_back(median(std::move(c)));
    interior.push_back(median(std::move(in)));
  }
  const double numerator = median(std::move(cross)), denominator = median(std::move(interior));
  const double ratio = denominator > 0.0 ? numerator / denominator
                                         : numerator == 0.0 ? 1.0 : std::numeric_limits<double>::infinity();
  if (count && !(ratio <= limit)) {
    throw AssetError("Replica self-check ratio " + std::to_string(ratio) + " exceeds " + std::to_string(limit) +
                     ": cross-edge radiance is inconsistent with interior variation (check face/tile mapping)");
  }
  Tree out = Tree::dict();
  out["ratio"] = Tree::real(ratio);
  out["edge_median"] = Tree::real(numerator);
  out["interior_median"] = Tree::real(denominator);
  out["edges_tested"] = Tree::integer(static_cast<int64_t>(count));
  out["boundary_edges"] = Tree::integer(adj.boundary);
  out["nonmanifold_edges"] = Tree::integer(adj.nonmanifold);
  return out;
}

}  // namespace

ReplicaResult import_replica(const std::string& scene_dir, const ReplicaInputs& in, const ReplicaOptions& opt) {
  const fs::path dir(scene_dir);
  if (in.tile_size != kSourceTile) {
    throw AssetError(scene_dir + ": tileSize " + std::to_string(in.tile_size) + " is not " +
                     std::to_string(kSourceTile));
  }
  if (opt.tile_texels != 0 && opt.tile_texels != 4 && opt.tile_texels != 8 && opt.tile_texels != 16) {
    throw AssetError("tile_texels must be auto, 4, 8 or 16, got " + std::to_string(opt.tile_texels));
  }

  if (opt.workers < 1 || !(opt.self_check_ratio > 0.0) || !std::isfinite(opt.self_check_ratio)) {
    throw AssetError("Replica workers must be positive and self_check_ratio positive and finite");
  }

  // --- the mesh: vertices, and quads only --------------------------------------------------------------------------
  if (!fs::exists(dir / "mesh.ply")) throw AssetError(scene_dir + ": missing mesh.ply");
  const PlyElements els = read_ply((dir / "mesh.ply").string());
  const auto vit = els.find("vertex"), fit = els.find("face");
  if (vit == els.end() || fit == els.end()) throw AssetError(scene_dir + "/mesh.ply: needs vertex and face elements");
  const PlyElement& vert = vit->second;
  for (const char* name : {"x", "y", "z"}) {
    const auto c = vert.find(name);
    if (c == vert.end() || c->second.list) {
      throw AssetError(scene_dir + "/mesh.ply: vertex property " + name + " missing");
    }
  }
  const size_t nv = vert.at("x").values.size();
  if (nv > std::numeric_limits<uint32_t>::max()) throw AssetError(scene_dir + "/mesh.ply: too many vertices");
  if (vert.at("y").values.size() != nv || vert.at("z").values.size() != nv) {
    throw AssetError(scene_dir + "/mesh.ply: vertex x, y and z differ in length");
  }
  // Replica keeps each quad's four corners and their normals, so the generic triangle cleaner cannot run here.
  // Check its numeric invariant before converting positions or normals into the float32 mesh.
  for (const char* name : {"x", "y", "z", "nx", "ny", "nz"}) {
    const auto c = vert.find(name);
    if (c == vert.end() || c->second.list) continue;
    for (const double v : c->second.values) {
      if (!std::isfinite(v) || std::fabs(v) > static_cast<double>(std::numeric_limits<float>::max())) {
        throw AssetError(scene_dir + "/mesh.ply: vertex property " + name + " is not finite in float32");
      }
    }
  }
  auto face_it = fit->second.find("vertex_indices");
  if (face_it == fit->second.end()) face_it = fit->second.find("vertex_index");
  if (face_it == fit->second.end() || !face_it->second.list) {
    throw AssetError(scene_dir + "/mesh.ply: face element has no vertex_indices list");
  }
  const PlyColumn& faces = face_it->second;
  const size_t nq = faces.rows();
  if (nq == 0) throw AssetError(scene_dir + "/mesh.ply: no faces");
  if (nq > std::numeric_limits<uint32_t>::max() / 4) throw AssetError(scene_dir + "/mesh.ply: too many faces");
  for (size_t f = 0; f < nq; ++f) {
    if (faces.len(f) != 4) {
      throw AssetError(scene_dir + "/mesh.ply: face " + std::to_string(f) + " has " + std::to_string(faces.len(f)) +
                       " vertices; a Replica scene is quads only");
    }
    for (size_t k = 0; k < 4; ++k) {
      const double v = faces.values[faces.offsets[f] + k];
      if (!(v >= 0.0 && v < static_cast<double>(nv) && std::floor(v) == v)) {
        throw AssetError(scene_dir + "/mesh.ply: face " + std::to_string(f) + " names vertex " +
                         std::to_string(v) + " of " + std::to_string(nv) + " (expected an integer index)");
      }
    }
  }
  const std::vector<std::vector<uint32_t>> chunks =
      read_sorted_faces(dir / "habitat" / "sorted_faces.bin", static_cast<int64_t>(nq));

  // --- T, G, P and the packed atlas's size -------------------------------------------------------------------------
  const int T = opt.tile_texels != 0 ? opt.tile_texels : (static_cast<int64_t>(nq) <= kAutoQuadLimit ? 8 : 4);
  const int G = T / 2, P = T + 2 * G, block = kSourceTile / T;
  int64_t S = static_cast<int64_t>(std::sqrt(static_cast<double>(nq)));      // tiles per side: ceil(sqrt(quads))
  while (S * S < static_cast<int64_t>(nq)) ++S;
  while (S > 1 && (S - 1) * (S - 1) >= static_cast<int64_t>(nq)) --S;
  const int64_t side = S * P;
  if (side * side > (int64_t{1} << 32)) {
    throw AssetError(scene_dir + ": the packed atlas (" + std::to_string(side) + " x " + std::to_string(side) +
                     ") has more than 2^32 texels; use a smaller tile_texels");
  }
  if (33 * side > (int64_t{1} << 30)) {
    throw AssetError(scene_dir + ": atlas side " + std::to_string(side) + " breaks (|uv| + 32) * side <= 2^30");
  }

  // --- the atlas: chunk by chunk, each face's tile box-meaned to T x T and framed by clamped gutters -----------------
  std::vector<uint16_t> bits(static_cast<size_t>(side * side * 3), 0);
  std::vector<uint32_t> tile_of(nq);                       // face -> its position in sorted_faces order
  std::vector<uint16_t> tile(static_cast<size_t>(T * T * 3));
  const std::vector<double>& value = binary16_table();
  int64_t k = 0;                                           // the running position over all chunks
  std::vector<int64_t> chunk_dims, chunk_quads;
  for (size_t c = 0; c < chunks.size(); ++c) {
    const fs::path path = dir / "textures" / (std::to_string(c) + "-color-ptex.hdr");
    std::error_code ec;
    const uintmax_t bytes = fs::file_size(path, ec);
    if (ec) throw AssetError(path.string() + ": missing chunk atlas");
    const int64_t dim = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(bytes) / 6.0)));
    if (dim <= 0 || static_cast<uintmax_t>(dim * dim) * 6 != bytes || dim % kSourceTile != 0) {
      throw AssetError(path.string() + ": " + std::to_string(bytes) +
                       " bytes are not a square RGB binary16 atlas whose side is a multiple of " +
                       std::to_string(kSourceTile));
    }
    const int64_t W = dim / kSourceTile;
    const std::vector<uint32_t>& chunk = chunks[c];
    if (static_cast<int64_t>(chunk.size()) > W * W) {
      throw AssetError(path.string() + ": chunk " + std::to_string(c) + " holds " + std::to_string(chunk.size()) +
                       " faces, more than the " + std::to_string(W * W) + " tiles of its atlas");
    }
    std::vector<uint16_t> src(static_cast<size_t>(dim * dim * 3));
    {
      std::ifstream f(path, std::ios::binary);
      if (!f.read(reinterpret_cast<char*>(src.data()), static_cast<std::streamsize>(bytes))) {
        throw AssetError(path.string() + ": cannot read the chunk atlas");
      }
    }
    chunk_dims.push_back(dim);
    chunk_quads.push_back(static_cast<int64_t>(chunk.size()));
    for (size_t i = 0; i < chunk.size(); ++i, ++k) {
      tile_of[chunk[i]] = static_cast<uint32_t>(k);
      const int64_t x0 = (static_cast<int64_t>(i) % W) * kSourceTile, y0 = (static_cast<int64_t>(i) / W) * kSourceTile;
      for (int ty = 0; ty < T; ++ty) {
        for (int tx = 0; tx < T; ++tx) {
          for (int ch = 0; ch < 3; ++ch) {
            double sum = 0.0;
            for (int by = 0; by < block; ++by) {
              for (int bx = 0; bx < block; ++bx) {
                const int64_t sx = x0 + tx * block + bx, sy = y0 + ty * block + by;
                const double v = value[src[static_cast<size_t>((sy * dim + sx) * 3 + ch)]];
                if (!(v >= 0.0 && v <= kMaxBinary16)) {
                  throw AssetError(path.string() + ": face " + std::to_string(chunk[i]) + " (tile " +
                                   std::to_string(i) + " of chunk " + std::to_string(c) + ") texel (" +
                                   std::to_string(sx - x0) + ", " + std::to_string(sy - y0) +
                                   ") is negative or not finite");
                }
                sum += v;
              }
            }
            tile[static_cast<size_t>((ty * T + tx) * 3 + ch)] = binary16_bits(sum / (block * block));
          }
        }
      }
      // The padded tile: texel (r, c) of the P x P cell reads the interior texel clamped into [0, T - 1].
      const int64_t col = k % S, row = k / S;
      for (int r = 0; r < P; ++r) {
        const int sy = std::clamp(r - G, 0, T - 1);
        uint16_t* dst = &bits[static_cast<size_t>(((row * P + r) * side + col * P) * 3)];
        for (int cc = 0; cc < P; ++cc) {
          std::memcpy(dst + 3 * cc, &tile[static_cast<size_t>((sy * T + std::clamp(cc - G, 0, T - 1)) * 3)], 6);
        }
      }
    }
  }

  Adjacency adj = adjacency(faces);
  Tree check = self_check(bits, tile_of, adj, T, S, side, opt.self_check_ratio);
  const int workers = std::min(opt.workers, static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency())));
  copy_neighbours(bits, tile_of, adj, T, S, side, workers);

  // --- the mesh: 4 vertices per quad, uv into the padded atlas ------------------------------------------------------
  ReplicaPart part;
  geo::Mesh& m = part.mesh;
  m.positions.resize(nq * 4 * 3);
  m.uvs.resize(nq * 4 * 2);
  m.indices.resize(nq * 6);
  const auto to_uv = [side](int64_t texels) {      // a position in the atlas's texels (v down) as a fp32 uv
    return static_cast<float>(static_cast<double>(texels) / static_cast<double>(side));
  };
  const PlyColumn *px = &vert.at("x"), *py = &vert.at("y"), *pz = &vert.at("z");
  const auto nx = vert.find("nx"), ny = vert.find("ny"), nz = vert.find("nz");
  const bool normals = nx != vert.end() && ny != vert.end() && nz != vert.end() && !nx->second.list &&
                       !ny->second.list && !nz->second.list && nx->second.values.size() == nv &&
                       ny->second.values.size() == nv && nz->second.values.size() == nv;
  if (normals) m.normals.resize(nq * 4 * 3);
  for (size_t q = 0; q < nq; ++q) {
    const int64_t col = tile_of[q] % S, row = tile_of[q] / S;
    for (size_t j = 0; j < 4; ++j) {
      const auto v = static_cast<size_t>(faces.values[faces.offsets[q] + j]);
      const size_t o = 4 * q + j;
      m.positions[3 * o] = static_cast<float>(px->values[v]);
      m.positions[3 * o + 1] = static_cast<float>(py->values[v]);
      m.positions[3 * o + 2] = static_cast<float>(pz->values[v]);
      if (normals) {
        m.normals[3 * o] = static_cast<float>(nx->second.values[v]);
        m.normals[3 * o + 1] = static_cast<float>(ny->second.values[v]);
        m.normals[3 * o + 2] = static_cast<float>(nz->second.values[v]);
      }
      m.uvs[2 * o] = to_uv(col * P + G + kCornerUv[j][0] * T);
      m.uvs[2 * o + 1] = to_uv(row * P + G + kCornerUv[j][1] * T);
    }
    const auto a = static_cast<uint32_t>(4 * q);     // the strip's diagonal 0-2 is shared by both triangles
    const uint32_t tri[6] = {a, a + 1, a + 2, a, a + 2, a + 3};
    std::copy(tri, tri + 6, &m.indices[6 * q]);
  }
  m.has_normals = normals;
  m.has_uvs = true;

  Material& mat = part.material;
  mat.base_color = {1.0f, 1.0f, 1.0f};
  mat.baked = true;
  mat.texture_linear = LinearImage{side, side, std::move(bits)};
  mat.texture_tile = P;

  ReplicaResult out;
  out.parts.push_back(std::move(part));
  Tree& info = out.info = Tree::dict();
  info["format"] = Tree::string("replica");
  info["quads"] = Tree::integer(static_cast<int64_t>(nq));
  info["chunks"] = Tree::integer(static_cast<int64_t>(chunks.size()));
  info["chunk_quads"] = Tree::ints(chunk_quads);
  info["chunk_atlas_dims"] = Tree::ints(chunk_dims);
  info["tile_texels"] = Tree::integer(T);
  info["gutter_texels"] = Tree::integer(G);
  info["tile_pitch"] = Tree::integer(P);
  info["atlas_side"] = Tree::integer(side);
  info["split_size"] = Tree::real(in.split_size);
  info["baked_materials"] = Tree::ints({0});
  info["self_check"] = std::move(check);
  return out;
}

Tree replica_reports(const std::vector<std::vector<std::array<double, 3>>>& polygons,
                     const std::optional<std::array<double, 3>>& gravity) {
  double area = 0.0;
  for (const auto& polygon : polygons) {
    if (polygon.size() < 3) throw AssetError("glass.sur: boundary_w needs at least three vertices");
    std::array<double, 3> sum{};
    for (size_t i = 0; i < polygon.size(); ++i) {
      const auto& a = polygon[i];
      const auto& b = polygon[(i + 1) % polygon.size()];
      for (const double x : a) if (!std::isfinite(x)) throw AssetError("glass.sur: boundary_w is not finite");
      sum[0] += a[1] * b[2] - a[2] * b[1];
      sum[1] += a[2] * b[0] - a[0] * b[2];
      sum[2] += a[0] * b[1] - a[1] * b[0];
    }
    area += 0.5 * std::hypot(sum[0], sum[1], sum[2]);
  }
  if (!std::isfinite(area)) throw AssetError("glass.sur: boundary_w area is not finite");
  Tree out = Tree::dict(), warnings = Tree::list();
  out["glass_reflectors"] = Tree::integer(static_cast<int64_t>(polygons.size()));
  out["glass_area_m2"] = Tree::real(area);
  if (!polygons.empty()) warnings.push_back(Tree::string("Replica glass reflectors are rendered as captured"));
  if (gravity) {
    const auto& g = *gravity;
    const double norm = std::hypot(g[0], g[1], g[2]);
    if (!(norm > 0.0) || !std::isfinite(norm)) throw AssetError("Replica gravity_dir must be finite and nonzero");
    const double tilt = std::acos(std::clamp(-g[2] / norm, -1.0, 1.0)) * (180.0 / std::acos(-1.0));
    out["gravity_dir"] = Tree::reals({g[0], g[1], g[2]});
    out["gravity_tilt_deg"] = Tree::real(tilt);
    if (tilt > 1.0) warnings.push_back(Tree::string("Replica gravity tilts " + std::to_string(tilt) + " deg from -Z (over 1 deg)"));
  }
  out["warnings"] = std::move(warnings);
  return out;
}

}  // namespace balla::scene::assets
