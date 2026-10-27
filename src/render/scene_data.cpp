#include "render/scene_data.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <numbers>

#include "core/camera/injective_domain.h"
#include "core/pmath/trig.h"
#include "core/rng/purpose.h"
#include "core/rng/sobol.h"
#include "core/rng/subsample.h"
#include "core/traj/spline_build.h"
#include "render/ao_proxy.h"
#include "render/patches.h"
#include "render/shade/ao_traced.h"
#include "render/shade/bsdf_tables.h"
#include "render/shade/lights.h"
#include "render/texture.h"

namespace balla::render {
namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

RigidPose to_pose(const std::array<double, 16>& t) {
  RigidPose p;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) p.r.m[i][j] = t[4 * i + j];
  }
  p.p = {t[3], t[7], t[11]};
  return p;
}

double dot3(const std::array<double, 3>& a, const std::array<float, 3>& b) {
  return (a[0] * static_cast<double>(b[0]) + a[1] * static_cast<double>(b[1])) + a[2] * static_cast<double>(b[2]);
}

// Checks the settings and builds the sub-sample warp of their pixel response.
rng::WarpTable<float> make_warp(const SceneSettings& s) {
  if (s.n_subsamples < 1 || s.n_subsamples > static_cast<uint32_t>(kMaxSubsamples)) {
    throw std::invalid_argument("render: n_subsamples must be in [1, " + std::to_string(kMaxSubsamples) + "]");
  }
  for (double w : s.sensor_rgb_weights) {
    if (!std::isfinite(w) || w < 0.0) throw std::invalid_argument("render: sensor_rgb_weights must be finite and >= 0");
  }
  if (!(s.sensor_rgb_weights[0] + s.sensor_rgb_weights[1] + s.sensor_rgb_weights[2] > 0.0)) {
    throw std::invalid_argument("render: sensor_rgb_weights must not all be 0");
  }
  if (s.lut_samples_per_px < 1) throw std::invalid_argument("render: lut_samples_per_px must be >= 1");
  if (s.warp_cells < rng::kMinWarpCells || s.warp_cells > rng::kMaxWarpCells) {
    throw std::invalid_argument("render: warp_cells must be in [" + std::to_string(rng::kMinWarpCells) + ", " +
                                std::to_string(rng::kMaxWarpCells) + "]");
  }
  return rng::WarpTable<float>(rng::PixelResponse(s.aperture_px, s.blur_sigma_px, s.truncation_sigmas), s.warp_cells);
}

// Appends a body-frame mesh at world pose t to a static world whose has_normals / has_uvs / has_patches are set:
// positions and normals transformed in fp64 and rounded once, zero normals and UVs where the mesh has none, the
// mesh's global patch ids (one per triangle) under has_patches. Indices count from vertex_base + the world's first
// vertex (a held set's tail follows the static world's vertices).
template <class MaterialOf>
void append_static(GeometryData& world, const RigidPose& t, uint32_t id, const std::vector<float>& positions,
                   const std::vector<float>& normals, const std::vector<float>& uvs,
                   const std::vector<uint32_t>& indices, MaterialOf material_of, const std::vector<uint32_t>& patches,
                   size_t vertex_base = 0) {
  const size_t nv = positions.size() / 3, nt = indices.size() / 3, v0 = world.num_vertices(), t0 = world.num_triangles();
  if (static_cast<uint64_t>(vertex_base) + v0 + nv > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("render: static world has more than 2^32 vertices");
  }
  const auto base = static_cast<uint32_t>(vertex_base + v0);
  world.positions.resize(3 * (v0 + nv));
  if (world.has_normals) world.normals.resize(3 * (v0 + nv));
  if (world.has_uvs) world.uvs.resize(2 * (v0 + nv));
  world.indices.resize(3 * (t0 + nt));
  world.materials.resize(t0 + nt);
  world.instances.resize(t0 + nt, id);
  if (world.has_patches) world.patches.insert(world.patches.end(), patches.begin(), patches.end());
  for (size_t v = 0; v < nv; ++v) {
    const Vec3d x{positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]};
    store3(&world.positions[3 * (v0 + v)], t.r * x + t.p);
    if (world.has_normals) {
      Vec3d n{0.0, 0.0, 0.0};
      if (!normals.empty()) n = t.r * Vec3d{normals[3 * v], normals[3 * v + 1], normals[3 * v + 2]};
      store3(&world.normals[3 * (v0 + v)], n);
    }
    if (world.has_uvs) {
      world.uvs[2 * (v0 + v)] = uvs.empty() ? 0.0f : uvs[2 * v];
      world.uvs[2 * (v0 + v) + 1] = uvs.empty() ? 0.0f : uvs[2 * v + 1];
    }
  }
  for (size_t f = 0; f < nt; ++f) {
    for (size_t c = 0; c < 3; ++c) world.indices[3 * (t0 + f) + c] = base + indices[3 * f + c];
    world.materials[t0 + f] = material_of(f);
  }
}

// Appends the set Y_l of the light with index `light` in the bundle (plan p2f §4.6): the first kLightPoints points of a
// 2-D Owen-scrambled Sobol sequence seeded by Philox(kLightSampleSet, camera 0, pixel = light, index 0), stored by the
// 32 x 32 cell each lies in (render/shade/lights.h light_cell; the net holds exactly one per cell), as the local
// coordinates (a, b) of SceneView::light_points: a rect's (2u - 1, 2v - 1) and a disk's concentric image of that square
// (Shirley and Chiu, JGT 1997), the unit disk, both scaled by the half sizes at the lookup. The rect's values are exact
// in fp32 (u, v have 24 bits); the disk's go through the portable trigonometry (core/pmath). The map is the one pbrt-v4
// (`SampleUniformDiskConcentric`, used by `Disk::Sample`) and Mitsuba 3 (`warp::square_to_uniform_disk_concentric`,
// used by the `disk` shape's `sample_position`; equal to it to 1e-7 in v3.9.1, measured) apply to the unit square:
// it keeps the cells of a stratified net compact on the disk (RENDER.md, "F8a-1: conventions").
void append_light_points(std::vector<float>& out, uint64_t run_seed, uint32_t light, bool disk) {
  const uint32_t seed = rng::draw(run_seed, rng::Purpose::kLightSampleSet, 0, light, 0).v[0];
  const size_t base = out.size();
  out.resize(base + 2 * static_cast<size_t>(kLightPoints));
  std::vector<bool> filled(static_cast<size_t>(kLightPoints), false);
  for (uint32_t i = 0; i < static_cast<uint32_t>(kLightPoints); ++i) {
    uint32_t zx, zy;
    rng::owen_sobol_2d(i, seed, &zx, &zy);
    const uint32_t cell = light_cell(zx, zy);
    if (filled[cell]) throw std::logic_error("render: a light's point set has two points in one cell");
    filled[cell] = true;
    float a = 2.0f * rng::to_unit_interval<float>(zx) - 1.0f, b = 2.0f * rng::to_unit_interval<float>(zy) - 1.0f;
    if (disk && (a != 0.0f || b != 0.0f)) {
      const double da = static_cast<double>(a), db = static_cast<double>(b);
      const bool a_major = std::abs(da) > std::abs(db);
      const double r = a_major ? da : db;
      const double theta = a_major ? (std::numbers::pi / 4.0) * (db / da) : std::numbers::pi / 2.0 - (std::numbers::pi / 4.0) * (da / db);
      double sn, cs;
      pmath::sincos(theta, &sn, &cs);
      a = static_cast<float>(r * cs);
      b = static_cast<float>(r * sn);
    }
    out[base + 2 * cell] = a;
    out[base + 2 * cell + 1] = b;
  }
}

bool single_material(const GeometryData& g) {
  return !g.materials.empty() &&
         std::all_of(g.materials.begin(), g.materials.end(), [&](uint32_t m) { return m == g.materials[0]; });
}

// The cameras' NIR fractions as a flat [camera * light groups + group] table (missing cameras and empty rows are 0), with
// the checks of SceneData's constructor (std::invalid_argument for a row of the wrong length or an f outside [0, 1]).
std::vector<double> nir_fractions(const SceneBundle& b, const std::vector<std::vector<double>>& nir_fraction) {
  const size_t num_cameras = b.cameras.size(), num_light_groups = b.light_groups.size();
  if (nir_fraction.size() > num_cameras) {
    throw std::invalid_argument("render: nir_fraction has " + std::to_string(nir_fraction.size()) +
                                " cameras, the bundle " + std::to_string(num_cameras));
  }
  std::vector<double> nir_f(num_cameras * num_light_groups, 0.0);
  for (size_t ci = 0; ci < nir_fraction.size(); ++ci) {
    const std::vector<double>& row = nir_fraction[ci];
    if (row.empty()) continue;
    const std::string where = "render: camera '" + b.cameras[ci].name + "' nir_fraction";
    if (row.size() != num_light_groups) {
      throw std::invalid_argument(where + " needs one value per light group (" + std::to_string(num_light_groups) +
                                  ") or []");
    }
    for (size_t g = 0; g < row.size(); ++g) {
      if (!(row[g] >= 0.0 && row[g] <= 1.0)) throw std::invalid_argument(where + " must lie in [0, 1]");
      nir_f[ci * num_light_groups + g] = row[g];
    }
  }
  return nir_f;
}

// A static body split from the static world (scans.md §4.2, SceneView::baked_proxies): a trace geometry of its own with
// trajectory -1, its triangles in the world frame exactly as append_static puts them into the world (so its rays and
// hits are the merged world's), one instance id, and the movers' bounding sphere (about the AABB centre, rounded
// outwards) of those world-frame positions. Under crease (has_patches) it carries the body's global patch ids.
template <class MaterialOf>
GeometryData split_static(const Body& body, MaterialOf material_of, const std::vector<uint32_t>& patches, bool crease) {
  const Mesh& mesh = body.mesh;
  GeometryData g;
  g.instance = body.id;
  g.has_normals = !mesh.normals.empty();
  g.has_uvs = !mesh.uvs.empty();
  g.has_patches = crease;
  append_static(g, to_pose(body.T_world_body), body.id, mesh.positions, mesh.normals, mesh.uvs, mesh.indices,
                material_of, patches);
  g.instances.clear();  // GeometryView::instance
  const std::vector<float>& x = g.positions;
  Vec3d lo{x[0], x[1], x[2]}, hi = lo;
  for (size_t v = 0; v < x.size() / 3; ++v) {
    lo = {std::min<double>(lo.x, x[3 * v]), std::min<double>(lo.y, x[3 * v + 1]), std::min<double>(lo.z, x[3 * v + 2])};
    hi = {std::max<double>(hi.x, x[3 * v]), std::max<double>(hi.y, x[3 * v + 1]), std::max<double>(hi.z, x[3 * v + 2])};
  }
  g.centre = 0.5 * (lo + hi);
  double r2 = 0.0;
  for (size_t v = 0; v < x.size() / 3; ++v) {
    const Vec3d d = Vec3d{x[3 * v], x[3 * v + 1], x[3 * v + 2]} - g.centre;
    r2 = std::max(r2, dot(d, d));
  }
  g.radius = static_cast<float>(std::sqrt(r2) * (1.0 + 1e-6) + 1e-9);
  return g;
}

}  // namespace

static_assert(static_cast<int32_t>(MoverAo::kProxy) == kMoverAoProxy &&
              static_cast<int32_t>(MoverAo::kTraced) == kMoverAoTraced);  // SceneView::mover_ao

bool SceneData::needs_two_channels(const SceneBundle& b, const std::vector<std::vector<double>>& nir_fraction) {
  const std::vector<double> f = nir_fractions(b, nir_fraction);
  return !std::all_of(f.begin(), f.end(), [&](double x) { return x == f.front(); });
}

SceneData::SceneData(const SceneBundle& b, const SceneSettings& settings,
                     const std::vector<std::vector<double>>& nir_fraction)
    : fixed64_(settings.pixel_integral == PixelIntegral::kFixed64 ? 1 : 0),
      unlit_mask_(settings.unlit_mask ? 1 : 0),
      texture_shadow_(settings.texture_shadow == TextureShadow::kWeighted ? 1 : 0),
      mover_ao_(static_cast<int32_t>(settings.mover_ao)),
      warp_(make_warp(settings)) {
  validate(b);
  const std::array<double, 3>& w = settings.sensor_rgb_weights;

  // The cameras' NIR fractions f[camera][group] (P2-F F11a). All equal (f = 0 included): one albedo, the NIR folded in at
  // level 0 when f != 0; else two channels and a mix per (camera, group).
  const size_t num_light_groups = b.light_groups.size();
  const std::vector<double> nir_f = nir_fractions(b, nir_fraction);
  const double nir_f0 = nir_f.empty() ? 0.0 : nir_f[0];
  const bool nir_equal = std::all_of(nir_f.begin(), nir_f.end(), [&](double f) { return f == nir_f0; });
  const bool nir_two_channel = !nir_equal;
  const double nir_fold = nir_equal ? nir_f0 : 0.0;  // the folded f; the channels are mixed in fp64 at level 0
  const bool nir_on = nir_two_channel || nir_fold != 0.0;
  nir_two_channel_ = nir_two_channel ? 1 : 0;
  const auto mix = [](double vis, double nir, double f) { return (1.0 - f) * vis + f * nir; };

  // Trajectories (positions are already relative to origin_world, contract C1).
  spline_nodes_.reserve(b.trajectories.size());
  piece_views_.reserve(b.trajectories.size());
  for (const Trajectory& t : b.trajectories) {
    const size_t n = t.pos.size() / 3;
    std::vector<Quat<double>> q(n);
    std::vector<Vec3d> p(n);
    for (size_t c = 0; c < n; ++c) {
      q[c] = {t.rot[4 * c], t.rot[4 * c + 1], t.rot[4 * c + 2], t.rot[4 * c + 3]};
      p[c] = {t.pos[3 * c], t.pos[3 * c + 1], t.pos[3 * c + 2]};
    }
    if (t.pieces.empty()) {
      spline_nodes_.push_back(traj::build_spline_nodes(q, p, {0.0, 0.0, 0.0}));
      piece_views_.emplace_back();
      const int32_t h = static_cast<int32_t>(t.h_ns);
      trajectories_.push_back({{spline_nodes_.back().data(), static_cast<int32_t>(n) - t.order + 1, h, 1e9 / h},
                               t.t0_ns, t.order, t.span_begin_ns, t.span_end_ns});
      continue;
    }
    // Pieces (blender.md §4.7.7): each piece's nodes from its own rows (its first node has d = dp = 0, as
    // _core.spline_eval_pieces_f64 builds them), stored at the rows' offsets in one array.
    std::vector<traj::SplineNode<double>> nodes;
    nodes.reserve(n);
    for (const TrajectoryPiece& piece : t.pieces) {
      const auto c0 = static_cast<size_t>(piece.ctrl_begin), rows = static_cast<size_t>(piece.ctrl_end) - c0;
      const auto part = traj::build_spline_nodes(std::span<const Quat<double>>(q).subspan(c0, rows),
                                                 std::span<const Vec3d>(p).subspan(c0, rows), {0.0, 0.0, 0.0});
      nodes.insert(nodes.end(), part.begin(), part.end());
    }
    spline_nodes_.push_back(std::move(nodes));
    std::vector<PieceView> views;
    for (const TrajectoryPiece& piece : t.pieces) {
      const int32_t h = static_cast<int32_t>(piece.h_ns);
      views.push_back({piece.begin_ns, piece.t0_ns,
                       {spline_nodes_.back().data() + piece.ctrl_begin, piece.ctrl_end - piece.ctrl_begin - t.order + 1,
                        h, 1e9 / h}});
    }
    piece_views_.push_back(std::move(views));
    TrajectoryView tv{piece_views_.back()[0].spline, t.t0_ns, t.order, t.span_begin_ns, t.span_end_ns};
    tv.pieces = piece_views_.back().data();
    tv.num_pieces = static_cast<int32_t>(piece_views_.back().size());
    trajectories_.push_back(tv);
  }

  // Bodies in ascending id order; materials and textures in the same order.
  std::vector<const Body*> sorted;
  for (const Body& body : b.bodies) sorted.push_back(&body);
  std::sort(sorted.begin(), sorted.end(), [](const Body* x, const Body* y) { return x->id < y->id; });
  std::vector<uint32_t> material_base(sorted.size());
  for (size_t i = 0; i < sorted.size(); ++i) {
    const Body& body = *sorted[i];
    material_base[i] = static_cast<uint32_t>(materials_.size());
    const std::vector<Material> fallback{Material{}};
    const std::vector<Material>& mats = body.materials.empty() ? fallback : body.materials;
    const bool has_uvs = !body.mesh.uvs.empty();
    for (const Material& m : mats) {
      const std::array<double, 3> wb{w[0] * static_cast<double>(m.base_color[0]), w[1] * static_cast<double>(m.base_color[1]),
                                     w[2] * static_cast<double>(m.base_color[2])};
      MaterialView mv{static_cast<float>((wb[0] + wb[1]) + wb[2]), static_cast<float>(dot3(w, m.emissive)), -1, -1};
      if (m.has_lobe()) {  // C1 feature "pbr" (P2-F F6): alpha = r^2, so alpha^2 = r^4; the fields stay 0 without a lobe
        const double r2 = static_cast<double>(m.roughness) * static_cast<double>(m.roughness);
        mv.alpha2 = static_cast<float>(r2 * r2);
        mv.metallic = m.metallic;
        mv.specular = m.specular;
        mv.lobe = 1;
        specular_ = true;
      }
      // An 8-bit sRGB texture, or a baked material's binary16 linear one (scans.md §4.1); validate() allows one.
      auto pyramid = [&](std::vector<float>& texels, int32_t max_levels) {
        return m.texture.empty() ? build_linear_mip_pyramid(m.texture_linear.data(), m.texture_width, m.texture_height,
                                                            wb, texels, max_levels)
                                 : build_mono_mip_pyramid(m.texture.data(), m.texture_width, m.texture_height,
                                                          m.texture_channels, wb, texels, max_levels);
      };
      if (nir_on && m.shading != Shading::kBaked) {  // lit materials: a baked one is radiance, with no NIR albedo
        // P2-F F11a: the visible and NIR albedo (texture.h band_images) folded at level 0 or as two channels. A material
        // with a texture of either band is textured, also when only its NIR is (nir_form texture).
        const BandImages im = band_images(m, wb);
        if (im.width > 0 && has_uvs) {
          mv.texture = static_cast<int32_t>(texture_levels_.size());
          // A tile atlas (S5a) is mono and on its own grid: it cannot take two channels or a resampled NIR image.
          if (m.texture_tile > 0 && (nir_two_channel || im.width != m.texture_width || im.height != m.texture_height)) {
            throw std::invalid_argument("render: texture_tile cannot be combined with two-channel near-infrared albedo "
                                        "or a near-infrared image of another size (a tile atlas is read per cell, mono)");
          }
          const int32_t max_levels =
              m.texture_tile > 0 ? std::countr_zero(static_cast<uint32_t>(m.texture_tile)) + 1 : 0;
          if (nir_two_channel) {
            texture_levels_.push_back(build_mip_pyramid2(im.vis, im.nir, im.width, im.height, texels_));
          } else {
            std::vector<double> level(im.vis.size());
            for (size_t t = 0; t < level.size(); ++t) level[t] = mix(im.vis[t], im.nir[t], nir_fold);
            texture_levels_.push_back(build_mip_pyramid(std::move(level), im.width, im.height, texels_, max_levels));
          }
          texture_channels_.push_back(nir_two_channel ? 2 : 1);
          texture_tiles_.push_back(m.texture_tile);
        } else {
          double vis = im.vis_const, nir = im.nir_const;
          if (im.width > 0) {  // textures without UVs cannot be placed: their means (the 1 x 1 top levels)
            std::vector<float> scratch;
            const auto lv_vis = build_mip_pyramid(im.vis, im.width, im.height, scratch);
            vis = scratch[lv_vis.back().offset];
            scratch.clear();
            const auto lv_nir = build_mip_pyramid(im.nir, im.width, im.height, scratch);
            nir = scratch[lv_nir.back().offset];
          }
          mv.albedo = static_cast<float>(nir_two_channel ? vis : mix(vis, nir, nir_fold));
          mv.nir_albedo = static_cast<float>(nir_two_channel ? nir : 0.0);
        }
      } else if (!m.texture.empty() || !m.texture_linear.empty()) {
        if (has_uvs) {
          // A tile atlas (scans.md §4.5.6, S5a) stops at one texel per cell: log2(P) + 1 levels (P a power of two).
          const int32_t max_levels =
              m.texture_tile > 0 ? std::countr_zero(static_cast<uint32_t>(m.texture_tile)) + 1 : 0;
          mv.texture = static_cast<int32_t>(texture_levels_.size());
          texture_levels_.push_back(pyramid(texels_, max_levels));
          texture_channels_.push_back(1);
          texture_tiles_.push_back(m.texture_tile);
        } else {
          // A texture without UVs cannot be placed; its mean (the 1 x 1 top level) is the albedo.
          std::vector<float> scratch;
          const auto levels = pyramid(scratch, 0);
          mv.albedo = scratch[levels.back().offset];
        }
      }
      if (m.shading == Shading::kBaked) {
        // S_g = fl32(double(K) double(a_g)): K already holds the run multiplier (scans.md §3.4); without shares the
        // radiance is all in group 0.
        mv.baked = static_cast<int32_t>(baked_scale_.size());
        for (size_t g = 0; g < b.light_groups.size(); ++g) {
          const double share = m.baked_groups.empty() ? (g == 0 ? 1.0 : 0.0) : static_cast<double>(m.baked_groups[g]);
          baked_scale_.push_back(static_cast<float>(static_cast<double>(m.baked_scale) * share));
        }
      }
      materials_.push_back(mv);
    }
  }

  // Trace geometries: the merged static world first, then one per mover. Both are assembled in
  // locals (bodies interleave by id) and moved into geometries_ at the end.
  // With baked proxies (scans.md §4.2) the world holds the scan bodies only: every other static body, lit or baked, is
  // split into a geometry of its own among the movers (split_static), so that it shadows the scan as inserted.
  const bool split = !baked_scale_.empty() && std::any_of(b.lights.begin(), b.lights.end(), [](const Light& l) {
    return l.proxy;
  });
  baked_proxies_ = split ? 1 : 0;
  GeometryData world;
  std::vector<GeometryData> movers;
  world.radius = kInfinity;
  // accuracy.shading_patches crease (P2-F F5): each body's patches (render/patches.h) as global ids, in body order;
  // rule (b) cuts a hard edge of any angle where either face's material has a lobe (P2-F F6).
  const bool crease = settings.shading_patches == ShadingPatches::kCrease;
  std::vector<uint8_t> lobe_flags;
  if (specular_) {
    for (const MaterialView& mv : materials_) lobe_flags.push_back(static_cast<uint8_t>(mv.lobe));
  }
  world.has_patches = crease;
  for (size_t i = 0; i < sorted.size(); ++i) {
    const Body& body = *sorted[i];
    if (!body.trajectory && (!split || body.scan)) {
      world.has_normals |= !body.mesh.normals.empty();
      world.has_uvs |= !body.mesh.uvs.empty();
    }
  }
  for (size_t i = 0; i < sorted.size(); ++i) {
    const Body& body = *sorted[i];
    const Mesh& mesh = body.mesh;
    const size_t nv = mesh.positions.size() / 3, nt = mesh.indices.size() / 3;
    auto material_of = [&](size_t tri) {
      return material_base[i] + (mesh.material_ids.empty() ? 0u : mesh.material_ids[tri]);
    };
    std::vector<uint32_t> patches;
    if (crease && nt > 0) {
      std::vector<uint32_t> mats(nt);
      for (size_t f = 0; f < nt; ++f) mats[f] = material_of(f);
      const Patches p = build_patches({mesh.positions, mesh.normals, mesh.indices}, mats, lobe_flags);
      const size_t base = patch_material_.size();
      if (base + p.count > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("render: more than 2^32 shading patches");
      }
      patch_material_.resize(base + p.count);
      patches.resize(nt);
      for (size_t f = 0; f < nt; ++f) {
        patches[f] = static_cast<uint32_t>(base) + p.of_triangle[f];
        patch_material_[patches[f]] = mats[f];  // one material per patch (rule (a))
      }
    }
    if (!body.trajectory && (!split || body.scan)) {
      bodies_.push_back({body.id, body.semantic, 0});
      append_static(world, to_pose(body.T_world_body), body.id, mesh.positions, mesh.normals, mesh.uvs, mesh.indices,
                    material_of, patches);
      continue;
    }
    if (!body.trajectory) {  // split (scans.md §4.2); without triangles never hit, as a mover's
      bodies_.push_back({body.id, body.semantic, nt == 0 ? -1 : static_cast<int32_t>(movers.size()) + 1});
      if (nt > 0) {
        movers.push_back(split_static(body, material_of, patches, crease));
        if (settings.mover_ao == MoverAo::kTraced) {
          // P2-F F9b indexes ao_reach_ by trace geometry and pushes the world's 0 at the first mover: the world's first,
          // then this static body's own (it casts no ambient ray: reach 0), so the reaches stay aligned (S3a-1).
          if (ao_reach_.empty()) ao_reach_.push_back(0.0f);
          ao_reach_.push_back(0.0f);
        }
      }
      continue;
    }
    if (nt == 0) {  // a dynamic body without triangles is never hit (it may still carry lights)
      bodies_.push_back({body.id, body.semantic, -1});
      continue;
    }
    bodies_.push_back({body.id, body.semantic, static_cast<int32_t>(movers.size()) + 1});
    GeometryData g;
    g.instance = body.id;
    g.trajectory = *body.trajectory;
    g.positions = mesh.positions;
    g.indices = mesh.indices;
    g.normals = mesh.normals;
    g.uvs = mesh.uvs;
    g.has_normals = !mesh.normals.empty();
    g.has_uvs = !mesh.uvs.empty();
    g.has_patches = crease;
    g.patches = std::move(patches);
    for (size_t f = 0; f < nt; ++f) g.materials.push_back(material_of(f));
    // Bounding sphere about the AABB centre, rounded outwards (sphere culling must be conservative).
    Vec3d lo{mesh.positions[0], mesh.positions[1], mesh.positions[2]}, hi = lo;
    for (size_t v = 0; v < nv; ++v) {
      lo = {std::min<double>(lo.x, mesh.positions[3 * v]), std::min<double>(lo.y, mesh.positions[3 * v + 1]),
            std::min<double>(lo.z, mesh.positions[3 * v + 2])};
      hi = {std::max<double>(hi.x, mesh.positions[3 * v]), std::max<double>(hi.y, mesh.positions[3 * v + 1]),
            std::max<double>(hi.z, mesh.positions[3 * v + 2])};
    }
    g.centre = 0.5 * (lo + hi);
    double r2 = 0.0;
    for (size_t v = 0; v < nv; ++v) {
      const Vec3d d = Vec3d{mesh.positions[3 * v], mesh.positions[3 * v + 1], mesh.positions[3 * v + 2]} - g.centre;
      r2 = std::max(r2, dot(d, d));
    }
    g.radius = static_cast<float>(std::sqrt(r2) * (1.0 + 1e-6) + 1e-9);
    if (settings.mover_ao == MoverAo::kTraced) {
      // accuracy.mover_ao traced (P2-F F9b): the mover's rays reach 2 d0 of its proxy (none without one).
      if (ao_reach_.empty()) ao_reach_.push_back(0.0f);  // the static world
      ao_reach_.push_back(static_cast<float>(2.0 * fit_ao_proxy(mesh.positions, mesh.indices).d0));
      traces_ao_ = traces_ao_ || ao_reach_.back() > 0.0f;
    }
    if (settings.mover_ao == MoverAo::kProxy) {
      // accuracy.mover_ao proxy (P2-F F9a): the mover's sphere set in its body frame, posed by its geometry's rows.
      const AoProxy proxy = fit_ao_proxy(mesh.positions, mesh.indices);
      if (!proxy.spheres.empty()) {
        const auto bi = static_cast<int32_t>(ao_bodies_.size());
        ao_bodies_.push_back({body.id, static_cast<int32_t>(movers.size()) + 1, static_cast<float>(proxy.d0)});
        for (const AoProxySphere& sp : proxy.spheres) {
          const Vec3d c{sp.centre[0], sp.centre[1], sp.centre[2]};
          ao_spheres_.push_back({{static_cast<float>(c.x), static_cast<float>(c.y), static_cast<float>(c.z)},
                                 static_cast<float>(sp.radius), bi});
          ao_centres_.push_back(c);
        }
      }
    }
    movers.push_back(std::move(g));
  }
  geometries_.push_back(std::move(world));
  for (GeometryData& g : movers) geometries_.push_back(std::move(g));
  for (GeometryData& g : geometries_) {
    g.single_material = single_material(g);
    pose_geometries_.push_back({g.trajectory, g.centre, g.radius});
  }

  // Cameras: injective domain over the sub-sample support, fp32 ray LUT, per-pixel pattern seeds.
  const double support = warp_.x_max();
  for (size_t ci = 0; ci < b.cameras.size(); ++ci) {
    const Camera& c = b.cameras[ci];
    auto k = [&](size_t i) { return i < c.dist.size() ? c.dist[i] : 0.0; };
    CameraView view{};
    std::unique_ptr<camera::RayLut<float>> lut;
    if (c.model == CameraModelKind::kRadtan) {
      camera::RadtanCamera<double> cam{c.fx, c.fy, c.cx, c.cy, k(0), k(1), k(2), k(3), k(4), 0.0, 0.0};
      cam.r_lim = camera::radtan_injective_domain(cam, c.width, c.height, support).r_lim;
      lut = std::make_unique<camera::RayLut<float>>(cam, c.width, c.height, support, settings.lut_samples_per_px);
      view.model = CameraModel::kRadtan;
      view.radtan = {static_cast<float>(cam.fx), static_cast<float>(cam.fy), static_cast<float>(cam.cx),
                     static_cast<float>(cam.cy), static_cast<float>(cam.k1), static_cast<float>(cam.k2),
                     static_cast<float>(cam.p1), static_cast<float>(cam.p2), static_cast<float>(cam.k3),
                     static_cast<float>(cam.r_lim), 0.0f};
    } else {
      camera::KannalaBrandtCamera<double> cam{c.fx, c.fy, c.cx, c.cy, k(0), k(1), k(2), k(3), 0.0};
      cam.theta_lim = camera::kannala_brandt_injective_domain(cam, c.width, c.height, support).theta_lim;
      lut = std::make_unique<camera::RayLut<float>>(cam, c.width, c.height, support, settings.lut_samples_per_px);
      view.model = CameraModel::kKannalaBrandt;
      view.kb = {static_cast<float>(cam.fx), static_cast<float>(cam.fy), static_cast<float>(cam.cx),
                 static_cast<float>(cam.cy), static_cast<float>(cam.k1), static_cast<float>(cam.k2),
                 static_cast<float>(cam.k3), static_cast<float>(cam.k4), static_cast<float>(cam.theta_lim)};
    }
    auto data = std::unique_ptr<CameraData>(new CameraData{view, std::move(*lut), {}, {}});
    const uint32_t npix = static_cast<uint32_t>(c.width) * static_cast<uint32_t>(c.height);
    data->seeds.resize(npix);
    for (uint32_t p = 0; p < npix; ++p) {
      data->seeds[p] = rng::subsample_pattern_seed(settings.seed, static_cast<uint32_t>(ci), p);
    }
    data->view.lut = data->lut.view();
    data->view.pattern_seeds = data->seeds.data();
    data->view.width = c.width;
    data->view.height = c.height;
    if (nir_two_channel) {  // zero otherwise: a folded scene's f is in its albedos
      for (size_t g = 0; g < num_light_groups; ++g) {
        data->view.nir_fraction[g] = static_cast<float>(nir_f[ci * num_light_groups + g]);
      }
    }
    cameras_.push_back(std::move(data));
    pose_cameras_.push_back({c.rig_trajectory, to_pose(c.T_rig_cam)});
  }

  // Lights: ambient and background terms fold into per-group constants; the others get pose-table rows. Materials see
  // the sensor response w (band albedo and emission, DESIGN §7.2); a light's colour only scales it
  // relative to white, so it is weighted by w / sum(w): white light of intensity I delivers I.
  num_groups_ = static_cast<int32_t>(b.light_groups.size());
  const double w_sum = (w[0] + w[1]) + w[2];
  std::vector<uint32_t> area_light_ids;  // the bundle index of each area light, in LightView::area_index order
  for (size_t light_index = 0; light_index < b.lights.size(); ++light_index) {
    const Light& l = b.lights[light_index];
    const float intensity = static_cast<float>(static_cast<double>(l.intensity) * dot3(w, l.color) / w_sum);
    if (l.kind == BundleLightKind::kAmbient) {
      ambient_[static_cast<size_t>(l.group)] += intensity;
      if (l.proxy) fill_[static_cast<size_t>(l.group)] += intensity;  // A^f_g: the proxy ambient, part of A_g
      continue;
    }
    if (l.kind == BundleLightKind::kBackground) {
      background_[static_cast<size_t>(l.group)] += intensity;
      continue;
    }
    LightView lv{};
    switch (l.kind) {
      case BundleLightKind::kPoint: lv.kind = LightKind::kPoint; break;
      case BundleLightKind::kSpot: lv.kind = LightKind::kSpot; break;
      case BundleLightKind::kDirectional: lv.kind = LightKind::kDirectional; break;
      // Area lights (P2-F F8a-1): the data of the shadow rays; the shading of them is F8a-2's, and the Renderer
      // constructor refuses them until then.
      case BundleLightKind::kRect:
      case BundleLightKind::kDisk: {
        const bool disk = l.kind == BundleLightKind::kDisk;
        lv.kind = disk ? LightKind::kDisk : LightKind::kRect;
        lv.half_u = static_cast<float>(0.5 * l.size[0]);
        lv.half_v = static_cast<float>(0.5 * l.size[1]);
        // c_L: the analytic form factor integrates the inscribed 32-gon, whose area is 16 sin(pi/16) of the unit disk's
        // pi; the gain restores the disk's radiance (plan p2f §4.6).
        constexpr double kPi = std::numbers::pi;
        lv.area_gain = disk ? static_cast<float>(kPi / (16.0 * pmath::sin(kPi / 16.0))) : 1.0f;
        lv.point_offset = static_cast<int32_t>(light_points_.size() / 2);
        lv.area_index = num_area_lights_++;
        append_light_points(light_points_, settings.seed, static_cast<uint32_t>(light_index), disk);
        area_light_ids.push_back(static_cast<uint32_t>(light_index));
        break;
      }
      case BundleLightKind::kAmbient:
      case BundleLightKind::kBackground: throw std::logic_error("render: ambient and background lights have no pose row");
    }
    lv.group = l.group;
    lv.intensity = intensity;
    lv.cos_outer = std::cos(l.spot_outer_rad);
    lv.cos_inner = std::cos(l.spot_inner_rad);
    lv.shadow_bit = l.casts_shadows ? num_shadow_lights_++ : -1;
    lv.proxy = l.proxy ? 1 : 0;
    if (lv.proxy != 0 && lv.shadow_bit >= 0) {  // P_g, M_g, b_g: validate() lets a proxy light cast shadows only
      const auto g = static_cast<size_t>(l.group);
      proxy_mask_[g] |= 1u << lv.shadow_bit;
      if (proxy_slot_[g] < 0) proxy_slot_[g] = lv.shadow_bit;  // shadow bits ascend with the lights
    }
    lights_.push_back(lv);
    PoseLightDef def{-1, {{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0}}, {l.position[0], l.position[1], l.position[2]},
                     {0, 0, 1}};
    if (lv.kind != LightKind::kPoint) {
      const Vec3d d{l.direction[0], l.direction[1], l.direction[2]};
      def.direction = (1.0 / std::sqrt(dot(d, d))) * d;
    }
    if (is_area(lv.kind)) {
      const Vec3d a{l.axis[0], l.axis[1], l.axis[2]};
      def.axis = (1.0 / std::sqrt(dot(a, a))) * a;
    }
    if (l.body) {
      const Body& carrier = b.bodies[static_cast<size_t>(*l.body)];
      if (carrier.trajectory) {
        def.trajectory = *carrier.trajectory;
      } else {
        def.attach = to_pose(carrier.T_world_body);
      }
    }
    pose_lights_.push_back(def);
  }
  if (traces_ao_ && num_shadow_lights_ >= kMaxShadowLights) {
    throw std::invalid_argument("render: accuracy.mover_ao traced takes bit 31 of the occlusion word, so at most " +
                                std::to_string(kMaxShadowLights - 1) + " shadow-casting lights");
  }

  // The seeds that give each sub-sample of a pixel a point of each area light's set (plan p2f §4.6): Philox(
  // kLightPadding, camera, pixel, index = the light's bundle index), one word per pixel and area light.
  if (num_area_lights_ > 0) {
    for (size_t ci = 0; ci < cameras_.size(); ++ci) {
      CameraData& c = *cameras_[ci];
      const auto npix = static_cast<size_t>(c.view.width) * static_cast<size_t>(c.view.height);
      c.light_seeds.resize(npix * static_cast<size_t>(num_area_lights_));
      for (size_t p = 0; p < npix; ++p) {
        for (int32_t a = 0; a < num_area_lights_; ++a) {
          c.light_seeds[p * static_cast<size_t>(num_area_lights_) + static_cast<size_t>(a)] =
              rng::draw(settings.seed, rng::Purpose::kLightPadding, static_cast<uint32_t>(ci), static_cast<uint32_t>(p),
                        area_light_ids[static_cast<size_t>(a)]).v[0];
        }
      }
      c.view.light_seeds = c.light_seeds.data();
    }
  }

  // Views over the host arrays.
  for (const GeometryData& g : geometries_) {
    geometry_views_.push_back({g.positions.data(), g.indices.data(), g.has_normals ? g.normals.data() : nullptr,
                               g.has_uvs ? g.uvs.data() : nullptr, g.materials.data(),
                               g.instances.empty() ? nullptr : g.instances.data(), g.instance, g.num_vertices(),
                               g.num_triangles(),
                               g.single_material, g.radius, g.has_patches ? g.patches.data() : nullptr,
                               g.trajectory >= 0 ? 1 : 0});
  }
  for (size_t t = 0; t < texture_levels_.size(); ++t) {
    texture_views_.push_back({texture_levels_[t].data(), static_cast<int32_t>(texture_levels_[t].size()),
                              texture_channels_[t], texture_tiles_[t]});
  }
  for (const auto& c : cameras_) camera_views_.push_back(c->view);
}

std::unique_ptr<TraceSet> SceneData::trace_set(std::span<const int32_t> held, std::span<const RigidPose> poses) const {
  auto t = std::make_unique<TraceSet>();
  const GeometryData& world = geometries_[0];
  GeometryData& tail = t->tail;
  tail.has_normals = world.has_normals;
  tail.has_uvs = world.has_uvs;
  tail.has_patches = world.has_patches;
  size_t nv = 0, nt = 0;
  for (const int32_t g : held) {
    const GeometryData& m = geometries_[static_cast<size_t>(g)];
    tail.has_normals |= m.has_normals;
    tail.has_uvs |= m.has_uvs;
    nv += m.num_vertices();
    nt += m.num_triangles();
  }
  tail.positions.reserve(3 * nv);
  if (tail.has_normals) tail.normals.reserve(3 * nv);
  if (tail.has_uvs) tail.uvs.reserve(2 * nv);
  tail.indices.reserve(3 * nt);
  tail.materials.reserve(nt);
  tail.instances.reserve(nt);
  if (tail.has_patches) tail.patches.reserve(nt);
  std::vector<int32_t> trace(geometries_.size(), 0);  // trace geometry of each geometry
  t->geometry.push_back(0);
  for (size_t g = 1, h = 0; g < geometries_.size(); ++g) {
    const GeometryData& m = geometries_[g];
    if (h < held.size() && held[h] == static_cast<int32_t>(g)) {
      append_static(tail, poses[h++], m.instance, m.positions, m.normals, m.uvs, m.indices,
                    [&](size_t f) { return m.materials[f]; }, m.patches, world.num_vertices());
    } else {
      trace[g] = static_cast<int32_t>(t->geometry.size());
      t->geometry.push_back(static_cast<int32_t>(g));
    }
  }
  // single_material of the world followed by the tail.
  tail.single_material = world.num_triangles() == 0
                             ? single_material(tail)
                             : world.single_material != 0 && single_material(tail) && tail.materials[0] == world.materials[0];
  for (const int32_t g : t->geometry) t->pose_geometries.push_back(pose_geometries_[static_cast<size_t>(g)]);
  for (const BodyView& b : bodies_) t->bodies.push_back({b.id, b.semantic, b.geometry > 0 ? trace[static_cast<size_t>(b.geometry)] : b.geometry});
  // Mover AO: a held mover's spheres move to the world frame at its held pose (fp64, rounded once, as its triangles).
  t->ao_bodies = ao_bodies_;
  t->ao_spheres = ao_spheres_;
  for (AoBodyView& b : t->ao_bodies) {
    const auto h = std::lower_bound(held.begin(), held.end(), b.geometry);
    const bool is_held = h != held.end() && *h == b.geometry;
    if (!is_held) {
      b.geometry = trace[static_cast<size_t>(b.geometry)];
      continue;
    }
    const RigidPose& pose = poses[static_cast<size_t>(h - held.begin())];
    const auto bi = static_cast<int32_t>(&b - t->ao_bodies.data());
    for (size_t i = 0; i < ao_spheres_.size(); ++i) {
      if (ao_spheres_[i].body == bi) store3(t->ao_spheres[i].centre, pose.r * ao_centres_[i] + pose.p);
    }
    b.geometry = 0;
  }
  return t;
}

SceneView SceneData::host_view() const {
  SceneView v{};
  v.geometries = geometry_views_.data();
  v.num_geometries = static_cast<int32_t>(geometry_views_.size());
  v.bodies = bodies_.data();
  v.num_bodies = static_cast<int32_t>(bodies_.size());
  v.materials = materials_.data();
  v.textures = texture_views_.data();
  v.texels = texels_.data();
  v.cameras = camera_views_.data();
  v.num_cameras = static_cast<int32_t>(camera_views_.size());
  v.lights = lights_.data();
  v.num_lights = static_cast<int32_t>(lights_.size());
  v.light_points = light_points_.empty() ? nullptr : light_points_.data();
  v.num_area_lights = num_area_lights_;
  v.num_shadow_lights = num_shadow_lights_;
  for (size_t j = 0; j < lights_.size(); ++j) {
    if (lights_[j].shadow_bit >= 0) v.shadow_light[lights_[j].shadow_bit] = static_cast<int32_t>(j);
  }
  v.num_groups = num_groups_;
  for (int g = 0; g < kMaxLightGroups; ++g) v.ambient[g] = ambient_[static_cast<size_t>(g)];
  v.has_background = 0;
  for (int g = 0; g < kMaxLightGroups; ++g) {
    v.background[g] = background_[static_cast<size_t>(g)];
    if (v.background[g] != 0.0f) v.has_background = 1;
  }
  v.fixed64 = fixed64_;
  v.unlit_mask = unlit_mask_;
  v.texture_shadow = texture_shadow_;
  v.specular = specular_ ? 1 : 0;
  if (specular_) v.bsdf = bsdf::host_tables();
  v.patch_material = patch_material_.empty() ? nullptr : patch_material_.data();
  v.mover_ao = mover_ao_;
  v.num_ao_spheres = static_cast<int32_t>(ao_spheres_.size());
  v.ao_bodies = ao_bodies_.empty() ? nullptr : ao_bodies_.data();
  v.ao_spheres = ao_spheres_.empty() ? nullptr : ao_spheres_.data();
  v.ao_reach = traces_ao_ ? ao_reach_.data() : nullptr;  // traced with a mover that has a proxy
  v.nir_two_channel = nir_two_channel_;
  v.warp = warp_.view();
  v.baked_scale = baked_scale_.empty() ? nullptr : baked_scale_.data();
  v.has_baked = baked_scale_.empty() ? 0 : 1;
  v.baked_proxies = baked_proxies_;
  for (size_t g = 0; g < static_cast<size_t>(kMaxLightGroups); ++g) {
    v.fill[g] = fill_[g];
    v.proxy_mask[g] = proxy_mask_[g];
    v.proxy_slot[g] = proxy_slot_[g];
  }
  return v;
}

PoseSceneView SceneData::pose_view() const {
  return {trajectories_.data(), pose_cameras_.data(), pose_geometries_.data(),
          static_cast<int32_t>(pose_geometries_.size()), pose_lights_.data(),
          static_cast<int32_t>(pose_lights_.size())};
}

}  // namespace balla::render
