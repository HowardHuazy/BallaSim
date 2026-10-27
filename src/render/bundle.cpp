#include "render/bundle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>

namespace balla::render {
namespace {

__extension__ typedef __int128 i128;

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("scene bundle: " + what); }

void require(bool ok, const std::string& what) {
  if (!ok) fail(what);
}

template <class T>
bool all_finite(const std::vector<T>& v) {
  for (const T& x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

// T must be rigid (orthonormal rotation with det +1, last row 0 0 0 1): normals and distances are
// transformed without inverse transposes.
void check_rigid(const std::array<double, 16>& t, const std::string& what) {
  for (double x : t) require(std::isfinite(x), what + " is not finite");
  require(t[12] == 0.0 && t[13] == 0.0 && t[14] == 0.0 && t[15] == 1.0,
          what + " must have last row (0, 0, 0, 1)");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double d = 0.0;
      for (int k = 0; k < 3; ++k) d += t[4 * k + i] * t[4 * k + j];
      require(std::abs(d - (i == j ? 1.0 : 0.0)) <= 1e-6, what + " is not a rigid transform");
    }
  }
  const double det = t[0] * (t[5] * t[10] - t[6] * t[9]) - t[1] * (t[4] * t[10] - t[6] * t[8]) +
                     t[2] * (t[4] * t[9] - t[5] * t[8]);
  require(det > 0.0, what + " is a reflection");
}

// IEEE binary16 bits (Material::texture_linear): an all-ones exponent is Inf or NaN; the sign bit on a non-zero
// magnitude is a negative value (-0 passes, as NumPy's `t < 0` is false for it, bundle.py).
constexpr bool half_finite(uint16_t h) { return (h & 0x7C00u) != 0x7C00u; }
constexpr bool half_negative(uint16_t h) { return (h & 0x8000u) != 0 && (h & 0x7FFFu) != 0; }

// The rules of the C1 feature "baked" for one material (bundle.py Material.validate, scans.md §3.1).
void validate_baked_fields(const Material& mat, size_t num_groups, const std::string& mn) {
  const bool baked = mat.shading == Shading::kBaked;
  require(mat.shading == Shading::kLit || baked, mn + ": shading must be lit or baked");
  require(std::isfinite(mat.baked_scale) && mat.baked_scale >= 0.0f, mn + ": baked_scale must be finite and >= 0");
  if (!mat.baked_groups.empty()) {
    require(baked, mn + ": baked_groups is set on a lit material (shares split baked radiance)");
    require(mat.baked_groups.size() == num_groups, mn + ": baked_groups needs one share per light group");
    double sum = 0.0;
    for (float a : mat.baked_groups) {
      require(std::isfinite(a) && a >= 0.0f, mn + ": baked_groups shares must be finite and >= 0");
      sum += static_cast<double>(a);
    }
    require(std::abs(sum - 1.0) <= 1e-6, mn + ": baked_groups shares must sum to 1 (1e-6)");
  }
  if (!mat.texture_linear.empty()) {
    require(baked, mn + ": texture_linear is set on a lit material (linear textures are baked radiance)");
    require(mat.texture.empty(), mn + ": has both texture and texture_linear");
    require(mat.texture_width > 0 && mat.texture_height > 0 && mat.texture_channels == 3 &&
                mat.texture_linear.size() ==
                    static_cast<size_t>(mat.texture_width) * static_cast<size_t>(mat.texture_height) * 3u,
            mn + ": texture_linear must be H x W x 3 and match its size");
    bool finite = true, negative = false;
    for (uint16_t h : mat.texture_linear) {
      finite = finite && half_finite(h);
      negative = negative || half_negative(h);
    }
    require(finite, mn + ": texture_linear contains NaN or Inf");
    require(!negative, mn + ": texture_linear has negative texels");
  }
  if (baked) {
    require(mat.emissive[0] == 0.0f && mat.emissive[1] == 0.0f && mat.emissive[2] == 0.0f,
            mn + ": emissive must be 0 on a baked material (baked radiance already contains emission)");
  }
  const int32_t p = mat.texture_tile;
  if (p != 0) {
    require(p >= 4 && (p & (p - 1)) == 0, mn + ": texture_tile must be 0 or a power of two >= 4");
    const bool textured = !mat.texture.empty() || !mat.texture_linear.empty();
    require(textured && mat.texture_width % p == 0 && mat.texture_height % p == 0,
            mn + ": texture_tile must divide the texture's width and height");
  }
}

// The rules of the C1 features "pbr" and "nir" for one material (bundle.py Material._validate_pbr_nir; p2f.md §4.3,
// §4.9, §4.11).
void validate_pbr_nir_fields(const Material& mat, const std::string& mn) {
  const auto unit = [](float x) { return std::isfinite(x) && x >= 0.0f && x <= 1.0f; };
  require(unit(mat.roughness) && unit(mat.metallic) && unit(mat.specular),
          mn + ": roughness, metallic and specular must be in [0, 1]");
  const bool baked = mat.shading == Shading::kBaked;
  require(!baked || (mat.roughness == 1.0f && mat.metallic == 0.0f && mat.specular == 0.0f),
          mn + ": a baked material is radiance: roughness, metallic and specular keep their defaults");
  require(mat.nir_form >= NirForm::kAssumed && mat.nir_form <= NirForm::kTexture, mn + ": unknown nir_form");
  require(mat.nir_params.empty() || (mat.nir_params.size() == 2 && all_finite(mat.nir_params)),
          mn + ": nir_params must be 2 finite values");
  // F11a: scale's s and scalar's c are albedos, so >= 0 (affine's a, b are clamped into [0, 1] at level 0).
  require(mat.nir_params.empty() || mat.nir_params[0] >= 0.0 ||
              (mat.nir_form != NirForm::kScale && mat.nir_form != NirForm::kScalar),
          mn + ": nir_params[0] (the scale s, or the scalar c) must be >= 0");
  const bool texture = mat.nir_form == NirForm::kTexture;
  require(texture == !mat.nir_texture.empty(),
          mn + (texture ? ": nir_form texture needs nir_texture" : ": nir_texture is set on a form other than texture"));
  if (texture) {
    require(mat.nir_texture_width > 0 && mat.nir_texture_height > 0 &&
                mat.nir_texture.size() ==
                    static_cast<size_t>(mat.nir_texture_width) * static_cast<size_t>(mat.nir_texture_height),
            mn + ": nir_texture must be H x W and match its size");
  }
  require(!baked || (mat.nir_form == NirForm::kAssumed && mat.nir_params.empty()),
          mn + ": a baked material is radiance: it has no NIR albedo (nir_form, nir_params, nir_texture)");
}

// Mesh.used_materials of bundle.py: the material ids the triangles use (0 when the mesh has none).
bool uses_baked_material(const Body& b) {
  const auto baked = [&](size_t i) { return i < b.materials.size() && b.materials[i].shading == Shading::kBaked; };
  if (b.mesh.material_ids.empty()) return baked(0);
  return std::any_of(b.mesh.material_ids.begin(), b.mesh.material_ids.end(), [&](uint16_t id) { return baked(id); });
}

void validate_mesh(const Body& b, size_t num_groups, const std::string& name) {
  const Mesh& m = b.mesh;
  require(m.positions.size() % 3 == 0, name + ": positions must be V x 3");
  require(m.indices.size() % 3 == 0, name + ": indices must be T x 3");
  const size_t nv = m.positions.size() / 3, nt = m.indices.size() / 3;
  require(nv <= std::numeric_limits<uint32_t>::max() && nt <= std::numeric_limits<uint32_t>::max(),
          name + ": mesh too large");
  require(all_finite(m.positions), name + ": positions are not finite");
  for (uint32_t i : m.indices) require(i < nv, name + ": vertex index out of range");
  require(m.normals.empty() || m.normals.size() == 3 * nv, name + ": normals must be V x 3");
  require(all_finite(m.normals), name + ": normals are not finite");
  require(m.uvs.empty() || m.uvs.size() == 2 * nv, name + ": uvs must be V x 2");
  require(all_finite(m.uvs), name + ": uvs are not finite");
  require(m.material_ids.empty() || m.material_ids.size() == nt, name + ": material_ids must have T entries");
  const size_t nm = std::max<size_t>(1, b.materials.size());
  for (uint16_t id : m.material_ids) require(id < nm, name + ": material id out of range");
  for (size_t i = 0; i < b.materials.size(); ++i) {
    const Material& mat = b.materials[i];
    const std::string mn = name + " material " + std::to_string(i);
    for (float c : mat.base_color) require(std::isfinite(c) && c >= 0.0f, mn + ": base_color must be finite and >= 0");
    for (float c : mat.emissive) require(std::isfinite(c) && c >= 0.0f, mn + ": emissive must be finite and >= 0");
    validate_baked_fields(mat, num_groups, mn);
    validate_pbr_nir_fields(mat, mn);
    if (!mat.texture.empty()) {
      require(mat.texture_width > 0 && mat.texture_height > 0, mn + ": texture has no size");
      require(mat.texture_channels == 3 || mat.texture_channels == 4, mn + ": texture must have 3 or 4 channels");
      require(mat.texture.size() == static_cast<size_t>(mat.texture_width) * static_cast<size_t>(mat.texture_height) *
                                        static_cast<size_t>(mat.texture_channels),
              mn + ": texture size does not match its shape");
    }
  }
  // Texel indices are int32 (render/shade/ewa.h): (|uv| + 32) x the texture size must stay below 2^30, which covers
  // the filter window and leaves a factor 2 for fp32 rounding.
  int64_t tex_size = 0;
  for (const Material& mat : b.materials) {
    if (!mat.texture.empty() || !mat.texture_linear.empty()) {
      tex_size = std::max<int64_t>({tex_size, mat.texture_width, mat.texture_height});
    }
    if (!mat.nir_texture.empty()) {
      tex_size = std::max<int64_t>({tex_size, mat.nir_texture_width, mat.nir_texture_height});
    }
  }
  if (tex_size > 0 && !m.uvs.empty()) {
    double uv_max = 0.0;
    for (float x : m.uvs) uv_max = std::max(uv_max, static_cast<double>(std::abs(x)));
    require((uv_max + 32.0) * static_cast<double>(tex_size) <= 0x1p30,
            name + ": |uv| x texture size must stay below 2^30 (int32 texel indices)");
  }
}

// The rules of the C1 feature "pieces" for one trajectory with pieces (bundle.py Trajectory._validate_pieces,
// blender.md §3.7): P >= 2; begin_0 = span begin, strictly increasing begins, the last before the span's end; t0_ns and
// h_ns are piece 0's; per piece 1 <= h <= 2^31 - 1, control rows that follow each other, cover rot and pos and hold
// >= K rows, a join in {-1, 0, 1} (piece 0: 0), a support [t0, t0 + (rows - K + 1) h] covering the piece's interval,
// and the pi/2 rule over its own consecutive controls (a join may turn by more: each piece is its own spline).
void validate_pieces(const Trajectory& t, size_t n, const std::string& name) {
  const std::vector<TrajectoryPiece>& ps = t.pieces;
  require(ps.size() >= 2, name + ": pieces needs at least 2 pieces (none for one)");
  require(ps.front().begin_ns == t.span_begin_ns && ps.back().begin_ns < t.span_end_ns,
          name + ": piece 0 must begin at the span's begin and the last piece before its end");
  require(t.t0_ns == ps.front().t0_ns && t.h_ns == ps.front().h_ns, name + ": t0_ns and h_ns must equal piece 0's");
  for (size_t p = 1; p < ps.size(); ++p) {  // first, as bundle.py: a piece's interval needs the next begin
    require(ps[p].begin_ns > ps[p - 1].begin_ns, name + " piece " + std::to_string(p) +
                                                     ": begin_ns must exceed the previous piece's");
  }
  int64_t next_row = 0;
  for (size_t p = 0; p < ps.size(); ++p) {
    const TrajectoryPiece& q = ps[p];
    const std::string at = name + " piece " + std::to_string(p);
    require(q.h_ns >= 1 && q.h_ns <= std::numeric_limits<int32_t>::max(), at + ": h_ns must be in [1, 2^31)");
    const int64_t rows = int64_t{q.ctrl_end} - q.ctrl_begin;
    require(q.ctrl_begin == next_row && q.ctrl_end <= static_cast<int64_t>(n) && rows >= t.order,
            at + ": control rows must follow the previous piece's, lie within rot and pos and hold at least K");
    next_row = q.ctrl_end;
    require(p == 0 ? q.join == 0 : (q.join >= -1 && q.join <= 1), at + ": join must be -1, 0 or 1 (piece 0: 0)");
    // The support's end in 128 bits: t0 + segments h may leave int64 for a far t0.
    const int64_t end = p + 1 < ps.size() ? ps[p + 1].begin_ns : t.span_end_ns;
    const i128 hi = static_cast<i128>(q.t0_ns) + static_cast<i128>(rows - t.order + 1) * q.h_ns;
    require(q.t0_ns <= q.begin_ns && q.begin_ns < end && static_cast<i128>(end) <= hi,
            at + ": its spline's support does not cover the piece's interval");
    for (int64_t c = q.ctrl_begin + 1; c < q.ctrl_end; ++c) {
      const auto i = static_cast<size_t>(c);
      double d = 0.0;
      for (size_t k = 0; k < 4; ++k) d += t.rot[4 * i + k] * t.rot[4 * (i - 1) + k];
      require(2.0 * std::acos(std::min(std::abs(d), 1.0)) <= std::numbers::pi / 2 + 1e-12,
              at + ": consecutive control rotations differ by more than pi/2");
    }
  }
  require(next_row == static_cast<int64_t>(n), name + ": the pieces' control rows must cover rot and pos");
}

}  // namespace

void validate(const SceneBundle& s) {
  require(s.t_end_ns >= s.t_start_ns, "t_end_ns < t_start_ns");
  for (double x : s.origin_world) require(std::isfinite(x), "origin_world is not finite");

  for (size_t i = 0; i < s.trajectories.size(); ++i) {
    const Trajectory& t = s.trajectories[i];
    const std::string name = "trajectory " + std::to_string(i);
    require(t.order == 4 || t.order == 6, name + ": K must be 4 or 6");
    require(t.h_ns > 0 && t.h_ns <= std::numeric_limits<int32_t>::max(), name + ": h_ns must be in [1, 2^31)");
    require(t.rot.size() % 4 == 0 && t.pos.size() % 3 == 0 && t.rot.size() / 4 == t.pos.size() / 3,
            name + ": rot (N x 4) and pos (N x 3) disagree");
    const size_t n = t.pos.size() / 3;
    require(n >= static_cast<size_t>(t.order), name + ": needs at least K control points");
    require(all_finite(t.rot) && all_finite(t.pos), name + ": control points are not finite");
    for (size_t c = 0; c < n; ++c) {
      const double q2 = t.rot[4 * c] * t.rot[4 * c] + t.rot[4 * c + 1] * t.rot[4 * c + 1] +
                        t.rot[4 * c + 2] * t.rot[4 * c + 2] + t.rot[4 * c + 3] * t.rot[4 * c + 3];
      require(q2 > 0.5 && q2 < 2.0, name + ": control rotation is not a unit quaternion");
    }
    const int64_t segments = static_cast<int64_t>(n) - t.order + 1;
    require(t.span_begin_ns <= t.span_end_ns, name + ": empty span");
    if (!t.pieces.empty()) {
      validate_pieces(t, n, name);  // each piece's support covers its part of the span
      continue;
    }
    require(t.span_begin_ns >= t.t0_ns && t.span_end_ns <= t.t0_ns + segments * t.h_ns,
            name + ": span exceeds the spline's knot range");
  }

  std::set<uint32_t> ids;
  for (size_t i = 0; i < s.bodies.size(); ++i) {
    const Body& b = s.bodies[i];
    const std::string name = "body " + std::to_string(i) + " ('" + b.name + "')";
    require(b.id >= 1 && b.id != 0xFFFFFFFFu, name + ": id must be >= 1 and not 0xFFFFFFFF");
    require(ids.insert(b.id).second, name + ": duplicate id " + std::to_string(b.id));
    // Ground truth stores semantics as u16 with 0xFFFF reserved as invalid (DESIGN §5.1).
    require(b.semantic <= 0xFFFEu, name + ": semantic must be <= 0xFFFE (u16; 0xFFFF is the invalid label)");
    if (b.trajectory) {
      require(*b.trajectory >= 0 && static_cast<size_t>(*b.trajectory) < s.trajectories.size(),
              name + ": trajectory index out of range");
    } else {
      check_rigid(b.T_world_body, name + " T_world_body");
    }
    validate_mesh(b, s.light_groups.size(), name);
    require(std::isfinite(b.normal_prefilter_m) && b.normal_prefilter_m >= 0.0,
            name + ": normal_prefilter_m must be finite and >= 0");
    if (b.scan) {
      require(!b.trajectory, name + ": a scan body is static (scan with a trajectory)");
      require(uses_baked_material(b), name + ": scan is set but none of the body's used materials is baked");
    }
  }

  require(!s.cameras.empty(), "no cameras");
  require(s.cameras.size() <= 256, "more than 256 cameras (the RNG counter keeps 8 camera bits)");
  for (size_t i = 0; i < s.cameras.size(); ++i) {
    const Camera& c = s.cameras[i];
    const std::string name = "camera " + std::to_string(i) + " ('" + c.name + "')";
    require(c.width > 0 && c.height > 0, name + ": invalid size");
    require(static_cast<uint64_t>(c.width) * static_cast<uint64_t>(c.height) <= std::numeric_limits<uint32_t>::max(),
            name + ": too many pixels");
    require(std::isfinite(c.fx) && std::isfinite(c.fy) && c.fx > 0 && c.fy > 0 && std::isfinite(c.cx) &&
                std::isfinite(c.cy),
            name + ": invalid intrinsics");
    require(all_finite(c.dist), name + ": distortion is not finite");
    if (c.model == CameraModelKind::kRadtan) {
      require(c.dist.size() == 4 || c.dist.size() == 5 || c.dist.empty(), name + ": radtan needs k1, k2, p1, p2[, k3]");
    } else {
      require(c.dist.size() == 4 || c.dist.empty(), name + ": kb needs k1..k4");
    }
    require(c.rig_trajectory >= 0 && static_cast<size_t>(c.rig_trajectory) < s.trajectories.size(),
            name + ": rig_trajectory out of range");
    check_rigid(c.T_rig_cam, name + " T_rig_cam");
  }

  require(!s.light_groups.empty(), "light_groups must contain at least the default group");
  require(s.light_groups.size() <= 8, "more than 8 light groups (DESIGN §7.5)");
  size_t shadow_lights = 0;
  for (size_t i = 0; i < s.lights.size(); ++i) {
    const Light& l = s.lights[i];
    const std::string name = "light " + std::to_string(i);
    require(l.group >= 0 && static_cast<size_t>(l.group) < s.light_groups.size(), name + ": group out of range");
    require(std::isfinite(l.intensity) && l.intensity >= 0.0f, name + ": intensity must be finite and >= 0");
    for (float c : l.color) require(std::isfinite(c) && c >= 0.0f, name + ": color must be finite and >= 0");
    for (double x : l.position) require(std::isfinite(x), name + ": position is not finite");
    for (double x : l.direction) require(std::isfinite(x), name + ": direction is not finite");
    if (l.body) require(*l.body >= 0 && static_cast<size_t>(*l.body) < s.bodies.size(), name + ": body index out of range");
    if (l.kind == BundleLightKind::kSpot || l.kind == BundleLightKind::kDirectional) {
      const double n2 = l.direction[0] * l.direction[0] + l.direction[1] * l.direction[1] +
                        l.direction[2] * l.direction[2];
      require(n2 > 1e-24, name + ": direction must be non-zero");
    }
    if (l.kind == BundleLightKind::kSpot) {
      // A hard cone edge would be a visibility jump (DESIGN §6.1 J3), which Phase 1 does not model.
      require(l.spot_inner_rad >= 0.0f && l.spot_inner_rad < l.spot_outer_rad &&
                  l.spot_outer_rad <= static_cast<float>(std::numbers::pi),
              name + ": spot needs 0 <= spot_inner_rad < spot_outer_rad <= pi");
      // The falloff divides by cos_inner - cos_outer in fp32 (shade/lights.h; as scene_data.cpp computes them).
      require(std::cos(l.spot_inner_rad) > std::cos(l.spot_outer_rad),
              name + ": spot cone too narrow: cos(spot_inner_rad) must exceed cos(spot_outer_rad) in fp32");
    }
    if (area_light(l.kind)) {
      // C1 feature "area_light" (bundle.py Light.validate; p2f.md §4.6).
      const auto norm = [](const std::array<double, 3>& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
      require(std::isfinite(l.size[0]) && std::isfinite(l.size[1]) && l.size[0] > 0.0 && l.size[1] > 0.0,
              name + ": a rect or disk light needs size [s_u, s_v] > 0 (metres)");
      for (double x : l.axis) require(std::isfinite(x), name + ": axis is not finite");
      require(std::abs(norm(l.direction) - 1.0) <= 1e-6 && std::abs(norm(l.axis) - 1.0) <= 1e-6,
              name + ": a rect or disk light needs unit direction and axis vectors");
      const double c = l.axis[0] * l.direction[0] + l.axis[1] * l.direction[1] + l.axis[2] * l.direction[2];
      require(std::abs(c) < 1e-6, name + ": axis must lie in the emitter's plane (|axis . direction| < 1e-6)");
    } else {
      require(l.size == std::array<double, 2>{0.0, 0.0} && l.axis == std::array<double, 3>{1.0, 0.0, 0.0},
              name + ": size and axis belong to rect and disk lights");
    }
    if (l.proxy) {
      require(l.casts_shadows || !positional(l.kind),
              name + ": a proxy point, spot or directional light must cast shadows (it shadows inserted objects on "
                     "the scan)");
      require(!l.body || !s.bodies[static_cast<size_t>(*l.body)].trajectory,
              name + ": a proxy stands for captured, static lighting (its body has a trajectory)");
    }
    if (l.casts_shadows && positional(l.kind)) ++shadow_lights;
  }
  require(shadow_lights <= 32, "more than 32 shadow-casting lights");
  const bool proxies = std::any_of(s.lights.begin(), s.lights.end(), [](const Light& l) { return l.proxy; });
  const bool scan = std::any_of(s.bodies.begin(), s.bodies.end(), [](const Body& b) { return b.scan; });
  require(!proxies || scan, "proxy lights need a scan body (mark the scanned room's body `scan: true`)");
}

}  // namespace balla::render
