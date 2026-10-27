// The statics of a Blender exchange in C1 terms (docs/design/phase3/blender.md §3.8, §4.1, §4.3-§4.6, §4.8): bodies
// with meshes split by (vertex, normal, uv) in their body frame, their materials and textures, instance ids and
// semantics, the cameras (names, intrinsics, calibrations), the lights, the world and the light groups, the 16 m rule
// with the optional crop, and the conversion report with each item's bound_kind and the matched-mode overrides (§1
// guarantee 3, §3.12, §6.1).
//
// Provenance (PHASE3.md §0 rule 4): the Blender facts used here are the manual's and the headless oracles' (DS d5,
// CV E4, RI E3/E4, RI2 gn instance keys, R2 c3): object keys, instance persistent ids, loop triangles as Cycles'
// corner triangles, corner and point normal domains, UVs with v up; camera.cpp and lights.cpp name theirs. No Blender
// source was used.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "render/bundle.h"
#include "scene/blender/exchange_in.h"
#include "scene/geometry.h"
#include "scene/tree.h"

namespace balla::scene::blender {

enum class Policy { kStrict, kLenient };
enum class BoundKind { kMeasured, kSampled, kNone };

// severity: refuse | warn | approx | exact | info | error (§6.1). Under strict, every item with bound_kind none that
// is not info is a refusal; the warn items carry their bound.
struct ReportItem {
  std::string code, severity, object, message;
  double bound = 0.0;
  BoundKind bound_kind = BoundKind::kNone;
};

// §3.11's conversion rows, as C++ options until BL-3b binds them (PHASE3.md §0 rule 1).
struct ConvertOptions {
  Policy policy = Policy::kStrict;
  std::optional<double> lm_per_w;
  bool world_as_ambient = false;
  std::optional<double> crop_radius_m;   // (0, 16]; null: refuse static geometry beyond 16 m (GEO-16M)
  int64_t max_triangles = 20'000'000;    // after instancing (GEO-BUDGET)
  double calib_tol_px = 0.01, fisheye_fit_tol_px = 0.05, rigidity_tol_m = 1e-5;
};
Tree options_record(const ConvertOptions& o);   // every option, in declaration order (blender_cache_key, meta)

// ---- materials and textures (§4.5; materials.cpp) ----

// One exchange material in C1 terms (§4.5): base_color = constant x texture (at most one image), a constant emissive
// radiance (k = lm_per_w applied), and the map from Blender's (u, v) on the material's UV map to C1's (u', v'): its
// Mapping nodes, the v flip and MIRROR's halving. `items` are its report items (object: the material's name).
struct MaterialOut {
  render::Material mat;                  // base_color and emissive; the texture is `image` (Statics::textures)
  int32_t image = -1;                    // images[] index of its texture (-1: none)
  bool mirror = false;                   // MIRROR: the C1 texture is the 2W x 2H tile
  bool unit_uv = false;                  // EXTEND or CLIP: exact only while the coordinates stay in [0, 1]
  std::string uv_map;                    // the UV map it names ("": the mesh's active render map)
  std::array<double, 6> uv{1.0, 0.0, 0.0, 0.0, -1.0, 1.0};   // u' = uv[0] u + uv[1] v + uv[2], v' = uv[3] u + ...
  std::vector<std::string> uv_maps;      // every map its Image Textures read ("": the default; GEO-UV2 per mesh)
  std::vector<ReportItem> items;
  bool emitter() const { return mat.emissive[0] > 0.0f || mat.emissive[1] > 0.0f || mat.emissive[2] > 0.0f; }
};
MaterialOut convert_material(const MaterialIn& m, const std::vector<ImageIn>& images, const ConvertOptions& o);

// A C1 texture: sRGB codes, rows top-down, RGB (alpha dropped), the MIRROR tile when `mirror`.
struct TextureOut {
  int32_t image = -1;
  bool mirror = false;
  int32_t width = 0, height = 0;
  std::vector<uint8_t> rgb;              // height x width x 3
};
// Reads the image's file (`im.path`, a uint8 .npy in C or Fortran order, released before it returns). Byte sRGB codes
// pass bitwise; byte Linear Rec.709 and Non-Color are re-encoded to the nearest sRGB code (TEX-CS warn, <= 2.63 %
// relative, CV E6); float images (until C1's texture_linear), other colour spaces and images written without pixels
// (file null: a missing file, or a tiled, sequence or movie image) are refused (TEX-CS). Alpha (2 or
// 4 channels, probe m2): the Color output of an sRGB or Linear image is the codes associated in the byte domain,
// floor(c a / 255), for alpha_mode STRAIGHT, and the raw codes for CHANNEL_PACKED, NONE and every Non-Color image;
// texels with alpha < 1 under another or an unwritten alpha_mode are MAT-ALPHA (refused under strict; associated as
// STRAIGHT, Blender's default, under lenient). MIRROR becomes [[T, T mirrored in u], [T mirrored in v, T mirrored in
// both]] with halved uvs. BRG-PROTO (std::invalid_argument) for a file that is not such a .npy or does not match the
// record's width, height and channels.
struct TextureResult {
  std::optional<TextureOut> tex;         // empty when refused
  std::vector<ReportItem> items;         // object: the image's id
};
TextureResult convert_texture(const ImageIn& im, int32_t index, bool mirror, Policy policy);

// The UV source of a material slot for split_mesh: its map ("" the default one) and its affine map to C1's uv; an
// untextured slot keeps the default (u, 1 - v) of the default map.
struct SlotUv {
  bool textured = false, unit = false;
  std::string map;
  std::array<double, 6> A{1.0, 0.0, 0.0, 0.0, -1.0, 1.0};
};

struct BodyStatic {
  int32_t object = -1;                   // objects[] index (the instancer for an `inherit` body)
  uint32_t id = 0, semantic = 0;
  std::string name;
  geo::Mesh mesh;                        // body frame, metres: S and scale_length applied, recentred at c
  std::vector<render::Material> materials;  // per body material, >= 1: §4.5's conversion, Lambert 0.8 for no material
  std::vector<int32_t> texture_of;       // per body material: its Statics::textures index (-1: none)
  std::vector<int32_t> material_of;      // per body material: the exchange materials[] index (-1: none); the
                                         // parts' slot lists, each distinct list once (<= kMaxBodyMaterials)
  std::array<double, 9> S{};             // polar scale factor of matrix_world (BU), row-major
  std::array<double, 3> c{};             // recentring offset in the scaled frame (metres)
  double rho = 0.0;                      // mesh radius about the body origin (metres)
  geo::M4 T_world_export{};              // world (Blender origin, metres) <- body at the export frame
  std::optional<geo::M4> T_world_static; // the same for static candidates (§4.7.1; the session confirms them)
  enum class Role { kBody, kCarrier } role = Role::kBody;
  std::vector<int32_t> parts;            // the objects whose meshes it holds (several for `inherit`)
};

// An exchanged ID that the bodies use and that carries an `asset` record (PHASE3.md H4): its kind (object, mesh,
// material, image), exchange index and name, the record {license, copyright, author} as written, and the bodies.
struct AssetUse {
  std::string kind, name;
  int32_t index = -1;
  Tree record;
  std::vector<std::string> bodies;
};

// ---- cameras (§4.3; camera.cpp) ----

// Blender's render size: W = (resolution_x * resolution_percentage) // 100 and likewise H (integer division, DS d6),
// the pixel aspect and the border flag.
struct RenderSize {
  int32_t W = 0, H = 0;
  double pax = 1.0, pay = 1.0;
  bool border = false;
};
RenderSize render_size(const ExchangeIn& ex);

// A camera in C1 terms (§3.8). `cam` is the C1 camera: the calibration's model when the camera has one (it is
// authoritative), else Blender's own (a pinhole as radtan with zero distortion, or kb for the fisheyes).
struct CameraStatic {
  int32_t index = -1, object = -1;        // cameras[] and objects[] indices
  render::Camera cam;                     // name: the C1 name; rig_trajectory: set when the bundle is assembled
  std::string sensor;                     // the preset's sensor, else the camera's sensor family ("": none)
  double pixel_pitch_m = 0.0, f_number = 2.0;
  std::array<double, 4> blender{};        // Blender's own pinhole part (fx, fy, cx, cy) at the render size
  int32_t rig_object = -1;                // the camera object (§4.7.9: BL-8 shares rigs of rigidly coupled cameras)
  geo::M4 T_rig_cam{};                    // diag(1, -1, -1, 1): Blender's camera frame (-z forward) to C1's (§4.1)
  geo::M4 T_world_export{};               // world (Blender origin, metres) <- rig at the export frame (polar factor)
  std::optional<geo::M4> T_world_static;  // the same for a static candidate
  double theta_max = 0.0;                 // the largest field angle over the image (rad; pixel centres +- 0.5 px)
  double clip_start_m = 0.0;
  bool calibrated = false, simulate = true;
};

// The Blender settings that reproduce a calibration's pinhole part (§4.3 inverse, `ballasim blender camera-settings`):
// sensor fit HORIZONTAL, lens = fx sw / W, pixel aspect (1, fx / fy) or (fy / fx, 1), shift_x = (W / 2 - 0.5 - cx) /
// W, shift_y = (cy + 0.5 - H / 2) (fx / fy) / W, resolution W x H at 100 %.
struct BlenderCamera {
  std::string sensor_fit = "HORIZONTAL";
  double lens_mm = 0.0, sensor_width_mm = 36.0, shift_x = 0.0, shift_y = 0.0;
  int32_t res_x = 0, res_y = 0;
  double pixel_aspect_x = 1.0, pixel_aspect_y = 1.0;
};
BlenderCamera camera_settings(const render::Camera& calib, double sensor_width_mm = 36.0);

// A calibration record (CameraIn::calibration) as a C1 camera, with a preset's sensor, pixel pitch and f-number.
// std::invalid_argument "CAM-MODEL: ..." or "CAM-CALIB: ..." for a record it cannot use.
struct Calibration {
  render::Camera cam;
  std::string sensor;
  std::optional<double> pixel_pitch_m, f_number;
};
Calibration calibration_camera(const Tree& record);

// FISHEYE_LENS_POLYNOMIAL as Kannala-Brandt: theta(r) = -(k0 + k1 r + ... + k4 r^4) (r in mm on the sensor; probe m3)
// sampled at 2000 radii in [0, r_max_mm], the samples with 0 <= theta <= theta_cap kept, and r = f (theta + k1 theta^3
// + ... + k4 theta^9) fitted by least squares (Householder QR). std::invalid_argument when theta(r) is not increasing
// there or fewer than 16 samples remain.
struct PolyFit {
  double f_mm = 0.0;
  std::array<double, 4> k{};
  double max_err_mm = 0.0, theta_max = 0.0;
  int samples = 0;
};
PolyFit fit_lens_polynomial(const std::array<double, 5>& k, double r_max_mm, double theta_cap);

// Blender's own camera at the render size (§4.3): PERSP as a pinhole (radtan, zero distortion), FISHEYE_EQUISOLID
// and FISHEYE_LENS_POLYNOMIAL as kb, with Blender's pixel pitch (film over pixels along the fitted axis) and its
// notes (CAM-FOV info, the polynomial fit's approx item); `refusal` (CAM-MODEL) for anything else. Report objects are
// left empty. BRG-PROTO (std::invalid_argument) for malformed values.
struct BlenderModel {
  render::Camera cam;
  double pitch_m = 0.0;
  std::vector<ReportItem> notes;
  std::optional<ReportItem> refusal;
};
BlenderModel blender_camera(const Tree& props, const RenderSize& rs, double fisheye_fit_tol_px,
                            const std::string& where);

// ---- lights and the world (§4.4; lights.cpp) ----

struct LightStatic {
  int32_t object = -1;                    // objects[] index (-1: the world's background or ambient)
  std::string name;
  render::Light light;                    // body: the Statics::bodies index of its carrier, with position and direction
                                          // in that body's frame; none: world frame (Blender origin, metres)
};
struct LightGroupOut {
  std::string name;
  Tree waveform;
};

// How a light moves (§4.4): kWorld when its object is a static candidate (no carrier), kParent when it moves rigidly
// with its parent object (it has no motion of its own and an OBJECT parent), kOwn otherwise (its own animation,
// constraints or rigid body, or a bone or vertex parent): a meshless carrier of the light object itself.
enum class LightMotion { kWorld, kParent, kOwn };
LightMotion light_motion(const ExchangeIn& ex, int32_t object, const std::vector<bool>& is_static);

struct Statics {
  std::vector<BodyStatic> bodies;
  std::vector<TextureOut> textures;      // shared by the bodies' materials (one per image and MIRROR flag)
  std::vector<std::string> camera_names; // C1 names of exchange cameras[], in order (§3.4)
  std::vector<CameraStatic> cameras;     // the converted cameras (a refused camera is left out)
  std::vector<LightStatic> lights;       // lights in exchange order, then the world's background and ambient
  std::vector<LightGroupOut> groups;     // C1 light groups: "default" (constant) first, then the used ones in order
  std::vector<AssetUse> assets;
  std::vector<ReportItem> report;
  Tree matched = Tree::list();           // §3.12 matched_overrides (render, light, world, camera)
  int64_t triangles = 0;                 // after instancing, before the split (GEO-BUDGET's count)
  bool refused() const;
};

// The cameras, the lights, the world and the light groups of an exchange whose bodies are converted (convert_statics
// calls them last); `is_static` per objects[] index.
void convert_cameras(const ExchangeIn& ex, const ConvertOptions& o, const std::vector<bool>& is_static, Statics& st);
void convert_lights(const ExchangeIn& ex, const ConvertOptions& o, const std::vector<bool>& is_static, Statics& st);

// §3.4 camera names: [A-Za-z0-9_-] kept, every other character (one UTF-8 code point) '_', empty -> "camera", unique
// in order with _2, _3, ... against a taken set that starts as {"default"} (the resolver's all-cameras block).
std::vector<std::string> camera_names(const std::vector<std::string>& name_fulls);

// A = R S with R a rotation; for det A < 0, R = polar(-A) and S = R^T A (a mirror, §4.1). Row-major 3 x 3.
struct Polar {
  std::array<double, 9> R, S;
  bool mirror;
};
Polar polar(const std::array<double, 9>& a);

// A mesh split by (vertex, normal, uv) (§4.6), mapped by x -> M co + o (M: 3 x 3 row-major, metres per BU), with
// normals by M^-T (unit or 0), uv (u, 1 - v), the winding reversed when det M < 0, and degenerate and duplicate
// triangles dropped as the asset importers drop them (geo::kept_faces, rel_eps 1e-7). Positions stay float64 until
// the body is recentred. With `slot_uv` (per material slot), a textured slot's corners take their uv from the slot's
// map through its affine A (MAT-BAKE when the map is missing, or when EXTEND/CLIP coordinates leave [0, 1]).
// Throws GeometryError for a mesh it cannot convert.
struct GeometryError : std::invalid_argument {
  std::string code;
  GeometryError(std::string c, const std::string& message) : std::invalid_argument(message), code(std::move(c)) {}
};
struct SplitMesh {
  std::vector<double> positions;         // V x 3
  std::vector<uint32_t> indices;         // T x 3
  std::vector<float> normals, uvs;       // V x 3, V x 2 (empty without a UV map)
  std::vector<uint16_t> material_ids;    // T (slot index)
  int64_t degenerate = 0, duplicate = 0;
};
SplitMesh split_mesh(const MeshIn& mesh, const std::array<double, 9>& M, const std::array<double, 3>& o, int slots,
                     std::span<const SlotUv> slot_uv = {});
// Concatenates parts (part j's material ids offset by bases[j], its first material in the body), recentres at the
// AABB centre c, stores float32 and gives the radius rho about the new origin (material ids only when the body has
// more than one material: total > 1). Parts are released as they are copied.
geo::Mesh finish_mesh(std::vector<SplitMesh>& parts, const std::vector<int>& bases, int total, std::array<double, 3>* c,
                      double* rho);
// The most materials a body can index (material_ids are uint16).
inline constexpr int kMaxBodyMaterials = 65536;

Statics convert_statics(const ExchangeIn& ex, const ConvertOptions& o);

// origin_world (§4.1): the AABB centre of the given motion samples (moving bodies, carriers, cameras; world metres),
// else of the static geometry; {0, 0, 0} for an empty scene.
std::array<double, 3> origin_world(const Statics& s, std::span<const std::array<double, 3>> samples);
// The samples of a provisional origin before the motion session (`blender check`, the static assembly): the
// export-frame positions of the simulated cameras (the bundle's; one that is not simulated is left out of it) and of
// every body that is not a static candidate.
std::vector<std::array<double, 3>> provisional_samples(const ExchangeIn& ex, const Statics& s);
// The exact static reach max_i |R x_i + t - origin| (§3.7).
double static_reach(const BodyStatic& b, const std::array<double, 3>& origin);
// §4.6: with crop_radius_m, crops every static body to the ball about origin (CROP info with the dropped area; a
// body cropped to nothing is removed); without it, GEO-16M refuses a static body beyond 16 m.
void place_statics(Statics& s, const std::array<double, 3>& origin, const ConvertOptions& o);
struct Cropped {
  geo::Mesh mesh;
  double dropped_area_m2 = 0.0;
};
// Keeps the triangles whose three vertices lie within r of the origin (T: origin <- body), drops those whose AABB
// misses the ball, and splits the others at their edge midpoints (normals and uvs interpolated) until the longest
// edge is below r/64.
Cropped crop_mesh(const geo::Mesh& mesh, const geo::M4& T, double r);

// The modifier types whose result depends on nothing but the object's own data and settings while they read no
// other object (BL-4b probe p5, 4.2.1 = 4.5.3): ARRAY, BEVEL, BOOLEAN, DATA_TRANSFER, DECIMATE, EDGE_SPLIT, MASK,
// MIRROR, MULTIRES, NORMAL_EDIT, REMESH, SCREW, SKIN, SOLIDIFY, SUBSURF, TRIANGULATE, UV_PROJECT, UV_WARP,
// VOLUME_TO_MESH, WEIGHTED_NORMAL, WELD, WIREFRAME. Every other type can change the geometry over the span without a
// key: deform types, and constructive ones that follow the frame (Build, Explode, Fluid, Dynamic Paint, Ocean, a Mesh
// Sequence Cache, a particle instance, ...), or a type this list does not know.
bool steady_modifier(const std::string& type);
// Whether an object's evaluated geometry can change over the span without a key on its transform (§4.6): a deform
// modifier (is_deform_modified), animated shape keys, geometry nodes, a NODES, cloth or soft-body modifier, a modifier
// shown for render that is not steady_modifier or that reads other objects (its record's `objects`, which may move),
// or a keyed or driven `modifiers[...]` path (in the object's animation or the strips its NLA plays). Its mesh is not
// looked at: an instancer hidden from render (mesh null) still moves its instances when it deforms.
bool may_change_shape(const ExchangeIn& ex, const ObjectIn& ob);
// Rigidity probes (§4.6): the objects with a mesh for which may_change_shape holds, and GEO-DEFORM for probed
// object-space vertices that move more than tol or change the topology.
std::vector<int32_t> rigidity_probe_objects(const ExchangeIn& ex);
std::optional<ReportItem> check_rigidity(const std::string& object, std::span<const float> co0,
                                         std::span<const std::vector<float>> probes, double tol_bu);

// The assets record of an exchange (§3.4 `asset`; PHASE3.md H4) as a list for blender check and bundle.meta.
Tree assets_doc(const std::vector<AssetUse>& assets);

// conversion_report.json (§3.12) with the statics' matched_overrides (the frames area adds its own).
Tree report_doc(const std::vector<ReportItem>& items, Policy policy, const Tree& matched = Tree::list());
const char* bound_kind_name(BoundKind k);

}  // namespace balla::scene::blender
