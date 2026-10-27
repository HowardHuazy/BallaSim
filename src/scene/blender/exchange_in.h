// Typed views of a Blender exchange (`ballasim.blender_exchange/1`, docs/design/phase3/blender.md §3.4, §3.8): the
// RNA values and arrays the add-on writes, filled by python/ballasim/blender/exchange.py through the _scene binding
// after it has checked the file's version and keys and computed its digest. Nothing here maps a Blender value to C1;
// convert.h does.
//
// Filled from BL-2a on: version, digest, units, time, span, render, meshes, objects, cameras (object, c1_name,
// sensor_family, simulate), light object indices, light_groups, semantic_classes, collection_semantics and the
// add-on's report; from BL-2b on the materials' node graphs and the images (records and pixel files); from BL-3a on the
// lights (properties, node graphs, their data's and node trees' animation), the world (graph, colour, animation) and
// the cameras' properties, calibrations and data animation; from BL-4b on each object's data_animation (the
// exporter's animation dump). Animations stay records, with the F-curves' key tables beside them (anim_keys); the
// motion session (BL-9, motion/session.h) types them for the declared keys.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "scene/tree.h"

namespace balla::scene::blender {

// The exchange format this engine reads (major, highest minor) and the statics converter's version (bundle.meta,
// blender_cache_key).
inline constexpr std::array<int, 2> kExchangeFormat{1, 0};
inline constexpr int kConverterVersion = 1;

struct MeshIn {
  std::string id;
  enum class Normals { kPoint, kFace, kCorner } domain = Normals::kCorner;
  std::vector<float> co;              // V x 3, BU, object space        <- arrays mesh.<id>.co
  std::vector<int32_t> corner_vert;   // C                              <- mesh.<id>.corner_vert
  std::vector<int32_t> tri_corners;   // T x 3                          <- mesh.<id>.tri_corners
  std::vector<int16_t> tri_mat;       // T                              <- mesh.<id>.tri_mat
  std::vector<float> normals;         // C x 3 (kFace, kCorner) or V x 3 (kPoint)
  std::map<std::string, std::vector<float>> uvs;  // C x 2 each, v up    <- mesh.<id>.uv.<k>
  std::string active_render_uv;
  Tree asset;                         // {license, copyright, author} or null (PHASE3.md H4)
  size_t num_vertices() const { return co.size() / 3; }
  size_t num_triangles() const { return tri_corners.size() / 3; }
};

struct ObjectIn {
  std::string key;                    // the key array as canonical JSON text (unique per exchange)
  std::string name_full, type;        // type: Blender's object type (MESH, EMPTY, CAMERA, LIGHT, CURVES, ...)
  bool is_instance = false, is_editable = true;
  int32_t instancer = -1, instance_object = -1, parent = -1, mesh = -1, animation = -1;
  int32_t data_animation = -1;        // its data block's animation (a curve's eval_time, a light's energy; BL-4b)
  std::array<int64_t, 8> persistent_id{};
  std::string parent_type, parent_bone;
  std::vector<int32_t> material_slots;
  std::array<double, 16> matrix_world{};   // row-major, the float32 values Blender holds
  Tree visibility = Tree::dict(), deform = Tree::dict(), rigid_body, field, constraints = Tree::list();
  std::string lightgroup, instancer_ids = "per_instance";
  int32_t pass_index = 0;
  int64_t instance_id = 0, instance_block = 0;
  std::string semantic;                // the object's class name ("" = none)
  std::vector<std::string> collections;  // its collections, nearest first (for Collection.ballasim.semantic)
  Tree asset;
};

// A material's node graph as Blender holds it (blender.md §3.4): sockets by identifier (the Mix node repeats its
// names, Mix and Add Shader repeat "Shader"), raw values, and the live links (neither muted nor invalid).
struct SocketIn {
  std::string identifier, name, type;  // type: Blender's socket type (VALUE, RGBA, VECTOR, SHADER, ...)
  bool enabled = true, linked = false;
  std::array<double, 4> value{}, def{};  // default_value and the node's declared default; n components (0: none)
  int n = 0, n_def = 0;
};
struct NodeIn {
  std::string name, type, data_type, blend_type;  // type: the bl_idname (ShaderNodeBsdfPrincipled) or the RNA type
  Tree props = Tree::dict();                     // (BSDF_PRINCIPLED); props: the node's own properties
  std::vector<SocketIn> inputs, outputs;
};
struct LinkIn {
  std::string from_node, from_socket, to_node, to_socket;  // node names, socket identifiers
};
struct MaterialIn {
  std::string id, name, emission_sampling;
  bool has_graph = false;              // false: graph null (a material without nodes)
  std::vector<NodeIn> nodes;
  std::vector<LinkIn> links;
  Tree asset;
};
// An image record and the file of its pixels (textures/<sha256>.npy: H x W x C raw byte codes or float32
// scene-linear values, Blender's bottom-up row order). The converter reads a byte image's file only when a material
// uses it, one image at a time (convert_texture); the reader has checked that `path` is a file in textures/.
struct ImageIn {
  std::string id, colorspace, extension;
  std::string alpha_mode;              // Image.alpha_mode (STRAIGHT, PREMUL, CHANNEL_PACKED, NONE); "" not written
  std::string path;                    // "" for a float image or one written without pixels (refused, TEX-CS)
  bool is_float = false;
  int32_t w = -1, h = -1, c = -1;      // the record's width, height and channels (-1: not written)
  Tree asset;
};

// A light's or the world's node graph (§3.4: the materials' graph format; present = false for null).
struct GraphIn {
  bool present = false;
  std::vector<NodeIn> nodes;
  std::vector<LinkIn> links;
};

// calibration: the reader's record of cameras[].calibration (python/ballasim/blender/calibration.py: {source, name,
// model, distortion, width, height, intrinsics, coefficients, sensor, pixel_pitch_m, f_number, error}) or null.
struct CameraIn {
  int32_t object = -1;
  std::string c1_name;                 // as the add-on wrote it ("" = not written); checked against camera_names()
  Tree props = Tree::dict(), calibration;
  std::string sensor_family;
  bool simulate = true;
  int32_t animation = -1;              // the camera data's animation: animations[] index (-1: none)
};

// Keys on a light's or the world's node tree live on the tree's own animation data, not on the light's or the
// world's (probe m3b: `nodes["Emission"].inputs[1].default_value`), hence a second animation index for each.
struct LightIn {
  int32_t object = -1;
  std::string type;
  Tree props = Tree::dict(), light_linking;
  GraphIn graph;                       // node_graph (the light's node tree when it uses nodes)
  int32_t animation = -1;              // the light data's animation: animations[] index (-1: none)
  int32_t graph_animation = -1;        // node_graph_animation: its node tree's animation (-1: none)
};

struct WorldIn {
  bool present = false;                // world null: no world
  GraphIn graph;                       // null when World.use_nodes is off: Cycles renders `color` (probe m3b)
  Tree color;                          // World.color [r, g, b] (null: Blender's default 0.05 grey)
  std::string lightgroup;
  Tree cycles_visibility = Tree::dict(), asset;
  int32_t animation = -1;              // the world's own animation (World.color, use_nodes, ray visibility)
  int32_t graph_animation = -1;        // its node tree's animation
};

struct AddonItem {
  std::string code, object, message;
};

struct ExchangeIn {
  std::array<int32_t, 2> version{1, 0};
  std::string digest;                  // exchange digest (§3.4), computed by the reader
  Tree units = Tree::dict(), color, time = Tree::dict(), span = Tree::dict(), render = Tree::dict(),
       rigidbody_world, cache_files;
  std::array<double, 3> gravity{0.0, 0.0, -9.81};
  bool use_gravity = true;
  std::vector<MeshIn> meshes;
  std::vector<ObjectIn> objects;
  std::vector<CameraIn> cameras;
  std::vector<LightIn> lights;
  std::vector<MaterialIn> materials;
  std::vector<ImageIn> images;
  Tree animations = Tree::list();
  // The F-curves' key tables, anim.<key>.keys float64[n, 7] (§3.4 code tables), by the name the F-curve record gives
  // (`keyframes`); filled by the motion session's binding (the statics and the checks leave it empty), read by its
  // declared keys (motion/session.h key_scene).
  std::map<std::string, std::vector<double>> anim_keys;
  WorldIn world;
  Tree light_groups = Tree::list(), semantic_classes = Tree::dict(), collection_semantics = Tree::dict();
  std::vector<AddonItem> addon_report;
  double scale_length() const;         // units.scale_length (metres per BU), 1 when absent
};

}  // namespace balla::scene::blender
