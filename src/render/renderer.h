// BallaSim renderer (contract C2, docs/dev/CONTRACTS.md; DESIGN §4.3, §6.1, §6.9, §7, §9.1).
//
// A deterministic, pure function of (scene, camera, pixel, time): per pixel-sample it traces the
// N fixed sub-samples at the exact poses of that time, shades per cluster and returns the
// pixel-area-mean irradiance per light group, a visibility signature, and optionally the
// sub-sample instances and the centre-ray ground truth. Results do not depend on batching, thread
// count or request order. Requests are batched internally into launches of up to
// max_rays_per_launch rays, shadow rays included (>= 2 M is the efficient operating point, G6 t6;
// the default 2^24 takes 1-2.5 % less GPU time than 2^22; every launch stays below 50 ms, RENDER.md).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "render/bundle.h"
#include "render/pose.h"
#include "render/scene_data.h"
#include "render/shade/codes.h"
#include "render/types.h"

namespace balla::render {

enum class BackendKind { kEmbree, kOptix };

struct RendererOptions {
  BackendKind backend = BackendKind::kOptix;
  SceneSettings scene;
  int num_threads = 0;                        // CPU threads; 0 = hardware concurrency - 1
  uint64_t max_rays_per_launch = uint64_t{1} << 24;
  uint64_t pose_cache_bytes = uint64_t{256} << 20;  // the pose table is flushed beyond this
  bool census = false;  // render_into's P2-B B0 census (render/census.h; RenderStats::census)
  // execution.pv_outputs (P2-D D2's cost measurement): render_into's calls at N >= kPvMeasureN without a PvOut of their
  // own write P&V's outputs (render/shade/codes.h) to scratch that is dropped (RenderStats::pv_*); outputs unchanged.
  bool pv_outputs = false;
  // P2-F F11a (sensor.nir_fraction of each camera, p2f.md §4.9; no row of its own: the engine fills it from the cameras'
  // SensorSpecs): per camera in bundle order, the fraction of its photo-electrons from the near infrared per light group
  // ([] = 0). Equal values everywhere fold the NIR into the albedos, differing ones shade with two channels (SceneData).
  std::vector<std::vector<double>> nir_fraction;
};

// execution.pv_outputs writes in the calls of at least this N: P&V's eligibility (N >= the band's N_band; plan p2d.md
// §3.1).
inline constexpr uint32_t kPvMeasureN = 256;

struct RenderFlags {
  bool sub_instance = false;  // per sub-sample instance ids (jump localization)
  bool centre = false;        // centre-ray ground truth of each request's pixel
  bool sample_class = false;  // each pixel-sample's class word (types.h kClass*; render_into's sample_class)
  uint8_t pv = 0;  // P2-D D2's P&V outputs (render/shade/codes.h): 1 of the edge-class pixel-samples, 2 of every one
  // The dense channel shadow_factor (scans.md §3.7): RenderResult::shadow_factor.
  bool shadow_factor = false;
};

// Stage times of one render call, ms. trace and shade (incl. the result copy to pinned memory) split the
// GPU's busy time: spans of the two batch slots overlap, so each instant counts once, for the open span
// that ends first. host is host work outside launches and GPU waits: validation, result allocation,
// request building, pose keys, staging and result copies. On the Embree path trace and shade are host time.
// With RendererOptions::census, census holds the census of the render_into calls (render/census.h, kCensus*; the
// pixel-samples of failed calls included).
// probe_ms: the GPU time of probe_into's launches (trace and state kernel; P2-D D2), apart from trace and shade. With
// RendererOptions::pv_outputs, pv_entries and pv_bytes count the scratch outputs written (entries, their bytes).
struct RenderStats {
  double trace_ms = 0.0, shade_ms = 0.0, host_ms = 0.0, probe_ms = 0.0;
  bool census_on = false;
  std::array<uint64_t, kCensusWords> census{};
  bool pv_on = false;
  uint64_t pv_entries = 0, pv_bytes = 0;
};

struct RenderResult {
  size_t count = 0;
  int32_t num_groups = 0;
  uint32_t subsamples = 0;
  std::vector<float> irradiance;       // count x num_groups, linear, relative units
  std::vector<uint64_t> signature;     // count
  std::vector<uint32_t> sub_instance;  // count x subsamples when requested
  std::vector<GtSample> centre;        // count when requested
  std::vector<uint32_t> sample_class;  // count when requested
  // RenderFlags::shadow_factor, count x num_groups: per light group the ratio of irradiance to the irradiance with every
  // proxy factor set to 1 (scans.md §3.7; shade.h cluster_radiance's `unshadowed`), 1 where the latter is 0 and
  // everywhere in a scene without baked proxies.
  std::vector<float> shadow_factor;
  // RenderFlags::pv: per request whether it wrote, and its codes (pv_code_words each) and table (pv_table_words each),
  // in request order; zeros where it did not, and in table rows past its header's count.
  std::vector<uint8_t> pv_written;
  std::vector<uint32_t> pv_codes, pv_table;
  uint32_t pv_code_words = 0, pv_table_words = 0;
  RenderStats stats;
};

// Renderer::probe's states (render/shade/codes.h ProbeOut), one per probe, and the call's stage times.
struct ProbeResult {
  std::vector<ProbeOut> out;
  RenderStats stats;
};

struct DenseFrame {
  std::vector<float> irradiance;    // height x width x num_groups when requested
  std::vector<GtSample> gt;         // height x width when requested
  std::vector<float> shadow_factor; // height x width x num_groups when requested (RenderResult::shadow_factor)
};

// True if this build has the OptiX backend and a CUDA device with an OptiX-capable driver.
bool optix_available();

class Renderer {
 public:
  Renderer(SceneBundle bundle, RendererOptions options);
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  // Preprocesses the scene (geometry, mono mip textures, LUTs, seeds) and builds the acceleration
  // structures. Called implicitly by the first query.
  void build();
  bool built() const;

  // Base of the int32 time offsets; flushes the pose table. Requests far from the base (more than
  // 2^31 ns) re-base automatically, so this is a hint that keeps the pose cache warm.
  void set_window(int64_t base_ns);
  int64_t window() const;

  // One pixel-sample per (cam[i], pixel[i], t_ns[i]); pixel = y * width + x. A wrapper of
  // render_into's chunk loop (rows from the pose cache, in staged groups of requests) and, for the
  // centre flag, of gt_into; the stage timers of pending render_into calls, host time included, are
  // taken with its own.
  RenderResult render(std::span<const uint16_t> cam, std::span<const uint32_t> pixel,
                      std::span<const int64_t> t_ns, RenderFlags flags = {});

  // P2-D D2: the state of sub-sample k[i] of pixel[i] at t_ns[i] (ProbeOut, render/shade/probe.h; f of shadow light
  // light[i]): a wrapper of probe_into's chunk loop with rows from the pose cache, as render().
  ProbeResult probe(std::span<const uint16_t> cam, std::span<const uint32_t> pixel, std::span<const uint32_t> k,
                    std::span<const int64_t> t_ns, std::span<const uint32_t> light);

  // Centre-ray ground truth at distorted pixel positions (pixel centres at integers): gt_into per
  // camera.
  std::vector<GtSample> gt(std::span<const uint16_t> cam, std::span<const float> x,
                           std::span<const float> y, std::span<const int64_t> t_ns);

  // Every pixel of one camera at one time. shadow_factor: the dense channel of scans.md §3.7 (the same pass as the
  // irradiance, no second trace).
  DenseFrame dense(uint16_t cam, int64_t t_ns, bool irradiance, bool gt, bool shadow_factor = false);

  // The start diagnostic of scenes with baked proxies (scans.md §4.2): from the centre ray of every pixel of camera cam at
  // t_ns that hits a baked receiver, each proxy shadow light's ray is traced against the scan alone (geometry 0); the
  // result holds, per proxy light in bundle light order, the share of the traced rays the scan blocks (0 where none is
  // traced). A proxy inside scanned fixture geometry reads near 1. Empty in a scene without baked proxies.
  std::vector<float> proxy_visibility(uint16_t cam, int64_t t_ns);
  // The static bodies split from the scan geometry (scans.md §4.2): trace geometries of their own with trajectory -1;
  // 0 in a scene without baked proxies. Builds the renderer.
  int32_t split_geometries();

  // Held movers (ENGINE_CPP §0.1 I3b): the bodies `ids` (movers with triangles; other ids are ignored) are
  // traced as part of the static world at their poses at t_ns, until the next call: one world structure per
  // held set, and the trace geometries and pose rows shrink with it. The caller guarantees that they are
  // exactly still at every time it queries meanwhile (holds covering a window). Results then equal those of a
  // bundle with these bodies static at those poses, up to the BVH's choice between coincident surfaces (and
  // exactly when their ids follow every static body's), except that under accuracy.mover_ao proxy they keep casting
  // their proxies' AO from those poses (a static body casts none). The structure is rebuilt only when the set or a pose
  // changes, from the static world's arrays and the held movers' triangles appended after them (host work
  // O(held triangles)); an empty set restores the movers. Returns whether it rebuilt; a rebuild drops the
  // pose cache and ends the device window (begin_window follows before assign_pose_rows and render_into).
  // If the rebuild fails (e.g. out of device memory), the movers are restored and the exception propagates. Under
  // accuracy.mover_ao traced (P2-F F9b) with a mover that traces AO, a set holding a mover raises
  // std::invalid_argument: the ambient rays look at movers only (the engine holds none then).
  bool hold(std::span<const uint32_t> ids, int64_t t_ns);
  // Whether hold() merges movers into the world: false in a scene with baked proxies (scans.md §4.2: some material
  // baked, some light a proxy), whose held movers stay trace geometries of their own (a proxy shadow on the scan needs
  // its blocker apart from the scan's world); hold() then keeps the movers and returns false, and the engine skips it.
  bool holds_movers() const;
  // Device memory kept for held sets (the merged world's tail, its tables and structure), for the
  // caller's budget.
  size_t held_device_bytes() const;

  // Entry points of the engine (ENGINE_CPP §1.4, §1.7), on both backends. Pointers are in the
  // backend's memory space (device pointers on OptiX, host pointers on Embree); on OptiX work is queued
  // on `stream` (a cudaStream_t) and never waits for the host, except assign_pose_rows' one sync; on
  // Embree it is done when the call returns. render() and gt() are their wrappers.
  //
  // Window pose table: begin_window drops it and checks that [t_w, t_w + W] lies inside camera
  // cam's valid time; assign_pose_rows replaces each request's pose_slot, a slot of the clock, by
  // the row of that slot's exact poses, evaluating the rows of newly used slots.
  void begin_window(uint16_t cam, const WindowClock& clock, void* stream);
  void assign_pose_rows(SampleRequest* requests, size_t n, void* stream);
  // Renders requests whose pose_slot is a row of the window pose table: irradiance row out_index[i]
  // (num_groups floats) and signature out_index[i] receive request i (row i without out_index), and when
  // sample_class is not null its class word (shade.h sample_class: the engine's counters). One call renders one N:
  // every request's k_count is `subsamples` (0: options().scene.n_subsamples; at most kMaxSubsamples), the
  // engine's N classes (ENGINE_CPP P2-B B2) taking one call each; on the host backend a request with another
  // k_count raises std::invalid_argument (device requests are the caller's to keep uniform).
  // With pv (P2-D D2; at most kPvMaxLights shadow-casting lights, else std::invalid_argument), the P&V outputs of the
  // call's edge-class pixel-samples (render/shade/codes.h; PvOut::slot's rows as irradiance's), in executor memory.
  void render_into(const SampleRequest* requests, size_t n, const uint32_t* out_index, float* irradiance,
                   uint64_t* signature, void* stream, uint32_t* sample_class = nullptr, uint32_t subsamples = 0,
                   const PvOut* pv = nullptr);
  // P2-D D2: the state of single sub-samples (render/shade/probe.h): request i (k_count 1, sub-sample k_begin, its
  // pose_slot a row of the window pose table as render_into's; flags RequestKind::kProbe | shadow light <<
  // kProbeLightShift, the light of ProbeOut::f) into out[i]. Traced with the tracer's sample (1 + L rays each), no
  // shading; timed apart (RenderStats::probe_ms). On the host backend a k_count other than 1 or a light out of range
  // raises std::invalid_argument (device requests are the caller's to check).
  void probe_into(const SampleRequest* requests, size_t n, ProbeOut* out, void* stream);
  // Stage timers of the render_into calls since the last take (waits for their work).
  RenderStats take_render_stats();
  // Centre-ray ground truth of one camera: query i at its own exact poses; consecutive queries of
  // equal time share them. check_gt validates host copies of the queries as gt() does. gt_into checks
  // only the camera index, since its queries may be device memory: the caller passes queries that
  // check_gt accepts (gt() and Engine::gt do). A time outside the camera's valid time gives undefined
  // results on both backends; positions stay memory-safe, because the camera LUT clamps. shadow (when
  // not null) receives 2 num_shadow_lights() words per query, (state, blocker t) per shadow light
  // (render/shade/lights.h: shadow_states; the plan pass of ENGINE_CPP I8); normal (when not null) 3 floats per query,
  // the hit triangle's unit geometric normal in the camera frame, facing the camera (NaN on a miss; P2-D D1's
  // plan-pass normals, render/gt.h).
  void gt_into(uint16_t cam, const GtQuery* queries, size_t n, GtSample* out, void* stream,
               uint32_t* shadow = nullptr, float* normal = nullptr);
  void check_gt(uint16_t cam, const float* x, const float* y, const int64_t* t_ns, size_t n) const;

  BackendKind backend() const;
  const SceneBundle& bundle() const;
  const RendererOptions& options() const;
  int32_t num_groups() const;
  int32_t num_shadow_lights() const;  // the shadow-casting point, spot and directional lights (shadow bits)
  bool has_specular() const;          // any material with a specular lobe (C1 feature "pbr"; P2-F F6)
  // P2-F F11a: whether options().nir_fraction differs between (camera, light group) pairs (two-channel albedo textures,
  // SceneData::needs_two_channels), decided from the bundle and options without building the scene. std::invalid_argument
  // for a row of the wrong length or an f outside [0, 1], as the build raises.
  bool nir_two_channel() const;
  // Whether render_into accepts a PvOut (and render a RenderFlags::pv): false on the device under nir_two_channel(),
  // which has no P&V twin of the two-channel kernels (a PvOut then raises std::invalid_argument). The engine's P&V
  // eligibility reads this (ENGINE_CPP I8), as it reads num_shadow_lights() and the traced-AO rule.
  bool pv_supported() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::render
