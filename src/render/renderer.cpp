#include "render/renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/time/time.h"
#include "render/embree/embree_tracer.h"
#include "render/executor.h"
#include "render/pose_table.h"
#include "render/thread_pool.h"
#include "render/tracer.h"
#if BALLA_RENDER_HAVE_OPTIX
#include "render/device_backend.h"
#endif

namespace balla::render {

bool optix_available() {
#if BALLA_RENDER_HAVE_OPTIX
  return optix_device_available();
#else
  return false;
#endif
}

namespace {

using Clock = std::chrono::steady_clock;

// render() stages and synchronizes at most this many launches of requests at a time.
constexpr size_t kGroupLaunches = 8;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// A stage's span of one batch on the executor clock (stage 0 trace, 1 shade, 2 probe launches).
struct Span {
  double begin, end;
  int stage;
};

// GPU time per stage. A batch queued on one slot waits for the other slot's work, so spans overlap; each
// instant is credited once, to the open span that ends first (that work ran, the later ones were queued).
void credit_spans(std::vector<Span>& spans, double* stage_ms) {
  std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.begin < b.begin; });
  std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> open;  // (end, stage)
  double t = std::numeric_limits<double>::lowest();
  for (size_t i = 0; i < spans.size() || !open.empty();) {
    if (open.empty()) t = std::max(t, spans[i].begin);
    for (; i < spans.size() && spans[i].begin <= t; ++i) open.emplace(spans[i].end, spans[i].stage);
    while (!open.empty() && open.top().first <= t) open.pop();
    if (open.empty()) continue;
    const double next = i < spans.size() ? std::min(open.top().first, spans[i].begin) : open.top().first;
    stage_ms[open.top().second] += next - t;
    t = next;
  }
}

constexpr const char* kPvNirRefusal =
    "render: P&V outputs are not built with two-channel near-infrared albedo (cameras whose sensor.nir_fraction differs "
    "between light groups) on the device";

}  // namespace

struct Renderer::Impl {
  Impl(SceneBundle b, RendererOptions o)
      : bundle(std::move(b)),
        options(o),
        pool(o.num_threads > 0 ? o.num_threads
                               : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1)) {
    validate(bundle);
    // P2-F's C1 features (p2f.md §4.11): the specular lobe renders from F6 (shade.h lobe_radiance), area lights from
    // F8a-2. The "nir" fields render since F11a (SceneData: the NIR folded in at level 0, or two channels mixed per
    // camera and light group, by sensor.nir_fraction). Tile atlases (C1 texture_tile) render since S5a.
    for (const Light& l : bundle.lights) {
      if (area_light(l.kind)) {
        throw std::invalid_argument("render: rect and disk lights (C1 feature 'area_light') are not rendered by this "
                                    "build yet (P2-F F8a-2)");
      }
    }
    // SceneView::baked_proxies (SceneData's rule), known before the build: the engine asks holds_movers() first.
    const auto baked = [](const Body& body) {
      return std::any_of(body.materials.begin(), body.materials.end(),
                         [](const Material& m) { return m.shading == Shading::kBaked; });
    };
    baked_proxies = std::any_of(bundle.bodies.begin(), bundle.bodies.end(), baked) &&
                    std::any_of(bundle.lights.begin(), bundle.lights.end(), [](const Light& l) { return l.proxy; });
    if (options.max_rays_per_launch == 0) throw std::invalid_argument("render: max_rays_per_launch must be > 0");
    if (options.backend == BackendKind::kOptix && !optix_available()) {
      throw std::runtime_error(
          "render: the OptiX backend is unavailable (built without OptiX, no CUDA device, or no OptiX driver)");
    }
    // Times a camera can be queried at: the intersection of the spans of every trajectory its pose
    // slots evaluate (rig, movers, lights carried by movers). Checked before any work is queued.
    for (const Camera& c : bundle.cameras) {
      std::vector<int32_t> used{c.rig_trajectory};
      for (const Body& body : bundle.bodies) {
        if (body.trajectory && !body.mesh.indices.empty()) used.push_back(*body.trajectory);
      }
      for (const Light& l : bundle.lights) {
        if (!positional(l.kind) || !l.body) continue;
        const Body& carrier = bundle.bodies[static_cast<size_t>(*l.body)];
        if (carrier.trajectory) used.push_back(*carrier.trajectory);
      }
      std::pair<int64_t, int64_t> span{INT64_MIN, INT64_MAX};
      for (int32_t t : used) {
        span.first = std::max(span.first, bundle.trajectories[static_cast<size_t>(t)].span_begin_ns);
        span.second = std::min(span.second, bundle.trajectories[static_cast<size_t>(t)].span_end_ns);
      }
      valid_time.push_back(span);
    }
  }

  SceneBundle bundle;
  RendererOptions options;
  ThreadPool pool;
  std::vector<std::pair<int64_t, int64_t>> valid_time;  // per camera
  bool baked_proxies = false;                           // SceneView::baked_proxies (holds_movers)
  std::unique_ptr<SceneData> scene;
  std::unique_ptr<Executor> executor;
  std::unique_ptr<Tracer> tracer;
  std::unique_ptr<PoseTable> poses;  // render()'s pose cache
  int64_t window_ns = 0;
  // The held set in use (hold): its geometries (ascending) and their poses.
  std::vector<int32_t> held;
  std::vector<RigidPose> held_poses;
  // Stage-timer marks of chunks not yet taken: (begin, end, stage) mark indices.
  struct PendingSpan {
    uint32_t begin, end;
    int stage;
  };
  std::vector<PendingSpan> pending;
  double pending_host_ms = 0.0;
  std::array<uint64_t, kCensusWords> census_taken{};  // the executor's census totals at the last take
  std::array<uint64_t, 2> pv_taken{};                 // the executor's pv_scratch totals at the last take
  bool timing = false;

  // All or nothing: a failed build (e.g. out of memory) leaves the renderer unbuilt, and the next call
  // builds again. Moving a unique_ptr keeps its pointee, so the views taken from sc and ex stay valid.
  void build() {
    auto sc = std::make_unique<SceneData>(bundle, options.scene, options.nir_fraction);
    // P2-F F6 x F11a: lobe_radiance shades a cluster with one albedo (its Fresnel F0 and diffuse weight), not one per
    // light group, so a scene with a lobe and two-channel NIR would silently ignore the NIR channel: refused.
    if (sc->nir_two_channel() && sc->has_specular()) {
      throw std::invalid_argument("render: specular materials (C1 feature 'pbr', specular or metallic > 0) are not built "
                                  "with two-channel near-infrared albedo (cameras whose sensor.nir_fraction differs "
                                  "between light groups)");
    }
    // S3a-2 x P2-F F6, F9, F11a: the device's baked-proxy kernels (kBaked) are non-V1 kernels (no kV1 x kBaked twin: a
    // third twin axis would triple the build), so a scene with baked proxies and a lobe, mover AO or two-channel NIR
    // is refused on both backends, which then agree; U3 of scans.md §4.2 takes them with the fill's AO.
    if (sc->baked_proxies() &&
        (sc->has_specular() || sc->nir_two_channel() || sc->traces_ao() || !sc->ao_bodies().empty())) {
      throw std::invalid_argument("render: baked proxies (a scan with proxy lights) are not built with specular materials "
                                  "(C1 feature 'pbr'), accuracy.mover_ao proxy or traced, or two-channel near-infrared "
                                  "albedo (cameras whose sensor.nir_fraction differs between light groups)");
    }
    // P2-F F11a: the device's P&V kernels (kPv) have no two-channel NIR variant (device_executor.cu launch_shade).
    if (options.backend == BackendKind::kOptix && sc->nir_two_channel() && options.pv_outputs) {
      throw std::invalid_argument("render: execution.pv_outputs is not built with two-channel near-infrared albedo "
                                  "(cameras whose sensor.nir_fraction differs between light groups) on the device");
    }
    std::unique_ptr<Executor> ex;
    std::unique_ptr<Tracer> tr;
    if (options.backend == BackendKind::kEmbree) {
      ex = std::make_unique<HostExecutor>(pool);
      tr = std::make_unique<EmbreeTracer>(pool);
    } else {
#if BALLA_RENDER_HAVE_OPTIX
      DeviceBackend d = make_optix_backend();
      ex = std::move(d.executor);
      tr = std::move(d.tracer);
#endif
    }
    ex->upload_scene(*sc);
    tr->build(*sc, ex->scene_view());
    const auto geoms = static_cast<int32_t>(sc->geometries().size());
    const auto lights = static_cast<int32_t>(sc->lights().size());
    // The pose cache holds pose_cache_bytes of rows, and at least one full render launch.
    const uint64_t slot_bytes = static_cast<uint64_t>(geoms) * sizeof(PoseRow) +
                                static_cast<uint64_t>(lights) * sizeof(LightRow);
    const uint64_t launch_requests = std::max<uint64_t>(1, options.max_rays_per_launch / options.scene.n_subsamples);
    auto ps = std::make_unique<PoseTable>(geoms, lights,
                                          static_cast<size_t>(std::max(launch_requests, options.pose_cache_bytes / slot_bytes)));
    ps->reset(window_ns);
    scene = std::move(sc);
    executor = std::move(ex);
    tracer = std::move(tr);
    poses = std::move(ps);
  }

  void ensure_built() {
    if (!scene) build();
  }

  // P2-F F11a: whether this renderer shades on the device with two-channel NIR (no P&V kernel variant of it).
  bool device_two_channel() const { return tracer->memory_space() != MemorySpace::kHost && scene->nir_two_channel(); }

  void check_camera_time(uint16_t cam, int64_t t_ns) const {
    if (cam >= bundle.cameras.size()) {
      throw std::invalid_argument("render: camera index " + std::to_string(cam) + " out of range");
    }
    const auto [lo, hi] = valid_time[cam];
    if (t_ns < lo || t_ns > hi) {
      throw std::invalid_argument("render: time " + std::to_string(t_ns) + " ns lies outside [" + std::to_string(lo) +
                                  ", " + std::to_string(hi) + "] ns, where camera " + std::to_string(cam) +
                                  " and every trajectory it sees are defined");
    }
  }

  void check_pixel(uint16_t cam, uint32_t pixel) const {
    const Camera& c = bundle.cameras[cam];
    if (pixel >= static_cast<uint32_t>(c.width) * static_cast<uint32_t>(c.height)) {
      throw std::invalid_argument("render: pixel " + std::to_string(pixel) + " out of range for camera " + std::to_string(cam));
    }
  }

  // Slot of one request in the current chunk. Returns false, closing the chunk, when the request
  // needs a reset of the pose table: its time is no int32 offset from the base, or the table is
  // full. The first request of a chunk performs that reset instead (re-basing to its window).
  bool take_slot(uint16_t cam, int64_t t_ns, bool chunk_empty, uint32_t* slot, int32_t* offset) {
    if (!time::offset_fits(t_ns, poses->base())) {
      if (!chunk_empty) return false;
      poses->reset(time::window_start(time::window_index(t_ns)));
    }
    if (!poses->slot(cam, t_ns, slot)) {
      if (!chunk_empty) return false;
      poses->reset(poses->base());
      poses->slot(cam, t_ns, slot);  // cannot fail on an empty table
    }
    *offset = time::to_offset(t_ns, poses->base());
    return true;
  }

  // Requests per launch at n_sub sub-samples per request: a launch traces each sub-sample ray plus its shadow rays;
  // the budget counts them all, which also keeps every kernel far below the display watchdog (DESIGN §4.2: <= 50 ms).
  size_t requests_per_launch(uint32_t n_sub) const {
    const uint64_t rays = static_cast<uint64_t>(n_sub) * (1 + static_cast<uint64_t>(scene->num_shadow_lights()) +
                                                          (scene->traces_ao() ? 1 : 0));  // P2-F F9b's ambient ray
    return static_cast<size_t>(std::max<uint64_t>(1, options.max_rays_per_launch / rays));
  }

  // The chunk loop (render_into, and render() through it): chunks of requests_per_launch() requests rotate
  // through the executor's batch slots, whose streams start after the caller's work; the caller's stream
  // continues after theirs. Request i writes row out_index[i] (row i when null), and its sub-sample
  // instances at row i when sub is not null. Every request traces n_sub sub-samples (its k_count).
  // With pv, the P&V outputs of the chunks (slot rows as irr's).
  // irr_unshadowed (rows as irr's; the dense channel shadow_factor): the irradiance with every proxy factor 1, written
  // for chunks with static words (scenes with baked proxies and shadow lights).
  void trace_shade(const SampleRequest* requests, size_t n, const PoseTableView& pv, const uint32_t* out_index,
                   float* irr, uint64_t* sig, uint32_t* sub, void* stream, uint32_t n_sub, uint32_t* klass = nullptr,
                   const PvOut* pv_out = nullptr, float* irr_unshadowed = nullptr) {
    Executor& e = *executor;
    if (!timing) {
      e.start_timing(stream);
      timing = true;
    }
    const size_t per_launch = requests_per_launch(n_sub), groups = bundle.light_groups.size();
    const uint32_t slots = e.batch_slots();
    for (uint32_t bs = 0; bs < slots; ++bs) e.join(e.stream(bs), stream);
    uint32_t chunk = 0;
    try {
      for (size_t begin = 0; begin < n; begin += per_launch, ++chunk) {
        const uint32_t bs = chunk % slots;
        const auto count = static_cast<uint32_t>(std::min(per_launch, n - begin));
        const TraceBatch batch{requests + begin, count, n_sub, pv, e.stream(bs), bs};
        HitRecord* hits = e.hit_buffer(bs, static_cast<size_t>(count) * n_sub);
        // Occlusion words: the shadow bits and, under accuracy.mover_ao traced (P2-F F9b), the AO bit.
        uint32_t* occluded = scene->num_shadow_lights() > 0 || scene->traces_ao()
                                 ? e.occlusion_buffer(bs, static_cast<size_t>(count) * n_sub)
                                 : nullptr;
        // Scenes with baked proxies (scans.md §4.2) also trace the static words that U2's shading reads.
        uint32_t* statics = scene->baked_proxies() && occluded != nullptr
                                ? e.static_occlusion_buffer(bs, static_cast<size_t>(count) * n_sub)
                                : nullptr;
        const uint32_t m0 = e.timing_mark(bs);
        tracer->sample(batch, hits, occluded, statics);
        const uint32_t m1 = e.timing_mark(bs);
        PvOut chunk_pv = pv_out != nullptr ? *pv_out : PvOut{};
        if (pv_out != nullptr && out_index == nullptr && chunk_pv.slot != nullptr) chunk_pv.slot += begin;
        const PvOut* cpv = pv_out != nullptr ? &chunk_pv : nullptr;
        float* unshadowed = statics != nullptr ? irr_unshadowed : nullptr;
        if (out_index != nullptr) {
          e.shade_into(bs, batch, hits, occluded, irr, sig, nullptr, out_index + begin, klass, cpv, statics, unshadowed);
        } else {
          e.shade_into(bs, batch, hits, occluded, irr + begin * groups, sig + begin,
                       sub != nullptr ? sub + begin * n_sub : nullptr, nullptr,
                       klass != nullptr ? klass + begin : nullptr, cpv, statics,
                       unshadowed != nullptr ? unshadowed + begin * groups : nullptr);
        }
        const uint32_t m2 = e.timing_mark(bs);
        if (options.census) {  // after the shade timer: the census is not render time
          e.census_into(bs, batch, hits, out_index != nullptr ? out_index + begin : nullptr,
                        out_index != nullptr || klass == nullptr ? klass : klass + begin);
        }
        pending.push_back({m0, m1, 0});
        pending.push_back({m1, m2, 1});
      }
    } catch (...) {
      // The caller's stream follows the queued chunks also when one fails.
      for (uint32_t bs = 0; bs < slots; ++bs) e.join(stream, e.stream(bs));
      throw;
    }
    for (uint32_t bs = 0; bs < std::min(slots, chunk); ++bs) e.join(stream, e.stream(bs));
  }

  // P2-D D2, probe_into's chunk loop (and probe()'s): chunks of requests_per_launch(1) one-sub-sample requests rotate
  // through the batch slots as trace_shade's; each is traced with the tracer's sample and its states written by the
  // executor's probe_into, the pair timed as stage 2 (RenderStats::probe_ms).
  void probe_chunks(const SampleRequest* requests, size_t n, const PoseTableView& pv, ProbeOut* out, void* stream) {
    Executor& e = *executor;
    if (!timing) {
      e.start_timing(stream);
      timing = true;
    }
    const size_t per_launch = requests_per_launch(1);
    const uint32_t slots = e.batch_slots();
    for (uint32_t bs = 0; bs < slots; ++bs) e.join(e.stream(bs), stream);
    uint32_t chunk = 0;
    try {
      for (size_t begin = 0; begin < n; begin += per_launch, ++chunk) {
        const uint32_t bs = chunk % slots;
        const auto count = static_cast<uint32_t>(std::min(per_launch, n - begin));
        const TraceBatch batch{requests + begin, count, 1, pv, e.stream(bs), bs};
        HitRecord* hits = e.hit_buffer(bs, count);
        uint32_t* occluded = scene->num_shadow_lights() > 0 ? e.occlusion_buffer(bs, count) : nullptr;
        // A scene with baked proxies traces the occlusion word by U2's inserted / scan rule (lights.h shadow_mask),
        // which needs a static-word buffer to select it: a probe's mask is render_into's word, as ever. The static word
        // itself is not part of a probe's state (ProbeOut).
        uint32_t* statics = scene->baked_proxies() && occluded != nullptr ? e.static_occlusion_buffer(bs, count) : nullptr;
        const uint32_t m0 = e.timing_mark(bs);
        tracer->sample(batch, hits, occluded, statics);
        e.probe_into(bs, batch, hits, occluded, out + begin);
        pending.push_back({m0, e.timing_mark(bs), 2});
      }
    } catch (...) {
      for (uint32_t bs = 0; bs < slots; ++bs) e.join(stream, e.stream(bs));
      throw;
    }
    for (uint32_t bs = 0; bs < std::min(slots, chunk); ++bs) e.join(stream, e.stream(bs));
  }

  // A probe's flags (RequestKind::kProbe, its shadow light); on the host its k_count and light are checked.
  void check_probe(const SampleRequest& q, size_t i) const {
    const uint32_t light = static_cast<uint32_t>(q.flags) >> kProbeLightShift;
    const auto lights = static_cast<uint32_t>(std::max(1, scene->num_shadow_lights()));
    if (q.k_count != 1 || light >= lights) {
      throw std::invalid_argument("probe: request " + std::to_string(i) + " has k_count " + std::to_string(q.k_count) +
                                  " and shadow light " + std::to_string(light) + " (a probe traces 1 sub-sample, lights < " +
                                  std::to_string(lights) + ")");
    }
  }

  // The GT path (gt_into, and gt() and render()'s centre flag through it): chunks of at most 64 k queries,
  // fewer when their worst case of one pose row each would exceed 64 MB, on slot 0 after the caller's work.
  // visibility: the shadow words are the start diagnostic's (GtBatch::proxy_visibility), not the plan pass's states.
  void centre(uint16_t cam, const GtQuery* queries, size_t n, GtSample* out, void* stream, uint32_t* shadow = nullptr,
              float* normal = nullptr, bool visibility = false) {
    Executor& e = *executor;
    const size_t row_bytes = sizeof(PoseRow) * scene->geometries().size() + sizeof(LightRow) * scene->lights().size();
    const size_t chunk = std::clamp<size_t>((size_t{64} << 20) / std::max<size_t>(1, row_bytes), 4096, 65536);
    const auto lights = static_cast<size_t>(scene->num_shadow_lights());
    e.join(e.stream(0), stream);
    try {
      for (size_t begin = 0; begin < n; begin += chunk) {
        const auto count = static_cast<uint32_t>(std::min(chunk, n - begin));
        PoseTableView pv{};
        const GtRequest* requests = e.gt_query_rows(0, cam, queries + begin, count, &pv);
        GtHit* hits = e.gt_hit_buffer(0, count);
        tracer->centre({requests, count, pv, e.stream(0), 0, visibility}, hits,
                       shadow != nullptr ? shadow + 2 * begin * lights : nullptr);
        e.evaluate_gt_into(0, requests, hits, count, pv, out + begin, normal != nullptr ? normal + 3 * begin : nullptr);
      }
    } catch (...) {
      e.join(stream, e.stream(0));  // as trace_shade: the caller's stream follows the queued chunks
      throw;
    }
    e.join(stream, e.stream(0));
  }

  // centre() on host queries of any cameras (query(i) of camera cam[i] into out[i]): one call per camera, on
  // the queries of that camera in order.
  template <class Query>
  void centre_host(const uint16_t* cam, size_t n, const Query& query, GtSample* out) {
    Executor& e = *executor;
    void* stream = e.stream(0);
    std::vector<size_t> index;
    std::vector<GtQuery> queries;
    std::vector<GtSample> samples;
    for (size_t c = 0; c < bundle.cameras.size(); ++c) {
      index.clear();
      queries.clear();
      for (size_t i = 0; i < n; ++i) {
        if (cam[i] != c) continue;
        index.push_back(i);
        queries.push_back(query(i));
      }
      const size_t m = index.size();
      if (m == 0) continue;
      samples.resize(m);
      const auto* q = static_cast<const GtQuery*>(e.stage(kStageQueries, queries.data(), m * sizeof(GtQuery), true, stream));
      auto* g = static_cast<GtSample*>(e.stage(kStageGt, samples.data(), m * sizeof(GtSample), false, stream));
      centre(static_cast<uint16_t>(c), q, m, g, stream);
      e.unstage(samples.data(), g, m * sizeof(GtSample), stream);
      for (size_t k = 0; k < m; ++k) out[index[k]] = samples[k];
    }
  }
};

Renderer::Renderer(SceneBundle bundle, RendererOptions options)
    : impl_(std::make_unique<Impl>(std::move(bundle), options)) {}
Renderer::~Renderer() = default;

void Renderer::build() {
  if (!impl_->scene) impl_->build();
}
bool Renderer::built() const { return impl_->scene != nullptr; }

void Renderer::set_window(int64_t base_ns) {
  impl_->window_ns = base_ns;
  if (impl_->poses) impl_->poses->reset(base_ns);
}
int64_t Renderer::window() const { return impl_->window_ns; }

BackendKind Renderer::backend() const { return impl_->options.backend; }
const SceneBundle& Renderer::bundle() const { return impl_->bundle; }
const RendererOptions& Renderer::options() const { return impl_->options; }
int32_t Renderer::num_groups() const { return static_cast<int32_t>(impl_->bundle.light_groups.size()); }

RenderResult Renderer::render(std::span<const uint16_t> cam, std::span<const uint32_t> pixel,
                              std::span<const int64_t> t_ns, RenderFlags flags) {
  Impl& s = *impl_;
  if (cam.size() != pixel.size() || cam.size() != t_ns.size()) {
    throw std::invalid_argument("render: cam, pixel and t_ns need the same length");
  }
  const Clock::time_point t_call = Clock::now();
  for (size_t i = 0; i < cam.size(); ++i) {
    s.check_camera_time(cam[i], t_ns[i]);
    s.check_pixel(cam[i], pixel[i]);
  }
  s.ensure_built();
  const uint32_t n_sub = s.options.scene.n_subsamples;
  const size_t n = cam.size();
  RenderResult res;
  res.count = n;
  res.num_groups = num_groups();
  res.subsamples = n_sub;
  res.irradiance.resize(n * static_cast<size_t>(res.num_groups));
  res.signature.resize(n);
  if (flags.sub_instance) res.sub_instance.resize(n * n_sub);
  if (flags.centre) res.centre.resize(n);
  if (flags.sample_class) res.sample_class.resize(n);
  // The dense channel shadow_factor: a scene with baked proxies (and shadow lights, so static words) traces the
  // unshadowed irradiance with the render and divides; any other scene has no proxy factor, so 1 everywhere.
  const bool unshadowed_pass = flags.shadow_factor && s.scene->baked_proxies() && s.scene->num_shadow_lights() > 0;
  std::vector<float> irr_unshadowed;
  if (flags.shadow_factor) res.shadow_factor.assign(n * static_cast<size_t>(res.num_groups), 1.0f);
  if (unshadowed_pass) irr_unshadowed.resize(res.shadow_factor.size());
  // P2-D D2: each group's P&V outputs into host staging (capacity: the group), then in request order into res.
  const int32_t lights = s.scene->num_shadow_lights();
  if (flags.pv != 0 && lights > kPvMaxLights) {
    throw std::invalid_argument("render: P&V outputs need at most " + std::to_string(kPvMaxLights) +
                                " shadow-casting lights");
  }
  if (flags.pv != 0 && s.scene->traces_ao()) {  // P2-F F9b x P2-D D2: the code has no AO bit (codes.h)
    throw std::invalid_argument("render: P&V outputs are not available under accuracy.mover_ao traced");
  }
  if (flags.pv != 0 && s.scene->has_specular()) {  // P2-F F6 x P2-D D2: the table's Lambert shares (codes.h)
    throw std::invalid_argument("render: P&V outputs are not available in a scene with a specular material");
  }
  if (flags.pv != 0 && s.device_two_channel()) throw std::invalid_argument(kPvNirRefusal);
  const uint32_t code_words = pv_code_words(n_sub, lights), table_words = pv_table_words(res.num_groups, lights);
  std::vector<uint32_t> pv_slot, pv_codes, pv_table;
  uint32_t pv_count = 0;
  if (flags.pv != 0) {
    res.pv_written.assign(n, 0);
    res.pv_codes.assign(n * code_words, 0);
    res.pv_table.assign(n * table_words, 0);
    res.pv_code_words = code_words;
    res.pv_table_words = table_words;
  }

  // A wrapper of render_into's chunk loop and gt_into's GT path: requests of any cameras and times take
  // their rows from the pose cache, in groups of at most kGroupLaunches launches (a group also ends where
  // the cache must be reset: the table is full, or a time is no int32 offset from its base); each group is
  // staged in executor memory, rendered and copied back. The next group's requests are built while the
  // executor renders this one (its pose-cache slots are only taken on the host; their rows are evaluated
  // by the next sync_poses, after this group's copy-back has waited for its work).
  Executor& e = *s.executor;
  void* stream = e.stream(0);
  const auto groups = static_cast<size_t>(res.num_groups);
  const size_t group_max = kGroupLaunches * s.requests_per_launch(n_sub);
  std::vector<SampleRequest> requests, next;
  // The requests of the group starting at `begin` into `out`; returns its end.
  const auto build_group = [&](size_t begin, std::vector<SampleRequest>& out) {
    out.clear();
    const size_t end = std::min(n, begin + group_max);
    for (size_t i = begin; i < end; ++i) {
      uint32_t slot;
      int32_t offset;
      if (!s.take_slot(cam[i], t_ns[i], out.empty(), &slot, &offset)) return i;
      out.push_back({cam[i], static_cast<uint16_t>(RequestKind::kSample), pixel[i], 0, n_sub, offset, slot});
    }
    return end;
  };
  size_t end = build_group(0, requests);
  for (size_t begin = 0; begin < n;) {
    const PoseTableView pv = e.sync_poses(*s.poses, 0);  // on slot 0's stream, which is `stream`
    const size_t m = end - begin;
    float* irr_host = res.irradiance.data() + begin * groups;
    uint32_t* sub_host = flags.sub_instance ? res.sub_instance.data() + begin * n_sub : nullptr;
    const auto* req =
        static_cast<const SampleRequest*>(e.stage(kStageRequests, requests.data(), m * sizeof(SampleRequest), true, stream));
    auto* irr = static_cast<float*>(e.stage(kStageIrradiance, irr_host, m * groups * sizeof(float), false, stream));
    auto* sig = static_cast<uint64_t*>(e.stage(kStageSignature, res.signature.data() + begin, m * sizeof(uint64_t), false,
                                               stream));
    uint32_t* sub = sub_host != nullptr ? static_cast<uint32_t*>(e.stage(kStageSubInstance, sub_host,
                                                                         m * n_sub * sizeof(uint32_t), false, stream))
                                        : nullptr;
    uint32_t* cls = flags.sample_class ? static_cast<uint32_t*>(e.stage(kStageClass, res.sample_class.data() + begin,
                                                                          m * sizeof(uint32_t), false, stream))
                                       : nullptr;
    float* irr_u = unshadowed_pass ? static_cast<float*>(e.stage(kStageIrrUnshadowed,
                                                                 irr_unshadowed.data() + begin * groups,
                                                                 m * groups * sizeof(float), false, stream))
                                   : nullptr;
    PvOut pvo;
    if (flags.pv != 0) {
      pv_slot.resize(m);
      pv_codes.resize(m * code_words);
      pv_table.resize(m * table_words);
      pv_count = 0;
      pvo.slot = static_cast<uint32_t*>(e.stage(kStagePvSlot, pv_slot.data(), m * sizeof(uint32_t), false, stream));
      pvo.codes = static_cast<uint32_t*>(
          e.stage(kStagePvCodes, pv_codes.data(), pv_codes.size() * sizeof(uint32_t), false, stream));
      pvo.table = static_cast<uint32_t*>(
          e.stage(kStagePvTable, pv_table.data(), pv_table.size() * sizeof(uint32_t), false, stream));
      pvo.count = static_cast<uint32_t*>(e.stage(kStagePvCount, &pv_count, sizeof(uint32_t), true, stream));
      pvo.capacity = static_cast<uint32_t>(m);
      pvo.all = flags.pv == 2 ? 1u : 0u;
    }
    s.trace_shade(req, m, pv, nullptr, irr, sig, sub, stream, n_sub, cls, flags.pv != 0 ? &pvo : nullptr, irr_u);
    const size_t next_end = end < n ? build_group(end, next) : end;
    e.unstage(irr_host, irr, m * groups * sizeof(float), stream);
    if (irr_u != nullptr) {
      e.unstage(irr_unshadowed.data() + begin * groups, irr_u, m * groups * sizeof(float), stream);
      for (size_t i = begin * groups; i < end * groups; ++i) {  // the ratio is 1 where the denominator is 0
        res.shadow_factor[i] = irr_unshadowed[i] > 0.0f ? res.irradiance[i] / irr_unshadowed[i] : 1.0f;
      }
    }
    e.unstage(res.signature.data() + begin, sig, m * sizeof(uint64_t), stream);
    if (sub != nullptr) e.unstage(sub_host, sub, m * n_sub * sizeof(uint32_t), stream);
    if (cls != nullptr) e.unstage(res.sample_class.data() + begin, cls, m * sizeof(uint32_t), stream);
    if (flags.pv != 0) {
      e.unstage(pv_slot.data(), pvo.slot, m * sizeof(uint32_t), stream);
      e.unstage(pv_codes.data(), pvo.codes, pv_codes.size() * sizeof(uint32_t), stream);
      e.unstage(pv_table.data(), pvo.table, pv_table.size() * sizeof(uint32_t), stream);
      const uint32_t row_words = pv_row_words(res.num_groups, lights);
      for (size_t i = 0; i < m; ++i) {
        const uint32_t entry = pv_slot[i];
        if (entry >= m) continue;  // kPvNone (the capacity holds every request: no kPvOverflow)
        res.pv_written[begin + i] = 1;
        std::copy_n(pv_codes.begin() + static_cast<ptrdiff_t>(entry) * code_words, code_words,
                    res.pv_codes.begin() + static_cast<ptrdiff_t>((begin + i) * code_words));
        const uint32_t* t = pv_table.data() + static_cast<size_t>(entry) * table_words;
        std::copy_n(t, 1 + std::min(t[0] & 7u, kPvRows) * row_words,
                    res.pv_table.begin() + static_cast<ptrdiff_t>((begin + i) * table_words));
      }
    }
    if (flags.centre) {
      s.centre_host(cam.data() + begin, m, [&](size_t k) {
        const size_t i = begin + k;
        const auto w = static_cast<uint32_t>(s.bundle.cameras[cam[i]].width);
        return GtQuery{t_ns[i], static_cast<float>(pixel[i] % w), static_cast<float>(pixel[i] / w)};
      }, res.centre.data() + begin);
    }
    begin = end;
    end = next_end;
    requests.swap(next);
  }
  res.stats = take_render_stats();
  // host: the rest of the call (validation, request building, pose rows, staging, waits beyond the GPU's spans),
  // plus the host time of the pending render_into calls whose stage timers were taken with its own.
  res.stats.host_ms += std::max(0.0, ms_since(t_call) - res.stats.trace_ms - res.stats.shade_ms);
  return res;
}

std::vector<GtSample> Renderer::gt(std::span<const uint16_t> cam, std::span<const float> x,
                                   std::span<const float> y, std::span<const int64_t> t_ns) {
  Impl& s = *impl_;
  if (cam.size() != x.size() || cam.size() != y.size() || cam.size() != t_ns.size()) {
    throw std::invalid_argument("gt: cam, x, y and t_ns need the same length");
  }
  for (size_t i = 0; i < cam.size(); ++i) check_gt(cam[i], &x[i], &y[i], &t_ns[i], 1);
  s.ensure_built();
  std::vector<GtSample> out(cam.size());
  s.centre_host(cam.data(), cam.size(), [&](size_t i) { return GtQuery{t_ns[i], x[i], y[i]}; }, out.data());
  return out;
}

void Renderer::check_gt(uint16_t cam, const float* x, const float* y, const int64_t* t_ns, size_t n) const {
  const Impl& s = *impl_;
  // Positions must stay inside the support whose rays the camera LUT and injective domain cover:
  // pixel centres +- the sub-sample reach.
  const double reach = support_px(s.options.scene);
  for (size_t i = 0; i < n; ++i) {
    s.check_camera_time(cam, t_ns[i]);
    const Camera& c = s.bundle.cameras[cam];
    const double px = static_cast<double>(x[i]), py = static_cast<double>(y[i]);
    if (!(px >= -reach && px <= c.width - 1 + reach && py >= -reach && py <= c.height - 1 + reach)) {
      throw std::invalid_argument("gt: pixel position (" + std::to_string(px) + ", " + std::to_string(py) +
                                  ") lies outside camera " + std::to_string(cam) + "'s image");
    }
  }
}

void Renderer::begin_window(uint16_t cam, const WindowClock& clock, void* stream) {
  Impl& s = *impl_;
  s.ensure_built();
  s.check_camera_time(cam, clock.t_w);  // every slot time used lies in [t_w, t_w + W]
  s.check_camera_time(cam, clock.t_w + time::kWindowNs);
  s.executor->begin_window(cam, clock, stream);
}

void Renderer::assign_pose_rows(SampleRequest* requests, size_t n, void* stream) {
  if (n > UINT32_MAX) throw std::invalid_argument("render: more than 2^32 requests in one call");
  impl_->ensure_built();
  impl_->executor->assign_pose_rows(requests, static_cast<uint32_t>(n), stream);
}

void Renderer::render_into(const SampleRequest* requests, size_t n, const uint32_t* out_index, float* irradiance,
                           uint64_t* signature, void* stream, uint32_t* sample_class, uint32_t subsamples,
                           const PvOut* pv) {
  Impl& s = *impl_;
  s.ensure_built();
  const Clock::time_point t_host = Clock::now();
  const uint32_t n_sub = subsamples != 0 ? subsamples : s.options.scene.n_subsamples;
  if (n_sub > static_cast<uint32_t>(kMaxSubsamples)) {
    throw std::invalid_argument("render_into: subsamples must be in [1, " + std::to_string(kMaxSubsamples) + "]");
  }
  const int32_t lights = s.scene->num_shadow_lights();
  if (pv != nullptr && lights > kPvMaxLights) {
    throw std::invalid_argument("render_into: P&V outputs need at most " + std::to_string(kPvMaxLights) +
                                " shadow-casting lights");
  }
  if (pv != nullptr && s.scene->traces_ao()) {  // P2-F F9b x P2-D D2: the code has no AO bit (codes.h)
    throw std::invalid_argument("render_into: P&V outputs are not available under accuracy.mover_ao traced");
  }
  if (pv != nullptr && s.scene->has_specular()) {  // P2-F F6 x P2-D D2: the table's Lambert shares (codes.h)
    throw std::invalid_argument("render_into: P&V outputs are not available in a scene with a specular material");
  }
  if (pv != nullptr && s.device_two_channel()) throw std::invalid_argument(kPvNirRefusal);
  if (n == 0) return;
  if (s.tracer->memory_space() == MemorySpace::kHost) {  // host requests: the uniform N is checked
    for (size_t i = 0; i < n; ++i) {
      if (requests[i].k_count != n_sub) {
        throw std::invalid_argument("render_into: request " + std::to_string(i) + " has k_count " +
                                    std::to_string(requests[i].k_count) + ", the call renders " +
                                    std::to_string(n_sub) + " sub-samples per request");
      }
    }
  }
  // execution.pv_outputs: the calls P&V would take outputs of (N >= kPvMeasureN) write them to scratch, and their
  // entries and bytes are counted on slot 0 after the call (pv_scratch_add), the caller's stream after that.
  Executor& e = *s.executor;
  PvOut scratch;
  const bool measure = pv == nullptr && s.options.pv_outputs && n_sub >= kPvMeasureN && lights <= kPvMaxLights &&
                       !s.scene->traces_ao() &&  // P2-F F9b: P&V is not eligible under accuracy.mover_ao traced
                       !s.scene->has_specular();  // nor in a scene with a lobe (P2-F F6)
  const auto groups = static_cast<int32_t>(s.bundle.light_groups.size());
  if (measure) {
    scratch = e.pv_scratch(n, pv_code_words(n_sub, lights), pv_table_words(groups, lights), stream);
    pv = &scratch;
  }
  s.trace_shade(requests, n, e.window_poses(), out_index, irradiance, signature, nullptr, stream, n_sub, sample_class,
                pv);
  if (measure) {
    e.join(e.stream(0), stream);
    e.pv_scratch_add(sizeof(uint32_t) * (static_cast<uint64_t>(pv_code_words(n_sub, lights)) +
                                         pv_table_words(groups, lights)),
                     e.stream(0));
    e.join(stream, e.stream(0));
  }
  s.pending_host_ms += ms_since(t_host);
}

void Renderer::probe_into(const SampleRequest* requests, size_t n, ProbeOut* out, void* stream) {
  Impl& s = *impl_;
  s.ensure_built();
  if (s.scene->traces_ao()) {  // P2-F F9b x P2-D D2: a probe's mask would carry the AO bit only with shadow lights
    throw std::invalid_argument("probe_into: probes are not available under accuracy.mover_ao traced");
  }
  const Clock::time_point t_host = Clock::now();
  if (n == 0) return;
  if (s.tracer->memory_space() == MemorySpace::kHost) {
    for (size_t i = 0; i < n; ++i) s.check_probe(requests[i], i);
  }
  s.probe_chunks(requests, n, s.executor->window_poses(), out, stream);
  s.pending_host_ms += ms_since(t_host);
}

ProbeResult Renderer::probe(std::span<const uint16_t> cam, std::span<const uint32_t> pixel, std::span<const uint32_t> k,
                            std::span<const int64_t> t_ns, std::span<const uint32_t> light) {
  Impl& s = *impl_;
  const size_t n = cam.size();
  if (pixel.size() != n || k.size() != n || t_ns.size() != n || light.size() != n) {
    throw std::invalid_argument("probe: cam, pixel, k, t_ns and light need the same length");
  }
  const Clock::time_point t_call = Clock::now();
  for (size_t i = 0; i < n; ++i) {
    s.check_camera_time(cam[i], t_ns[i]);
    s.check_pixel(cam[i], pixel[i]);
    if (k[i] >= static_cast<uint32_t>(kMaxSubsamples) || light[i] >= (uint32_t{1} << (16 - kProbeLightShift))) {
      throw std::invalid_argument("probe: sub-sample " + std::to_string(k[i]) + " or light " + std::to_string(light[i]) +
                                  " out of range");
    }
  }
  s.ensure_built();
  if (s.scene->traces_ao()) {  // P2-F F9b x P2-D D2: as probe_into
    throw std::invalid_argument("probe: probes are not available under accuracy.mover_ao traced");
  }
  ProbeResult res;
  res.out.resize(n);
  Executor& e = *s.executor;
  void* stream = e.stream(0);
  const size_t group_max = kGroupLaunches * s.requests_per_launch(1);
  std::vector<SampleRequest> requests;
  for (size_t begin = 0; begin < n;) {
    requests.clear();
    size_t end = begin;
    for (; end < n && end - begin < group_max; ++end) {
      uint32_t slot;
      int32_t offset;
      if (!s.take_slot(cam[end], t_ns[end], requests.empty(), &slot, &offset)) break;
      const auto flags = static_cast<uint16_t>(static_cast<uint32_t>(RequestKind::kProbe) | light[end] << kProbeLightShift);
      requests.push_back({cam[end], flags, pixel[end], k[end], 1, offset, slot});
      s.check_probe(requests.back(), end);
    }
    const size_t m = end - begin;
    const PoseTableView pv = e.sync_poses(*s.poses, 0);
    const auto* req =
        static_cast<const SampleRequest*>(e.stage(kStageRequests, requests.data(), m * sizeof(SampleRequest), true, stream));
    auto* out = static_cast<ProbeOut*>(e.stage(kStageProbes, res.out.data() + begin, m * sizeof(ProbeOut), false, stream));
    s.probe_chunks(req, m, pv, out, stream);
    e.unstage(res.out.data() + begin, out, m * sizeof(ProbeOut), stream);
    begin = end;
  }
  res.stats = take_render_stats();
  res.stats.host_ms += std::max(0.0, ms_since(t_call) - res.stats.probe_ms);
  return res;
}

RenderStats Renderer::take_render_stats() {
  Impl& s = *impl_;
  RenderStats out;
  out.host_ms = s.pending_host_ms;
  out.census_on = s.options.census;
  out.pv_on = s.options.pv_outputs;
  if (s.timing) {
    Executor& e = *s.executor;
    for (uint32_t bs = 0; bs < e.batch_slots(); ++bs) e.wait(bs);
    if (s.options.census) {  // the executor's running totals since the last take
      std::array<uint64_t, kCensusWords> totals{};
      e.census_totals(totals.data());
      for (size_t i = 0; i < totals.size(); ++i) out.census[i] = totals[i] - s.census_taken[i];
      s.census_taken = totals;
    }
    if (s.options.pv_outputs) {  // execution.pv_outputs: the scratch totals since the last take
      std::array<uint64_t, 2> totals{};
      e.pv_scratch_totals(totals.data());
      out.pv_entries = totals[0] - s.pv_taken[0];
      out.pv_bytes = totals[1] - s.pv_taken[1];
      s.pv_taken = totals;
    }
    std::vector<Span> spans;
    spans.reserve(s.pending.size());
    for (const Impl::PendingSpan& p : s.pending) spans.push_back({e.timing_ms(p.begin), e.timing_ms(p.end), p.stage});
    double stage_ms[3] = {0.0, 0.0, 0.0};
    credit_spans(spans, stage_ms);
    out.trace_ms = stage_ms[0];
    out.shade_ms = stage_ms[1];
    out.probe_ms = stage_ms[2];
    e.reset_timing();
  }
  s.pending.clear();
  s.pending_host_ms = 0.0;
  s.timing = false;
  return out;
}

void Renderer::gt_into(uint16_t cam, const GtQuery* queries, size_t n, GtSample* out, void* stream, uint32_t* shadow,
                       float* normal) {
  Impl& s = *impl_;
  s.ensure_built();
  if (cam >= s.bundle.cameras.size()) throw std::invalid_argument("gt_into: camera index out of range");
  if (n > 0) s.centre(cam, queries, n, out, stream, s.scene->num_shadow_lights() > 0 ? shadow : nullptr, normal);
}

bool Renderer::has_specular() const {
  for (const Body& body : impl_->bundle.bodies) {
    for (const Material& m : body.materials) {
      if (m.has_lobe()) return true;
    }
  }
  return false;
}

int32_t Renderer::num_shadow_lights() const {
  int32_t n = 0;
  for (const Light& l : impl_->bundle.lights) n += positional(l.kind) && l.casts_shadows ? 1 : 0;
  return n;
}

bool Renderer::nir_two_channel() const {
  return SceneData::needs_two_channels(impl_->bundle, impl_->options.nir_fraction);
}

bool Renderer::pv_supported() const { return !(impl_->options.backend == BackendKind::kOptix && nir_two_channel()); }

bool Renderer::hold(std::span<const uint32_t> ids, int64_t t_ns) {
  Impl& s = *impl_;
  s.ensure_built();
  if (!holds_movers()) return false;  // baked proxies: the movers stay apart (the engine does not ask; a guard)
  const SceneData& sc = *s.scene;
  std::vector<int32_t> held;
  for (const uint32_t id : ids) {
    const auto b = std::lower_bound(sc.bodies().begin(), sc.bodies().end(), id,
                                    [](const BodyView& v, uint32_t x) { return v.id < x; });
    // Movers only: a static body split from the world (trajectory -1, scans.md §4.2) is never held.
    if (b != sc.bodies().end() && b->id == id && b->geometry > 0 &&
        sc.geometries()[static_cast<size_t>(b->geometry)].trajectory >= 0) {
      held.push_back(b->geometry);
    }
  }
  if (!held.empty() && sc.traces_ao()) {  // P2-F F9b: no AO ray looks into the static world
    throw std::invalid_argument("hold: accuracy.mover_ao traced traces every mover as a mover (no held set)");
  }
  std::sort(held.begin(), held.end());
  held.erase(std::unique(held.begin(), held.end()), held.end());
  std::vector<RigidPose> poses(held.size());
  for (size_t i = 0; i < held.size(); ++i) {
    const int32_t traj = sc.geometries()[static_cast<size_t>(held[i])].trajectory;
    BodyState b;
    if (!body_state(sc.trajectories()[static_cast<size_t>(traj)], t_ns, &b)) {
      throw std::invalid_argument("hold: time " + std::to_string(t_ns) + " ns lies outside a held body's trajectory");
    }
    poses[i] = b.pose;
  }
  if (held == s.held && (held.empty() || std::memcmp(poses.data(), s.held_poses.data(), poses.size() * sizeof(RigidPose)) == 0)) {
    return false;
  }
  const std::unique_ptr<TraceSet> next = held.empty() ? nullptr : sc.trace_set(held, poses);
  s.held.clear();
  s.held_poses.clear();
  try {
    s.executor->use_traces(next.get());
    s.tracer->use_traces(next.get(), s.executor->scene_view());
  } catch (...) {
    // Back to the movers, which needs no memory (a failed build, e.g. out of device memory, leaves no
    // backend on a partial set), then rethrow.
    s.executor->use_traces(nullptr);
    s.tracer->use_traces(nullptr, s.executor->scene_view());
    s.poses->reshape(s.executor->scene_view().num_geometries);
    throw;
  }
  s.held = std::move(held);
  s.held_poses = std::move(poses);
  s.poses->reshape(s.executor->scene_view().num_geometries);
  return true;
}

bool Renderer::holds_movers() const { return !impl_->baked_proxies; }

size_t Renderer::held_device_bytes() const {
  const Impl& s = *impl_;
  return s.scene ? s.executor->held_device_bytes() + s.tracer->held_device_bytes() : 0;
}

DenseFrame Renderer::dense(uint16_t cam, int64_t t_ns, bool irradiance, bool gt_channels, bool shadow_factor) {
  Impl& s = *impl_;
  if (cam >= s.bundle.cameras.size()) throw std::invalid_argument("dense: camera index out of range");
  const Camera& c = s.bundle.cameras[cam];
  const size_t npix = static_cast<size_t>(c.width) * static_cast<size_t>(c.height);
  DenseFrame frame;
  if (irradiance || shadow_factor) {
    std::vector<uint16_t> cams(npix, cam);
    std::vector<uint32_t> pixels(npix);
    for (size_t p = 0; p < npix; ++p) pixels[p] = static_cast<uint32_t>(p);
    std::vector<int64_t> times(npix, t_ns);
    RenderFlags flags;
    flags.shadow_factor = shadow_factor;
    RenderResult r = render(cams, pixels, times, flags);
    if (irradiance) frame.irradiance = std::move(r.irradiance);
    frame.shadow_factor = std::move(r.shadow_factor);
  }
  if (gt_channels) {
    std::vector<uint16_t> cams(npix, cam);
    std::vector<float> xs(npix), ys(npix);
    for (size_t p = 0; p < npix; ++p) {
      xs[p] = static_cast<float>(p % static_cast<size_t>(c.width));
      ys[p] = static_cast<float>(p / static_cast<size_t>(c.width));
    }
    std::vector<int64_t> times(npix, t_ns);
    frame.gt = gt(cams, xs, ys, times);
  }
  return frame;
}

std::vector<float> Renderer::proxy_visibility(uint16_t cam, int64_t t_ns) {
  Impl& s = *impl_;
  if (cam >= s.bundle.cameras.size()) throw std::invalid_argument("proxy_visibility: camera index out of range");
  s.check_camera_time(cam, t_ns);
  s.ensure_built();
  const auto bits = static_cast<size_t>(s.scene->num_shadow_lights());
  if (!s.scene->baked_proxies() || bits == 0) return {};
  const Camera& c = s.bundle.cameras[cam];
  const size_t npix = static_cast<size_t>(c.width) * static_cast<size_t>(c.height);
  std::vector<GtQuery> queries(npix);
  for (size_t p = 0; p < npix; ++p) {
    queries[p] = {t_ns, static_cast<float>(p % static_cast<size_t>(c.width)),
                  static_cast<float>(p / static_cast<size_t>(c.width))};
  }
  std::vector<GtSample> samples(npix);
  std::vector<uint32_t> states(2 * npix * bits);
  Executor& e = *s.executor;
  void* stream = e.stream(0);
  const auto* q = static_cast<const GtQuery*>(e.stage(kStageQueries, queries.data(), npix * sizeof(GtQuery), true, stream));
  auto* g = static_cast<GtSample*>(e.stage(kStageGt, samples.data(), npix * sizeof(GtSample), false, stream));
  auto* sh = static_cast<uint32_t*>(
      e.stage(kStageShadowStates, states.data(), states.size() * sizeof(uint32_t), false, stream));
  s.centre(cam, q, npix, g, stream, sh, nullptr, true);
  e.unstage(samples.data(), g, npix * sizeof(GtSample), stream);
  e.unstage(states.data(), sh, states.size() * sizeof(uint32_t), stream);
  std::vector<float> out;
  for (const LightView& l : s.scene->lights()) {
    if (l.proxy == 0 || l.shadow_bit < 0) continue;
    uint64_t open = 0, blocked = 0;
    for (size_t p = 0; p < npix; ++p) {
      const uint32_t v = states[2 * (p * bits + static_cast<size_t>(l.shadow_bit))];
      open += v == 1;
      blocked += v == 2;
    }
    out.push_back(open + blocked > 0 ? static_cast<float>(static_cast<double>(blocked) / static_cast<double>(open + blocked))
                                     : 0.0f);
  }
  return out;
}

int32_t Renderer::split_geometries() {
  Impl& s = *impl_;
  s.ensure_built();
  const auto& geometries = s.scene->geometries();
  int32_t n = 0;
  for (size_t g = 1; g < geometries.size(); ++g) n += geometries[g].trajectory < 0 ? 1 : 0;
  return n;
}

}  // namespace balla::render
