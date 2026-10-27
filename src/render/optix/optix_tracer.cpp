#include "render/optix/optix_tracer.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/optix/optix_function_table.h"
#include "render/device_backend.h"
#include "render/optix/launch_params.h"
#include "render/scene_data.h"

namespace balla::render {
extern const unsigned char kRenderProgramsIr[];
extern const std::size_t kRenderProgramsIrSize;

namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    static_cast<void>(cudaGetLastError());  // a caught failure must not fail the next launch check
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
  }
}

void check(OptixResult res, const char* what, const char* log = nullptr) {
  if (res != OPTIX_SUCCESS) {
    std::string msg = std::string("OptiX error in ") + what + ": " + optixGetErrorName(res);
    if (log != nullptr && log[0] != '\0') msg += "\n" + std::string(log);
    throw std::runtime_error(msg);
  }
}

class DeviceMemory {
 public:
  DeviceMemory() = default;
  explicit DeviceMemory(size_t bytes) { allocate(bytes); }
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;
  DeviceMemory(DeviceMemory&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) { o.ptr_ = nullptr; }
  DeviceMemory& operator=(DeviceMemory&& o) noexcept {
    if (this != &o) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = o.ptr_;
      bytes_ = o.bytes_;
      o.ptr_ = nullptr;
    }
    return *this;
  }
  ~DeviceMemory() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  void allocate(size_t bytes) {
    if (ptr_ != nullptr) check(cudaFree(ptr_), "cudaFree");
    ptr_ = nullptr;  // empty if the allocation fails
    bytes_ = 0;
    void* p = nullptr;
    if (bytes > 0) check(cudaMalloc(&p, bytes), "cudaMalloc");
    ptr_ = p;
    bytes_ = bytes;
  }
  CUdeviceptr ptr() const { return reinterpret_cast<CUdeviceptr>(ptr_); }
  size_t bytes() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// The first five are made with the tracer (kSampleShadowAo: accuracy.mover_ao traced, P2-F F9b, kSampleShadow's rays plus
// each hit's ambient ray); the baked ones (scans.md §4.2: the sample-shadow and centre-shadow pipelines of scenes with
// baked proxies) when build() sees such a scene, so lit scenes pay for neither.
enum Pipeline : int { kSample = 0, kSampleShadow, kCentre, kCentreShadow, kSampleShadowAo, kSampleShadowBaked,
                      kCentreShadowBaked, kNumPipelines };
constexpr int kLitPipelines = kSampleShadowBaked;
constexpr const char* kRaygen[kNumPipelines] = {"__raygen__sample",         "__raygen__sample_shadow",
                                                "__raygen__centre",         "__raygen__centre_shadow",
                                                "__raygen__sample_shadow_ao", "__raygen__sample_shadow_baked",
                                                "__raygen__centre_shadow_baked"};

}  // namespace

struct OptixTracer::Impl {
  explicit Impl(DeviceExecutor& e) : executor(e), stream(static_cast<cudaStream_t>(e.stream(0))) {}
  ~Impl() {
    cudaDeviceSynchronize();
    for (OptixPipeline p : pipelines) {
      if (p != nullptr) optixPipelineDestroy(p);
    }
    for (OptixProgramGroup g : groups) {
      if (g != nullptr) optixProgramGroupDestroy(g);
    }
    if (module != nullptr) optixModuleDestroy(module);
    if (context != nullptr) optixDeviceContextDestroy(context);
  }

  DeviceExecutor& executor;
  cudaStream_t stream;  // builds; launches use the stream of their batch's slot
  OptixDeviceContext context = nullptr;
  OptixModule module = nullptr;
  // the raygens of kRaygen; null miss; null hit group.
  OptixProgramGroup groups[kNumPipelines + 2] = {};
  OptixPipeline pipelines[kNumPipelines] = {};
  DeviceMemory raygen_records[kNumPipelines], miss_record, hit_record;
  OptixShaderBindingTable sbt[kNumPipelines] = {};
  std::vector<DeviceMemory> gas;                       // per geometry of the scene
  std::vector<OptixTraversableHandle> scene_handles;  // 0 for geometries without triangles
  DeviceMemory handles;                               // scene_handles on the device
  DeviceMemory held_gas, held_handles;                // a held set's merged world and trace handles (use_traces)
  bool held = false;                                  // held_handles in use
  std::vector<DeviceMemory> params;  // launch parameters per executor slot
  SceneView view{};

  // Module, the shared miss and hit-group programs and the pipelines [first, last) of Pipeline.
  void create_pipelines(int first, int last);
  OptixTraversableHandle build_gas(const GeometryView& g, DeviceMemory& out);
  static void upload_handles(DeviceMemory& out, const std::vector<OptixTraversableHandle>& h) {
    out.allocate(h.size() * sizeof(OptixTraversableHandle));
    check(cudaMemcpy(reinterpret_cast<void*>(out.ptr()), h.data(), h.size() * sizeof(OptixTraversableHandle),
                     cudaMemcpyHostToDevice),
          "GAS handles");
    // Pageable cudaMemcpy may return before its DMA lands; launches run on non-blocking streams.
    check(cudaDeviceSynchronize(), "GAS handles");
  }
  // Launches on a slot's stream; the slot's parameter buffer is reused only by later work of the
  // same stream, so stream order protects it.
  void launch(Pipeline p, const optix::LaunchParams& lp, size_t width, void* launch_stream, uint32_t slot) {
    if (width == 0) return;
    if (width > (size_t{1} << 30)) throw std::invalid_argument("render: OptiX launch wider than 2^30 rays");
    const auto st = static_cast<cudaStream_t>(launch_stream);
    const DeviceMemory& buffer = params.at(slot);
    check(cudaMemcpyAsync(reinterpret_cast<void*>(buffer.ptr()), &lp, sizeof(lp), cudaMemcpyHostToDevice, st),
          "launch parameters");
    check(optixLaunch(pipelines[p], st, buffer.ptr(), sizeof(lp), &sbt[p], static_cast<unsigned>(width), 1, 1),
          "optixLaunch");
  }
  optix::LaunchParams base(const PoseTableView& poses) const {
    optix::LaunchParams lp{};
    lp.scene = view;
    lp.poses = poses;
    lp.handles = reinterpret_cast<const OptixTraversableHandle*>((held ? held_handles : handles).ptr());
    return lp;
  }
};

void OptixTracer::Impl::create_pipelines(int first, int last) {
  OptixModuleCompileOptions module_options = {};
  module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  OptixPipelineCompileOptions pipeline_options = {};
  pipeline_options.usesMotionBlur = 0;
  pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_options.numPayloadValues = 0;  // optixTraverse + hit-object queries (programs.cu)
  pipeline_options.numAttributeValues = 2;
  pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_options.pipelineLaunchParamsVariableName = "params";
  pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  char log[8192];
  size_t log_size = sizeof(log);
  log[0] = '\0';
  if (module == nullptr) {
    check(optixModuleCreate(context, &module_options, &pipeline_options,
                            reinterpret_cast<const char*>(kRenderProgramsIr), kRenderProgramsIrSize, log, &log_size,
                            &module),
          "optixModuleCreate", log);
  }

  // The raygens trace with optixTraverse and never invoke a closest-hit or miss program: the SBT's one
  // miss and one hit-group record hold null programs.
  constexpr int kMiss = kNumPipelines, kHit = kNumPipelines + 1;
  if (miss_record.ptr() == 0) {
    OptixProgramGroupDesc desc[2] = {};
    desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    OptixProgramGroupOptions group_options = {};
    log_size = sizeof(log);
    check(optixProgramGroupCreate(context, desc, 2, &group_options, log, &log_size, groups + kMiss),
          "optixProgramGroupCreate", log);
    SbtRecord miss, hit;
    check(optixSbtRecordPackHeader(groups[kMiss], &miss), "optixSbtRecordPackHeader");
    check(optixSbtRecordPackHeader(groups[kHit], &hit), "optixSbtRecordPackHeader");
    miss_record.allocate(sizeof(miss));
    hit_record.allocate(sizeof(hit));
    check(cudaMemcpy(reinterpret_cast<void*>(miss_record.ptr()), &miss, sizeof(miss), cudaMemcpyHostToDevice), "SBT");
    check(cudaMemcpy(reinterpret_cast<void*>(hit_record.ptr()), &hit, sizeof(hit), cudaMemcpyHostToDevice), "SBT");
  }

  for (int p = first; p < last; ++p) {
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = module;
    desc.raygen.entryFunctionName = kRaygen[p];
    OptixProgramGroupOptions group_options = {};
    log_size = sizeof(log);
    check(optixProgramGroupCreate(context, &desc, 1, &group_options, log, &log_size, groups + p),
          "optixProgramGroupCreate", log);
    // One pipeline per raygen: the compile-time specialization G6 measured as essential.
    OptixProgramGroup linked[3] = {groups[p], groups[kMiss], groups[kHit]};
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    check(optixPipelineCreate(context, &pipeline_options, &link_options, linked, 3, log, &log_size, &pipelines[p]),
          "optixPipelineCreate", log);
    OptixStackSizes stack = {};
    for (OptixProgramGroup g : linked) {
      check(optixUtilAccumulateStackSizes(g, &stack, pipelines[p]), "optixUtilAccumulateStackSizes");
    }
    unsigned dc_traversal = 0, dc_state = 0, continuation = 0;
    check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &dc_traversal, &dc_state, &continuation),
          "optixUtilComputeStackSizes");
    check(optixPipelineSetStackSize(pipelines[p], dc_traversal, dc_state, continuation, 1), "optixPipelineSetStackSize");

    SbtRecord rg;
    check(optixSbtRecordPackHeader(groups[p], &rg), "optixSbtRecordPackHeader");
    raygen_records[p].allocate(sizeof(rg));
    check(cudaMemcpy(reinterpret_cast<void*>(raygen_records[p].ptr()), &rg, sizeof(rg), cudaMemcpyHostToDevice), "SBT");
    sbt[p].raygenRecord = raygen_records[p].ptr();
    sbt[p].missRecordBase = miss_record.ptr();
    sbt[p].missRecordStrideInBytes = sizeof(SbtRecord);
    sbt[p].missRecordCount = 1;
    sbt[p].hitgroupRecordBase = hit_record.ptr();
    sbt[p].hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    sbt[p].hitgroupRecordCount = 1;
  }
}

OptixTraversableHandle OptixTracer::Impl::build_gas(const GeometryView& g, DeviceMemory& out) {
  CUdeviceptr vertices = reinterpret_cast<CUdeviceptr>(g.positions);
  const unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  OptixBuildInput input = {};
  input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  OptixBuildInputTriangleArray& tri = input.triangleArray;
  tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  tri.vertexStrideInBytes = 3 * sizeof(float);
  tri.numVertices = g.num_vertices;
  tri.vertexBuffers = &vertices;
  tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  tri.indexStrideInBytes = 3 * sizeof(uint32_t);
  tri.numIndexTriplets = g.num_triangles;
  tri.indexBuffer = reinterpret_cast<CUdeviceptr>(g.indices);
  tri.flags = &flags;
  tri.numSbtRecords = 1;

  OptixAccelBuildOptions options = {};
  options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  options.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sizes = {};
  check(optixAccelComputeMemoryUsage(context, &options, &input, 1, &sizes), "optixAccelComputeMemoryUsage");
  DeviceMemory temp(sizes.tempSizeInBytes), uncompacted(sizes.outputSizeInBytes), compacted_size(sizeof(size_t));
  OptixAccelEmitDesc emit = {};
  emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit.result = compacted_size.ptr();
  OptixTraversableHandle handle = 0;
  check(optixAccelBuild(context, stream, &options, &input, 1, temp.ptr(), temp.bytes(), uncompacted.ptr(),
                        uncompacted.bytes(), &handle, &emit, 1),
        "optixAccelBuild");
  size_t size = 0;
  check(cudaMemcpyAsync(&size, reinterpret_cast<void*>(compacted_size.ptr()), sizeof(size), cudaMemcpyDeviceToHost, stream),
        "compacted size");
  check(cudaStreamSynchronize(stream), "optixAccelBuild");
  if (size < uncompacted.bytes()) {
    out.allocate(size);
    check(optixAccelCompact(context, stream, handle, out.ptr(), size, &handle), "optixAccelCompact");
    check(cudaStreamSynchronize(stream), "optixAccelCompact");
  } else {
    out = std::move(uncompacted);
  }
  return handle;
}

OptixTracer::OptixTracer(DeviceExecutor& executor) : impl_(std::make_unique<Impl>(executor)) {
  check(balla::optix::init_function_table(), "optixInit");
  check(cudaFree(nullptr), "CUDA context initialization");
  OptixDeviceContextOptions options = {};
  check(optixDeviceContextCreate(nullptr, &options, &impl_->context), "optixDeviceContextCreate");
  impl_->create_pipelines(0, kLitPipelines);
  for (uint32_t s = 0; s < executor.batch_slots(); ++s) impl_->params.emplace_back(sizeof(optix::LaunchParams));
}

OptixTracer::~OptixTracer() = default;

void OptixTracer::build(const SceneData&, const SceneView& view) {
  Impl& s = *impl_;
  s.view = view;
  // Scenes with baked proxies add the two baked pipelines (scans.md §4.2, §7), made once per tracer, here.
  if (view.baked_proxies != 0 && s.pipelines[kSampleShadowBaked] == nullptr) {
    s.create_pipelines(kLitPipelines, kNumPipelines);
  }
  const std::vector<GeometryView>& geos = s.executor.device_geometries();
  s.scene_handles.assign(geos.size(), 0);
  s.gas.clear();
  s.gas.resize(geos.size());
  for (size_t g = 0; g < geos.size(); ++g) {
    if (geos[g].num_triangles > 0) s.scene_handles[g] = s.build_gas(geos[g], s.gas[g]);
  }
  Impl::upload_handles(s.handles, s.scene_handles);
}

void OptixTracer::use_traces(const TraceSet* t, const SceneView& view) {
  Impl& s = *impl_;
  s.held = false;  // the scene's own handles first (needs no memory)
  s.view = view;
  check(cudaDeviceSynchronize(), "trace set");  // queued launches read the held GAS and handles
  s.held_gas = DeviceMemory();
  if (t == nullptr) return;
  const GeometryView& world = s.executor.device_geometries()[0];  // the executor's merged world
  Impl::upload_handles(s.held_handles,
                       t->select(s.scene_handles.data(), world.num_triangles > 0 ? s.build_gas(world, s.held_gas) : 0));
  s.held = true;
}

size_t OptixTracer::held_device_bytes() const { return impl_->held_gas.bytes() + impl_->held_handles.bytes(); }

void OptixTracer::sample(const TraceBatch& b, HitRecord* hits, uint32_t* occluded, uint32_t* static_occluded) {
  // Static words come with shadow rays and a scene whose baked pipeline exists; without them the buffer is zeroed.
  const bool baked =
      static_occluded != nullptr && occluded != nullptr && impl_->pipelines[kSampleShadowBaked] != nullptr;
  if (static_occluded != nullptr && !baked && b.num_requests > 0) {
    check(cudaMemsetAsync(static_occluded, 0, static_cast<size_t>(b.num_requests) * b.subsamples * sizeof(uint32_t),
                          static_cast<cudaStream_t>(b.stream)),
          "static word");
  }
  optix::LaunchParams lp = impl_->base(b.poses);
  lp.requests = b.requests;
  lp.num_requests = b.num_requests;
  lp.subsamples = b.subsamples;
  lp.static_occluded = static_occluded;
  lp.hits = hits;
  lp.occluded = occluded;
  const bool ao = impl_->view.mover_ao == static_cast<int32_t>(MoverAo::kTraced) && impl_->view.ao_reach != nullptr;
  impl_->launch(baked ? kSampleShadowBaked : occluded == nullptr ? kSample : (ao ? kSampleShadowAo : kSampleShadow), lp,
                static_cast<size_t>(b.num_requests) * b.subsamples, b.stream, b.slot);
}

void OptixTracer::centre(const GtBatch& b, GtHit* out, uint32_t* shadow) {
  optix::LaunchParams lp = impl_->base(b.poses);
  lp.gt_requests = b.requests;
  lp.gt_hits = out;
  lp.num_gt = b.count;
  lp.shadow_states = shadow;
  const bool baked = shadow != nullptr && impl_->pipelines[kCentreShadowBaked] != nullptr;
  lp.proxy_visibility = baked && b.proxy_visibility ? 1u : 0u;
  impl_->launch(baked ? kCentreShadowBaked : shadow != nullptr ? kCentreShadow : kCentre, lp, b.count, b.stream,
                b.slot);
}

bool optix_device_available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
  return balla::optix::init_function_table() == OPTIX_SUCCESS;
}

DeviceBackend make_optix_backend() {
  DeviceBackend b;
  auto executor = std::make_unique<DeviceExecutor>();
  b.tracer = std::make_unique<OptixTracer>(*executor);
  b.executor = std::move(executor);
  return b;
}

}  // namespace balla::render
