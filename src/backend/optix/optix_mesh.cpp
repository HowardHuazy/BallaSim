#include "backend/optix/optix_mesh.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/optix/mesh_programs.h"
#include "backend/optix/optix_function_table.h"

namespace balla::optix {
extern const unsigned char kMeshProgramsIr[];
extern const std::size_t kMeshProgramsIrSize;
}  // namespace balla::optix

namespace balla::backend {
namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
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

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes_ > 0) check(cudaMalloc(&ptr_, bytes_), "cudaMalloc");
  }
  DeviceBuffer(const void* host, size_t bytes) : DeviceBuffer(bytes) {
    if (bytes_ > 0) check(cudaMemcpy(ptr_, host, bytes_, cudaMemcpyHostToDevice), "cudaMemcpy");
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  CUdeviceptr ptr() const { return reinterpret_cast<CUdeviceptr>(ptr_); }
  template <class T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }
  template <class T>
  void download(std::vector<T>& out) const {
    check(cudaMemcpy(out.data(), ptr_, out.size() * sizeof(T), cudaMemcpyDeviceToHost),
          "cudaMemcpy");
  }

 private:
  void* ptr_ = nullptr;
  size_t bytes_;
};

// Owns every OptiX object of one query; destroyed in reverse order of creation.
struct OptixState {
  OptixDeviceContext context = nullptr;
  OptixModule module = nullptr;
  OptixProgramGroup groups[3] = {nullptr, nullptr, nullptr};  // raygen, miss, hitgroup
  OptixPipeline pipeline = nullptr;

  OptixState() = default;
  OptixState(const OptixState&) = delete;
  OptixState& operator=(const OptixState&) = delete;
  ~OptixState() {
    if (pipeline) optixPipelineDestroy(pipeline);
    for (OptixProgramGroup g : groups) {
      if (g) optixProgramGroupDestroy(g);
    }
    if (module) optixModuleDestroy(module);
    if (context) optixDeviceContextDestroy(context);
  }
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct Gas {
  OptixTraversableHandle handle = 0;
  std::unique_ptr<DeviceBuffer> memory;
};

Gas build_gas(OptixDeviceContext context, const TriangleMeshView& mesh,
              const DeviceBuffer& vertices, const DeviceBuffer& indices) {
  CUdeviceptr vertex_ptr = vertices.ptr();
  const unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  OptixBuildInput input = {};
  input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  OptixBuildInputTriangleArray& tri = input.triangleArray;
  tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  tri.vertexStrideInBytes = 3 * sizeof(float);
  tri.numVertices = static_cast<unsigned int>(mesh.num_vertices);
  tri.vertexBuffers = &vertex_ptr;
  tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  tri.indexStrideInBytes = 3 * sizeof(uint32_t);
  tri.numIndexTriplets = static_cast<unsigned int>(mesh.num_triangles);
  tri.indexBuffer = indices.ptr();
  tri.flags = &flags;
  tri.numSbtRecords = 1;

  OptixAccelBuildOptions options = {};
  options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  options.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sizes = {};
  check(optixAccelComputeMemoryUsage(context, &options, &input, 1, &sizes),
        "optixAccelComputeMemoryUsage");
  DeviceBuffer temp(sizes.tempSizeInBytes);
  Gas gas;
  gas.memory = std::make_unique<DeviceBuffer>(sizes.outputSizeInBytes);
  check(optixAccelBuild(context, nullptr, &options, &input, 1, temp.ptr(), sizes.tempSizeInBytes,
                        gas.memory->ptr(), sizes.outputSizeInBytes, &gas.handle, nullptr, 0),
        "optixAccelBuild");
  check(cudaDeviceSynchronize(), "optixAccelBuild");
  return gas;
}

void create_pipeline(OptixState& s) {
  OptixModuleCompileOptions module_options = {};
  module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  OptixPipelineCompileOptions pipeline_options = {};
  pipeline_options.usesMotionBlur = 0;
  pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_options.numPayloadValues = 4;
  pipeline_options.numAttributeValues = 2;
  pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_options.pipelineLaunchParamsVariableName = "params";
  pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  char log[4096];
  size_t log_size = sizeof(log);
  log[0] = '\0';
  check(optixModuleCreate(s.context, &module_options, &pipeline_options,
                          reinterpret_cast<const char*>(optix::kMeshProgramsIr),
                          optix::kMeshProgramsIrSize, log, &log_size, &s.module),
        "optixModuleCreate", log);

  OptixProgramGroupDesc desc[3] = {};
  desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  desc[0].raygen.module = s.module;
  desc[0].raygen.entryFunctionName = "__raygen__mesh";
  desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  desc[1].miss.module = s.module;
  desc[1].miss.entryFunctionName = "__miss__mesh";
  desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  desc[2].hitgroup.moduleCH = s.module;
  desc[2].hitgroup.entryFunctionNameCH = "__closesthit__mesh";
  OptixProgramGroupOptions group_options = {};
  log_size = sizeof(log);
  check(optixProgramGroupCreate(s.context, desc, 3, &group_options, log, &log_size, s.groups),
        "optixProgramGroupCreate", log);

  OptixPipelineLinkOptions link_options = {};
  link_options.maxTraceDepth = 1;
  log_size = sizeof(log);
  check(optixPipelineCreate(s.context, &pipeline_options, &link_options, s.groups, 3, log,
                            &log_size, &s.pipeline),
        "optixPipelineCreate", log);

  OptixStackSizes stack = {};
  for (OptixProgramGroup g : s.groups) {
    check(optixUtilAccumulateStackSizes(g, &stack, s.pipeline), "optixUtilAccumulateStackSizes");
  }
  unsigned int dc_traversal = 0, dc_state = 0, continuation = 0;
  check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &dc_traversal, &dc_state, &continuation),
        "optixUtilComputeStackSizes");
  check(optixPipelineSetStackSize(s.pipeline, dc_traversal, dc_state, continuation, 1),
        "optixPipelineSetStackSize");
}

}  // namespace

MeshHits optix_intersect(const TriangleMeshView& mesh, const RayBatchView& rays) {
  check_indices(mesh);
  // OptiX launches at most 2^30 threads; this also keeps the device index 3 * i and the unsigned
  // params.count and launch width exact.
  if (rays.count > (size_t{1} << 30)) {
    throw std::invalid_argument("optix_intersect: more than 2^30 rays (the OptiX launch limit)");
  }
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) throw std::runtime_error("OptiX: no CUDA device");
  check(cudaFree(nullptr), "CUDA context initialization");
  check(optix::init_function_table(), "optixInit");
  if (mesh.num_triangles == 0) return all_misses(rays.count);  // OptiX rejects an empty GAS

  OptixState s;
  OptixDeviceContextOptions context_options = {};
  check(optixDeviceContextCreate(nullptr, &context_options, &s.context),
        "optixDeviceContextCreate");

  const DeviceBuffer vertices(mesh.vertices, 3 * mesh.num_vertices * sizeof(float));
  const DeviceBuffer indices(mesh.indices, 3 * mesh.num_triangles * sizeof(uint32_t));
  const Gas gas = build_gas(s.context, mesh, vertices, indices);
  create_pipeline(s);

  SbtRecord records[3];
  for (int i = 0; i < 3; ++i) {
    check(optixSbtRecordPackHeader(s.groups[i], &records[i]), "optixSbtRecordPackHeader");
  }
  const DeviceBuffer raygen_record(&records[0], sizeof(SbtRecord));
  const DeviceBuffer miss_record(&records[1], sizeof(SbtRecord));
  const DeviceBuffer hit_record(&records[2], sizeof(SbtRecord));
  OptixShaderBindingTable sbt = {};
  sbt.raygenRecord = raygen_record.ptr();
  sbt.missRecordBase = miss_record.ptr();
  sbt.missRecordStrideInBytes = sizeof(SbtRecord);
  sbt.missRecordCount = 1;
  sbt.hitgroupRecordBase = hit_record.ptr();
  sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
  sbt.hitgroupRecordCount = 1;

  const DeviceBuffer origins(rays.origins, 3 * rays.count * sizeof(float));
  const DeviceBuffer directions(rays.directions, 3 * rays.count * sizeof(float));
  const DeviceBuffer t(rays.count * sizeof(float));
  const DeviceBuffer prim(rays.count * sizeof(unsigned int));
  const DeviceBuffer u(rays.count * sizeof(float));
  const DeviceBuffer v(rays.count * sizeof(float));
  optix::MeshLaunchParams params = {};
  params.handle = gas.handle;
  params.origins = origins.as<const float>();
  params.directions = directions.as<const float>();
  params.t = t.as<float>();
  params.prim = prim.as<unsigned int>();
  params.u = u.as<float>();
  params.v = v.as<float>();
  params.count = static_cast<unsigned int>(rays.count);
  const DeviceBuffer params_buffer(&params, sizeof(params));

  MeshHits hits;
  hits.t.resize(rays.count);
  hits.prim.resize(rays.count);
  hits.u.resize(rays.count);
  hits.v.resize(rays.count);
  if (rays.count > 0) {
    check(optixLaunch(s.pipeline, nullptr, params_buffer.ptr(), sizeof(params), &sbt,
                      static_cast<unsigned int>(rays.count), 1, 1),
          "optixLaunch");
    check(cudaDeviceSynchronize(), "optixLaunch");
    t.download(hits.t);
    prim.download(hits.prim);
    u.download(hits.u);
    v.download(hits.v);
  }
  return hits;
}

}  // namespace balla::backend
