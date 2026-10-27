#include "render/cuda/device_executor.h"

#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "kernels/device_array.h"
#include "render/gt.h"
#include "render/pose.h"
#include "render/scene_data.h"
#include "render/shade/codes.h"
#include "render/shade/probe.h"
#include "render/shade/shade.h"

namespace balla::render {
namespace {

constexpr unsigned kBlock = 128;

// kernels::check_cuda that also clears the thread's last error, so a caught failure (an allocation the
// device cannot hold) does not fail the next launch check or CUB call.
void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) static_cast<void>(cudaGetLastError());
  kernels::check_cuda(err, what);
}

// Device allocation that only grows; its contents are not preserved when it does.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  void* ensure(size_t bytes) {
    if (bytes > capacity_) {
      if (ptr_ != nullptr) check(cudaFree(ptr_), "cudaFree");
      ptr_ = nullptr;  // empty if the allocation fails
      capacity_ = 0;
      void* p = nullptr;
      check(cudaMalloc(&p, bytes), "cudaMalloc");
      ptr_ = p;
      capacity_ = bytes;
    }
    return ptr_;
  }
  size_t capacity() const { return capacity_; }
  void swap(Buffer& o) noexcept {
    std::swap(ptr_, o.ptr_);
    std::swap(capacity_, o.capacity_);
  }
  template <class T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
  size_t capacity_ = 0;
};

constexpr unsigned kFullMask = 0xFFFFFFFFu;

__device__ float warp_sum(float v) {
  for (unsigned off = kLanes / 2; off > 0; off /= 2) v = v + __shfl_xor_sync(kFullMask, v, off);
  return v;
}

__device__ uint64_t warp_sum(uint64_t v) {
  for (unsigned off = kLanes / 2; off > 0; off /= 2) v += __shfl_xor_sync(kFullMask, v, off);
  return v;
}

// The xor butterfly with multiplication: render/shade/ao.h butterfly_product's schedule.
__device__ float warp_product(float v) {
  for (unsigned off = kLanes / 2; off > 0; off /= 2) v = v * __shfl_xor_sync(kFullMask, v, off);
  return v;
}

// One warp per pixel-sample (DESIGN §6.9 "Reductions"): lane l owns sub-samples l, l + 32, ...;
// clusters are found with warp votes in order of their first sub-sample, each lane sums its members'
// terms in increasing sub-sample order, and the xor butterfly combines the lanes, exactly the
// schedule of shade_pixel_sample on the CPU.
//
// Single pass over the sub-samples: every lane first reconstructs the surfaces of all its sub-samples
// and looks up their textures, whatever their cluster, so the lanes of different clusters do that
// work together; the cluster loop then only adds its members' terms (the light terms are computed
// there, from the stored point and normal: the same values), reduces and shades. Formerly each
// cluster ran the surface and texture work with only its member lanes active.
//
// kSlots (sub-samples per lane, >= ceil(N / 32)) and kShadow (shadow-light capacity) are compile-time
// so the per-lane state stays in registers; they do not change the arithmetic. Records are laid out as
// traced (render/tracer.h): request r's start at r * stride, stride = the batch's sub-samples.
// kShadeMinBlocks: the minimum resident blocks per SM that __launch_bounds__ asks of each instantiation, from ptxas
// on sm_75 (ENGINE_CPP §0.1 "P2-B B0 notes"; asked, then resident): <2, k <= 4> ask six, as before the class word
// (80 registers; it would take 85); <2, 32> and <8, k <= 4> ask three, the most without spills (140 and 164-166
// registers; at four they spill 0.1-0.5 KB); <8, 32> and <32, 32> ask one and get two (212 and 255 registers since
// P2-B B3's fixed64, 209 before; only <32, 32> spills).
template <uint32_t kSlots, int32_t kShadow>
constexpr int kShadeMinBlocks = kSlots <= 2 ? (kShadow <= 4 ? 6 : 3) : kSlots <= 8 ? (kShadow <= 4 ? 3 : 1) : 1;
// The V1 kernels (kV1: mover AO, P2-F F9a, and specular lobes, F6; kV1 absorbs F9a's kAO twins) carry the lobe's code
// and registers, so they ask less than their non-V1 twins, from ptxas on sm_75 (F6's record: RENDER.md "F6"): the most
// blocks whose spills stay within the twins' (at most 68 B): <2, k <= 1> five (96 registers, 0-56 B), <2, 4> four (128;
// at five 216 B), <2, 32> three (168, none: the kernel holds no unweighted pair), <8, 0> three (167, none), <8, 1> and
// <8, 4> two (178 and 216; at three 8-412 B), <8, 32> and <32, 32> one as their twins (236 and 255 registers; <32, 32>
// spills 3980 B, its twin 3708 B).
template <uint32_t kSlots, int32_t kShadow, bool kV1>
constexpr int kShadeMinBlocksV1 =
    !kV1 ? kShadeMinBlocks<kSlots, kShadow>
         : kSlots <= 2 ? (kShadow <= 1 ? 5 : kShadow <= 4 ? 4 : 3) : kSlots <= 8 ? (kShadow == 0 ? 3 : kShadow <= 4 ? 2 : 1) : 1;
// The kBaked kernels (scans S3a-2) are not V1. At six blocks the 2-slot, capacity-4 one holds 80 registers and spills 264 /
// 680 B (stack 696 B): T12 (batch 5) measured the N = 64 shade stage at 124 ms against U0's 58 ms (U2 / U0 1.94), while
// the 8-slot baked kernel (168 registers, three blocks) adds 6 % at N = 256. They ask three, as <2, 32> does (168 registers,
// 8 B of spills).
template <uint32_t kSlots, int32_t kShadow, bool kV1, bool kBaked>
constexpr int kShadeMinBlocksBaked = kBaked && kSlots <= 2 ? 3 : kShadeMinBlocksV1<kSlots, kShadow, kV1>;

// The unweighted pair's lights of a kernel's ClusterSums (shade.h unweighted_lights; plan p2f.md §4.10: the pair
// exists in kShadow <= 4 kernels only): the 2- and 8-slot kernels of kMaxShadowLights shadow lights are launched only in
// a scene with more than kUnweightedLights shadow lights, where R^s = R and the pair can never be read, so they carry
// none; the 32-slot kernel takes every scene above N = 256 whatever its lights and keeps it.
template <uint32_t kSlots, int32_t kShadow, bool kV1>
constexpr int32_t kUnwLightsOf =
    unweighted_lights(kShadow, kV1, kShadow <= kUnweightedLights || kSlots == kMaxSubsamples / kLanes);
static_assert(kUnwLightsOf<2, 4, true> == 4 && kUnwLightsOf<8, 1, true> == 1 && kUnwLightsOf<2, 0, true> == 0 &&
                  kUnwLightsOf<2, kMaxShadowLights, true> == 0 && kUnwLightsOf<8, kMaxShadowLights, true> == 0 &&
                  kUnwLightsOf<32, kMaxShadowLights, true> == 4 && kUnwLightsOf<8, kMaxShadowLights, false> == 0,
              "the unweighted pair's capacity per kernel");

// kUnlit: accuracy.unlit_mask (SceneView::unlit_mask, P2-C C2) compiled in; the kernels without it are the unmasked
// ones as they were (the mask's code would cost the N <= 64 kernels registers and spills whether or not it runs).
// kPv: P2-D D2's outputs (render/shade/codes.h) compiled in, written for the pixel-samples pv_claim gives an entry: the
// class word, then per non-environment cluster its table row (lane 0, after the cluster's radiance) and its members'
// row codes (4 bits per slot), then each slot's codes OR-reduced over the lanes that share a code word. Only render_into
// with a PvOut launches them (N > 64: the 8- and 32-slot kernels), so the others keep their registers.
// kV1: the radiance-v1 path compiled in, for a scene with accuracy.mover_ao proxy or traced (SceneView::mover_ao,
// P2-F F9a/F9b; render/shade/ao.h) or a specular material (SceneView::specular, P2-F F6; render/shade/bsdf.h): each
// cluster's 1 - O, one proxy sphere per lane and pass, multiplied by the xor butterfly (shade.h's host schedule); under
// traced (render/shade/ao_traced.h) the members' open share of their ambient rays, counted by ballots; the class
// word's specular bit; the unweighted shadow pair of ClusterSums<., kV1> and lobe_radiance for a cluster whose
// material has a lobe; for a scene with two-channel NIR (SceneView::nir_two_channel, P2-F F11a; render/shade/nir.h:
// F11a's kNir flag is folded into kV1 too) a second albedo per slot (rho2), one more cluster sum (ClusterSums<., kV1>
// t[7]) and the mix per light group. The AO branches test scene.mover_ao and the lookups test the texture's channels at
// run time (one kernel serves these features: F6 folds F9a's kAO into kV1, since a second twin axis would have doubled
// the build again); the kernels without kV1 are the Lambert kernels as they were.
// The two axes (kPv, kV1) cross: under proxy or traced a PvOut's kernel is the kPv twin with kV1, whose table rows store
// L_s,g with 1 - O (cluster_radiance's rad), so pv_outputs stays output-invariant under accuracy.mover_ao. P&V is not
// available in a scene with a lobe (Renderer::has_specular).
// kBaked: scenes with baked proxies (SceneView::baked_proxies; scans.md §4.2) compiled in, launched when the chunk has
// static words (Tracer::sample's static_occluded): they enter the signature, and a baked cluster takes U2's factor
// (shade.h accumulate_baked, cluster_radiance). These kernels hold accuracy.unlit_mask as a runtime switch instead of a
// twin, and the lit kernels are the ones they were. irradiance_unshadowed (scans.md §3.7, the dense channel
// shadow_factor; nullptr in runs): the baked kernels also write the irradiance with every proxy factor set to 1; the
// other kernels never read it. There is no kV1 x kBaked twin (a third twin axis would triple the build): the baked
// kernels are non-V1, so a scene with baked proxies and a lobe, mover AO or two-channel NIR is refused at
// Renderer::build (U3 of scans.md §4.2 takes them with the fill's AO).
template <uint32_t kSlots, int32_t kShadow, bool kUnlit = false, bool kPv = false, bool kV1 = false, bool kBaked = false>
__global__ void __launch_bounds__(kBlock, kShadeMinBlocksBaked<kSlots, kShadow, kV1, kBaked>)
    shade_kernel(SceneView scene, PoseTableView poses, const SampleRequest* requests, uint32_t count,
                             uint32_t stride, const HitRecord* hits, const uint32_t* occluded,
                             const uint32_t* static_occluded, float* irradiance, float* irradiance_unshadowed,
                             uint64_t* signature, uint32_t* sub_instance, const uint32_t* out_index,
                             uint32_t* klass, PvOut pv) {
  const uint32_t lane = threadIdx.x & (kLanes - 1);
  const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / kLanes;
  if (warp >= count) return;  // the whole warp leaves together
  const SampleRequest req = requests[warp];
  const uint32_t n = req.k_count;
  const size_t base = static_cast<size_t>(warp) * stride;
  hits += base;
  if (occluded != nullptr) occluded += base;
  if constexpr (kBaked) {
    if (static_occluded != nullptr) static_occluded += base;
  }
  if (sub_instance != nullptr) sub_instance += base;
  const CameraView& cam = scene.cameras[req.cam];
  // fixed64 (shade.h, D-P2-3): slots j < kPrefixSlots hold the prefix k < 64, so kernels of at most kPrefixSlots slots
  // (N <= 64) shade as phase1 and compile without the switch.
  constexpr uint32_t kPrefixSlots = kTexturePrefix / kLanes;
  uint32_t n_terms = n;
  if constexpr (kSlots > kPrefixSlots) n_terms = member_count(scene.fixed64, n);
  const float footprint = subsample_footprint_variance(n_terms);

  // Per sub-sample: cluster key (instance, material; the patch for material under accuracy.shading_patches crease,
  // shade.h hit_key), geometry (-1: environment), shadow mask, and the surface terms (point, shading normal, texture
  // albedo).
  uint32_t inst[kSlots], mat[kSlots], occ[kSlots];
  int32_t geom[kSlots];
  Vec3f point[kSlots], normal[kSlots];
  float rho[kSlots];
  float rho2[kV1 ? kSlots : 1];  // kV1: the NIR albedo of each slot (F11a), used in a two-channel scene
  uint32_t undone = 0;
  uint64_t sig = 0;
  uint32_t geometry_hash = 0;  // the class word's (shade.h sample_class)
  bool tex = false, spec = false;  // spec: a hit material with a lobe and alpha <= 1/2 (kV1; shade.h class_specular)
  uint32_t cached_id = kInstanceNone;
  int32_t cached = -1;
#pragma unroll
  for (uint32_t j = 0; j < kSlots; ++j) {
    const uint32_t k = lane + kLanes * j;
    inst[j] = kInstanceNone;
    mat[j] = occ[j] = 0;
    geom[j] = -1;
    point[j] = normal[j] = {0.0f, 0.0f, 0.0f};
    rho[j] = 0.0f;
    if constexpr (kV1) rho2[j] = 0.0f;
    if (k >= n) continue;
    const HitRecord h = hits[k];
    inst[j] = h.instance;
    occ[j] = occluded != nullptr ? occluded[k] : 0u;
    if (h.instance != kInstanceNone) {
      if (h.instance != cached_id) {
        const BodyView* b = find_body(scene, h.instance);
        cached = b != nullptr ? b->geometry : -1;
        cached_id = h.instance;
      }
      geom[j] = cached;
      if (cached >= 0) mat[j] = hit_key(scene.geometries[cached], h.prim);
    }
    const uint64_t gterm = geometry_term(k, inst[j], mat[j]);
    sig += gterm + shadow_term(k, occ[j]);
    if constexpr (kBaked) {  // the static word's term (shade.h static_term), where it is not 0
      const uint32_t st = static_occluded[k];
      if (st != 0) sig += static_term(k, st);
    }
    geometry_hash += static_cast<uint32_t>(gterm >> 32);
    undone |= 1u << j;
    if (sub_instance != nullptr) sub_instance[k] = h.instance;
    if (geom[j] < 0) continue;  // environment cluster (misses): no surface terms
    const GeometryView& geo = scene.geometries[geom[j]];
    const MaterialView& m = scene.materials[key_material(scene, mat[j])];
    const bool textured = m.texture >= 0 && geo.uvs != nullptr;
    tex = tex || textured;
    if constexpr (kV1) spec = spec || class_specular(m);
    const Vec3f d = subsample_direction(scene, cam, req.pixel, req.k_begin + k);
    const SurfaceSample ss = surface_sample(geo, pose_row(poses, req.pose_slot, geom[j]), h.prim, d, textured);
    point[j] = ss.p;
    normal[j] = ss.n_shade;
    if constexpr (kUnlit || kBaked) {
      if ((kUnlit || scene.unlit_mask != 0) && occ[j] != 0) {
        // accuracy.unlit_mask (shade.h lit_occlusion): the signature's term moves to the masked word, which the class
        // word and the cluster loop then take (a cleared bit adds its f = 0 to sum f V: the radiance is unchanged).
        const uint32_t lit = lit_occlusion(scene, poses, req.pose_slot, ss.p, ss.n_shade, occ[j]);
        sig += shadow_term(k, lit) - shadow_term(k, occ[j]);
        occ[j] = lit;
      }
    }
    if (textured) {
      // F11a: a two-channel scene's lit textures hold (visible, NIR) per texel; a baked texture there is mono (channels).
      if constexpr (kV1) {
        if (scene.textures[m.texture].channels == 2) {
          const Albedo2 a = subsample_albedo2(cam, scene.textures[m.texture], scene.texels, ss, footprint);
          rho[j] = a.vis;
          rho2[j] = a.nir;
        } else {
          rho[j] = subsample_albedo(cam, scene.textures[m.texture], scene.texels, ss, footprint);
        }
      } else {
        rho[j] = subsample_albedo(cam, scene.textures[m.texture], scene.texels, ss, footprint);
      }
    }
  }
  sig = warp_sum(sig);
  // The class word (shade.h sample_class), reduced and stored here so that none of it stays live in the cluster loop:
  // two or more clusters is a sub-sample whose key differs from sub-sample 0's; two butterflies reduce the sums and
  // the masks.
  [[maybe_unused]] uint32_t pv_entry = kPvNone;
  if (klass != nullptr || kPv) {
    const uint32_t inst_0 = __shfl_sync(kFullMask, inst[0], 0), mat_0 = __shfl_sync(kFullMask, mat[0], 0);
    uint32_t shadowed = 0, occ_or = 0, occ_and = ~0u;
    bool other = false;
#pragma unroll
    for (uint32_t j = 0; j < kSlots; ++j) {
      if (lane + kLanes * j >= n) continue;
      other = other || inst[j] != inst_0 || mat[j] != mat_0;
      if (geom[j] < 0) continue;
      shadowed += static_cast<uint32_t>(__popc(occ[j]));
      occ_or |= occ[j];
      occ_and &= occ[j];
    }
    uint32_t sums = (geometry_hash << 16) + shadowed, masks = class_masks(occ_or, occ_and, scene.num_shadow_lights);
    for (unsigned off = kLanes / 2; off > 0; off /= 2) {
      sums += __shfl_xor_sync(kFullMask, sums, off);
      masks |= __shfl_xor_sync(kFullMask, masks, off);
    }
    const uint32_t word = sample_class(sums, __any_sync(kFullMask, other), masks, __any_sync(kFullMask, tex),
                                       kV1 && __any_sync(kFullMask, spec));
    if (lane == 0 && klass != nullptr) klass[out_index != nullptr ? out_index[warp] : warp] = word;
    if constexpr (kPv) {
      if (lane == 0) {
        pv_entry = pv_claim(pv, word);
        if (pv.slot != nullptr) pv.slot[out_index != nullptr ? out_index[warp] : warp] = pv_entry;
      }
      pv_entry = __shfl_sync(kFullMask, pv_entry, 0);
    }
  }
  // kPv: the non-environment clusters so far, and per slot (4 bits each) its row code, kPvCodeEnv until a cluster
  // takes it.
  [[maybe_unused]] uint32_t pv_clusters = 0;
  [[maybe_unused]] uint32_t pv_code_row[(kSlots + 7) / 8];
  if constexpr (kPv) {
#pragma unroll
    for (uint32_t i = 0; i < (kSlots + 7) / 8; ++i) pv_code_row[i] = 0x66666666u;  // kPvCodeEnv in every nibble
  }

  float irr[kMaxLightGroups];
  for (int32_t g = 0; g < scene.num_groups; ++g) irr[g] = 0.0f;
  [[maybe_unused]] float irr_u[kMaxLightGroups];  // kBaked with irradiance_unshadowed: the unshadowed sums
  if constexpr (kBaked) {
    if (irradiance_unshadowed != nullptr) {
      for (int32_t g = 0; g < scene.num_groups; ++g) irr_u[g] = 0.0f;
    }
  }
  const float inv_n = 1.0f / static_cast<float>(n);
  for (;;) {
    // Cluster leader: the lowest sub-sample not yet assigned (slot-major, then lane).
    // The leader's key is taken from its slot with unrolled selects, keeping the slots in registers.
    uint32_t leader_lane = kLanes, my_inst = kInstanceNone, my_mat = 0;
    int32_t my_geom = -1;
#pragma unroll
    for (uint32_t j = 0; j < kSlots; ++j) {
      const uint32_t votes = __ballot_sync(kFullMask, (undone >> j) & 1u);
      if (leader_lane == kLanes && votes != 0) {
        leader_lane = static_cast<uint32_t>(__ffs(static_cast<int>(votes))) - 1;
        my_inst = inst[j];
        my_mat = mat[j];
        my_geom = geom[j];
      }
    }
    if (leader_lane == kLanes) break;
    const uint32_t inst0 = __shfl_sync(kFullMask, my_inst, leader_lane);
    const uint32_t mat0 = __shfl_sync(kFullMask, my_mat, leader_lane);
    const int32_t geom0 = __shfl_sync(kFullMask, my_geom, leader_lane);
    uint32_t members = 0;
#pragma unroll
    for (uint32_t j = 0; j < kSlots; ++j) {
      if (((undone >> j) & 1u) && inst[j] == inst0 && mat[j] == mat0) members |= 1u << j;
    }
    undone &= ~members;
    if (geom0 < 0) {  // environment cluster: the background radiance of the misses, if any (shade.h)
      if (inst0 == kInstanceNone && scene.has_background != 0) {
        uint32_t miss_count = 0;
#pragma unroll
        for (uint32_t j = 0; j < kSlots; ++j) {
          miss_count += static_cast<uint32_t>(__popc(__ballot_sync(kFullMask, ((members >> j) & 1u) != 0)));
        }
        const float w = static_cast<float>(miss_count) * inv_n;
        for (int32_t g = 0; g < scene.num_groups; ++g) irr[g] += w * scene.background[g];
        if constexpr (kBaked) {
          if (irradiance_unshadowed != nullptr) {
            for (int32_t g = 0; g < scene.num_groups; ++g) irr_u[g] += w * scene.background[g];
          }
        }
      }
      continue;
    }
    if constexpr (kPv) {
      const uint32_t code = pv_clusters < kPvRows ? pv_clusters : kPvCodeOverflow;
#pragma unroll
      for (uint32_t j = 0; j < kSlots; ++j) {
        if ((members >> j) & 1u) {
          const uint32_t shift = 4u * (j % 8u);
          pv_code_row[j / 8] = (pv_code_row[j / 8] & ~(0xFu << shift)) | (code << shift);
        }
      }
    }
    using Sums = ClusterSums<kShadow, kV1, kUnwLightsOf<kSlots, kShadow, kV1>>;
    Sums sums;
    sums.clear();
    uint32_t cluster_count = 0, term_count = 0;
    // fixed64: a cluster with a member in the prefix slots (its leader) takes the per-member terms from them only.
    bool prefix = false;
    if constexpr (kSlots > kPrefixSlots) {
      prefix = scene.fixed64 != 0 && __any_sync(kFullMask, (members & ((1u << kPrefixSlots) - 1u)) != 0);
    }
    const MaterialView& m = scene.materials[key_material(scene, mat0)];
    const bool textured = m.texture >= 0 && scene.geometries[geom0].uvs != nullptr;
    // accuracy.texture_shadow weighted (shade.h accumulate): a textured cluster's members weight f by their rho[j].
    const bool weighted = textured && scene.texture_shadow != 0;
    [[maybe_unused]] const bool unweighted = kV1 && wants_unweighted(scene, m, textured);  // F6's unweighted pair
    uint32_t ao_blocked = 0;  // accuracy.mover_ao traced: the members whose ambient ray is blocked
    const bool light_terms = m.baked < 0;  // a baked cluster takes no light sums (shade.h cluster_radiance)
    const bool u2 = kBaked && !light_terms;  // ... but U2's factor sums, with the weighted form on the same switch
#pragma unroll
    for (uint32_t j = 0; j < kSlots; ++j) {
      const bool member = ((members >> j) & 1u) != 0;
      const uint32_t votes = static_cast<uint32_t>(__popc(__ballot_sync(kFullMask, member)));
      const bool member_terms = j < kPrefixSlots || !prefix;
      cluster_count += votes;
      if (member_terms) term_count += votes;
      if constexpr (kV1) {
        if (scene.mover_ao == kMoverAoTraced) {
          ao_blocked += static_cast<uint32_t>(__popc(__ballot_sync(kFullMask, member && (occ[j] & kAoTracedMask) != 0)));
        }
      }
      if (member) {
        float rho_nir = 0.0f;
        if constexpr (kV1) rho_nir = rho2[j];
        accumulate(sums, scene, poses, req.pose_slot, point[j], normal[j], rho[j], occ[j], member_terms, weighted,
                   unweighted, light_terms, rho_nir, cam.nir_fraction);
        if constexpr (kBaked) {
          if (u2) {  // the static word matters only where a proxy is blocked
            accumulate_baked(sums, scene, poses, req.pose_slot, point[j], normal[j], rho[j], occ[j],
                             occ[j] != 0 ? static_occluded[lane + kLanes * j] : 0u, weighted);
          }
        }
      }
    }
#pragma unroll
    for (int32_t e = 0; e < Sums::kSize; ++e) sums.t[e] = warp_sum(sums.t[e]);
    float ao = 1.0f;  // accuracy.mover_ao: 1 - O at the cluster (proxy: shade.h ao_visibility_host's schedule)
    if constexpr (kV1) {
      Vec3f ao_x, ao_n;
      if (scene.mover_ao == kMoverAoTraced) {
        ao = ao_traced_visibility(cluster_count, ao_blocked);
      } else if (scene.mover_ao == kMoverAoProxy && ao_receiver(term_count, sums, &ao_x, &ao_n)) {
        for (int32_t base = 0; base < scene.num_ao_spheres; base += static_cast<int32_t>(kLanes)) {
          ao = ao * warp_product(ao_sphere_factor(scene, poses, req.pose_slot, inst0, ao_x, ao_n,
                                                  base + static_cast<int32_t>(lane)));
        }
      }
    }
    float rad[kMaxLightGroups];
    [[maybe_unused]] float rad_u[kMaxLightGroups];
    // rad: with 1 - O (P2-F F9a), as the P&V row stores it; a baked cluster takes U2's factor (u2, scans.md §4.2)
    if constexpr (kBaked) {
      cluster_radiance(scene, poses, req, m, textured, term_count, cluster_count, sums, rad, ao, u2,
                       irradiance_unshadowed != nullptr ? rad_u : nullptr);
    } else {
      cluster_radiance(scene, poses, req, m, textured, term_count, cluster_count, sums, rad, ao, u2);
    }
    if constexpr (kPv) {
      if (pv_entry < kPvOverflow && pv_clusters < kPvRows && lane == 0) {
        const int32_t groups = scene.num_groups, lights = scene.num_shadow_lights;
        pv_table_row(pv.table + static_cast<size_t>(pv_entry) * pv_table_words(groups, lights) + 1 +
                         pv_clusters * pv_row_words(groups, lights),
                     scene, poses, req, inst0, mat0, m, textured, cluster_count, term_count, inv_n, sums, rad);
      }
      ++pv_clusters;
    }
    const float w = static_cast<float>(cluster_count) * inv_n;
    for (int32_t g = 0; g < scene.num_groups; ++g) irr[g] += w * rad[g];
    if constexpr (kBaked) {
      if (irradiance_unshadowed != nullptr) {
        for (int32_t g = 0; g < scene.num_groups; ++g) irr_u[g] += w * rad_u[g];
      }
    }
  }
  if constexpr (kPv) {
    if (pv_entry < kPvOverflow) {  // warp-uniform: the codes of the `stride` sub-samples (zeros past n), the header
      const int32_t lights = scene.num_shadow_lights;
      const uint32_t bits = pv_code_bits(lights), group = 32u / bits;
      uint32_t* codes = pv.codes + static_cast<size_t>(pv_entry) * pv_code_words(stride, lights);
#pragma unroll
      for (uint32_t j = 0; j < kSlots; ++j) {
        if (kLanes * j >= stride) break;
        const uint32_t k = lane + kLanes * j;
        const uint32_t row = (pv_code_row[j / 8] >> (4u * (j % 8u))) & 0xFu;
        uint32_t v = k < n ? pv_code(row, occ[j]) << (lane % group * bits) : 0u;
        for (uint32_t off = 1; off < group; off *= 2) v |= __shfl_xor_sync(kFullMask, v, off);
        if (lane % group == 0 && k < stride) codes[k * bits / 32u] = v;
      }
      if (lane == 0) {
        pv.table[static_cast<size_t>(pv_entry) * pv_table_words(scene.num_groups, lights)] =
            (pv_clusters < kPvRows ? pv_clusters : kPvRows) | (pv_clusters > kPvRows ? kPvTableOverflow : 0u);
      }
    }
  }
  if (lane == 0) {
    const size_t row = out_index != nullptr ? out_index[warp] : warp;  // render_into writes its own rows
    for (int32_t g = 0; g < scene.num_groups; ++g) irradiance[row * scene.num_groups + g] = irr[g];
    if constexpr (kBaked) {
      if (irradiance_unshadowed != nullptr) {
        for (int32_t g = 0; g < scene.num_groups; ++g) irradiance_unshadowed[row * scene.num_groups + g] = irr_u[g];
      }
    }
    signature[row] = sig;
  }
}

// The census of a chunk (render/census.h), one thread per pixel-sample: warp sums, then one atomic add per warp and
// counter (integer sums: the totals do not depend on the order).
__global__ void census_kernel(SceneView scene, PoseTableView poses, const SampleRequest* requests, uint32_t count,
                              uint32_t stride, const HitRecord* hits, const uint32_t* out_index, const uint32_t* klass,
                              unsigned long long* totals) {
  const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t v[kCensusWords] = {};
  if (r < count) {
    const size_t row = out_index != nullptr ? out_index[r] : r;
    census_add(v, requests[r], census_sample(scene, poses, requests[r], hits + static_cast<size_t>(r) * stride),
               klass != nullptr ? klass + row : nullptr);
  }
#pragma unroll
  for (int i = 0; i < kCensusWords; ++i) {
    const uint64_t s = warp_sum(v[i]);
    if ((threadIdx.x & (kLanes - 1)) == 0 && s != 0) atomicAdd(totals + i, static_cast<unsigned long long>(s));
  }
}

// P2-D D2: one thread per probe (render/shade/probe.h).
__global__ void probe_kernel(SceneView scene, PoseTableView poses, const SampleRequest* requests, uint32_t count,
                             const HitRecord* hits, const uint32_t* occluded, ProbeOut* out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = probe_state(scene, poses, requests[i], hits[i], occluded != nullptr ? occluded[i] : 0u);
}

// execution.pv_outputs: a call's entries and their bytes into the running totals.
__global__ void pv_add_kernel(const uint32_t* count, uint64_t entry_bytes, unsigned long long* totals) {
  totals[0] += *count;
  totals[1] += static_cast<unsigned long long>(*count) * entry_bytes;
}

__global__ void gt_kernel(SceneView scene, PoseTableView poses, const GtRequest* requests, const GtHit* hits,
                          uint32_t count, GtSample* out, float* normal) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = gt_eval(scene, poses, requests[i], hits[i], normal != nullptr ? normal + 3 * i : nullptr);
}

unsigned blocks(size_t n) { return static_cast<unsigned>((n + kBlock - 1) / kBlock); }

// One warp per pixel-sample; the smallest instantiation that holds N sub-samples and the scene's
// shadow lights: 2 or 8 slots (N <= 64, N <= 256) with 0, 1, 4 or 32 shadow lights, and 32 slots with 32 (N <= 1024);
// with accuracy.unlit_mask and shadow lights, its masked twin (without shadow lights there is nothing to mask); with a
// PvOut (P2-D D2), its kPv twin, 8 slots from N = 1 (the same arithmetic as 2 slots: empty slots add nothing); with
// accuracy.mover_ao proxy and proxy spheres, or traced and a mover with a reach (P2-F F9b), or a scene with a specular
// material (P2-F F6), the kV1 twin of that (F9a's AO twins, F6's lobes).
// A chunk with static words (a scene with baked proxies, scans.md §4.2) takes the kBaked kernels instead: 2, 8 and 32
// slots (kPv twins from 8) with 4 or 32 shadow lights; accuracy.unlit_mask is a runtime switch there. They are non-V1
// kernels: Renderer::build refuses the scenes that would need a kV1 x kBaked twin (a lobe, mover AO, two-channel NIR).
void launch_shade(const SceneView& view, const TraceBatch& b, const HitRecord* hits, const uint32_t* occluded,
                  const uint32_t* static_occluded, float* irr, float* irr_unshadowed, uint64_t* sig, uint32_t* sub,
                  const uint32_t* out_index, uint32_t* klass, const PvOut* pv, cudaStream_t stream) {
  const unsigned grid = blocks(static_cast<size_t>(b.num_requests) * kLanes);
  const int32_t shadow = view.num_shadow_lights;
  const PvOut pv_arg = pv != nullptr ? *pv : PvOut{};
  auto launch = [&](auto kernel) {
    kernel<<<grid, kBlock, 0, stream>>>(view, b.poses, b.requests, b.num_requests, b.subsamples, hits, occluded,
                                         static_occluded, irr, irr_unshadowed, sig, sub, out_index, klass, pv_arg);
  };
  auto by_shadow = [&](auto slots, auto unlit, auto with_pv, auto v1) {
    constexpr uint32_t kSlots = decltype(slots)::value;
    constexpr bool kUnlit = decltype(unlit)::value, kPv = decltype(with_pv)::value, kV1 = decltype(v1)::value;
    if (shadow == 0) {
      launch(shade_kernel<kSlots, 0, false, kPv, kV1>);
    } else if (shadow <= 1) {
      launch(shade_kernel<kSlots, 1, kUnlit, kPv, kV1>);
    } else if (shadow <= 4) {
      launch(shade_kernel<kSlots, 4, kUnlit, kPv, kV1>);
    } else {
      launch(shade_kernel<kSlots, kMaxShadowLights, kUnlit, kPv, kV1>);
    }
  };
  auto by_slots = [&](auto unlit, auto with_pv, auto v1) {
    constexpr bool kUnlit = decltype(unlit)::value, kPv = decltype(with_pv)::value, kV1 = decltype(v1)::value;
    if constexpr (!kPv) {
      if (b.subsamples <= 64) {
        by_shadow(std::integral_constant<uint32_t, 2>{}, unlit, with_pv, v1);
        return;
      }
    }
    if (b.subsamples <= 256) {
      by_shadow(std::integral_constant<uint32_t, 8>{}, unlit, with_pv, v1);
    } else {  // up to 1024: 32 slots, the slot masks' width
      static_assert(kMaxSubsamples / kLanes <= 32, "the slot masks are 32-bit");
      launch(shade_kernel<kMaxSubsamples / kLanes, kMaxShadowLights, kUnlit, kPv, kV1>);
    }
  };
  auto by_pv = [&](auto unlit, auto v1) {
    if (pv != nullptr) {
      by_slots(unlit, std::true_type{}, v1);
    } else {
      by_slots(unlit, std::false_type{}, v1);
    }
  };
  auto by_unlit = [&](auto v1) {
    if (view.unlit_mask != 0) {
      by_pv(std::true_type{}, v1);
    } else {
      by_pv(std::false_type{}, v1);
    }
  };
  // The baked-proxy kernels (kBaked): capacity 4 or 32 shadow lights (the capacity does not change the arithmetic).
  auto baked_by_shadow = [&](auto slots, auto with_pv) {
    constexpr uint32_t kSlots = decltype(slots)::value;
    constexpr bool kPv = decltype(with_pv)::value;
    if (shadow <= 4 && kSlots <= 8) {
      launch(shade_kernel<kSlots, 4, false, kPv, false, true>);
    } else {
      launch(shade_kernel<kSlots, kMaxShadowLights, false, kPv, false, true>);
    }
  };
  auto baked_by_slots = [&](auto with_pv) {
    constexpr bool kPv = decltype(with_pv)::value;
    if constexpr (!kPv) {
      if (b.subsamples <= 64) {
        baked_by_shadow(std::integral_constant<uint32_t, 2>{}, with_pv);
        return;
      }
    }
    if (b.subsamples <= 256) {
      baked_by_shadow(std::integral_constant<uint32_t, 8>{}, with_pv);
    } else {
      launch(shade_kernel<kMaxSubsamples / kLanes, kMaxShadowLights, false, kPv, false, true>);
    }
  };
  // P2-F F11a: P&V outputs with two-channel NIR are not built (F11a's kNir twins had no kPv variant; after the fold into
  // kV1 the kPv x kV1 kernels carry the NIR term, but no check covers their table rows with it).
  if (view.nir_two_channel != 0 && pv != nullptr) {
    throw std::invalid_argument("render: P&V outputs with two-channel near-infrared albedo (cameras whose "
                                "sensor.nir_fraction differs) are not built");
  }
  const bool v1 = (view.mover_ao == kMoverAoProxy && view.num_ao_spheres > 0) ||
                  (view.mover_ao == kMoverAoTraced && view.ao_reach != nullptr) || view.specular != 0 ||
                  view.nir_two_channel != 0;
  if (static_occluded != nullptr) {
    if (v1) {  // Renderer::build refuses these scenes already: no kV1 x kBaked kernels exist
      throw std::invalid_argument("render: baked proxies with a specular lobe, mover AO or two-channel near-infrared "
                                  "albedo are not built");
    }
    if (pv != nullptr) {
      baked_by_slots(std::true_type{});
    } else {
      baked_by_slots(std::false_type{});
    }
  } else if (v1) {
    by_unlit(std::true_type{});
  } else {
    by_unlit(std::false_type{});
  }
  check(cudaGetLastError(), "shade kernel launch");
}

struct PoseKey {
  int64_t t_ns;
  uint32_t cam;
  uint32_t reserved;
};

// fp64 pose rows of new slots (DESIGN §6.9: one kernel evaluates the exact pose of every (node,
// body)); the same compute_slot as the host, so the rows are bitwise equal. Request times were
// checked against the trajectory spans before any work was queued.
__global__ void pose_kernel(PoseSceneView scene, const PoseKey* keys, uint32_t count, size_t first, PoseRow* rows,
                            LightRow* light_rows) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const size_t slot = first + i;
  compute_slot(scene, keys[i].cam, keys[i].t_ns, rows + slot * static_cast<size_t>(scene.num_geometries),
               light_rows + slot * static_cast<size_t>(scene.num_lights));
}

constexpr uint32_t kRowNone = 0xFFFFFFFFu, kRowPending = 0xFFFFFFFEu;

// Window pose table, step 1: marks the slots the requests use (racing writes store the same value);
// *bad becomes the largest slot out of range (0: none, as slot 0 is always in range).
__global__ void mark_slots_kernel(const SampleRequest* requests, uint32_t n, uint32_t n_slots, uint32_t* index,
                                  uint32_t* bad) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t slot = requests[i].pose_slot;
  if (slot >= n_slots) {
    atomicMax(bad, slot);
  } else if (index[slot] == kRowNone) {
    index[slot] = kRowPending;
  }
}

// A refused call: its marks are undone and *bad is cleared.
__global__ void unmark_slots_kernel(uint32_t* index, uint32_t n_slots, uint32_t* bad) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k == 0) *bad = 0;
  if (k < n_slots && index[k] == kRowPending) index[k] = kRowNone;
}

__global__ void flag_new_kernel(const uint32_t* index, uint32_t n_slots, uint32_t* flag) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k <= n_slots) flag[k] = k < n_slots && index[k] == kRowPending;
}

// Step 2: new slots get the rows first + position (exclusive scan of the flags) and their poses.
__global__ void new_rows_kernel(PoseSceneView scene, WindowClock clock, uint32_t cam, uint32_t n_slots,
                                const uint32_t* position, uint32_t first, uint32_t* index, PoseRow* rows,
                                LightRow* light_rows) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_slots || index[k] != kRowPending) return;
  const uint32_t row = first + position[k];
  index[k] = row;
  compute_slot(scene, cam, slot_time(clock, k), rows + static_cast<size_t>(row) * scene.num_geometries,
               light_rows + static_cast<size_t>(row) * scene.num_lights);
}

__global__ void map_rows_kernel(SampleRequest* requests, uint32_t n, const uint32_t* index) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) requests[i].pose_slot = index[requests[i].pose_slot];
}

// gt_into: a query starts a row when its time differs from the previous query's.
__global__ void gt_flag_kernel(const GtQuery* q, uint32_t n, uint32_t* flag) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flag[i] = i == 0 || q[i].t_ns != q[i - 1].t_ns;
}

// row[i] = inclusive scan of the flags - 1; the first query of each row evaluates it.
__global__ void gt_rows_kernel(PoseSceneView scene, uint32_t cam, const GtQuery* q, uint32_t n, const uint32_t* row,
                               PoseRow* rows, LightRow* light_rows, GtRequest* requests) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t r = row[i] - 1;
  if (i == 0 || q[i].t_ns != q[i - 1].t_ns) {
    compute_slot(scene, cam, q[i].t_ns, rows + static_cast<size_t>(r) * scene.num_geometries,
                 light_rows + static_cast<size_t>(r) * scene.num_lights);
  }
  requests[i] = {static_cast<uint16_t>(cam), 0, q[i].x, q[i].y, r};
}

constexpr uint32_t kBatchSlots = 2;

// One batch slot: its stream and buffers.
struct Slot {
  cudaStream_t stream = nullptr;
  Buffer hits, occluded, static_occluded, gt_requests, gt_hits, pose_keys;
};

}  // namespace

struct DeviceExecutor::Impl {
  Slot slots[kBatchSlots];
  std::vector<std::unique_ptr<Buffer>> scene_storage;
  // In use (a held set's after use_traces) and the scene's own: geometries (host structs holding device
  // pointers) and the views (device pointers).
  std::vector<GeometryView> geometries, own_geometries;
  SceneView view{}, own_view{};
  PoseSceneView pose_scene{}, own_pose{};
  // The static world's arrays (trace geometry 0). A held set's merged world uses them with its tail after
  // the world's values: grown, keeping those values (zeros for normals and UVs the world has none of),
  // when a tail does not fit. world_bytes: their size without a tail; world_moved: one grew since the
  // world's own view was last pointed at them.
  Buffer world_positions, world_normals, world_uvs, world_indices, world_materials, world_instances, world_patches;
  size_t world_bytes = 0;
  bool world_moved = false;
  Buffer held_geometries, held_bodies, held_pose, held_ao_bodies, held_ao_spheres;  // a held set's tables
  // Pose-table mirror, shared by the slots. Rows uploaded on one slot's stream are published to the
  // other through rows_ready.
  Buffer rows, light_rows;
  cudaEvent_t rows_ready = nullptr;
  bool rows_recorded = false;
  uint64_t generation = ~uint64_t{0};
  size_t mirrored = 0;
  int32_t num_geometries = 0, num_lights = 0;

  // Window pose table: per slot its row (kRowNone, kRowPending), the rows, and scan scratch.
  struct WindowTable {
    uint16_t cam = 0;
    WindowClock clock{};
    uint32_t n_slots = 0, n_rows = 0, row_capacity = 0;
    Buffer index, flag, position, rows, light_rows, scan_tmp;
  } window;
  Buffer gt_flag, gt_row, gt_rows, gt_light_rows, gt_scan_tmp;  // gt_into scratch (one chunk)
  Buffer staged[kStageBuffers];                                  // the host API's staging
  Buffer census;  // census_into's running totals (kCensusWords), zeroed when first used
  Buffer pv_codes, pv_table, pv_count, pv_totals;  // pv_scratch's outputs, its count, the running totals
  cudaEvent_t join_event = nullptr;
  cudaEvent_t timing_epoch = nullptr;
  std::vector<cudaEvent_t> timing_events;
  uint32_t timing_used = 0;

  // Here, not in ~DeviceExecutor, so a constructor that throws still destroys what it created.
  ~Impl() {
    cudaDeviceSynchronize();
    for (Slot& slot : slots) {
      if (slot.stream != nullptr) cudaStreamDestroy(slot.stream);
    }
    if (rows_ready != nullptr) cudaEventDestroy(rows_ready);
    if (join_event != nullptr) cudaEventDestroy(join_event);
    if (timing_epoch != nullptr) cudaEventDestroy(timing_epoch);
    for (cudaEvent_t e : timing_events) cudaEventDestroy(e);
  }

  // Synchronous upload of n values at element `offset` of b (grown to hold them; nullptr when b stays empty).
  template <class T>
  const T* put_into(Buffer& b, const std::vector<T>& v, size_t offset = 0) {
    if (offset + v.size() == 0) return nullptr;
    b.ensure((offset + v.size()) * sizeof(T));
    if (!v.empty()) {
      check(cudaMemcpy(b.as<T>() + offset, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "scene upload");
    }
    return b.as<T>();
  }
  // Synchronous upload of a scene array into its own allocation (nullptr for empty arrays).
  template <class T>
  const T* put(const std::vector<T>& v) {
    scene_storage.push_back(std::make_unique<Buffer>());
    return put_into(*scene_storage.back(), v);
  }
  GeometryView put(const GeometryData& g) {
    return {put(g.positions), put(g.indices), g.has_normals ? put(g.normals) : nullptr,
            g.has_uvs ? put(g.uvs) : nullptr, put(g.materials), put(g.instances), g.instance,
            g.num_vertices(), g.num_triangles(), g.single_material, g.radius,
            g.has_patches ? put(g.patches) : nullptr, g.trajectory >= 0 ? 1 : 0};
  }
  // A held set's tail after the world's `keep` values of a world array: grows b first, keeping them (zeros
  // when b never held them).
  template <class T>
  const T* put_tail(Buffer& b, size_t keep, const std::vector<T>& tail) {
    const size_t bytes = (keep + tail.size()) * sizeof(T);
    if (bytes > b.capacity()) {
      Buffer grown;
      grown.ensure(bytes);
      if (keep > 0) {
        check(b.capacity() >= keep * sizeof(T)
                  ? cudaMemcpy(grown.as<void>(), b.as<void>(), keep * sizeof(T), cudaMemcpyDeviceToDevice)
                  : cudaMemset(grown.as<void>(), 0, keep * sizeof(T)),
              "held world");
      }
      b.swap(grown);
      world_moved = true;
    }
    return put_into(b, tail, keep);
  }

  template <class T>
  T* stage(Buffer& b, const std::vector<T>& host, cudaStream_t stream) {
    T* p = static_cast<T*>(b.ensure(std::max<size_t>(1, host.size()) * sizeof(T)));
    if (!host.empty()) {
      check(cudaMemcpyAsync(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "staging");
    }
    return p;
  }
};

DeviceExecutor::DeviceExecutor() : impl_(std::make_unique<Impl>()) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) throw std::runtime_error("render: no CUDA device");
  for (Slot& slot : impl_->slots) check(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking), "cudaStreamCreate");
  check(cudaEventCreateWithFlags(&impl_->rows_ready, cudaEventDisableTiming), "cudaEventCreate");
  check(cudaEventCreateWithFlags(&impl_->join_event, cudaEventDisableTiming), "cudaEventCreate");
  check(cudaEventCreate(&impl_->timing_epoch), "cudaEventCreate");
}

DeviceExecutor::~DeviceExecutor() = default;

uint32_t DeviceExecutor::batch_slots() const { return kBatchSlots; }
void* DeviceExecutor::stream(uint32_t slot) const { return impl_->slots[slot].stream; }
const std::vector<GeometryView>& DeviceExecutor::device_geometries() const { return impl_->geometries; }
const SceneView& DeviceExecutor::scene_view() const { return impl_->view; }

void DeviceExecutor::upload_scene(const SceneData& scene) {
  Impl& s = *impl_;
  SceneView v = scene.host_view();
  const GeometryData& w = scene.geometries()[0];
  s.own_geometries.push_back({s.put_into(s.world_positions, w.positions), s.put_into(s.world_indices, w.indices),
                              w.has_normals ? s.put_into(s.world_normals, w.normals) : nullptr,
                              w.has_uvs ? s.put_into(s.world_uvs, w.uvs) : nullptr,
                              s.put_into(s.world_materials, w.materials), s.put_into(s.world_instances, w.instances),
                              w.instance, w.num_vertices(), w.num_triangles(), w.single_material, w.radius,
                              w.has_patches ? s.put_into(s.world_patches, w.patches) : nullptr, 0});
  s.world_bytes = s.world_positions.capacity() + s.world_normals.capacity() + s.world_uvs.capacity() +
                  s.world_indices.capacity() + s.world_materials.capacity() + s.world_instances.capacity() +
                  s.world_patches.capacity();
  for (size_t g = 1; g < scene.geometries().size(); ++g) s.own_geometries.push_back(s.put(scene.geometries()[g]));
  s.geometries = s.own_geometries;
  v.geometries = s.put(s.geometries);
  v.bodies = s.put(scene.bodies());
  v.materials = s.put(scene.materials());
  v.patch_material = s.put(scene.patch_material());  // nullptr under accuracy.shading_patches material
  v.ao_bodies = s.put(scene.ao_bodies());  // accuracy.mover_ao proxy (nullptr under off)
  v.ao_spheres = s.put(scene.ao_spheres());
  v.ao_reach = scene.traces_ao() ? s.put(scene.ao_reach()) : nullptr;  // accuracy.mover_ao traced
  std::vector<TextureView> textures;
  for (size_t t = 0; t < scene.texture_levels().size(); ++t) {
    const auto& levels = scene.texture_levels()[t];
    textures.push_back({s.put(levels), static_cast<int32_t>(levels.size()), scene.texture_channels()[t],
                        scene.texture_tiles()[t]});
  }
  v.textures = s.put(textures);
  v.texels = s.put(scene.texels());
  v.baked_scale = s.put(scene.baked_scale());  // nullptr without baked materials (has_baked 0)
  std::vector<CameraView> cameras;
  for (const auto& c : scene.cameras()) {
    CameraView cv = c->view;
    cv.lut.rays = s.put(c->lut.rays());
    cv.pattern_seeds = s.put(c->seeds);
    cv.light_seeds = s.put(c->light_seeds);  // nullptr without area lights
    cameras.push_back(cv);
  }
  v.cameras = s.put(cameras);
  v.lights = s.put(scene.lights());
  if (v.specular != 0) {  // P2-F F6: the BSDF's six coefficient tables (render/shade/bsdf.h), kTableSize floats each
    const auto table = [&](const float* t) { return s.put(std::vector<float>(t, t + kTableSize)); };
    const bsdf::BsdfTables h = v.bsdf;
    v.bsdf = {table(h.spec_a), table(h.spec_b), table(h.ltc_m00), table(h.ltc_m02), table(h.ltc_m20), table(h.ltc_m22)};
  }
  v.light_points = s.put(scene.light_points());
  v.warp.cdf = s.put(scene.warp().cdf());
  v.warp.slope = s.put(scene.warp().slope());
  v.warp.guide = s.put(scene.warp().guide());
  s.view = v;
  s.num_geometries = static_cast<int32_t>(scene.geometries().size());
  s.num_lights = static_cast<int32_t>(scene.lights().size());

  std::vector<TrajectoryView> trajectories = scene.trajectories();
  for (size_t t = 0; t < trajectories.size(); ++t) {
    const std::vector<traj::SplineNode<double>>& nodes = scene.spline_nodes()[t];
    const traj::SplineNode<double>* d_nodes = s.put(nodes);
    trajectories[t].spline.nodes = d_nodes;
    // A piecewise trajectory (blender.md §4.7.7): its pieces' views point into the same node array at their rows'
    // offsets; the device copies keep the offsets and point into the device array.
    if (!scene.piece_views()[t].empty()) {
      std::vector<PieceView> pieces = scene.piece_views()[t];
      for (PieceView& p : pieces) p.spline.nodes = d_nodes + (p.spline.nodes - nodes.data());
      trajectories[t].spline.nodes = pieces[0].spline.nodes;
      trajectories[t].pieces = s.put(pieces);
    }
  }
  s.pose_scene = {s.put(trajectories), s.put(scene.pose_cameras()), s.put(scene.pose_geometries()), s.num_geometries,
                  s.put(scene.pose_lights()), s.num_lights};
  s.own_view = s.view;
  s.own_pose = s.pose_scene;
  // cudaMemcpy from pageable memory may return before its DMA lands, and the slot streams are
  // non-blocking: wait here so no later stream work can read a partial scene.
  check(cudaDeviceSynchronize(), "scene upload");
}

void DeviceExecutor::use_traces(const TraceSet* t) {
  Impl& s = *impl_;
  // The scene's own geometries first (host state only): a failure below leaves them in use.
  s.geometries = s.own_geometries;
  s.view = s.own_view;
  s.pose_scene = s.own_pose;
  s.num_geometries = s.own_view.num_geometries;
  s.generation = ~uint64_t{0};  // the pose rows change shape: the mirror and the window table start over
  s.window.n_slots = s.window.n_rows = s.window.row_capacity = 0;
  check(cudaDeviceSynchronize(), "trace set");  // queued work reads the arrays of the set in use
  GeometryView& w = s.own_geometries[0];
  // The world's own view follows its arrays once put_tail moved them, also when a later put_tail throws
  // (if following fails too, world_moved stays set and the next call follows them).
  auto follow_world = [&] {
    if (!s.world_moved) return;
    if (w.positions != nullptr) w.positions = s.world_positions.as<float>();
    if (w.indices != nullptr) w.indices = s.world_indices.as<uint32_t>();
    if (w.normals != nullptr) w.normals = s.world_normals.as<float>();
    if (w.uvs != nullptr) w.uvs = s.world_uvs.as<float>();
    if (w.materials != nullptr) w.materials = s.world_materials.as<uint32_t>();
    if (w.instances != nullptr) w.instances = s.world_instances.as<uint32_t>();
    if (w.patches != nullptr) w.patches = s.world_patches.as<uint32_t>();
    s.geometries[0] = w;
    check(cudaMemcpy(const_cast<GeometryView*>(s.own_view.geometries), &w, sizeof(GeometryView),
                     cudaMemcpyHostToDevice),
          "trace set upload");
    check(cudaDeviceSynchronize(), "trace set upload");  // pageable copies may return before their DMA lands
    s.world_moved = false;
  };
  const size_t nv = w.num_vertices, nt = w.num_triangles;
  GeometryView merged{};
  if (t != nullptr) {
    const GeometryData& tail = t->tail;
    try {
      merged = {s.put_tail(s.world_positions, 3 * nv, tail.positions),
                s.put_tail(s.world_indices, 3 * nt, tail.indices),
                tail.has_normals ? s.put_tail(s.world_normals, 3 * nv, tail.normals) : nullptr,
                tail.has_uvs ? s.put_tail(s.world_uvs, 2 * nv, tail.uvs) : nullptr,
                s.put_tail(s.world_materials, nt, tail.materials),
                s.put_tail(s.world_instances, nt, tail.instances),
                0,
                static_cast<uint32_t>(nv) + tail.num_vertices(),
                static_cast<uint32_t>(nt) + tail.num_triangles(),
                tail.single_material,
                w.sphere_radius,
                tail.has_patches ? s.put_tail(s.world_patches, nt, tail.patches) : nullptr, 0};
    } catch (...) {
      follow_world();  // an earlier put_tail may have freed the arrays the view pointed at
      throw;
    }
  }
  follow_world();
  if (t == nullptr) return;
  const std::vector<GeometryView> geometries = t->select(s.own_geometries.data(), merged);
  const GeometryView* views = s.put_into(s.held_geometries, geometries);
  const BodyView* bodies = s.put_into(s.held_bodies, t->bodies);
  const PoseGeometryDef* pose = s.put_into(s.held_pose, t->pose_geometries);
  const AoBodyView* ao_bodies = s.put_into(s.held_ao_bodies, t->ao_bodies);
  const AoSphere* ao_spheres = s.put_into(s.held_ao_spheres, t->ao_spheres);
  check(cudaDeviceSynchronize(), "trace set upload");
  s.geometries = geometries;
  s.view.geometries = views;
  s.view.bodies = bodies;
  s.view.ao_bodies = ao_bodies;
  s.view.ao_spheres = ao_spheres;
  s.pose_scene.geometries = pose;
  s.num_geometries = s.view.num_geometries = s.pose_scene.num_geometries = static_cast<int32_t>(geometries.size());
}

size_t DeviceExecutor::held_device_bytes() const {
  const Impl& s = *impl_;
  return s.world_positions.capacity() + s.world_normals.capacity() + s.world_uvs.capacity() +
         s.world_indices.capacity() + s.world_materials.capacity() + s.world_instances.capacity() +
         s.world_patches.capacity() - s.world_bytes +
         s.held_geometries.capacity() + s.held_bodies.capacity() + s.held_pose.capacity() +
         s.held_ao_bodies.capacity() + s.held_ao_spheres.capacity();
}

PoseTableView DeviceExecutor::sync_poses(PoseTable& table, uint32_t slot_index) {
  Impl& s = *impl_;
  Slot& slot = s.slots[slot_index];
  const size_t n = table.num_slots();
  const size_t row_bytes = sizeof(PoseRow) * static_cast<size_t>(s.num_geometries);
  const size_t light_bytes = sizeof(LightRow) * static_cast<size_t>(s.num_lights);
  const bool reset = table.generation() != s.generation;
  const bool grow = n * row_bytes > s.rows.capacity() || n * light_bytes > s.light_rows.capacity();
  if (reset || grow) {
    // Rows are rewritten from the start or reallocated: wait for every batch still reading them.
    check(cudaDeviceSynchronize(), "pose table reset");
    s.generation = table.generation();
    s.mirrored = 0;
    if (grow) {
      const size_t slots = std::max(n, 2 * (s.rows.capacity() / row_bytes));
      s.rows.ensure(slots * row_bytes);
      s.light_rows.ensure(std::max<size_t>(1, slots * light_bytes));
    }
  } else if (s.rows_recorded) {
    check(cudaStreamWaitEvent(slot.stream, s.rows_ready, 0), "pose rows wait");  // rows of the other slot
  }
  if (s.mirrored < n) {
    std::vector<PoseKey> keys;
    keys.reserve(n - s.mirrored);
    for (size_t i = s.mirrored; i < n; ++i) keys.push_back({table.keys()[i].t_ns, table.keys()[i].cam, 0});
    const PoseKey* d_keys = s.stage(slot.pose_keys, keys, slot.stream);
    pose_kernel<<<blocks(keys.size()), kBlock, 0, slot.stream>>>(s.pose_scene, d_keys, static_cast<uint32_t>(keys.size()),
                                                                 s.mirrored, s.rows.as<PoseRow>(),
                                                                 s.light_rows.as<LightRow>());
    check(cudaGetLastError(), "pose kernel launch");
    s.mirrored = n;
    check(cudaEventRecord(s.rows_ready, slot.stream), "pose rows event");
    s.rows_recorded = true;
  }
  return {s.rows.as<PoseRow>(), s.light_rows.as<LightRow>(), s.num_geometries, s.num_lights};
}

HitRecord* DeviceExecutor::hit_buffer(uint32_t slot, size_t count) {
  return static_cast<HitRecord*>(impl_->slots[slot].hits.ensure(std::max<size_t>(1, count) * sizeof(HitRecord)));
}

uint32_t* DeviceExecutor::occlusion_buffer(uint32_t slot, size_t count) {
  return static_cast<uint32_t*>(impl_->slots[slot].occluded.ensure(std::max<size_t>(1, count) * sizeof(uint32_t)));
}

uint32_t* DeviceExecutor::static_occlusion_buffer(uint32_t slot, size_t count) {
  return static_cast<uint32_t*>(
      impl_->slots[slot].static_occluded.ensure(std::max<size_t>(1, count) * sizeof(uint32_t)));
}

GtHit* DeviceExecutor::gt_hit_buffer(uint32_t slot, size_t count) {
  return static_cast<GtHit*>(impl_->slots[slot].gt_hits.ensure(std::max<size_t>(1, count) * sizeof(GtHit)));
}

void DeviceExecutor::wait(uint32_t slot) { check(cudaStreamSynchronize(impl_->slots[slot].stream), "render batch"); }

void DeviceExecutor::shade_into(uint32_t slot, const TraceBatch& b, const HitRecord* hits, const uint32_t* occluded,
                                float* irr, uint64_t* sig, uint32_t* sub, const uint32_t* out_index, uint32_t* klass,
                                const PvOut* pv, const uint32_t* static_occluded, float* irr_unshadowed) {
  if (b.num_requests == 0) return;
  launch_shade(impl_->view, b, hits, occluded, static_occluded, irr, irr_unshadowed, sig, sub, out_index, klass, pv,
               impl_->slots[slot].stream);
}

void DeviceExecutor::probe_into(uint32_t slot, const TraceBatch& b, const HitRecord* hits, const uint32_t* occluded,
                                ProbeOut* out) {
  if (b.num_requests == 0) return;
  probe_kernel<<<blocks(b.num_requests), kBlock, 0, impl_->slots[slot].stream>>>(impl_->view, b.poses, b.requests,
                                                                                  b.num_requests, hits, occluded, out);
  check(cudaGetLastError(), "probe kernel launch");
}

PvOut DeviceExecutor::pv_scratch(size_t n, size_t code_words, size_t table_words, void* stream) {
  Impl& s = *impl_;
  if (s.pv_totals.capacity() == 0) {  // first use: zeroed before any kernel adds to it
    s.pv_totals.ensure(2 * sizeof(unsigned long long));
    check(cudaMemset(s.pv_totals.as<void>(), 0, 2 * sizeof(unsigned long long)), "pv totals");
    check(cudaDeviceSynchronize(), "pv totals");
  }
  if (n * code_words * sizeof(uint32_t) > s.pv_codes.capacity() ||
      n * table_words * sizeof(uint32_t) > s.pv_table.capacity()) {
    check(cudaDeviceSynchronize(), "pv scratch");  // queued kernels may still write the buffers a grow frees
  }
  PvOut o;
  o.codes = static_cast<uint32_t*>(s.pv_codes.ensure(std::max<size_t>(1, n * code_words) * sizeof(uint32_t)));
  o.table = static_cast<uint32_t*>(s.pv_table.ensure(std::max<size_t>(1, n * table_words) * sizeof(uint32_t)));
  o.count = static_cast<uint32_t*>(s.pv_count.ensure(sizeof(uint32_t)));
  o.capacity = static_cast<uint32_t>(n);
  check(cudaMemsetAsync(o.count, 0, sizeof(uint32_t), static_cast<cudaStream_t>(stream)), "pv count");
  return o;
}

void DeviceExecutor::pv_scratch_add(uint64_t entry_bytes, void* stream) {
  Impl& s = *impl_;
  pv_add_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(s.pv_count.as<uint32_t>(), entry_bytes,
                                                                  s.pv_totals.as<unsigned long long>());
  check(cudaGetLastError(), "pv totals kernel launch");
}

void DeviceExecutor::pv_scratch_totals(uint64_t* totals) {
  Impl& s = *impl_;
  totals[0] = totals[1] = 0;
  if (s.pv_totals.capacity() == 0) return;
  const cudaStream_t stream = s.slots[0].stream;  // pv_scratch_add's (Renderer::render_into), done: the caller waited
  check(cudaMemcpyAsync(totals, s.pv_totals.as<void>(), 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream),
        "pv totals");
  check(cudaStreamSynchronize(stream), "pv totals");
}

void DeviceExecutor::census_into(uint32_t slot, const TraceBatch& b, const HitRecord* hits, const uint32_t* out_index,
                                 const uint32_t* klass) {
  Impl& s = *impl_;
  if (s.census.capacity() == 0) {  // first use: zeroed before any slot's kernel reads it
    s.census.ensure(kCensusWords * sizeof(unsigned long long));
    check(cudaMemset(s.census.as<void>(), 0, kCensusWords * sizeof(unsigned long long)), "census");
    check(cudaDeviceSynchronize(), "census");
  }
  if (b.num_requests == 0) return;
  census_kernel<<<blocks(b.num_requests), kBlock, 0, s.slots[slot].stream>>>(
      s.view, b.poses, b.requests, b.num_requests, b.subsamples, hits, out_index, klass,
      s.census.as<unsigned long long>());
  check(cudaGetLastError(), "census kernel launch");
}

void DeviceExecutor::census_totals(uint64_t* totals) {
  Impl& s = *impl_;
  std::fill(totals, totals + kCensusWords, uint64_t{0});
  if (s.census.capacity() == 0) return;
  const cudaStream_t stream = s.slots[0].stream;  // the caller waited for every slot
  check(cudaMemcpyAsync(totals, s.census.as<void>(), kCensusWords * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream),
        "census");
  check(cudaStreamSynchronize(stream), "census");
}

void DeviceExecutor::begin_window(uint16_t cam, const WindowClock& clock, void* stream) {
  Impl::WindowTable& w = impl_->window;
  w.cam = cam;
  w.clock = clock;
  w.n_slots = num_slots(clock);
  w.n_rows = 0;
  // Queued work of the previous window reads the rows only through render_into, which the caller's
  // stream joins on return: rewriting the index and the rows in that stream's order is safe.
  w.index.ensure(w.n_slots * sizeof(uint32_t));
  w.flag.ensure((w.n_slots + 1) * sizeof(uint32_t));
  w.position.ensure((w.n_slots + 2) * sizeof(uint32_t));  // the scan's n_slots + 1 values, then the bad slot
  check(cudaMemsetAsync(w.index.as<void>(), 0xFF, w.n_slots * sizeof(uint32_t), static_cast<cudaStream_t>(stream)),
        "window pose table reset");
  check(cudaMemsetAsync(w.position.as<uint32_t>() + w.n_slots + 1, 0, sizeof(uint32_t),
                        static_cast<cudaStream_t>(stream)),
        "window pose table reset");
}

void DeviceExecutor::assign_pose_rows(SampleRequest* requests, uint32_t n, void* stream_ptr) {
  Impl& s = *impl_;
  Impl::WindowTable& w = s.window;
  if (n == 0) return;
  if (w.n_slots == 0) throw std::logic_error("render: assign_pose_rows needs begin_window first (also after hold)");
  const auto stream = static_cast<cudaStream_t>(stream_ptr);
  uint32_t* index = w.index.as<uint32_t>();
  uint32_t* bad = w.position.as<uint32_t>() + w.n_slots + 1;
  mark_slots_kernel<<<blocks(n), kBlock, 0, stream>>>(requests, n, w.n_slots, index, bad);
  flag_new_kernel<<<blocks(w.n_slots + 1), kBlock, 0, stream>>>(index, w.n_slots, w.flag.as<uint32_t>());
  check(cudaGetLastError(), "pose slot kernels");
  size_t tmp_bytes = 0;
  check(cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, w.flag.as<uint32_t>(), w.position.as<uint32_t>(),
                                      w.n_slots + 1, stream),
        "pose slot scan");
  check(cub::DeviceScan::ExclusiveSum(w.scan_tmp.ensure(tmp_bytes), tmp_bytes, w.flag.as<uint32_t>(),
                                      w.position.as<uint32_t>(), w.n_slots + 1, stream),
        "pose slot scan");
  uint32_t counts[2] = {0, 0};  // new slots, the largest slot out of range
  check(cudaMemcpyAsync(counts, w.position.as<uint32_t>() + w.n_slots, sizeof(counts), cudaMemcpyDeviceToHost, stream),
        "pose slot count");
  check(cudaStreamSynchronize(stream), "pose slot count");
  if (counts[1] != 0) {  // refused as on the host: the table and the requests stay as they were
    unmark_slots_kernel<<<blocks(w.n_slots), kBlock, 0, stream>>>(index, w.n_slots, bad);
    check(cudaGetLastError(), "pose slot kernels");
    throw std::invalid_argument("render: pose slot " + std::to_string(counts[1]) + " out of range");
  }
  const uint32_t fresh = counts[0];
  if (fresh > 0) {
    const size_t row_bytes = sizeof(PoseRow) * static_cast<size_t>(s.num_geometries);
    const size_t light_bytes = sizeof(LightRow) * static_cast<size_t>(s.num_lights);
    if (w.n_rows + fresh > w.row_capacity) {  // grow, keeping the rows of this window
      const uint32_t cap = std::max(w.n_rows + fresh, std::min(w.n_slots, 2 * w.row_capacity + 64));
      Buffer rows, lights;
      rows.ensure(cap * row_bytes);
      lights.ensure(std::max<size_t>(1, cap * light_bytes));
      if (w.n_rows > 0) {
        check(cudaMemcpyAsync(rows.as<void>(), w.rows.as<void>(), w.n_rows * row_bytes, cudaMemcpyDeviceToDevice, stream),
              "pose rows");
        if (light_bytes > 0) {
          check(cudaMemcpyAsync(lights.as<void>(), w.light_rows.as<void>(), w.n_rows * light_bytes,
                                cudaMemcpyDeviceToDevice, stream),
                "pose rows");
        }
      }
      check(cudaStreamSynchronize(stream), "pose rows");
      w.rows.swap(rows);
      w.light_rows.swap(lights);
      w.row_capacity = cap;
    }
    new_rows_kernel<<<blocks(w.n_slots), kBlock, 0, stream>>>(s.pose_scene, w.clock, w.cam, w.n_slots,
                                                              w.position.as<uint32_t>(), w.n_rows, index,
                                                              w.rows.as<PoseRow>(), w.light_rows.as<LightRow>());
    check(cudaGetLastError(), "pose row kernel");
    w.n_rows += fresh;
  }
  map_rows_kernel<<<blocks(n), kBlock, 0, stream>>>(requests, n, index);
  check(cudaGetLastError(), "pose slot kernels");
}

PoseTableView DeviceExecutor::window_poses() const {
  const Impl::WindowTable& w = impl_->window;
  if (w.n_slots == 0) throw std::logic_error("render: render_into needs begin_window first (also after hold)");
  return {w.rows.as<PoseRow>(), w.light_rows.as<LightRow>(), impl_->num_geometries, impl_->num_lights};
}

const GtRequest* DeviceExecutor::gt_query_rows(uint32_t slot_index, uint16_t cam, const GtQuery* queries, uint32_t n,
                                               PoseTableView* poses) {
  Impl& s = *impl_;
  if (n == 0) {  // no launch: a grid of zero blocks is an error
    *poses = {nullptr, nullptr, s.num_geometries, s.num_lights};
    return nullptr;
  }
  Slot& slot = s.slots[slot_index];
  auto* flag = static_cast<uint32_t*>(s.gt_flag.ensure(n * sizeof(uint32_t)));
  auto* row = static_cast<uint32_t*>(s.gt_row.ensure(n * sizeof(uint32_t)));
  auto* rows = static_cast<PoseRow*>(s.gt_rows.ensure(n * sizeof(PoseRow) * static_cast<size_t>(s.num_geometries)));
  auto* lights = static_cast<LightRow*>(
      s.gt_light_rows.ensure(std::max<size_t>(1, n * sizeof(LightRow) * static_cast<size_t>(s.num_lights))));
  auto* requests = static_cast<GtRequest*>(slot.gt_requests.ensure(n * sizeof(GtRequest)));
  gt_flag_kernel<<<blocks(n), kBlock, 0, slot.stream>>>(queries, n, flag);
  check(cudaGetLastError(), "gt flag kernel");
  size_t tmp_bytes = 0;
  check(cub::DeviceScan::InclusiveSum(nullptr, tmp_bytes, flag, row, n, slot.stream), "gt row scan");
  check(cub::DeviceScan::InclusiveSum(s.gt_scan_tmp.ensure(tmp_bytes), tmp_bytes, flag, row, n, slot.stream),
        "gt row scan");
  gt_rows_kernel<<<blocks(n), kBlock, 0, slot.stream>>>(s.pose_scene, cam, queries, n, row, rows, lights, requests);
  check(cudaGetLastError(), "gt row kernel");
  *poses = {rows, lights, s.num_geometries, s.num_lights};
  return requests;
}

void DeviceExecutor::evaluate_gt_into(uint32_t slot, const GtRequest* requests, const GtHit* hits, uint32_t count,
                                      const PoseTableView& poses, GtSample* out, float* normal) {
  if (count == 0) return;
  gt_kernel<<<blocks(count), kBlock, 0, impl_->slots[slot].stream>>>(impl_->view, poses, requests, hits, count, out,
                                                                      normal);
  check(cudaGetLastError(), "gt kernel launch");
}

void DeviceExecutor::join(void* waiter, void* signaller) {
  check(cudaEventRecord(impl_->join_event, static_cast<cudaStream_t>(signaller)), "stream join");
  check(cudaStreamWaitEvent(static_cast<cudaStream_t>(waiter), impl_->join_event, 0), "stream join");
}

void DeviceExecutor::start_timing(void* stream) {
  check(cudaEventRecord(impl_->timing_epoch, static_cast<cudaStream_t>(stream)), "timer event");
}

uint32_t DeviceExecutor::timing_mark(uint32_t slot) {
  Impl& s = *impl_;
  if (s.timing_used == s.timing_events.size()) {
    cudaEvent_t e = nullptr;
    check(cudaEventCreate(&e), "cudaEventCreate");
    s.timing_events.push_back(e);
  }
  check(cudaEventRecord(s.timing_events[s.timing_used], s.slots[slot].stream), "timer event");
  return s.timing_used++;
}

double DeviceExecutor::timing_ms(uint32_t index) {
  float ms = 0.0f;
  check(cudaEventElapsedTime(&ms, impl_->timing_epoch, impl_->timing_events[index]), "timer event");
  return ms;
}

void DeviceExecutor::reset_timing() { impl_->timing_used = 0; }

void* DeviceExecutor::stage(StageBuffer id, void* host, size_t bytes, bool upload, void* stream) {
  void* p = impl_->staged[id].ensure(std::max<size_t>(1, bytes));
  if (upload && bytes > 0) {
    check(cudaMemcpyAsync(p, host, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)), "staging");
  }
  return p;
}

void DeviceExecutor::unstage(void* host, const void* staged, size_t bytes, void* stream) {
  if (bytes > 0) {
    check(cudaMemcpyAsync(host, staged, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)), "results");
  }
  check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "results");
}

}  // namespace balla::render
