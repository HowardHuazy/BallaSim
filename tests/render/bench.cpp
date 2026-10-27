// Renderer throughput on a synthetic EVIMO2-matched scene (G6 §2.1): a displaced, textured
// tabletop of ~545 k triangles in a room box plus 8 movers of ~217 k triangles each (2.28 M in
// total), a 640 x 480 radtan camera with EVIMO2-like intrinsics on a moving rig, N = 64 sub-samples,
// one shadow-casting point light plus ambient. Reports G sub-samples/s end to end (pose table, ray
// generation, multi-trace, shadow rays, cluster shading, signature, host transfers).
//
// Request sets: "uniform" (random 8 x 8 tiles over the image) and "movers" (pixels inside the
// movers' projected bounding spheres, the realistic worst case, G6 §2.2); times on a 2^13 ns
// lattice inside one window (shared pose slots, as in the integrator's level-synchronous rounds)
// or, with "unique", a distinct time per pixel-sample.
//
//   balla_render_bench [--smoke] [--backend embree|optix|all] [--threads n] [--requests n]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "core/traj/spline_build.h"
#include "render/renderer.h"
#include "scenes.h"

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;

namespace {

constexpr int64_t kT0 = 2'000'000'000, kT1 = 3'000'000'000;
constexpr int kW = 640, kH = 480;
constexpr double kF = 556.6;

// Smooth value noise for displacement and texture.
double smooth_noise(double x, double y, uint32_t seed) {
  double v = 0.0, amp = 1.0, freq = 1.0;
  for (int o = 0; o < 4; ++o) {
    v += amp * std::sin(freq * (1.7 * x + 0.3 * seed) + 2.1 * std::cos(freq * (1.3 * y - 0.7 * seed)));
    amp *= 0.5;
    freq *= 2.1;
  }
  return v;
}

Mesh terrain(int n, double sx, double sy, double amp) {
  Mesh m;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      const double u = static_cast<double>(i) / n, v = static_cast<double>(j) / n;
      const double x = (u - 0.5) * sx, y = (v - 0.5) * sy;
      m.positions.insert(m.positions.end(), {static_cast<float>(x), static_cast<float>(y),
                                             static_cast<float>(amp * smooth_noise(8 * x, 8 * y, 1))});
      m.uvs.insert(m.uvs.end(), {static_cast<float>(u), static_cast<float>(v)});
    }
  }
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const uint32_t a = static_cast<uint32_t>(j * (n + 1) + i), b = a + 1, c = a + static_cast<uint32_t>(n) + 1, d = c + 1;
      m.indices.insert(m.indices.end(), {a, b, d, a, d, c});
    }
  }
  return m;
}

// Displaced UV sphere with per-vertex normals: n rings x n segments, ~2 n^2 triangles.
Mesh blob(int n, double radius, uint32_t seed) {
  Mesh m;
  for (int j = 0; j <= n; ++j) {
    const double th = 3.14159265358979 * j / n;
    for (int i = 0; i <= n; ++i) {
      const double ph = 2 * 3.14159265358979 * i / n;
      const Vec3<double> dir{std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};
      const double r = radius * (1.0 + 0.15 * smooth_noise(3 * dir.x, 3 * dir.y + dir.z, seed));
      m.positions.insert(m.positions.end(), {static_cast<float>(r * dir.x), static_cast<float>(r * dir.y),
                                             static_cast<float>(r * dir.z)});
      m.normals.insert(m.normals.end(), {static_cast<float>(dir.x), static_cast<float>(dir.y), static_cast<float>(dir.z)});
      m.uvs.insert(m.uvs.end(), {static_cast<float>(static_cast<double>(i) / n), static_cast<float>(static_cast<double>(j) / n)});
    }
  }
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const uint32_t a = static_cast<uint32_t>(j * (n + 1) + i), b = a + 1, c = a + static_cast<uint32_t>(n) + 1, d = c + 1;
      if (j > 0) m.indices.insert(m.indices.end(), {a, b, d});
      if (j < n - 1) m.indices.insert(m.indices.end(), {a, d, c});
    }
  }
  return m;
}

std::vector<uint8_t> procedural_texture(int size, uint32_t seed) {
  std::vector<uint8_t> t(static_cast<size_t>(size) * size * 3);
  for (int j = 0; j < size; ++j) {
    for (int i = 0; i < size; ++i) {
      const double v = 0.5 + 0.25 * smooth_noise(20.0 * i / size, 20.0 * j / size, seed) +
                       (((i / 32 + j / 32) % 2) ? 0.15 : -0.15);
      for (int c = 0; c < 3; ++c) {
        t[(static_cast<size_t>(j) * size + i) * 3 + c] =
            static_cast<uint8_t>(std::clamp(255.0 * (v + 0.05 * c), 0.0, 255.0));
      }
    }
  }
  return t;
}

struct Scene {
  SceneBundle bundle;
  std::vector<double> radius;  // mover bounding radii (approximate, for request generation)
};

Scene make_scene(bool smoke, bool shadows) {
  Scene sc;
  SceneBundle& s = sc.bundle;
  s.light_groups = {"default"};
  // The world frame is the camera's initial frame; the rig drifts and turns slowly (hand-held).
  s.trajectories.push_back(moving_trajectory(kT0, kT1, quat_identity<double>(), {0.0, 0.0, 0.0},
                                             {0.12, 0.03, 0.05}, {0.15, 0.25, 0.05}));
  Camera cam = pinhole(kW, kH, kF, 0);
  cam.dist = {-0.39, 0.20, 0.0005, -0.0003};
  s.cameras.push_back(cam);
  // Tabletop (~545 k triangles at full size) filling the background, tilted away from the camera,
  // inside a 12-triangle room.
  Body table = make_body(1, terrain(smoke ? 60 : 522, 1.6, 1.2, 0.01), {0.8f, 0.75f, 0.7f});
  table.materials[0].texture = procedural_texture(smoke ? 128 : 1024, 3);
  table.materials[0].texture_width = table.materials[0].texture_height = smoke ? 128 : 1024;
  table.materials[0].texture_channels = 3;
  table.T_world_body = {1, 0, 0, 0, 0, std::cos(0.35), -std::sin(0.35), 0.0, 0, std::sin(0.35), std::cos(0.35), 1.2, 0, 0, 0, 1};
  s.bodies.push_back(table);
  Body walls = make_body(2, box_mesh(3.5f, 3.5f, 3.5f), {0.6f, 0.6f, 0.6f});  // seen from inside (two-sided)
  walls.T_world_body = translation(0, 0, 1.0);
  s.bodies.push_back(walls);
  // Eight movers over the table.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (uint32_t i = 0; i < 8; ++i) {
    const double r = 0.05 + 0.03 * std::abs(u(rng));
    const Vec3<double> p0{0.25 * u(rng), 0.15 * u(rng) - 0.05, 0.8 + 0.15 * u(rng)};
    const Vec3<double> v{0.15 * u(rng), 0.1 * u(rng), 0.05 * u(rng)};
    const Vec3<double> w{3 * u(rng), 3 * u(rng), 3 * u(rng)};
    s.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{u(rng), u(rng), u(rng)}), p0, v, w));
    Body b = make_body(10 + i, blob(smoke ? 24 : 330, r, i), {0.4f + 0.07f * static_cast<float>(i), 0.5f, 0.45f});
    if (i % 2 == 0) {
      b.materials[0].texture = procedural_texture(smoke ? 64 : 512, 10 + i);
      b.materials[0].texture_width = b.materials[0].texture_height = smoke ? 64 : 512;
      b.materials[0].texture_channels = 3;
    }
    b.trajectory = static_cast<int32_t>(i) + 1;
    s.bodies.push_back(b);
    sc.radius.push_back(1.2 * r);
  }
  s.lights.push_back(point_light({0.4, -0.9, 0.3}, 2.0f, shadows));
  s.lights.push_back(ambient_light(0.15f));
  return sc;
}

struct Pose {
  Mat3<double> r;
  Vec3<double> p;
};

Pose eval_pose(const Trajectory& tr, int64_t t) {
  std::vector<Quat<double>> qs;
  std::vector<Vec3<double>> ps;
  for (size_t c = 0; c < tr.pos.size() / 3; ++c) {
    qs.push_back({tr.rot[4 * c], tr.rot[4 * c + 1], tr.rot[4 * c + 2], tr.rot[4 * c + 3]});
    ps.push_back({tr.pos[3 * c], tr.pos[3 * c + 1], tr.pos[3 * c + 2]});
  }
  const auto nodes = traj::build_spline_nodes(qs, ps, {0, 0, 0});
  const traj::SplineView<double> v{nodes.data(), static_cast<int32_t>(nodes.size()) - 5,
                                   static_cast<int32_t>(tr.h_ns), 1e9 / static_cast<double>(tr.h_ns)};
  traj::SegmentTime st{};
  traj::locate(t, tr.t0_ns, v.h_ns, v.num_segments, &st);
  const auto k = traj::evaluate<6>(v, st);
  return {to_matrix(k.q), k.p};
}

struct Requests {
  std::vector<uint16_t> cam;
  std::vector<uint32_t> pixel;
  std::vector<int64_t> t;
};

// Interleaves the bits of x and y (Morton code of 8 x 8 tiles).
uint32_t morton(uint32_t x, uint32_t y) {
  uint32_t m = 0;
  for (int b = 0; b < 16; ++b) m |= ((x >> b) & 1u) << (2 * b) | ((y >> b) & 1u) << (2 * b + 1);
  return m;
}

// mode: 0 uniform tiles, 1 movers. unique: a distinct time per request. Requests are ordered as
// the integrator issues them inside a launch (DESIGN §6.9): by time, then tile Morton code, then
// pixel. Each seed draws its times in its own window, so repeated calls never reuse cached poses.
Requests make_requests(const Scene& sc, size_t n, int mode, bool unique, uint32_t seed) {
  Requests q;
  std::mt19937 rng(seed);
  const int64_t window = ((kT0 + 100'000'000) & ~((int64_t{1} << 23) - 1)) + (int64_t{seed % 64} << 23);
  std::uniform_int_distribution<int64_t> node(0, 1023), any(0, (int64_t{1} << 23) - 1);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  size_t idle = 0;  // consecutive draws that added nothing (movers out of view)
  while (q.cam.size() < n) {
    if (++idle > 10000) {
      std::fprintf(stderr, "bench: the movers are not visible; check the scene\n");
      std::exit(1);
    }
    const size_t before = q.cam.size();
    const int64_t t = window + (unique ? any(rng) : node(rng) * 8192);
    if (mode == 0) {
      const int tx = static_cast<int>(unit(rng) * (kW / 8)), ty = static_cast<int>(unit(rng) * (kH / 8));
      for (int k = 0; k < 64 && q.cam.size() < n; ++k) {
        q.cam.push_back(0);
        q.pixel.push_back(static_cast<uint32_t>((ty * 8 + k / 8) * kW + tx * 8 + k % 8));
        q.t.push_back(unique ? window + any(rng) : t);
      }
    } else {
      const Pose cam = eval_pose(sc.bundle.trajectories[0], t);
      const size_t m = static_cast<size_t>(unit(rng) * 8);
      const Pose b = eval_pose(sc.bundle.trajectories[m + 1], t);
      const Vec3<double> d = b.p - cam.p;
      const Vec3<double> c{cam.r.m[0][0] * d.x + cam.r.m[1][0] * d.y + cam.r.m[2][0] * d.z,
                           cam.r.m[0][1] * d.x + cam.r.m[1][1] * d.y + cam.r.m[2][1] * d.z,
                           cam.r.m[0][2] * d.x + cam.r.m[1][2] * d.y + cam.r.m[2][2] * d.z};
      if (c.z < 0.1) continue;
      const double cx = 319.5 + kF * c.x / c.z, cy = 239.5 + kF * c.y / c.z, rp = kF * sc.radius[m] / c.z;
      for (int k = 0; k < 64 && q.cam.size() < n; ++k) {
        const double a = 6.2831853 * unit(rng), rr = rp * std::sqrt(unit(rng));
        const int x = static_cast<int>(cx + rr * std::cos(a)), y = static_cast<int>(cy + rr * std::sin(a));
        if (x < 0 || y < 0 || x >= kW || y >= kH) continue;
        q.cam.push_back(0);
        q.pixel.push_back(static_cast<uint32_t>(y * kW + x));
        q.t.push_back(unique ? window + any(rng) : t);
      }
    }
    if (q.cam.size() > before) idle = 0;
  }
  std::vector<size_t> order(q.cam.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  auto key = [&](size_t i) {
    const uint32_t x = q.pixel[i] % kW, y = q.pixel[i] / kW;
    return std::tuple(q.t[i], morton(x / 8, y / 8), q.pixel[i]);
  };
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return key(a) < key(b); });
  Requests sorted;
  for (size_t i : order) {
    sorted.cam.push_back(q.cam[i]);
    sorted.pixel.push_back(q.pixel[i]);
    sorted.t.push_back(q.t[i]);
  }
  return sorted;
}

// G samples/s over calls with fresh request sets (new windows, nothing cached); the first set
// warms up (GPU clocks, allocations).
double measure(Renderer& r, const std::vector<Requests>& sets) {
  r.render(sets[0].cam, sets[0].pixel, sets[0].t);
  size_t samples = 0;
  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 1; i < sets.size(); ++i) {
    r.render(sets[i].cam, sets[i].pixel, sets[i].t);
    samples += sets[i].cam.size() * r.options().scene.n_subsamples;
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return static_cast<double>(samples) / s * 1e-9;
}

}  // namespace

int main(int argc, char** argv) {
  bool smoke = false;
  std::string backend = "all";
  int threads = 6;
  size_t requests = size_t{1} << 17;  // 8.4 M sub-sample rays per call
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--smoke")) smoke = true;
    else if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--requests") && i + 1 < argc) requests = std::strtoull(argv[++i], nullptr, 10);
  }
  if (smoke) requests = 4096;
  std::vector<BackendKind> kinds;
  if (backend == "all" || backend == "embree") kinds.push_back(BackendKind::kEmbree);
  if ((backend == "all" || backend == "optix") && optix_available()) kinds.push_back(BackendKind::kOptix);
  for (bool shadows : {true, false}) {
    const Scene sc = make_scene(smoke, shadows);
    size_t tris = 0;
    for (const Body& b : sc.bundle.bodies) tris += b.mesh.indices.size() / 3;
    std::printf("scene: %zu triangles, 8 movers, N = 64, %s\n", tris, shadows ? "point light with shadows + ambient" : "no shadows");
    for (BackendKind k : kinds) {
      RendererOptions o;
      o.backend = k;
      o.num_threads = threads;
      Renderer r(sc.bundle, o);
      const auto t0 = std::chrono::steady_clock::now();
      r.build();
      const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  %-6s build %.2f s\n", backend_name(k).c_str(), build_s);
      for (int mode : {0, 1}) {
        for (bool unique : {false, true}) {
          const int repeats = smoke ? 1 : (k == BackendKind::kOptix ? 8 : 1);
          std::vector<Requests> sets;
          for (int i = 0; i <= repeats; ++i) {
            sets.push_back(make_requests(sc, requests, mode, unique, 100 + static_cast<uint32_t>(4 * i + mode)));
          }
          const double g = measure(r, sets);
          std::printf("  %-6s %-7s %-13s %8.3f G samples/s\n", backend_name(k).c_str(), mode == 0 ? "uniform" : "movers",
                      unique ? "unique times" : "lattice times", g);
        }
      }
    }
  }
  return 0;
}
