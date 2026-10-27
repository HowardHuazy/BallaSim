// CUDA execution of the sensor kernel (pixel.h) in fp32: one thread per pixel, the same BALLA_HD code
// as the CPU path, so the GPU and the CPU fp32 kernel produce bitwise identical states and events
// under the floating-point policy (DESIGN §4.6, §12.1). Host-only interface (no CUDA headers).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "sensor/launch.h"
#include "sensor/maps.h"
#include "sensor/pixel.h"
#include "sensor/sensor_model.h"

namespace balla::sensor {

// Number of visible CUDA devices (0 without a device or a usable driver).
int sensor_cuda_device_count();

class CudaSensorKernel {
 public:
  // Copies the maps and the fp32 noise tables to the device; kc.tables is re-pointed at them.
  CudaSensorKernel(const KernelConstants<float>& kc, const std::vector<PixelMap>& maps,
                   const std::vector<float>& tables);
  ~CudaSensorKernel();
  CudaSensorKernel(const CudaSensorKernel&) = delete;
  CudaSensorKernel& operator=(const CudaSensorKernel&) = delete;

  // The pixel states, and the high-pass filter's side states while its kernels run (enable_hpf; hpf empty otherwise).
  void upload_states(const std::vector<PixelState>& states, const std::vector<HpfState>& hpf);
  void download_states(std::vector<PixelState>& states) const;
  void download_hpf(std::vector<HpfState>& hpf) const;
  // Runs the high-pass filter's kernels from now on (pixel.h kHpf): kc's filter constants, the per-pixel rates hpf_b
  // (empty: kc.hpf_b for every pixel) and the side states (zero until the next upload_states).
  void enable_hpf(const KernelConstants<float>& kc, const std::vector<float>& hpf_b);

  // A region-of-interest model (SensorModel's roi): the frame pixel id of each of its pixels (KernelConstants::
  // pixel_ids, a device copy), which keys the draws and names the events.
  void set_pixel_ids(const std::vector<uint32_t>& ids);

  // Replaces the launch policy (and re-allocates the event buffer at its capacity).
  void set_policy(const GpuLaunchPolicy& policy);
  // SensorModel::set_photometry: the scale and the clamp of kc, and the maps (their gains).
  void set_photometry(const KernelConstants<float>& kc, const std::vector<PixelMap>& maps);
  // SensorModel::set_glare: the next call's glare ramp.
  void set_glare(const GlareRamp<float>& glare);
  // Veiling glare (SensorModel::held_frame): the input each pixel holds after a call on these runs (its run's last
  // sample, else its held input; no runs: the held inputs), (pixels x groups) on the host, after `stream`.
  std::vector<float> end_frame(const DeviceRuns& runs, void* stream);

  // Advances every pixel from t_state to t_end on device-resident runs (SensorModel::advance_device;
  // first the pixel ids are checked, and with check_inputs the runs, counting the non-finite and clamped
  // samples they take). The events accumulate in the kernel's device buffer, unsorted; the work counters
  // are added to stats.
  DeviceEvents advance_device(const DeviceRuns& runs, int64_t t_state, int64_t t_end, void* stream, bool check_inputs,
                              KernelStats& stats, GpuLaunchStats& launch_stats);
  std::vector<RawEvent> download(const DeviceEvents& events) const;

  // The same on host runs (SensorModel::advance_runs, validated there): uploads them and returns the
  // events unsorted.
  std::vector<RawEvent> advance(const uint32_t* pixel_ids, size_t n, const int64_t* t, const float* irr,
                                size_t n_samples, const int64_t* row_start, const int32_t* row_count, int64_t t_state,
                                int64_t t_end, KernelStats& stats, GpuLaunchStats& launch_stats);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::sensor
