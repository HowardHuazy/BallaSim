// Factory of the GPU backend (OptiX tracer + CUDA executor sharing one stream). Implemented only
// in builds with OptiX (BALLA_RENDER_HAVE_OPTIX); renderer.cpp calls it behind that macro.
#pragma once

#include <memory>

#include "render/executor.h"
#include "render/tracer.h"

namespace balla::render {

struct DeviceBackend {
  std::unique_ptr<Executor> executor;
  std::unique_ptr<Tracer> tracer;
};

// Throws std::runtime_error without a CUDA device or an OptiX-capable driver.
DeviceBackend make_optix_backend();

// True if a CUDA device is visible and optixInit() succeeds.
bool optix_device_available();

}  // namespace balla::render
