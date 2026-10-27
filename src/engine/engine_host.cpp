// The host instantiation of the engine (engine_impl.h; ENGINE_CPP §1.7): Embree, the CPU sensor and
// serial primitives; and make_engine, which picks the instantiation by the renderer's backend.
#include <algorithm>

#include "engine/engine_impl.h"

namespace balla::engine {

#if BALLA_ENGINE_HAVE_DEVICE
std::unique_ptr<Engine> make_device_engine(render::SceneBundle bundle, EngineConfig config,
                                           const std::vector<std::optional<SensorSpec>>& sensors);  // engine.cu
#endif

std::unique_ptr<Engine> make_engine(render::SceneBundle bundle, EngineConfig config,
                                    const std::vector<std::optional<SensorSpec>>& sensors) {
  // P2-F F11a: each simulated camera's sensor.nir_fraction, to the renderer (one SceneData for the cameras: it mixes
  // the albedo channels per camera and light group). A camera the run does not simulate is never queried: it takes the
  // first simulated camera's row, so that it alone does not make the scene two-channel.
  config.render.nir_fraction.resize(std::max(config.render.nir_fraction.size(), sensors.size()));
  const std::vector<double>* first = nullptr;
  for (size_t i = 0; i < sensors.size(); ++i) {
    if (!sensors[i]) continue;
    config.render.nir_fraction[i] = sensors[i]->nir_fraction;
    if (first == nullptr) first = &sensors[i]->nir_fraction;
  }
  for (size_t i = 0; first != nullptr && i < sensors.size(); ++i) {
    if (!sensors[i]) config.render.nir_fraction[i] = *first;
  }
  if (config.render.backend == render::BackendKind::kEmbree) {
    return std::make_unique<engine_detail::EngineOn<Space::kHost>>(std::move(bundle), std::move(config), sensors);
  }
#if BALLA_ENGINE_HAVE_DEVICE
  return make_device_engine(std::move(bundle), std::move(config), sensors);
#else
  throw std::invalid_argument("engine: this build has no device engine (CUDA and OptiX); use the Embree renderer");
#endif
}

bool device_available() {
#if BALLA_ENGINE_HAVE_DEVICE
  return sensor::sensor_cuda_available() && render::optix_available();
#else
  return false;
#endif
}

}  // namespace balla::engine
