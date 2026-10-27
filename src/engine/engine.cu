// The device instantiation of the engine (engine_impl.h; ENGINE_CPP §1.3): OptiX, the CUDA sensor and CUB.
#include "engine/engine_impl.h"

namespace balla::engine {

template class WindowEngine<Space::kDevice>;

std::unique_ptr<Engine> make_device_engine(render::SceneBundle bundle, EngineConfig config,
                                           const std::vector<std::optional<SensorSpec>>& sensors) {
  return std::make_unique<engine_detail::EngineOn<Space::kDevice>>(std::move(bundle), std::move(config), sensors);
}

}  // namespace balla::engine
