// The single definition of the OptiX function table (see optix_function_table.h).
#include "backend/optix/optix_function_table.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <mutex>

namespace balla::optix {

OptixResult init_function_table() {
  static std::once_flag once;
  static OptixResult result = OPTIX_ERROR_UNKNOWN;
  std::call_once(once, [] { result = optixInit(); });
  return result;
}

}  // namespace balla::optix
