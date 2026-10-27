// The process's OptiX entry points. optix_stubs.h calls every OptiX function through one global
// table, g_optixFunctionTable, which optixInit() fills from the driver. The table must be defined
// exactly once per binary (optix_function_table_definition.h), so it lives in its own library,
// balla_optix_runtime, which every OptiX user links (balla_optix and the renderer's device
// backend) instead of defining a copy or borrowing another library's.
#pragma once

#include <optix_types.h>

namespace balla::optix {

// Loads the driver's OptiX entry points into the function table, once per process and thread-safe;
// later calls return the first call's result (OPTIX_SUCCESS, or the reason the driver lacks OptiX).
OptixResult init_function_table();

}  // namespace balla::optix
