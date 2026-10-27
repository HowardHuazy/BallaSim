// The generated coefficient tables of bsdf.h (bsdf_tables.inc, written by tools/bsdf_tables/gen_tables.cpp; the
// ctest bsdf_tables_regen requires a regeneration to be bitwise equal). Host only: include it where the tables are
// handed out (bindings, tests, and from F6 the device upload); bsdf.h itself takes BsdfTables views.
#pragma once

#include "render/shade/bsdf.h"
#include "render/shade/bspline_table.h"

namespace balla::render::bsdf {

namespace generated {
#include "render/shade/bsdf_tables.inc"
}  // namespace generated

inline BsdfTables host_tables() {
  return {generated::kSpecA, generated::kSpecB, generated::kLtcM00,
          generated::kLtcM02, generated::kLtcM20, generated::kLtcM22};
}

}  // namespace balla::render::bsdf
