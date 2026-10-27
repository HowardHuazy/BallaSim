# The browser UI's event wire (docs/design/phase3/ui.md §3.10): LOD order, BSEV packing, gathers and rate bins.
# Host-only integer code, linked by _core (ballasim._core.wire, src/bindings/core_wire.cpp) and balla_core_tests.
add_library(balla_wire STATIC wire.cpp)
target_link_libraries(balla_wire PUBLIC balla_core)
balla_warnings(balla_wire)
balla_lint_fp_flags(balla_wire)
