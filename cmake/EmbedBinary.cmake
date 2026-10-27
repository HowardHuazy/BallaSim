# balla_embed_binary(OUTPUT <file.cpp> INPUT <file-or-genex> SYMBOL <id> NAMESPACE <ns>
#                    [DEPENDS <targets>...])
# Generates a C++ source defining `extern const unsigned char <ns>::<SYMBOL>[]` and
# `extern const std::size_t <ns>::<SYMBOL>Size` with the bytes of INPUT. List OUTPUT as a source
# of the consuming target; the custom command then runs whenever INPUT changes.
function(balla_embed_binary)
  cmake_parse_arguments(ARG "" "INPUT;SYMBOL;NAMESPACE;OUTPUT" "DEPENDS" ${ARGN})
  add_custom_command(
    OUTPUT "${ARG_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${ARG_INPUT}" "-DOUTPUT=${ARG_OUTPUT}"
            "-DSYMBOL=${ARG_SYMBOL}" "-DNAMESPACE=${ARG_NAMESPACE}"
            -P "${PROJECT_SOURCE_DIR}/cmake/scripts/embed_binary.cmake"
    DEPENDS ${ARG_DEPENDS} "${ARG_INPUT}" "${PROJECT_SOURCE_DIR}/cmake/scripts/embed_binary.cmake"
    COMMENT "Embedding ${ARG_SYMBOL}"
    VERBATIM COMMAND_EXPAND_LISTS)
endfunction()
