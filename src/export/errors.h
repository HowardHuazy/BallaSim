// The exporters' refusal (docs/dev/EXPORT.md): invalid input or options, named in the message. ballasim._export
// translates it to ballasim.export.errors.ExportError (a ValueError); every other exception keeps its own type.
#pragma once

#include <stdexcept>

namespace balla::exporter {  // 'export' is a C++ keyword

struct Refused : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace balla::exporter
