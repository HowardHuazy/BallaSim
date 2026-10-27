// The refusals of the outputs' modules (ballasim._imu, ballasim._export; docs/dev/EXPORT.md): a translator from the
// C++ type E to ballasim.export.errors.ExportError (a ValueError) with E's message. Other exceptions are left to the
// next translator.
#pragma once

#include <nanobind/nanobind.h>

#include <exception>

namespace balla::bindings {

// Call once in the module's NB_MODULE body. The translator holds one reference to the class for the life of the
// process (translators outlive the module object).
template <class E>
void register_export_error() {
  namespace nb = nanobind;
  nb::object type = nb::module_::import_("ballasim.export.errors").attr("ExportError");
  PyObject* cls = type.release().ptr();
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void* payload) {
        try {
          std::rethrow_exception(p);
        } catch (const E& e) {
          PyErr_SetString(static_cast<PyObject*>(payload), e.what());
        }
      },
      cls);
}

}  // namespace balla::bindings
