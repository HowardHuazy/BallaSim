// NumPy array helpers shared by the extension modules.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace balla::bindings {

namespace nb = nanobind;

// Read-only, C-contiguous CPU input arrays; nanobind converts other dtypes and layouts.
template <class T, ssize_t... Shape>
using In = nb::ndarray<const T, nb::shape<Shape...>, nb::c_contig, nb::device::cpu>;

// Read-only CPU input arrays of any layout whose dtype must match exactly. Bind them with
// nb::arg(...).noconvert(): int64 ns times and uint32 RNG inputs must never be cast implicitly
// from floats or other integer types, which would truncate or wrap silently (DESIGN §5.1).
template <class T, ssize_t... Shape>
using ExactIn = nb::ndarray<const T, nb::shape<Shape...>, nb::device::cpu>;

template <class T>
using Out = nb::ndarray<nb::numpy, T>;

// Hands a vector to NumPy without copying; the capsule frees it with the array.
template <class T>
Out<T> to_numpy(std::vector<T>&& data, std::initializer_list<size_t> shape) {
  size_t count = 1;
  for (const size_t s : shape) count *= s;
  if (count > data.size()) throw std::logic_error("to_numpy: the shape exceeds the data");
  auto* owned = new std::vector<T>(std::move(data));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return Out<T>(owned->data(), shape, owner);
}

// Mesh indices (T, 3) as an owned copy, each checked < num_vertices (std::invalid_argument otherwise). The copy
// follows the strides (any layout) and is made while the GIL is held, so only checked indices reach a query that
// runs with the GIL released, even if the caller writes to its array meanwhile.
inline std::vector<uint32_t> checked_indices(const ExactIn<uint32_t, -1, 3>& indices, size_t num_vertices) {
  const size_t n = indices.shape(0);
  std::vector<uint32_t> out(3 * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t k = 0; k < 3; ++k) {
      out[3 * t + k] = indices(t, k);
      if (out[3 * t + k] >= num_vertices) throw std::invalid_argument("vertex index out of range");
    }
  }
  return out;
}

// uint8 0/1 flags as a NumPy bool array.
inline Out<bool> to_numpy_bool(std::vector<uint8_t>&& flags) {
  auto* owned = new std::vector<uint8_t>(std::move(flags));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<uint8_t>*>(p); });
  return Out<bool>(owned->data(), {owned->size()}, owner);
}

}  // namespace balla::bindings
