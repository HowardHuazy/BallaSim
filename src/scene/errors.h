// The Python scene side's error types, so ballasim._scene raises what the Python raises: numpy.linalg.LinAlgError
// (a singular or not positive-definite system, a LAPACK routine that did not converge) and
// ballasim.traj.fit.TrajectoryError (input a trajectory cannot represent), both ValueErrors in Python, and
// ZeroDivisionError (a zero step or knot spacing, where the Python divides by it). Other std::invalid_argument errors
// become ValueError, other std::runtime_error RuntimeError.
#pragma once

#include <stdexcept>

namespace balla::scene {

struct LinAlgError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct TrajectoryError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

struct ZeroDivisionError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ballasim.scene.assets.geometry.AssetError (a ValueError): an asset that cannot be imported faithfully.
struct AssetError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

}  // namespace balla::scene
