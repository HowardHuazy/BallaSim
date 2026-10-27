"""Window sampling (docs/dev/ENGINE.md loop steps 2-4): the dyadic lattice, refinement by signature and by the
midpoint test, the sample budget, the fixed-rate grid and the sensor's sample runs.

Thin wrappers (ENGINE_CPP §1.9): waveform_range, MidpointTest, level_cap, sample_window and fixed_rate_window are
``ballasim._engine``'s, which runs them on the host instantiation of the engine (src/engine/sampling.h,
window.cpp) with the signatures of the NumPy module they replaced at ENGINE_CPP I5. ``sample_window`` and
``fixed_rate_window`` take a Python callable as the renderer (:data:`RenderFn`) and return a :class:`WindowRuns`;
requests go to it in slot order (pixel-major in the given order, then time; ENGINE_CPP I1 notes), so ``rank`` is
only checked. ``ballasim.run`` runs the same code on the engine's own memory space, inside ``CameraEngine.step``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ballasim._engine import MidpointTest, fixed_rate_window, level_cap, sample_window, waveform_range

__all__ = ["MidpointTest", "RenderFn", "WindowRuns", "fixed_rate_window", "level_cap", "sample_window",
           "waveform_range"]

#: render(pixel_ids u32, t_ns i64) -> (irradiance f32 (n, G), signature u64 (n,)), requests in the given order.
RenderFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class WindowRuns:
    """Sample runs of one window, ready for ``SensorModel.advance(..., offsets=...)``."""

    pixel_ids: np.ndarray         # uint32 (n,)
    t_ns: np.ndarray              # int64 (S,), strictly increasing within each run
    irradiance: np.ndarray        # float32 (S, G)
    offsets: np.ndarray           # int64 (n + 1,)
    last_irradiance: np.ndarray   # float32 (n, G): each pixel's sample at t_w + W
    last_signature: np.ndarray    # uint64 (n,)
    stats: dict = field(default_factory=dict)
