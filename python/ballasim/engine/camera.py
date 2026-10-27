"""Per-camera geometry of the planner: 8x8 tiles, pixel rays, image-speed factors, tile cones and the Morton order.

All quantities come from the core's fp64 camera models (``_core.RadtanCamera``, ``_core.KannalaBrandtCamera``),
restricted to their injective domains around the sub-sample support. For a camera-frame point at range R along the
unit ray d, the image speed is at most ||J(d)||_2 (|a| / R + |omega_C|), J(d) the projection Jacobian at the unit
ray; ``speed_factor`` is the maximum of ||J(d)||_2 over the pixel positions a tile's sub-samples reach and its 8
neighbours (DESIGN §6.7's prefactor in range form, exact for any camera model).

Thin wrapper (ENGINE_CPP §1.9): ``CameraGeometry`` is ``ballasim._engine``'s (src/engine/motion.*), with the
signature of the NumPy class it replaced at ENGINE_CPP I5; the tile edge and the sub-sample reach are plan.h's,
exposed as ``_engine.TILE`` and ``_engine.SUPPORT_PX``.
"""

from __future__ import annotations

from ballasim._engine import CameraGeometry

__all__ = ["CameraGeometry"]
