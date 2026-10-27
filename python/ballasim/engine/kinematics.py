"""Motion of the bundle's rig, bodies and lights over a window, and distances to the bodies' meshes (the planner's
inputs, docs/dev/ENGINE.md loop step 1).

Speed bounds are the rigorous per-segment hull bounds of the spline core (M2 Theorem B3), maximized over the
segments a window touches; "held" means exactly still over the closed window (inside a hold interval, or on
segments whose controls are all equal). ``MeshDistance`` gives a lower bound within 2 REL of the exact distance
from a point to a mesh (Morton clusters of CLUSTER triangles) and a cone from the point that holds every vertex.

Thin wrappers (ENGINE_CPP §1.9): SceneMotion, MeshDistance, BodyWindow and LightWindow are ``ballasim._engine``'s
(src/engine/motion.*), with the signatures of the NumPy module they replaced at ENGINE_CPP I5;
``SceneMotion.camera_window`` returns a :class:`WindowMotion`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ballasim._engine import BodyWindow, LightWindow, MeshDistance, SceneMotion

__all__ = ["BodyWindow", "LightWindow", "MeshDistance", "SceneMotion", "WindowMotion"]


@dataclass(frozen=True)
class WindowMotion:
    v: float          # bound on the speed of the trajectory's origin (m/s)
    omega: float      # bound on the angular speed (rad/s)
    held: bool        # exactly still over the window
