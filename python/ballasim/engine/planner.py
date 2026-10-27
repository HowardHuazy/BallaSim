"""Per-window tile planner (docs/dev/ENGINE.md loop step 1; DESIGN §6.7): for every 8x8 tile of a camera, a bound
v_hat on the image speed over the window from static content under camera motion, moving bodies, moving shadows
of static lights and moving lights, the lattice level l = clamp(ceil(log2(v_hat W / d_px)), l_min, l_max), and
STATIC tiles, which get no samples.

Thin wrappers (ENGINE_CPP §1.9): Planner, _Receivers (the shadow-receiver test), dilate_by_reach and
moving_light_edge_speed are ``ballasim._engine``'s (src/engine/motion.*, plan.h, on the host).
``Planner(geometry, cam_index, motion, d_px, refine_tol_log, min_dt_ns, crop, plan_pass).plan(t_w)`` returns a
:class:`TilePlan` (the accuracy keys of the same names; l_min and l_max follow from them in C++, as in the engine);
``plan_pass(t_ns)`` returns the camera-z depth of every pixel centre (NaN on a miss), taken as float32, or with the keyword
switch ``shadow_band`` (ENGINE_CPP I8) a tuple (depth, instance, shadow states): the instance id of
every centre-ray hit (uint32) and, per pixel and shadow light, the shadow state and the bits of the blocker's ray
parameter (uint32 pairs, flattened; render/shade/lights.h). The keyword switch ``band_seeds`` (P2-B B4a) takes the
same tuple, optionally with the carried samples' class words (uint32 per pixel) as a fourth item, and adds the band N's
tile map (the tiles within a seed's reach, bool) to the plan's stats as ``band``. The keyword ``mover_ao`` ('off',
'proxy' or 'traced'; P2-F F9b) marks the moving bodies' AO receivers from the depths alone (stats ``ao_movers`` and
``ao_pixels``). ``ballasim.run`` runs the same planner inside ``CameraEngine.step``, its per-pixel stage on the engine's
memory space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ballasim._engine import Planner, _Receivers, dilate_by_reach, moving_light_edge_speed, obliquity_budget

__all__ = ["Planner", "TilePlan", "_Receivers", "dilate_by_reach", "moving_light_edge_speed", "obliquity_budget"]


@dataclass
class TilePlan:
    static: np.ndarray            # (tiles_y, tiles_x) bool: no samples in this window
    level: np.ndarray             # (tiles_y, tiles_x) int8: lattice level of dynamic tiles, -1 where static
    v_hat: np.ndarray             # (tiles_y, tiles_x) float64: image-speed bound (px/s)
    stats: dict = field(default_factory=dict)
