"""Scene description: the C1 bundle, asset import, YAML authoring and reference ray casting.

Submodules: :mod:`.bundle` (C1 dataclasses, validation, persistence), :mod:`.assets` (glTF/OBJ/PLY
import with a content-hash cache), :mod:`.authoring` (scene YAML -> bundle), :mod:`.raycast`
(dense depth/instance by Embree) and :mod:`.transforms` (rotation and rigid-transform helpers).
"""

from .bundle import (
    Body,
    BundleError,
    Camera,
    Light,
    LightGroup,
    Material,
    Mesh,
    SceneBundle,
    Trajectory,
)

__all__ = ["Body", "BundleError", "Camera", "Light", "LightGroup", "Material", "Mesh", "SceneBundle", "Trajectory"]
