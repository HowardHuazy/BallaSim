"""EVIMO2 v2 replay: read a sequence, build its C1 SceneBundle, validate it against the release.

- :func:`build_bundle` (:mod:`.adapter`): sequence directory -> SceneBundle with fitted trajectories; :func:`relight`
  rebuilds its lights for other lighting values (:class:`Evimo2Lighting`) without a rebuild.
- :mod:`.dataset`: sequence metadata, GT archives and the object catalogue (read-only).
- :mod:`.clocks`: per-camera clock offsets.
- :mod:`.validate`: P1-M1 depth/mask reproduction (``python -m ballasim.evimo2.validate``).
"""

from .adapter import Evimo2Build, Evimo2Lighting, Evimo2Options, build_bundle, relight

__all__ = ["Evimo2Build", "Evimo2Lighting", "Evimo2Options", "build_bundle", "relight"]
