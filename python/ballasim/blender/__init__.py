"""Blender scenes (docs/design/phase3/blender.md): the engine side of the BallaSim Blender add-on.

The add-on (a separate, GPL-3.0-or-later repository) writes an exchange directory ``scene.bsx/`` with the scene's raw
RNA values and arrays; it never imports this package. :mod:`.exchange` reads and checks that directory (versions,
unknown keys, the content digest); :mod:`.convert` turns its statics into C1 terms through ``ballasim._scene`` (C++:
``src/scene/blender/``), computes the bundle cache key and loads a converted bundle as a run's scene source.
"""
