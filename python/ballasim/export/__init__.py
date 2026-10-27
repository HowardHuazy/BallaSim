"""Synchronized outputs (docs/dev/IMU.md, docs/dev/EXPORT.md; docs/design/phase3/outputs.md): the per-camera IMU, a
post-process of a finished run in ``ballasim._imu`` (always built), and the format exporters in ``ballasim._export``
(optional: liblz4, libzstd and libbz2). Python here only reads runs, writes the JSON records and parses the command
line (``python -m ballasim imu`` and ``export``, :mod:`ballasim.export.cli`); refusals raise :class:`ExportError`.
"""

from .errors import ExportError

__all__ = ["ExportError"]
