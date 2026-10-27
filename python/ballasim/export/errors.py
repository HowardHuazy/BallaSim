"""The outputs' refusal (docs/dev/EXPORT.md)."""


class ExportError(ValueError):
    """An IMU simulation or an export refused its input or options; the message names the reason.

    ``ballasim._imu`` and ``ballasim._export`` raise it for their C++ refusals (``imu::Refused``,
    ``exporter::Refused``), and the Python side for its own. ``python -m ballasim`` prints it on one line and exits
    with 2 (``-v`` shows the traceback). Internal invariants stay plain ValueErrors with their tracebacks."""
