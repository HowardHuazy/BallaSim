"""Continuous-time trajectories: fitting measured poses and keyframes to C1 splines.

- :func:`fit_smoothing`: robust smoothing of measured poses (re-timed Vicon, gating, Cauchy IRLS,
  exact jerk penalties, lambda by GCV, holds), DESIGN §5.4 Phase 1 subset.
- :func:`fit_interpolating`: exact interpolation of authored keyframes with holds (M2 §C.7).
- :func:`retime_stamps`: Vicon stamp re-timing (G1).
- :func:`evaluate`: spline evaluation through ``ballasim._core`` (never re-implemented here).
"""

from .evaluate import evaluate, pose_matrices
from .fit import (
    FitResult,
    PoseSamples,
    SmoothingOptions,
    TrajectoryError,
    fit_interpolating,
    fit_smoothing,
)
from .retime import Retiming, retime_stamps

__all__ = ["FitResult", "PoseSamples", "Retiming", "SmoothingOptions", "TrajectoryError", "evaluate",
           "fit_interpolating", "fit_smoothing", "pose_matrices", "retime_stamps"]
