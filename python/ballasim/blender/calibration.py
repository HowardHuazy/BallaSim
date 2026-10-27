"""Camera calibrations of a Blender exchange (docs/design/phase3/blender.md §3.4 ``cameras[].calibration``, §4.3).

The add-on records which calibration a camera uses: an EVIMO2 preset by name, or the text of a Kalibr camchain or an
OpenCV calibration file. This module only reads them into one record per camera; the C++ converter decides what the
record means (``_scene.blender_statics``: the calibration replaces Blender's model, CAM-CALIB compares the pinholes,
unsupported models are CAM-MODEL):

    {source, name, model, distortion, width, height, intrinsics: [fx, fy, cx, cy], coefficients, sensor,
     pixel_pitch_m, f_number, error}

``model`` is Kalibr's camera_model (``pinhole``, ``omni``, ``ds``, ``eucm``) or ``pinhole``; ``distortion`` is
Kalibr's distortion_model (``radtan``, ``equidistant``, ``fov``, ``none``), an OpenCV file's ``distortion_model``
(ROS: ``plumb_bob``, ``rational_polynomial`` → ``rational``, ``equidistant``) or, without one, its coefficient count
(4 or 5 ``radtan``, 8 ``rational``, 12 ``thin_prism``, 14 ``tilted``). A file this module cannot read gives a record
with ``error`` (CAM-CALIB), never an exception.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

SOURCES = ("preset", "kalibr", "opencv")
_COUNT_MODEL = {4: "radtan", 5: "radtan", 8: "rational", 12: "thin_prism", 14: "tilted"}
_ROS_MODEL = {"plumb_bob": "radtan", "radtan": "radtan", "rational_polynomial": "rational",
              "equidistant": "equidistant", "fisheye": "equidistant"}


class CalibrationError(ValueError):
    """A calibration file or preset this reader cannot use."""


def _yaml(text: str) -> Any:
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    # OpenCV's FileStorage writes a "%YAML:1.0" directive and "!!opencv-matrix" tags that YAML 1.2 loaders refuse.
    text = re.sub(r"^%YAML[: ]1\.0[^\n]*\n", "", text)
    text = text.replace("!!opencv-matrix", "")
    try:
        return YAML(typ="safe", pure=True).load(text)
    except YAMLError as e:
        raise CalibrationError(f"not YAML: {e}") from None


def _floats(v, what: str, n: int | None = None) -> list[float]:
    if isinstance(v, Mapping):   # an OpenCV matrix {rows, cols, dt, data} or a ROS matrix {rows, cols, data}
        v = v.get("data")
    if not isinstance(v, list) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        raise CalibrationError(f"{what} is not a list of numbers")
    if n is not None and len(v) != n:
        raise CalibrationError(f"{what} needs {n} numbers, has {len(v)}")
    return [float(x) for x in v]


def preset(name: str) -> dict:
    """An EVIMO2 camera preset (``ballasim.scene.authoring.CAMERA_PRESETS``) as a calibration record."""
    from ballasim.scene.authoring import CAMERA_PRESETS

    p = CAMERA_PRESETS.get(name)
    if p is None:
        raise CalibrationError(f"unknown preset {name!r} (known: {', '.join(sorted(CAMERA_PRESETS))})")
    return {"source": "preset", "name": name, "model": "pinhole",
            "distortion": "radtan" if p["model"] == "radtan" else "equidistant", "width": p["width"],
            "height": p["height"], "intrinsics": [p["fx"], p["fy"], p["cx"], p["cy"]],
            "coefficients": [float(x) for x in p["dist"]], "sensor": p["sensor"],
            "pixel_pitch_m": p["pixel_pitch_m"], "f_number": p["f_number"], "error": None}


def kalibr(text: str, camera: str | int | None = None) -> dict:
    """One camera of a Kalibr camchain (``cam0: {camera_model, intrinsics, distortion_model, distortion_coeffs,
    resolution}``); ``camera`` names it (``cam1`` or 1), default the only one."""
    doc = _yaml(text)
    if not isinstance(doc, Mapping):
        raise CalibrationError("not a Kalibr camchain (a mapping of cameras)")
    keys = [k for k, v in doc.items() if isinstance(v, Mapping) and "intrinsics" in v]
    if camera is None or camera == "":
        if len(keys) != 1:
            raise CalibrationError(f"name the camera: the camchain holds {', '.join(map(str, keys)) or 'none'}")
        camera = keys[0]
    key = f"cam{camera}" if isinstance(camera, int) or str(camera).isdigit() else str(camera)
    cam = doc.get(key)
    if not isinstance(cam, Mapping) or "intrinsics" not in cam:
        raise CalibrationError(f"no camera {key} (the camchain holds {', '.join(map(str, keys)) or 'none'})")
    res = _floats(cam.get("resolution"), f"{key}.resolution", 2)
    return {"source": "kalibr", "name": key, "model": str(cam.get("camera_model", "pinhole")),
            "distortion": str(cam.get("distortion_model", "none")), "width": int(res[0]), "height": int(res[1]),
            "intrinsics": _floats(cam["intrinsics"], f"{key}.intrinsics"),
            "coefficients": _floats(cam.get("distortion_coeffs", []), f"{key}.distortion_coeffs"),
            "sensor": None, "pixel_pitch_m": None, "f_number": None, "error": None}


def opencv(text: str, camera: str | None = None) -> dict:
    """An OpenCV calibration (FileStorage YAML: image_width, image_height, camera_matrix, distortion_coefficients; or a
    ROS camera_info YAML with distortion_model); ``camera`` names a sub-mapping holding one."""
    doc = _yaml(text)
    if camera:
        doc = doc.get(camera) if isinstance(doc, Mapping) else None
    if not isinstance(doc, Mapping):
        raise CalibrationError(f"no calibration mapping{f' {camera}' if camera else ''}")

    def first(*names):
        for n in names:
            if n in doc:
                return doc[n]
        raise CalibrationError(f"no {' / '.join(names)}")

    K = _floats(first("camera_matrix", "cameraMatrix", "K"), "camera_matrix", 9)
    if K[1] != 0.0 or K[3] != 0.0 or K[6] != 0.0 or K[7] != 0.0 or K[8] != 1.0:
        raise CalibrationError("camera_matrix is not [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (skew is not supported)")
    D = _floats(first("distortion_coefficients", "dist_coeffs", "distCoeffs", "D"), "distortion_coefficients")
    if "distortion_model" in doc:
        model = _ROS_MODEL.get(str(doc["distortion_model"]))
        if model is None:
            raise CalibrationError(f"distortion_model {doc['distortion_model']!r}")
    else:
        model = _COUNT_MODEL.get(len(D))
        if model is None:
            raise CalibrationError(f"{len(D)} distortion coefficients (OpenCV writes 4, 5, 8, 12 or 14)")
    return {"source": "opencv", "name": camera or "", "model": "pinhole", "distortion": model,
            "width": int(first("image_width", "width")), "height": int(first("image_height", "height")),
            "intrinsics": [K[0], K[4], K[2], K[5]], "coefficients": D, "sensor": None, "pixel_pitch_m": None,
            "f_number": None, "error": None}


def read_calibration(rec: Mapping | None) -> dict | None:
    """The calibration record of one ``cameras[]`` entry (None: the camera has none). Unreadable files and unknown
    presets give a record with ``error``, which the converter refuses as CAM-CALIB."""
    if not isinstance(rec, Mapping) or rec.get("source") in (None, "none"):
        return None
    source = rec.get("source")
    try:
        if source == "preset":
            return preset(str(rec.get("preset")))
        if source in ("kalibr", "opencv"):
            text = rec.get("file_text")
            if not isinstance(text, str):
                raise CalibrationError("file_text is missing")
            return kalibr(text, rec.get("camera")) if source == "kalibr" else opencv(text, rec.get("camera"))
        raise CalibrationError(f"unknown calibration source {source!r} (one of {', '.join(SOURCES)})")
    except (CalibrationError, KeyError, TypeError, ValueError) as e:
        return {"source": str(source), "name": str(rec.get("preset") or rec.get("camera") or ""), "error": str(e)}


def calibrations(doc: Mapping) -> list[dict | None]:
    """The calibration records of an exchange's ``cameras[]``, in order."""
    return [read_calibration(c.get("calibration") if isinstance(c, Mapping) else None)
            for c in doc.get("cameras") or []]
