"""Keyframes of the analytic motions in scenes/lab_tabletop.yaml.

Prints YAML keyframe blocks for the movers whose motion follows a formula: the cracker box (slide,
then a tip about its far bottom edge), the soup can (rolling without slipping), the knife (flick),
the airplane (arc with roll) and the police car (circular drive). Paste the output into the scene
file after changing a motion; the scene file stays the single source the loader reads.

Run: python scenes/lab_tabletop_keys.py [--assets DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ballasim.scene.assets import load_asset

DESK_TOP = 0.788   # m, height of the Poly Haven metal_office_desk


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x * x)


def key(t, p, r, hold=None):
    q = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
    q = q if q[0] >= 0 else -q
    s = (f"      - {{t_s: {t:.3f}, t_m: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}], "
         f"q_wxyz: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    return s + (f", hold_s: {hold:.3f}}}" if hold else "}")


def rz(deg):
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def cracker(box_thickness):
    """Slide 1.5-2.3 s (+20 deg yaw), rest, tip 90 deg over the far bottom edge in 3.0-3.35 s."""
    p0, p1 = np.array([-0.55, -0.08, DESK_TOP]), np.array([-0.32, -0.02, DESK_TOP])
    out = [key(0.0, p0, rz(90), 1.5)]
    for t in np.arange(1.6, 2.25, 0.1):
        s = smoothstep((t - 1.5) / 0.8)
        out.append(key(t, p0 + (p1 - p0) * s, rz(90 + 20 * s)))
    r1 = rz(110)
    out.append(key(2.3, p1, r1, 0.7))
    thin = r1 @ np.array([1.0, 0.0, 0.0])                      # box thickness axis (body x) in the world
    pivot = p1 + 0.5 * box_thickness * thin
    axis = np.cross([0.0, 0.0, 1.0], thin)
    axis /= np.linalg.norm(axis)
    for t in np.arange(3.025, 3.351, 0.025):
        ra = Rotation.from_rotvec(np.radians(90.0 * smoothstep((t - 3.0) / 0.35)) * axis).as_matrix()
        out.append(key(t, pivot + ra @ (p1 - pivot), ra @ r1, (8.0 - 3.35) if t > 3.349 else None))
    return out


def soup_can(radius):
    """Lying on its side (axis along world y), rolling without slipping from x = 0.12 to -0.24."""
    lying = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    x0, x1, y, z = 0.12, -0.24, 0.22, DESK_TOP + radius
    out = [key(0.0, [x0, y, z], lying, 2.0)]
    for t in np.arange(2.05, 3.451, 0.05):
        x = x0 + (x1 - x0) * smoothstep((t - 2.0) / 1.5)
        out.append(key(t, [x, y, z], Rotation.from_euler("y", (x - x0) / radius).as_matrix() @ lying))
    out.append(key(3.5, [x1, y, z], Rotation.from_euler("y", (x1 - x0) / radius).as_matrix() @ lying, 4.5))
    return out


def knife():
    """A flick: 125 deg of yaw and a 12 cm slide in 0.25 s."""
    p0, p1 = np.array([0.40, -0.18, DESK_TOP]), np.array([0.30, -0.10, DESK_TOP])
    out = [key(0.0, p0, np.eye(3), 5.0)]
    for t in np.arange(5.05, 5.21, 0.05):
        s = smoothstep((t - 5.0) / 0.25)
        out.append(key(t, p0 + (p1 - p0) * s, rz(125 * s)))
    out.append(key(5.25, p1, rz(125), 2.75))
    return out


def airplane():
    """Arc over the desk from left to right in 1.2-4.2 s, heading along the path, +-25 deg roll."""
    c, radius = np.array([0.0, 0.25, 0.90]), 0.6

    def pose(t):
        s = smoothstep((t - 1.2) / 3.0)
        a = np.pi * (1 - s)
        p = c + np.array([radius * np.cos(a), -0.15 * np.sin(a), 0.15 * np.sin(a)])
        v = -np.array([-radius * np.sin(a), -0.15 * np.cos(a), 0.15 * np.cos(a)])
        r = Rotation.from_euler("zyx", [np.arctan2(v[1], v[0]), 0.0, np.radians(25) * np.sin(2 * np.pi * s)])
        return p, r.as_matrix()

    out = [key(0.0, *pose(1.2), 1.2)]
    out += [key(t, *pose(t)) for t in np.arange(1.45, 4.0, 0.25)]
    out.append(key(4.2, *pose(4.2), 3.8))
    return out


def police_car():
    """Drive 120 deg along a 0.35 m circle on the desk in 3.5-6.0 s, body +y along the path."""
    def pose(t):
        a = np.radians(200 - 120 * smoothstep((t - 3.5) / 2.5))
        p = np.array([0.20 + 0.35 * np.cos(a), 0.05 + 0.35 * np.sin(a), DESK_TOP])
        return p, Rotation.from_euler("z", np.arctan2(-np.cos(a), np.sin(a)) - np.pi / 2).as_matrix()

    out = [key(0.0, *pose(3.5), 3.5)]
    out += [key(t, *pose(t)) for t in np.arange(3.75, 5.8, 0.25)]
    out.append(key(6.0, *pose(6.0), 2.0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--assets", default="/data/assets")
    args = ap.parse_args(argv)
    root = Path(args.assets)
    box = load_asset(root / "ycb/003_cracker_box/extracted/003_cracker_box/google_64k/textured.obj", scale=1.0)
    can = load_asset(root / "ycb/005_tomato_soup_can/extracted/005_tomato_soup_can/google_64k/textured.obj", scale=1.0)
    blocks = {"cracker_box": cracker(np.ptp(box.mesh.positions[:, 0])),
              "soup_can": soup_can(0.5 * np.ptp(can.mesh.positions[:, 0])),
              "knife": knife(), "airplane": airplane(), "police_car": police_car()}
    for name, lines in blocks.items():
        print(f"# {name}\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
