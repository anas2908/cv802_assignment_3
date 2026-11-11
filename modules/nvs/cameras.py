"""Cameras for novel-view synthesis: one pinhole camera type shared by the renderers, the GUI and the video paths.

Convention (OpenCV / COLMAP): x right, y down, z forward; R, t map world -> camera (x_cam = R x_world + t).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Camera:
    W: int
    H: int
    fx: float
    fy: float
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cx: float | None = None
    cy: float | None = None
    name: str = ""

    def __post_init__(self):
        self.R, self.t = np.asarray(self.R, np.float64), np.asarray(self.t, np.float64)
        self.cx = self.W / 2.0 if self.cx is None else self.cx
        self.cy = self.H / 2.0 if self.cy is None else self.cy

    @property
    def center(self):
        return -self.R.T @ self.t

    @property
    def fovx(self):
        return 2 * math.atan(self.W / (2 * self.fx))

    @property
    def fovy(self):
        return 2 * math.atan(self.H / (2 * self.fy))

    def scaled(self, s):
        """Same view at another resolution (s = 0.25 gives a quarter-size preview)."""
        return Camera(max(8, int(round(self.W * s))), max(8, int(round(self.H * s))), self.fx * s, self.fy * s,
                      self.R, self.t, self.cx * s, self.cy * s, self.name)

    def c2w(self):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.R.T, self.center
        return m

    def to_dict(self):
        return {"W": self.W, "H": self.H, "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "R": self.R.tolist(), "t": self.t.tolist(), "name": self.name}

    @classmethod
    def from_dict(cls, d):
        return cls(d["W"], d["H"], d["fx"], d["fy"], np.array(d["R"]), np.array(d["t"]), d.get("cx"), d.get("cy"), d.get("name", ""))


def look_at(eye, target, up, W, H, fx, fy=None, name=""):
    """Camera at `eye` looking at `target`, with `up` pointing up in the image."""
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)                  # image x to the right when y (= -up) points down: x = y × z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])              # rows: camera axes in world coordinates
    return Camera(W, H, fx, fy or fx, R, -R @ eye, name=name)


def orbit_path(centre, up, radius, height, n_frames, W, H, fx, start_dir=None, turns=1.0, bob=0.0):
    """A full circle around `centre` at `height` above it (along `up`), always looking at the centre.
    `bob` adds a gentle up-and-down wave (fraction of the radius) so the path is not flat."""
    up = np.asarray(up, np.float64) / np.linalg.norm(up)
    a = np.asarray(start_dir if start_dir is not None else [1.0, 0.0, 0.0], np.float64)
    a = a - up * (a @ up)
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    cams = []
    for i in range(n_frames):
        th = 2 * math.pi * turns * i / n_frames
        h = height + bob * radius * math.sin(2 * th)
        eye = np.asarray(centre) + radius * (math.cos(th) * a + math.sin(th) * b) + h * up
        cams.append(look_at(eye, centre, up, W, H, fx, name="orbit_%04d" % i))
    return cams


def _slerp(q0, q1, s):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def rot_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def keyframe_path(keys, n_frames, closed=False):
    """Smooth path through keyframe cameras: Catmull-Rom on positions, slerp on orientations."""
    if len(keys) == 1:
        return [keys[0]] * n_frames
    P = [k.center for k in keys]
    Q = [rot_to_quat(k.R) for k in keys]
    segs = len(keys) if closed else len(keys) - 1
    out = []
    for i in range(n_frames):
        u = segs * i / (n_frames - (0 if closed else 1))
        s = min(int(u), segs - 1)
        f = u - s
        idx = lambda j: j % len(keys) if closed else min(max(j, 0), len(keys) - 1)
        p0, p1, p2, p3 = P[idx(s - 1)], P[idx(s)], P[idx(s + 1)], P[idx(s + 2)]
        pos = 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
        R = quat_to_rot(_slerp(Q[idx(s)], Q[idx(s + 1)], f))
        k = keys[idx(s)]
        out.append(Camera(k.W, k.H, k.fx, k.fy, R, -R @ pos, name="key_%04d" % i))
    return out


def save_path(cams, path):
    json.dump([c.to_dict() for c in cams], open(path, "w"))


def load_path(path):
    return [Camera.from_dict(d) for d in json.load(open(path))]
