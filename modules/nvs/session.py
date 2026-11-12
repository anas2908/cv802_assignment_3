"""Novel-view results folder + one renderer front-end for all methods (used by the GUI and the tests).

A results folder (made by training/export_bundle.py) holds:
  scene.json        photo cameras (train/test), scene centre and up direction, list of models, metrics
  models/*.npz      trained models (packed, see packing.py): Gaussians, TensoRF planes and lines, NeRF weights
  points.ply        the person's point cloud (for orientation in the 3D view and the GUI's mesh extraction)
  orbit_path.json   the preset camera path of the result videos
  photos/*.jpg      the held-out photos, to compare a render with the real photo
  meshes/           Sub-project 2 meshes (rasterised by the GUI)
  videos/*.mp4      the result videos along the preset path
  metrics.json      every run's scores (main runs and the point-cloud-start ablation, per photo)
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .cameras import Camera, keyframe_path, load_path
from .gaussians import GaussianModel, best_device, render as render_gaussians
from .nerf import NeRFModel, render as render_nerf
from .tensorf import TensoRFModel, render as render_tensorf


class NVSSession:
    def __init__(self, folder, device=None):
        self.root = Path(folder)
        self.meta = json.load(open(self.root / "scene.json"))
        self.device = device or best_device()
        self.cameras = [Camera.from_dict(c) for c in self.meta["cameras"]]
        self.models = self.meta["models"]                     # id -> {label, file, kind, background}
        self._loaded = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- models
    def labels(self):
        return [(k, v["label"]) for k, v in self.models.items()]

    def model(self, mid):
        if mid not in self._loaded:
            info = self.models[mid]
            path = self.root / info["file"]
            if info["kind"] == "tensorf":
                self._loaded[mid] = TensoRFModel.load(path, self.device)
            elif info["kind"] == "nerf":
                self._loaded[mid] = NeRFModel.load(path, self.device)
            else:
                self._loaded[mid] = GaussianModel.load(path, kind=info["kind"], device=self.device)
        return self._loaded[mid]

    def background(self, mid):
        return tuple(self.models[mid].get("background", [1.0, 1.0, 1.0]))

    @torch.no_grad()
    def render(self, mid, cam: Camera):
        """uint8 image (H, W, 3) of model `mid` seen from `cam`, and the seconds it took"""
        with self._lock:
            t0 = time.time()
            m = self.model(mid)
            bg = self.background(mid)
            if self.models[mid]["kind"] == "tensorf":
                img = render_tensorf(m, cam, white_bg=bg[0] > 0.5)
            elif self.models[mid]["kind"] == "nerf":
                img = render_nerf(m, cam)
            else:
                img = render_gaussians(m, cam, bg=bg)
            out = (img.clamp(0, 1) * 255).round().byte().cpu().numpy()
            if self.device.type == "mps":
                torch.mps.synchronize()
            return np.ascontiguousarray(out), time.time() - t0

    # ---------------------------------------------------------------- paths and photos
    def orbit(self):
        return load_path(self.root / self.meta["orbit_path"])

    def photo(self, name):
        p = self.root / "photos" / (Path(name).stem + ".jpg")
        return p if p.exists() else None

    def test_names(self):
        return [c.name for c in self.cameras if c.name in set(self.meta["split"]["test"])]

    @staticmethod
    def path_from_keyframes(keys, n_frames):
        return keyframe_path(keys, n_frames)

    def export_video(self, mid, cams, out, fps=30, progress=None):
        import imageio.v2 as imageio
        w = imageio.get_writer(str(out), fps=fps, quality=9, macro_block_size=8)
        for i, c in enumerate(cams):
            img, _ = self.render(mid, c)
            w.append_data(img)
            if progress:
                progress(i + 1, len(cams))
        w.close()
        return out
