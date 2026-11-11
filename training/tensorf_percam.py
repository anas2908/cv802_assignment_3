"""TensoRF data loader for photos taken with several cameras (our 125 photos come from 4 iPhone lenses).

Same interface as TensoRF's Blender loader, but every frame carries its own intrinsics and image size
(transforms_{train,test}.json written by prep_scenes.py; camera-to-world in OpenGL axes like Blender). The photos
are already composited on white. setup_envs.sh copies this file into TensoRF/dataLoader/ and registers it as 'percam'.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import get_ray_directions, get_rays


class PerCamDataset(Dataset):
    def __init__(self, datadir, split="train", downsample=1.0, is_stack=False, N_vis=-1):
        self.root_dir, self.split, self.is_stack, self.N_vis = datadir, split, is_stack, N_vis
        self.downsample = downsample
        self.transform = T.ToTensor()
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        with open(os.path.join(datadir, f"transforms_{split}.json")) as f:
            self.meta = json.load(f)
        self.scene_bbox = torch.tensor(self.meta["aabb"], dtype=torch.float32)
        self.near_far = list(self.meta["near_far"])
        self.white_bg = bool(self.meta.get("white_bg", True))
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.read_meta()

    def read_meta(self):
        self.image_paths, self.poses, self.all_rays, self.all_rgbs, self.img_whs, self.intrinsics = [], [], [], [], [], []
        frames = self.meta["frames"]
        step = 1 if self.N_vis < 0 else max(1, len(frames) // self.N_vis)
        for fr in frames[::step]:
            img = Image.open(os.path.join(self.root_dir, fr["file_path"])).convert("RGB")
            w, h = img.size
            s = 1.0 / self.downsample
            if s != 1.0:
                w, h = int(round(w * s)), int(round(h * s))
                img = img.resize((w, h), Image.LANCZOS)
            fx, fy, cx, cy = fr["fl_x"] * s, fr["fl_y"] * s, fr["cx"] * s, fr["cy"] * s
            dirs = get_ray_directions(h, w, [fx, fy], center=[cx, cy])
            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            c2w = torch.FloatTensor(np.array(fr["transform_matrix"]) @ self.blender2opencv)
            rays_o, rays_d = get_rays(dirs, c2w)
            self.poses.append(c2w)
            self.image_paths.append(fr["file_path"])
            self.img_whs.append((w, h))
            self.intrinsics.append((fx, fy, cx, cy))
            self.all_rgbs.append(self.transform(img).view(3, -1).permute(1, 0))
            self.all_rays.append(torch.cat([rays_o, rays_d], 1))
        self.poses = torch.stack(self.poses)
        self.img_wh = self.img_whs[0]
        if not self.is_stack:   # training: one flat buffer of rays; testing keeps one entry per photo (sizes differ)
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        return {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}
