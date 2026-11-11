"""TensoRF (VM decomposition) renderer in plain PyTorch (CUDA, Apple MPS or CPU) for a trained model.

It reproduces the official forward pass (Chen et al. 2022) for inference:
  density   sigma(x) = softplus( sum_r  P_r^{XY}(x,y) L_r^{Z}(z) + P_r^{XZ} L_r^{Y} + P_r^{YZ} L_r^{X}  - 10 )
  colour    c(x, d) = MLP( B [P^{XY} L^{Z}, P^{XZ} L^{Y}, P^{YZ} L^{X}],  d )  (features + positional encodings)
  rendering C = sum_i T_i (1 - exp(-sigma_i delta_i)) c_i  + T_N * white,   T_i = prod_{j<i} exp(-sigma_j delta_j)
with samples every `stepSize` along the ray inside the box, skipping empty space with the saved occupancy grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import best_device

MAT = [[0, 1], [0, 2], [1, 2]]
VEC = [2, 1, 0]


def _pe(x, freqs):
    fb = 2.0 ** torch.arange(freqs, device=x.device, dtype=x.dtype)
    p = (x[..., None] * fb).reshape(*x.shape[:-1], freqs * x.shape[-1])
    return torch.cat([torch.sin(p), torch.cos(p)], -1)


class TensoRFModel:
    def __init__(self, z: dict, device=None):
        self.device = device or best_device()
        t = lambda k: torch.as_tensor(np.asarray(z[k], np.float32), device=self.device)
        self.aabb = t("aabb")
        self.grid = np.asarray(z["gridSize"]).astype(int)
        self.near, self.far = [float(v) for v in z["near_far"]]
        self.density_shift, self.distance_scale = float(z["density_shift"]), float(z["distance_scale"])
        self.weight_thres = float(z["rayMarch_weight_thres"])
        self.view_pe, self.fea_pe = int(z["view_pe"]), int(z["fea_pe"])
        self.dplane = [t("density_plane.%d" % i) for i in range(3)]
        self.dline = [t("density_line.%d" % i) for i in range(3)]
        self.aplane = [t("app_plane.%d" % i) for i in range(3)]
        self.aline = [t("app_line.%d" % i) for i in range(3)]
        self.basis = t("basis_mat.weight")
        self.mlp = [(t("renderModule.mlp.%d.weight" % i), t("renderModule.mlp.%d.bias" % i)) for i in (0, 2, 4)]
        size = self.aabb[1] - self.aabb[0]
        self.inv = 2.0 / size
        units = size / torch.as_tensor(self.grid - 1, dtype=torch.float32, device=self.device)
        self.step = float(units.mean()) * float(z["step_ratio"])
        self.n_samples = int(float(torch.sqrt((size ** 2).sum())) / self.step) + 1
        self.mask = None
        if "alpha_shape" in z:
            shape = tuple(int(v) for v in z["alpha_shape"])
            vol = np.unpackbits(np.asarray(z["alpha_bits"]))[: int(np.prod(shape))].reshape(shape[-3:])   # saved as (1,1,D,H,W)
            self.mask = torch.as_tensor(vol.astype(bool), device=self.device)            # (D=z, H=y, W=x)
            self.mask_aabb = t("alpha_aabb")
            self.mask_inv = 2.0 / (self.mask_aabb[1] - self.mask_aabb[0])

    @classmethod
    def load(cls, path, device=None):
        path = Path(path)
        if path.suffix == ".npz":
            return cls(dict(np.load(path)), device)
        return cls(convert_checkpoint(path), device)

    # ---------------------------------------------------------------- fields
    def _sample2d(self, planes, lines, xyz):
        """plane x line features for normalised points xyz (N,3) in [-1,1]: list of (C_i, N) per mode"""
        out = []
        for i in range(3):
            gp = xyz[:, MAT[i]].view(1, -1, 1, 2)
            gl = torch.stack([torch.zeros_like(xyz[:, VEC[i]]), xyz[:, VEC[i]]], -1).view(1, -1, 1, 2)
            p = F.grid_sample(planes[i], gp, align_corners=True).view(planes[i].shape[1], -1)
            l = F.grid_sample(lines[i], gl, align_corners=True).view(lines[i].shape[1], -1)
            out.append(p * l)
        return out

    def density(self, xyz):
        f = sum(m.sum(0) for m in self._sample2d(self.dplane, self.dline, xyz))
        return F.softplus(f + self.density_shift)

    def colour(self, xyz, dirs):
        feat = torch.cat(self._sample2d(self.aplane, self.aline, xyz), 0).T @ self.basis.T
        x = [feat, dirs]
        if self.fea_pe > 0:
            x.append(_pe(feat, self.fea_pe))
        if self.view_pe > 0:
            x.append(_pe(dirs, self.view_pe))
        h = torch.cat(x, -1)
        for k, (w, b) in enumerate(self.mlp):
            h = h @ w.T + b
            if k < 2:
                h = torch.relu(h)
        return torch.sigmoid(h)

    def occupied(self, pts):
        """the saved occupancy grid, sampled like grid_sample (a point counts if any of its 8 neighbours is set)"""
        if self.mask is None:
            return torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
        D, H, W = self.mask.shape
        n = (pts - self.mask_aabb[0]) * self.mask_inv - 1
        g = (n + 1) / 2 * torch.as_tensor([W - 1, H - 1, D - 1], dtype=torch.float32, device=pts.device)
        lo = torch.floor(g).long()
        frac = g - lo
        hit = torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (frac[:, 0] if dx else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1]) * (frac[:, 2] if dz else 1 - frac[:, 2])
                    ix, iy, iz = lo[:, 0] + dx, lo[:, 1] + dy, lo[:, 2] + dz
                    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (iz >= 0) & (iz < D)
                    v = torch.zeros_like(hit)
                    v[inside] = self.mask[iz[inside], iy[inside], ix[inside]]
                    hit |= v & (w > 0)
        return hit

    # ---------------------------------------------------------------- rendering
    @torch.no_grad()
    def render_rays(self, rays_o, rays_d, white_bg=True, chunk=8192):
        out = []
        for s in range(0, rays_o.shape[0], chunk):
            out.append(self._render_chunk(rays_o[s:s + chunk], rays_d[s:s + chunk], white_bg))
        return torch.cat(out, 0)

    def _render_chunk(self, o, d, white_bg):
        vec = torch.where(d == 0, torch.full_like(d, 1e-6), d)
        ra, rb = (self.aabb[1] - o) / vec, (self.aabb[0] - o) / vec
        tmin = torch.minimum(ra, rb).amax(-1).clamp(min=self.near, max=self.far)
        z = tmin[:, None] + self.step * torch.arange(self.n_samples, device=o.device, dtype=torch.float32)[None]
        pts = o[:, None, :] + d[:, None, :] * z[..., None]                         # (R, S, 3)
        valid = ~((self.aabb[0] > pts) | (pts > self.aabb[1])).any(-1)
        if self.mask is not None and valid.any():
            occ = self.occupied(pts[valid])
            v2 = valid.clone()
            v2[valid] = occ
            valid = v2
        dists = torch.cat([z[:, 1:] - z[:, :-1], torch.zeros_like(z[:, :1])], -1)
        sigma = torch.zeros(z.shape, device=o.device)
        norm = (pts - self.aabb[0]) * self.inv - 1
        if valid.any():
            sigma[valid] = self.density(norm[valid])
        alpha = 1 - torch.exp(-sigma * dists * self.distance_scale)
        T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], -1), -1)
        w = alpha * T[:, :-1]
        rgb = torch.zeros(*z.shape, 3, device=o.device)
        m = w > self.weight_thres
        if m.any():
            rgb[m] = self.colour(norm[m], d[:, None, :].expand_as(pts)[m])
        acc = w.sum(-1)
        c = (w[..., None] * rgb).sum(-2)
        if white_bg:
            c = c + (1 - acc[:, None])
        return c.clamp(0, 1)


@torch.no_grad()
def render(model: TensoRFModel, cam, white_bg=True):
    """Image (H, W, 3) for a Camera (OpenCV axes); pixel centres at +0.5 like the official loader."""
    dev = model.device
    W, H = int(cam.W), int(cam.H)
    j, i = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=dev), torch.arange(W, dtype=torch.float32, device=dev), indexing="ij")
    d = torch.stack([(i + 0.5 - cam.cx) / cam.fx, (j + 0.5 - cam.cy) / cam.fy, torch.ones_like(i)], -1).reshape(-1, 3)
    d = d / d.norm(dim=-1, keepdim=True)
    R = torch.as_tensor(np.asarray(cam.R, np.float32), device=dev)
    rays_d = d @ R                                        # camera -> world (R^T d)
    rays_o = torch.as_tensor(np.asarray(cam.center, np.float32), device=dev).expand_as(rays_d)
    return model.render_rays(rays_o, rays_d, white_bg).reshape(H, W, 3)


def convert_checkpoint(path, out=None, half=True):
    """Official TensoRF .th checkpoint -> plain arrays (optionally saved as .npz; float16 halves the size)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    kw, sd = ck["kwargs"], ck["state_dict"]
    z = {k: (v.numpy().astype(np.float16) if half and v.ndim >= 2 and "mlp" not in k and "basis" not in k else v.numpy())
         for k, v in sd.items() if not k.startswith("alphaMask")}
    z["aabb"] = np.asarray(kw["aabb"].cpu() if torch.is_tensor(kw["aabb"]) else kw["aabb"], np.float32)
    for k in ("gridSize", "near_far", "density_shift", "distance_scale", "rayMarch_weight_thres", "view_pe", "fea_pe", "step_ratio"):
        z[k] = np.asarray(kw[k])
    if "alphaMask.mask" in ck:
        z["alpha_shape"], z["alpha_bits"] = np.asarray(ck["alphaMask.shape"]), np.asarray(ck["alphaMask.mask"])
        z["alpha_aabb"] = np.asarray(ck["alphaMask.aabb"].cpu(), np.float32)
    if out:
        np.savez_compressed(out, **z)
    return z
