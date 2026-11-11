"""NeRF renderer in plain PyTorch (CUDA, Apple MPS or CPU) for a model trained with nerf-pytorch.

It repeats nerf-pytorch's inference (Mildenhall et al. 2020, run_nerf.py / run_nerf_helpers.py) for the trained
coarse and fine networks:
  encoding     gamma(p) = (p, sin(2^0 p), cos(2^0 p), ..., sin(2^(L-1) p), cos(2^(L-1) p)),  L = 10 / 4 (directions)
  network      8 layers of 256 + ReLU, encoded position concatenated after the 5th; density from the last layer;
               256-feature + encoded unit direction -> 128 -> RGB
  rendering    64 evenly spaced depths between the ray's near and far (here: where it enters and leaves the
               person's box, as in training), then 128 more from the coarse weights (inverse CDF, deterministic);
               alpha_i = 1 - exp(-relu(sigma_i) delta_i |d|),  C = sum_i alpha_i T_i c_i + (1 - sum_i alpha_i T_i) * white
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import best_device


def embed(x, L):
    out = [x]
    for f in 2.0 ** torch.linspace(0.0, L - 1, L, device=x.device):
        out += [torch.sin(x * f), torch.cos(x * f)]
    return torch.cat(out, -1)


class _MLP:
    def __init__(self, sd, dev, skips=(4,)):
        t = lambda k: torch.as_tensor(np.asarray(sd[k], np.float32), device=dev)
        n = len([k for k in sd if k.startswith("pts_linears.") and k.endswith(".weight")])
        self.pts = [(t("pts_linears.%d.weight" % i), t("pts_linears.%d.bias" % i)) for i in range(n)]
        self.views = [(t("views_linears.0.weight"), t("views_linears.0.bias"))]
        self.feature = (t("feature_linear.weight"), t("feature_linear.bias"))
        self.alpha = (t("alpha_linear.weight"), t("alpha_linear.bias"))
        self.rgb = (t("rgb_linear.weight"), t("rgb_linear.bias"))
        self.skips = skips

    def __call__(self, x, d):
        h = x
        for i, (w, b) in enumerate(self.pts):
            h = F.relu(F.linear(h, w, b))
            if i in self.skips:
                h = torch.cat([x, h], -1)
        alpha = F.linear(h, *self.alpha)
        h = torch.cat([F.linear(h, *self.feature), d], -1)
        for w, b in self.views:
            h = F.relu(F.linear(h, w, b))
        return torch.cat([F.linear(h, *self.rgb), alpha], -1)


def _composite(raw, z, d, white):
    dists = torch.cat([z[:, 1:] - z[:, :-1], torch.full_like(z[:, :1], 1e10)], -1) * d.norm(dim=-1, keepdim=True)
    rgb = torch.sigmoid(raw[..., :3])
    alpha = 1.0 - torch.exp(-F.relu(raw[..., 3]) * dists)
    w = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    c = (w[..., None] * rgb).sum(-2)
    if white:
        c = c + (1.0 - w.sum(-1, keepdim=True))
    return c, w


def _sample_pdf(bins, weights, n):
    weights = weights + 1e-5
    pdf = weights / weights.sum(-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], -1)
    u = torch.linspace(0.0, 1.0, n, device=bins.device).expand(cdf.shape[0], n).contiguous()
    inds = torch.searchsorted(cdf.contiguous(), u, right=True)
    below, above = (inds - 1).clamp_min(0), inds.clamp_max(cdf.shape[-1] - 1)
    cb, ca = torch.gather(cdf, 1, below), torch.gather(cdf, 1, above)
    bb, ba = torch.gather(bins, 1, below), torch.gather(bins, 1, above)
    den = torch.where(ca - cb < 1e-5, torch.ones_like(ca), ca - cb)
    return bb + (u - cb) / den * (ba - bb)


class NeRFModel:
    def __init__(self, z: dict, device=None):
        self.device = device or best_device()
        self.coarse = _MLP({k[7:]: v for k, v in z.items() if k.startswith("coarse.")}, self.device)
        self.fine = _MLP({k[5:]: v for k, v in z.items() if k.startswith("fine.")}, self.device)
        self.lo = torch.as_tensor(np.asarray(z["aabb"][0], np.float32), device=self.device)
        self.hi = torch.as_tensor(np.asarray(z["aabb"][1], np.float32), device=self.device)
        self.Lx, self.Ld = int(z.get("multires", 10)), int(z.get("multires_views", 4))
        self.n_coarse, self.n_fine = int(z.get("N_samples", 64)), int(z.get("N_importance", 128))
        self.white = bool(z.get("white_bkgd", True))

    @classmethod
    def load(cls, path, device=None):
        return cls(dict(np.load(Path(path))), device)

    def near_far(self, o, d):
        inv = 1.0 / torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
        t0, t1 = (self.lo - o) * inv, (self.hi - o) * inv
        n = torch.minimum(t0, t1).amax(-1).clamp_min(0.05)
        return n, torch.maximum(torch.maximum(t0, t1).amin(-1), n + 1e-3)

    def _query(self, net, pts, vd):
        x = embed(pts.reshape(-1, 3), self.Lx)
        v = embed(vd[:, None].expand_as(pts).reshape(-1, 3), self.Ld)
        return net(x, v).reshape(*pts.shape[:2], 4)

    @torch.no_grad()
    def render_rays(self, o, d, chunk=8192):
        out = []
        for s in range(0, o.shape[0], chunk):
            oo, dd = o[s:s + chunk], d[s:s + chunk]
            vd = dd / dd.norm(dim=-1, keepdim=True)
            n, f = self.near_far(oo, dd)
            t = torch.linspace(0.0, 1.0, self.n_coarse, device=oo.device)
            z = n[:, None] * (1 - t) + f[:, None] * t
            raw = self._query(self.coarse, oo[:, None] + dd[:, None] * z[..., None], vd)
            _, w = _composite(raw, z, dd, self.white)
            z2 = _sample_pdf(0.5 * (z[:, 1:] + z[:, :-1]), w[:, 1:-1], self.n_fine)
            zz, _ = torch.sort(torch.cat([z, z2], -1), -1)
            raw = self._query(self.fine, oo[:, None] + dd[:, None] * zz[..., None], vd)
            out.append(_composite(raw, zz, dd, self.white)[0])
        return torch.cat(out, 0)


@torch.no_grad()
def render(model: NeRFModel, cam):
    """Image (H, W, 3) for a Camera (OpenCV axes); rays through pixel centres, z = 1 in the camera as in training."""
    dev = model.device
    W, H = int(cam.W), int(cam.H)
    j, i = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=dev), torch.arange(W, dtype=torch.float32, device=dev), indexing="ij")
    d = torch.stack([(i + 0.5 - cam.cx) / cam.fx, (j + 0.5 - cam.cy) / cam.fy, torch.ones_like(i)], -1).reshape(-1, 3)
    R = torch.as_tensor(np.asarray(cam.R, np.float32), device=dev)
    rays_d = d @ R
    rays_o = torch.as_tensor(np.asarray(cam.center, np.float32), device=dev).expand_as(rays_d)
    return model.render_rays(rays_o, rays_d).reshape(H, W, 3).clamp(0, 1)


def convert_checkpoint(path, aabb, out=None, half=False):
    """nerf-pytorch checkpoint (.tar) -> plain arrays (+ the box the rays were sampled in), optionally saved as .npz"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    z = {"coarse." + k: v.numpy() for k, v in ck["network_fn_state_dict"].items()}
    z.update({"fine." + k: v.numpy() for k, v in ck["network_fine_state_dict"].items()})
    if half:
        z = {k: v.astype(np.float16) for k, v in z.items()}
    z.update(aabb=np.asarray(aabb, np.float32), multires=10, multires_views=4, N_samples=64, N_importance=128,
             white_bkgd=True, iterations=ck["global_step"])
    if out:
        np.savez_compressed(out, **z)
    return z
