"""Gaussian-splat renderer in plain PyTorch (CUDA, Apple MPS or CPU) for trained 3DGS and Mip-Splatting models.

It follows the official CUDA rasterisers (Kerbl et al. 2023; Yu et al. 2024) step by step, so a model trained on the
cluster renders the same on a laptop without CUDA:
  1. world -> camera, keep Gaussians in front of the near plane (z > 0.2)
  2. 3D covariance  Sigma = R S S^T R^T  from the scale and rotation (quaternion) of each Gaussian
     (Mip-Splatting first adds its 3D smoothing filter: scale^2 + filter^2, opacity scaled to keep the integral)
  3. 2D covariance  Sigma' = J W Sigma W^T J^T  (EWA splatting, J = Jacobian of the perspective projection)
     plus a small screen-space filter: +0.3 px^2 for 3DGS, Mip-Splatting's 2D Mip filter (+0.1, opacity rescaled)
  4. colour from degree-3 spherical harmonics in the viewing direction
  5. 16x16 tiles; per tile the Gaussians it touches, sorted near to far; front-to-back alpha blending
         C = sum_i c_i a_i T_i,  T_i = prod_{j<i} (1 - a_j),  a_i = min(0.99, o_i exp(-0.5 d^T Sigma'^-1 d))
     skipping a_i < 1/255 and stopping when T would fall below 1e-4, as the CUDA kernel does.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .packing import unpack

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
         1.445305721320277, -0.5900435899266435)
TILE = 16


def best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ loading
def read_ply_vertices(path):
    """Read the binary little-endian vertex table written by the official training code (no plyfile needed)."""
    with open(path, "rb") as f:
        header, props, n = [], [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}[t]))
            elif line == "end_header":
                break
        assert "format binary_little_endian 1.0" in header, "expected a binary little-endian PLY"
        data = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=props, count=n)
    return data


class GaussianModel:
    """Trained Gaussians: means (N,3), log-scales (N,3), quaternions (N,4, w first), opacity logits (N,1),
    SH coefficients (N,16,3); filter_3d (N,1) for Mip-Splatting models."""

    def __init__(self, means, scales, rots, opac, sh, filter_3d=None, kind="3dgs", device=None):
        self.device = device or best_device()
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=self.device)
        self.kind = kind
        self.means, sc, self.opacity_logit = t(means), t(scales), t(opac).reshape(-1)
        q = t(rots)
        self.quat = q / q.norm(dim=1, keepdim=True)
        self.sh = t(sh)
        self.scale = torch.exp(sc)
        self.opacity = torch.sigmoid(self.opacity_logit)
        if filter_3d is not None:   # Mip-Splatting 3D smoothing filter, baked into scale and opacity
            f2 = t(filter_3d).reshape(-1, 1) ** 2
            s2 = self.scale ** 2
            coef = torch.sqrt(torch.prod(s2, 1) / torch.prod(s2 + f2, 1))
            self.scale = torch.sqrt(s2 + f2)
            self.opacity = self.opacity * coef
        self.sh_degree = int(round(math.sqrt(self.sh.shape[1]))) - 1
        self.cov3d = self._covariance()

    @classmethod
    def load(cls, path, kind=None, device=None):
        path = Path(path)
        if path.suffix == ".npz":
            z = unpack(np.load(path))                  # packed files (8/16-bit codes) or plain arrays
            kind = kind or (str(z["kind"]) if "kind" in z else "3dgs")
            if "sh" in z:
                sh = z["sh"]
            else:                                      # colour stored as SH degree 0 + the higher bands (N, 3*(K-1))
                dc = np.asarray(z["sh_dc"], np.float32).reshape(-1, 1, 3)
                sh = dc if "sh_rest" not in z else np.concatenate([dc, z["sh_rest"].reshape(len(dc), -1, 3)], 1)
            return cls(z["means"], z["scales"], z["rots"], z["opacity"], sh,
                       z["filter_3d"] if "filter_3d" in z else None, kind, device)
        v = read_ply_vertices(path)
        names = v.dtype.names
        means = np.stack([v["x"], v["y"], v["z"]], 1)
        dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)[:, None, :]
        rest_names = sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
        rest = np.stack([v[n] for n in rest_names], 1).reshape(len(v), 3, -1).transpose(0, 2, 1) if rest_names \
            else np.zeros((len(v), 0, 3), np.float32)
        sh = np.concatenate([dc, rest], 1)
        scales = np.stack([v["scale_%d" % i] for i in range(3)], 1)
        rots = np.stack([v["rot_%d" % i] for i in range(4)], 1)
        filt = v["filter_3D"][:, None] if "filter_3D" in names else None
        return cls(means, scales, rots, v["opacity"][:, None], sh, filt, kind or ("mip" if filt is not None else "3dgs"), device)

    def _covariance(self):
        w, x, y, z = self.quat.unbind(1)
        R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
        M = R * self.scale[:, None, :]              # R S
        return M @ M.transpose(1, 2)                # R S S^T R^T

    def __len__(self):
        return self.means.shape[0]


# ------------------------------------------------------------------ rendering
def eval_sh(deg, sh, dirs):
    """Colour from spherical harmonics (same constants and order as the official code)."""
    res = SH_C0 * sh[:, 0]
    if deg > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        res = res - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if deg > 1:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            res = (res + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5] + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6]
                   + SH_C2[3] * xz * sh[:, 7] + SH_C2[4] * (xx - yy) * sh[:, 8])
            if deg > 2:
                res = (res + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9] + SH_C3[1] * xy * z * sh[:, 10]
                       + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                       + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + SH_C3[5] * z * (xx - yy) * sh[:, 14]
                       + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return res + 0.5


def auto_tile(cam):
    return 4 if max(cam.W, cam.H) <= 400 else 8 if max(cam.W, cam.H) <= 900 else 16


@torch.no_grad()
def render(model: GaussianModel, cam, bg=(1.0, 1.0, 1.0), batch=None, kernel_size=0.1, tile=None):
    """Render an image (H, W, 3) in [0, 1] for a camera with fields W, H, fx, fy, R (world->cam 3x3), t (3,).
    Like the official code the principal point is the image centre. `tile` (pixels) only changes speed: small
    images are faster with small tiles, since each tile then holds fewer Gaussians."""
    dev = model.device
    TILE = tile or auto_tile(cam)
    # Gaussians per step: as many as fit in ~24M per-pixel values (fast on small images, bounded memory on big ones)
    batch = batch or int(min(256, max(32, 24e6 / (int(cam.W) * int(cam.H)))))
    W, H = int(cam.W), int(cam.H)
    R = torch.as_tensor(np.asarray(cam.R, np.float32), device=dev)
    t = torch.as_tensor(np.asarray(cam.t, np.float32), device=dev)
    fx, fy = float(cam.fx), float(cam.fy)
    pc = model.means @ R.T + t
    z = pc[:, 2]
    tanx, tany = W / (2 * fx), H / (2 * fy)
    keep = z > 0.2            # like the official rasteriser, drop only what is behind the near plane: a large Gaussian
    idx = torch.nonzero(keep).squeeze(1)   # centred outside the view can still cover it (tiles are clipped below)
    if idx.numel() == 0:
        return torch.ones(H, W, 3, device=dev) * torch.tensor(bg, device=dev)
    pc, z = pc[idx], z[idx]
    cov3 = model.cov3d[idx]
    opac = model.opacity[idx]

    # EWA: Jacobian of the projection, with x/z, y/z clamped to 1.3 tan(fov/2) like the official code
    tx = (pc[:, 0] / z).clamp(-1.3 * tanx, 1.3 * tanx) * z
    ty = (pc[:, 1] / z).clamp(-1.3 * tany, 1.3 * tany) * z
    J = torch.zeros(len(z), 2, 3, device=dev)
    J[:, 0, 0], J[:, 0, 2] = fx / z, -fx * tx / (z * z)
    J[:, 1, 1], J[:, 1, 2] = fy / z, -fy * ty / (z * z)
    T = J @ R[None]
    cov2 = T @ cov3 @ T.transpose(1, 2)
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    if model.kind == "mip":       # 2D Mip filter: Gaussian blur of one pixel's footprint, opacity keeps the energy
        det0 = (a * c - b * b).clamp_min(0)
        a, c = a + kernel_size, c + kernel_size
        det = a * c - b * b
        opac = opac * torch.sqrt((det0 / det.clamp_min(1e-12)).clamp_min(0.000025))
    else:                         # 3DGS low-pass: +0.3 px^2
        a, c = a + 0.3, c + 0.3
        det = a * c - b * b
    ok = det > 0
    det_inv = 1.0 / det.clamp_min(1e-12)
    conic = torch.stack([c * det_inv, -b * det_inv, a * det_inv], 1)
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam))
    u = fx * pc[:, 0] / z + (W - 1) / 2.0      # pixel centres at integer coordinates, as in the CUDA kernel
    v = fy * pc[:, 1] / z + (H - 1) / 2.0

    # colour from SH in the direction camera -> Gaussian
    campos = -R.T @ t
    dirs = model.means[idx] - campos
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    rgb = eval_sh(model.sh_degree, model.sh[idx], dirs).clamp_min(0.0)

    # tiles touched by each Gaussian's 3-sigma square
    gx, gy = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    x0 = ((u - radius) / TILE).floor().clamp(0, gx).long()
    x1 = ((u + radius + TILE - 1) / TILE).floor().clamp(0, gx).long()
    y0 = ((v - radius) / TILE).floor().clamp(0, gy).long()
    y1 = ((v + radius + TILE - 1) / TILE).floor().clamp(0, gy).long()
    nt = (x1 - x0) * (y1 - y0)
    ok &= (nt > 0) & (radius > 0)
    sel = torch.nonzero(ok).squeeze(1)
    if sel.numel() == 0:
        return torch.ones(H, W, 3, device=dev) * torch.tensor(bg, device=dev)
    u, v, conic, opac, rgb, z = u[sel], v[sel], conic[sel], opac[sel], rgb[sel], z[sel]
    x0, x1, y0, y1, nt = x0[sel], x1[sel], y0[sel], y1[sel], nt[sel]

    # expand to (tile, Gaussian) pairs, sort by tile then depth
    order = torch.argsort(z)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order), device=dev)
    gid = torch.repeat_interleave(torch.arange(len(nt), device=dev), nt)
    start = torch.cumsum(nt, 0) - nt
    k = torch.arange(len(gid), device=dev) - start[gid]
    w_ = (x1 - x0)[gid]
    tx_ = x0[gid] + k % w_
    ty_ = y0[gid] + torch.div(k, w_, rounding_mode="floor")
    tile = ty_ * gx + tx_
    key = tile * len(nt) + rank[gid]
    key, perm = torch.sort(key)
    gid, tile = gid[perm], tile[perm]
    ntiles = gx * gy
    counts = torch.zeros(ntiles, dtype=torch.long, device=dev).index_add_(0, tile, torch.ones_like(tile))
    offs = torch.cumsum(counts, 0) - counts
    lmax = int(counts.max().item())

    # pixel coordinates of every tile: (ntiles, 256)
    ly, lx = torch.meshgrid(torch.arange(TILE, device=dev), torch.arange(TILE, device=dev), indexing="ij")
    tiles = torch.arange(ntiles, device=dev)
    px = ((tiles % gx) * TILE)[:, None].float() + lx.reshape(1, -1).float()
    py = (torch.div(tiles, gx, rounding_mode="floor") * TILE)[:, None].float() + ly.reshape(1, -1).float()

    Tacc = torch.ones(ntiles, TILE * TILE, device=dev)
    C = torch.zeros(ntiles, TILE * TILE, 3, device=dev)
    alive = torch.ones(ntiles, TILE * TILE, dtype=torch.bool, device=dev)
    for s in range(0, lmax, batch):
        j = torch.arange(s, min(s + batch, lmax), device=dev)
        valid = j[None, :] < counts[:, None]                                  # (ntiles, B)
        active = torch.nonzero(valid.any(1) & alive.any(1)).squeeze(1)
        if active.numel() == 0:
            continue
        valid = valid[active]
        pidx = (offs[active, None] + j[None, :]).clamp_max(len(gid) - 1)
        g = gid[pidx]                                                          # (A, B)
        dx = u[g][..., None] - px[active][:, None, :]                          # (A, B, 256): mean - pixel
        dy = v[g][..., None] - py[active][:, None, :]
        cn = conic[g]
        power = -0.5 * (cn[..., 0:1] * dx * dx + cn[..., 2:3] * dy * dy) - cn[..., 1:2] * dx * dy
        alpha = torch.clamp(opac[g][..., None] * torch.exp(power), max=0.99)
        alpha = torch.where((power <= 0) & (alpha >= 1.0 / 255.0) & valid[..., None], alpha, torch.zeros_like(alpha))
        # transmittance inside this batch, then early stop when T would drop below 1e-4
        log1m = torch.log1p(-alpha)
        Tin = Tacc[active][:, None, :] * torch.exp(torch.cumsum(log1m, 1) - log1m)     # T before each Gaussian
        Tafter = Tin * (1 - alpha)
        contrib = (Tafter >= 1e-4) & alive[active][:, None, :]
        stop = torch.cumsum((~contrib & (alpha > 0)).int(), 1) > 0                       # once stopped, stays stopped
        contrib &= ~stop
        wgt = torch.where(contrib, alpha * Tin, torch.zeros_like(alpha))
        C[active] += torch.einsum("abp,abc->apc", wgt, rgb[g])
        a_eff = torch.where(contrib, alpha, torch.zeros_like(alpha))
        Tacc[active] = Tacc[active] * torch.exp(torch.log1p(-a_eff).sum(1))
        alive[active] &= ~stop.any(1)
    out = C + Tacc[..., None] * torch.tensor(bg, device=dev)
    img = out.reshape(gy, gx, TILE, TILE, 3).permute(0, 2, 1, 3, 4).reshape(gy * TILE, gx * TILE, 3)
    return img[:H, :W].clamp(0, 1)
