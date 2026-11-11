"""Train NeRF with the nerf-pytorch code on our photos, then score it and render the result video.

Run from inside the nerf-pytorch repository with the gs environment:
    python train_nerf.py --data <DATA>/person --out <RUNS>/nerf_person --minutes 60 [--path orbit.json --video v.mp4]

The official training loop assumes one camera for all photos; ours come from 4 lenses, so this script builds the
rays itself (per photo intrinsics, like the TensoRF loader) and otherwise follows nerf-pytorch's train():
its create_nerf / render, the paper's Blender settings (configs/lego.txt: 64 + 128 samples, 1024 rays per step,
view directions, white background, lr 5e-4 decaying 10x every 500k steps), loss = MSE(fine) + MSE(coarse).
Rays sample only between their entry and exit of the person's box (render() takes a near/far per ray).
Training stops after --minutes (the paper uses ~200k steps; far more than our time allows).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.getcwd())
sys.path.insert(1, str(CODE / "training"))
torch.set_default_tensor_type("torch.cuda.FloatTensor")     # as in nerf-pytorch's __main__
from run_nerf import config_parser, create_nerf, render  # noqa: E402  (official repo)
from run_nerf_helpers import img2mse, mse2psnr  # noqa: E402

from nvs_metrics import all_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--minutes", type=float, default=60)
ap.add_argument("--max-iters", type=int, default=200000)
ap.add_argument("--path", default="")
ap.add_argument("--video", default="")
ap.add_argument("--video-scale", type=float, default=0.5)
ap.add_argument("--precrop-iters", type=int, default=2000)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

# nerf-pytorch's options, set as in configs/lego.txt (the paper's synthetic-scene settings)
args = config_parser().parse_args([
    "--expname", out.name, "--basedir", str(out.parent), "--datadir", a.data, "--dataset_type", "blender",
    "--no_batching", "--use_viewdirs", "--white_bkgd", "--lrate_decay", "500", "--N_samples", "64",
    "--N_importance", "128", "--N_rand", "1024", "--no_reload"])
render_train, render_test, _, grad_vars, optimizer = create_nerf(args)
render_train["ndc"] = render_test["ndc"] = False
render_train["lindisp"] = render_test["lindisp"] = False

meta = {s: json.load(open(Path(a.data) / f"transforms_{s}.json")) for s in ("train", "test")}
lo, hi = torch.tensor(meta["train"]["aabb"][0]), torch.tensor(meta["train"]["aabb"][1])


def rays_of(W, H, fx, fy, cx, cy, c2w_cv):
    """Rays through the pixel centres; c2w in OpenCV axes (x right, y down, z forward)."""
    j, i = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    d = torch.stack([(i + 0.5 - cx) / fx, (j + 0.5 - cy) / fy, torch.ones_like(i)], -1).reshape(-1, 3)
    c2w = torch.as_tensor(np.asarray(c2w_cv, np.float32))
    return c2w[:3, 3].expand(d.shape[0], 3).clone(), d @ c2w[:3, :3].T


def near_far(o, d):
    """Where each ray enters and leaves the person's box (render() accepts one near/far per ray)."""
    inv = 1.0 / torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
    t0, t1 = (lo - o) * inv, (hi - o) * inv
    n = torch.minimum(t0, t1).amax(-1).clamp_min(0.05)
    f = torch.maximum(torch.maximum(t0, t1).amin(-1), n + 1e-3)
    return n[:, None], f[:, None]


def frame_rays(fr):
    c2w = np.asarray(fr["transform_matrix"]) @ np.diag([1.0, -1.0, -1.0, 1.0])       # OpenGL -> OpenCV axes
    return rays_of(fr["w"], fr["h"], fr["fl_x"], fr["fl_y"], fr["cx"], fr["cy"], c2w)


def load_rgb(fr):
    return torch.as_tensor(np.asarray(Image.open(Path(a.data) / fr["file_path"]).convert("RGB"), np.float32) / 255).reshape(-1, 3)


# all training rays on the GPU (109 photos, ~82M rays), sampled at random like nerf-pytorch's batching mode;
# as nerf-pytorch does for white-background scenes (precrop_frac 0.5), the first steps use only the centre half of
# each photo, otherwise the empty background pulls every density to zero and the network stays white
O, D, C, CENTRE = [], [], [], []
for fr in tqdm(meta["train"]["frames"], desc="rays", leave=False):
    o, d = frame_rays(fr)
    O.append(o), D.append(d), C.append(load_rgb(fr))
    j, i = torch.meshgrid(torch.arange(fr["h"]), torch.arange(fr["w"]), indexing="ij")
    CENTRE.append(((i >= fr["w"] // 4) & (i < 3 * fr["w"] // 4) & (j >= fr["h"] // 4) & (j < 3 * fr["h"] // 4)).reshape(-1))
O, D, C = torch.cat(O), torch.cat(D), torch.cat(C)
CENTRE = torch.nonzero(torch.cat(CENTRE)).squeeze(1)
NEAR, FAR = near_far(O, D)
print("training rays:", O.shape[0], flush=True)


def draw(o, d, chunk=1024 * 32):
    n, f = near_far(o, d)
    rgb, _, _, _ = render(0, 0, None, chunk=chunk, rays=(o, d), near=n, far=f, **render_test)
    return rgb


# ---- training (stops at --minutes or --max-iters)
t0 = time.time()
step = 0
est = a.max_iters
bar = tqdm(total=est, desc="Training")
while step < a.max_iters and time.time() - t0 < a.minutes * 60:
    idx = CENTRE[torch.randint(0, CENTRE.shape[0], (args.N_rand,))] if step < a.precrop_iters else \
        torch.randint(0, O.shape[0], (args.N_rand,))
    rgb, _, _, extras = render(0, 0, None, chunk=args.chunk, rays=(O[idx], D[idx]), near=NEAR[idx], far=FAR[idx],
                               retraw=True, **render_train)
    loss = img2mse(rgb, C[idx])
    psnr = mse2psnr(loss)
    if "rgb0" in extras:
        loss = loss + img2mse(extras["rgb0"], C[idx])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1
    lr = args.lrate * (0.1 ** (step / (args.lrate_decay * 1000)))      # nerf-pytorch's schedule
    for g in optimizer.param_groups:
        g["lr"] = lr
    if step % 100 == 0:
        el = time.time() - t0
        # the bar's total follows the time budget, so the dashboard's ETA is the training time left
        bar.total = min(a.max_iters, int(step * a.minutes * 60 / max(el, 1e-6)))
        bar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr.item():.2f}")
        bar.update(100)
bar.close()
train_seconds = time.time() - t0
ckpt = out / ("%06d.tar" % step)
torch.save({"global_step": step, "network_fn_state_dict": render_train["network_fn"].state_dict(),
            "network_fine_state_dict": render_train["network_fine"].state_dict(),
            "optimizer_state_dict": optimizer.state_dict()}, ckpt)
json.dump({"train_seconds": round(train_seconds), "iterations": step}, open(out / "train_time.json", "w"))
print("trained %d steps in %.0f min" % (step, train_seconds / 60), flush=True)

# ---- held-out photos
import torchvision  # noqa: E402

ev = out / "eval"
(ev / "renders").mkdir(parents=True, exist_ok=True)
(ev / "gt").mkdir(parents=True, exist_ok=True)
per, t_all = {}, 0.0
with torch.no_grad():
    for fr in tqdm(meta["test"]["frames"], desc="test views"):
        o, d = frame_rays(fr)
        torch.cuda.synchronize()
        t1 = time.time()
        img = draw(o, d).clamp(0, 1).reshape(fr["h"], fr["w"], 3).permute(2, 0, 1)
        torch.cuda.synchronize()
        t_all += time.time() - t1
        gt = load_rgb(fr).reshape(fr["h"], fr["w"], 3).permute(2, 0, 1)
        name = Path(fr["file_path"]).stem
        torchvision.utils.save_image(img, ev / "renders" / (name + ".png"))
        torchvision.utils.save_image(gt, ev / "gt" / (name + ".png"))
        per[name] = all_metrics(img, gt)
n_params = sum(p.numel() for p in grad_vars)
res = {"method": "nerf", "test_views": len(per), "n_params": int(n_params), "iterations": step,
       "model_mb": round(ckpt.stat().st_size / 2 ** 20, 1), "fps": round(len(per) / t_all, 3),
       "resolution": [meta["test"]["frames"][0]["w"], meta["test"]["frames"][0]["h"]],
       "mean": {k: float(np.mean([m[k] for m in per.values()])) for k in ("psnr", "ssim", "lpips")}, "per_view": per}

# ---- result video along the preset orbit (half size: NeRF draws slowly)
if a.path and a.video:
    import imageio.v2 as imageio
    frames = []
    with torch.no_grad():
        for c in tqdm(json.load(open(a.path)), desc="video"):
            s = a.video_scale
            W, H = int(round(c["W"] * s)), int(round(c["H"] * s))
            R, t = np.asarray(c["R"]), np.asarray(c["t"])
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
            o, d = rays_of(W, H, c["fx"] * s, c["fy"] * s, c.get("cx", c["W"] / 2) * s, c.get("cy", c["H"] / 2) * s, c2w)
            img = draw(o, d).clamp(0, 1).reshape(H, W, 3)
            frames.append((img.cpu().numpy() * 255).round().astype(np.uint8))
    Path(a.video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(a.video, frames, fps=30, quality=9, macro_block_size=8)
    res["video"] = a.video
json.dump(res, open(ev / "metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_view"}, indent=1))
