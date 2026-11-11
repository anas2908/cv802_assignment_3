"""Score a trained TensoRF model on the held-out photos and render the result video (same outputs as eval_gs.py).

Run from inside the official TensoRF repository with the gs environment:
    python eval_tensorf.py --ckpt <run>/tensorf_person.th --data <DATA>/person --out <run>/eval [--path p.json --video v.mp4]
TensoRF's own evaluation assumes one image size, so the photos (4 lenses, two orientations) are rendered one by one.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.getcwd())
sys.path.insert(1, str(CODE / "training"))
from dataLoader.percam import PerCamDataset  # noqa: E402  (registered by setup_envs.sh)
from models.tensoRF import TensorVMSplit  # noqa: E402
from renderer import OctreeRender_trilinear_fast  # noqa: E402

from nvs_metrics import all_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--path", default="")
ap.add_argument("--video", default="")
a = ap.parse_args()
torch.set_grad_enabled(False)
dev = torch.device("cuda")

ckpt = torch.load(a.ckpt, map_location=dev)
kw = ckpt["kwargs"]
kw.update({"device": dev})
model = TensorVMSplit(**kw)
model.load(ckpt)
test = PerCamDataset(a.data, split="test", is_stack=True)
out = Path(a.out)
(out / "renders").mkdir(parents=True, exist_ok=True)
(out / "gt").mkdir(parents=True, exist_ok=True)


def draw(rays, w, h):
    rgb, *_ = OctreeRender_trilinear_fast(rays.to(dev), model, chunk=16384, N_samples=-1, ndc_ray=False,
                                          white_bg=test.white_bg, device=dev)
    return rgb.clamp(0, 1).reshape(h, w, 3).permute(2, 0, 1)


per, t_all = {}, 0.0
for rays, rgbs, (w, h), p in zip(test.all_rays, test.all_rgbs, test.img_whs, test.image_paths):
    name = Path(p).stem
    torch.cuda.synchronize()
    t0 = time.time()
    img = draw(rays, w, h)
    torch.cuda.synchronize()
    t_all += time.time() - t0
    gt = rgbs.reshape(h, w, 3).permute(2, 0, 1).to(dev)
    torchvision.utils.save_image(img, out / "renders" / (name + ".png"))
    torchvision.utils.save_image(gt, out / "gt" / (name + ".png"))
    per[name] = all_metrics(img, gt)

res = {"method": "tensorf", "test_views": len(per), "n_params": int(sum(p.numel() for p in model.parameters())),
       "model_mb": round(Path(a.ckpt).stat().st_size / 2 ** 20, 1), "fps": round(len(per) / t_all, 2),
       "resolution": list(test.img_whs[0]),
       "mean": {k: float(np.mean([m[k] for m in per.values()])) for k in ("psnr", "ssim", "lpips")}, "per_view": per}

if a.path and a.video:
    import imageio.v2 as imageio
    frames = []
    for c in json.load(open(a.path)):
        W, H = c["W"], c["H"]
        j, i = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
        d = torch.stack([(i + 0.5 - c["cx"]) / c["fx"], (j + 0.5 - c["cy"]) / c["fy"], torch.ones_like(i)], -1)
        d = d / d.norm(dim=-1, keepdim=True)
        R, t = torch.tensor(c["R"], dtype=torch.float32), torch.tensor(c["t"], dtype=torch.float32)
        rays_d = (d.reshape(-1, 3) @ R)            # camera -> world: R^T d, written as d @ R
        rays_o = (-R.T @ t).expand_as(rays_d)
        img = draw(torch.cat([rays_o, rays_d], 1), W, H)
        frames.append((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    Path(a.video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(a.video, frames, fps=30, quality=9, macro_block_size=8)
    res["video"] = a.video

json.dump(res, open(out / "metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_view"}, indent=1))
