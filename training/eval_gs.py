"""Score a trained 3DGS or Mip-Splatting model on the held-out photos and render the result video.

Run from inside the official repository (its modules are imported), with that repository's environment:
    python eval_gs.py --method 3dgs|mip -m <run dir> [--path path.json --video out.mp4] [--check-ours]
Writes <run>/eval/{renders,gt}/*.png and <run>/eval/metrics.json (PSNR, SSIM, LPIPS per photo and mean,
rendering speed, number of Gaussians, model size). --check-ours also renders the test photos with the plain-PyTorch
renderer used by the GUI and reports how close it is to the official CUDA rasteriser.
"""
import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.getcwd())
sys.path.insert(1, str(CODE / "training"))
sys.path.insert(2, str(CODE))
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402  (official repo)
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: E402

from nvs_metrics import all_metrics  # noqa: E402

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--method", required=True, choices=["3dgs", "mip"])
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--path", default="")
parser.add_argument("--video", default="")
parser.add_argument("--check-ours", action="store_true")
parser.add_argument("--zoom", action="store_true", help="also score at 1/4 size and at 1.6x (original photos)")
parser.add_argument("--hires", default="", help="folder of the 1600-px person-on-white photos (RGBA) for --zoom")
parser.add_argument("--quiet", action="store_true")
args = get_combined_args(parser)
dataset, pipe = model.extract(args), pipeline.extract(args)
torch.set_grad_enabled(False)

gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
bg = torch.tensor([1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0], device="cuda")


def draw(view):
    if args.method == "mip":
        return render(view, gaussians, pipe, bg, kernel_size=dataset.kernel_size)["render"]
    return render(view, gaussians, pipe, bg)["render"]


run = Path(dataset.model_path)
out = run / "eval"
(out / "renders").mkdir(parents=True, exist_ok=True)
(out / "gt").mkdir(parents=True, exist_ok=True)
views = scene.getTestCameras()
per = {}
for v in views:
    img = draw(v).clamp(0, 1)
    gt = v.original_image[0:3].cuda()
    torchvision.utils.save_image(img, out / "renders" / (v.image_name + ".png"))
    torchvision.utils.save_image(gt, out / "gt" / (v.image_name + ".png"))
    per[v.image_name] = all_metrics(img, gt)

# speed: render every test view 5 times after a warm-up
draw(views[0])
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    for v in views:
        draw(v)
torch.cuda.synchronize()
fps = 5 * len(views) / (time.time() - t0)

ply = sorted((run / "point_cloud").glob("iteration_*/point_cloud.ply"))[-1]
res = {"method": args.method, "test_views": len(views), "n_gaussians": int(gaussians.get_xyz.shape[0]),
       "model_mb": round(ply.stat().st_size / 2 ** 20, 1), "fps": round(fps, 1),
       "resolution": [int(views[0].image_width), int(views[0].image_height)],
       "mean": {k: float(np.mean([m[k] for m in per.values()])) for k in ("psnr", "ssim", "lpips")}, "per_view": per}

if args.check_ours:   # the GUI's plain-PyTorch renderer against the official CUDA rasteriser, same views
    from modules.nvs.cameras import Camera
    from modules.nvs.gaussians import GaussianModel as OurModel, render as our_render
    ours = OurModel.load(ply, kind=args.method, device=torch.device("cuda"))
    d, t_ours = [], 0.0
    for v in views:
        W, H = int(v.image_width), int(v.image_height)
        cam = Camera(W, H, W / (2 * np.tan(v.FoVx / 2)), H / (2 * np.tan(v.FoVy / 2)), np.asarray(v.R).T, np.asarray(v.T))
        torch.cuda.synchronize()
        t1 = time.time()
        mine = our_render(ours, cam, bg=tuple(bg.tolist())).permute(2, 0, 1)
        torch.cuda.synchronize()
        t_ours += time.time() - t1
        ref = draw(v).clamp(0, 1)
        d.append(10 * np.log10(1.0 / max(torch.mean((mine - ref) ** 2).item(), 1e-12)))
        torchvision.utils.save_image(mine, out / "renders" / (v.image_name + "__ours.png"))
    res["ours_vs_official_psnr"] = {"mean": float(np.mean(d)), "min": float(np.min(d))}
    res["ours_fps_cuda"] = round(len(views) / t_ours, 2)

if args.zoom:   # same views, other image sizes: zoomed out (1/4, ground truth averaged down) and zoomed in (original photos)
    import torch.nn.functional as F
    from PIL import Image
    res["zoom"] = {}
    for label in ("x0.25", "x1.6"):
        ms = []
        for v in views:
            gt = v.original_image[0:3].cuda()
            if label == "x0.25":
                h, w = round(gt.shape[1] / 4), round(gt.shape[2] / 4)
                gt = F.interpolate(gt[None], size=(h, w), mode="area")[0]
            else:
                im = Image.open(Path(args.hires) / Path(v.image_name).name if Path(v.image_name).suffix else
                                next(Path(args.hires).glob(v.image_name + ".*"))).convert("RGBA")
                im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
                gt = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1).cuda()
                h, w = gt.shape[1], gt.shape[2]
            cam = MiniCam(w, h, v.FoVy, v.FoVx, v.znear, v.zfar, v.world_view_transform, v.full_proj_transform)
            ms.append(all_metrics(draw(cam).clamp(0, 1), gt))
            if v is views[0]:
                torchvision.utils.save_image(draw(cam).clamp(0, 1), out / ("zoom_%s_%s.png" % (label, Path(v.image_name).stem)))
        res["zoom"][label] = {k: float(np.mean([m[k] for m in ms])) for k in ("psnr", "ssim", "lpips")}

if args.path and args.video:
    import imageio.v2 as imageio
    cams = json.load(open(args.path))
    frames = []
    for c in cams:
        Rc, tc = np.asarray(c["R"]), np.asarray(c["t"])
        fovx, fovy = 2 * np.arctan(c["W"] / (2 * c["fx"])), 2 * np.arctan(c["H"] / (2 * c["fy"]))
        wv = torch.tensor(getWorld2View2(Rc.T, tc)).transpose(0, 1).cuda()
        proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        img = draw(MiniCam(c["W"], c["H"], fovy, fovx, 0.01, 100.0, wv, full)).clamp(0, 1)
        frames.append((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.video, frames, fps=30, quality=9, macro_block_size=8)
    res["video"] = str(args.video)

json.dump(res, open(out / "metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_view"}, indent=1))
