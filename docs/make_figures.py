"""Figures for the report and the slides, from the evaluation outputs (run on the cluster after training).

    python docs/make_figures.py <SP3 root>          -> docs/figs/*.jpg + docs/figs/numbers.json
"""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SP3 = Path(sys.argv[1])
OUT = ROOT / "docs" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {"nerf": "nerf_person", "tensorf": "tensorf_person", "3dgs": "3dgs_person_rand", "mip": "mip_person_rand"}
ABLATION = {"3dgs_pcl": "3dgs_person", "mip_pcl": "mip_person", "3dgs_full_pcl": "3dgs_full"}   # started from the SP1 point cloud
LABEL = {"gt": "Real photo (held out)", "nerf": "NeRF", "tensorf": "TensoRF", "3dgs": "3D Gaussian Splatting", "mip": "Mip-Splatting"}
try:
    FONT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 30)
except OSError:
    FONT = ImageFont.load_default()
split = json.load(open(SP3 / "data" / "person" / "split.json"))
test = [Path(n).stem for n in split["test"]]


def find(run, stem, sub="renders"):
    d = SP3 / "runs" / run / "eval" / sub
    for p in (d / (stem + ".png"), d / (stem + ".jpg.png")):
        if p.exists():
            return p
    return None


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def person_box(img, pad=12):
    m = (np.abs(img.astype(int) - 255).sum(-1) > 30)
    ys, xs = np.nonzero(m)
    return max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad), max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)


def label(img, text, h=44):
    im = Image.new("RGB", (img.shape[1], img.shape[0] + h), "white")
    im.paste(Image.fromarray(img), (0, h))
    d = ImageDraw.Draw(im)
    w = d.textlength(text, font=FONT)
    d.text(((img.shape[1] - w) / 2, 6), text, fill=(40, 40, 40), font=FONT)
    return np.asarray(im)


def hcat(imgs, gap=10):
    h = max(i.shape[0] for i in imgs)
    out = []
    for k, i in enumerate(imgs):
        if i.shape[0] < h:
            i = np.concatenate([i, np.full((h - i.shape[0], i.shape[1], 3), 255, np.uint8)])
        out.append(i)
        if k < len(imgs) - 1:
            out.append(np.full((h, gap, 3), 255, np.uint8))
    return np.concatenate(out, 1)


def vcat(imgs, gap=10):
    w = max(i.shape[1] for i in imgs)
    out = []
    for k, i in enumerate(imgs):
        if i.shape[1] < w:
            i = np.concatenate([i, np.full((i.shape[0], w - i.shape[1], 3), 255, np.uint8)], 1)
        out.append(i)
        if k < len(imgs) - 1:
            out.append(np.full((gap, w, 3), 255, np.uint8))
    return np.concatenate(out, 0)


def save(img, name, width=None):
    im = Image.fromarray(img)
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(OUT / name, quality=90)


methods = [m for m in ("nerf", "tensorf", "3dgs", "mip") if find(RUNS[m], test[0])]

# 1. held-out photos next to the three methods: one horizontal strip, a short label under every crop
SHORT = {"gt": "Photo", "nerf": "NeRF", "tensorf": "TensoRF", "3dgs": "3DGS", "mip": "Mip-Splat."}


def caption_below(img, text, h=40):
    im = Image.new("RGB", (img.shape[1], img.shape[0] + h), "white")
    im.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(im)
    d.text(((img.shape[1] - d.textlength(text, font=FONT)) / 2, img.shape[0] + 4), text, fill=(40, 40, 40), font=FONT)
    return np.asarray(im)


def strip(stems, H=640):
    groups = []
    for stem in stems:
        gt = load(find(RUNS[methods[0]], stem, "gt"))
        y0, y1, x0, x1 = person_box(gt)
        tiles = [gt[y0:y1, x0:x1]] + [load(find(RUNS[m], stem))[y0:y1, x0:x1] for m in methods]
        tiles = [np.asarray(Image.fromarray(t).resize((round(t.shape[1] * H / t.shape[0]), H), Image.LANCZOS)) for t in tiles]
        groups.append(hcat([caption_below(t, SHORT[k]) for t, k in zip(tiles, ["gt"] + methods)], 4))
    return hcat(groups, 40)


picks = [test[0], test[4], test[9]] if len(test) > 9 else test[:3]
save(strip(picks), "fig_strip.jpg", 3000)          # report: three views
save(strip(picks[:2]), "fig_compare.jpg", 2600)    # slides: two views

# 2. zoomed out to 1/4: aliasing of 3DGS vs Mip-Splatting (crop of the chest, enlarged with square pixels)
z = {}
for m in ("3dgs", "mip"):
    p = sorted((SP3 / "runs" / RUNS[m] / "eval").glob("zoom_x0.25_*.png"))
    if p:
        z[m] = load(p[0])
if len(z) == 2:
    stem = Path(sorted((SP3 / "runs" / RUNS["3dgs"] / "eval").glob("zoom_x0.25_*.png"))[0]).stem.replace("zoom_x0.25_", "").replace(".jpg", "")
    gt = Image.open(find(RUNS["3dgs"], stem, "gt")).convert("RGB")
    gt = np.asarray(gt.resize(z["3dgs"].shape[1::-1][::1], Image.BOX))
    y0, y1, x0, x1 = person_box(gt, 2)
    y1 = y0 + (y1 - y0) * 45 // 100                                     # head and shirt, where aliasing shows
    crop = lambda a: np.asarray(Image.fromarray(a[y0:y1, x0:x1]).resize(((x1 - x0) * 6, (y1 - y0) * 6), Image.NEAREST))
    save(hcat([label(crop(gt), "Photo"), label(crop(z["3dgs"]), "3DGS"), label(crop(z["mip"]), "Mip-Splatting")]),
         "fig_zoom.jpg", 2400)

# 3. frames of the result videos (preset orbit), one row per method
for m, run in list(RUNS.items()) + [("3dgs_full", "3dgs_full_rand")]:
    v = SP3 / "results" / "videos" / (run + ".mp4")
    if not v.exists():
        continue
    r = imageio.get_reader(str(v))
    fr = [f for i, f in enumerate(r) if i in (0, 40, 80, 120, 160, 200)]
    save(hcat(fr), "fig_orbit_%s.jpg" % m, 2400)

# 4. the inputs: a sample of the 125 photos
imgs = sorted((SP3 / "data" / "full" / "images").iterdir())
sample = [np.asarray(Image.open(p).convert("RGB").resize((150, 200) if Image.open(p).height > Image.open(p).width else (200, 150))) for p in imgs[::6][:20]]
rows = [hcat(sample[k:k + 10], 6) for k in range(0, len(sample), 10)]
save(vcat(rows, 6), "fig_inputs.jpg", 2400)

# numbers for the slides and the report
num = {}
for m, run in list(RUNS.items()) + [("3dgs_full", "3dgs_full_rand")] + list(ABLATION.items()):
    f = SP3 / "runs" / run / "eval" / "metrics.json"
    if f.exists():
        d = json.load(open(f))
        d.pop("per_view", None)
        t = SP3 / "runs" / run / "train_time.json"
        d["train_seconds"] = json.load(open(t))["train_seconds"] if t.exists() else None
        num[m] = d
json.dump(num, open(OUT / "numbers.json", "w"), indent=1)
print("figures:", sorted(p.name for p in OUT.glob("*.jpg")))
