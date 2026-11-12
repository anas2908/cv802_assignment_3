"""Collect the trained models and results into the folder the GUI opens (results/ in the repository).

    python training/export_bundle.py <SP3 root> <out folder> [--sp2-hub <cv802_assignment_2>/surface/ui/hub]

The whole repository has to stay under 100 MB (it is also submitted as one zip), so the models are packed
(modules/nvs/packing.py) and every packed model is rendered on held-out photos and compared with the official
renders, so the packing cannot change a result unnoticed:

  * Gaussians whose opacity is below 1/255 are dropped: the official rasterisers skip any splat with alpha < 1/255,
    so these never draw a pixel (lossless).
  * Gaussian positions float32; scales, rotations, opacity and the SH degree-0 colour float16; the higher SH bands
    as 8-bit codes. The renders match the official ones to ~70 dB PSNR (identical 8-bit images).
  * TensoRF: the density and appearance planes as 8-bit codes per feature channel (~60-65 dB, identical images).
  * NeRF: float32 weights as trained.
  * Room (extra, 0.9M Gaussians): spherical harmonics up to degree 1 only, positions as 16-bit codes in the scene
    box and rotations as 8-bit codes. Its renders differ from the official ones (~26 dB PSNR), mostly in
    view-dependent shading; all reported scores come from the official models and renderers.
  * Videos re-encoded with H.264 (CRF 20, room CRF 23).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from modules.nvs.cameras import Camera  # noqa: E402
from modules.nvs.gaussians import GaussianModel, read_ply_vertices, render as render_gs  # noqa: E402
from modules.nvs.nerf import NeRFModel, convert_checkpoint as convert_nerf, render as render_nerf  # noqa: E402
from modules.nvs.packing import quantize  # noqa: E402
from modules.nvs.tensorf import TensoRFModel, convert_checkpoint, render as render_tf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sp3")
ap.add_argument("out")
ap.add_argument("--sp2-hub", default="/home/anas.khan/cv802_surface_repo/surface/ui/hub")
a = ap.parse_args()
SP3, OUT = Path(a.sp3), Path(a.out)
for sub in ("models", "photos", "meshes", "videos"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

MODELS = [  # id, label, run folder, kind, background, packing options
    ("nerf", "NeRF", "nerf_person", "nerf", [1.0, 1.0, 1.0], {}),
    ("3dgs", "3D Gaussian Splatting", "3dgs_person_rand", "3dgs", [1.0, 1.0, 1.0], {}),   # random start: images only
    ("mip", "Mip-Splatting", "mip_person_rand", "mip", [1.0, 1.0, 1.0], {}),
    ("tensorf", "TensoRF", "tensorf_person", "tensorf", [1.0, 1.0, 1.0], {}),
    ("3dgs_full", "3DGS, full room (extra)", "3dgs_full_rand", "3dgs", [0.0, 0.0, 0.0], {"sh_coeffs": 4, "compact": True}),
]
ABLATION = [  # the same Gaussian runs started from the Sub-project 1 point cloud (scores only, no model shipped)
    ("3dgs_pointcloud_start", "3dgs_person"), ("mip_pointcloud_start", "mip_person"), ("3dgs_full_pointcloud_start", "3dgs_full"),
]


def gs_npz(ply, out, kind, sh_coeffs=16, compact=False):
    """official Gaussian .ply -> packed .npz (see the module docstring); returns (Gaussians kept, Gaussians trained)"""
    v = read_ply_vertices(ply)
    n = v.dtype.names
    rest = sorted([k for k in n if k.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    sh = np.concatenate([np.stack([v["f_dc_%d" % i] for i in range(3)], 1)[:, None, :],
                         np.stack([v[k] for k in rest], 1).reshape(len(v), 3, -1).transpose(0, 2, 1)], 1)
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    scales = np.stack([v["scale_%d" % i] for i in range(3)], 1).astype(np.float32)
    rots = np.stack([v["rot_%d" % i] for i in range(4)], 1).astype(np.float32)
    opac = v["opacity"][:, None].astype(np.float32)
    filt = v["filter_3D"][:, None].astype(np.float32) if "filter_3D" in n else None
    op = 1.0 / (1.0 + np.exp(-opac[:, 0].astype(np.float64)))
    if filt is not None:                       # Mip-Splatting's 3D filter lowers the opacity it draws with
        s2, f2 = np.exp(scales.astype(np.float64)) ** 2, filt.astype(np.float64) ** 2
        op = op * np.sqrt(np.prod(s2, 1) / np.prod(s2 + f2, 1))
    keep = op >= 1.0 / 255.0
    arrs = {"scales": scales[keep].astype(np.float16), "opacity": opac[keep].astype(np.float16),
            "sh_dc": sh[keep, :1].astype(np.float16)}
    if compact:
        q = rots[keep] / np.linalg.norm(rots[keep], axis=1, keepdims=True)
        arrs.update(quantize("means", means[keep], 16, 1))
        arrs.update(quantize("rots", q, 8, 1))
    else:
        arrs.update(means=means[keep], rots=rots[keep].astype(np.float16))
    if sh_coeffs > 1:
        arrs.update(quantize("sh_rest", sh[keep, 1:sh_coeffs].reshape(int(keep.sum()), -1), 8, 1))
    if filt is not None:
        arrs["filter_3d"] = filt[keep].astype(np.float16)
    np.savez_compressed(out, kind=kind, **arrs)
    return int(keep.sum()), len(v)


def tensorf_npz(ck, out):
    z = convert_checkpoint(ck)
    packed = {}
    for k, val in z.items():
        if k.startswith(("density_plane.", "app_plane.")):    # (1, channels, H, W): one range per channel
            packed.update(quantize(k, val, 8, 1))
        else:
            packed[k] = val
    np.savez_compressed(out, **packed)


def cameras():
    cams, split = [], {}
    for s in ("train", "test"):
        meta = json.load(open(SP3 / "data" / "person" / f"transforms_{s}.json"))
        split[s] = []
        for f in meta["frames"]:
            c2w = np.asarray(f["transform_matrix"]) @ np.diag([1.0, -1.0, -1.0, 1.0])      # OpenGL -> OpenCV axes
            R = c2w[:3, :3].T
            name = Path(f["file_path"]).name
            cams.append(Camera(f["w"], f["h"], f["fl_x"], f["fl_y"], R, -R @ c2w[:3, 3], f["cx"], f["cy"], name).to_dict())
            split[s].append(name)
    return cams, split


def psnr(x, y):
    return float(10 * np.log10(1.0 / max(np.mean((x - y) ** 2), 1e-12)))


import imageio.v2 as imageio  # noqa: E402

cams, split = cameras()
by_name = {c["name"]: Camera.from_dict(c) for c in cams}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models, fidelity = {}, {}
for mid, label, run, kind, bg, opts in MODELS:
    rdir = SP3 / "runs" / run
    out = OUT / "models" / (mid + ".npz")
    if kind == "nerf":
        cks = sorted(rdir.glob("*.tar"))
        if not cks:
            print("skip", mid, "(not trained)")
            continue
        aabb = json.load(open(SP3 / "data" / "person" / "transforms_train.json"))["aabb"]
        convert_nerf(cks[-1], aabb, out)
        model = NeRFModel.load(out, dev)
    elif kind == "tensorf":
        ck = rdir / (run + ".th")
        if not ck.exists():
            print("skip", mid, "(not trained)")
            continue
        tensorf_npz(ck, out)
        model = TensoRFModel.load(out, dev)
    else:
        plys = sorted(rdir.glob("point_cloud/iteration_*/point_cloud.ply"))
        if not plys:
            print("skip", mid, "(not trained)")
            continue
        kept, total = gs_npz(plys[-1], out, kind, **opts)
        model = GaussianModel.load(out, kind=kind, device=dev)
    models[mid] = {"label": label, "file": "models/" + out.name, "kind": kind, "background": bg,
                   "size_mb": round(out.stat().st_size / 1e6, 1)}
    if kind not in ("tensorf", "nerf"):
        models[mid]["gaussians"] = kept
        models[mid]["dropped_invisible"] = total - kept
        if model.sh_degree < 3:
            models[mid]["note"] = "spherical harmonics up to degree %d only (the trained model has degree 3), to keep the repository small" % model.sh_degree
    # packing check: our renderer on the packed model vs the official render, on 4 held-out photos
    ev = rdir / "eval" / "renders"
    checks = []
    for name in split["test"][::4]:
        ref = next((r for r in (ev / (Path(name).stem + ".png"), ev / (name + ".png")) if r.exists()), None)   # 3DGS keeps the extension
        if ref is None:
            continue
        r = imageio.imread(ref)[..., :3].astype(np.float32) / 255
        c = by_name[name]
        with torch.no_grad():
            if kind == "tensorf":
                img = render_tf(model, c)
            elif kind == "nerf":
                img = render_nerf(model, c)
            else:   # the official Gaussian rasterisers use the image centre as principal point
                img = render_gs(model, Camera(c.W, c.H, c.fx, c.fy, c.R, c.t), bg=tuple(bg))
        img = (img.clamp(0, 1) * 255).round().cpu().numpy() / 255          # as the GUI shows it (8 bit)
        checks.append(psnr(img, r))
    if checks:
        fidelity[mid] = {"min": round(min(checks), 1), "mean": round(float(np.mean(checks)), 1), "views": len(checks)}
    print(mid, models[mid], "PSNR vs official renders:", fidelity.get(mid), flush=True)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

# point cloud for orientation (subsampled person cloud) and for the GUI's Sub-project 2 mesh extraction
from plyfile import PlyData, PlyElement  # noqa: E402
pv = PlyData.read(str(SP3 / "data" / "person" / "sparse" / "0" / "points3D.ply"))["vertex"]
idx = np.random.default_rng(0).choice(len(pv), min(80000, len(pv)), replace=False)
v = np.zeros(len(idx), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
for k in ("x", "y", "z", "red", "green", "blue"):
    v[k] = np.asarray(pv[k])[idx]
PlyData([PlyElement.describe(v, "vertex")]).write(str(OUT / "points.ply"))

# held-out photos (to compare a render with the real photo in the GUI)
for name in split["test"]:
    imageio.imwrite(OUT / "photos" / (Path(name).stem + ".jpg"),
                    imageio.imread(SP3 / "data" / "person" / "images" / name)[..., :3], quality=92)

# Sub-project 2 meshes, rasterised by the GUI: the viewer-ready textured meshes of the SP2 results, kept in the
# cameras' world frame (the SP2 web page rotated its copies, so they are rebuilt here from the OBJ)
meshes = []
sp2 = Path(a.sp2_hub).resolve().parents[2]
zf = sp2 / "results" / "vggsfm.zip"
if zf.exists():
    import io as _io
    import zipfile
    from PIL import Image
    sys.path.insert(0, str(Path(a.sp2_hub)))
    from build_hub_data import glb, read_obj, vertex_normals  # noqa: E402   (Sub-project 2's GLB writer)
    z = zipfile.ZipFile(zf)
    for key, label in (("poisson", "Screened Poisson (own, SP2)"), ("nksr", "NKSR (SP2)")):
        base = next((n.rsplit("/", 1)[0] for n in z.namelist() if n.startswith("vggsfm/%s/" % key) and n.endswith("/mesh.obj")), None)
        if base is None:
            continue
        V, VT, FV, FT = read_obj(z.read(base + "/mesh.obj").decode())
        uniq, inv = np.unique(np.stack([FV.ravel(), FT.ravel()], 1), axis=0, return_inverse=True)
        uv = VT[uniq[:, 1]].copy()
        uv[:, 1] = 1 - uv[:, 1]
        img = Image.open(_io.BytesIO(z.read(base + "/texture.jpg"))).convert("RGB").resize((2048, 2048), Image.LANCZOS)
        jb = _io.BytesIO()
        img.save(jb, "JPEG", quality=85)
        nrm = vertex_normals(V, FV)[uniq[:, 0]].astype(np.float32)          # for lighting in the 3D view
        b = glb([("POSITION", V[uniq[:, 0]].astype(np.float32), 5126, False), ("NORMAL", nrm, 5126, False),
                 ("TEXCOORD_0", uv.astype(np.float32), 5126, False)],
                indices=inv.reshape(-1, 3), image=jb.getvalue())
        (OUT / "meshes" / (key + ".glb")).write_bytes(b)
        meshes.append({"label": label, "file": "meshes/%s.glb" % key, "triangles": int(len(FV))})

# result videos along the preset orbit (rendered by the evaluation scripts), re-encoded small
import imageio_ffmpeg  # noqa: E402
shutil.copy2(SP3 / "data" / "orbit_path.json", OUT / "orbit_path.json")
for old in (OUT / "videos").glob("*.mp4"):
    old.unlink()
for mid, label, run, kind, bg, opts in MODELS:
    mp4 = SP3 / "results" / "videos" / (run + ".mp4")
    if mp4.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4), "-c:v", "libx264",
                        "-preset", "slow", "-crf", "23" if mid == "3dgs_full" else "20", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(OUT / "videos" / (mid + ".mp4"))], check=True)


# scores: the GUI's copy in scene.json, and every run (main + point-cloud-start ablation, per photo) in metrics.json
def run_metrics(run, per_view=True):
    m = SP3 / "runs" / run / "eval" / "metrics.json"
    if not m.exists():
        return None
    d = json.load(open(m))
    d.pop("video", None)
    if not per_view:
        d.pop("per_view", None)
    t = SP3 / "runs" / run / "train_time.json"
    if t.exists():
        d.update(json.load(open(t)))
    d["run"] = run
    return d


metrics = {mid: run_metrics(run, False) for mid, _, run, *_ in MODELS if run_metrics(run, False)}
everything = {mid: run_metrics(run) for mid, _, run, *_ in MODELS}
everything.update({k: run_metrics(run) for k, run in ABLATION})
json.dump({"note": "16 held-out photos (every 8th), 750x1000; PSNR [dB], SSIM, LPIPS (VGG); fps on one RTX 5000 Ada; "
                   "*_pointcloud_start = ablation, same settings but Gaussians started from the Sub-project 1 point cloud",
           "runs": {k: v for k, v in everything.items() if v}}, open(OUT / "metrics.json", "w"), indent=1)
scene = json.load(open(SP3 / "data" / "scenes.json"))
json.dump({"cameras": cams, "split": split, "centre": scene["centre"], "up": scene["up"], "points": "points.ply",
           "orbit_path": "orbit_path.json", "models": models, "meshes": meshes, "metrics": metrics,
           "packed_vs_official_psnr": fidelity}, open(OUT / "scene.json", "w"), indent=1)
print(json.dumps({"models": list(models), "meshes": len(meshes), "fidelity": fidelity}))
