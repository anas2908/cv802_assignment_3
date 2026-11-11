"""Collect the trained models and results into the folder the GUI opens (results/ in the repository).

    python training/export_bundle.py <SP3 root> <out folder> [--sp2-hub <cv802_assignment_2>/surface/ui/hub]

Models are stored compactly (.npz): Gaussian positions stay float32 (float16 would move them by ~2 mm, about a
pixel), everything else float16. Each converted model is rendered on two held-out photos and compared with the
official renders, so a conversion error cannot go unnoticed.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from modules.nvs.cameras import Camera  # noqa: E402
from modules.nvs.gaussians import GaussianModel, read_ply_vertices, render as render_gs  # noqa: E402
from modules.nvs.nerf import NeRFModel, convert_checkpoint as convert_nerf, render as render_nerf  # noqa: E402
from modules.nvs.tensorf import TensoRFModel, convert_checkpoint, render as render_tf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sp3")
ap.add_argument("out")
ap.add_argument("--sp2-hub", default="/home/anas.khan/cv802_surface_repo/surface/ui/hub")
a = ap.parse_args()
SP3, OUT = Path(a.sp3), Path(a.out)
for sub in ("models", "photos", "meshes", "videos"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

MODELS = [  # id, label, run folder, kind, background
    ("nerf", "NeRF", "nerf_person", "nerf", [1.0, 1.0, 1.0]),
    ("3dgs", "3D Gaussian Splatting", "3dgs_person_rand", "3dgs", [1.0, 1.0, 1.0]),     # random start: images only
    ("mip", "Mip-Splatting", "mip_person_rand", "mip", [1.0, 1.0, 1.0]),
    ("tensorf", "TensoRF", "tensorf_person", "tensorf", [1.0, 1.0, 1.0]),
    ("3dgs_full", "3DGS, full room (extra)", "3dgs_full_rand", "3dgs", [0.0, 0.0, 0.0]),
]


def gs_npz(ply, out, kind, sh_coeffs=16):
    """sh_coeffs=4 keeps spherical harmonics up to degree 1 only (smaller file, slightly flatter view-dependent colour)."""
    v = read_ply_vertices(ply)
    n = v.dtype.names
    rest = sorted([k for k in n if k.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    sh = np.concatenate([np.stack([v["f_dc_%d" % i] for i in range(3)], 1)[:, None, :],
                         np.stack([v[k] for k in rest], 1).reshape(len(v), 3, -1).transpose(0, 2, 1)], 1)
    arrs = {"means": np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32),
            "scales": np.stack([v["scale_%d" % i] for i in range(3)], 1).astype(np.float16),
            "rots": np.stack([v["rot_%d" % i] for i in range(4)], 1).astype(np.float16),
            "opacity": v["opacity"][:, None].astype(np.float16), "sh": sh[:, :sh_coeffs].astype(np.float16)}
    if "filter_3D" in n:
        arrs["filter_3d"] = v["filter_3D"][:, None].astype(np.float16)
    np.savez_compressed(out, kind=kind, **arrs)
    return len(v)


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


cams, split = cameras()
by_name = {c["name"]: Camera.from_dict(c) for c in cams}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models, fidelity = {}, {}
for mid, label, run, kind, bg in MODELS:
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
        convert_checkpoint(ck, out)
        model = TensoRFModel.load(out, dev)
    else:
        plys = sorted(rdir.glob("point_cloud/iteration_*/point_cloud.ply"))
        if not plys:
            print("skip", mid, "(not trained)")
            continue
        gs_npz(plys[-1], out, kind)
        for coeffs in (4, 1):                          # GitHub refuses files over 100 MB: SH degree 1, then 0
            if out.stat().st_size > 90 * 2 ** 20:
                gs_npz(plys[-1], out, kind, sh_coeffs=coeffs)
        model = GaussianModel.load(out, kind=kind, device=dev)
    models[mid] = {"label": label, "file": "models/" + out.name, "kind": kind, "background": bg,
                   "size_mb": round(out.stat().st_size / 2 ** 20, 1)}
    if kind not in ("tensorf", "nerf") and model.sh_degree < 3:
        models[mid]["note"] = "stored with spherical harmonics up to degree %d to stay under GitHub's file limit" % model.sh_degree
    # conversion check against the official renders of two held-out photos (person models)
    ev = rdir / "eval" / "renders"
    checks = []
    for name in split["test"][:2]:
        ref = next((r for r in (ev / (Path(name).stem + ".png"), ev / (name + ".png")) if r.exists()), None)   # 3DGS keeps the extension
        if ref is None:
            continue
        import imageio.v2 as imageio
        r = imageio.imread(ref)[..., :3].astype(np.float32) / 255
        c = by_name[name]
        if kind == "tensorf":
            img = render_tf(model, c).cpu().numpy()
        elif kind == "nerf":
            img = render_nerf(model, c).cpu().numpy()
        else:   # the official Gaussian rasterisers use the image centre as principal point
            img = render_gs(model, Camera(c.W, c.H, c.fx, c.fy, c.R, c.t), bg=tuple(bg)).cpu().numpy()
        checks.append(psnr(img, r))
    if checks:
        fidelity[mid] = round(min(checks), 1)
    print(mid, models[mid], "conversion check (PSNR vs official render):", checks)

# point cloud for orientation (subsampled person cloud)
from plyfile import PlyData, PlyElement  # noqa: E402
pv = PlyData.read(str(SP3 / "data" / "person" / "sparse" / "0" / "points3D.ply"))["vertex"]
idx = np.random.default_rng(0).choice(len(pv), min(80000, len(pv)), replace=False)
v = np.zeros(len(idx), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
for k in ("x", "y", "z", "red", "green", "blue"):
    v[k] = np.asarray(pv[k])[idx]
PlyData([PlyElement.describe(v, "vertex")]).write(str(OUT / "points.ply"))

# held-out photos (to compare a render with the real photo in the GUI)
import imageio.v2 as imageio  # noqa: E402
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

shutil.copy2(SP3 / "data" / "orbit_path.json", OUT / "orbit_path.json")
for old in (OUT / "videos").glob("*.mp4"):
    old.unlink()
for mid, label, run, kind, bg in MODELS:        # the main runs' orbit videos, named by method
    mp4 = SP3 / "results" / "videos" / (run + ".mp4")
    if mp4.exists():
        shutil.copy2(mp4, OUT / "videos" / (mid + ".mp4"))

metrics = {}
for mid, label, run, kind, bg in MODELS:
    m = SP3 / "runs" / run / "eval" / "metrics.json"
    if m.exists():
        d = json.load(open(m))
        d.pop("per_view", None)
        t = SP3 / "runs" / run / "train_time.json"
        if t.exists():
            d["train_seconds"] = json.load(open(t))["train_seconds"]
        metrics[mid] = d
scene = json.load(open(SP3 / "data" / "scenes.json"))
json.dump({"cameras": cams, "split": split, "centre": scene["centre"], "up": scene["up"], "points": "points.ply",
           "orbit_path": "orbit_path.json", "models": models, "meshes": meshes, "metrics": metrics,
           "conversion_check_psnr": fidelity}, open(OUT / "scene.json", "w"), indent=1)
print(json.dumps({"models": list(models), "meshes": len(meshes), "fidelity": fidelity}))
