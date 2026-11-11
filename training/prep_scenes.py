"""Build the two training scenes from the Project-1 VGGSfM cameras and the 125 light-shirt photos.

Input: the undistorted, person-masked scene made for 2DGS in Sub-project 2 (same photos, same cameras).

DATA/person/  images/*.png  RGB, person composited on white (person mask), max side --max-side
              sparse/0/{cameras,images}.bin  undistorted PINHOLE cameras; points3D.ply initial points (our cloud)
              transforms_{train,test}.json   the same cameras for TensoRF (per-frame intrinsics, OpenGL c2w)
DATA/full/    images/*.jpg  undistorted photos with the room, same cameras; points3D.ply = sparse points + a
              coarse random fill of the room box (3DGS needs some points where the background is)
Both: split.json — test = every 8th image in name order (the 3DGS/Mip-Splatting --eval rule), the rest train.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="Sub-project 2 2DGS scene folder (images/, sparse/0, _und/, scene.json)")
ap.add_argument("--out", required=True)
ap.add_argument("--max-side", type=int, default=1000)
ap.add_argument("--hold", type=int, default=8)
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)
rec = pycolmap.Reconstruction(str(src / "sparse" / "0"))
scene = json.load(open(src / "scene.json"))


def pose(im):
    cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    return cfw.rotation.matrix(), np.asarray(cfw.translation)


ims = sorted(rec.images.values(), key=lambda im: Path(im.name).name)
names = [Path(im.name).name for im in ims]
test = [n for i, n in enumerate(names) if i % a.hold == 0]
train = [n for n in names if n not in test]


def resized(img):
    s = min(1.0, a.max_side / max(img.size))
    return img if s >= 1 else img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)


def write_ply(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    v = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = np.asarray(xyz, np.float32).T
    v["red"], v["green"], v["blue"] = np.asarray(rgb, np.uint8).T
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def common(dst):
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "sparse" / "0").mkdir(parents=True)
    for f in ("cameras.bin", "images.bin"):
        shutil.copy2(src / "sparse" / "0" / f, dst / "sparse" / "0" / f)
    json.dump({"train": train, "test": test, "hold": a.hold}, open(dst / "split.json", "w"), indent=1)


# ---- person on white
P = out / "person"
common(P)
shutil.copy2(src / "sparse" / "0" / "points3D.ply", P / "sparse" / "0" / "points3D.ply")
sizes = {}
for n in names:
    # RGB already composited on white: with RGBA input the official COLMAP loaders apply the mask to the render
    # (3DGS) or to the ground truth (Mip-Splatting), which fights a white background
    img = resized(Image.open(src / "images" / n).convert("RGBA"))
    img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img).convert("RGB")
    img.save(P / "images" / n, format="PNG")          # PNG data under the original name, as 3DGS looks it up
    sizes[n] = img.size

# TensoRF: per-frame pinhole intrinsics at the saved size, camera-to-world in OpenGL axes (x right, y up, z back)
cams = {}
for im in ims:
    c = rec.cameras[im.camera_id]
    fx, fy, cx, cy = (c.params[0], c.params[0], c.params[1], c.params[2]) if len(c.params) == 3 else c.params[:4]
    n = Path(im.name).name
    w, h = sizes[n]
    sx, sy = w / c.width, h / c.height
    R, t = pose(im)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
    c2w[:3, 1:3] *= -1                                  # OpenCV -> OpenGL camera axes
    cams[n] = {"file_path": "images/" + n, "w": w, "h": h, "fl_x": fx * sx, "fl_y": fy * sy, "cx": cx * sx, "cy": cy * sy,
               "transform_matrix": c2w.tolist()}
pts = np.asarray([p.xyz for p in rec.points3D.values()]) if len(rec.points3D) else None
centre, diag = np.asarray(scene["centre"]), float(scene["diag"])
lo, hi = centre - 0.55 * diag, centre + 0.55 * diag          # box around the person, a little larger than the cloud
dist = [np.linalg.norm(np.asarray(cams[n]["transform_matrix"])[:3, 3] - centre) for n in names]
near, far = max(0.05, min(dist) - 0.6 * diag), max(dist) + 0.6 * diag
for split, lst in (("train", train), ("test", test)):
    json.dump({"aabb": [lo.tolist(), hi.tolist()], "near_far": [near, far], "white_bg": True,
               "frames": [cams[n] for n in lst]}, open(P / f"transforms_{split}.json", "w"), indent=1)

# ---- full scene (room kept)
F = out / "full"
common(F)
for n in names:
    resized(Image.open(src / "_und" / "images" / n).convert("RGB")).save(F / "images" / n, format="JPEG", quality=95)
cam_pos = np.asarray([np.asarray(cams[n]["transform_matrix"])[:3, 3] for n in names])
blo, bhi = np.minimum(cam_pos.min(0), lo) - 0.3 * diag, np.maximum(cam_pos.max(0), hi) + 0.3 * diag
rng = np.random.default_rng(0)
fill = rng.uniform(blo, bhi, size=(60000, 3))
xyz = [fill]
rgb = [np.full((len(fill), 3), 128)]
if pts is not None and len(pts):
    xyz.append(pts)
    rgb.append(np.asarray([p.color for p in rec.points3D.values()]))
from plyfile import PlyData
person = PlyData.read(str(src / "sparse" / "0" / "points3D.ply"))["vertex"]
xyz.append(np.stack([person["x"], person["y"], person["z"]], 1))
rgb.append(np.stack([person["red"], person["green"], person["blue"]], 1))
write_ply(F / "sparse" / "0" / "points3D.ply", np.concatenate(xyz), np.concatenate(rgb))

meta = {"images": len(names), "train": len(train), "test": len(test), "max_side": a.max_side, "centre": centre.tolist(),
        "diag": diag, "aabb": [lo.tolist(), hi.tolist()], "near_far": [near, far], "sparse_points": 0 if pts is None else len(pts),
        "room_box": [blo.tolist(), bhi.tolist()],
        "up": (lambda u: (u / np.linalg.norm(u)).tolist())(np.mean([np.asarray(cams[n]["transform_matrix"])[:3, 1] for n in names], 0))}
json.dump(meta, open(out / "scenes.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
