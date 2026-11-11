"""Scenes whose Gaussians start from random points, so the input is the photos and their cameras only.

    python training/make_random_init.py <DATA>            -> DATA/person_rand, DATA/full_rand

Same photos, cameras and train/test split as DATA/person and DATA/full (linked, not copied); only the starting point
cloud differs: uniform random positions in the scene box with grey colour, as the 3DGS code does for its synthetic
scenes (100k points in a cube, colour SH2RGB(random/255) = grey). No point of the Sub-project 1 cloud is used.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

DATA = Path(sys.argv[1])
meta = json.load(open(DATA / "scenes.json"))
rng = np.random.default_rng(0)
for src, dst, box, n in (("person", "person_rand", meta["aabb"], 100_000), ("full", "full_rand", meta["room_box"], 300_000)):
    s, d = DATA / src, DATA / dst
    if d.exists():
        shutil.rmtree(d)
    (d / "sparse" / "0").mkdir(parents=True)
    os.symlink(s / "images", d / "images")
    for f in ("cameras.bin", "images.bin"):
        shutil.copy2(s / "sparse" / "0" / f, d / "sparse" / "0" / f)
    for f in ("split.json", "transforms_train.json", "transforms_test.json"):
        if (s / f).exists():
            shutil.copy2(s / f, d / f)
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    xyz = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
    v = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["red"] = v["green"] = v["blue"] = 128
    PlyData([PlyElement.describe(v, "vertex")]).write(str(d / "sparse" / "0" / "points3D.ply"))
    print(dst, n, "random points in", np.round(lo, 2).tolist(), np.round(hi, 2).tolist())
