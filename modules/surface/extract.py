"""Sub-project 2 inside the GUI: a textured-colour mesh from the loaded point cloud with our Screened Poisson.

normals (PCA + MST orientation, normals.py) -> screened Poisson solve + marching tetrahedra (poisson.py) ->
trim faces far from the points -> colour each vertex from its nearest input point.
"""
from __future__ import annotations

import time

import numpy as np

from .knn import knn
from .normals import estimate_normals
from .poisson import reconstruct, trim


def extract_mesh(xyz, rgb, res=128, alpha=4.0, trim_voxels=2.0, normals=None, log=print):
    """xyz (N,3), rgb (N,3) in [0,1] -> (V (M,3), F (K,3), vertex colours (M,3), info)."""
    t0 = time.time()
    xyz = np.asarray(xyz, np.float64)
    if normals is None:
        log("normals: PCA on 24 neighbours, oriented along a minimum spanning tree")
        normals = estimate_normals(xyz)
    log("Poisson solve on a %d^3 grid (screening %.0f)" % (res, alpha))
    V, F, info = reconstruct(xyz, normals, res, alpha, log=log)
    if trim_voxels > 0:
        V, F = trim(V, F, xyz, info["voxel_size"], trim_voxels)
    _, idx = knn(V, xyz, k=1)
    colours = np.asarray(rgb, np.float64)[np.asarray(idx).reshape(-1)]
    info.update(seconds=round(time.time() - t0, 1), vertices=int(len(V)), triangles=int(len(F)))
    return V, F, colours, info
