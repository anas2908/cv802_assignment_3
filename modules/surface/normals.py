"""Point-cloud normal estimation (our own implementation; no open3d).

1. PCA normals: the eigenvector of the smallest eigenvalue of the covariance of each point's
   k nearest neighbours (Hoppe et al. 1992).
2. Consistent orientation: a minimum spanning tree over the symmetric kNN graph with edge cost
   ``1 - |n_i . n_j|`` (Hoppe's Riemannian graph); normals are flipped while walking the tree
   breadth-first so neighbouring normals agree.
3. Global outward check per connected component: each point votes with the sign of
   ``n_i . (p_i - c_i)``, where ``c_i`` is the centroid of a *large* neighbourhood (a local proxy
   for the medial axis: inside a limb/torso it lies in the interior). The whole component is
   flipped if the majority votes inward. Unlike a single global centroid this also works for arms
   and legs of a person.

Optionally, a final relabelling pass flips isolated points whose normal disagrees with most of
their neighbours (cleans MST flip errors on thin parts).
"""
from __future__ import annotations

import numpy as np

from .knn import knn

__all__ = ["estimate_normals", "pca_normals", "orient_normals", "outward_votes"]


def pca_normals(xyz, k=24, idx=None, device=None):
    """Unoriented PCA normals. Returns ``(normals (N,3), curvature (N,), knn_idx (N,k))``."""
    xyz = np.asarray(xyz, np.float64)
    if idx is None:
        _, idx = knn(xyz, xyz, k, device=device)
    N = len(xyz)
    normals = np.empty((N, 3))
    curv = np.empty(N)
    for s in range(0, N, 200000):
        nb = xyz[idx[s:s + 200000]]                       # (n,k,3)
        nb = nb - nb.mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb) / nb.shape[1]
        w, v = np.linalg.eigh(cov)                        # ascending eigenvalues
        normals[s:s + 200000] = v[:, :, 0]
        curv[s:s + 200000] = w[:, 0] / np.maximum(w.sum(1), 1e-30)
    return normals, curv, idx


def outward_votes(xyz, normals, k_big=200, device=None):
    """Per-point outward score ``n_i . (p_i - c_i)/|p_i - c_i|`` with ``c_i`` the centroid of the
    ``k_big`` nearest neighbours (positive = pointing outwards)."""
    xyz = np.asarray(xyz, np.float64)
    k_big = int(min(k_big, len(xyz)))
    _, ib = knn(xyz, xyz, k_big, device=device)
    c = np.empty_like(xyz)
    for s in range(0, len(xyz), 50000):
        c[s:s + 50000] = xyz[ib[s:s + 50000]].mean(1)
    d = xyz - c
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return (normals * d).sum(1)


def orient_normals(xyz, normals, idx, k_big=200, device=None, smooth_iters=2):
    """Consistently orient ``normals`` (MST propagation over the kNN graph ``idx`` + per-component
    outward majority check). Returns ``(oriented normals, info dict)``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree

    xyz = np.asarray(xyz, np.float64)
    n = normals.copy()
    N, k = idx.shape
    rows = np.repeat(np.arange(N), k)
    cols = idx.reshape(-1)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    w = 1.0 - np.abs((n[rows] * n[cols]).sum(1)) + 1e-6   # strictly positive (0 means "no edge")
    G = coo_matrix((w, (rows, cols)), shape=(N, N)).tocsr()
    G = G.maximum(G.T)                                      # symmetric kNN graph
    T = minimum_spanning_tree(G)
    T = T.maximum(T.T)
    ncomp, labels = connected_components(T, directed=False)
    flips = 0
    visited = np.zeros(N, bool)
    for comp in range(ncomp):
        members = np.flatnonzero(labels == comp)
        root = members[0]
        if visited[root]:
            continue
        order, pred = breadth_first_order(T, root, directed=False, return_predecessors=True)
        visited[order] = True
        # Walk in BFS order: each node is aligned with its (already final) predecessor.
        for v in order[1:]:
            p = pred[v]
            if n[v] @ n[p] < 0:
                n[v] = -n[v]
                flips += 1
    # Optional cleanup: flip points that disagree with the (weighted) majority of their neighbours.
    for _ in range(smooth_iters):
        agree = np.einsum("nkj,nj->nk", n[idx[:, 1:]], n)
        bad = (np.sign(agree).sum(1) < -(k - 1) * 0.3)
        if not bad.any():
            break
        n[bad] = -n[bad]
    # Global outward check per component.
    votes = outward_votes(xyz, n, k_big=k_big, device=device)
    comp_flipped = 0
    for comp in range(ncomp):
        m = labels == comp
        if np.sum(np.sign(votes[m])) < 0:
            n[m] = -n[m]
            votes[m] = -votes[m]
            comp_flipped += 1
    info = {
        "components": int(ncomp),
        "components_flipped": int(comp_flipped),
        "mst_flips": int(flips),
        "outward_fraction": float(np.mean(votes > 0)),
        "neighbour_agreement": float(np.mean(np.einsum("nkj,nj->nk", n[idx[:, 1:]], n) > 0)),
    }
    return n, info


def estimate_normals(xyz, k=24, k_big=200, device=None, return_info=False):
    """Oriented normals for a point cloud (PCA + MST orientation + outward check).

    ``k`` controls the PCA/graph neighbourhood, ``k_big`` the neighbourhood whose centroid acts as
    the local inside reference for the outward vote."""
    normals, curv, idx = pca_normals(xyz, k=k, device=device)
    n, info = orient_normals(xyz, normals, idx, k_big=k_big, device=device)
    info["k"] = int(k)
    info["k_big"] = int(k_big)
    info["mean_curvature_ratio"] = float(curv.mean())
    return (n, info) if return_info else n
