"""k-nearest-neighbour search: chunked brute force on the GPU (torch), scipy cKDTree on the CPU.

``knn(query, ref, k)`` returns ``(dist float64 (Q,k), idx int64 (Q,k))`` sorted by distance
(Euclidean, not squared). Both arrays are numpy on return.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["knn", "get_device", "nearest"]


def get_device(prefer: str | None = None):
    """Pick a torch device: ``prefer`` / ``$CV802_DEVICE`` ('cuda'|'cpu') or CUDA if available."""
    import torch

    want = prefer or os.environ.get("CV802_DEVICE")
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _knn_cpu(query, ref, k, workers=-1):
    from scipy.spatial import cKDTree

    tree = cKDTree(ref)
    d, i = tree.query(query, k=k, workers=workers)
    if k == 1:
        d, i = d[:, None], i[:, None]
    return d.astype(np.float64), i.astype(np.int64)


def _knn_torch(query, ref, k, device, budget_bytes=None):
    import torch

    q = torch.as_tensor(np.ascontiguousarray(query), dtype=torch.float32, device=device)
    r = torch.as_tensor(np.ascontiguousarray(ref), dtype=torch.float32, device=device)
    # Centre for numerical stability of the |q|^2 - 2 q.r + |r|^2 expansion.
    c = r.mean(0, keepdim=True)
    q = q - c
    r = r - c
    rn = (r * r).sum(1)
    if budget_bytes is None:
        budget_bytes = int(os.environ.get("CV802_KNN_BUDGET_MB", "1024")) * (1 << 20)
    # A (chunk x R) float32 distance matrix plus topk workspace.
    chunk = max(256, min(65536, budget_bytes // max(1, 4 * 3 * r.shape[0])))
    D = np.empty((q.shape[0], k), np.float64)
    I = np.empty((q.shape[0], k), np.int64)
    for s in range(0, q.shape[0], chunk):
        qs = q[s:s + chunk]
        d2 = (qs * qs).sum(1, keepdim=True) - 2.0 * qs @ r.T + rn[None]
        vals, idx = torch.topk(d2, k, dim=1, largest=False, sorted=True)
        # Recompute exact distances for the selected pairs (removes expansion round-off).
        diff = qs[:, None, :] - r[idx]
        exact = diff.norm(dim=2)
        exact, order = exact.sort(dim=1)
        idx = torch.gather(idx, 1, order)
        D[s:s + chunk] = exact.double().cpu().numpy()
        I[s:s + chunk] = idx.cpu().numpy()
        del d2, vals, diff
    return D, I


def knn(query, ref, k: int = 1, device=None, force_cpu: bool = False):
    """k nearest neighbours of each ``query`` row among ``ref`` rows (3D points).

    Uses the GPU when available (brute force in chunks, exact distances recomputed for the
    selected pairs) and falls back to scipy's cKDTree on CPU (or if CUDA errors/OOMs)."""
    query = np.asarray(query, np.float64)
    ref = np.asarray(ref, np.float64)
    k = int(min(k, len(ref)))
    if len(query) == 0:
        return np.zeros((0, k)), np.zeros((0, k), np.int64)
    if not force_cpu:
        try:
            dev = get_device(device) if not hasattr(device, "type") else device
            if dev.type == "cuda":
                return _knn_torch(query, ref, k, dev)
        except Exception as e:  # pragma: no cover - CUDA OOM / unavailable
            import sys
            print(f"[knn] GPU path failed ({type(e).__name__}: {e}); falling back to cKDTree", file=sys.stderr)
    return _knn_cpu(query, ref, k)


def nearest(query, ref, device=None):
    """Nearest neighbour: returns ``(dist (Q,), idx (Q,))``."""
    d, i = knn(query, ref, 1, device=device)
    return d[:, 0], i[:, 0]
