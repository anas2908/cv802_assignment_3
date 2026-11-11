"""Screened Poisson Surface Reconstruction, implemented from scratch (no Poisson/marching-cubes API).

Kazhdan, Bolitho, Hoppe, "Poisson Surface Reconstruction", SGP 2006;
Kazhdan & Hoppe, "Screened Poisson Surface Reconstruction", ACM TOG 2013.

Pipeline (all our own code; numpy/scipy/torch arithmetic only):
 1. oriented normals: PCA + MST propagation (surface/common/normals.py, our implementation);
 2. density-adaptive sample weights and splatting of the inward normal field -n into a vector field V
    on a regular grid (trilinear splat + [1,2,1] smoothing ~ quadratic B-spline filter);
 3. divergence of V and Jacobi-preconditioned CG solve of the screened Poisson system
        (-L + beta * S^T W S) chi = -div V + beta * S^T W c,
    where L is the 7-point Laplacian (chi = 0 on the grid boundary), S trilinear interpolation at the
    samples and c the iso-value of the unscreened solution; coarse-to-fine (cascadic multigrid) warm starts;
 4. iso-surface chi = mean chi(samples), extracted with our own marching tetrahedra (Kuhn 6-tet split,
    welded edge vertices -> watertight, consistently oriented);
 5. density-based trimming (cf. PoissonRecon's SurfaceTrimmer): drop faces far from every input sample,
    then tiny components.
"""
# =====================================================================================================
# HOW TO READ THIS FILE
#
# The idea in one paragraph: we look for a 3D function chi ("indicator") that is ~1 inside the object
# and ~0 outside. Its gradient is zero everywhere except at the surface, where it points INWARD
# (= minus the surface normal). So the oriented normals tell us what grad(chi) should look like.
# Asking "which chi has a gradient closest to these arrows?" (least squares) gives the Poisson
# equation  laplacian(chi) = div(V).  We solve it on a 3D grid, then cut out the surface where chi
# crosses a threshold (the "iso-value").
#
# Reading order (search for the "STEP" labels):
#   extract.py              STEP 0  (in the GUI) get normals, call reconstruct, trim, colour the mesh
#   reconstruct()           STEP 1  fit cloud into the grid, density weights, coarse->fine solves
#   solve_level()           STEP 2  build the equation at one grid size and solve it
#     _corners/splat/interp         helpers: points <-> grid (trilinear weights)
#     smooth121                     helper: blur the arrow field
#     neg_laplacian/divergence      helpers: the two differential operators
#     pcg()                 STEP 3  the iterative solver (conjugate gradient)
#   marching_tetrahedra()   STEP 4  turn the solved 3D field chi into triangles
#   trim()                  STEP 5  remove the "balloon" surface where there were no input points
#
# Notation used in the comments:  R = grid size per axis (e.g. 256), N = number of input points,
# S = interpolation grid->points, S^T = splat points->grid, W = per-point weights.
# =====================================================================================================
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from .knn import knn

# GPU if available (CUDA, or Apple silicon through MPS), else CPU
DEV = torch.device("cuda" if torch.cuda.is_available() else
                   "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu")


def progress(run: Path | None, pct: int, stage: str, msg: str = "") -> None:
    # prints progress and writes run/progress.json (the UI reads this file to show a progress bar)
    print(f"[{pct:3d}%] {stage}: {msg}", flush=True)
    if run is not None:
        (run / "progress.json").write_text(json.dumps({"percent": pct, "stage": stage, "message": msg}))


# ---------------------------------------------------------------- trilinear splat / interpolation
# helpers that move values between the N input points and the R x R x R grid
def _corners(g: torch.Tensor, R: int):
    # g: (N,3) point positions in grid units. Each point lies inside one grid cube; return that
    # cube's 8 corners as (flat grid index, trilinear weight) pairs. Weights of the 8 corners sum to 1,
    # and a corner gets more weight the closer the point is to it.
    i0 = torch.floor(g).long().clamp(0, R - 2)   # lower corner of the cube containing each point
    f = g - i0                                   # fractional position inside the cube, in [0,1]^3
    out = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                # weight = product over x,y,z of (f if we take the upper corner else 1-f)
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                idx = ((i0[:, 0] + dx) * R + (i0[:, 1] + dy)) * R + (i0[:, 2] + dz)  # 3D index -> flat index
                out.append((idx, w))
    return out


def splat(corners, vals: torch.Tensor, R: int) -> torch.Tensor:
    """S^T: scatter per-sample values (N,) or (N,C) to the grid."""
    # each point "pours" its value into its 8 corners, split by the trilinear weights;
    # index_add_ sums contributions when several points hit the same corner
    shape = (R ** 3,) + tuple(vals.shape[1:])
    grid = torch.zeros(shape, device=vals.device, dtype=vals.dtype)
    for idx, w in corners:
        grid.index_add_(0, idx, vals * (w if vals.ndim == 1 else w[:, None]))
    return grid.reshape((R, R, R) + tuple(vals.shape[1:]))


def interp(corners, grid: torch.Tensor) -> torch.Tensor:
    """S: trilinear interpolation of a scalar grid at the samples."""
    # the exact mirror of splat: read the grid at each point as a weighted sum of its 8 corners.
    # Using the same weights both ways keeps the system symmetric, which CG (STEP 3) requires.
    flat = grid.reshape(-1)
    return sum(flat[idx] * w for idx, w in corners)


def smooth121(x: torch.Tensor, times: int = 1) -> torch.Tensor:
    # blur with the kernel [1/4, 1/2, 1/4] along x, then y, then z (separable 3D blur).
    # Applied twice it approximates the quadratic B-spline filter of the paper: it spreads each
    # point's spike over a few voxels so the field is smooth enough to differentiate.
    k = torch.tensor([0.25, 0.5, 0.25], device=x.device, dtype=x.dtype)
    for _ in range(times):
        for axis in range(3):
            xp = Fn.pad(x.movedim(axis, -1), (1, 1))  # field is zero near the grid border
            x = (k[0] * xp[..., :-2] + k[1] * xp[..., 1:-1] + k[2] * xp[..., 2:]).movedim(-1, axis)
    return x


def neg_laplacian(x: torch.Tensor) -> torch.Tensor:
    # -laplacian with the 7-point stencil: 6 * centre - (sum of the 6 face neighbours).
    # Measures how much a voxel differs from its neighbours (0 for a perfectly smooth/linear field).
    p = Fn.pad(x[None, None], (1, 1, 1, 1, 1, 1))[0, 0]  # chi = 0 outside the grid (Dirichlet)
    return 6 * x - (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
                    + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])


def divergence(V: torch.Tensor) -> torch.Tensor:
    # div V = dVx/dx + dVy/dy + dVz/dz, each by central differences (next - previous) / 2.
    # Large where the arrows converge/diverge, i.e. right at the surface.
    d = torch.zeros(V.shape[:3], device=V.device, dtype=V.dtype)
    for a in range(3):
        f = V[..., a]                                        # component a of the vector field
        fp = Fn.pad(f[None, None], (1, 1, 1, 1, 1, 1))[0, 0]
        sl_p = [slice(1, -1)] * 3
        sl_m = [slice(1, -1)] * 3
        sl_p[a] = slice(2, None)                             # "next" voxel along axis a
        sl_m[a] = slice(None, -2)                            # "previous" voxel along axis a
        d += 0.5 * (fp[tuple(sl_p)] - fp[tuple(sl_m)])
    return d


# ---------------------------------------------------------------- STEP 3: the solver
def pcg(apply_A, b, x0, diag, iters, tol=1e-5, log=print):
    # Preconditioned Conjugate Gradient: solves A x = b for a symmetric positive-definite A,
    # given only a function apply_A(x) (the matrix is never built: R^3 x R^3 would be far too big).
    # Preconditioner = Jacobi (divide by the diagonal of A), which speeds up convergence.
    # There is no training "loss" here; the quantity to watch is the relative residual
    # |b - A x| / |b| printed every 25 iterations: it should fall by orders of magnitude.
    x = x0.clone()                 # starting guess (zeros, or the upsampled coarser solution)
    r = b - apply_A(x)             # residual: how far A x is from b
    z = r / diag                   # preconditioned residual
    p = z.clone()                  # first search direction
    rz = (r * z).sum()
    bn = b.norm().clamp_min(1e-30)
    for it in range(iters):
        Ap = apply_A(p)
        alpha = rz / (p * Ap).sum()    # optimal step length along p
        x += alpha * p                 # update the solution
        r -= alpha * Ap                # update the residual without recomputing A x
        rel = (r.norm() / bn).item() if it % 25 == 0 or it == iters - 1 else None
        if rel is not None:
            log(f"    cg it {it:4d} rel.residual {rel:.2e}")
            if rel < tol:              # converged early
                break
        z = r / diag
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p      # new direction, "conjugate" to the previous ones
        rz = rz_new
    return x


# ---------------------------------------------------------------- STEP 2: Poisson solve at one resolution
def solve_level(gpts, normals, w, R, beta_scale, chi0=None, target=None, iters=300, log=print):
    # gpts: (N,3) points in grid units of THIS level; normals: (N,3) outward unit normals; w: (N,) weights.
    # beta_scale = alpha (0 -> classic 2006 Poisson); target = iso-value c the points should sit on
    # (None -> unscreened). chi0 = warm start. Returns the solved grid chi (R,R,R).
    cor = _corners(gpts, R)
    # the arrow field V: each point pushes its INWARD normal (-n), scaled by its weight, onto the grid
    V = splat(cor, -normals * w[:, None], R)          # inward field (grad of indicator: 1 inside, 0 outside)
    V = torch.stack([smooth121(V[..., a], 2) for a in range(3)], -1)  # blur each component (x,y,z)
    # right-hand side of  -laplacian(chi) = -div(V)
    rhs = -divergence(V)
    ws = splat(cor, w, R)                             # total point weight landing on each voxel
    touched = (ws > 0).sum().item()                   # how many voxels have points nearby
    # screening strength: alpha rescaled by (voxels touched / total weight) so the same --alpha
    # behaves similarly at every resolution; 0 for the unscreened solves
    beta = beta_scale * touched / w.sum().item() if target is not None else 0.0
    # diagonal of A for the Jacobi preconditioner: 6 from the Laplacian stencil + the screening part
    diag = 6.0 + beta * splat([(i, c * c) for i, c in cor], w, R)
    if beta > 0:
        rhs = rhs + beta * target * ws                # screening: pull chi(points) toward target

    def A(x):
        # left-hand side operator: (-laplacian + beta * S^T W S) x
        y = neg_laplacian(x)
        if beta > 0:
            # screening term: read chi at the points (S), weight it (W), pour it back to the grid (S^T).
            # Together with the rhs term this penalises chi(point) != target, so the surface is pinned
            # to the input points instead of smoothing over them (sharper fingers/face).
            y = y + beta * splat(cor, w * interp(cor, x), R)
        return y

    x0 = chi0 if chi0 is not None else torch.zeros_like(rhs)
    chi = pcg(A, rhs, x0, diag, iters, log=log)
    return chi, cor, beta


def upsample(chi: torch.Tensor, R: int) -> torch.Tensor:
    # resize a coarse solution to R^3 (trilinear) to use as the starting guess of the next finer level
    return Fn.interpolate(chi[None, None], size=(R, R, R), mode="trilinear", align_corners=True)[0, 0]


# ---------------------------------------------------------------- STEP 4: marching tetrahedra (own implementation)
# Each grid cube is split into 6 tetrahedra that all share the cube diagonal corner 0 -> corner 7
# (Kuhn split). A tetrahedron has only 3 cases (1, 2 or 3 corners inside), so no 256-entry lookup
# table and no ambiguous cases as in marching cubes.
_CUBE = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])  # corner id = 4x+2y+z
_TETS = []
for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):  # Kuhn split along 0->7
    # walk from corner (0,0,0) to (1,1,1) raising one axis at a time; the 4 visited corners form a tet
    c = np.zeros(3, int)
    path = [c.copy()]
    for a in perm:
        c[a] = 1
        path.append(c.copy())
    _TETS.append([4 * p[0] + 2 * p[1] + p[2] for p in path])
_TETS = np.array(_TETS)  # (6,4)


def marching_tetrahedra(field: np.ndarray, iso: float):
    """Iso-surface {field = iso} of a regular grid. Inside = field > iso. Returns (V grid coords, F)."""
    R0, R1, R2 = field.shape
    f = field - iso                     # now the surface is where f = 0
    ins = f > 0                         # True for grid vertices inside the object
    # cubes with a sign change
    c = np.zeros((R0 - 1, R1 - 1, R2 - 1), np.int8)
    for (x, y, z) in _CUBE:
        c += ins[x:R0 - 1 + x, y:R1 - 1 + y, z:R2 - 1 + z]   # count inside corners of every cube
    cubes = np.argwhere((c > 0) & (c < 8))   # only cubes with some (not all) corners inside can hold surface
    corner = cubes[:, None, :] + _CUBE[None]                               # (C,8,3)
    gid = (corner[..., 0] * R1 + corner[..., 1]) * R2 + corner[..., 2]     # global vertex ids (C,8)
    tv = gid[:, _TETS].reshape(-1, 4)                                      # (C*6,4)
    flat = f.reshape(-1)
    tf = flat[tv]                       # f at the 4 corners of every tet
    ti = tf > 0                         # which tet corners are inside
    n_in = ti.sum(1)
    keep = (n_in > 0) & (n_in < 4)      # tets crossed by the surface
    tv, tf, ti, n_in = tv[keep], tf[keep], ti[keep], n_in[keep]

    # Each output vertex lies on a tet edge joining an inside corner to an outside corner, so a
    # triangle is stored as 3 such edges and vertices are created afterwards (one per unique edge).
    edges, tris_e = [], []        # triangle = 3 edges (a,b) between an inside and an outside vertex
    order = np.argsort(~ti, axis=1, kind="stable")                         # inside vertices first
    sv = np.take_along_axis(tv, order, 1)
    for k in (1, 2, 3):
        m = n_in == k
        s = sv[m]
        if k == 1:      # one inside vertex a, outside b,c,d
            tri = [[(s[:, 0], s[:, 1]), (s[:, 0], s[:, 2]), (s[:, 0], s[:, 3])]]
        elif k == 3:    # one outside vertex d
            tri = [[(s[:, 0], s[:, 3]), (s[:, 1], s[:, 3]), (s[:, 2], s[:, 3])]]
        else:           # inside a,b ; outside c,d -> quad (a,c)(a,d)(b,d)(b,c)
            tri = [[(s[:, 0], s[:, 2]), (s[:, 0], s[:, 3]), (s[:, 1], s[:, 3])],
                   [(s[:, 0], s[:, 2]), (s[:, 1], s[:, 3]), (s[:, 1], s[:, 2])]]
        for t in tri:
            tris_e.append(np.stack([np.stack(e, 1) for e in t], 1))       # (T,3,2) [inside, outside]
    E = np.concatenate(tris_e, 0)                                          # (T,3,2)
    # one id per edge; np.unique merges the same edge seen from neighbouring tets/cubes, so adjacent
    # triangles share vertices ("welded") and the mesh is watertight
    keys = E[..., 0].astype(np.int64) * (R0 * R1 * R2) + E[..., 1]
    uniq, inv = np.unique(keys.reshape(-1), return_inverse=True)
    a = uniq // (R0 * R1 * R2)          # inside end of each unique edge
    b = uniq % (R0 * R1 * R2)           # outside end
    fa, fb = flat[a], flat[b]
    # place the vertex where f crosses 0 along the edge (linear interpolation)
    t = (fa / (fa - fb))[:, None]                                          # fa > 0 > fb
    pa = np.stack(np.unravel_index(a, f.shape), 1).astype(np.float64)
    pb = np.stack(np.unravel_index(b, f.shape), 1).astype(np.float64)
    V = pa + t * (pb - pa)
    F = inv.reshape(-1, 3)              # triangles as indices into V
    # orient: normal must point from inside to outside (outward)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    out_dir = pb[np.searchsorted(uniq, keys[:, 0])] - pa[np.searchsorted(uniq, keys[:, 0])]
    flip = (n * out_dir).sum(1) < 0
    F[flip] = F[flip][:, ::-1]          # reverse vertex order = flip the triangle
    good = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])   # drop degenerate triangles
    return V, F[good]


def keep_large_components(V, F, min_frac=0.01):
    # remove small floating pieces: keep connected components with >= min_frac of all triangles
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    if len(F) == 0:
        return V, F
    # graph whose nodes are vertices and whose edges are triangle edges
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    g = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(V), len(V)))
    _, lab = connected_components(g, directed=False)
    flab = lab[F[:, 0]]                 # component label of each triangle
    cnt = np.bincount(flab)             # triangles per component
    F = F[cnt[flab] >= min_frac * len(F)]
    # drop unused vertices and renumber the triangles
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


# ---------------------------------------------------------------- STEP 1: driver
def reconstruct(xyz, normals, depth_res=256, alpha=4.0, margin=8, log=print, run=None):
    # xyz: (N,3) points, normals: (N,3) outward normals. Returns the watertight mesh (V, F) + info.
    # fit the cloud into a depth_res^3 grid, leaving `margin` empty voxels on every side
    lo, hi = xyz.min(0), xyz.max(0)
    center = (lo + hi) / 2
    scale = (depth_res - 1 - 2 * margin) / (hi - lo).max()   # voxels per world unit
    to_grid = lambda p: (p - center) * scale + (depth_res - 1) / 2
    from_grid = lambda g: (g - (depth_res - 1) / 2) / scale + center

    # density-adaptive weights (2006 paper): w_i proportional to 1 / local sampling density
    # spacing = mean distance to the 8 nearest neighbours (column 0 is the point itself). A point in a
    # sparse region stands for a larger patch of surface (area ~ spacing^2), so it gets more weight;
    # otherwise densely scanned regions (e.g. the face) would dominate the solve.
    d, _ = knn(xyz, xyz, k=9)
    spacing = np.asarray(d)[:, 1:].mean(1)
    w = spacing ** 2
    w = w / w.mean()

    # coarse -> fine: e.g. 64^3, 128^3, 256^3. Each level starts from the upsampled previous answer,
    # so the cheap coarse grid finds the overall shape and the fine grid only adds detail.
    levels = [r for r in (depth_res // 4, depth_res // 2, depth_res) if r >= 32]
    chi, iters = None, {levels[0]: 600, levels[-1]: 250}     # CG iterations per level (300 otherwise)
    nT = torch.tensor(normals, dtype=torch.float32, device=DEV)
    wT = torch.tensor(w, dtype=torch.float32, device=DEV)
    for li, R in enumerate(levels):
        progress(run, 10 + 15 * li, "poisson solve", f"unscreened level {R}^3")
        s = (R - 1) / (depth_res - 1)                         # rescale point positions to this level
        g = torch.tensor(to_grid(xyz) * s, dtype=torch.float32, device=DEV)
        chi0 = upsample(chi, R) if chi is not None else None
        chi, cor, _ = solve_level(g, nT, wT, R, 0.0, chi0, None, iters.get(R, 300), log)
    # c0 = weighted mean of chi at the input points = "the value of chi on the surface"
    c0 = (interp(cor, chi) * wT).sum().item() / wT.sum().item()
    log(f"  unscreened iso {c0:.5f}")
    beta = 0.0
    if alpha > 0:
        # screened solve (2013 paper) at full resolution, warm-started from the unscreened chi and
        # pulling chi(points) toward c0 so the surface sticks to the points
        progress(run, 55, "poisson solve", f"screened (alpha={alpha}) at {depth_res}^3")
        chi, cor, beta = solve_level(g, nT, wT, depth_res, alpha, chi, c0, 250, log)
    # final threshold: weighted mean of chi at the input points
    sample_vals = interp(cor, chi)
    iso = (sample_vals * wT).sum().item() / wT.sum().item()
    log(f"  iso {iso:.5f}; chi(samples) std {sample_vals.std().item():.5f}; beta {beta:.3f}")

    progress(run, 75, "marching tetrahedra", f"iso={iso:.4f}")
    Vg, F = marching_tetrahedra(chi.float().cpu().numpy(), iso)   # surface where chi = iso (grid units)
    V = from_grid(Vg)                                             # back to world coordinates
    return V, F, {"iso": iso, "unscreened_iso": c0, "beta": beta, "levels": levels,
                  "voxel_size": float(1 / scale), "median_spacing": float(np.median(spacing))}


# ---------------------------------------------------------------- STEP 5: trimming
def trim(V, F, xyz, voxel, factor=2.0, min_frac=0.01):
    """Density trimming: remove faces whose centroid is farther than factor*voxel from every sample."""
    # Poisson always returns a CLOSED surface, so where there were no points (under the feet, gaps)
    # it inflates a smooth "balloon". Faces far from every input point are that balloon: cut them,
    # then drop the small pieces left over.
    cen = V[F].mean(1)                  # centre of each triangle
    d, _ = knn(cen, xyz, k=1)           # distance to the nearest input point
    keep = np.asarray(d).ravel() <= factor * voxel
    return keep_large_components(V, F[keep], min_frac)
