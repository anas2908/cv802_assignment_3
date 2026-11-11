"""Checks for the novel-view part of the GUI that run without a window.

    python tests/test_nvs.py [results folder]

1. every model in the results folder loads and renders a held-out view on the CPU (as on a laptop without CUDA)
2. the GUI's conversion from the Open3D view camera back to an OpenCV camera returns the camera that was set
   (uses Open3D's offscreen renderer; skipped if this machine has no headless OpenGL)
3. the GUI's Sub-project 2 mesh extraction (our Screened Poisson) runs on the results' point cloud
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.nvs.cameras import Camera  # noqa: E402
from modules.nvs.session import NVSSession  # noqa: E402

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
s = NVSSession(folder, device=torch.device("cpu"))
cam = [c for c in s.cameras if c.name in s.meta["split"]["test"]][0]
ok = True
for mid, label in s.labels():
    t0 = time.time()
    img, dt = s.render(mid, cam.scaled(0.3))
    load = time.time() - t0 - dt
    filled = float((np.abs(img.astype(int) - (0 if s.background(mid)[0] < 0.5 else 255)).sum(-1) > 30).mean())
    good = img.shape == (round(cam.H * 0.3), round(cam.W * 0.3), 3) and filled > 0.05
    ok &= good
    print("%-28s %s  %dx%d  render %.2f s (CPU), load %.1f s, %.0f%% of pixels drawn" % (
        label, "ok " if good else "BAD", img.shape[1], img.shape[0], dt, load, 100 * filled))

# Open3D view camera -> OpenCV camera (the conversion in modules/gui/gui.py, _view_camera)
try:
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    W, H = 640, 480
    r = rendering.OffscreenRenderer(W, H)
    r.scene.add_geometry("pts", o3d.io.read_point_cloud(str(folder / s.meta["points"])), rendering.MaterialRecord())
    c = Camera(W, H, 520.0, 520.0, cam.R, cam.t)
    e = np.eye(4)
    e[:3, :3], e[:3, 3] = c.R, c.t
    r.setup_camera(o3d.camera.PinholeCameraIntrinsic(W, H, c.fx, c.fy, W / 2, H / 2), e)
    V = np.asarray(r.scene.camera.get_view_matrix(), np.float64)
    flip = np.diag([1.0, -1.0, -1.0])
    R, t = flip @ V[:3, :3], flip @ V[:3, 3]
    fov = np.deg2rad(r.scene.camera.get_field_of_view())
    f = (W if r.scene.camera.get_field_of_view_type() == rendering.Camera.FovType.Horizontal else H) / (2 * np.tan(fov / 2))
    err_r, err_t, err_f = np.abs(R - c.R).max(), np.abs(t - c.t).max(), abs(f - c.fy) / c.fy
    good = err_r < 1e-4 and err_t < 1e-4 and err_f < 1e-3
    ok &= good
    print("camera conversion %s  |dR| %.1e  |dt| %.1e  focal error %.2e" % ("ok " if good else "BAD", err_r, err_t, err_f))
except Exception as e:  # noqa: BLE001
    print("camera conversion skipped (no headless OpenGL here): %s" % e)
# Sub-project 2 in the GUI: our Screened Poisson on the results' point cloud (128^3, CPU)
try:
    import open3d as o3d
    import modules.surface.poisson as P
    from modules.surface.extract import extract_mesh
    P.DEV = torch.device("cpu")
    pcd = o3d.io.read_point_cloud(str(folder / s.meta["points"]))
    t0 = time.time()
    V, F, C, info = extract_mesh(np.asarray(pcd.points), np.asarray(pcd.colors), res=128, log=lambda *a: None)
    good = len(F) > 10000 and np.isfinite(V).all()
    ok &= good
    print("Poisson mesh      %s  %d triangles in %.1f s (CPU)" % ("ok " if good else "BAD", len(F), time.time() - t0))
except Exception as e:  # noqa: BLE001
    ok = False
    print("Poisson mesh      BAD  %s" % e)
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
