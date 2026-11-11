"""Figure of what the GUI's 3D view draws, made with the same renderers in offscreen mode (no window needed):
(a) the point cloud and photo cameras, (b) a Sub-project 2 mesh rasterised by Open3D, (c) the novel view of a
trained model from the same camera, (d) the real held-out photo.

    python docs/make_gui_figure.py results docs/figs/fig_gui.jpg     (env with open3d + torch)
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.nvs.cameras import Camera  # noqa: E402
from modules.nvs.session import NVSSession  # noqa: E402

folder, out = Path(sys.argv[1]), Path(sys.argv[2])
s = NVSSession(folder, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
test = [c for c in s.cameras if c.name in s.meta["split"]["test"]]
c0 = test[0]
W, H = 450, 600
cam = Camera(W, H, c0.fy * H / c0.H, c0.fy * H / c0.H, c0.R, c0.t)
far = Camera(W, H, cam.fx * 0.55, cam.fy * 0.55, c0.R, c0.t)        # wider view to see the cameras around the person


def offscreen(geoms, camera):
    r = rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    for name, g, mat in geoms:
        if isinstance(g, rendering.TriangleMeshModel):
            r.scene.add_model(name, g)
        else:
            r.scene.add_geometry(name, g, mat)
    e = np.eye(4)
    e[:3, :3], e[:3, 3] = camera.R, camera.t
    r.setup_camera(o3d.camera.PinholeCameraIntrinsic(W, H, camera.fx, camera.fy, W / 2, H / 2), e)
    r.scene.scene.set_sun_light(camera.R[2] + 0.5 * camera.R[1], [1, 1, 1], 75000)      # light travels with the camera's view, from slightly above
    r.scene.scene.enable_sun_light(True)
    return np.asarray(r.render_to_image())


unlit = rendering.MaterialRecord()
unlit.shader = "defaultUnlit"
unlit.point_size = 2.0
line = rendering.MaterialRecord()
line.shader = "unlitLine"
line.line_width = 1.5
geoms = [("pts", o3d.io.read_point_cloud(str(folder / s.meta["points"])), unlit)]
tset = set(s.meta["split"]["test"])
for i, c in enumerate(s.cameras):
    ls = o3d.geometry.LineSet.create_camera_visualization(int(c.W), int(c.H), np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]]),
                                                          np.vstack([np.hstack([c.R, c.t[:, None]]), [0, 0, 0, 1]]), scale=0.12)
    ls.colors = o3d.utility.Vector3dVector([[0.95, 0.45, 0.1] if c.name in tset else [0.2, 0.45, 0.85]] * len(ls.lines))
    geoms.append(("cam%d" % i, ls, line))
back = Camera(W, H, far.fx, far.fy, far.R, far.t)
eye = s.meta["centre"] + 2.2 * (np.asarray(far.center) - s.meta["centre"])           # step back to show the camera ring
from modules.nvs.cameras import look_at  # noqa: E402
wide = look_at(eye, s.meta["centre"], s.meta["up"], W, H, far.fx)
panels = [("Points + photo cameras", offscreen(geoms, wide))]
# Sub-project 2 inside the app: our Screened Poisson on the open point cloud, rasterised by Open3D
from modules.surface.extract import extract_mesh  # noqa: E402
pcd = o3d.io.read_point_cloud(str(folder / s.meta["points"]))
V, F, C, info = extract_mesh(np.asarray(pcd.points), np.asarray(pcd.colors), res=256, log=lambda *a: None)
mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
mesh.compute_vertex_normals()
lit = rendering.MaterialRecord()
lit.shader = "defaultLit"
panels.append(("Mesh extracted in the app (Poisson)", offscreen([("mesh", mesh, lit)], cam)))
img, _ = s.render("3dgs", cam)
panels.append(("Novel view: 3DGS", img))
if "mip" in s.models:
    panels.append(("Novel view: Mip-Splatting", s.render("mip", cam)[0]))
photo = s.photo(c0.name)
if photo:
    ph = Image.open(photo).convert("RGB").resize((W, round(W * c0.H / c0.W)))
    panels.append(("Real photo (held out)", np.asarray(ph)[:H]))

try:
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 22)
except OSError:
    font = ImageFont.load_default()
canvas = Image.new("RGB", (len(panels) * (W + 12), H + 40), "white")
d = ImageDraw.Draw(canvas)
for k, (t, im) in enumerate(panels):
    canvas.paste(Image.fromarray(np.asarray(im)[..., :3]), (k * (W + 12), 40))
    d.text((k * (W + 12) + (W - d.textlength(t, font=font)) / 2, 8), t, fill=(40, 40, 40), font=font)
canvas.save(out, quality=90)
print("saved", out, len(panels), "panels")
