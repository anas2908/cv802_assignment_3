"""Preset camera path for the result videos: one smooth circle around the person at the photos' average height
and distance, with a gentle up-and-down wave, starting in front of the person (the first photo's side)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.nvs.cameras import orbit_path, save_path  # noqa: E402

data = Path(sys.argv[1])            # DATA folder (scenes.json, person/transforms_train.json)
out = Path(sys.argv[2])
W, H, N = 720, 960, 240             # portrait frames, 8 s at 30 fps
meta = json.load(open(data / "scenes.json"))
frames = json.load(open(data / "person" / "transforms_train.json"))["frames"]
centre, up = np.asarray(meta["centre"]), np.asarray(meta["up"])
pos = np.asarray([np.asarray(f["transform_matrix"])[:3, 3] for f in frames])
rel = pos - centre
height = float(np.median(rel @ up))
radius = float(np.median(np.linalg.norm(rel - np.outer(rel @ up, up), axis=1)))
f = float(np.median([fr["fl_y"] / fr["h"] for fr in frames])) * H * 1.1     # photos' field of view, slightly zoomed in
start = rel[0] - up * (rel[0] @ up)
cams = orbit_path(centre, up, radius, height, N, W, H, f, start_dir=start, bob=0.12)
save_path(cams, out)
print(json.dumps({"frames": N, "size": [W, H], "radius": radius, "height": height, "focal": f}))
