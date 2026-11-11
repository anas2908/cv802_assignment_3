"""Progress dashboard for the tmux session: one bar per step, live % and ETA from the training logs."""
import json
import os
import re
import time
from pathlib import Path

SP3 = Path(os.environ.get("CV802_SP3_ROOT", "/l/users/%s/cv802_sp3" % os.environ.get("USER", "")))
STATUS, LOGS = SP3 / "status", SP3 / "logs"
# step id, label, estimated minutes (used for the overall ETA until the step reports its own)
STEPS = [
    ("setup", "Environments + official code", 20),
    ("data", "Data: person on white + full room", 5),
    ("train_3dgs", "Train 3DGS (random start)", 12),
    ("train_mip", "Train Mip-Splatting (random start)", 15),
    ("train_tensorf", "Train TensoRF", 40),
    ("train_3dgs_full", "Train 3DGS room (random start)", 35),
    ("train_nerf", "Train NeRF (nerf-pytorch)", 75),
    ("eval", "Scores on 16 held-out photos", 15),
    ("videos", "Result MP4s along the camera path", 20),
    ("gui", "GUI in the starter code + Mac renderer", 120),
    ("report", "One-page report + references", 60),
    ("slides", "Slides (SP1/SP2 theme)", 60),
    ("push", "New GitHub repo + fresh-clone test", 30),
]
TQDM = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:?]+)")
C = {"done": "\033[32m", "running": "\033[33m", "failed": "\033[31m", "pending": "\033[2m", "end": "\033[0m", "b": "\033[1m"}


def secs(s):
    if "?" in s:
        return None
    t = 0
    for p in s.split(":"):
        t = t * 60 + int(p)
    return t


def fmt(s):
    if s is None:
        return "  —  "
    s = int(s)
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60) if s >= 3600 else "%2d:%02d" % (s // 60, s % 60)


def log_progress(step):
    f = LOGS / (step + ".log")
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        fh.seek(max(0, f.stat().st_size - 20000))
        tail = fh.read().decode("utf-8", "ignore")
    m = None
    for m in TQDM.finditer(tail.replace("\r", "\n")):
        pass
    if not m:
        return None
    done, total = int(m.group(2)), int(m.group(3))
    return 100.0 * done / max(total, 1), secs(m.group(5)), "%d / %d iterations" % (done, total)


def bar(pct, w=16):
    n = int(round(w * pct / 100))
    return "█" * n + "░" * (w - n)


now = time.time()
rows, remaining = [], 0.0
for sid, label, est in STEPS:
    st = {"percent": 0, "state": "pending", "message": ""}
    try:
        st.update(json.load(open(STATUS / (sid + ".json"))))
    except (OSError, ValueError):
        pass
    pct, eta, msg = float(st["percent"]), None, st.get("message", "")
    if st["state"] == "running":
        lp = log_progress(sid)
        if lp:
            pct, eta, msg = lp[0], lp[1], lp[2]
        if eta is None:
            eta = est * 60 * (1 - pct / 100)
    if st["state"] == "done":
        pct = 100.0
    if st["state"] != "done":
        remaining += eta if eta is not None else est * 60
    rows.append((st["state"], label, pct, eta, msg))

ndone = sum(r[0] == "done" for r in rows)
overall = 100.0 * sum(r[2] for r in rows) / (100.0 * len(rows))
print(C["b"] + "CV802 Sub-project 3 · Novel-view synthesis" + C["end"] + "   " + time.strftime("%H:%M:%S"))
print("Overall  %s %5.1f%%   %d/%d steps done   ETA %s (finish ≈ %s)" % (
    bar(overall, 24), overall, ndone, len(rows), fmt(remaining), time.strftime("%H:%M", time.localtime(now + remaining))))
print()
# live GPU and CPU use of the allocation's node
try:
    import subprocess
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
    name, util, used, total, temp, power = q[0], float(q[1]), float(q[2]), float(q[3]), q[4], q[5]
    procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    print("GPU      %s  compute %s %5.1f%%   %s" % (C["b"] + name + C["end"], bar(util, 16), util, "%s °C · %s W" % (temp, power)))
    print("VRAM     %s %5.1f%%   %.1f / %.1f GB · %d process%s" % (bar(100 * used / total, 24), 100 * used / total, used / 1024,
                                                                  total / 1024, len(procs), "" if len(procs) == 1 else "es"))
except Exception as e:  # noqa: BLE001
    print("GPU      (nvidia-smi unavailable: %s)" % e)
ncpu = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", "12").split("(")[0])
load = os.getloadavg()[0]
print("CPU      %s %5.1f%%   load %.1f on %d allocated cores" % (bar(min(100, 100 * load / ncpu), 24), min(100, 100 * load / ncpu), load, ncpu))
print()
icon = {"done": "✔", "running": "▶", "failed": "✘", "pending": "·"}
for state, label, pct, eta, msg in rows:
    print("%s%s %-34s %s %5.1f%% %s %s%s" % (C[state], icon[state], label, bar(pct), pct,
                                              "ETA " + fmt(eta) if state == "running" else "         ", msg[:30], C["end"]))
