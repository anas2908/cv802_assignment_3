"""Write the report's numbers (numbers.tex) and tables (table_main.tex, table_zoom.tex) from docs/figs/numbers.json,
so every number in the report comes from the evaluation files."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.load(open(HERE.parent / "figs" / "numbers.json"))
ORDER = [("nerf", "NeRF~\\cite{mildenhall2020nerf}"), ("tensorf", "TensoRF~\\cite{chen2022tensorf}"), ("3dgs", "3DGS~\\cite{kerbl20233dgs}"), ("mip", "Mip-Splatting~\\cite{yu2024mip}")]
ORDER = [(k, n) for k, n in ORDER if k in N]


def best(key, sign):
    vals = [N[k]["mean"][key] if key in N[k]["mean"] else N[k][key] for k, _ in ORDER]
    return max(vals) if sign > 0 else min(vals)


def cell(v, b, fmt):
    s = fmt % v
    return "\\textbf{%s}" % s if abs(v - b) < 1e-12 else s


def size(k):
    d = N[k]
    return "%.0fk" % (d["n_gaussians"] / 1e3) if "n_gaussians" in d else "%.1fM" % (d["n_params"] / 1e6)


rows = []
bp, bs, bl = best("psnr", 1), best("ssim", 1), best("lpips", -1)
bt = min(N[k]["train_seconds"] for k, _ in ORDER)
bf = max(N[k]["fps"] for k, _ in ORDER)
for k, name in ORDER:
    d = N[k]
    rows.append("%s & %s & %s & %s & %s & %s & %s & %.0f \\\\" % (
        name, cell(d["mean"]["psnr"], bp, "%.2f"), cell(d["mean"]["ssim"], bs, "%.3f"), cell(d["mean"]["lpips"], bl, "%.3f"),
        cell(d["train_seconds"] / 60, bt / 60, "%.0f"), cell(d["fps"], bf, "%.1f") if d["fps"] < 10 else cell(d["fps"], bf, "%.0f"),
        size(k), d["model_mb"]))
(HERE / "table_main.tex").write_text(r"""\begin{tabular}{@{}lccccccc@{}}\toprule
Method & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & Train (min) & FPS$\uparrow$ & Primitives & MB \\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

z = [(k, n) for k, n in ORDER if "zoom" in N[k]]
if z:
    lines = []
    for lab, key in (("1/4 size (zoom out)", "x0.25"), ("1$\\times$ (training size)", None), ("1.6$\\times$ (zoom in)", "x1.6")):
        vals = [(N[k]["zoom"][key] if key else N[k]["mean"]) for k, _ in z]
        bp = max(v["psnr"] for v in vals)
        lines.append(lab + " & " + " & ".join(cell(v["psnr"], bp, "%.2f") + " / %.3f" % v["lpips"] for v in vals) + " \\\\")
    (HERE / "table_zoom.tex").write_text("\\begin{tabular}{@{}l" + "c" * len(z) + "@{}}\\toprule\nRendered at & "
                                         + " & ".join(n.split("~")[0] for _, n in z) + " \\\\\\midrule\n" + "\n".join(lines)
                                         + "\n\\bottomrule\\end{tabular}\n")

mac = []
for k, tag in (("nerf", "Nf"), ("tensorf", "Tf"), ("3dgs", "Gs"), ("mip", "Mip"), ("3dgs_full", "Full")):
    if k not in N:
        continue
    d = N[k]
    mac += ["\\newcommand{\\%sPSNR}{%.2f}" % (tag, d["mean"]["psnr"]), "\\newcommand{\\%sFPS}{%.0f}" % (tag, d["fps"]),
            "\\newcommand{\\%sMin}{%.0f}" % (tag, (d.get("train_seconds") or 0) / 60)]
    if "iterations" in d:
        mac.append("\\newcommand{\\%sIters}{%dk}" % (tag, round(d["iterations"] / 1000)))
    if "ours_vs_official_psnr" in d:
        mac.append("\\newcommand{\\%sOurs}{%.0f}" % (tag, d["ours_vs_official_psnr"]["mean"]))
    if "zoom" in d:
        mac.append("\\newcommand{\\%sZoomOut}{%.2f}" % (tag, d["zoom"]["x0.25"]["psnr"]))
(HERE / "numbers.tex").write_text("\n".join(mac) + "\n")
print("tables written:", [p.name for p in HERE.glob("table_*.tex")])
