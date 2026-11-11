"""CV802 Assignment 3 deck in the plain style of the Assignment 1 and 2 slides (white, Arial, black titles, grey text).

python docs/slides/build_slides.py [out.pptx]      (python-pptx, Pillow)
Inputs: docs/figs/ (figures and numbers.json from docs/make_figures.py)."""
import json
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "CV802_SP3_novel_view_synthesis.pptx"
N = json.load(open(FIG / "numbers.json"))
BLACK, GREY, LINE = RGBColor(0, 0, 0), RGBColor(0x59, 0x59, 0x59), RGBColor(0xBF, 0xC5, 0xCC)
PIE = ["4285F4", "DB4437", "F4B400", "0F9D58", "AB47BC", "00ACC1", "FF7043", "9E9D24", "5C6BC0"]
GREEN, BLUE = "D9EAD3", "CFE2F3"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(10), Inches(5.625)
BLANK = prs.slide_layouts[6]
count = [0]


def text(slide, s, x, y, w, h, size=20, color=GREY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.name = "Arial"; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
    return tb


def bullets(slide, items, x, y, w, h, size=18, gap=6):
    """Plain bullet list; an item starting with '  ' is a sub-point (smaller, indented)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        sub = it.startswith("  ")
        p.space_after = Pt(gap)
        r = p.add_run(); r.text = ("–  " if sub else "•  ") + it.strip()
        r.font.name = "Arial"; r.font.size = Pt(size - 3 if sub else size); r.font.color.rgb = GREY
        if sub:
            p.level = 1
    return tb


def slide(title=None, number=True):
    s = prs.slides.add_slide(BLANK)
    count[0] += 1
    if title:
        text(s, title, 0.42, 0.3, 9.2, 0.75, size=32, color=BLACK)
    if number:
        text(s, str(count[0]), 9.25, 5.22, 0.5, 0.3, size=11, color=GREY, align=PP_ALIGN.RIGHT)
    return s


def pic(s, path, x, y, w=None, h=None):
    """Place an image inside the box (x, y, w, h) keeping its aspect ratio; returns its (x, y, w, h)."""
    iw, ih = Image.open(path).size
    if w and h:
        sc = min(w / iw, h / ih); pw, ph = iw * sc, ih * sc; x, y = x + (w - pw) / 2, y + (h - ph) / 2
    elif w:
        pw, ph = w, w * ih / iw
    else:
        pw, ph = h * iw / ih, h
    s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(pw), Inches(ph))
    return x, y, pw, ph


def border(cell, color="BFC5CC", width=9525):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tag), w=str(width), cap="flat", cmpd="sng", algn="ctr")
        sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr"), val=color)
        etree.SubElement(ln, qn("a:prstDash"), val="solid")


def table(s, rows, x, y, widths, size=12, row_h=0.3, align=None, fills=None, merge=()):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * len(rows))).table
    tblPr = t._tbl.tblPr
    for k in ("firstRow", "bandRow"):
        tblPr.set(k, "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"   # "No Style, Table Grid"
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j); border(c)
            if fills and (i, j) in fills:
                c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string(fills[(i, j)])
            else:
                c.fill.background()
            c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.text = ""
            p.alignment = (align[j] if align else (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT))
            r = p.add_run(); r.text = str(val)
            r.font.name = "Arial"; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = BLACK
    for r0, c0, r1, c1 in merge:              # group headers, centred
        t.cell(r0, c0).merge(t.cell(r1, c1))
        t.cell(r0, c0).text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    return t


def arrow(s, x1, y1, x2, y2):
    ln = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = GREY; ln.line.width = Pt(1.25)
    etree.SubElement(ln.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")


def pie(s, title, cats, vals, x, y, w, h, fmt='0'):
    cd = CategoryChartData(); cd.categories = cats; cd.add_series(title, vals)
    ch = s.shapes.add_chart(XL_CHART_TYPE.PIE, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.has_title = True; ch.chart_title.text_frame.text = title
    tp = ch.chart_title.text_frame.paragraphs[0]; tp.runs[0].font.size = Pt(14); tp.runs[0].font.name = "Arial"; tp.runs[0].font.bold = False; tp.runs[0].font.color.rgb = GREY
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.RIGHT; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11); ch.legend.font.name = "Arial"; ch.legend.font.color.rgb = GREY
    plot = ch.plots[0]
    for i, pt in enumerate(plot.series[0].points):
        pt.format.fill.solid(); pt.format.fill.fore_color.rgb = RGBColor.from_string(PIE[i % len(PIE)])
        pt.format.line.color.rgb = RGBColor(255, 255, 255)
    plot.has_data_labels = True
    dl = plot.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False; dl.show_value = True
    dl.font.size = Pt(11); dl.font.name = "Arial"; dl.font.color.rgb = RGBColor(255, 255, 255); dl.font.bold = True
    dl.position = XL_LABEL_POSITION.CENTER
    return ch


def scatter(s, title, pts, x, y, w, h, xtitle, ytitle, xfmt, yfmt, xr, yr, ystep, log=True):
    """pts: [(label, x, y, label position)], one labelled point per method."""
    cd = XyChartData()
    for lab, xv, yv, _ in pts:
        cd.add_series(lab).add_data_point(xv, yv)
    ch = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.has_title = True; ch.chart_title.text_frame.text = title
    tp = ch.chart_title.text_frame.paragraphs[0]; tp.runs[0].font.size = Pt(14); tp.runs[0].font.name = "Arial"; tp.runs[0].font.bold = False; tp.runs[0].font.color.rgb = GREY
    ch.has_legend = False
    for i, (ser, (lab, xv, yv, pos)) in enumerate(zip(ch.plots[0].series, pts)):
        ser.marker.style = XL_MARKER_STYLE.CIRCLE; ser.marker.size = 10
        ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = RGBColor.from_string(PIE[i % len(PIE)])
        ser.marker.format.line.color.rgb = RGBColor(255, 255, 255)
        ser.format.line.fill.background()
        dl = ser.points[0].data_label; dl.has_text_frame = True; dl.text_frame.text = lab; dl.position = pos
        r = dl.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(10); r.font.name = "Arial"; r.font.color.rgb = GREY
    for ax, t, fmt, (lo, hi) in ((ch.category_axis, xtitle, xfmt, xr), (ch.value_axis, ytitle, yfmt, yr)):
        ax.minimum_scale, ax.maximum_scale = lo, hi
        ax.has_major_gridlines = True; ax.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE6, 0xEA)
        ax.format.line.color.rgb = LINE
        ax.tick_labels.number_format = fmt; ax.tick_labels.number_format_is_linked = False
        ax.tick_labels.font.size = Pt(10); ax.tick_labels.font.name = "Arial"; ax.tick_labels.font.color.rgb = GREY
        ax.has_title = True; ax.axis_title.text_frame.text = t
        r = ax.axis_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(11); r.font.name = "Arial"; r.font.bold = False; r.font.color.rgb = GREY
    ch.value_axis.major_unit = ystep
    if log:
        lb = etree.Element(qn("c:logBase")); lb.set("val", "10")
        ch.category_axis._element.find(qn("c:scaling")).insert(0, lb)
    return ch


def num(v):
    return float(str(v).replace(",", "").replace(" min", "").replace(" s", "").replace("k", "e3").replace("M", "e6"))


def rank_fills(rows, cols, start=1):
    """cols: {column index: +1 higher is better / -1 lower is better}; rows before `start` are headers.
    Best value(s) green, second best blue."""
    out = {}
    for j, sign in cols.items():
        vals = sorted({num(r[j]) for r in rows[start:]}, reverse=(sign > 0))
        for i, r in enumerate(rows[start:], start):
            v = num(r[j])
            if v == vals[0]: out[(i, j)] = GREEN
            elif len(vals) > 1 and v == vals[1]: out[(i, j)] = BLUE
    return out


def legend(s):
    for k, (col, lab) in enumerate([(GREEN, "best"), (BLUE, "second best")]):
        x = 6.9 + k * 1.15
        b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(0.62), Inches(0.2), Inches(0.2))
        b.fill.solid(); b.fill.fore_color.rgb = RGBColor.from_string(col); b.line.color.rgb = LINE; b.shadow.inherit = False
        text(s, lab, x + 0.25, 0.55, 1.0, 0.35, size=12)


NAMES = [("nerf", "NeRF"), ("tensorf", "TensoRF"), ("3dgs", "3DGS"), ("mip", "Mip-Splatting")]
NAMES = [(k, n) for k, n in NAMES if k in N]
dur = lambda t: f"{t / 60:.0f} min" if t >= 90 else f"{t:.0f} s"
prims = lambda d: f"{d['n_gaussians'] / 1e3:.0f}k" if "n_gaussians" in d else f"{d['n_params'] / 1e6:.1f}M"

# 1 title
s = slide(number=False)
text(s, "CV802", 0.3, 1.3, 9.4, 1.0, size=54, color=BLACK, align=PP_ALIGN.CENTER)
text(s, "Assignment 3: Novel-view Synthesis", 0.3, 2.3, 9.4, 0.9, size=40, color=BLACK, align=PP_ALIGN.CENTER)
text(s, "Anas Khan", 0.3, 3.75, 9.4, 0.7, size=28, color=GREY, align=PP_ALIGN.CENTER)

# 2 problem statement
s = slide("Problem Statement:")
pic(s, FIG / "fig_inputs.jpg", 0.42, 1.15, w=4.0, h=3.0)
arrow(s, 4.55, 2.65, 5.1, 2.65)
if (FIG / "fig_orbit_3dgs.jpg").exists():
    pic(s, FIG / "fig_orbit_3dgs.jpg", 5.2, 1.15, w=4.4, h=3.0)
text(s, "125 photos + their cameras (Assignment 1)", 0.42, 4.25, 4.0, 0.4, size=13, align=PP_ALIGN.CENTER)
text(s, "pictures from new viewpoints", 5.2, 4.25, 4.4, 0.4, size=13, align=PP_ALIGN.CENTER)
text(s, "Output: a video along a preset camera path (.mp4) and drag-to-view inside the starter GUI", 0.42, 4.75, 9.2, 0.4,
     size=15, align=PP_ALIGN.CENTER)

# 3 inputs
s = slide("Inputs")
pic(s, FIG / "fig_inputs.jpg", 0.42, 1.05, w=9.16, h=2.4)
bullets(s, ["The same 125 photos as Assignments 1 and 2 (4 iPhone lenses + video frames)",
            "Cameras from Assignment 1 (VGGSfM), undistorted to pinhole",
            "Person masks → person on a white background; resized to 1000 px",
            "109 photos for training, 16 held out for scoring (every 8th, the 3DGS rule)"], 0.42, 3.55, 9.2, 1.7, size=15, gap=3)

# 4 methods
s = slide("Methods")
rows = [["Method", "Venue", "Year", "Scene stored as"],
        ["NeRF", "ECCV", "2020", "one neural network (8 layers × 256)"],
        ["TensoRF", "ECCV", "2022", "a grid split into planes and lines"],
        ["3D Gaussian Splatting (3DGS)", "ACM TOG / SIGGRAPH", "2023", "~200k coloured 3D blobs"],
        ["Mip-Splatting", "CVPR", "2024", "3DGS + two anti-aliasing filters"]]
table(s, rows, 0.42, 1.1, [3.0, 2.0, 0.8, 3.36], size=13, row_h=0.42, align=[PP_ALIGN.LEFT] * 4)
text(s, "Official code (NeRF: nerf-pytorch, the standard PyTorch version of the authors' TensorFlow code), same photos and cameras.\nTensoRF, 3DGS, Mip-Splatting: 30,000 steps. NeRF: 60 minutes of training (the paper trains for 200,000+ steps).", 0.42, 3.4, 9.2, 0.8, size=13)

# 5-7 how each method works (one idea per slide)
s = slide("TensoRF: how it works")
bullets(s, ["Space is a 3D grid of features: density (how solid) and colour",
            "A full 300×300×300 grid is huge, so it is stored as planes × lines:",
            "  feature(x, y, z) = Σ  plane_XY(x, y) · line_Z(z) + plane_XZ · line_Y + plane_YZ · line_X",
            "A pixel = shoot a ray, sample points along it, add up their colours:",
            "  C = Σ  Tᵢ · (1 − e^(−σᵢ δᵢ)) · cᵢ ,   Tᵢ = how much light still gets through",
            "Small network turns the feature + viewing direction into a colour"], 0.42, 1.05, 9.2, 4.2, size=17)
s = slide("3D Gaussian Splatting: how it works")
bullets(s, ["The scene is ~200k 3D Gaussians (soft ellipsoids): position, size, rotation, opacity, colour",
            "Start from our point cloud (Assignment 1); add, split and delete blobs while training",
            "To draw: project each blob to the screen (a 2D ellipse), sort front to back, blend:",
            "  Σ′ = J W Σ Wᵀ Jᵀ        (3D shape → 2D shape on screen)",
            "  C = Σ  cᵢ αᵢ Πⱼ₍ⱼ<ᵢ₎ (1 − αⱼ)",
            "No ray marching → real time (hundreds of frames per second)"], 0.42, 1.05, 9.2, 4.2, size=17)
s = slide("Mip-Splatting: how it works")
bullets(s, ["3DGS, plus two filters so a blob is never sharper than the camera can see",
            "3D filter: each blob gets a minimum size, set by the closest training photo",
            "  Σ ← Σ + s²·I        (no blobs thinner than one pixel of any photo)",
            "2D filter: on screen, each blob is blurred to at least one pixel",
            "Why it matters: zoom out and 3DGS blobs become too thin → flicker and jagged edges",
            "Same quality at the training size; better when the camera moves closer or further"], 0.42, 1.05, 9.2, 4.2, size=17)

# 8 experiments
s = slide("Experiments")
exp = ["5 trainings: NeRF, TensoRF, 3DGS, Mip-Splatting (person), 3DGS full room (extra)",
       "Scores on the 16 held-out photos: PSNR, SSIM, LPIPS",
       "Zoom test: render 3DGS and Mip-Splatting at ¼ size and 1.6× size",
       "Result video: same 8 s orbit for every method",
       "Check: our GUI renderer vs the official renderer (same pixels?)"]
bullets(s, exp, 0.42, 1.0, 5.0, 3.9, size=15, gap=8)
tm = [(n, N[k]["train_seconds"] / 60) for k, n in NAMES + [("3dgs_full", "3DGS full room")] if k in N and N[k].get("train_seconds")]
if tm:
    pie(s, "Training minutes (%.0f in total)" % sum(t for _, t in tm), [t[0] for t in tm], [round(t[1]) for t in tm], 5.4, 1.0, 4.3, 3.9)
text(s, "One RTX 5000 Ada (32 GB), trainings run side by side.", 0.42, 5.08, 7, 0.3, size=11)

# 9 results figure
s = slide("Results on held-out photos")
pic(s, FIG / "fig_compare.jpg", 0.42, 1.0, w=9.16, h=4.15)

# 10 comparison table
s = slide("Comparison")
legend(s)
rows = [["Method", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "Training ↓", "FPS ↑", "Primitives", "Size (MB)"]]
for k, n in NAMES:
    d = N[k]
    rows.append([n, f"{d['mean']['psnr']:.2f}", f"{d['mean']['ssim']:.3f}", f"{d['mean']['lpips']:.3f}", dur(d["train_seconds"]),
                 f"{d['fps']:.1f}" if d["fps"] < 10 else f"{d['fps']:.0f}", prims(d), f"{d['model_mb']:.0f}"])
table(s, rows, 0.42, 1.2, [1.75, 0.95, 0.95, 1.0, 1.15, 0.85, 1.25, 1.26], size=13, row_h=0.42,
      fills=rank_fills(rows, {1: 1, 2: 1, 3: -1, 4: -1, 5: 1}))
text(s, "16 held-out photos, 750×1000. PSNR in dB. FPS = images per second on the GPU. Primitives: Gaussians (3DGS, Mip) or numbers in the grid (TensoRF).",
     0.42, 3.15, 9.2, 0.6, size=11)

# 11 quality vs time and speed
s = slide("Quality vs time and speed")
L = XL_LABEL_POSITION
pts_t = [(n, N[k]["train_seconds"] / 60, N[k]["mean"]["psnr"], L.ABOVE if i % 2 == 0 else L.BELOW) for i, (k, n) in enumerate(NAMES)]
pts_f = [(n, N[k]["fps"], N[k]["mean"]["psnr"], L.ABOVE if i % 2 == 0 else L.BELOW) for i, (k, n) in enumerate(NAMES)]
lo = min(p[2] for p in pts_t) - 1.0
hi = max(p[2] for p in pts_t) + 1.0
scatter(s, "PSNR vs training time", pts_t, 0.3, 0.95, 4.65, 4.2, "Training (min, log scale)", "PSNR (dB) ↑", "0", "0.0", (1, 100), (lo, hi), 0.5)
scatter(s, "PSNR vs rendering speed", pts_f, 5.05, 0.95, 4.65, 4.2, "Images per second (log scale)", "PSNR (dB) ↑", "0", "0.0", (0.1, 1000), (lo, hi), 0.5)
text(s, "Higher is better; left chart: further left is faster to train, right chart: further right draws faster.", 0.42, 5.15, 7, 0.3, size=11)

# 12 zoom test
if all("zoom" in N[k] for k in ("3dgs", "mip") if k in N):
    s = slide("Zoom test: 3DGS vs Mip-Splatting")
    pic(s, FIG / "fig_zoom.jpg", 0.42, 1.0, w=5.4, h=4.1)
    rows = [["Rendered at", "3DGS", "Mip"]]
    for lab, key in (("¼ size", "x0.25"), ("1× (training)", None), ("1.6×", "x1.6")):
        g = N["3dgs"]["zoom"][key] if key else N["3dgs"]["mean"]
        m = N["mip"]["zoom"][key] if key else N["mip"]["mean"]
        rows.append([lab, f"{g['psnr']:.2f}", f"{m['psnr']:.2f}"])
    fills = {}
    for i in range(1, len(rows)):
        j = 1 if num(rows[i][1]) >= num(rows[i][2]) else 2
        fills[(i, j)] = GREEN
    table(s, rows, 6.05, 1.3, [1.45, 0.95, 0.95], size=13, row_h=0.42, fills=fills)
    text(s, "Left: rendered at ¼ size, enlarged ×6 (pixels kept square). Table: PSNR (dB), 16 held-out photos; ¼ = photo averaged down, 1.6× = original 1600 px photos.", 6.05, 3.15, 3.5, 1.2, size=11)
    text(s, "Zoomed out, 3DGS blobs get thinner than a pixel and the image breaks up; Mip-Splatting stays clean.", 6.05, 4.05, 3.5, 1.0, size=13)

# 13 result videos
s = slide("Result videos (preset 8 s orbit)")
y = 1.0
for k, n in NAMES:
    f = FIG / f"fig_orbit_{k}.jpg"
    if f.exists():
        text(s, n, 0.42, y + 0.45, 1.5, 0.4, size=13)
        pic(s, f, 1.9, y, w=7.7, h=1.3)
        y += 1.38
text(s, "Frames 0, 40, 80, 120, 160, 200 of 240. The full videos are in results/videos/.", 0.42, 5.15, 7, 0.3, size=11)

# 14 extra: full room
if (FIG / "fig_orbit_3dgs_full.jpg").exists():
    s = slide("Extra: 3DGS with the room")
    pic(s, FIG / "fig_orbit_3dgs_full.jpg", 0.42, 1.1, w=9.16, h=2.6)
    d = N.get("3dgs_full")
    if d:
        bullets(s, [f"Same photos without the person masks: {d['mean']['psnr']:.2f} dB PSNR on the held-out photos",
                    f"{d['n_gaussians'] / 1e6:.2f}M Gaussians (person alone: ~200k), trained in {d['train_seconds'] / 60:.0f} min",
                    "The room is only seen from where the photos were taken, so views far from them break up"],
                0.42, 3.85, 9.2, 1.4, size=15, gap=4)

# 15 GUI
s = slide("GUI: inside the course starter code")
gui_fig = FIG / "fig_gui.jpg"
if gui_fig.exists():
    pic(s, gui_fig, 0.42, 0.95, w=9.16, h=2.25)
    text(s, "What the GUI's 3D view draws, from one held-out photo's camera (offscreen render of the same code)", 0.42, 3.2, 9.2, 0.3,
         size=10, align=PP_ALIGN.CENTER)
bullets(s, ["Same Open3D app as Assignment 1, new panels",
            "File > Open novel-view results; method menu",
            "Drag / pan / zoom → the novel view follows",
            "Held-out photo vs render, one click"], 0.42, 3.55, 4.5, 1.7, size=14, gap=3)
bullets(s, ["Keyframes or preset orbit → Export MP4",
            "Assignment 2 meshes rasterised in the same view",
            "Runs on a laptop: our PyTorch renderers",
            "  (CUDA, Apple GPU or CPU)"], 5.1, 3.55, 4.5, 1.7, size=14, gap=3)

# 16 renderer check
if "ours_vs_official_psnr" in N.get("3dgs", {}):
    s = slide("GUI renderer = official renderer")
    rows = [["Model", "Same pixels? (PSNR vs official)", "Official FPS", "Ours FPS (CUDA)"]]
    for k, n in (("3dgs", "3DGS"), ("mip", "Mip-Splatting")):
        d = N.get(k, {})
        if "ours_vs_official_psnr" in d:
            rows.append([n, f"{d['ours_vs_official_psnr']['mean']:.0f} dB", f"{d['fps']:.0f}", f"{d['ours_fps_cuda']:.1f}"])
    table(s, rows, 0.42, 1.2, [2.2, 3.4, 1.6, 2.0], size=13, row_h=0.42)
    bullets(s, ["Above ~50 dB two images look identical; >100 dB = differences of rounding only",
                "Ours follows the CUDA code step by step, in plain PyTorch, so it also runs without CUDA",
                "Slower than CUDA, so the GUI draws a small preview while dragging, full size when the mouse stops"],
            0.42, 2.75, 9.2, 2.3, size=14, gap=6)

# 17 findings
s = slide("Findings")
best_k = max(NAMES, key=lambda kn: N[kn[0]]["mean"]["psnr"])
fast_k = min(NAMES, key=lambda kn: N[kn[0]]["train_seconds"])
items = [f"Best quality on held-out photos: {best_k[1]} ({N[best_k[0]]['mean']['psnr']:.2f} dB)",
         f"Fastest to train: {fast_k[1]} ({dur(N[fast_k[0]]['train_seconds'])}); Gaussians draw in real time",
         "At the training size 3DGS ≈ Mip-Splatting; zoomed out Mip-Splatting is clearly better"]
if "zoom" in N.get("mip", {}):
    items[-1] += f" (+{N['mip']['zoom']['x0.25']['psnr'] - N['3dgs']['zoom']['x0.25']['psnr']:.1f} dB)"
items += ["Clean masks matter: the white background makes the person easy to learn",
          "NeRF: same rendering idea as TensoRF, but one big network → slowest to train and to draw"]
bullets(s, items, 0.42, 1.0, 9.2, 4.2, size=16, gap=8)

# 18 references
s = slide("References")
refs = ["B. Mildenhall et al. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ECCV 2020.",
        "A. Chen, Z. Xu, A. Geiger, J. Yu, H. Su. TensoRF: Tensorial Radiance Fields. ECCV 2022.",
        "B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM TOG 42(4), 2023.",
        "Z. Yu, A. Chen, B. Huang, T. Sattler, A. Geiger. Mip-Splatting: Alias-free 3D Gaussian Splatting. CVPR 2024.",
        "M. Zwicker, H. Pfister, J. van Baar, M. Gross. EWA Volume Splatting. IEEE Visualization 2001.",
        "J. Wang, N. Karaev, C. Rupprecht, D. Novotny. VGGSfM: Visual Geometry Grounded Deep Structure from Motion. CVPR 2024.",
        "Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE TIP 13(4), 2004.",
        "R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018.",
        "Q.-Y. Zhou, J. Park, V. Koltun. Open3D: A Modern Library for 3D Data Processing. arXiv:1801.09847, 2018.",
        "M. Tancik et al. Nerfstudio: A Modular Framework for Neural Radiance Field Development. SIGGRAPH 2023."]
text(s, "\n".join("[%d] %s" % (i + 1, r) for i, r in enumerate(refs)), 0.42, 1.0, 9.2, 4.2, size=11)

prs.save(OUT)
print("saved", OUT, count[0], "slides")
