# CV802 Assignment 3: Novel-view Synthesis

The 125 photos of a person from Assignments 1 and 2, with the cameras recovered in Assignment 1 (VGGSfM), turned into
3D scenes that can be drawn from **any** viewpoint, with the four methods of the course list:

| Method | Paper | Code used | Scene stored as |
|---|---|---|---|
| NeRF | Mildenhall et al., ECCV 2020 | nerf-pytorch (PyTorch port of the original TensorFlow code) | one network (8 layers × 256) |
| TensoRF | Chen et al., ECCV 2022 | official | a feature grid split into planes and lines, read by a small network |
| 3D Gaussian Splatting (3DGS) | Kerbl et al., ACM TOG 2023 | official | ~200k coloured 3D Gaussians |
| Mip-Splatting | Yu et al., CVPR 2024 | official | 3DGS + a 3D and a 2D anti-aliasing filter |

plus an extra **3DGS of the whole room** (the same photos without the person masks). All methods are trained with
their official code; NeRF for one hour (the paper trains for 200k+ steps).

The code is built **on the course starter code** (the Open3D GUI of Assignment 1, `main.py` and `modules/`): the same
window now opens trained novel-view models, renders them live from the 3D view's camera while you drag, compares a
render with the real held-out photo, exports MP4 videos along a preset or hand-made camera path, and rasterises the
Assignment 2 meshes.

## Open the GUI (laptop, no CUDA needed)

```bash
git clone https://github.com/anas2908/cv802_assignment_3.git
cd cv802_assignment_3
bash run_gui.sh
```

`run_gui.sh` creates `.venv` the first time (Python 3.10–3.12; on a Mac without one: `brew install python@3.11`),
installs `requirements.txt` and runs `python main.py results`. The trained models ship in `results/`. On Apple
silicon the renderers run on the GPU through PyTorch (MPS); elsewhere on CUDA or the CPU.

In the window (panel on the right):

- **Novel-view synthesis**
  - *Method*: NeRF, TensoRF, 3D Gaussian Splatting, Mip-Splatting, 3DGS full room (NeRF and TensoRF march rays and
    are slow to draw on a laptop; the Gaussian methods are interactive)
  - *Live novel view*: drag (turn), right-drag (pan), scroll (zoom): the novel view follows the 3D view's camera.
    While moving, a small preview is drawn (*Moving* slider = its size); when the mouse stops, the full image
    (*Still* slider)
  - *Held-out photos*: jump to one of the 16 photos not used in training; *Show the real photo* to compare
  - *Add keyframe* / *Clear*, *Path* (preset orbit or my keyframes), *Seconds*, **Export MP4…**
- **Meshes (Sub-project 2)**: the Screened Poisson and NKSR meshes, textured, drawn by Open3D's rasteriser in the
  same view (also *File > Open mesh…* for any `.obj/.ply/.glb`)
- The Assignment 1 panels (COLMAP, camera list, point size …) are unchanged.

`python main.py <folder>` also opens a COLMAP result (Assignment 1) and `python main.py <mesh>` a mesh.

## Results

<!-- RESULTS -->

Videos along the preset 8-second orbit: [`results/videos/`](results/videos/). Report: [`docs/report/main.pdf`](docs/report/main.pdf).
Slides: [`docs/slides/`](docs/slides/).

## How the renderers in the GUI work

The official 3DGS and Mip-Splatting renderers are CUDA-only. [`modules/nvs/gaussians.py`](modules/nvs/gaussians.py)
re-implements their rasteriser in plain PyTorch, step by step (projection with the EWA Jacobian, spherical-harmonics
colour, 16×16-pixel tiles sorted by depth, front-to-back alpha blending with the same thresholds), and
[`modules/nvs/tensorf.py`](modules/nvs/tensorf.py) and [`modules/nvs/nerf.py`](modules/nvs/nerf.py) repeat TensoRF's and
nerf-pytorch's ray marching for their trained weights (nothing is trained with these files). On the held-out photos they
reproduce the official renders (PSNR above 100 dB between the two, i.e. identical up to rounding;
`training/eval_gs.py --check-ours`). `tests/test_nvs.py` checks the models and the GUI's camera conversion without a
window.

## Training (cluster, one NVIDIA GPU)

```bash
bash training/setup_envs.sh                     # official code + environments (3DGS, Mip-Splatting, TensoRF)
python training/prep_scenes.py --src <2DGS scene of Sub-project 2> --out $DATA   # photos, masks, cameras, split
bash training/train.sh train_3dgs 3dgs person    # also: mip person, tensorf person, 3dgs full, nerf person --minutes 60
python training/eval_gs.py --method 3dgs -m <run> --check-ours --zoom --hires <1600px photos> --path orbit.json --video out.mp4
python training/eval_tensorf.py --ckpt <run>/tensorf_person.th --data $DATA/person --out <run>/eval
python training/export_bundle.py <SP3 root> results    # models (.npz), cameras, photos, meshes, videos -> results/
```

Paths are set in `training/env.sh`. The data: 125 photos, undistorted with the Assignment 1 cameras, person masks →
white background, 1000 px; 109 photos train, every 8th photo (16) is held out. `training/dashboard.py` shows progress
in tmux.

## Layout

```text
main.py, modules/gui/      course starter GUI (Assignment 1) + novel-view and mesh panels
modules/colmap/            Assignment 1 SfM (unchanged)
modules/nvs/               cameras and paths, PyTorch renderers (Gaussians, TensoRF), results-folder session
training/                  data preparation, training, evaluation, export, progress dashboard
results/                   trained models (.npz), cameras, held-out photos, SP2 meshes, result videos
docs/                      report (LaTeX), slides (python-pptx), figures
tests/                     tests (SfM of Assignment 1, novel-view renderers and camera conversion)
```

## References

Mildenhall et al., NeRF, ECCV 2020 · Chen et al., TensoRF, ECCV 2022 · Kerbl et al., 3D Gaussian Splatting, ACM TOG
2023 · Yu et al., Mip-Splatting, CVPR 2024 · Zwicker et al., EWA Volume Splatting, IEEE Visualization 2001 · Wang et
al., VGGSfM, CVPR 2024 · Zhou et al., Open3D, arXiv 2018. Full list in the report.
