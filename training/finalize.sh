#!/usr/bin/env bash
# After training: score every model (one at a time, so the speed numbers are fair), render the result videos, export
# results/ for the GUI, then rebuild the figures, report and slides and run the tests.  Run on the GPU node.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
HI="${CV802_HIRES:-/l/users/${USER}/cv802_sp2/outputs/twodgs/twodgs-v2-30k-1600/work/scene/images}"
DOCS_PY="${CV802_DOCS_PY:-/l/users/${USER}/cv802_sp2/envs/docs/bin/python}"
V="${SP3}/results/videos"; mkdir -p "${V}"
trap 'status eval 0 failed "see logs/finalize.log"' ERR

if [[ -z "${SKIP_EVAL:-}" ]]; then        # SKIP_EVAL=1: only re-export and rebuild (NeRF scores itself in train_nerf.py)
status eval 5 running "3DGS: held-out photos, zoom, renderer check"
( cd "${TP}/gaussian-splatting" && "${GS_PY}" "${CODE}/training/eval_gs.py" --method 3dgs -m "${RUNS}/3dgs_person" --check-ours --zoom \
    --hires "${HI}" --path "${DATA}/orbit_path.json" --video "${V}/3dgs_person.mp4" )
status eval 30 running "Mip-Splatting"
( cd "${TP}/mip-splatting" && "${MIP_PY}" "${CODE}/training/eval_gs.py" --method mip -m "${RUNS}/mip_person" --check-ours --zoom \
    --hires "${HI}" --path "${DATA}/orbit_path.json" --video "${V}/mip_person.mp4" )
status eval 55 running "3DGS full room"
( cd "${TP}/gaussian-splatting" && "${GS_PY}" "${CODE}/training/eval_gs.py" --method 3dgs -m "${RUNS}/3dgs_full" \
    --path "${DATA}/orbit_path.json" --video "${V}/3dgs_full.mp4" )
status eval 70 running "TensoRF"
( cd "${TP}/TensoRF" && "${GS_PY}" "${CODE}/training/eval_tensorf.py" --ckpt "${RUNS}/tensorf_person/tensorf_person.th" \
    --data "${DATA}/person" --out "${RUNS}/tensorf_person/eval" --path "${DATA}/orbit_path.json" --video "${V}/tensorf_person.mp4" )
fi
status eval 100 done "5 models scored"
status videos 100 done "5 orbit videos (8 s)"

status gui 85 running "exporting results/ for the GUI"
"${GS_PY}" "${CODE}/training/export_bundle.py" "${SP3}" "${CODE}/results"
"${MIP_PY}" "${CODE}/docs/make_gui_figure.py" "${CODE}/results" "${CODE}/docs/figs/fig_gui.jpg"
CUDA_VISIBLE_DEVICES= "${MIP_PY}" "${CODE}/tests/test_nvs.py" "${CODE}/results"
status gui 100 done "results/ exported, tests pass"

status report 70 running "figures + LaTeX"
"${GS_PY}" "${CODE}/docs/make_figures.py" "${SP3}"
( cd "${CODE}/docs/report" && "${GS_PY}" build_tables.py && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null \
    && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main | grep "Output written" )
status report 100 done "docs/report/main.pdf"
status slides 70 running "building pptx"
"${DOCS_PY}" "${CODE}/docs/slides/build_slides.py"
status slides 100 done "docs/slides/*.pptx"
