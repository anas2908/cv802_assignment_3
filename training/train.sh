#!/usr/bin/env bash
# train.sh <step id> <method: 3dgs|mip|tensorf> <scene: person|full> [extra args for the official train.py]
# Trains one model with the official code; progress goes to logs/<step>.log and status/<step>.json (dashboard).
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
STEP="$1" METHOD="$2" SCENE="$3"; shift 3
RUN="${RUNS}/${METHOD}_${SCENE}"; LOG="${LOGS}/${STEP}.log"
mkdir -p "${RUN}"
status "${STEP}" 1 running "starting ${METHOD} on ${SCENE}"
trap 'status "${STEP}" 0 failed "see logs/${STEP}.log"' ERR
T0=$(date +%s)
case "${METHOD}" in
  3dgs)
    BG=(); [[ "${SCENE}" == person ]] && BG=(-w)
    ( cd "${TP}/gaussian-splatting" && "${GS_PY}" train.py -s "${DATA}/${SCENE}" -m "${RUN}" --eval "${BG[@]}" --disable_viewer \
        --test_iterations 7000 30000 --save_iterations 30000 "$@" ) > "${LOG}" 2>&1 ;;
  mip)
    BG=(); [[ "${SCENE}" == person ]] && BG=(-w)
    ( cd "${TP}/mip-splatting" && "${MIP_PY}" train.py -s "${DATA}/${SCENE}" -m "${RUN}" --eval "${BG[@]}" -r 1 --port $((6100 + RANDOM % 800)) \
        --test_iterations 7000 30000 --save_iterations 30000 "$@" ) > "${LOG}" 2>&1 ;;
  tensorf)
    ( cd "${TP}/TensoRF" && "${GS_PY}" train.py --config "${CODE}/training/tensorf_person.txt" --datadir "${DATA}/${SCENE}" \
        --basedir "${RUNS}" --expname "tensorf_${SCENE}" "$@" ) > "${LOG}" 2>&1 ;;
  nerf)   # trains for a time budget, then scores the held-out photos and renders the orbit video itself
    ( cd "${TP}/nerf-pytorch" && "${GS_PY}" "${CODE}/training/train_nerf.py" --data "${DATA}/${SCENE}" --out "${RUN}" \
        --path "${DATA}/orbit_path.json" --video "${SP3}/results/videos/nerf_${SCENE}.mp4" "$@" ) > "${LOG}" 2>&1
    status "${STEP}" 100 done "$(python3 -c "import json;d=json.load(open('${RUN}/train_time.json'));print('%d steps in %d min' % (d['iterations'], d['train_seconds'] // 60))")"
    exit 0 ;;
  *) echo "unknown method ${METHOD}"; exit 2 ;;
esac
echo "{\"train_seconds\": $(( $(date +%s) - T0 ))}" > "${RUN}/train_time.json"
status "${STEP}" 100 done "trained in $(( ($(date +%s) - T0) / 60 )) min"
