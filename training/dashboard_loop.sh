#!/usr/bin/env bash
# tmux helper: redraw the Sub-project 3 dashboard every 60 s (computed on the GPU node of the allocation).
D="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOB="${1:?usage: dashboard_loop.sh <slurm job id>}"
while true; do
  OUT=$(timeout 60 srun --jobid="$JOB" --overlap -n 1 -c 1 --mem=500M python3 "$D/dashboard.py" 2>&1)
  clear
  echo "$OUT"
  echo
  echo -e "\033[2mjob $JOB · refreshes every 60 s · Ctrl-b d to leave\033[0m"
  sleep 60
done
