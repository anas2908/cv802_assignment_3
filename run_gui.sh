#!/usr/bin/env bash
# One command on a laptop (macOS or Linux): create .venv on first use, install the requirements, open the GUI on the
# trained results. No CUDA needed (Apple silicon uses its GPU through PyTorch MPS). Needs Python 3.10-3.12
# (open3d 0.19 has no wheels for 3.9 or 3.13); on a Mac: brew install python@3.11
#   bash run_gui.sh            (or: bash run_gui.sh <results folder | mesh.obj | colmap folder>)
set -euo pipefail
cd "$(dirname "$0")"

ok() { "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; }
if [ -x .venv/bin/python ] && ! ok .venv/bin/python; then
  echo "Removing .venv: it was made with $(.venv/bin/python --version 2>&1), which open3d does not support"
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3.11 python3.12 python3.10 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12 \
           /opt/homebrew/bin/python3.10 /usr/local/bin/python3.11 /usr/local/bin/python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1 && ok "$(command -v "$c")"; then PY="$(command -v "$c")"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10, 3.11 or 3.12 is needed (found: $(python3 --version 2>&1))."
    echo "On a Mac:  brew install python@3.11   then run  bash run_gui.sh  again."
    exit 1
  fi
  echo "Creating .venv with $("$PY" --version) and installing requirements (first run only, a few minutes)"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
fi
exec .venv/bin/python main.py "${1:-results}"
