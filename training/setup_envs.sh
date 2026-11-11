#!/usr/bin/env bash
# Build the training environments and fetch the official code (run on the GPU node).
#   gs  : Python 3.10, PyTorch 2.1 cu121, official 3DGS rasteriser + simple-knn; also runs TensoRF (pure PyTorch)
#   mip : same base, Mip-Splatting's own rasteriser (same module name as 3DGS's, so it needs its own env)
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
status setup 2 running "cloning official code"

clone() {  # clone <url> <dir> [commit]: pin to a commit when given, and record the one used
  [[ -d "$2/.git" ]] || git clone -q --recursive "$1" "$2"
  [[ -n "${3:-}" ]] && git -C "$2" checkout -q "$3"
  git -C "$2" submodule update --init --recursive -q
  echo "$(basename "$2") $(git -C "$2" rev-parse --short HEAD)" >> "${TP}/versions.txt"
}
: > "${TP}/versions.txt"
clone https://github.com/graphdeco-inria/gaussian-splatting "${TP}/gaussian-splatting" "${GS_COMMIT:-}"
clone https://github.com/autonomousvision/mip-splatting "${TP}/mip-splatting" "${MIP_COMMIT:-}"
clone https://github.com/apchenstu/TensoRF "${TP}/TensoRF" "${TENSORF_COMMIT:-}"

NVCC_DIR=""
for d in /apps/local/cuda-12.2/bin /usr/local/cuda-12.2/bin /usr/local/cuda/bin $(ls -d /apps/local/cuda*/bin 2>/dev/null | sort -r); do
  [[ -x "$d/nvcc" ]] && { NVCC_DIR="$d"; break; }
done
echo "nvcc: ${NVCC_DIR:-none}"; export PATH="${NVCC_DIR}:${PATH}" CUDA_HOME="$(dirname "${NVCC_DIR}")"
export CC=gcc-12 CXX=g++-12

base_env() {  # base_env <env dir>
  [[ -x "$1/bin/python" ]] || "${UV}" venv -q --python 3.10 "$1"
  "${UV}" pip install -q --python "$1/bin/python" torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
  "${UV}" pip install -q --python "$1/bin/python" "numpy<2" plyfile tqdm opencv-python-headless lpips scikit-image imageio \
      imageio-ffmpeg configargparse kornia tensorboard "setuptools<70" wheel ninja pillow scipy
}
status setup 20 running "gs environment"
base_env "${ENVS}/gs"
if ! "${GS_PY}" -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
  "${UV}" pip install -q --python "${GS_PY}" --no-build-isolation \
      "${TP}/gaussian-splatting/submodules/diff-gaussian-rasterization" "${TP}/gaussian-splatting/submodules/simple-knn"
fi
status setup 60 running "mip environment"
base_env "${ENVS}/mip"
if ! "${MIP_PY}" -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
  "${UV}" pip install -q --python "${MIP_PY}" --no-build-isolation \
      "${TP}/mip-splatting/submodules/diff-gaussian-rasterization" "${TP}/mip-splatting/submodules/simple-knn"
fi
"${UV}" pip install -q --python "${MIP_PY}" trimesh open3d==0.18.0   # mip-splatting train.py imports open3d
for P in "${GS_PY}" "${MIP_PY}"; do
  "$P" -c "import torch, diff_gaussian_rasterization, simple_knn; print('OK', torch.__version__, torch.cuda.is_available())"
done
status setup 100 done "environments ready"

# TensoRF: register our multi-camera loader ('percam') next to the official ones
cp "${CODE}/training/tensorf_percam.py" "${TP}/TensoRF/dataLoader/percam.py"
grep -q "percam" "${TP}/TensoRF/dataLoader/__init__.py" || printf "\nfrom .percam import PerCamDataset\ndataset_dict['percam'] = PerCamDataset\n" >> "${TP}/TensoRF/dataLoader/__init__.py"
grep -q "'percam'" "${TP}/TensoRF/opt.py" || sed -i "s/'own_data'\]/'own_data', 'percam']/" "${TP}/TensoRF/opt.py"
