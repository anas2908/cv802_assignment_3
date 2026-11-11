# Shared paths for the training pipeline (sourced by every script in training/).
export SP3="${CV802_SP3_ROOT:-/l/users/${USER}/cv802_sp3}"
export CODE="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TP="${SP3}/third_party" ENVS="${SP3}/envs" DATA="${SP3}/data" RUNS="${SP3}/runs" LOGS="${SP3}/logs" STATUS="${SP3}/status"
export UV="${CV802_UV:-/l/users/${USER}/cv802_sp2/tools/uv/uv}"
export UV_CACHE_DIR="${CV802_UV_CACHE:-/l/users/${USER}/cv802_sp2/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${CV802_UV_PYTHONS:-/l/users/${USER}/cv802_sp2/tools/python}"
export TORCH_HOME="${SP3}/cache/torch" XDG_CACHE_HOME="${SP3}/cache/xdg"
export GS_PY="${ENVS}/gs/bin/python" MIP_PY="${ENVS}/mip/bin/python"
export PREP_PY="${CV802_PREP_PY:-/l/users/${USER}/cv802_sp2/envs/base/bin/python}"   # has pycolmap
export TORCH_CUDA_ARCH_LIST="8.9"
mkdir -p "${TP}" "${ENVS}" "${DATA}" "${RUNS}" "${LOGS}" "${STATUS}"

# progress line for the tmux dashboard: status <step> <percent> <state> <message>
status() { printf '{"step": "%s", "percent": %s, "state": "%s", "message": "%s", "time": %s}\n' "$1" "$2" "$3" "$4" "$(date +%s)" >| "${STATUS}/$1.json"; }
