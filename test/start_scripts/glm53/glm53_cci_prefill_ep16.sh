#!/usr/bin/env bash
# Standalone CCI Prefill + multi-master startup: two replicas, eight GPUs per replica.
# Usage: bash glm53_cci_prefill_ep16.sh <node_rank> <node_ip> <p0_ip>
# Positional arguments may be omitted when NODE_RANK, NODE_IP, P_MASTER_IP are set.
# Use a LightLLM image with compatible CUDA, PyTorch, DeepEP and DeepGEMM.
set -euo pipefail

if (($# != 0 && $# != 3)); then
  echo "Usage: $0 <node_rank> <node_ip> <p0_ip>" >&2
  exit 2
fi
ROLE=p
NODE_RANK="${1:-${NODE_RANK:-}}"
NODE_IP="${2:-${NODE_IP:-}}"
P_MASTER_IP="${3:-${P_MASTER_IP:-}}"
D_MASTER_IP="${D_MASTER_IP:-}"
: "${NODE_RANK:?Set the replica index within this CCI job: 0 or 1}"
: "${NODE_IP:?Set the routable IP of this replica}"
: "${P_MASTER_IP:?Set the routable IP of P job replica 0, identical in both jobs}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || { echo "NODE_RANK must be 0 or 1" >&2; exit 2; }

PYTHON="${PYTHON:-python}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-/mnt/vqa/models/GLM-5.3-Flash}"
MODEL_DIR="${MODEL_DIR:-/nvme/GLM-5.3-Flash}"
COPY_JOBS="${COPY_JOBS:-16}"
CODE_BRANCH="${CODE_BRANCH:-main}"
# Pin all four replicas to the same revision, even if main advances during startup.
CODE_REV="${CODE_REV:-70f3927fb6be3b932f1610831876d74e6f646321}"
CODE_DIR="${CODE_DIR:-/nvme/LightLLM-cci-glm53-${CODE_REV:0:12}}"
LOG_DIR="${LOG_DIR:-/nvme/glm53-cci-logs/${ROLE}-${NODE_RANK}}"
CONFIG_SERVER_PORT="${CONFIG_SERVER_PORT:-60088}"
PD_MASTER_BASE_PORT="${PD_MASTER_BASE_PORT:-60011}"
PD_MASTER_PROCESSES="${PD_MASTER_PROCESSES:-4}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
MAX_REQ_TOTAL_LEN="${MAX_REQ_TOTAL_LEN:-65536}"
RUNNING_MAX_REQ_SIZE="${RUNNING_MAX_REQ_SIZE:-256}"
BATCH_MAX_TOKENS="${BATCH_MAX_TOKENS:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.8}"
GRAPH_MAX_BATCH_SIZE="${GRAPH_MAX_BATCH_SIZE:-16}"
[[ "${PD_MASTER_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || { echo "PD_MASTER_PROCESSES must be positive" >&2; exit 2; }
for cmd in find xargs cp git curl "${PYTHON}"; do
  command -v "${cmd}" >/dev/null || { echo "Missing command: ${cmd}" >&2; exit 1; }
done

# 1. Copy the model to this replica's local NVMe before cloning/installing code.
[[ -f "${SOURCE_MODEL_DIR}/config.json" ]] || { echo "Missing model: ${SOURCE_MODEL_DIR}/config.json" >&2; exit 1; }
mkdir -p "${MODEL_DIR}" "${LOG_DIR}"
echo "[${ROLE}${NODE_RANK}] Copying model with ${COPY_JOBS} concurrent processes"
find "${SOURCE_MODEL_DIR}" -mindepth 1 -maxdepth 1 -print0 \
  | xargs -0 -r -P "${COPY_JOBS}" -I{} cp -a -- "{}" "${MODEL_DIR}/"

# 2. Clone a private checkout on this replica; never reset an existing checkout.
if [[ ! -e "${CODE_DIR}" ]]; then
  git clone --depth 1 --single-branch --branch "${CODE_BRANCH}" \
    https://github.com/ModelTC/LightLLM.git "${CODE_DIR}"
  if [[ "$(git -C "${CODE_DIR}" rev-parse HEAD)" != "${CODE_REV}" ]]; then
    git -C "${CODE_DIR}" fetch --depth 1 origin "${CODE_REV}"
    git -C "${CODE_DIR}" checkout --detach "${CODE_REV}"
  fi
fi
cd "${CODE_DIR}"
[[ "$(git rev-parse HEAD)" == "${CODE_REV}" ]] || {
  echo "CODE_DIR has a different revision; choose a new CODE_DIR or matching CODE_REV" >&2
  exit 1
}
echo "[${ROLE}${NODE_RANK}] Using code $(git rev-parse HEAD)"

# Config-server discovery uses HTTP GET. This revision incorrectly uses ws://
# for /registered_objects, which httpx rejects. Patch only that URL in our clone.
"${PYTHON}" - <<'PY'
from pathlib import Path

path = Path("lightllm/server/httpserver/pd_loop.py")
source = path.read_text()
old = 'f"ws://{args.config_server_host}:{get_shm_port_args().config_server_port}/registered_objects"'
new = 'f"http://{args.config_server_host}:{get_shm_port_args().config_server_port}/registered_objects"'
if old in source:
    if source.count(old) != 1:
        raise SystemExit("Unexpected config-server discovery code; review pd_loop.py")
    path.write_text(source.replace(old, new))
elif new not in source:
    raise SystemExit("Config-server discovery code changed; review pd_loop.py")
PY

# 3. The LightLLM image supplies native GPU libraries. Optionally refresh Python
# requirements; pip -e alone does not build DeepEP/DeepGEMM or install CUDA.
if [[ "${INSTALL_REQUIREMENTS:-0}" == 1 ]]; then
  "${PYTHON}" -m pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
fi
"${PYTHON}" -m pip install --no-build-isolation --no-deps -e .
"${PYTHON}" - <<'PY'
import importlib.util
import torch

missing = [name for name in ("deep_ep", "deep_gemm", "sgl_kernel", "flashinfer")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Use a compatible LightLLM image; missing native dependencies: {missing}")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"Each replica needs 8 visible GPUs; found {torch.cuda.device_count()}")
PY

# Internal P/D and discovery traffic must connect directly across both jobs.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${NODE_IP},${P_MASTER_IP}${D_MASTER_IP:+,${D_MASTER_IP}}"
export no_proxy="${NO_PROXY}"
export LOADWORKER="${LOADWORKER:-8}"
export PYTHONUNBUFFERED=1

PIDS=()
cleanup() {
  if ((${#PIDS[@]} > 0)); then
    kill -TERM "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_service() {
  local name="$1"
  shift
  "$@" >>"${LOG_DIR}/${name}.log" 2>&1 &
  PIDS+=("$!")
  echo "[${ROLE}${NODE_RANK}] Started ${name}, pid=$!, log=${LOG_DIR}/${name}.log"
}

wait_http() {
  local url="$1" deadline=$((SECONDS + STARTUP_TIMEOUT)) pid
  echo "[${ROLE}${NODE_RANK}] Waiting for ${url}"
  while ((SECONDS < deadline)); do
    for pid in "${PIDS[@]}"; do
      kill -0 "${pid}" 2>/dev/null || { echo "A local service exited; inspect ${LOG_DIR}" >&2; return 1; }
    done
    if curl --noproxy '*' -fsS --connect-timeout 2 --max-time 3 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for ${url}; inspect ${LOG_DIR} and CCI networking" >&2
  return 1
}

# 4. P0 hosts discovery and several independent, CPU-only PD master processes.
if [[ "${NODE_RANK}" == 0 ]]; then
  start_service config-server env CUDA_VISIBLE_DEVICES= "${PYTHON}" -m lightllm.server.api_server \
    --run_mode config_server --host "${P_MASTER_IP}" --port "${CONFIG_SERVER_PORT}" \
    --config_server_host "${P_MASTER_IP}" --config_server_port "${CONFIG_SERVER_PORT}"
fi
wait_http "http://${P_MASTER_IP}:${CONFIG_SERVER_PORT}/health"

if [[ "${NODE_RANK}" == 0 ]]; then
  for ((i = 0; i < PD_MASTER_PROCESSES; i++)); do
    start_service "pd-master-${i}" env CUDA_VISIBLE_DEVICES= "${PYTHON}" -m lightllm.server.api_server \
      --model_dir "${MODEL_DIR}" --model_name glm53 \
      --run_mode pd_master --pd_master_mode elastic --httpserver_workers 1 \
      --host "${P_MASTER_IP}" --port "$((PD_MASTER_BASE_PORT + i))" \
      --config_server_host "${P_MASTER_IP}" --config_server_port "${CONFIG_SERVER_PORT}" \
      --max_req_total_len "${MAX_REQ_TOTAL_LEN}" --tool_call_parser glm47 --reasoning_parser glm45
  done
  for ((i = 0; i < PD_MASTER_PROCESSES; i++)); do
    wait_http "http://${P_MASTER_IP}:$((PD_MASTER_BASE_PORT + i))/liveness"
  done
  echo "PD master endpoints: http://${P_MASTER_IP}:${PD_MASTER_BASE_PORT} through port $((PD_MASTER_BASE_PORT + PD_MASTER_PROCESSES - 1))"
fi

# 5. In LightLLM --tp is the total world size; this is EP16 + DP16 on 16 GPUs.
COMMON_ARGS=(
  --model_dir "${MODEL_DIR}" --model_name glm53
  --tp 16 --dp 16 --nnodes 2 --node_rank "${NODE_RANK}" --enable_ep_moe
  --host "${NODE_IP}"
  --config_server_host "${P_MASTER_IP}" --config_server_port "${CONFIG_SERVER_PORT}"
  --max_req_total_len "${MAX_REQ_TOTAL_LEN}" --running_max_req_size "${RUNNING_MAX_REQ_SIZE}"
  --batch_max_tokens "${BATCH_MAX_TOKENS}" --mem_fraction "${MEM_FRACTION}"
  --tool_call_parser glm47 --reasoning_parser glm45
  --linear_att_ssm_data_type float32
  --pd_trans_mode nccl --pd_kv_page_size 16384 --pd_kv_page_num 2
)
start_service prefill "${PYTHON}" -m lightllm.server.api_server "${COMMON_ARGS[@]}" \
  --run_mode prefill --disable_cudagraph --port "${P_HTTP_PORT:-8019}" \
  --nccl_host "${P_MASTER_IP}" --nccl_port "${P_NCCL_PORT:-29500}"

# Let CCI observe failures and stop our other local services on exit/signals.
status=0
wait -n "${PIDS[@]}" || status=$?
echo "A LightLLM service exited (status=${status}); inspect ${LOG_DIR}" >&2
exit 1
