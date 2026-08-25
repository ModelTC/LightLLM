#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-16666}"
MODEL_DIR="${MODEL_DIR:-/mnt/afs/models/Qwen3.5-27B}"
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-${draft_model:-}}"
MODEL_NAME="${MODEL_NAME:-sensenova-flash-lite-v41-20260816-fp8-step3k-dspark}"

CPU_CACHE_SIZE="${CPU_CACHE_SIZE:-600}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${MODEL_DIR}/chat_template.jinja}"
PD_MASTER_IP="${PD_MASTER_IP:-127.0.0.1}"

# DSpark checkpoint block_size. Prefill and decode must use the same value.
MTP_STEP="${MTP_STEP:-5}"

# DSpark expands each logical decode request into speculative verification rows.
# Start conservatively; raise both values together after checking GPU headroom.
D_MAX_REQ_SIZE="${D_MAX_REQ_SIZE:-16}"
D_GRAPH_MAX_BATCH_SIZE="${D_GRAPH_MAX_BATCH_SIZE:-16}"

if [[ -z "${DRAFT_MODEL_DIR}" ]]; then
  echo "DRAFT_MODEL_DIR is required (the DSpark draft checkpoint directory)." >&2
  exit 2
fi

export LOADWORKER="${LOADWORKER:-8}"
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL="${LIGHTLLM_TRITON_AUTOTUNE_LEVEL:-1}"
export LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING="${LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING:-1}"
export LIGHTLLM_LOG_LEVEL="${LIGHTLLM_LOG_LEVEL:-debug}"
export PYTHONUNBUFFERED=1
export LIGHTLLM_PD_SPLIT_MAX_NEW_TOKENS="${LIGHTLLM_PD_SPLIT_MAX_NEW_TOKENS:-4096}"

# Keep same-host PD WebSocket traffic away from HTTP/WebSocket proxies.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${PD_MASTER_IP},localhost"
export no_proxy="${NO_PROXY}"

RUN_NAME="${RUN_NAME:-qwen35-dspark-1p1d}"
LOG_DIR="${LOG_ROOT:-/mnt/afs/lightllm-runs}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

P_COMMON_ARGS=(
  --model_dir "${MODEL_DIR}"
  --model_name "${MODEL_NAME}"
  --graph_max_batch_size 32
  --running_max_req_size 32
  --mem_fraction 0.75
  --max_image_token_count 4096
  --max_image_pixels 3686400
  --batch_max_tokens 32768
  --chunked_prefill_size 16384
  --linear_att_cache_size 500
  --linear_att_hash_page_size 2048
  --linear_att_page_block_num 8
  --quant_type fp8w8a8-pt-sgl
  --mtp_mode dspark
  --mtp_draft_model_dir "${DRAFT_MODEL_DIR}"
  --mtp_step "${MTP_STEP}"
  --chat_template "${CHAT_TEMPLATE}"
  --pd_trans_mode nixl
  --pd_kv_page_size 4096
  --pd_master_ip "${PD_MASTER_IP}"
  --pd_master_port "${PORT}"
  --llm_prefill_att_backend fa3 flashqla
)

D_COMMON_ARGS=(
  --model_dir "${MODEL_DIR}"
  --model_name "${MODEL_NAME}"
  --graph_max_batch_size "${D_GRAPH_MAX_BATCH_SIZE}"
  --running_max_req_size "${D_MAX_REQ_SIZE}"
  --mem_fraction 0.75
  --max_image_token_count 4096
  --max_image_pixels 3686400
  --batch_max_tokens 256
  --linear_att_cache_size 500
  --quant_type fp8w8a8-pt-sgl
  --mtp_mode dspark
  --mtp_draft_model_dir "${DRAFT_MODEL_DIR}"
  --mtp_step "${MTP_STEP}"
  --chat_template "${CHAT_TEMPLATE}"
  --pd_trans_mode nixl
  --pd_kv_page_size 4096
  --pd_master_ip "${PD_MASTER_IP}"
  --pd_master_port "${PORT}"
)

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

start_prefill() {
  local name="$1"
  local devices="$2"
  local http_port="$3"
  local nccl_port="$4"

  echo "Starting ${name}: GPUs=${devices}, port=${http_port}"

  CUDA_VISIBLE_DEVICES="${devices}" \
    python -m lightllm.server.api_server \
      "${P_COMMON_ARGS[@]}" \
      --run_mode prefill \
      --disable_cudagraph \
      --enable_cpu_cache \
      --cpu_cache_storage_size "${CPU_CACHE_SIZE}" \
      --tp 4 \
      --dp 1 \
      --visual_dp 4 \
      --host 0.0.0.0 \
      --port "${http_port}" \
      --nccl_port "${nccl_port}" \
      >>"${LOG_DIR}/${name}.log" 2>&1 &

  PIDS+=("$!")
}

start_decode() {
  local name="$1"
  local devices="$2"
  local http_port="$3"
  local nccl_port="$4"

  echo "Starting ${name}: GPUs=${devices}, port=${http_port}"

  CUDA_VISIBLE_DEVICES="${devices}" \
    python -m lightllm.server.api_server \
      "${D_COMMON_ARGS[@]}" \
      --run_mode decode \
      --tp 4 \
      --dp 1 \
      --visual_dp 4 \
      --host 0.0.0.0 \
      --port "${http_port}" \
      --nccl_port "${nccl_port}" \
      >>"${LOG_DIR}/${name}.log" 2>&1 &

  PIDS+=("$!")
}

# Single node: Prefill on GPUs 0-3, Decode on GPUs 4-7.
start_prefill p1 "0,1,2,3" 28761 29761
start_decode d1 "4,5,6,7" 28762 29762

# CPU-only PD Master. This is the public service endpoint.
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --model_name "${MODEL_NAME}" \
  --run_mode pd_master \
  --pd_master_mode 1p1d \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max_image_token_count 4096 \
  --max_image_pixels 3686400 \
  --chat_template "${CHAT_TEMPLATE}" \
  >>"${LOG_DIR}/pd-master.log" 2>&1 &
PIDS+=("$!")

echo "PD Master is starting at http://${PD_MASTER_IP}:${PORT}"
echo "logs: ${LOG_DIR}"

# A fixed 1P1D deployment is incomplete as soon as any component exits.
if wait -n "${PIDS[@]}"; then
  echo "A LightLLM component exited unexpectedly" >&2
else
  echo "A LightLLM component failed" >&2
fi
exit 1
