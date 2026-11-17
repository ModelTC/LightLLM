#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "Usage: $0 <port> <model_dir> <cpu_cache_size_gb> [chat_template]" >&2
  exit 2
fi

PORT="$1"
MODEL_DIR="$2"
CPU_CACHE_SIZE="$3"
CHAT_TEMPLATE_ARGS=()
if [[ -n "${4:-}" ]]; then
  CHAT_TEMPLATE_ARGS=(--chat_template "$4")
fi

# GLM-5.3 Flash requires BF16 for dynamic FP8 KV.
LLM_KV_TYPE="${LLM_KV_TYPE:-fp8kv_dsa}"
DATA_TYPE="${DATA_TYPE:-bfloat16}"
MTP_STEP="${MTP_STEP:-3}"
PD_KV_PAGE_SIZE=16384
if [[ "$LLM_KV_TYPE" == "fp8kv_dsa" ]]; then
  PD_KV_PAGE_SIZE=32768
fi

export LOADWORKER="${LOADWORKER:-8}"
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL="${LIGHTLLM_TRITON_AUTOTUNE_LEVEL:-1}"
export LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING=1
export LIGHTLLM_LOG_LEVEL="${LIGHTLLM_LOG_LEVEL:-info}"

# Keep same-host PD WebSocket traffic away from HTTP/WebSocket proxies.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

P_COMMON_ARGS=(
  --model_dir "${MODEL_DIR}"
  --model_name glm53
  --data_type "${DATA_TYPE}"
  --llm_kv_type "${LLM_KV_TYPE}"
  --graph_max_batch_size 8
  --running_max_req_size 64
  --mem_fraction 0.80
  --max_image_token_count 4096
  --max_image_pixels 3686400
  --batch_max_tokens 8192
  --linear_att_cache_size 500
  --linear_att_hash_page_size 2048
  --linear_att_page_block_num 8
  --linear_att_ssm_data_type float32
  --enable_fused_shared_experts
  --mtp_mode eagle_with_att
  --mtp_draft_model_dir "${MODEL_DIR}"
  --mtp_step "${MTP_STEP}"
  --tool_call_parser glm47
  --reasoning_parser glm45
  "${CHAT_TEMPLATE_ARGS[@]}"
  --pd_trans_mode nccl
  --pd_kv_page_size "${PD_KV_PAGE_SIZE}"
  --pd_kv_page_num 2
  --pd_master_ip 127.0.0.1
  --pd_master_port "${PORT}"
)

D_COMMON_ARGS=(
  --model_dir "${MODEL_DIR}"
  --model_name glm53
  --data_type "${DATA_TYPE}"
  --llm_kv_type "${LLM_KV_TYPE}"
  --graph_max_batch_size 64
  --graph_max_len_in_batch 65536
  --running_max_req_size 64
  --mem_fraction 0.80
  --batch_max_tokens 256
  --linear_att_cache_size 500
  --linear_att_hash_page_size 2048
  --linear_att_ssm_data_type float32
  --enable_fused_shared_experts
  --mtp_mode eagle_with_att
  --mtp_draft_model_dir "${MODEL_DIR}"
  --mtp_step "${MTP_STEP}"
  --tool_call_parser glm47
  --reasoning_parser glm45
  "${CHAT_TEMPLATE_ARGS[@]}"
  --pd_trans_mode nccl
  --pd_kv_page_size "${PD_KV_PAGE_SIZE}"
  --pd_kv_page_num 2
  --pd_master_ip 127.0.0.1
  --pd_master_port "${PORT}"
)

PIDS=()

cleanup() {
  kill -TERM "${PIDS[@]}" 2>/dev/null || true
  wait "${PIDS[@]}" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Prefill 1: TP4 on GPUs 0-3.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lightllm.server.api_server \
  "${P_COMMON_ARGS[@]}" \
  --run_mode prefill \
  --disable_cudagraph \
  --enable_cpu_cache \
  --cpu_cache_storage_size "${CPU_CACHE_SIZE}" \
  --tp 4 \
  --host 0.0.0.0 \
  --port 28761 \
  --nccl_port 29761 &
PIDS+=("$!")

# Prefill 2: TP4 on GPUs 4-7.
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m lightllm.server.api_server \
  "${P_COMMON_ARGS[@]}" \
  --run_mode prefill \
  --disable_cudagraph \
  --enable_cpu_cache \
  --cpu_cache_storage_size "${CPU_CACHE_SIZE}" \
  --tp 4 \
  --host 0.0.0.0 \
  --port 28762 \
  --nccl_port 29762 &
PIDS+=("$!")

# Decode: TP4 on GPUs 8-11.
CUDA_VISIBLE_DEVICES=8,9,10,11 python -m lightllm.server.api_server \
  "${D_COMMON_ARGS[@]}" \
  --run_mode decode \
  --tp 4 \
  --host 0.0.0.0 \
  --port 28763 \
  --nccl_port 29763 &
PIDS+=("$!")

# Start PD Master last, after all Prefill and Decode processes have started.
CUDA_VISIBLE_DEVICES= python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --model_name glm53 \
  --run_mode pd_master \
  --pd_master_mode 2p1d \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max_image_token_count 4096 \
  --max_image_pixels 3686400 \
  --tool_call_parser glm47 \
  --reasoning_parser glm45 \
  "${CHAT_TEMPLATE_ARGS[@]}" &
PIDS+=("$!")

echo "GLM-5.3 Flash 2P1D is starting at http://127.0.0.1:${PORT} (P0/P1/D: TP4)"

# A fixed 2P1D deployment is incomplete as soon as any component exits.
wait -n "${PIDS[@]}"
exit 1
