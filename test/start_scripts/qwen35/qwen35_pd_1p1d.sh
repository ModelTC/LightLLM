#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "Usage: $0 <port> <model_dir> <cpu_cache_size_gb> <dspark_model_dir> [chat_template]" >&2
  exit 2
fi

PORT="$1"
MODEL_DIR="$2"
CPU_CACHE_SIZE="$3"
DSPARK_MODEL_DIR="$4"
CHAT_TEMPLATE_ARGS=()
if [[ -n "${5:-}" ]]; then
  CHAT_TEMPLATE_ARGS=(--chat_template "$5")
fi

# export PYTORCH_ALLOC_CONF=expandable_segments:True
export LOADWORKER=8
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING=1
export LIGHTLLM_LOG_LEVEL=debug

# Keep same-host PD WebSocket traffic away from HTTP/WebSocket proxies.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

P_COMMON_ARGS=(
  --model_dir "${MODEL_DIR}"
  --model_name qwen35_27b
  --graph_max_batch_size 8
  --running_max_req_size 8
  --mem_fraction 0.75
  --max_image_token_count 4096
  --max_image_pixels 3686400
  --batch_max_tokens 8192
  --linear_att_cache_size 500
  --linear_att_hash_page_size 2048
  --linear_att_page_block_num 8
  --quant_type fp8w8a8-pt-sgl
  --mtp_mode dspark
  --mtp_draft_model_dir "${DSPARK_MODEL_DIR}"
  --mtp_step 5
  "${CHAT_TEMPLATE_ARGS[@]}"
  --pd_trans_mode nccl
  --pd_kv_page_size 4096
  --pd_master_ip 127.0.0.1
  --pd_master_port "${PORT}"
)

D_COMMON_ARGS=(
  --model_dir "${MODEL_DIR}"
  --model_name qwen35_27b
  --graph_max_batch_size 64
  --running_max_req_size 64
  --mem_fraction 0.80
  --max_image_token_count 4096
  --max_image_pixels 3686400
  --batch_max_tokens 512
  --linear_att_cache_size 500
  --quant_type fp8w8a8-pt-sgl
  --mtp_mode dspark
  --mtp_draft_model_dir "${DSPARK_MODEL_DIR}"
  --mtp_step 5
  "${CHAT_TEMPLATE_ARGS[@]}"
  --pd_trans_mode nccl
  --pd_kv_page_size 4096
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

# Prefill: TP4 on GPUs 0-3.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lightllm.server.api_server \
  "${P_COMMON_ARGS[@]}" \
  --run_mode prefill \
  --disable_cudagraph \
  --enable_cpu_cache \
  --cpu_cache_storage_size "${CPU_CACHE_SIZE}" \
  --tp 4 \
  --visual_dp 4 \
  --host 0.0.0.0 \
  --port 28761 \
  --nccl_port 29761 &
PIDS+=("$!")

# Decode: TP4 on GPUs 4-7.
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m lightllm.server.api_server \
  "${D_COMMON_ARGS[@]}" \
  --run_mode decode \
  --tp 4 \
  --host 0.0.0.0 \
  --port 28762 \
  --nccl_port 29762 &
PIDS+=("$!")

# Start PD Master last, after Prefill and Decode have started.
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --model_name qwen35_27b \
  --run_mode pd_master \
  --pd_master_mode 1p1d \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max_image_token_count 4096 \
  --max_image_pixels 3686400 \
  "${CHAT_TEMPLATE_ARGS[@]}" &
PIDS+=("$!")

echo "1P1D D-Spark service is starting at http://127.0.0.1:${PORT}"

# A fixed 1P1D deployment is incomplete as soon as any component exits.
wait -n "${PIDS[@]}"
exit 1
