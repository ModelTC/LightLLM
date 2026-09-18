#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 <master_port> <node_ip> [model_dir]" >&2
  exit 2
fi

PORT="$1"
NODE_IP="$2"
MODEL_DIR="${3:-/nvme/models/GLM-5.3-Flash}"

# P/D must advertise an address reachable by the master and the other node.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${NODE_IP}"
export no_proxy="${NO_PROXY}"
export LOADWORKER="${LOADWORKER:-8}"

COMMON_ARGS=(
  --model_dir "${MODEL_DIR}"
  --model_name glm53
  --tp 4
  --batch_max_tokens 8192
  --running_max_req_size 64
  --mem_fraction 0.8
  --enable_fused_shared_experts
  --tool_call_parser glm47
  --reasoning_parser glm45
  --linear_att_ssm_data_type float32
  --pd_trans_mode nccl
  # One transfer page must also fit the global Conv/SSM/indexer-tail state.
  --pd_kv_page_size 16384
  --pd_kv_page_num 2
  --pd_master_ip 127.0.0.1
  --pd_master_port "${PORT}"
  --host "${NODE_IP}"
)

PIDS=()
cleanup() {
  kill -TERM "${PIDS[@]}" 2>/dev/null || true
  wait "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lightllm.server.api_server \
  "${COMMON_ARGS[@]}" \
  --run_mode prefill \
  --disable_cudagraph \
  --port "$((PORT + 1))" \
  --nccl_port "$((PORT + 101))" &
PIDS+=("$!")

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m lightllm.server.api_server \
  "${COMMON_ARGS[@]}" \
  --run_mode decode \
  --graph_max_batch_size 64 \
  --graph_max_len_in_batch 65536 \
  --port "$((PORT + 2))" \
  --nccl_port "$((PORT + 102))" &
PIDS+=("$!")

CUDA_VISIBLE_DEVICES= python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --model_name glm53 \
  --run_mode pd_master \
  --pd_master_mode 1p1d \
  --tool_call_parser glm47 \
  --reasoning_parser glm45 \
  --host 127.0.0.1 \
  --port "${PORT}" &
PIDS+=("$!")

echo "GLM-5.3 Flash 1P1D is starting at http://127.0.0.1:${PORT} (P TP4 + D TP4)"
wait -n "${PIDS[@]}"
exit 1
