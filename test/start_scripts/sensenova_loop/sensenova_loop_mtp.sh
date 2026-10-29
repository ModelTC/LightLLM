#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <loop_model_dir> <mtp_draft_model_dir> [server arguments...]" >&2
  exit 2
fi

MODEL_DIR="$1"
MTP_DIR="$2"
shift 2

export LOADWORKER="${LOADWORKER:-4}"
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL="${LIGHTLLM_TRITON_AUTOTUNE_LEVEL:-1}"

# A single MTP layer drafts recurrently, so mtp_step can exceed its layer count.
exec python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --mtp_draft_model_dir "${MTP_DIR}" \
  --mtp_mode eagle_with_att \
  --mtp_step "${MTP_STEP:-3}" \
  --tp "${TP:-4}" \
  --data_type bfloat16 \
  --llm_prefill_att_backend fa3 \
  --llm_decode_att_backend fa3 \
  --max_req_total_len 32768 \
  --batch_max_tokens 8192 \
  --running_max_req_size 32 \
  --graph_max_batch_size 32 \
  --linear_att_cache_size 64 \
  --reasoning_parser qwen3 \
  --port "${PORT:-8000}" \
  "$@"
