#!/usr/bin/env bash
# Launch SenseNova-Loop-9B with an optional DFlash2 draft.
set -euo pipefail
: "${MODEL_DIR:?Set MODEL_DIR to the matching BF16 SenseNova-Loop-9B checkpoint}"
: "${CUDA_VISIBLE_DEVICES:?Set four GPU IDs}"
METHOD="${METHOD:-dflash2}"
MTP_STEP="${MTP_STEP:-5}"
PYTHON_BIN="$(command -v "${PYTHON:-python3}")"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export LOADWORKER="${LOADWORKER:-4}"
export TOKENIZERS_PARALLELISM=false
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#gpu_ids[@]} -ne 4 ]]; then echo 'This script requires four GPU IDs.' >&2; exit 2; fi
case "$METHOD" in
    target) MTP_STEP=0; draft_args=() ;;
    dflash2)
        : "${DRAFT_MODEL_DIR:?Set DRAFT_MODEL_DIR to the compatible SenseNova-Loop-9B-DFlash2 checkpoint}"
        if [[ "$MTP_STEP" != 5 && "$MTP_STEP" != 7 ]]; then echo 'MTP_STEP must be 5 or 7 for this script.' >&2; exit 2; fi
        draft_args=(--mtp_mode dflash2 --mtp_draft_model_dir "$DRAFT_MODEL_DIR") ;;
    *) echo 'METHOD must be target or dflash2.' >&2; exit 2 ;;
esac
args=(--run_mode normal --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
    --nccl_port "${NCCL_PORT:-28765}" --tp 4 --model_dir "$MODEL_DIR"
    --model_name "${MODEL_NAME:-SenseNova-Loop-9B}" --data_type bfloat16
    --linear_att_ssm_data_type float32 --quant_type none --mem_fraction 0.8
    --disable_audio --output_constraint_mode none --vit_att_backend triton
    --max_req_total_len 524288 --running_max_req_size 64 --graph_max_batch_size 64
    --batch_max_tokens 8192 --chunked_prefill_size 8192 --linear_att_cache_size 500
    --linear_att_hash_page_size 2048 --linear_att_page_block_num 8
    --enable_cpu_cache --cpu_cache_storage_size 2 --enable_spec_acceptance_metrics
    --llm_prefill_att_backend fa3 triton --llm_decode_att_backend fa3 triton
    --mtp_step "$MTP_STEP")
if [[ "$METHOD" == dflash2 ]]; then args+=("${draft_args[@]}"); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "$PYTHON_BIN" -u -m lightllm.server.api_server "${args[@]}"; printf '\n'; exit 0
fi
exec "$PYTHON_BIN" -u -m lightllm.server.api_server "${args[@]}"
