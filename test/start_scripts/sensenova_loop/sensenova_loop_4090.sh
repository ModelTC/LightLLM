#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the SenseNova Loop AWQ checkpoint directory}"
: "${KV_CALIB:?Set KV_CALIB to the matching per_tensor FP8 KV calibration JSON}"
CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
case "$CONTEXT_LENGTH" in
    65536|131072|262144) ;;
    *) printf 'CONTEXT_LENGTH must be 65536, 131072 or 262144\n' >&2; exit 2 ;;
esac

# Explicit capacity avoids filling the rest of the 24 GiB card with KV pages.
# The extra 512 tokens cover the allocator's internal hold/padding slots.
command=(
    "${PYTHON_BIN:-python3}" -m lightllm.server.api_server
    --model_dir "$MODEL_DIR" --model_name sensenova_loop
    --tp 1 --dp 1 --data_type bfloat16 --port "${PORT:-8000}"
    --running_max_req_size 1
    --max_req_total_len "$CONTEXT_LENGTH"
    --max_total_token_num "$((CONTEXT_LENGTH + 512))"
    --batch_max_tokens 512 --chunked_prefill_size 512
    --linear_att_cache_size 2 --linear_att_ssm_data_type float32
    --graph_max_batch_size 1 --graph_max_len_in_batch "$CONTEXT_LENGTH"
    --llm_kv_type fp8kv_spt --kv_quant_calibration_config_path "$KV_CALIB"
    --llm_prefill_att_backend flashinfer triton
    --llm_decode_att_backend flashinfer triton
    --flashinfer_workspace_size 128
    --disable_vision --disable_audio
)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LOADWORKER=1
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q LOADWORKER=1 ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
[[ -f "$MODEL_DIR/config.json" ]] || { printf 'Missing model config.json\n' >&2; exit 2; }
[[ -f "$KV_CALIB" ]] || { printf 'Missing KV calibration JSON\n' >&2; exit 2; }
exec "${command[@]}"
