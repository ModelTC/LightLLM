#!/usr/bin/env bash
set -euo pipefail

ARGS=(
  --model_dir /mtc/models/Qwen3.5-122B-A10B
  --tp 8
  --port 8088
  --max_req_total_len 262144
  --linear_att_hash_page_size 8192
  --linear_att_page_block_num 8
)

if [[ "${ENABLE_DEEPEP:-0}" == "1" || "${ENABLE_DEEPEP:-}" == "true" ]]; then
  ARGS+=(--quant_cfg ../../advanced_config/mixed_quantization/qwen3_5-122b-moe-only-fp8.yaml --enable_ep_moe --dp 8 --batch_max_tokens 4096 --graph_max_batch_size 64 --chunked_prefill_size 2048 --mem_fraction 0.8 --linear_att_cache_size 300)
elif [[ -n "${QUANT_TYPE:-}" ]]; then
  ARGS+=(--mem_fraction 0.85 --quant_type "${QUANT_TYPE}" --linear_att_cache_size 3000)
else
  ARGS+=(--mem_fraction 0.85 --linear_att_cache_size 3000)
fi

ENABLE_CPU_CACHE_ARG=false
ENABLE_DISK_CACHE_ARG=false

if [[ "${ENABLE_DISK_CACHE:-0}" == "1" || "${ENABLE_DISK_CACHE:-}" == "true" || -n "${DISK_CACHE_STORAGE_SIZE:-}" || -n "${DISK_CACHE_DIR:-}" ]]; then
  ENABLE_DISK_CACHE_ARG=true
fi

if [[ "${ENABLE_CPU_CACHE:-0}" == "1" || "${ENABLE_CPU_CACHE:-}" == "true" || -n "${CPU_CACHE_STORAGE_SIZE:-}" || "${ENABLE_DISK_CACHE_ARG}" == "true" ]]; then
  ENABLE_CPU_CACHE_ARG=true
fi

if [[ "${ENABLE_CPU_CACHE_ARG}" == "true" ]]; then
  ARGS+=(--enable_cpu_cache)
fi

if [[ -n "${CPU_CACHE_STORAGE_SIZE:-}" ]]; then
  ARGS+=(--cpu_cache_storage_size "${CPU_CACHE_STORAGE_SIZE}")
fi

if [[ "${ENABLE_DISK_CACHE_ARG}" == "true" ]]; then
  ARGS+=(--enable_disk_cache)
fi

if [[ -n "${DISK_CACHE_STORAGE_SIZE:-}" ]]; then
  ARGS+=(--disk_cache_storage_size "${DISK_CACHE_STORAGE_SIZE}")
fi

if [[ -n "${DISK_CACHE_DIR:-}" ]]; then
  ARGS+=(--disk_cache_dir "${DISK_CACHE_DIR}")
fi

LOADWORKER="${LOADWORKER:-18}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  python -m lightllm.server.api_server "${ARGS[@]}"