#!/usr/bin/env bash

PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8010}"

LIGHTLLM_ENABLE_GQA_DIVERSE_DECODE_FAST_KERNEL=1 \
GHTLLM_USE_TRITON_FP8_SCALED_MM=1 \
LOADWORKER=8 \
"${PYTHON_BIN}" -m lightllm.server.api_server \
  --port "${PORT}" \
  --model_dir /data0/models/Qwen3-Omni-30B-A3B-Instruct/ \
  --tp 4 \
  --graph_max_batch_size 16 \
  --chunked_prefill_size 8192 \
  --disable_vision \
  --disable_audio \
  --batch_max_tokens 8192 \
  --max_req_total_len 65536 \
  --mem_fraction 0.70 \
  --data_type bfloat16 \
  --enable_prefill_cudagraph \
  --schedule_time_interval 0.005 \
  --graph_max_len_in_batch 8192 \
  --llm_decode_att_backend triton
