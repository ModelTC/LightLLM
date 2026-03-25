#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/FlashMLA:${ROOT_DIR}/sglang/sgl-kernel/python:${PYTHONPATH:-}"

pytest \
  "${ROOT_DIR}/unit_tests/models/deepseek3_2/triton_kernel/test_destindex_copy_kv_flashmla_fp8.py" \
  "${ROOT_DIR}/unit_tests/models/deepseek3_2/test_flashmla_fp8_sparse_decode.py" \
  -s

python "${ROOT_DIR}/test/kernel/benchmark_deepseekv32_fp8kv_dsa.py" --tokens-list 10000 100000 1000000 --page-size-list 1 64 128 256 --iters 100 --warmup 20
python "${ROOT_DIR}/test/kernel/benchmark_deepseekv32_sparse_decode_fp8_vs_bf16.py" --batch 64 --heads 128 --cache-tokens-list 10000 100000 1000000 --page-size-list 1 64 128 256 --topk 2048 --iters 100 --warmup 20 --check-correctness
