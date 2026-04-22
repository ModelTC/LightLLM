#!/usr/bin/env bash
set -euo pipefail

export BS=8
export ILEN=10000
export OLEN=500

export ROOT_DIR=/mtc/niushengxiao/lightllm
export MODEL_DIR=/mtc/niushengxiao/Qwen3-235B-A22B
export PROFILE_DIR=$ROOT_DIR/profiles
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

rm -rf "$PROFILE_DIR/qwen3_static_decode"


vllm bench latency \
  --model "$MODEL_DIR" \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --batch-size "$BS" \
  --input-len "$ILEN" \
  --output-len "$OLEN" \
  --num-iters-warmup 1 \
  --disable-detokenize \
  --profile \
  --enforce-eager \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir "$PROFILE_DIR/qwen3_static_decode" \
  --profiler-config.delay_iterations 1 \
  --profiler-config.max_iterations 1 \
  --profiler-config.ignore_frontend true
