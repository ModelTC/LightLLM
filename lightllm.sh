export BS=8
export ILEN=10000
export OLEN=500
export NCCL_HOST=127.0.0.1
export NCCL_PORT=28765
export MODEL_DIR=${MODEL_DIR:-../Qwen3-235B-A22B}
export TP=${TP:-8}
export BENCH_MODE=${1:-${BENCH_MODE:-both}}
export CACHE_HIT_LEN=${CACHE_HIT_LEN:-$((ILEN - 1))}

run_static_forward_bench() {
  echo "[LightLLM] run cold prefill+decode benchmark"
  LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=18 \
  python test/benchmark/static_inference/test_model.py \
    --model_dir "$MODEL_DIR" \
    --tp "$TP" \
    --nccl_host "$NCCL_HOST" \
    --nccl_port "$NCCL_PORT" \
    --data_type bfloat16 \
    --graph_max_batch_size 200 \
    --enable_prefill_cudagraph \
    --llm_prefill_att_backend flashinfer \
    --llm_decode_att_backend flashinfer \
    --sampling_backend sglang_kernel \
    --batch_size "$BS" \
    --input_len "$ILEN" \
    --output_len "$OLEN" \
    --torch_profile
}

run_cache_hit_forward_bench() {
  echo "[LightLLM] run cache-hit prefill+decode benchmark, cache_hit_len=${CACHE_HIT_LEN}"
  LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=18 \
  python test/benchmark/static_inference/test_model.py \
    --model_dir "$MODEL_DIR" \
    --tp "$TP" \
    --nccl_host "$NCCL_HOST" \
    --nccl_port "$NCCL_PORT" \
    --data_type bfloat16 \
    --graph_max_batch_size 200 \
    --enable_prefill_cudagraph \
    --llm_prefill_att_backend flashinfer \
    --llm_decode_att_backend flashinfer \
    --sampling_backend sglang_kernel \
    --batch_size "$BS" \
    --input_len "$ILEN" \
    --output_len "$OLEN" \
    --cache_hit_len "$CACHE_HIT_LEN" \
    --torch_profile
}

case "$BENCH_MODE" in
  static_forward)
    run_static_forward_bench
    ;;
  cache_hit)
    run_cache_hit_forward_bench
    ;;
  both)
    run_static_forward_bench
    run_cache_hit_forward_bench
    ;;
  *)
    echo "Unsupported BENCH_MODE: $BENCH_MODE" >&2
    echo "Supported values: static_forward, cache_hit, both" >&2
    exit 1
    ;;
esac
