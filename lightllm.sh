export BS=8
export ILEN=10000
export OLEN=500
export NCCL_HOST=127.0.0.1
export NCCL_PORT=28765

LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=18 \
python test/benchmark/static_inference/test_model.py \
  --model_dir ../Qwen3-235B-A22B \
  --tp 8 \
  --nccl_host "$NCCL_HOST" \
  --nccl_port "$NCCL_PORT" \
  --data_type bfloat16 \
  --graph_max_batch_size 200 \
  --llm_prefill_att_backend flashinfer \
  --llm_decode_att_backend flashinfer \
  --sampling_backend sglang_kernel \
  --batch_size "$BS" \
  --input_len "$ILEN" \
  --output_len "$OLEN" \
  --torch_profile
