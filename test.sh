python3 test/benchmark/service/benchmark_client.py \
  --url http://127.0.0.1:60011/generate \
  --tokenizer_path /mtc/bianzhuohang/models/Qwen/Qwen2.5-14B \
  --server_api lightllm \
  --dump_file result.json \
  --seed 42 \
  --dump_file pd_random.json