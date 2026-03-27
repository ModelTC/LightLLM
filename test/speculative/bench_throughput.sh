python /data/chenjunyi/project/lightllm/test/benchmark/service/benchmark_sharegpt.py \
    --port 8090 \
    --num-prompts 500 \
    --tokenizer /data/chenjunyi/models/qwen3-8b \
    --dataset /data/chenjunyi/project/lightllm/datasets/gsm8k.json \
    --history-turns 1