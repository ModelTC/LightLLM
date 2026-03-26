python /data/nvme0/chenjunyi/project/lightllm/test/benchmark/service/benchmark_sharegpt.py \
    --port 8088 \
    --num-prompts 10 \
    --tokenizer /mtc/models/qwen3-8b \
    --dataset /data/nvme0/chenjunyi/project/lightllm/datasets/gsm8k.json \
    --history-turns 1