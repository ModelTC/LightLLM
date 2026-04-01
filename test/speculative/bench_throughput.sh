#!/bin/bash
# 默认值
PORT=8088
NUM_PROMPTS=1000
TOKENIZER="/data/chenjunyi/models/qwen3-8b"
DATASET="/data/chenjunyi/project/lightllm/datasets/gsm8k.json"
HISTORY_TURNS=1
CONCURRENCY=128
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -n|--num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        -t|--tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -h|--history-turns)
            HISTORY_TURNS="$2"
            shift 2
            ;;
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done
# 执行 Python 脚本
python /data/chenjunyi/project/lightllm/test/benchmark/service/benchmark_sharegpt.py \
    --port "$PORT" \
    --num-prompts "$NUM_PROMPTS" \
    --tokenizer "$TOKENIZER" \
    --dataset "$DATASET" \
    --history-turns "$HISTORY_TURNS" \
    --concurrency "$CONCURRENCY"