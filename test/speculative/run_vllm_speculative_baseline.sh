#!/bin/bash

# =============================================================================
# vLLM Speculative Decoding Baseline Experiment Script
# Function: Run vLLM default draft-model speculative decoding baseline for
#           different mtp steps (mapped to num_speculative_tokens), and collect
#           throughput/latency metrics with the same benchmark script.
# =============================================================================

set -euo pipefail

# Keep default GPU visibility aligned with existing LightLLM experiment scripts.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,6}"
# Reduce allocator fragmentation risk during model warmup.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# =============================================================================
# Configurable Parameters
# =============================================================================
PROJECT_DIR="/data/nvme0/chenjunyi/project/lightllm"
BENCH_PY_SCRIPT="${PROJECT_DIR}/test/benchmark/service/benchmark_sharegpt.py"
DATASET="${PROJECT_DIR}/datasets/gsm8k.json"

# Keep defaults close to existing LightLLM qwen3-32b setup.
MODEL_DIR="/mtc/models/qwen3-32b"
DRAFT_MODEL_DIR="/mtc/models/qwen3-32b-eagle3"
TOKENIZER="/mtc/models/qwen3-32b"

SAMPLES=1000
CONCURRENCY=256
PORT=8088
TP=4
MAX_MODEL_LEN=16384
MAX_NUM_BATCHED_TOKENS=200000
MAX_NUM_SEQS=256
GPU_MEMORY_UTILIZATION=0.6
MAX_CUDAGRAPH_CAPTURE_SIZE=256
ATTENTION_BACKEND="FLASH_ATTN"
DISABLE_CUSTOM_ALL_REDUCE=1
MTP_STEPS=(5)

RESULTS_DIR="${PROJECT_DIR}/experiment_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATASET_NAME=$(basename "${DATASET}" .json)
EXPERIMENT_SUBDIR="${RESULTS_DIR}/${DATASET_NAME}_${TIMESTAMP}_vllm_spec_default"
RESULTS_FILE="${EXPERIMENT_SUBDIR}/results.csv"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model-dir PATH              Main model path (default: ${MODEL_DIR})"
    echo "  --draft-model-dir PATH        Draft model path (default: ${DRAFT_MODEL_DIR})"
    echo "  --dataset PATH                Dataset path (default: ${DATASET})"
    echo "  --tokenizer PATH              Tokenizer path (default: ${TOKENIZER})"
    echo "  --samples NUM                 Number of prompts (default: ${SAMPLES})"
    echo "  --concurrency NUM             Concurrency (default: ${CONCURRENCY})"
    echo "  --port PORT                   Service port (default: ${PORT})"
    echo "  --tp NUM                      Tensor parallel size (default: ${TP})"
    echo "  --mtp-steps LIST              Comma-separated mtp steps (default: 5)"
    echo "  --num-speculative-tokens NUM  Backward-compatible alias, equals one mtp step"
    echo "  --max-model-len NUM           vLLM max model len (default: ${MAX_MODEL_LEN})"
    echo "  --max-num-batched-tokens NUM  vLLM max batched tokens (default: ${MAX_NUM_BATCHED_TOKENS})"
    echo "  --max-num-seqs NUM            vLLM max number of concurrent seqs (default: ${MAX_NUM_SEQS})"
    echo "  --max-cudagraph-capture-size NUM  vLLM max cudagraph capture size (default: ${MAX_CUDAGRAPH_CAPTURE_SIZE})"
    echo "  --gpu-memory-utilization F    GPU memory utilization (default: ${GPU_MEMORY_UTILIZATION})"
    echo "  --attention-backend NAME      vLLM attention backend (default: ${ATTENTION_BACKEND})"
    echo "  --enable-custom-all-reduce    Enable custom all-reduce (default: disabled)"
    echo "  --results-dir DIR             Results base dir (default: ${RESULTS_DIR})"
    echo "  --help                        Show this help"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --draft-model-dir)
            DRAFT_MODEL_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --mtp-steps)
            IFS=',' read -ra MTP_STEPS <<< "$2"
            shift 2
            ;;
        --num-speculative-tokens)
            MTP_STEPS=("$2")
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-num-batched-tokens)
            MAX_NUM_BATCHED_TOKENS="$2"
            shift 2
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --max-cudagraph-capture-size)
            MAX_CUDAGRAPH_CAPTURE_SIZE="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --attention-backend)
            ATTENTION_BACKEND="$2"
            shift 2
            ;;
        --enable-custom-all-reduce)
            DISABLE_CUSTOM_ALL_REDUCE=0
            shift 1
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Recompute result paths in case dataset/results-dir was overridden.
DATASET_NAME=$(basename "${DATASET}" .json)
EXPERIMENT_SUBDIR="${RESULTS_DIR}/${DATASET_NAME}_${TIMESTAMP}_vllm_spec_default"
RESULTS_FILE="${EXPERIMENT_SUBDIR}/results.csv"

mkdir -p "${EXPERIMENT_SUBDIR}"

echo "timestamp,engine,mode,mtp_step,dataset,samples,concurrency,throughput,avg_latency,avg_ttft,avg_inter_token_latency" > "${RESULTS_FILE}"

wait_for_server() {
    local max_attempts=600
    local attempt=0
    echo "Waiting for vLLM server to start..."
    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "vLLM server started"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "vLLM server startup timeout"
    return 1
}

extract_benchmark_metrics() {
    local log_file="$1"
    local throughput=""
    local avg_latency=""
    local avg_ttft=""
    local avg_inter_token_latency=""

    throughput=$(grep -oP 'Throughput: \K[\d.]+' "$log_file" | tail -1)
    avg_latency=$(grep -oP 'Average latency: \K[\d.]+' "$log_file" | tail -1)
    avg_ttft=$(grep -oP 'Average time to first token: \K[\d.]+' "$log_file" | tail -1)
    avg_inter_token_latency=$(grep -oP 'Average inter-token latency: \K[\d.]+' "$log_file" | tail -1)

    echo "${throughput:-NA},${avg_latency:-NA},${avg_ttft:-NA},${avg_inter_token_latency:-NA}"
}

kill_vllm() {
    echo "Stopping vLLM server..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 1
    echo "vLLM server stopped"
}

trap 'kill_vllm' EXIT

echo "=============================================="
echo "vLLM Speculative Baseline Started"
echo "=============================================="
echo "Model: ${MODEL_DIR}"
echo "Draft model: ${DRAFT_MODEL_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Dataset: ${DATASET}"
echo "Samples: ${SAMPLES}"
echo "Concurrency: ${CONCURRENCY}"
echo "TP: ${TP}"
echo "Port: ${PORT}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "Max batched tokens: ${MAX_NUM_BATCHED_TOKENS}"
echo "Max num seqs: ${MAX_NUM_SEQS}"
echo "Max cudagraph capture size: ${MAX_CUDAGRAPH_CAPTURE_SIZE}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Attention backend: ${ATTENTION_BACKEND}"
echo "Disable custom all reduce: ${DISABLE_CUSTOM_ALL_REDUCE}"
echo "MTP steps: ${MTP_STEPS[*]}"
echo "Results directory: ${EXPERIMENT_SUBDIR}"
echo "=============================================="

for MTP_STEP in "${MTP_STEPS[@]}"; do
    echo ""
    echo "--- Running mtp step: ${MTP_STEP} ---"

    LOG_FILE="${EXPERIMENT_SUBDIR}/log_vllm_spec_default_step${MTP_STEP}_${TIMESTAMP}.txt"
    BENCH_LOG="${EXPERIMENT_SUBDIR}/bench_vllm_spec_default_step${MTP_STEP}_${TIMESTAMP}.txt"

    SPECULATIVE_CONFIG=$(printf '{"model": "%s", "num_speculative_tokens": %s, "method": "draft_model"}' \
        "${DRAFT_MODEL_DIR}" "${MTP_STEP}")
    CUSTOM_ALL_REDUCE_FLAG=""
    if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
        CUSTOM_ALL_REDUCE_FLAG="--disable-custom-all-reduce"
    fi

    kill_vllm

    echo "Starting vLLM server with speculative_config=${SPECULATIVE_CONFIG}"
    (
        vllm serve "${MODEL_DIR}" \
            --host 0.0.0.0 \
            --port "${PORT}" \
            --served-model-name DeepSeek-R1 \
            -tp "${TP}" \
            --max_model_len "${MAX_MODEL_LEN}" \
            --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}" \
            --max_num_seqs "${MAX_NUM_SEQS}" \
            --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
            --attention-backend "${ATTENTION_BACKEND}" \
            ${CUSTOM_ALL_REDUCE_FLAG} \
            --speculative_config "${SPECULATIVE_CONFIG}"
    ) > "${LOG_FILE}" 2>&1 &

    SERVER_PID=$!
    echo "vLLM PID: ${SERVER_PID}"

    if ! wait_for_server; then
        echo "vLLM server failed to start for mtp step ${MTP_STEP}. Check log: ${LOG_FILE}"
        RESULT_LINE="${TIMESTAMP},vllm,speculative_draft_model_default,${MTP_STEP},${DATASET},${SAMPLES},${CONCURRENCY},NA,NA,NA,NA"
        echo "${RESULT_LINE}" >> "${RESULTS_FILE}"
        continue
    fi

    sleep 5

    echo "Running benchmark with benchmark_sharegpt.py (OpenAI API mode)..."
    python "${BENCH_PY_SCRIPT}" \
        --use_openai_api \
        --port "${PORT}" \
        --num-prompts "${SAMPLES}" \
        --tokenizer "${TOKENIZER}" \
        --dataset "${DATASET}" \
        --history-turns 1 \
        --concurrency "${CONCURRENCY}" 2>&1 | tee "${BENCH_LOG}"

    cat "${BENCH_LOG}" >> "${LOG_FILE}"

    BENCH_METRICS=$(extract_benchmark_metrics "${LOG_FILE}")
    RESULT_LINE="${TIMESTAMP},vllm,speculative_draft_model_default,${MTP_STEP},${DATASET},${SAMPLES},${CONCURRENCY},${BENCH_METRICS}"
    echo "${RESULT_LINE}" >> "${RESULTS_FILE}"

    echo "Completed mtp step ${MTP_STEP}: ${RESULT_LINE}"
done

echo ""
echo "=============================================="
echo "All Experiments Completed"
echo "=============================================="
echo "Results file: ${RESULTS_FILE}"
cat "${RESULTS_FILE}"
