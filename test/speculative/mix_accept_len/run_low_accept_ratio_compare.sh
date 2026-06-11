#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SINGLE_RATIO_SCRIPT="${SCRIPT_DIR}/run_low_accept_single_ratio.sh"

GSM8K_DATASET="${LIGHTLLM_LOW_ACCEPT_GSM8K_DATASET}"
LOW_ACCEPT_DATASET="${LIGHTLLM_LOW_ACCEPT_DATASET}"
TOKENIZER="${LIGHTLLM_QWEN3_32B_TOKENIZER}"
OUTPUT_ROOT="${LIGHTLLM_PROJECT_ROOT}/experiment_results"
RATIOS="0.1,0.25,0.5,0.6,0.75,1.0"
MODES="static_fa3,dynamic_triton"
MTP_STEP=3
PORT=8088
SAMPLES=1000
CONCURRENCY=256
REPEATS=3
SEED=0
BENCH_RETRIES=3
BENCH_RETRY_WAIT=10

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --gsm8k PATH             GSM8K dataset path"
    echo "  --low-accept PATH        Low-accept dataset path"
    echo "  --tokenizer PATH         Tokenizer path"
    echo "  --ratios CSV             Ratios, e.g. 0.1,0.25,0.5,0.6,0.75,1.0"
    echo "  --modes CSV              Modes, default: static_fa3,dynamic_triton"
    echo "  --mtp-step N             MTP step passed to startup scripts"
    echo "  --port N                 Server port"
    echo "  --samples N              Benchmark sample count"
    echo "  --concurrency N          Benchmark concurrency"
    echo "  --repeats N              Repeats per test config"
    echo "  --seed N                 Seed for dataset mixing"
    echo "  --bench-retries N        Retry count per benchmark run on failure"
    echo "  --bench-retry-wait N     Seconds to wait before benchmark retry"
    echo "  --output-root DIR        Result root directory"
    echo "  --help                   Show help"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gsm8k)
            GSM8K_DATASET="$2"
            shift 2
            ;;
        --low-accept)
            LOW_ACCEPT_DATASET="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --ratios)
            RATIOS="$2"
            shift 2
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --mtp-step)
            MTP_STEP="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
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
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --bench-retries)
            BENCH_RETRIES="$2"
            shift 2
            ;;
        --bench-retry-wait)
            BENCH_RETRY_WAIT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
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

EXPERIMENT_DIR="${OUTPUT_ROOT}/low_accept_ratio_compare_${TIMESTAMP}"
mkdir -p "${EXPERIMENT_DIR}"

IFS=',' read -ra RATIO_LIST <<< "${RATIOS}"

echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Ratios: ${RATIOS}"
echo "Modes: ${MODES}"
echo "Samples: ${SAMPLES}, Concurrency: ${CONCURRENCY}, Repeats: ${REPEATS}"

for ratio in "${RATIO_LIST[@]}"; do
    bash "${SINGLE_RATIO_SCRIPT}" \
        --ratio "${ratio}" \
        --experiment-dir "${EXPERIMENT_DIR}" \
        --gsm8k "${GSM8K_DATASET}" \
        --low-accept "${LOW_ACCEPT_DATASET}" \
        --tokenizer "${TOKENIZER}" \
        --modes "${MODES}" \
        --mtp-step "${MTP_STEP}" \
        --port "${PORT}" \
        --samples "${SAMPLES}" \
        --concurrency "${CONCURRENCY}" \
        --repeats "${REPEATS}" \
        --seed "${SEED}" \
        --bench-retries "${BENCH_RETRIES}" \
        --bench-retry-wait "${BENCH_RETRY_WAIT}"
done

echo ""
echo "Final results:"
cat "${EXPERIMENT_DIR}/final_results.csv"

echo ""
echo "All run results:"
cat "${EXPERIMENT_DIR}/all_runs.csv"
