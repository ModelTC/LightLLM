#!/bin/bash

# =============================================================================
# MTP Automated Experiment Script
# Function: Automatically run experiments with different MTP steps and modes, collect ttft, latency, throughput and other metrics
# =============================================================================

set -e
export CUDA_VISIBLE_DEVICES=1,2,3,4,6

# =============================================================================
# Configurable Parameters (can be overridden via command line arguments)
# =============================================================================

# Base directory configuration
PROJECT_DIR="/data/nvme0/chenjunyi/project/lightllm"
BASE_DIR="${PROJECT_DIR}/test/speculative"
SCRIPTS_DIR="${BASE_DIR}/qwen3-32b"
HELPER_SCRIPT="${BASE_DIR}/helper.py"
BENCH_SCRIPT="${BASE_DIR}/bench_throughput.sh"
DATASET="${PROJECT_DIR}/datasets/gsm8k.json"
TOKENIZER="/mtc/models/qwen3-32b"

# Experiment configuration
MTP_STEPS=(4 8 12)
MODES=("dynamic_triton" "static_triton" "static_fa3")
SAMPLES=1000
CONCURRENCY=64

# Server configuration
PORT=8088
TP=2
MAX_TOTAL_TOKEN_NUM=200000

# Log and result configuration
RESULTS_DIR="${PROJECT_DIR}/experiment_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Extract dataset name for directory (e.g., gsm8k.json -> gsm8k)
DATASET_NAME=$(basename "${DATASET}" .json)
# Create result directory: {results_dir}/{dataset}_{timestamp}/
EXPERIMENT_SUBDIR="${RESULTS_DIR}/${DATASET_NAME}_${TIMESTAMP}"
RESULTS_FILE="${EXPERIMENT_SUBDIR}/results.csv"

# =============================================================================
# Parse Command Line Arguments
# =============================================================================

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --scripts-dir DIR        Startup script directory (default: ${SCRIPTS_DIR})"
    echo "  --dataset PATH           Dataset path (default: ${DATASET})"
    echo "  --tokenizer PATH         Tokenizer path (default: ${TOKENIZER})"
    echo "  --samples NUM            Number of samples (default: ${SAMPLES})"
    echo "  --concurrency NUM        Concurrency (default: ${CONCURRENCY})"
    echo "  --mtp-steps STEPS        MTP step list, comma separated (default: 4,8,12)"
    echo "  --modes MODES            Run mode list, comma separated (default: dynamic_triton,static_triton,static_fa3)"
    echo "  --port PORT              API server port (default: ${PORT})"
    echo "  --results-dir DIR        Result output directory (default: ${RESULTS_DIR})"
    echo "  --help                   Show this help message"
    echo ""
    echo "Directory Structure:"
    echo "  Results will be saved to: ${RESULTS_DIR}/{dataset}_{timestamp}/"
    echo "  Example: ${RESULTS_DIR}/gsm8k_20260402_120000/"
    echo ""
    echo "Examples:"
    echo "  $0 --samples 500 --concurrency 32 --mtp-steps 4,8"
    echo "  $0 --scripts-dir ${BASE_DIR}/qwen3-32b --tokenizer /mtc/models/qwen3-32b"
    exit 1
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scripts-dir)
            SCRIPTS_DIR="$2"
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
        --mtp-steps)
            IFS=',' read -ra MTP_STEPS <<< "$2"
            shift 2
            ;;
        --modes)
            IFS=',' read -ra MODES <<< "$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
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

# =============================================================================
# Initialization
# =============================================================================

# Create results directory and experiment subdirectory
mkdir -p "${EXPERIMENT_SUBDIR}"

# Write CSV header
echo "timestamp,mode,mtp_step,dataset,samples,concurrency,throughput,avg_latency,avg_ttft,avg_inter_token_latency,mtp_avg_token_per_step,mtp_avg_verify_tokens_per_step" > "${RESULTS_FILE}"

echo "=============================================="
echo "MTP Automated Experiment Started"
echo "=============================================="
echo "Startup script directory: ${SCRIPTS_DIR}"
echo "Dataset: ${DATASET}"
echo "Tokenizer: ${TOKENIZER}"
echo "Samples: ${SAMPLES}"
echo "Concurrency: ${CONCURRENCY}"
echo "MTP Steps: ${MTP_STEPS[*]}"
echo "Modes: ${MODES[*]}"
echo "Results directory: ${EXPERIMENT_SUBDIR}"
echo "=============================================="

# =============================================================================
# Helper Functions
# =============================================================================

# Wait for server to start
wait_for_server() {
    local max_attempts=600
    local attempt=0
    echo "Waiting for server to start..."
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "Server started"
            return 0
        fi
        sleep 2
        ((attempt++))
    done
    echo "Server startup timeout"
    return 1
}

# Extract metrics from benchmark output
extract_benchmark_metrics() {
    local log_file="$1"
    local throughput=""
    local avg_latency=""
    local avg_ttft=""
    local avg_inter_token_latency=""

    # Extract benchmark output from log
    throughput=$(grep -oP 'Throughput: \K[\d.]+' "$log_file" | tail -1)
    avg_latency=$(grep -oP 'Average latency: \K[\d.]+' "$log_file" | tail -1)
    avg_ttft=$(grep -oP 'Average time to first token: \K[\d.]+' "$log_file" | tail -1)
    avg_inter_token_latency=$(grep -oP 'Average inter-token latency: \K[\d.]+' "$log_file" | tail -1)

    echo "${throughput:-NA},${avg_latency:-NA},${avg_ttft:-NA},${avg_inter_token_latency:-NA}"
}

# Extract MTP metrics using helper.py
extract_mtp_metrics() {
    local log_file="$1"
    local mtp_avg_token_per_step=""
    local mtp_avg_verify_tokens_per_step=""

    # Use helper.py to extract, parse from output
    local helper_output
    helper_output=$(python3 "${HELPER_SCRIPT}" "$log_file" 2>&1)

    # Extract mtp_avg_token_per_step average (format: metric_name.avg = value)
    mtp_avg_token_per_step=$(echo "$helper_output" | grep "mtp_avg_token_per_step.avg =" | awk -F'= ' '{print $2}')

    # Extract mtp_avg_verify_tokens_per_step average
    mtp_avg_verify_tokens_per_step=$(echo "$helper_output" | grep "mtp_avg_verify_tokens_per_step.avg =" | awk -F'= ' '{print $2}')

    echo "${mtp_avg_token_per_step:-NA},${mtp_avg_verify_tokens_per_step:-NA}"
}

# Kill lightllm process
kill_lightllm() {
    echo "Stopping server..."
    # Kill all lightllm processes forcefully - multiple rounds to ensure complete cleanup
    pkill -9 -f "lightllm" 2>/dev/null || true
    sleep 1
    pkill -9 -f "lightllm" 2>/dev/null || true
    sleep 1
    pkill -9 -f "lightllm" 2>/dev/null || true
    sleep 1
    echo "Server stopped"
}

# =============================================================================
# Main Experiment Loop
# =============================================================================

trap 'kill_lightllm' EXIT

for MODE in "${MODES[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running mode: ${MODE}"
    echo "=============================================="

    for MTP_STEP in "${MTP_STEPS[@]}"; do
        echo ""
        echo "--- MTP Step: ${MTP_STEP} ---"

        # Determine startup script
        if [[ "${MODE}" == "dynamic_triton" ]]; then
            STARTUP_SCRIPT="${SCRIPTS_DIR}/dynamic_triton.sh"
        elif [[ "${MODE}" == "static_triton" ]]; then
            STARTUP_SCRIPT="${SCRIPTS_DIR}/static_triton.sh"
        elif [[ "${MODE}" == "static_fa3" ]]; then
            STARTUP_SCRIPT="${SCRIPTS_DIR}/static_fa3.sh"
        else
            echo "Unknown mode: ${MODE}"
            continue
        fi

        if [[ ! -f "${STARTUP_SCRIPT}" ]]; then
            echo "Startup script not found: ${STARTUP_SCRIPT}"
            continue
        fi

        # Create log file for this experiment
        LOG_FILE="${EXPERIMENT_SUBDIR}/log_${MODE}_step${MTP_STEP}_${TIMESTAMP}.txt"

        # Ensure previous process is stopped first
        kill_lightllm

        # Start server (redirect log)
        echo "Starting server: ${STARTUP_SCRIPT} --mtp-step ${MTP_STEP}"
        echo "Log file: ${LOG_FILE}"

        # Use bash to start script, pass mtp-step parameter, redirect all output to log file
        bash "${STARTUP_SCRIPT}" --mtp-step "${MTP_STEP}" > "${LOG_FILE}" 2>&1 &
        SERVER_PID=$!

        # Wait for server to start
        if ! wait_for_server; then
            echo "Server failed to start, skipping current configuration"
            continue
        fi

        # Extra wait to ensure server is fully ready
        echo "Waiting for server to be ready..."
        sleep 5

        # Run benchmark
        echo "Running benchmark (samples=${SAMPLES}, concurrency=${CONCURRENCY})"

        BENCH_LOG="${EXPERIMENT_SUBDIR}/bench_${MODE}_step${MTP_STEP}_${TIMESTAMP}.txt"
        bash "${BENCH_SCRIPT}" \
            --port "${PORT}" \
            --num-prompts "${SAMPLES}" \
            --tokenizer "${TOKENIZER}" \
            --dataset "${DATASET}" \
            --concurrency "${CONCURRENCY}" 2>&1 | tee "${BENCH_LOG}"

        # Merge logs for easier analysis
        cat "${BENCH_LOG}" >> "${LOG_FILE}"

        # Extract benchmark metrics
        echo "Extracting benchmark metrics..."
        BENCH_METRICS=$(extract_benchmark_metrics "${LOG_FILE}")

        # Extract MTP metrics
        echo "Extracting MTP metrics..."
        MTP_METRICS=$(extract_mtp_metrics "${LOG_FILE}")

        # Write result
        RESULT_LINE="${TIMESTAMP},${MODE},${MTP_STEP},${DATASET},${SAMPLES},${CONCURRENCY},${BENCH_METRICS},${MTP_METRICS}"
        echo "${RESULT_LINE}" >> "${RESULTS_FILE}"

        echo "Result recorded: ${RESULT_LINE}"

        # Stop server
        kill_lightllm

        echo "Completed: Mode=${MODE}, MTP Step=${MTP_STEP}"
    done
done

echo ""
echo "=============================================="
echo "All Experiments Completed"
echo "=============================================="
echo "Results directory: ${EXPERIMENT_SUBDIR}"
echo ""
echo "Results Summary:"
cat "${RESULTS_FILE}"
echo ""
echo "Directory structure:"
ls -la "${EXPERIMENT_SUBDIR}"