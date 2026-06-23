#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SCRIPTS_DIR="${LIGHTLLM_LOW_ACCEPT_QWEN32_DIR}"
BENCH_SCRIPT="${LIGHTLLM_LOW_ACCEPT_BENCH_SCRIPT}"
HELPER_SCRIPT="${LIGHTLLM_LOW_ACCEPT_HELPER_SCRIPT}"
MIX_SCRIPT="${LIGHTLLM_LOW_ACCEPT_MIX_SCRIPT}"

GSM8K_DATASET="${LIGHTLLM_LOW_ACCEPT_GSM8K_DATASET}"
LOW_ACCEPT_DATASET="${LIGHTLLM_LOW_ACCEPT_DATASET}"
TOKENIZER="${LIGHTLLM_QWEN3_32B_TOKENIZER}"
OUTPUT_ROOT="${LIGHTLLM_PROJECT_ROOT}/experiment_results"
EXPERIMENT_DIR=""
RATIO=""
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
    echo "Usage: $0 --ratio 0.1 [options]"
    echo ""
    echo "Options:"
    echo "  --ratio N                Required low-accept mixing ratio"
    echo "  --experiment-dir DIR     Existing or new experiment directory for resume"
    echo "  --gsm8k PATH             GSM8K dataset path"
    echo "  --low-accept PATH        Low-accept dataset path"
    echo "  --tokenizer PATH         Tokenizer path"
    echo "  --modes CSV              Modes, default: static_fa3,dynamic_triton; supports static_fa3,dynamic_triton,dynamic_fa3,no_mtp_fa3"
    echo "  --mtp-step N             MTP step"
    echo "  --port N                 Server port"
    echo "  --samples N              Benchmark sample count"
    echo "  --concurrency N          Benchmark concurrency"
    echo "  --repeats N              Repeats per mode"
    echo "  --seed N                 Seed for dataset mixing"
    echo "  --bench-retries N        Retry count per benchmark run"
    echo "  --bench-retry-wait N     Seconds to wait before benchmark retry"
    echo "  --output-root DIR        Parent dir when auto-creating experiment-dir"
    echo "  --help                   Show help"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --experiment-dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
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

if [[ -z "${RATIO}" ]]; then
    usage
fi

if [[ -z "${EXPERIMENT_DIR}" ]]; then
    EXPERIMENT_DIR="${OUTPUT_ROOT}/low_accept_ratio_resume_${TIMESTAMP}"
fi

MIX_DIR="${EXPERIMENT_DIR}/datasets"
RESULTS_FILE="${EXPERIMENT_DIR}/final_results.csv"
RUN_RESULTS_FILE="${EXPERIMENT_DIR}/all_runs.csv"
mkdir -p "${EXPERIMENT_DIR}" "${MIX_DIR}"

if [[ ! -f "${RUN_RESULTS_FILE}" ]]; then
    echo "timestamp,mode,ratio,mtp_step,dataset,run_index,samples,concurrency,throughput,avg_latency,avg_ttft,avg_inter_token_latency,mtp_avg_token_per_step,mtp_avg_verify_tokens_per_step,pure_decode_time_per_token_ms,pure_decode_throughput" > "${RUN_RESULTS_FILE}"
fi
if [[ ! -f "${RESULTS_FILE}" ]]; then
    echo "timestamp,mode,ratio,mtp_step,dataset,selected_run,samples,concurrency,throughput,avg_latency,avg_ttft,avg_inter_token_latency,mtp_avg_token_per_step,mtp_avg_verify_tokens_per_step,pure_decode_time_per_token_ms,pure_decode_throughput" > "${RESULTS_FILE}"
fi

CURRENT_SERVER_PID=""

wait_for_server() {
    local server_log="$1"
    local server_pid="$2"
    local max_attempts=600
    local attempt=0
    while [[ $attempt -lt $max_attempts ]]; do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "Server process exited before becoming ready"
            return 1
        fi
        if [[ -f "${server_log}" ]] && grep -q "server start up ok" "${server_log}"; then
            if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
                return 0
            fi
        fi
        sleep 2
        ((attempt++))
    done
    return 1
}

cleanup_current_server() {
    local server_pid="${CURRENT_SERVER_PID:-}"
    if [[ -z "${server_pid}" ]]; then
        return 0
    fi

    # The server is launched with setsid, so the direct child pid is also the
    # process-group leader. Kill the whole group for this round only.
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${server_pid}" 2>/dev/null || true

    # Reap the direct child to avoid leaving a zombie owned by this shell.
    wait "${server_pid}" 2>/dev/null || true
    CURRENT_SERVER_PID=""
}

kill_lightllm() {
    cleanup_current_server

    pkill -TERM -f "lightllm::.*::api_server" 2>/dev/null || true
    sleep 1
    pkill -TERM -f "lightllm::.*::metric_manager" 2>/dev/null || true
    sleep 1
    pkill -TERM -f "lightllm::.*::detokenization_server" 2>/dev/null || true
    sleep 1
    pkill -TERM -f "hypercorn" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "lightllm::.*::api_server" 2>/dev/null || true
    pkill -KILL -f "lightllm::.*::metric_manager" 2>/dev/null || true
    pkill -KILL -f "lightllm::.*::detokenization_server" 2>/dev/null || true
    pkill -KILL -f "hypercorn" 2>/dev/null || true
    sleep 1
}

extract_benchmark_metrics() {
    local log_file="$1"
    local throughput avg_latency avg_ttft avg_inter_token_latency
    throughput=$(grep -oP 'Throughput: \K[\d.]+' "$log_file" | tail -1)
    avg_latency=$(grep -oP 'Average latency: \K[\d.]+' "$log_file" | tail -1)
    avg_ttft=$(grep -oP 'Average time to first token: \K[\d.]+' "$log_file" | tail -1)
    avg_inter_token_latency=$(grep -oP 'Average inter-token latency: \K[\d.]+' "$log_file" | tail -1)
    echo "${throughput:-NA},${avg_latency:-NA},${avg_ttft:-NA},${avg_inter_token_latency:-NA}"
}

extract_mtp_metrics() {
    local log_file="$1"
    local helper_output mtp_avg_token_per_step mtp_avg_verify_tokens_per_step pure_decode_time_per_token_ms pure_decode_throughput
    helper_output=$("${LIGHTLLM_SPEC_PYTHON}" "${HELPER_SCRIPT}" "$log_file" 2>&1)
    mtp_avg_token_per_step=$(echo "$helper_output" | grep "mtp_avg_token_per_step.avg =" | awk -F'= ' '{print $2}')
    mtp_avg_verify_tokens_per_step=$(echo "$helper_output" | grep "mtp_avg_verify_tokens_per_step.avg =" | awk -F'= ' '{print $2}')
    pure_decode_time_per_token_ms=$(echo "$helper_output" | grep "pure_decode_time_per_token_ms.global =" | awk -F'= ' '{print $2}')
    pure_decode_throughput=$(echo "$helper_output" | grep "pure_decode_throughput.global =" | awk -F'= ' '{print $2}')
    echo "${mtp_avg_token_per_step:-NA},${mtp_avg_verify_tokens_per_step:-NA},${pure_decode_time_per_token_ms:-NA},${pure_decode_throughput:-NA}"
}

run_benchmark_with_retry() {
    local bench_log="$1"
    local dataset_path="$2"
    local mode="$3"
    local attempt=1
    local rc=0

    while [[ "${attempt}" -le "${BENCH_RETRIES}" ]]; do
        echo "Running benchmark: mode=${mode}, ratio=${RATIO}, attempt=${attempt}"
        set +e
        bash "${BENCH_SCRIPT}" \
            --port "${PORT}" \
            --num-prompts "${SAMPLES}" \
            --tokenizer "${TOKENIZER}" \
            --dataset "${dataset_path}" \
            --concurrency "${CONCURRENCY}" > "${bench_log}" 2>&1
        rc=$?
        set -e
        if [[ "${rc}" -eq 0 ]]; then
            return 0
        fi
        echo "Benchmark failed: mode=${mode}, ratio=${RATIO}, attempt=${attempt}, rc=${rc}"
        if [[ "${attempt}" -lt "${BENCH_RETRIES}" ]]; then
            sleep "${BENCH_RETRY_WAIT}"
        fi
        ((attempt++))
    done
    return "${rc}"
}

build_mix_dataset() {
    local ratio_tag="${RATIO//./p}"
    local output_path="${MIX_DIR}/gsm8k_low_accept_ratio_${ratio_tag}.json"
    "${LIGHTLLM_SPEC_PYTHON}" "${MIX_SCRIPT}" \
        --gsm8k "${GSM8K_DATASET}" \
        --low-accept "${LOW_ACCEPT_DATASET}" \
        --ratio "${RATIO}" \
        --target-size "${SAMPLES}" \
        --seed "${SEED}" \
        --output "${output_path}" > /dev/null
    echo "${output_path}"
}

write_server_delta_log() {
    local src_log="$1"
    local dst_log="$2"
    local start_size="$3"
    tail -c "+$((start_size + 1))" "${src_log}" > "${dst_log}"
}

trap 'kill_lightllm' EXIT

DATASET_PATH=$(build_mix_dataset)
DATASET_NAME=$(basename "${DATASET_PATH}" .json)
IFS=',' read -ra MODE_LIST <<< "${MODES}"

echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Ratio: ${RATIO}"
echo "Modes: ${MODES}"
echo "Samples: ${SAMPLES}, Concurrency: ${CONCURRENCY}, Repeats: ${REPEATS}"

for mode in "${MODE_LIST[@]}"; do
    if [[ "${mode}" == "dynamic_triton" ]]; then
        STARTUP_SCRIPT="${SCRIPTS_DIR}/dynamic_triton.sh"
    elif [[ "${mode}" == "dynamic_fa3" ]]; then
        STARTUP_SCRIPT="${SCRIPTS_DIR}/dynamic_fa3.sh"
    elif [[ "${mode}" == "no_mtp_fa3" ]]; then
        STARTUP_SCRIPT="${SCRIPTS_DIR}/no_mtp_fa3.sh"
    elif [[ "${mode}" == "static_fa3" ]]; then
        STARTUP_SCRIPT="${SCRIPTS_DIR}/static_fa3.sh"
    else
        echo "Unsupported mode: ${mode}"
        exit 1
    fi

    CONFIG_DIR="${EXPERIMENT_DIR}/${mode}_${DATASET_NAME}"
    mkdir -p "${CONFIG_DIR}"
    SERVER_LOG="${CONFIG_DIR}/server.log"

    echo ""
    echo "=============================================="
    echo "Running mode=${mode}, ratio=${RATIO}, dataset=${DATASET_NAME}"
    echo "=============================================="

    kill_lightllm
    STARTUP_ARGS=(--port "${PORT}")
    if [[ "${mode}" != "no_mtp_fa3" ]]; then
        STARTUP_ARGS+=(--mtp-step "${MTP_STEP}")
    fi

    setsid bash "${STARTUP_SCRIPT}" "${STARTUP_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    CURRENT_SERVER_PID="${SERVER_PID}"

    if ! wait_for_server "${SERVER_LOG}" "${SERVER_PID}"; then
        echo "Server failed to start for mode=${mode}, ratio=${RATIO}"
        kill_lightllm
        continue
    fi
    sleep 10

    for run_index in $(seq 1 "${REPEATS}"); do
        BENCH_LOG="${CONFIG_DIR}/bench_run${run_index}.log"
        SERVER_DELTA_LOG="${CONFIG_DIR}/server_run${run_index}.log"
        server_size_before=$(wc -c < "${SERVER_LOG}")

        if ! run_benchmark_with_retry "${BENCH_LOG}" "${DATASET_PATH}" "${mode}"; then
            write_server_delta_log "${SERVER_LOG}" "${SERVER_DELTA_LOG}" "${server_size_before}"
            RUN_LINE="${TIMESTAMP},${mode},${RATIO},${MTP_STEP},${DATASET_PATH},${run_index},${SAMPLES},${CONCURRENCY},FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED"
            echo "${RUN_LINE}" >> "${RUN_RESULTS_FILE}"
            echo "Benchmark permanently failed for mode=${mode}, ratio=${RATIO}, run=${run_index}"
            break
        fi

        write_server_delta_log "${SERVER_LOG}" "${SERVER_DELTA_LOG}" "${server_size_before}"

        BENCH_METRICS=$(extract_benchmark_metrics "${BENCH_LOG}")
        MTP_METRICS=$(extract_mtp_metrics "${SERVER_DELTA_LOG}")
        RUN_LINE="${TIMESTAMP},${mode},${RATIO},${MTP_STEP},${DATASET_PATH},${run_index},${SAMPLES},${CONCURRENCY},${BENCH_METRICS},${MTP_METRICS}"
        echo "${RUN_LINE}" >> "${RUN_RESULTS_FILE}"

        if [[ "${run_index}" == "${REPEATS}" ]]; then
            FINAL_LINE="${TIMESTAMP},${mode},${RATIO},${MTP_STEP},${DATASET_PATH},${run_index},${SAMPLES},${CONCURRENCY},${BENCH_METRICS},${MTP_METRICS}"
            echo "${FINAL_LINE}" >> "${RESULTS_FILE}"
            echo "Selected final metrics: ${FINAL_LINE}"
        fi
    done

    kill_lightllm
done

echo ""
echo "Final results:"
cat "${RESULTS_FILE}"
