#!/bin/bash

# =============================================================================
# MTP 自动化实验脚本
# 功能：自动运行不同 MTP 步长和模式的实验，收集 ttft、latency、throughput 等指标
# =============================================================================

set -e
export CUDA_VISIBLE_DEVICES=4,5,6,7

# =============================================================================
# 可配置参数 (可通过命令行参数覆盖)
# =============================================================================

# 基础目录配置
PROJECT_DIR="/data/nvme0/chenjunyi/project/lightllm"
BASE_DIR="/data/nvme0/chenjunyi/project/lightllm/test/speculative"
SCRIPTS_DIR="${BASE_DIR}/qwen3-8b"
HELPER_SCRIPT="${BASE_DIR}/helper.py"
BENCH_SCRIPT="${BASE_DIR}/bench_throughput.sh"
DATASET="/data/nvme0/chenjunyi/project/lightllm/datasets/gsm8k.json"
TOKENIZER="/mtc/models/qwen3-8b"

# 实验配置
MTP_STEPS=(4 8 12)
MODES=("dynamic_triton" "static_triton" "static_fa3")
SAMPLES=1000
CONCURRENCY=64

# 服务器配置
PORT=8088
TP=2
MAX_TOTAL_TOKEN_NUM=200000

# 日志和结果配置
RESULTS_DIR="${PROJECT_DIR}/experiment_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.csv"

# =============================================================================
# 解析命令行参数
# =============================================================================

usage() {
    echo "用法：$0 [选项]"
    echo ""
    echo "选项:"
    echo "  --scripts-dir DIR        启动脚本所在目录 (默认：${SCRIPTS_DIR})"
    echo "  --dataset PATH           数据集路径 (默认：${DATASET})"
    echo "  --tokenizer PATH         tokenizer 路径 (默认：${TOKENIZER})"
    echo "  --samples NUM            样本数量 (默认：${SAMPLES})"
    echo "  --concurrency NUM        并发数 (默认：${CONCURRENCY})"
    echo "  --mtp-steps STEPS        MTP 步长列表，逗号分隔 (默认：4,8,12)"
    echo "  --modes MODES            运行模式列表，逗号分隔 (默认：dynamic_triton,static_triton,static_fa3)"
    echo "  --port PORT              API 服务端口 (默认：${PORT})"
    echo "  --results-dir DIR        结果输出目录 (默认：${RESULTS_DIR})"
    echo "  --help                   显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --samples 500 --concurrency 32 --mtp-steps 4,8"
    echo "  $0 --scripts-dir ${BASE_DIR}/qwen3-32b --tokenizer /mtc/models/qwen3-32b"
    exit 1
}

# 解析命名参数
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
            echo "未知参数：$1"
            usage
            ;;
    esac
done

# =============================================================================
# 初始化
# =============================================================================

# 创建结果目录
mkdir -p "${RESULTS_DIR}"

# 写入 CSV 表头
echo "timestamp,mode,mtp_step,dataset,samples,concurrency,throughput,avg_latency,avg_ttft,avg_inter_token_latency,mtp_avg_token_per_step,mtp_avg_verify_tokens_per_step" > "${RESULTS_FILE}"

echo "=============================================="
echo "MTP 自动化实验开始"
echo "=============================================="
echo "启动脚本目录：${SCRIPTS_DIR}"
echo "数据集：${DATASET}"
echo "Tokenizer: ${TOKENIZER}"
echo "Samples: ${SAMPLES}"
echo "Concurrency: ${CONCURRENCY}"
echo "MTP Steps: ${MTP_STEPS[*]}"
echo "Modes: ${MODES[*]}"
echo "结果文件：${RESULTS_FILE}"
echo "=============================================="

# =============================================================================
# 辅助函数
# =============================================================================

# 等待服务器启动
wait_for_server() {
    local max_attempts=60
    local attempt=0
    echo "等待服务器启动..."
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "服务器已启动"
            return 0
        fi
        sleep 2
        ((attempt++))
    done
    echo "服务器启动超时"
    return 1
}

# 提取 benchmark 输出中的指标
extract_benchmark_metrics() {
    local log_file="$1"
    local throughput=""
    local avg_latency=""
    local avg_ttft=""
    local avg_inter_token_latency=""

    # 从日志中提取 benchmark 输出
    throughput=$(grep -oP 'Throughput: \K[\d.]+' "$log_file" | tail -1)
    avg_latency=$(grep -oP 'Average latency: \K[\d.]+' "$log_file" | tail -1)
    avg_ttft=$(grep -oP 'Average time to first token: \K[\d.]+' "$log_file" | tail -1)
    avg_inter_token_latency=$(grep -oP 'Average inter-token latency: \K[\d.]+' "$log_file" | tail -1)

    echo "${throughput:-NA},${avg_latency:-NA},${avg_ttft:-NA},${avg_inter_token_latency:-NA}"
}

# 使用 helper.py 提取 MTP 指标
extract_mtp_metrics() {
    local log_file="$1"
    local mtp_avg_token_per_step=""
    local mtp_avg_verify_tokens_per_step=""

    # 使用 helper.py 提取，从输出中解析
    local helper_output
    helper_output=$(python3 "${HELPER_SCRIPT}" "$log_file" 2>&1)

    # 提取 mtp_avg_token_per_step 的均值
    mtp_avg_token_per_step=$(echo "$helper_output" | grep -A5 "mtp_avg_token_per_step" | grep "均值 (Avg):" | grep -oP '[\d.]+')

    # 提取 mtp_avg_verify_tokens_per_step 的均值
    mtp_avg_verify_tokens_per_step=$(echo "$helper_output" | grep -A5 "mtp_avg_verify_tokens_per_step" | grep "均值 (Avg):" | grep -oP '[\d.]+')

    echo "${mtp_avg_token_per_step:-NA},${mtp_avg_verify_tokens_per_step:-NA}"
}

# 杀死 lightllm 进程
kill_lightllm() {
    echo "正在停止服务器..."
    # 杀死 api_server 进程
    pkill -f "lightllm.server.api_server" 2>/dev/null || true
    sleep 2
    # 确认进程已杀死
    if pgrep -f "lightllm.server.api_server" > /dev/null; then
        echo "强制终止进程..."
        pkill -9 -f "lightllm.server.api_server" 2>/dev/null || true
    fi
    echo "服务器已停止"
}

# =============================================================================
# 主实验循环
# =============================================================================

trap 'kill_lightllm' EXIT

for MODE in "${MODES[@]}"; do
    echo ""
    echo "=============================================="
    echo "运行模式：${MODE}"
    echo "=============================================="

    for MTP_STEP in "${MTP_STEPS[@]}"; do
        echo ""
        echo "--- MTP Step: ${MTP_STEP} ---"

        # 确定启动脚本
        if [[ "${MODE}" == "dynamic_triton" ]]; then
            STARTUP_SCRIPT="${SCRIPTS_DIR}/dynamic_triton.sh"
        elif [[ "${MODE}" == "static_triton" ]]; then
            STARTUP_SCRIPT="${SCRIPTS_DIR}/static_triton.sh"
        elif [[ "${MODE}" == "static_fa3" ]]; then
            STARTUP_SCRIPT="${SCRIPTS_DIR}/static_fa3.sh"
        else
            echo "未知模式：${MODE}"
            continue
        fi

        if [[ ! -f "${STARTUP_SCRIPT}" ]]; then
            echo "启动脚本不存在：${STARTUP_SCRIPT}"
            continue
        fi

        # 创建本次实验的日志文件
        LOG_FILE="${RESULTS_DIR}/log_${MODE}_step${MTP_STEP}_${TIMESTAMP}.txt"

        # 确保先停止之前的进程
        kill_lightllm

        # 启动服务器 (重定向日志)
        echo "启动服务器：${STARTUP_SCRIPT} --mtp-step ${MTP_STEP}"
        echo "日志文件：${LOG_FILE}"

        # 使用 bash 启动脚本，传递 mtp-step 参数，重定向所有输出到日志文件
        bash "${STARTUP_SCRIPT}" --mtp-step "${MTP_STEP}" > "${LOG_FILE}" 2>&1 &
        SERVER_PID=$!

        # 等待服务器启动
        if ! wait_for_server; then
            echo "服务器启动失败，跳过当前配置"
            continue
        fi

        # 额外等待，确保服务器完全就绪
        echo "等待服务器完全就绪..."
        sleep 5

        # 运行 benchmark
        echo "运行 benchmark (samples=${SAMPLES}, concurrency=${CONCURRENCY})"

        BENCH_LOG="${RESULTS_DIR}/bench_${MODE}_step${MTP_STEP}_${TIMESTAMP}.txt"
        bash "${BENCH_SCRIPT}" \
            --port "${PORT}" \
            --num-prompts "${SAMPLES}" \
            --tokenizer "${TOKENIZER}" \
            --dataset "${DATASET}" \
            --concurrency "${CONCURRENCY}" 2>&1 | tee "${BENCH_LOG}"

        # 合并日志，方便后续分析
        cat "${BENCH_LOG}" >> "${LOG_FILE}"

        # 提取 benchmark 指标
        echo "提取 benchmark 指标..."
        BENCH_METRICS=$(extract_benchmark_metrics "${LOG_FILE}")

        # 提取 MTP 指标
        echo "提取 MTP 指标..."
        MTP_METRICS=$(extract_mtp_metrics "${LOG_FILE}")

        # 写入结果
        RESULT_LINE="${TIMESTAMP},${MODE},${MTP_STEP},${DATASET},${SAMPLES},${CONCURRENCY},${BENCH_METRICS},${MTP_METRICS}"
        echo "${RESULT_LINE}" >> "${RESULTS_FILE}"

        echo "结果已记录：${RESULT_LINE}"

        # 停止服务器
        kill_lightllm

        echo "完成：Mode=${MODE}, MTP Step=${MTP_STEP}"
    done
done

echo ""
echo "=============================================="
echo "所有实验完成"
echo "=============================================="
echo "结果文件：${RESULTS_FILE}"
echo ""
echo "结果汇总:"
cat "${RESULTS_FILE}"
