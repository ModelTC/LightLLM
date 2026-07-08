#!/bin/bash
# MTP Kernel Performance Benchmark Runner
# 用于比较 MTP kernel、FA3 和 MTP VSM 在不同负载下的性能

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/bench/benchmark_mtp_comparison.py"
RESULTS_DIR="${SCRIPT_DIR}/bench/results"

# 创建结果目录
mkdir -p "${RESULTS_DIR}"

# 默认参数
MODEL="all"
CUDA_GRAPH=""
DEVICE="cuda"

# 打印帮助
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL     测试模型类型 (llama3|deepseek_v2|load_dist|mtp|all)"
    echo "                        默认：all"
    echo "  -g, --cuda-graph      启用 CUDA Graph"
    echo "  -d, --device DEVICE   CUDA 设备 (默认：cuda)"
    echo "  --skip-decode         跳过 decode kernel 测试"
    echo "  --skip-mtp            跳过 MTP kernel 测试"
    echo "  -h, --help            显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  # 测试所有场景"
    echo "  $0"
    echo ""
    echo "  # 只测试 Llama-3 场景"
    echo "  $0 -m llama3"
    echo ""
    echo "  # 测试 DeepSeek-V2 场景，启用 CUDA Graph"
    echo "  $0 -m deepseek_v2 -g"
    echo ""
    echo "  # 测试不同负载分布的影响"
    echo "  $0 -m load_dist"
    echo ""
    echo "  # 只测试 MTP 不同 group_size 的影响"
    echo "  $0 -m mtp"
}

# 解析参数
SKIP_DECODE=""
SKIP_MTP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -g|--cuda-graph)
            CUDA_GRAPH="--cuda-graph"
            shift
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-decode)
            SKIP_DECODE="--skip-decode"
            shift
            ;;
        --skip-mtp)
            SKIP_MTP="--skip-mtp"
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "未知参数：$1"
            print_help
            exit 1
            ;;
    esac
done

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 输出文件
OUTPUT_FILE="${RESULTS_DIR}/benchmark_${MODEL}_${TIMESTAMP}.json"

# 打印配置信息
echo "========================================"
echo "MTP Kernel Performance Benchmark"
echo "========================================"
echo "Model: ${MODEL}"
echo "CUDA Graph: ${CUDA_GRAPH:-disabled}"
echo "Device: ${DEVICE}"
echo "Skip Decode: ${SKIP_DECODE:-no}"
echo "Skip MTP: ${SKIP_MTP:-no}"
echo "Output: ${OUTPUT_FILE}"
echo "========================================"
echo ""

# 运行 benchmark
python "${BENCHMARK_SCRIPT}" \
    --model "${MODEL}" \
    ${CUDA_GRAPH} \
    --device "${DEVICE}" \
    --output "${OUTPUT_FILE}" \
    ${SKIP_DECODE} \
    ${SKIP_MTP}

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Benchmark completed successfully!"
    echo "Results saved to: ${OUTPUT_FILE}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Benchmark failed!"
    echo "========================================"
    exit 1
fi
