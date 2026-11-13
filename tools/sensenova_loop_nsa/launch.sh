#!/usr/bin/env bash
# Same step519 / BF16 / MTP3 / YaRN512K / 2P(TP2)+D(TP4) as supplied baseline.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/bin/python}
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
MODE=${MODE:-candidate}
"$PYTHON_BIN" "$REPO/tools/sensenova_loop_nsa/prepare_runtime.py" --verify-only >/dev/null
case "$MODE" in
    baseline)
        export LIGHTLLM_ROOT=${LIGHTLLM_ROOT:-"$REPO/../work/original_runtime"}
        export DSA_TOPK_IMPL=packed DSA_SCORE_QUERY_TILE=4
        export DSA_SCORE_MEMORY_MIB=${DSA_SCORE_MEMORY_MIB:-256}
        ;;
    candidate)
        export LIGHTLLM_ROOT=${LIGHTLLM_ROOT:-"$REPO"}
        export DSA_TOPK_IMPL=${DSA_TOPK_IMPL:-auto} DSA_SCORE_QUERY_TILE=${DSA_SCORE_QUERY_TILE:-16}
        export DSA_PREFILL_POLICY=${DSA_PREFILL_POLICY:-auto} DSA_PREFILL_DENSE_Q_HEADS_MAX=${DSA_PREFILL_DENSE_Q_HEADS_MAX:-8}
        export DSA_SCORE_KEY_TILE=${DSA_SCORE_KEY_TILE:-128}
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | grep -q "H200"; then
            export DSA_SCORE_MEMORY_MIB=${DSA_SCORE_MEMORY_MIB:-4096}
            export DSA_FA3_NUM_SPLITS=${DSA_FA3_NUM_SPLITS:-0}
        else
            export DSA_SCORE_MEMORY_MIB=${DSA_SCORE_MEMORY_MIB:-1024}
        fi
        if [[ -z "${DSA_VLLM_TOPK_LIBRARY:-}" && -f /mnt/afs_agents/hongjiawei/code/loop_transformers/infer_kuihao/dsa/work/vllm_C_0221.so ]]; then
            export DSA_VLLM_TOPK_LIBRARY=/mnt/afs_agents/hongjiawei/code/loop_transformers/infer_kuihao/dsa/work/vllm_C_0221.so
        fi
        ;;
    *) printf 'MODE must be baseline or candidate\n' >&2; exit 2 ;;
esac
export MODEL_DIR=${MODEL_DIR:-/nvme/dsa_models/ckpt-519-bf16}
export MTP_DIR=${MTP_DIR:-/nvme/dsa_models/mtp_yarn512k_v2_1106}
export CHAT_TEMPLATE="$MODEL_DIR/chat_template.jinja" PYTHON_BIN
export MODEL_NAME=${MODEL_NAME:-DSA_STEP519_BSH_NSA}
export QUANT_TYPE=none MTP_STEP=3 MAX_CONTEXT=524288 PREFILL_TP=2 DECODE_TP=4
export DSA_SPARSE_ATTN_IMPL=${DSA_SPARSE_ATTN_IMPL:-fa3}
export DSA_SHORT_DENSE_FA3=${DSA_SHORT_DENSE_FA3:-auto}
export GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7} PORT=${PORT:-18150}
if [[ "${DRY_RUN:-0}" != 1 && "${REQUIRE_FREE_GPUS:-1}" == 1 ]]; then
    memory_usage=$(nvidia-smi --id="$GPU_IDS" --query-gpu=memory.used --format=csv,noheader,nounits)
    while IFS= read -r used; do
        if (( used > 512 )); then
            printf 'Selected GPUs are occupied; use an idle test allocation.\n' >&2
            exit 2
        fi
    done <<< "$memory_usage"
fi
export P_PORT=${P_PORT:-24061} P2_PORT=${P2_PORT:-24062} D_PORT=${D_PORT:-24063}
export P_NCCL_PORT=${P_NCCL_PORT:-25061} P2_NCCL_PORT=${P2_NCCL_PORT:-25062} D_NCCL_PORT=${D_NCCL_PORT:-25063}
XGRAMMAR_DIR=/mnt/afs_agents/hongjiawei/code/loop_transformers/infer_kuihao/deps/xgrammar-0.2.8-cp310-cu13
export PYTHONPATH="$XGRAMMAR_DIR/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$XGRAMMAR_DIR/cuda-runtime/nvidia/nccl/lib:$XGRAMMAR_DIR/cuda-runtime/nvidia/nvshmem/lib"
export PYTHONDONTWRITEBYTECODE=1
test -f "$MODEL_DIR/_complete"
exec bash "${SERVE_SCRIPT:-/mnt/afs-toolcall/yankuihao/code/dsa_quick_perf_20261107/serve_2p1d.sh}"
