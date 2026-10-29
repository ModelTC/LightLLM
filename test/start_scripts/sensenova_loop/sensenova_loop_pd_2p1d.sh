#!/usr/bin/env bash
# Two Prefill services and one Decode service; TP defaults to 2 / 4.
# Set MODEL_DIR and GPU_IDS as needed. DRY_RUN=1 prints the startup commands.
set -euo pipefail

LIGHTLLM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/mnt/afs_agents/yankuihao/model_ckpts/SenseNova-6.8-Loop-9B-Preview-mtp13200}"
PORT="${PORT:-8000}"
PREFILL_TP="${PREFILL_TP:-2}"
DECODE_TP="${DECODE_TP:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DRY_RUN="${DRY_RUN:-0}"

CHAT_TEMPLATE_ARGS=()
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
    CHAT_TEMPLATE_ARGS=(--chat_template "$(realpath -- "$CHAT_TEMPLATE")")
fi

export LOADWORKER="${LOADWORKER:-8}" LIGHTLLM_TRITON_AUTOTUNE_LEVEL="${LIGHTLLM_TRITON_AUTOTUNE_LEVEL:-1}"
export PYTHONUNBUFFERED=1
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} != 2 * PREFILL_TP + DECODE_TP )); then
    printf 'GPU_IDS must contain 2*PREFILL_TP+DECODE_TP GPU IDs.\n' >&2
    exit 2
fi
join_gpus() { local IFS=','; printf '%s' "$*"; }
P1_GPUS="$(join_gpus "${GPUS[@]:0:PREFILL_TP}")"
P2_GPUS="$(join_gpus "${GPUS[@]:PREFILL_TP:PREFILL_TP}")"
D_GPUS="$(join_gpus "${GPUS[@]:$((2 * PREFILL_TP)):DECODE_TP}")"

MODEL_ARGS=(
    --model_dir "$MODEL_DIR" --model_name "${MODEL_NAME:-sensenova_loop}"
    --host 0.0.0.0 --max_image_token_count 4096 "${CHAT_TEMPLATE_ARGS[@]}"
)
COMMON_ARGS=(
    "${MODEL_ARGS[@]}" --mtp_mode eagle_with_att --mtp_step 2
    --linear_att_cache_size 500
    --pd_trans_mode nixl --pd_kv_page_size 4096 --pd_master_ip 127.0.0.1 --pd_master_port "$PORT"
    --llm_prefill_att_backend fa3 triton --llm_decode_att_backend fa3 triton
)
P_ARGS=(
    "${COMMON_ARGS[@]}" --run_mode prefill --tp "$PREFILL_TP"
    --vit_att_backend triton
    --graph_max_batch_size 8 --running_max_req_size 8 --batch_max_tokens 8192
    --chunked_prefill_size 8192
    --linear_att_hash_page_size 2048 --linear_att_page_block_num 8
    --enable_cpu_cache --cpu_cache_storage_size "${CPU_CACHE_SIZE:-2}"
)
D_ARGS=(
    "${COMMON_ARGS[@]}" --run_mode decode --tp "$DECODE_TP"
    --graph_max_batch_size "${GRAPH_MAX_BATCH:-64}"
    --running_max_req_size "${MAX_REQUESTS:-64}" --batch_max_tokens 256
    --chunked_prefill_size 256 --port 28763 --nccl_port 29763
)
MASTER_ARGS=(
    "${MODEL_ARGS[@]}" --run_mode pd_master --pd_master_mode 2p1d --port "$PORT"
)

cd -- "$LIGHTLLM_ROOT"
PIDS=()
cleanup() {
    kill -TERM "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
    local role="$1" gpu_ids="$2"; shift 2
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_ids"
        printf '%q ' python -m lightllm.server.api_server "$@"
        printf '\n'
        return
    fi
    printf 'Starting %s on GPUs [%s]\n' "$role" "$gpu_ids"
    CUDA_VISIBLE_DEVICES="$gpu_ids" python -m lightllm.server.api_server "$@" &
    PIDS+=("$!")
}
start_server prefill-1 "$P1_GPUS" "${P_ARGS[@]}" --port 28761 --nccl_port 29761
start_server prefill-2 "$P2_GPUS" "${P_ARGS[@]}" --port 28762 --nccl_port 29762
start_server decode "$D_GPUS" "${D_ARGS[@]}"
start_server pd-master '' "${MASTER_ARGS[@]}"
if [[ "$DRY_RUN" == 1 ]]; then
    exit 0
fi

printf '2P1D is starting at http://127.0.0.1:%s\n' "$PORT"
wait -n "${PIDS[@]}"
exit 1
