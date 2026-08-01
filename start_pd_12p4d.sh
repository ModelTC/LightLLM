#!/usr/bin/env bash
set -euo pipefail

NNODES="${SENSECORE_PYTORCH_NNODES:?SENSECORE_PYTORCH_NNODES is required}"
NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:?SENSECORE_PYTORCH_NODE_RANK is required}"
DEVICE_COUNT="${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}"

if [[ "$NNODES" -ne 2 ]]; then
    echo "12P+4D requires exactly 2 nodes, got $NNODES." >&2
    exit 2
fi
if [[ "$NODE_RANK" -ne 0 && "$NODE_RANK" -ne 1 ]]; then
    echo "Unsupported node rank: $NODE_RANK." >&2
    exit 2
fi
if [[ "$DEVICE_COUNT" -ne 8 ]]; then
    echo "12P+4D requires exactly 8 GPUs per node, got $DEVICE_COUNT." >&2
    exit 2
fi

detect_node_ip() {
    hostname -I | tr ' ' '\n' | awk '/^[0-9]+\./ && $0 !~ /^127\./ { print; exit }'
}

MODEL_DIR="${MODEL_DIR:-/mtc/models/Qwen3.5-27B}"
LOG_DIR="${LOG_DIR:-/logs}"
PD_MASTER_PORT="${PD_MASTER_PORT:-8088}"
CPU_CACHE_GB="${CPU_CACHE_GB:-32}"
PD_MASTER_READY_TIMEOUT="${PD_MASTER_READY_TIMEOUT:-300}"
PD_TOPOLOGY_READY_TIMEOUT="${PD_TOPOLOGY_READY_TIMEOUT:-1200}"
LOCAL_SERVICE_READY_TIMEOUT="${LOCAL_SERVICE_READY_TIMEOUT:-900}"
PD_WARMUP_REQUESTS="${PD_WARMUP_REQUESTS:-96}"
P0_RUNNING_MAX_REQ_SIZE="${P0_RUNNING_MAX_REQ_SIZE:-256}"
P1_RUNNING_MAX_REQ_SIZE="${P1_RUNNING_MAX_REQ_SIZE:-128}"
D_RUNNING_MAX_REQ_SIZE="${D_RUNNING_MAX_REQ_SIZE:-64}"
P_MAX_TOTAL_TOKEN_NUM="${P_MAX_TOTAL_TOKEN_NUM:-1200000}"
D_MAX_TOTAL_TOKEN_NUM="${D_MAX_TOTAL_TOKEN_NUM:-2700000}"
P_LINEAR_ATT_CACHE_SIZE="${P_LINEAR_ATT_CACHE_SIZE:-16}"

if [[ "$NODE_RANK" -eq 0 ]]; then
    NODE_IP="${NODE_IP:-$MASTER_ADDR}"
else
    NODE_IP="${NODE_IP:-$(detect_node_ip)}"
fi

if [[ -z "$NODE_IP" ]]; then
    echo "Unable to discover local IP; set NODE_IP explicitly." >&2
    exit 2
fi
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Model directory does not exist: $MODEL_DIR" >&2
    exit 2
fi

export LOADWORKER="${LOADWORKER:-18}"
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL="${LIGHTLLM_TRITON_AUTOTUNE_LEVEL:-0}"
export LIGHTLLM_FP8_GEMM="${LIGHTLLM_FP8_GEMM:-sgl}"
export DISABLE_CHECK_MAX_LEN_INFER="${DISABLE_CHECK_MAX_LEN_INFER:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
    if (( ${#PIDS[@]} > 0 )); then
        kill "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_pd_master() {
    local ready=0
    for ((i = 0; i < PD_MASTER_READY_TIMEOUT; i++)); do
        if (echo >"/dev/tcp/${MASTER_ADDR}/${PD_MASTER_PORT}") >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "Cannot reach PD master at ${MASTER_ADDR}:${PD_MASTER_PORT}." >&2
        return 1
    fi
}

wait_for_local_registration() {
    local pid="$1"
    local log_file="$2"
    local mode="$3"
    local label="$4"

    for ((i = 0; i < LOCAL_SERVICE_READY_TIMEOUT; i++)); do
        if grep -q "Sent registration JSON.*mode.*${mode}" "$log_file" 2>/dev/null; then
            echo "$label registered with the PD master."
            return
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$label exited before registration. Check $log_file." >&2
            return 1
        fi
        sleep 1
    done

    echo "$label did not register after ${LOCAL_SERVICE_READY_TIMEOUT}s." >&2
    return 1
}

wait_for_pd_topology() {
    local waited=0
    local p_count=0
    local d_count=0
    echo "Waiting for two P services and one D service"
    while true; do
        p_count=$(grep -c "mode: prefill.*registed" "$LOG_DIR/pd-master.log" 2>/dev/null || true)
        d_count=$(grep -c "mode: decode.*registed" "$LOG_DIR/pd-master.log" 2>/dev/null || true)
        if (( p_count >= 2 && d_count >= 1 )); then
            return
        fi
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "A rank-0 service exited before topology readiness." >&2
                return 1
            fi
        done
        if (( waited >= PD_TOPOLOGY_READY_TIMEOUT )); then
            echo "Topology not ready after ${PD_TOPOLOGY_READY_TIMEOUT}s (P=$p_count D=$d_count)." >&2
            return 1
        fi
        sleep 1
        ((waited += 1))
    done
}

warm_pd_routes() {
    if [[ "$PD_WARMUP_REQUESTS" -eq 0 ]]; then
        echo "PD route warmup disabled."
        return
    fi

    echo "Warming ${PD_WARMUP_REQUESTS} unique P-to-D routes"
    PD_WARMUP_URL="http://${MASTER_ADDR}:${PD_MASTER_PORT}/v1/chat/completions" \
    PD_WARMUP_REQUESTS="$PD_WARMUP_REQUESTS" \
    python - <<'PY'
import concurrent.futures
import json
import os
import urllib.request

url = os.environ["PD_WARMUP_URL"]
count = int(os.environ["PD_WARMUP_REQUESTS"])


def send_one(index):
    body = json.dumps(
        {
            "model": "qwen35-27b",
            "messages": [
                {
                    "role": "user",
                    "content": f"route-{index:05d}-" + (f"unique-{index}-warmup " * 512),
                }
            ],
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
        }
    ).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise RuntimeError(f"warmup returned HTTP {response.status}")
        response.read()


with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
    list(pool.map(send_one, range(count)))
PY
    echo "PD topology is ready and unique routes are warm."
}

start_pd_master() {
    echo "Starting PD master at ${NODE_IP}:${PD_MASTER_PORT}"
    python -m lightllm.server.api_server \
        --model_dir "$MODEL_DIR" \
        --model_name qwen35-27b \
        --run_mode pd_master \
        --host "$NODE_IP" \
        --port "$PD_MASTER_PORT" \
        --max_req_total_len 262144 \
        --select_p_d_node_strategy cache_aware \
        >"$LOG_DIR/pd-master.log" 2>&1 &
    PIDS+=("$!")
}

start_p_service() {
    local gpu_ids="$1"
    local tp="$2"
    local dp="$3"
    local port="$4"
    local nccl_port="$5"
    local running_max_req_size="$6"
    local log_file="$7"
    local label="$8"

    echo "Starting $label at ${NODE_IP}:${port} GPUs=${gpu_ids} TP=${tp} DP=${dp}"
    /usr/bin/env -u DISABLE_CHECK_MAX_LEN_INFER \
    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    python -m lightllm.server.api_server \
        --model_dir "$MODEL_DIR" \
        --model_name qwen35-27b \
        --run_mode prefill \
        --host "$NODE_IP" \
        --port "$port" \
        --nccl_port "$nccl_port" \
        --pd_master_ip "$MASTER_ADDR" \
        --pd_master_port "$PD_MASTER_PORT" \
        --nnodes 1 \
        --node_rank 0 \
        --tp "$tp" \
        --dp "$dp" \
        --dp_balancer bs_balancer \
        --mem_fraction 0.80 \
        --max_total_token_num "$P_MAX_TOTAL_TOKEN_NUM" \
        --max_req_total_len 262144 \
        --running_max_req_size "$running_max_req_size" \
        --batch_max_tokens 4096 \
        --chunked_prefill_size 4096 \
        --graph_max_batch_size 64 \
        --disable_cudagraph \
        --disable_audio \
        --visual_gpu_ids 0 \
        --visual_infer_batch_size 1 \
        --visual_send_batch_size 1 \
        --max_image_pixels 4194304 \
        --max_image_token_count 4096 \
        --quant_type triton-fp8w8a8-pertensor \
        --gdn_prefill_backend flashqla \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend fa3 \
        --disable_symm_mem_allreduce \
        --disable_flashinfer_allreduce \
        --enable_cpu_cache \
        --cpu_cache_storage_size "$CPU_CACHE_GB" \
        --linear_att_hash_page_size 4096 \
        --linear_att_page_block_num 8 \
        --linear_att_cache_size "$P_LINEAR_ATT_CACHE_SIZE" \
        --mtp_mode eagle_with_att \
        --mtp_draft_model_dir "$MODEL_DIR" \
        --mtp_step 3 \
        --pd_kv_page_num 16 \
        --pd_kv_page_size 4096 \
        >"$log_file" 2>&1 &
    PIDS+=("$!")
}

start_d_service() {
    local log_file="$LOG_DIR/d.log"
    echo "Starting D at ${NODE_IP}:8122 GPUs=4,5,6,7 TP=4 DP=1"
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    python -m lightllm.server.api_server \
        --model_dir "$MODEL_DIR" \
        --model_name qwen35-27b \
        --run_mode decode \
        --host "$NODE_IP" \
        --port 8122 \
        --nccl_port 12322 \
        --pd_master_ip "$MASTER_ADDR" \
        --pd_master_port "$PD_MASTER_PORT" \
        --nnodes 1 \
        --node_rank 0 \
        --tp 4 \
        --dp 1 \
        --mem_fraction 0.80 \
        --max_total_token_num "$D_MAX_TOTAL_TOKEN_NUM" \
        --max_req_total_len 262144 \
        --running_max_req_size "$D_RUNNING_MAX_REQ_SIZE" \
        --batch_max_tokens 32768 \
        --chunked_prefill_size 8192 \
        --graph_max_batch_size 64 \
        --graph_split_batch_size 32 \
        --graph_grow_step_size 16 \
        --graph_max_len_in_batch 262144 \
        --disable_vision \
        --disable_audio \
        --disable_dynamic_prompt_cache \
        --quant_type triton-fp8w8a8-pertensor \
        --gdn_prefill_backend flashqla \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend fa3 \
        --linear_att_hash_page_size 4096 \
        --linear_att_page_block_num 8 \
        --mtp_mode eagle_with_att \
        --mtp_draft_model_dir "$MODEL_DIR" \
        --mtp_step 3 \
        --pd_kv_page_num 16 \
        --pd_kv_page_size 4096 \
        >"$log_file" 2>&1 &
    PIDS+=("$!")
}

if [[ "$NODE_RANK" -eq 0 ]]; then
    echo "Rank 0: PD master + 8P (TP8/DP4)"
    start_pd_master
    wait_for_pd_master
    start_p_service 0,1,2,3,4,5,6,7 8 4 8120 12320 "$P0_RUNNING_MAX_REQ_SIZE" "$LOG_DIR/p.log" "P0"
    wait_for_pd_topology
    warm_pd_routes
    echo "Public API: http://${MASTER_ADDR}:${PD_MASTER_PORT}"
else
    echo "Rank 1: 4P (TP4/DP2) + 4D (TP4/DP1)"
    wait_for_pd_master
    start_p_service 0,1,2,3 4 2 8121 12321 "$P1_RUNNING_MAX_REQ_SIZE" "$LOG_DIR/p.log" "P1"
    wait_for_local_registration "${PIDS[-1]}" "$LOG_DIR/p.log" prefill "P1"
    start_d_service
    wait_for_local_registration "${PIDS[-1]}" "$LOG_DIR/d.log" decode "D"
    echo "P1=${NODE_IP}:8121 D=${NODE_IP}:8122"
fi

set +e
wait -n
status=$?
set -e
echo "A rank-${NODE_RANK} service exited with status $status. Check $LOG_DIR." >&2
exit "$status"
