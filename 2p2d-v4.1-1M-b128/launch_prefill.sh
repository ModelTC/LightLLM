#!/usr/bin/env bash
set -euo pipefail
NODE_RANK=${1:?usage: bash launch_prefill.sh NODE_RANK [PREFILL_GROUP]}
ulimit -Sn 1048576
source "$(dirname -- "${BASH_SOURCE[0]}")/config.txt" prefill "${2:-1}"
export DG_JIT_CACHE_DIR="/root/.cache/lightllm-dsv41-opt/native-prefill-cold-83899ab89"
mkdir -p "$DG_JIT_CACHE_DIR"
WEIGHT_CACHE_ARGS=()
if [[ -n "${LIGHTLLM_WEIGHT_CACHE_DIR:-}" ]]; then
    WEIGHT_CACHE_ARGS=(--weight_cache_dir "$LIGHTLLM_WEIGHT_CACHE_DIR")
fi
export NVSHMEM_IB_ENABLE_IBGDA=0
/opt/conda/bin/python -c 'import lightllm,resource,sys; print(sys.executable,lightllm.__file__,resource.getrlimit(resource.RLIMIT_NOFILE),flush=True)'
exec /opt/conda/bin/python -u -m "${LIGHTLLM_SERVER_MODULE:-lightllm.server.api_server}" \
    --model_dir "$MODEL_DIR" \
    --model_name DeepSeek-V4.1-Flash-FP8 \
    --run_mode prefill \
    --pd_decode_allocation_policy "$PD_DECODE_ALLOCATION_POLICY" \
    --pd_trans_mode nixl \
    --pd_kv_page_size 2048 \
    --pd_kv_page_num 8 \
    --tp 16 \
    --dp 16 \
    --nnodes 2 \
    --node_rank "$NODE_RANK" \
    --nccl_host "$PREFILL_NCCL_HOST" \
    --nccl_port 28793 \
    --enable_ep_moe \
    --ep_moe_backend deepep_v2 \
    --data_type bfloat16 \
    --host 0.0.0.0 \
    --port 8019 \
    --httpserver_workers 4 \
    --pd_master_ip "$PD_MASTER_IP" \
    --pd_master_port "$PD_MASTER_PORT" \
    --running_max_req_size 512 \
    --max_req_total_len "$MAX_REQ_TOTAL_LEN" \
    --mtp_mode dspark \
    --mtp_step "$MTP_STEP" \
    --mtp_draft_model_dir "$MTP_DRAFT_MODEL_DIR" \
    --disable_audio \
    --batch_max_tokens 8192 \
    --chunked_prefill_size 8192 \
    --prefill_queue_strategy hrrn_resume \
    --mem_fraction 0.75 \
    --disable_cudagraph \
    --linear_att_page_block_num 16 \
    --enable_cpu_cache \
    --enable_dp_prompt_cache_fetch \
    --cpu_cache_storage_size 512 \
    --enable_disk_cache \
    --disk_cache_storage_size 4608 \
    --cache_placement_strategy legacy \
    --dp_balancer cache_aware \
    --disk_cache_dir /root/.cache/lightllm-dsv41-opt/pd_disk_cache_v41_b128_256k_20261118 \
    "${WEIGHT_CACHE_ARGS[@]}"
