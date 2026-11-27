#!/usr/bin/env bash
set -euo pipefail
NODE_RANK=${1:?usage: bash launch_decode.sh NODE_RANK}
ulimit -Sn 1048576
source "$(dirname -- "${BASH_SOURCE[0]}")/config.txt" decode
dp=16
nnodes=2
running_max_req_size=1024

# LightLLM caps the decode graph batch size at the per-rank request limit.
dp_per_node=$((dp / nnodes))
per_rank_requests=$((running_max_req_size / (dp_per_node > 0 ? dp_per_node : 1)))
batch_max_tokens=$(((per_rank_requests * (MTP_STEP + 1) + 7) / 8 * 8))
export NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE="$batch_max_tokens"

WEIGHT_CACHE_ARGS=()
if [[ -n "${LIGHTLLM_WEIGHT_CACHE_DIR:-}" ]]; then
    WEIGHT_CACHE_ARGS=(--weight_cache_dir "$LIGHTLLM_WEIGHT_CACHE_DIR")
fi
export NVSHMEM_IB_ENABLE_IBGDA=0
export LIGHTLLM_D_DECODE_TIMELOG_EVERY=128
/opt/conda/bin/python -c 'import lightllm,resource,sys; print(sys.executable,lightllm.__file__,resource.getrlimit(resource.RLIMIT_NOFILE),flush=True)'
exec /opt/conda/bin/python -u -m "${LIGHTLLM_SERVER_MODULE:-lightllm.server.api_server}" \
    --model_dir "$MODEL_DIR" \
    --model_name DeepSeek-V4.1-Flash-FP8 \
    --run_mode decode \
    --pd_decode_allocation_policy "$PD_DECODE_ALLOCATION_POLICY" \
    --pd_trans_mode nixl \
    --pd_kv_page_size 2048 \
    --pd_kv_page_num 8 \
    --tp 16 \
    --dp "$dp" \
    --nnodes "$nnodes" \
    --node_rank "$NODE_RANK" \
    --nccl_host "$DECODE_NCCL_HOST" \
    --nccl_port 28901 \
    --enable_ep_moe \
    --ep_moe_backend deepep_v2 \
    --data_type bfloat16 \
    --host 0.0.0.0 \
    --port 8121 \
    --httpserver_workers 4 \
    --pd_master_ip "$PD_MASTER_IP" \
    --pd_master_port "$PD_MASTER_PORT" \
    --running_max_req_size "$running_max_req_size" \
    --max_req_total_len "$MAX_REQ_TOTAL_LEN" \
    --mtp_mode dspark \
    --mtp_step "$MTP_STEP" \
    --mtp_draft_model_dir "$MTP_DRAFT_MODEL_DIR" \
    --disable_audio \
    --batch_max_tokens "$batch_max_tokens" \
    --mem_fraction 0.77 \
    --graph_max_batch_size 128 \
    --graph_split_batch_size 16 \
    --graph_grow_step_size 8 \
    --mtp_dynamic_verify \
    --sampling_backend flashinfer \
    --enable_profiling torch_profiler \
    "${WEIGHT_CACHE_ARGS[@]}"
