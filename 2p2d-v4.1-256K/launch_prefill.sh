#!/usr/bin/env bash
set -euo pipefail
NODE_RANK=${1:?usage: bash launch_prefill.sh NODE_RANK}
ulimit -Sn 1048576
source "$(dirname -- "${BASH_SOURCE[0]}")/config.txt"
export NVSHMEM_IB_ENABLE_IBGDA=0
/opt/conda/bin/python -c 'import lightllm,resource,sys; print(sys.executable,lightllm.__file__,resource.getrlimit(resource.RLIMIT_NOFILE),flush=True)'
exec /opt/conda/bin/python -u -m lightllm.server.api_server \
    --model_dir "$MODEL_DIR" \
    --model_name DeepSeek-V4.1-Flash-FP8 \
    --run_mode prefill \
    --pd_trans_mode nixl \
    --tp 16 \
    --dp 16 \
    --nnodes 2 \
    --node_rank "$NODE_RANK" \
    --nccl_host "$PREFILL_NCCL_HOST" \
    --nccl_port 28741 \
    --enable_ep_moe \
    --ep_moe_backend deepep_v2 \
    --data_type bfloat16 \
    --host 0.0.0.0 \
    --port 8019 \
    --httpserver_workers 4 \
    --pd_master_ip "$PD_MASTER_IP" \
    --pd_master_port "$PD_MASTER_PORT" \
    --running_max_req_size 256 \
    --max_req_total_len "$MAX_REQ_TOTAL_LEN" \
    --mtp_mode dspark \
    --mtp_step 5 \
    --mtp_draft_model_dir "$MTP_DRAFT_MODEL_DIR" \
    --disable_audio \
    --batch_max_tokens 8192 \
    --chunked_prefill_size 8192 \
    --mem_fraction 0.75 \
    --disable_cudagraph \
    --linear_att_page_block_num 16 \
    --enable_cpu_cache \
    --cpu_cache_storage_size 1024 \
    --enable_disk_cache \
    --disk_cache_storage_size 5120 \
    --disk_cache_dir /root/.cache/lightllm-dsv41-opt/pd_disk_cache
