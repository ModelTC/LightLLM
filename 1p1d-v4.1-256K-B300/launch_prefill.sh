#!/usr/bin/env bash
set -euo pipefail
ulimit -Sn 1048576
source "$(dirname -- "${BASH_SOURCE[0]}")/config.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3
/opt/conda/bin/python -c 'import lightllm,resource,sys; print(sys.executable,lightllm.__file__,resource.getrlimit(resource.RLIMIT_NOFILE),flush=True)'
exec /opt/conda/bin/python -u -m lightllm.server.api_server \
    --model_dir "$MODEL_DIR" \
    --model_name DeepSeek-V4.1-Flash-FP8 \
    --run_mode prefill \
    --pd_trans_mode nixl \
    --pd_kv_page_size 2048 \
    --pd_kv_page_num 8 \
    --tp 4 \
    --dp 4 \
    --nnodes 1 \
    --node_rank 0 \
    --nccl_host 127.0.0.1 \
    --nccl_port 28741 \
    --enable_ep_moe \
    --ep_moe_backend deepep_v2 \
    --data_type bfloat16 \
    --host "$PD_NODE_IP" \
    --port 8019 \
    --httpserver_workers 4 \
    --pd_master_ip "$PD_MASTER_IP" \
    --pd_master_port "$PD_MASTER_PORT" \
    --running_max_req_size 1024 \
    --max_req_total_len "$MAX_REQ_TOTAL_LEN" \
    --mtp_mode dspark \
    --mtp_step 3 \
    --mtp_draft_model_dir "$MODEL_DIR" \
    --disable_audio \
    --batch_max_tokens 16384 \
    --chunked_prefill_size 8192 \
    --mem_fraction 0.9 \
    --disable_cudagraph \
    --linear_att_page_block_num 16 \
    --enable_cpu_cache \
    --enable_dp_prompt_cache_fetch \
    --cpu_cache_storage_size 1024 \
    --enable_disk_cache \
    --disk_cache_storage_size 5120 \
    --cache_placement_strategy legacy \
    --dp_balancer cache_aware \
    --disk_cache_dir /mnt/afs-agentpro/lightllm_disk_cache
