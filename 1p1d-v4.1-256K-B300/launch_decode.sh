#!/usr/bin/env bash
set -euo pipefail
ulimit -Sn 1048576
source "$(dirname -- "${BASH_SOURCE[0]}")/config.txt"
export CUDA_VISIBLE_DEVICES=4,5,6,7
/opt/conda/bin/python -c 'import lightllm,resource,sys; print(sys.executable,lightllm.__file__,resource.getrlimit(resource.RLIMIT_NOFILE),flush=True)'
exec /opt/conda/bin/python -u -m lightllm.server.api_server \
    --model_dir "$MODEL_DIR" \
    --model_name DeepSeek-V4.1-Flash-FP8 \
    --run_mode decode \
    --pd_trans_mode nixl \
    --pd_kv_page_size 2048 \
    --pd_kv_page_num 8 \
    --tp 4 \
    --dp 4 \
    --node_rank 0 \
    --nccl_host 127.0.0.1 \
    --nccl_port 28841 \
    --enable_ep_moe \
    --ep_moe_backend deepep_v2 \
    --data_type bfloat16 \
    --host "$PD_NODE_IP" \
    --port 8121 \
    --httpserver_workers 4 \
    --pd_master_ip "$PD_MASTER_IP" \
    --pd_master_port "$PD_MASTER_PORT" \
    --running_max_req_size 1024 \
    --max_req_total_len "$MAX_REQ_TOTAL_LEN" \
    --mtp_mode dspark \
    --mtp_step 3 \
    --mtp_draft_model_dir "$MODEL_DIR" \
    --mtp_dynamic_verify \
    --batch_max_tokens 1024 \
    --mem_fraction 0.9 \
    --graph_max_batch_size 256 \
    --graph_split_batch_size 16
