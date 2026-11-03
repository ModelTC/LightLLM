#!/usr/bin/env bash
set -euo pipefail
NODE_RANK=${1:?usage: bash launch_decode.sh NODE_RANK}
ulimit -Sn 1048576
source "$(dirname -- "${BASH_SOURCE[0]}")/config.txt"
export PYTHONPATH="$RUN_DIR/deps/deepep-pr696:$PYTHONPATH"
export NVSHMEM_IB_ENABLE_IBGDA=1
/opt/conda/bin/python -c 'import lightllm,resource,sys; print(sys.executable,lightllm.__file__,resource.getrlimit(resource.RLIMIT_NOFILE),flush=True)'
exec /opt/conda/bin/python -u -m lightllm.server.api_server \
    --model_dir "$MODEL_DIR" \
    --model_name DeepSeek-V4.1-Flash-FP8 \
    --run_mode decode \
    --pd_trans_mode nixl \
    --pd_kv_page_size 2048 \
    --pd_kv_page_num 8 \
    --tp 16 \
    --dp 16 \
    --nnodes 2 \
    --node_rank "$NODE_RANK" \
    --nccl_host "$DECODE_NCCL_HOST" \
    --nccl_port 28841 \
    --enable_ep_moe \
    --ep_moe_backend auto \
    --data_type bfloat16 \
    --host 0.0.0.0 \
    --port 8121 \
    --httpserver_workers 4 \
    --pd_master_ip "$PD_MASTER_IP" \
    --pd_master_port "$PD_MASTER_PORT" \
    --running_max_req_size 768 \
    --max_req_total_len "$MAX_REQ_TOTAL_LEN" \
    --mtp_mode dspark \
    --mtp_step 3 \
    --mtp_dynamic_verify \
    --mtp_draft_model_dir "$MTP_DRAFT_MODEL_DIR" \
    --disable_audio \
    --batch_max_tokens 1024 \
    --mem_fraction 0.75 \
    --graph_max_batch_size 96 \
    --enable_profiling torch_profiler
