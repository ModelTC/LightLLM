#!/usr/bin/env bash
set -euo pipefail
ulimit -Sn 1048576
source "$(dirname -- "${BASH_SOURCE[0]}")/config.txt"
/opt/conda/bin/python -c 'import lightllm,resource,sys; print(sys.executable,lightllm.__file__,resource.getrlimit(resource.RLIMIT_NOFILE),flush=True)'
exec /opt/conda/bin/python -u -m lightllm.server.api_server \
    --model_dir "$MODEL_DIR" \
    --model_name DeepSeek-V4.1-Flash-FP8 \
    --run_mode pd_master \
    --disable_pd_cache_high_priority \
    --pd_master_mode "$PD_MASTER_MODE" \
    --pd_trans_mode nixl \
    --host 0.0.0.0 \
    --port "$PD_MASTER_PORT" \
    --httpserver_workers 4 \
    --max_req_total_len "$MAX_REQ_TOTAL_LEN"
