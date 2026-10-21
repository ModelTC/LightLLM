#!/usr/bin/env bash
set -euo pipefail

# Run the same script on both eight-GPU nodes, with node rank 0 and 1 respectively.
NODE_RANK=${1:?usage: bash tools/launch_deepseek_v41_dpep16.sh NODE_RANK NCCL_HOST}
NCCL_HOST=${2:?node 0 address is required}
SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ulimit -Sn 1048576
export PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH="$SOURCE_DIR"
export LOADWORKER=16
export OMP_NUM_THREADS=4
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export TRITON_CACHE_DIR=/root/.cache/lightllm-dsv41-opt/triton
export EP_NCCL_ROOT_DIR=/opt/conda/lib/python3.10/site-packages/nvidia/nccl
export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}
export LIGHTLLM_TRACE_DIR=/mnt/vqa/wanzihao/dsv41_dpep16_20261021/profile/trace
cd "$SOURCE_DIR"
command -v ibstat
/opt/conda/bin/python -c 'import lightllm, resource, sys; print(sys.executable, lightllm.__file__, resource.getrlimit(resource.RLIMIT_NOFILE), flush=True)'
exec /opt/conda/bin/python -u -m lightllm.server.api_server \
    --model_dir /mnt/vqa/models/DeepSeek-V4.1-Flash-1017-FP8 \
    --model_name DeepSeek-V4.1-Flash-FP8 \
    --tp 16 --dp 16 --nnodes 2 --node_rank "$NODE_RANK" \
    --nccl_host "$NCCL_HOST" --nccl_port 28941 \
    --enable_ep_moe --data_type bfloat16 \
    --batch_max_tokens 2048 --chunked_prefill_size 1024 \
    --max_req_total_len 8192 \
    --running_max_req_size 256 --graph_max_batch_size 32 \
    --enable_profiling torch_profiler \
    --disable_vision --disable_audio \
    --host 0.0.0.0 --port 18941
