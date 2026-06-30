#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2-FP8 CPU KV-cache smoke/perf launch on one 8xH200 node.
MACHINE="${MACHINE:-m39}"
CONTAINER_NAME="${CONTAINER_NAME:-lightllm_glm52_cpu_cache}"
IMAGE="${IMAGE:-ghcr.io/modeltc/lightllm:main}"
MODEL_DIR="${MODEL_DIR:-/mtc/models/GLM-5.2-FP8}"
PORT="${PORT:-8090}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
WORKTREE="${WORKTREE:-/mtc/sufubao/shared_home/sufubao/code/worktree-lightllm/support_glm52}"
AUTOTUNE_CACHE="${AUTOTUNE_CACHE:-/mtc/sufubao/shared_home/lightllm_autotune_cache}"
MAX_REQ_TOTAL_LEN="${MAX_REQ_TOTAL_LEN:-60000}"
CPU_CACHE_STORAGE_SIZE="${CPU_CACHE_STORAGE_SIZE:-16}"
CPU_CACHE_TOKEN_PAGE_SIZE="${CPU_CACHE_TOKEN_PAGE_SIZE:-64}"
PAR_ARGS="${PAR_ARGS---dp 8 --enable_ep_moe}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export CONTAINER_NAME IMAGE MODEL_DIR PORT GPUS WORKTREE AUTOTUNE_CACHE MAX_REQ_TOTAL_LEN
export CPU_CACHE_STORAGE_SIZE CPU_CACHE_TOKEN_PAGE_SIZE PAR_ARGS EXTRA_ARGS

LOCAL_MACHINE=""
case " $(hostname -I) " in
  *" 10.120.178.74 "*) LOCAL_MACHINE=m33 ;;
  *" 10.120.178.75 "*) LOCAL_MACHINE=m34 ;;
  *" 10.120.178.76 "*) LOCAL_MACHINE=m35 ;;
  *" 10.120.178.80 "*) LOCAL_MACHINE=m39 ;;
  *" 10.120.178.82 "*) LOCAL_MACHINE=m41 ;;
  *" 10.210.6.10 "*)   LOCAL_MACHINE=h100 ;;
esac

echo "Launching ${CONTAINER_NAME}: model=${MODEL_DIR} on ${MACHINE} GPUs=${GPUS} port=${PORT} image=${IMAGE}"
echo "CPU cache: ${CPU_CACHE_STORAGE_SIZE}GB, page=${CPU_CACHE_TOKEN_PAGE_SIZE}, kv=fp8kv_dsa"

if [[ "${MACHINE}" == "${LOCAL_MACHINE}" ]]; then
  RUN=(bash -s)
else
  RUN=(ssh "${MACHINE}"
       CONTAINER_NAME="${CONTAINER_NAME}" IMAGE="${IMAGE}" MODEL_DIR="${MODEL_DIR}"
       PORT="${PORT}" GPUS="${GPUS}" WORKTREE="${WORKTREE}" AUTOTUNE_CACHE="${AUTOTUNE_CACHE}"
       MAX_REQ_TOTAL_LEN="${MAX_REQ_TOTAL_LEN}" CPU_CACHE_STORAGE_SIZE="${CPU_CACHE_STORAGE_SIZE}"
       CPU_CACHE_TOKEN_PAGE_SIZE="${CPU_CACHE_TOKEN_PAGE_SIZE}" PAR_ARGS="${PAR_ARGS}" EXTRA_ARGS="${EXTRA_ARGS}"
       bash -s)
fi

"${RUN[@]}" <<'REMOTE'
set -euo pipefail

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} exists. Remove first: docker rm -f ${CONTAINER_NAME}" >&2
  exit 1
fi

docker run -d --init --name "${CONTAINER_NAME}" \
  --gpus all --privileged --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${PORT}:${PORT}" \
  -v /dev/shm/:/dev/shm/ \
  -v /mtc:/mtc \
  -v "${WORKTREE}:/lightllm" \
  -v "${AUTOTUNE_CACHE}:/lightllm/lightllm/common/triton_utils/autotune_kernel_configs" \
  -e CUDA_VISIBLE_DEVICES="${GPUS}" \
  -e LIGHTLLM_FUSED_ADD_RMSNORM="${LIGHTLLM_FUSED_ADD_RMSNORM:-1}" \
  -e LIGHTLLM_FUSED_AR_RMSNORM="${LIGHTLLM_FUSED_AR_RMSNORM:-1}" \
  -e LIGHTLLM_FUSED_SILU_QUANT="${LIGHTLLM_FUSED_SILU_QUANT:-1}" \
  -e LOADWORKER=18 \
  -e LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e PYTHONUNBUFFERED=1 \
  "${IMAGE}" \
  bash -lc "cd /lightllm && python -m lightllm.server.api_server \
      --model_dir ${MODEL_DIR} \
      --host 0.0.0.0 --port ${PORT} \
      --tp 8 ${PAR_ARGS} \
      --mem_fraction 0.85 \
      --max_req_total_len ${MAX_REQ_TOTAL_LEN} \
      --graph_max_batch_size 32 \
      --llm_kv_type fp8kv_dsa \
      --enable_cpu_cache \
      --cpu_cache_storage_size ${CPU_CACHE_STORAGE_SIZE} \
      --cpu_cache_token_page_size ${CPU_CACHE_TOKEN_PAGE_SIZE} \
      ${EXTRA_ARGS}"

echo "Launched ${CONTAINER_NAME}."
echo "Readiness: curl -s http://127.0.0.1:${PORT}/v1/models"
echo "Logs:      docker logs -f ${CONTAINER_NAME}"
REMOTE
