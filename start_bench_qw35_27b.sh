#!/usr/bin/env bash
set -euo pipefail

# Benchmark/dev launch for Qwen3.5-27B MTP on m39 GPU4 (TP1, port 18089).
# Mounts THIS worktree's lightllm/ over the image's installed package for dev iteration.
# MTP=1 (default) enables eagle_with_att mtp_step=3; MTP=0 runs the baseline.
# PROFILE=torch_profiler|nvtx adds --enable_profiling (trace dir mounted to host).

MTP="${MTP:-1}"
MTP_STEP="${MTP_STEP:-3}"
PROFILE="${PROFILE:-}"
CONTAINER_NAME="${CONTAINER_NAME:-lightllm_bench_qw35_dev}"
IMAGE="${IMAGE:-registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm:qwen35-stable-260703-294623f5}"
MODEL_DIR="${MODEL_DIR:-/nvme/models/Qwen3.5-27B}"
PORT="${PORT:-18089}"
GPUS="${GPUS:-4}"
WORKTREE="${WORKTREE:-/nvme/sufubao/m39-home/code/worktree-lightllm/qwen35-27b-260706}"
AUTOTUNE_CACHE="${AUTOTUNE_CACHE:-/mtc/sufubao/shared_home/lightllm_autotune_cache}"
TRACE_DIR="${TRACE_DIR:-${WORKTREE}/trace}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

MTP_FLAGS=""
if [[ "${MTP}" == "1" ]]; then
  MTP_FLAGS="--mtp_mode eagle_with_att --mtp_draft_model_dir ${MODEL_DIR} --mtp_step ${MTP_STEP}"
fi

PROFILE_FLAGS=""
if [[ -n "${PROFILE}" ]]; then
  PROFILE_FLAGS="--enable_profiling ${PROFILE}"
  mkdir -p "${TRACE_DIR}"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} already exists. Remove it first:  docker rm -f ${CONTAINER_NAME}"
  exit 1
fi

echo "Launching ${CONTAINER_NAME}: model=${MODEL_DIR} GPU=${GPUS} port=${PORT} MTP=${MTP} step=${MTP_STEP} profile=${PROFILE:-off} image=${IMAGE}"
echo "Dev mount: ${WORKTREE}/lightllm -> /lightllm/lightllm"

docker run -d --init --name "${CONTAINER_NAME}" \
  --gpus all --privileged --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${PORT}:${PORT}" \
  -v /dev/shm/:/dev/shm/ \
  -v /nvme:/nvme \
  -v "${WORKTREE}/lightllm:/lightllm/lightllm" \
  -v "${AUTOTUNE_CACHE}:/lightllm/lightllm/common/triton_utils/autotune_kernel_configs" \
  -v "${TRACE_DIR}:/lightllm_trace" \
  -e CUDA_VISIBLE_DEVICES="${GPUS}" \
  -e LOADWORKER=18 \
  -e LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 \
  -e LIGHTLLM_TRACE_DIR=/lightllm_trace \
  -e LIGHTLLM_QWEN35_MTP_PRENORM_HIDDENS="${PRENORM_HIDDENS:-0}" \
  -e LIGHTLLM_DISABLE_MTP_FUSED_GRAPH="${DISABLE_MTP_FUSED_GRAPH:-0}" \
  "${IMAGE}" \
  bash -lc "python -m lightllm.server.api_server \
      --model_dir ${MODEL_DIR} \
      --host 0.0.0.0 --port ${PORT} \
      --tp 1 \
      --max_total_token_num 40000 \
      --max_req_total_len 16384 \
      --mem_fraction 0.8 \
      --batch_max_tokens 8192 \
      --running_max_req_size 40 \
      --graph_max_batch_size 32 \
      ${MTP_FLAGS} ${PROFILE_FLAGS} ${EXTRA_FLAGS}"

echo "Launched ${CONTAINER_NAME} (MTP=${MTP})."
echo "Readiness: curl -s http://127.0.0.1:${PORT}/v1/models"
echo "Logs:      docker logs -f ${CONTAINER_NAME}"
