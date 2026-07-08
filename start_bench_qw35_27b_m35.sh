#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-27B MTP verification launch on m35 (TP1, port 18089).
# Mounts the shared-home copy of this worktree's lightllm/ over the image's install.
# Run from m39: bash start_bench_qw35_27b_m35.sh   (sync code first: rsync worktree -> shared home)
# MTP=1 (default) enables eagle_with_att mtp_step=3; MTP=0 runs the baseline.

MACHINE="${MACHINE:-m35}"
MTP="${MTP:-1}"
MTP_STEP="${MTP_STEP:-3}"
CONTAINER_NAME="${CONTAINER_NAME:-lightllm_verify_qw35_mtp}"
IMAGE="${IMAGE:-registry.ms-sc-01.maoshanwangtech.com/ms-ccr/lightllm:qwen35-stable-260703-294623f5}"
MODEL_DIR="${MODEL_DIR:-/nvme0/models/Qwen3.5-27B}"
PORT="${PORT:-18089}"
GPUS="${GPUS:-2}"
WORKTREE="${WORKTREE:-/mtc/sufubao/shared_home/sufubao/code/worktree-lightllm/rl_verl_rebase_main_mtp_optim}"
AUTOTUNE_CACHE="${AUTOTUNE_CACHE:-/mtc/sufubao/shared_home/lightllm_autotune_cache}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

MTP_FLAGS=""
if [[ "${MTP}" == "1" ]]; then
  MTP_FLAGS="--mtp_mode eagle_with_att --mtp_draft_model_dir ${MODEL_DIR} --mtp_step ${MTP_STEP}"
fi

echo "Launching ${CONTAINER_NAME} on ${MACHINE}: model=${MODEL_DIR} GPU=${GPUS} port=${PORT} MTP=${MTP} step=${MTP_STEP} image=${IMAGE}"
echo "Dev mount: ${WORKTREE}/lightllm -> /lightllm/lightllm"

ssh "${MACHINE}" \
  CONTAINER_NAME="${CONTAINER_NAME}" IMAGE="${IMAGE}" MODEL_DIR="${MODEL_DIR}" \
  PORT="${PORT}" GPUS="${GPUS}" WORKTREE="${WORKTREE}" AUTOTUNE_CACHE="${AUTOTUNE_CACHE}" \
  MTP_FLAGS="\"${MTP_FLAGS}\"" EXTRA_FLAGS="\"${EXTRA_FLAGS}\"" \
  PRENORM_HIDDENS="${PRENORM_HIDDENS:-0}" DISABLE_MTP_FUSED_GRAPH="${DISABLE_MTP_FUSED_GRAPH:-0}" \
  bash -s <<'REMOTE'
set -euo pipefail

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} already exists. Remove it first:  docker rm -f ${CONTAINER_NAME}"
  exit 1
fi

docker run -d --init --name "${CONTAINER_NAME}" \
  --gpus all --privileged --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${PORT}:${PORT}" \
  -v /dev/shm/:/dev/shm/ \
  -v /mnt/nvme0:/nvme0 \
  -v /mtc:/mtc \
  -v "${WORKTREE}/lightllm:/lightllm/lightllm" \
  -v "${AUTOTUNE_CACHE}:/lightllm/lightllm/common/triton_utils/autotune_kernel_configs" \
  -e CUDA_VISIBLE_DEVICES="${GPUS}" \
  -e LOADWORKER=18 \
  -e LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 \
  -e LIGHTLLM_QWEN35_MTP_PRENORM_HIDDENS="${PRENORM_HIDDENS}" \
  -e LIGHTLLM_DISABLE_MTP_FUSED_GRAPH="${DISABLE_MTP_FUSED_GRAPH}" \
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
      ${MTP_FLAGS} ${EXTRA_FLAGS}"

echo "Launched ${CONTAINER_NAME}."
echo "Readiness: curl -s http://127.0.0.1:${PORT}/v1/models"
echo "Logs:      docker logs -f ${CONTAINER_NAME}"
REMOTE
