#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

IMAGE_PREFIX="${IMAGE_PREFIX:-lightllm}"
CUDA_VERSION="${CUDA_VERSION:-12.8.0}"
IMAGE_TAG="${IMAGE_TAG:-nixl.deepep.cache-cuda${CUDA_VERSION}}"

DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg ENABLE_DEEPEP=1 \
  --build-arg ENABLE_NIXL=1 \
  --build-arg ENABLE_CACHE=1 \
  -t "${IMAGE_PREFIX}:${IMAGE_TAG}" .

