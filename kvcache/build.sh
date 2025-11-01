#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  # Re-exec with bash if script was invoked via sh so we can rely on bash features.
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"
GRPC_DIR="${THIRD_PARTY_DIR}/grpc"
GRPC_TAG="v1.71.0"

if [ ! -f "${GRPC_DIR}/CMakeLists.txt" ]; then
  # Populate gRPC locally once so recurring builds reuse the same checkout.
  echo "[kvcache] Populating third_party/grpc (${GRPC_TAG})..."
  rm -rf "${GRPC_DIR}"
  git clone --depth 1 --branch "${GRPC_TAG}" --recurse-submodules --shallow-submodules \
    https://github.com/grpc/grpc "${GRPC_DIR}"
fi

uv build \
  -C override=cmake.options.Python_ROOT_DIR=$PWD/.venv \
  -C override=cmake.options.CMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path,end='')") \
  -C override=cmake.options.CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -C 'override=cmake.build_args=["--parallel","8"]'

pip uninstall -y kvcache
pip install dist/*.whl
