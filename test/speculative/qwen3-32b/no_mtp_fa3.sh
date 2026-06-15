#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

MODEL_DIR="${LIGHTLLM_QWEN3_32B_MODEL_DIR:-/mtc/models/qwen3-32b}"
DRAFT_MODEL_DIR="${LIGHTLLM_QWEN3_32B_DRAFT_MODEL_DIR:-/mtc/models/qwen3-32b-eagle3}"

LOADWORKER=18 "${LIGHTLLM_SERVER_PYTHON}" -m lightllm.server.api_server --port 8088 \
--tp 2 \
--model_dir "${MODEL_DIR}" \
--graph_grow_step_size 1 \
--llm_decode_att_backend triton
