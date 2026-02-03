#!/bin/bash
# BFCL (Berkeley Function Calling Leaderboard) evaluation script for LightLLM
#
# Prerequisites:
# 1. Start LightLLM server:
#    python -m lightllm.server.api_server \
#        --model_dir /path/to/GLM-4.7-Flash \
#        --tp 1 \
#        --port 8000
#
# 2. Install dependencies:
#    pip install openai tqdm datasets

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-GLM-4.7-Flash}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
PORT="${PORT:-8000}"
TEST_CATEGORY="${TEST_CATEGORY:-simple}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Check if server is running
if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "Error: LightLLM server not running on port ${PORT}"
    echo "Start the server first with:"
    echo "  python -m lightllm.server.api_server --model_dir /path/to/model --tp 1 --port ${PORT}"
    exit 1
fi

echo "=========================================="
echo "BFCL Function Calling Evaluation"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Server: ${BASE_URL}"
echo "Test Category: ${TEST_CATEGORY}"
echo ""

# Run evaluation
python "$(dirname "$0")/eval_bfcl.py" \
    --model_name "${MODEL_NAME}" \
    --base_url "${BASE_URL}" \
    --test_category "${TEST_CATEGORY}" \
    --num_workers "${NUM_WORKERS}" \
    --output "bfcl_results_${TEST_CATEGORY}_$(date +%Y%m%d_%H%M%S).json"

echo ""
echo "Evaluation complete!"
