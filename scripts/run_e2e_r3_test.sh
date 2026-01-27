#!/bin/bash
# E2E Test Script for R3 Routing Capture Feature
#
# This script starts a LightLLM server with routing capture enabled,
# runs the client test, and verifies the results.
#
# Requirements:
#   - A MoE model (DeepSeek-V2/V3, Qwen-MoE, Mixtral, etc.)
#   - At least 1 GPU with sufficient memory
#   - LightLLM installed
#
# Usage:
#   ./scripts/run_e2e_r3_test.sh /path/to/moe/model [--tp N]

set -e

MODEL_DIR="${1:-}"
TP="${2:-1}"
PORT=8765

if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 /path/to/moe/model [--tp N]"
    echo ""
    echo "Example:"
    echo "  $0 /models/DeepSeek-V3 --tp 8"
    echo "  $0 /models/Qwen-MoE-A14B --tp 4"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

echo "=========================================="
echo "R3 E2E Test: Routing Capture Feature"
echo "=========================================="
echo "Model: $MODEL_DIR"
echo "TP: $TP"
echo "Port: $PORT"
echo ""

# Kill any existing server on the port
pkill -f "lightllm.server.api_server.*--port $PORT" 2>/dev/null || true
sleep 2

# Start server in background
echo "Starting LightLLM server..."
python -m lightllm.server.api_server \
    --model_dir "$MODEL_DIR" \
    --tp "$TP" \
    --port "$PORT" \
    --enable_return_routed_experts \
    --max_total_token_num 8000 \
    --batch_max_tokens 4000 \
    > /tmp/lightllm_r3_test.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "Log: /tmp/lightllm_r3_test.log"

# Wait for server to be ready
echo "Waiting for server to be ready..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waited ${WAITED}s..."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server failed to start within ${MAX_WAIT}s"
    echo "Server log:"
    tail -50 /tmp/lightllm_r3_test.log
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run client test
echo ""
echo "Running R3 client test..."
echo "=========================================="
python test_r3.py --url "http://localhost:$PORT"
TEST_RESULT=$?

# Cleanup
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Report result
echo ""
echo "=========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "E2E TEST PASSED!"
else
    echo "E2E TEST FAILED!"
    echo "Server log (last 30 lines):"
    tail -30 /tmp/lightllm_r3_test.log
fi
echo "=========================================="

exit $TEST_RESULT
