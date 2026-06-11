#!/bin/bash

if [[ -n "${LIGHTLLM_SPEC_COMMON_SH_LOADED:-}" ]]; then
    return 0
fi
LIGHTLLM_SPEC_COMMON_SH_LOADED=1

LIGHTLLM_SPEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTLLM_PROJECT_ROOT="${LIGHTLLM_PROJECT_ROOT:-$(cd "${LIGHTLLM_SPEC_DIR}/../.." && pwd)}"

if [[ -n "${LIGHTLLM_SPEC_PYTHON:-}" ]]; then
    :
elif command -v python3 >/dev/null 2>&1; then
    LIGHTLLM_SPEC_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    LIGHTLLM_SPEC_PYTHON="$(command -v python)"
else
    echo "python/python3 not found in PATH" >&2
    return 1
fi

LIGHTLLM_SERVER_PYTHON="${LIGHTLLM_SERVER_PYTHON:-${LIGHTLLM_SPEC_PYTHON}}"
LIGHTLLM_BENCHMARK_SCRIPT="${LIGHTLLM_BENCHMARK_SCRIPT:-${LIGHTLLM_PROJECT_ROOT}/test/benchmark/service/benchmark_sharegpt.py}"

export LIGHTLLM_PROJECT_ROOT
export LIGHTLLM_SPEC_DIR
export LIGHTLLM_SPEC_PYTHON
export LIGHTLLM_SERVER_PYTHON
export LIGHTLLM_BENCHMARK_SCRIPT
