#!/usr/bin/env bash
# Check nvidia_peermem (GPUDirect RDMA) for NIXL PD / UCX over IB.
# Usage: bash skills/test_model/qwen3-8b-pd-nixl/check_nvidia_peermem.sh [LOG_DIR]
#   LOG_DIR optional: scan prefill.log / decode.log for UCX GPUDirect lines.
set -euo pipefail

LOG_DIR="${1:-}"
FAIL=0

echo "=== nvidia_peermem ==="

if lsmod 2>/dev/null | awk '{print $1}' | grep -qx nvidia_peermem; then
  ver="$(cat /sys/module/nvidia_peermem/version 2>/dev/null || echo '?')"
  echo "OK: module loaded (version ${ver})"
else
  echo "FAIL: nvidia_peermem not loaded"
  FAIL=1
fi

if [[ -n "$LOG_DIR" ]]; then
  for f in prefill.log decode.log; do
    [[ -f "${LOG_DIR}/${f}" ]] || continue
    if grep -q 'GPUDirect RDMA is not detected' "${LOG_DIR}/${f}" 2>/dev/null; then
      echo "FAIL: ${f} -> GPUDirect RDMA is not detected (restart services after modprobe)"
      FAIL=1
    elif grep -q 'GPUDirect RDMA is detected' "${LOG_DIR}/${f}" 2>/dev/null; then
      echo "OK: ${f} -> GPUDirect RDMA is detected"
    fi
  done
fi

if [[ "$FAIL" -ne 0 ]]; then
  cat <<'EOF'

Enable GPUDirect RDMA:
  sudo modprobe nvidia_peermem
  lsmod | grep nvidia_peermem
  # cross-node: run on every host; then restart prefill / decode
EOF
  exit 1
fi

exit 0
