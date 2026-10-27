#!/usr/bin/env bash
# The validated four-host Replay8/FP8KV deployment. Run one role on each host.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: glm53_pd_2p2d.sh <cpu|p0|p1|d0|d1> [--dry-run]

Each GPU host has eight H100 GPUs. P0/P1 form one 16-GPU group;
D0/D1 form another. The role sets node_rank automatically within its group.
--dry-run prints the command and log path without starting or probing a service.

Path overrides: CPU_SOURCE, GPU_SOURCE, CPU_MODEL_DIR, MODEL_DIR,
CPU_EXTRA_PYTHONPATH, GPU_EXTRA_PYTHONPATH, RUN_DIR, PYTHON_BIN.
Address overrides: PD_MASTER_IP, PD_MASTER_PORT, P0_IP, P1_IP, D0_IP, D1_IP.
EOF
}

if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then usage; exit 0; fi
ROLE="$1"
shift
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
case "$ROLE" in cpu|p0|p1|d0|d1) ;; *) usage >&2; exit 2 ;; esac

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
CPU_SOURCE="${CPU_SOURCE:-/opt/lightllm-pd-master-60011-20261026/tuned}"
GPU_SOURCE="${GPU_SOURCE:-/mnt/niushengxiao/LightLLM-replay-fp8kv-20261027}"
CPU_MODEL_DIR="${CPU_MODEL_DIR:-/opt/lightllm-pd-master-60011-20261026/model_meta}"
MODEL_DIR="${MODEL_DIR:-/nvme/models/sensenova-flash-320b-v22-20261005-tp1-pp8-vpp3-cp8-ep16/v0-20261006-005328/checkpoint-600-fp8-e8m0/}"
PD_MASTER_IP="${PD_MASTER_IP:-10.119.16.95}"
PD_MASTER_PORT="${PD_MASTER_PORT:-60011}"
P0_IP="${P0_IP:-10.119.25.88}"
P1_IP="${P1_IP:-10.119.25.80}"
D0_IP="${D0_IP:-10.119.25.76}"
D1_IP="${D1_IP:-10.119.25.72}"
PROXY_BYPASS="${P0_IP},${P1_IP},${D0_IP},${D1_IP},${PD_MASTER_IP},127.0.0.1,localhost"

SERVICE_ENV=(
  LIGHTLLM_JSON_SCHEMA_MAX_LENGTH=65536
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false LIGHTLLM_LOG_LEVEL=info
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  PATH=/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  "NO_PROXY=${PROXY_BYPASS}" "no_proxy=${PROXY_BYPASS}"
)

if [[ "$ROLE" == "cpu" ]]; then
  SOURCE_DIR="$CPU_SOURCE"
  RUN_DIR="${RUN_DIR:-/opt/lightllm-pd-master-60011-20261026/cache-aware-20261027-restart}"
  PORT="$PD_MASTER_PORT"
  SERVICE_ENV+=(
    LIGHTLLM_PD_NODE_RESOURCE_WAIT_TIMEOUT_SECONDS=200
    LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS=0
    CUDA_VISIBLE_DEVICES=
    "PYTHONPATH=${SOURCE_DIR}:${CPU_EXTRA_PYTHONPATH:-/opt/lightllm-pd-master-60011-20261026/deps}"
  )
  ARGS=(
    --run_mode pd_master --model_dir "$CPU_MODEL_DIR" --host 0.0.0.0 --port "$PORT"
    --httpserver_workers 4 --select_p_d_node_strategy cache_aware
    --pd_master_mode 2p2d --pd_trans_mode nixl --disable_pd_cache_high_priority
    --running_max_req_size 128 --max_req_total_len 1048576
    --tool_call_parser glm47 --reasoning_parser glm45
  )
else
  SOURCE_DIR="$GPU_SOURCE"
  GPU_DEPLOY_RUN="${GPU_SOURCE}/outputs/cache-aware-1m-20261027-restart"
  RUN_DIR="${RUN_DIR:-$GPU_DEPLOY_RUN}"
  # These are the existing validated isolated packages, not files shipped in Git.
  DEFAULT_GPU_EXTRA="${GPU_DEPLOY_RUN}/runtime-deps-complete:${GPU_DEPLOY_RUN}/runtime-deps-complete/nvidia_cutlass_dsl/dsl_packages:/var/tmp/sensenova-2p2d-20261026-061910/cubin-deps:/mnt/niushengxiao/LightLLM/outputs/sensenova-2p2d-20261026-061910/deps/docker-pinned/nvidia_cutlass_dsl/dsl_packages:/mnt/niushengxiao/LightLLM/outputs/sensenova-2p2d-20261026-061910/deps/docker-pinned:/mnt/niushengxiao/LightLLM/outputs/sensenova-2p2d-20261026-061910/deps"
  SERVICE_ENV+=(
    LIGHTLLM_PD_KV_TRANSFER_TIMEOUT_SECONDS=600
    "PYTHONPATH=${SOURCE_DIR}:${GPU_EXTRA_PYTHONPATH:-$DEFAULT_GPU_EXTRA}"
    LD_LIBRARY_PATH=/opt/conda/lib CUDA_HOME=/usr/local/cuda
    CC=/usr/bin/gcc CXX=/usr/bin/g++
    LIGHTLLM_MODEL_INFER_RECURSION_LIMIT=5000 LIGHTLLM_TOKEN_HASH_LIST_SIZE=4096
    LIGHTLLM_DSV4_THINKING_EFFORT=low LIGHTLLM_USE_UE8M0_SCALES=1
    LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING=1
    PD_MASTER_MODE=2p2d LOADWORKER=16 MAX_REQ_TOTAL_LEN=1048576
    "TRACE_DIR=${RUN_DIR}/profile/${ROLE}"
    # Preserve the existing cache namespace; the name does not select topology.
    DISK_CACHE_NAME=1p1d-p0
  )
  case "$ROLE" in
    p0) RUN_MODE=prefill; NODE_RANK=0; NCCL_HOST="$P0_IP"; NCCL_PORT=29501; PORT=8019 ;;
    p1) RUN_MODE=prefill; NODE_RANK=1; NCCL_HOST="$P0_IP"; NCCL_PORT=29501; PORT=8019 ;;
    d0) RUN_MODE=decode;  NODE_RANK=0; NCCL_HOST="$D0_IP"; NCCL_PORT=29502; PORT=8121 ;;
    d1) RUN_MODE=decode;  NODE_RANK=1; NCCL_HOST="$D0_IP"; NCCL_PORT=29502; PORT=8121 ;;
  esac
  ARGS=(
    --model_dir "$MODEL_DIR" --run_mode "$RUN_MODE"
    --nnodes 2 --node_rank "$NODE_RANK" --nccl_host "$NCCL_HOST" --nccl_port "$NCCL_PORT"
    --pd_trans_mode nixl --pd_kv_page_size 16384 --pd_kv_page_num 1
    --tp 16 --dp 16 --enable_ep_moe --host 0.0.0.0 --port "$PORT"
    --pd_master_ip "$PD_MASTER_IP" --pd_master_port "$PD_MASTER_PORT"
    --graph_max_batch_size 16 --max_req_total_len 1048576
    --mtp_mode eagle_with_att --mtp_draft_model_dir "$MODEL_DIR" --mtp_step 3
    --quant_type fp8w8a8-b128-deepgemm --linear_att_ssm_data_type bfloat16
    --llm_kv_type fp8kv_dsa --tool_call_parser glm47 --reasoning_parser glm45
    --linear_att_hash_page_size 2048 --linear_att_cache_size 64
    --ssm_state_mode replay
    --ssm_replay_config '{"cache_len":8,"compute_mode":"precompute","fold_method":"raw"}'
  )
  if [[ "$RUN_MODE" == "prefill" ]]; then
    ARGS+=(
      --use_dp_split_mode_connect_pd_master
      --batch_max_tokens 4096 --chunked_prefill_size 2048
      --running_max_req_size 128 --mem_fraction 0.75 --disable_cudagraph
      --linear_att_page_block_num 8
      --enable_cpu_cache --cpu_cache_storage_size 1024
      --enable_disk_cache --disk_cache_storage_size 10240
      --disk_cache_dir /mnt/vqa/lightllm_disk_cache
    )
  else
    ARGS+=(
      --batch_max_tokens 1024 --running_max_req_size 256
      --max_total_token_num 2800000 --mem_fraction 0.85
      --enable_profiling torch_profiler
    )
  fi
fi

LOG_FILE="${RUN_DIR}/logs/${ROLE}.log"
printf 'role=%s\nsource_dir=%s\nlog_file=%s\n' "$ROLE" "$SOURCE_DIR" "$LOG_FILE"
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'command='
  printf '%q ' env "${SERVICE_ENV[@]}" "$PYTHON_BIN" -m lightllm.server.api_server "${ARGS[@]}"
  printf '\n'
  exit 0
fi

[[ -d "$SOURCE_DIR/lightllm" ]] || { echo "Source directory not found: $SOURCE_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
command -v flock >/dev/null || { echo 'flock is required for the per-role launch lock' >&2; exit 1; }
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids" "$RUN_DIR/profile/$ROLE"
exec 9>"${RUN_DIR}/pids/${ROLE}.launch.lock"
flock -n 9 || { echo "Role $ROLE already has a launcher holding its lock" >&2; exit 1; }
# Reject an occupied API port before loading weights; never stop another process.
"$PYTHON_BIN" - "$PORT" <<'PY'
import socket
import sys

with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("0.0.0.0", int(sys.argv[1])))
    except OSError:
        raise SystemExit(f"API port {sys.argv[1]} is occupied; no service was started")
PY
cd "$SOURCE_DIR"
printf '%s\n' "$$" > "${RUN_DIR}/pids/${ROLE}.pid"
printf '\n[%s] role=%s pid=%s source=%s\n' "$(date -u +%FT%TZ)" "$ROLE" "$$" "$SOURCE_DIR" >> "$LOG_FILE"
exec env "${SERVICE_ENV[@]}" "$PYTHON_BIN" -m lightllm.server.api_server "${ARGS[@]}" >> "$LOG_FILE" 2>&1
