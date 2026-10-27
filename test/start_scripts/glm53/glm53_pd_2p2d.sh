#!/usr/bin/env bash
# support_ds4 control plane: four independent PD masters + config server + HAProxy.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: glm53_pd_2p2d.sh <cpu|config|master0|master1|master2|master3|proxy|p0|p1|d0|d1> [--dry-run]

cpu runs config, four independent single-worker masters, and proxy; Ctrl-C stops only its children.
Each GPU host has eight H100 GPUs. P0/P1 form one 16-GPU group;
D0/D1 form another. The role sets node_rank automatically within its group.
--dry-run prints the command and log path without starting or probing a service.

Path overrides: CPU_SOURCE, GPU_SOURCE, CPU_MODEL_DIR, MODEL_DIR,
CPU_EXTRA_PYTHONPATH, GPU_EXTRA_PYTHONPATH, RUN_DIR, PYTHON_BIN.
Address overrides: PD_MASTER_IP, PD_MASTER_PORT (proxy), MASTER_BASE_PORT,
CONFIG_SERVER_HOST, CONFIG_SERVER_PORT, P0_IP, P1_IP, D0_IP, D1_IP.
Other overrides: HAPROXY_BIN, GPU_RUNTIME_ROOT. Default control ports: 60011, 60088, 60111-60114.
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
case "$ROLE" in cpu|config|master[0-3]|proxy|p0|p1|d0|d1) ;; *) usage >&2; exit 2 ;; esac

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
CPU_SOURCE="${CPU_SOURCE:-/opt/lightllm-pd-master-60011-20261026/glm53_optim}"
GPU_SOURCE="${GPU_SOURCE:-/mnt/niushengxiao/LightLLM-glm53-optim-20261027}"
CPU_MODEL_DIR="${CPU_MODEL_DIR:-/opt/lightllm-pd-master-60011-20261026/model_meta}"
MODEL_DIR="${MODEL_DIR:-/nvme/models/sensenova-flash-320b-v22-20261005-tp1-pp8-vpp3-cp8-ep16/v0-20261006-005328/checkpoint-600-fp8-e8m0/}"
PD_MASTER_IP="${PD_MASTER_IP:-10.119.16.95}"
PD_MASTER_PORT="${PD_MASTER_PORT:-60011}"
MASTER_BASE_PORT="${MASTER_BASE_PORT:-60111}"
CONFIG_SERVER_HOST="${CONFIG_SERVER_HOST:-$PD_MASTER_IP}"
CONFIG_SERVER_PORT="${CONFIG_SERVER_PORT:-60088}"
HAPROXY_BIN="${HAPROXY_BIN:-haproxy}"
P0_IP="${P0_IP:-10.119.25.88}"
P1_IP="${P1_IP:-10.119.25.80}"
D0_IP="${D0_IP:-10.119.25.76}"
D1_IP="${D1_IP:-10.119.25.72}"
PROXY_BYPASS="${P0_IP},${P1_IP},${D0_IP},${D1_IP},${PD_MASTER_IP},${CONFIG_SERVER_HOST},127.0.0.1,localhost"

SERVICE_ENV=(
  LIGHTLLM_JSON_SCHEMA_MAX_LENGTH=65536
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false LIGHTLLM_LOG_LEVEL=info
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  PATH=/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  "NO_PROXY=${PROXY_BYPASS}" "no_proxy=${PROXY_BYPASS}"
)

if [[ "$ROLE" == "cpu" ]]; then
  # All children have separate ports, process supervisors, request-id generators,
  # logs and locks. No --httpserver_workers 4 path is used.
  if [[ "$DRY_RUN" == "1" ]]; then
    for child in config master0 master1 master2 master3 proxy; do
      bash "$0" "$child" --dry-run
    done
    exit 0
  fi
  # Check every control-plane port before starting any child. Never stop an
  # existing deployment or attach it to this launcher's cleanup trap.
  "$PYTHON_BIN" - "$PD_MASTER_PORT" "$CONFIG_SERVER_PORT" "$MASTER_BASE_PORT" <<'CHECK'
import socket
import sys
front, config, base = map(int, sys.argv[1:])
ports = [front, config, *range(base, base + 4)]
if len(set(ports)) != len(ports) or not all(0 < p < 65536 for p in ports):
    raise SystemExit("Control-plane ports must be valid and distinct")
for port in ports:
    with socket.socket() as listener:
        try:
            listener.bind(("0.0.0.0", port))
        except OSError:
            raise SystemExit(f"Port {port} occupied; no control-plane process started")
CHECK
  command -v "$HAPROXY_BIN" >/dev/null || { echo 'HAProxy is required' >&2; exit 1; }
  children=()
  cleanup() {
    trap - EXIT INT TERM
    for pid in "${children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${children[@]}"; do wait "$pid" 2>/dev/null || true; done
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  for child in config master0 master1 master2 master3 proxy; do
    bash "$0" "$child" &
    children+=("$!")
  done
  # A failed child must not leave a partly running control plane behind.
  wait -n "${children[@]}"
  exit 1
fi

if [[ "$ROLE" == "config" || "$ROLE" == master[0-3] || "$ROLE" == "proxy" ]]; then
  SOURCE_DIR="$CPU_SOURCE"
  RUN_DIR="${RUN_DIR:-/opt/lightllm-pd-master-60011-20261026/glm53-optim-4masters}"
  SERVICE_ENV+=(
    LIGHTLLM_PD_NODE_RESOURCE_WAIT_TIMEOUT_SECONDS=200
    LIGHTLLM_PD_NODE_CONTINUATION_RESOURCE_WAIT_TIMEOUT_SECONDS=200
    LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS=0
    CUDA_VISIBLE_DEVICES=
    "PYTHONPATH=${SOURCE_DIR}:${CPU_EXTRA_PYTHONPATH:-/opt/lightllm-pd-master-60011-20261026/deps}"
  )
  if [[ "$ROLE" == "config" ]]; then
    PORT="$CONFIG_SERVER_PORT"
    ARGS=(--run_mode config_server --config_server_host "$CONFIG_SERVER_HOST" --config_server_port "$PORT")
  elif [[ "$ROLE" == "proxy" ]]; then
    PORT="$PD_MASTER_PORT"
    ARGS=()
  else
    PORT=$((MASTER_BASE_PORT + ${ROLE#master}))
    ARGS=(
      --run_mode pd_master --model_dir "$CPU_MODEL_DIR" --host "$PD_MASTER_IP" --port "$PORT"
      --httpserver_workers 1 --config_server_host "$CONFIG_SERVER_HOST" --config_server_port "$CONFIG_SERVER_PORT"
      --select_p_d_node_strategy cache_aware --pd_master_mode 2p2d --pd_trans_mode nixl
      --disable_pd_cache_high_priority --running_max_req_size 128 --max_req_total_len 1048576
      --tool_call_parser glm47 --reasoning_parser glm45
    )
  fi
else
  SOURCE_DIR="$GPU_SOURCE"
  GPU_DEPLOY_RUN="${GPU_RUNTIME_ROOT:-/mnt/niushengxiao/LightLLM-replay-fp8kv-20261027/outputs/cache-aware-1m-20261027-restart}"
  RUN_DIR="${RUN_DIR:-${GPU_SOURCE}/outputs/glm53-optim-4masters}"
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
    --config_server_host "$CONFIG_SERVER_HOST" --config_server_port "$CONFIG_SERVER_PORT"
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
  if [[ "$ROLE" == "proxy" ]]; then
    printf '%q ' "$HAPROXY_BIN" -W -db -f "${RUN_DIR}/haproxy.cfg"
    printf '\nbackends=%s:%s-%s\n' "$PD_MASTER_IP" "$MASTER_BASE_PORT" "$((MASTER_BASE_PORT + 3))"
  else
    printf '%q ' env "${SERVICE_ENV[@]}" "$PYTHON_BIN" -m lightllm.server.api_server "${ARGS[@]}"
  fi
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
if [[ "$ROLE" == "proxy" ]]; then
  {
    printf '%s\n' 'global' '  maxconn 8192' 'defaults' '  mode http' '  timeout connect 10s' \
      '  timeout client 24h' '  timeout server 24h' '  timeout http-request 60s' \
      'frontend glm53_api' "  bind 0.0.0.0:${PD_MASTER_PORT}" '  default_backend glm53_masters' \
      'backend glm53_masters' '  balance leastconn' '  option tcp-check'
    for index in 0 1 2 3; do
      printf '  server master%s %s:%s check inter 2s fall 3 rise 2\n' "$index" "$PD_MASTER_IP" "$((MASTER_BASE_PORT + index))"
    done
  } > "${RUN_DIR}/haproxy.cfg"
  "$HAPROXY_BIN" -c -f "${RUN_DIR}/haproxy.cfg"
  printf '%s\n' "$$" > "${RUN_DIR}/pids/${ROLE}.pid"
  exec "$HAPROXY_BIN" -W -db -f "${RUN_DIR}/haproxy.cfg" >> "$LOG_FILE" 2>&1
fi
cd "$SOURCE_DIR"
printf '%s\n' "$$" > "${RUN_DIR}/pids/${ROLE}.pid"
printf '\n[%s] role=%s pid=%s source=%s\n' "$(date -u +%FT%TZ)" "$ROLE" "$$" "$SOURCE_DIR" >> "$LOG_FILE"
exec env "${SERVICE_ENV[@]}" "$PYTHON_BIN" -m lightllm.server.api_server "${ARGS[@]}" >> "$LOG_FILE" 2>&1
