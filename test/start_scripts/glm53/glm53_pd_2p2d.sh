#!/usr/bin/env bash
# ConfigServer + four independent PD masters + a two-host P group and D group.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: glm53-pd <config|master|p0|p1|d0|d1> [--dry-run]
Run master once on each host/pod (port 60011, one worker each).
master0/master1/master2/master3 are aliases with distinct log names.
Set CONFIG_SERVER_HOST to the ConfigServer DNS name or reachable address.
ConfigServer binds to CONFIG_SERVER_BIND_HOST (default 0.0.0.0).
P0/P1 and D0/D1 each form a TP16/DP16 group; the role supplies node_rank.
Mount the same model at MODEL_DIR (default /model) on all GPU hosts.
Logs: RUN_DIR/logs/ROLE.log and stdout (RUN_DIR defaults /var/log/lightllm).
Overrides: CONFIG_SERVER_PORT, CONFIG_SERVER_BIND_HOST, MODEL_DIR, CPU_MODEL_DIR, SOURCE_DIR,
CPU_SOURCE, GPU_SOURCE, PYTHON_BIN, RUN_DIR, MASTER_PORT, DISK_CACHE_DIR.
Tuning overrides: P_BATCH_MAX_TOKENS, P_CHUNKED_PREFILL_SIZE,
P_MEM_FRACTION, P_MTP_STEP, P_SSM_STATE_MODE, LIGHTLLM_GLM53_PREFILL_LOGITS_MIB,
PD_KV_PAGE_SIZE, PD_KV_PAGE_NUM.
Use the same PD_KV_PAGE_SIZE/PD_KV_PAGE_NUM on every P and D role.
--dry-run prints commands and paths without starting services or writing files.
EOF
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
if [[ "$1" == --help || "$1" == -h ]]; then usage; exit 0; fi
ROLE="$1"; shift
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
case "$ROLE" in config|master|master[0-3]|p0|p1|d0|d1) ;; *) usage >&2; exit 2 ;; esac

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
MODEL_DIR="${MODEL_DIR:-/model}"
CPU_MODEL_DIR="${CPU_MODEL_DIR:-$MODEL_DIR}"
RUN_DIR="${RUN_DIR:-/var/log/lightllm}"
CONFIG_SERVER_PORT="${CONFIG_SERVER_PORT:-60088}"
if [[ "$ROLE" == config ]]; then
  # A Kubernetes Service address belongs to the virtual service, not this pod.
  # Peers use CONFIG_SERVER_HOST; the server binds on its own interfaces.
  CONFIG_SERVER_HOST="${CONFIG_SERVER_BIND_HOST:-0.0.0.0}"
else
  : "${CONFIG_SERVER_HOST:?Set CONFIG_SERVER_HOST to the ConfigServer service DNS name or address}"
fi
SOURCE_DIR="${SOURCE_DIR:-/lightllm}"
SERVICE_ENV=(
  LIGHTLLM_JSON_SCHEMA_MAX_LENGTH=65536
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false LIGHTLLM_LOG_LEVEL=info
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  "NO_PROXY=${CONFIG_SERVER_HOST},127.0.0.1,localhost,${NO_PROXY:-}"
  "no_proxy=${CONFIG_SERVER_HOST},127.0.0.1,localhost,${no_proxy:-}"
)

if [[ "$ROLE" == config || "$ROLE" == master* ]]; then
  SOURCE_DIR="${CPU_SOURCE:-$SOURCE_DIR}"
  SERVICE_ENV+=(
    CUDA_VISIBLE_DEVICES=
    LIGHTLLM_PD_NODE_RESOURCE_WAIT_TIMEOUT_SECONDS=200
    LIGHTLLM_PD_NODE_CONTINUATION_RESOURCE_WAIT_TIMEOUT_SECONDS=200
    LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS=0
    "PYTHONPATH=${SOURCE_DIR}${CPU_EXTRA_PYTHONPATH:+:${CPU_EXTRA_PYTHONPATH}}"
  )
  if [[ "$ROLE" == config ]]; then
    PORT="$CONFIG_SERVER_PORT"
    ARGS=(--run_mode config_server --config_server_host "$CONFIG_SERVER_HOST" --config_server_port "$PORT")
  else
    PORT="${MASTER_PORT:-60011}"
    ARGS=(
      --run_mode pd_master --model_dir "$CPU_MODEL_DIR" --host 0.0.0.0 --port "$PORT"
      --httpserver_workers 1 --config_server_host "$CONFIG_SERVER_HOST" --config_server_port "$CONFIG_SERVER_PORT"
      --select_p_d_node_strategy cache_aware --pd_master_mode 2p2d --pd_trans_mode nixl
      --disable_pd_cache_high_priority --running_max_req_size 128 --max_req_total_len 1048576
      --tool_call_parser glm47 --reasoning_parser glm45
    )
  fi
else
  SOURCE_DIR="${GPU_SOURCE:-$SOURCE_DIR}"
  SERVICE_ENV+=(
    LIGHTLLM_PD_KV_TRANSFER_TIMEOUT_SECONDS=600
    "PYTHONPATH=${SOURCE_DIR}${GPU_EXTRA_PYTHONPATH:+:${GPU_EXTRA_PYTHONPATH}}"
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
    CUDA_HOME=/usr/local/cuda CC=/usr/bin/gcc CXX=/usr/bin/g++
    LIGHTLLM_MODEL_INFER_RECURSION_LIMIT=5000 LIGHTLLM_TOKEN_HASH_LIST_SIZE=4096
    LIGHTLLM_DSV4_THINKING_EFFORT=low LIGHTLLM_USE_UE8M0_SCALES=1
    LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING=1 LOADWORKER=16
    "TRACE_DIR=${RUN_DIR}/profile/${ROLE}"
    DISK_CACHE_NAME=1p1d-p0
  )
  case "$ROLE" in
    p0) RUN_MODE=prefill; NODE_RANK=0; NCCL_PORT=29501; PORT=8019 ;;
    p1) RUN_MODE=prefill; NODE_RANK=1; NCCL_PORT=29501; PORT=8019 ;;
    d0) RUN_MODE=decode; NODE_RANK=0; NCCL_PORT=29502; PORT=8121 ;;
    d1) RUN_MODE=decode; NODE_RANK=1; NCCL_PORT=29502; PORT=8121 ;;
  esac
  ARGS=(
    --model_dir "$MODEL_DIR" --run_mode "$RUN_MODE"
    --nnodes 2 --node_rank "$NODE_RANK" --nccl_host "$CONFIG_SERVER_HOST" --nccl_port "$NCCL_PORT"
    --use_config_server_to_init_nccl
    --config_server_host "$CONFIG_SERVER_HOST" --config_server_port "$CONFIG_SERVER_PORT"
    --pd_trans_mode nixl --pd_kv_page_size "${PD_KV_PAGE_SIZE:-32768}" --pd_kv_page_num "${PD_KV_PAGE_NUM:-2}"
    --tp 16 --dp 16 --enable_ep_moe --host 0.0.0.0 --port "$PORT"
    --graph_max_batch_size 16 --max_req_total_len 1048576
    --mtp_mode eagle_with_att --mtp_draft_model_dir "$MODEL_DIR"
    --quant_type fp8w8a8-b128-deepgemm --linear_att_ssm_data_type bfloat16
    --llm_kv_type fp8kv_dsa --tool_call_parser glm47 --reasoning_parser glm45
    --linear_att_hash_page_size 2048 --linear_att_cache_size 64
  )
  if [[ "$RUN_MODE" == prefill ]]; then
    P_SSM_STATE_MODE="${P_SSM_STATE_MODE:-native}"
    ARGS+=(--mtp_step "${P_MTP_STEP:-1}" --ssm_state_mode "$P_SSM_STATE_MODE")
    case "$P_SSM_STATE_MODE" in
      replay) ARGS+=(--ssm_replay_config '{"cache_len":8,"compute_mode":"precompute","fold_method":"raw"}') ;;
      native) ;;
      *) echo 'P_SSM_STATE_MODE must be native or replay' >&2; exit 2 ;;
    esac
    SERVICE_ENV+=("LIGHTLLM_GLM53_PREFILL_LOGITS_MIB=${LIGHTLLM_GLM53_PREFILL_LOGITS_MIB:-512}")
    ARGS+=(
      --use_dp_split_mode_connect_pd_master
      --batch_max_tokens "${P_BATCH_MAX_TOKENS:-8192}" --chunked_prefill_size "${P_CHUNKED_PREFILL_SIZE:-8192}"
      --running_max_req_size 128 --mem_fraction "${P_MEM_FRACTION:-0.73}" --disable_cudagraph
      --linear_att_page_block_num 8
      --enable_cpu_cache --cpu_cache_storage_size 1024
      --enable_disk_cache --disk_cache_storage_size 10240
      --disk_cache_dir "${DISK_CACHE_DIR:-${RUN_DIR}/disk_cache/${ROLE}}"
    )
  else
    ARGS+=(
      --mtp_step 3 --ssm_state_mode replay
      --ssm_replay_config '{"cache_len":8,"compute_mode":"precompute","fold_method":"raw"}'
      --batch_max_tokens 1024 --running_max_req_size 256
      --max_total_token_num 2800000 --mem_fraction 0.85
      --enable_profiling torch_profiler
    )
  fi
fi

LOG_FILE="${RUN_DIR}/logs/${ROLE}.log"
printf 'role=%s\nsource_dir=%s\nlog_file=%s\n' "$ROLE" "$SOURCE_DIR" "$LOG_FILE"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'command='
  printf '%q ' env "${SERVICE_ENV[@]}" "$PYTHON_BIN" -m lightllm.server.api_server "${ARGS[@]}"
  printf '\n'
  exit 0
fi
[[ -d "$SOURCE_DIR/lightllm" ]] || { echo "Source directory not found: $SOURCE_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
command -v flock >/dev/null || { echo 'flock is required' >&2; exit 1; }
# Refuse occupied ports before creating runtime files or loading any weights.
"$PYTHON_BIN" - "$PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
if not 0 < port < 65536:
    raise SystemExit("Invalid API port")
with socket.socket() as listener:
    try:
        listener.bind(("0.0.0.0", port))
    except OSError:
        raise SystemExit(f"API port {port} occupied; no service was started")
PY
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids" "$RUN_DIR/profile/$ROLE"
exec 9>"${RUN_DIR}/pids/${ROLE}.launch.lock"
flock -n 9 || { echo "Role $ROLE already has a launcher" >&2; exit 1; }
cd "$SOURCE_DIR"
printf '%s\n' "$$" > "${RUN_DIR}/pids/${ROLE}.pid"
exec > >(tee -a "$LOG_FILE") 2>&1
printf '\n[%s] role=%s pid=%s source=%s\n' "$(date -u +%FT%TZ)" "$ROLE" "$$" "$SOURCE_DIR"
exec env "${SERVICE_ENV[@]}" "$PYTHON_BIN" -m lightllm.server.api_server "${ARGS[@]}"
