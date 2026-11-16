#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: glm53-pd-kernels <config|master|p0|p1|d0|d1> [--dry-run]
MODEL_DIR=/model CONFIG_SERVER_HOST=lightllm-config
PD_TOPOLOGY=ep16: two hosts per EP16 group (default).
PD_TOPOLOGY=ep8: each host is an independent EP8 group.
EP_MOE_BACKEND=auto|megamoe (megamoe requires ep8 and SM90).
KDA_PREFILL_BACKEND=auto|flashkda|triton (default auto).
P_MAX_TOTAL_TOKEN_NUM and D_MAX_TOTAL_TOKEN_NUM override KV capacities.
EP16 defaults: P=2249892, D=2800000. EP8 sizes KV from mem_fraction.
P_BATCH_MAX_TOKENS=8192 P_CHUNKED_PREFILL_SIZE=8192
P_MEM_FRACTION=0.73 D_MEM_FRACTION=0.85 MAX_REQ_TOTAL_LEN=1048576
PD_KV_PAGE_SIZE=32768 PD_KV_PAGE_NUM=2 must match across P and D.
CONFIG_SERVER_PORT=60088 PD_MASTER_PORT=60011
DISK_CACHE_DIR=/mnt/vqa/lightllm_disk_cache
--dry-run prints the command without starting a service.
EOF
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
if [[ "$1" == --help || "$1" == -h ]]; then usage; exit 0; fi
ROLE="$1"; shift
DRY_RUN=0
if [[ $# == 1 && "$1" == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ $# == 0 ]] || { usage >&2; exit 2; }
case "$ROLE" in config|master|p0|p1|d0|d1) ;; *) usage >&2; exit 2 ;; esac

MODEL_DIR="${MODEL_DIR:-/model}"
MTP_DRAFT_MODEL_DIR="${MTP_DRAFT_MODEL_DIR:-$MODEL_DIR}"
CONFIG_SERVER_HOST="${CONFIG_SERVER_HOST:-lightllm-config}"
CONFIG_SERVER_PORT="${CONFIG_SERVER_PORT:-60088}"
PD_MASTER_PORT="${PD_MASTER_PORT:-60011}"
MAX_REQ_TOTAL_LEN="${MAX_REQ_TOTAL_LEN:-1048576}"
PD_TOPOLOGY="${PD_TOPOLOGY:-ep16}"
EP_MOE_BACKEND="${EP_MOE_BACKEND:-auto}"
KDA_PREFILL_BACKEND="${KDA_PREFILL_BACKEND:-auto}"
SSM_REPLAY_CONFIG="${SSM_REPLAY_CONFIG:-}"
if [[ -z "$SSM_REPLAY_CONFIG" ]]; then
  SSM_REPLAY_CONFIG='{"cache_len":8,"compute_mode":"precompute","fold_method":"raw"}'
fi
case "$PD_TOPOLOGY" in ep16|ep8) ;; *) echo 'PD_TOPOLOGY must be ep16 or ep8' >&2; exit 2 ;; esac
case "$EP_MOE_BACKEND" in auto|megamoe) ;; *) echo 'EP_MOE_BACKEND must be auto or megamoe' >&2; exit 2 ;; esac
case "$KDA_PREFILL_BACKEND" in auto|flashkda|triton) ;; *) echo 'Invalid KDA_PREFILL_BACKEND' >&2; exit 2 ;; esac
if [[ "$EP_MOE_BACKEND" == megamoe && "$PD_TOPOLOGY" != ep8 ]]; then
  echo 'MegaMoE requires PD_TOPOLOGY=ep8: one NVLink node per EP group.' >&2
  exit 2
fi

export LIGHTLLM_TOKEN_HASH_LIST_SIZE="${LIGHTLLM_TOKEN_HASH_LIST_SIZE:-4096}"
export LIGHTLLM_MODEL_INFER_RECURSION_LIMIT="${LIGHTLLM_MODEL_INFER_RECURSION_LIMIT:-5000}"
export LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING="${LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING:-1}"
export LIGHTLLM_LOG_LEVEL="${LIGHTLLM_LOG_LEVEL:-info}"
export LIGHTLLM_JSON_SCHEMA_MAX_LENGTH="${LIGHTLLM_JSON_SCHEMA_MAX_LENGTH:-65536}"
export LIGHTLLM_PD_NODE_RESOURCE_WAIT_TIMEOUT_SECONDS="${LIGHTLLM_PD_NODE_RESOURCE_WAIT_TIMEOUT_SECONDS:-200}"
export LIGHTLLM_PD_NODE_CONTINUATION_RESOURCE_WAIT_TIMEOUT_SECONDS="${LIGHTLLM_PD_NODE_CONTINUATION_RESOURCE_WAIT_TIMEOUT_SECONDS:-200}"
export LIGHTLLM_PD_KV_TRANSFER_TIMEOUT_SECONDS="${LIGHTLLM_PD_KV_TRANSFER_TIMEOUT_SECONDS:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export LOADWORKER="${LOADWORKER:-16}"
export LIGHTLLM_USE_UE8M0_SCALES="${LIGHTLLM_USE_UE8M0_SCALES:-1}"
export LIGHTLLM_PD_UCX_TLS="${LIGHTLLM_PD_UCX_TLS:-^rc_gda}"
export LIGHTLLM_GLM53_COMPACT_PD_TRANSFER="${LIGHTLLM_GLM53_COMPACT_PD_TRANSFER:-1}"
export LIGHTLLM_GLM53_FAST_SINGLE_GROUP_TOPK="${LIGHTLLM_GLM53_FAST_SINGLE_GROUP_TOPK:-1}"
export LIGHTLLM_USE_SGL_RMSNORM="${LIGHTLLM_USE_SGL_RMSNORM:-0}"
export LIGHTLLM_VISUAL_RPC_INIT_TIMEOUT_SECONDS="${LIGHTLLM_VISUAL_RPC_INIT_TIMEOUT_SECONDS:-240}"
export NO_PROXY="$CONFIG_SERVER_HOST,127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$CONFIG_SERVER_HOST,127.0.0.1,localhost,${no_proxy:-}"

if [[ "$ROLE" == config ]]; then
  ARGS=(--run_mode config_server --config_server_host 0.0.0.0 --config_server_port "$CONFIG_SERVER_PORT")
elif [[ "$ROLE" == master ]]; then
  export LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS=0
  export LIGHTLLM_PD_CACHE_LIVE_PREFIX_FALLBACK=1
  export LIGHTLLM_PD_TOKEN_CACHE_MIB=128
  ARGS=(
    --model_dir "$MODEL_DIR" --run_mode pd_master --disable_pd_cache_high_priority
    --select_p_d_node_strategy cache_aware --pd_master_mode "${PD_MASTER_MODE:-2p2d}"
    --pd_trans_mode nixl --host 0.0.0.0 --port "$PD_MASTER_PORT"
    --config_server_host "$CONFIG_SERVER_HOST" --config_server_port "$CONFIG_SERVER_PORT"
    --httpserver_workers 1 --running_max_req_size 128 --max_req_total_len "$MAX_REQ_TOTAL_LEN"
    --tool_call_parser glm47 --reasoning_parser glm45
  )
else
  case "$ROLE" in
    p0) RUN_MODE=prefill; NODE_RANK=0; NCCL_PORT=29501; PORT=8019 ;;
    p1) RUN_MODE=prefill; NODE_RANK=1; NCCL_PORT=29501; PORT=8019 ;;
    d0) RUN_MODE=decode; NODE_RANK=0; NCCL_PORT=29502; PORT=8121 ;;
    d1) RUN_MODE=decode; NODE_RANK=1; NCCL_PORT=29502; PORT=8121 ;;
  esac
  NNODES=2; WORLD_SIZE=16
  if [[ "$PD_TOPOLOGY" == ep8 ]]; then
    NNODES=1; WORLD_SIZE=8; NODE_RANK=0
    case "$ROLE" in p1) NCCL_PORT=29503 ;; d1) NCCL_PORT=29504 ;; esac
  fi
  ARGS=(
    --model_dir "$MODEL_DIR" --run_mode "$RUN_MODE"
    --nnodes "$NNODES" --node_rank "$NODE_RANK"
    --nccl_host "$CONFIG_SERVER_HOST" --nccl_port "$NCCL_PORT" --use_config_server_to_init_nccl
    --pd_trans_mode nixl --pd_kv_page_size "${PD_KV_PAGE_SIZE:-32768}" --pd_kv_page_num "${PD_KV_PAGE_NUM:-2}"
    --tp "$WORLD_SIZE" --dp "$WORLD_SIZE" --enable_ep_moe --ep_moe_backend "$EP_MOE_BACKEND"
    --host 0.0.0.0 --port "$PORT"
    --config_server_host "$CONFIG_SERVER_HOST" --config_server_port "$CONFIG_SERVER_PORT"
    --max_req_total_len "$MAX_REQ_TOTAL_LEN"
    --mtp_mode eagle_with_att --mtp_draft_model_dir "$MTP_DRAFT_MODEL_DIR"
    --quant_type fp8w8a8-b128-deepgemm --linear_att_ssm_data_type bfloat16 --llm_kv_type fp8kv_dsa
    --llm_prefill_att_backend auto "$KDA_PREFILL_BACKEND"
    --tool_call_parser glm47 --reasoning_parser glm45
    --linear_att_hash_page_size 2048 --linear_att_cache_size 64
  )
  if [[ "$EP_MOE_BACKEND" == megamoe ]]; then ARGS+=(--expert_dtype fp8); fi
  if [[ "$RUN_MODE" == prefill ]]; then
    export DISK_CACHE_NAME="2p2d-${ROLE}"
    export LIGHTLLM_GLM53_PREFILL_LOGITS_MIB="${LIGHTLLM_GLM53_PREFILL_LOGITS_MIB:-512}"
    export LIGHTLLM_GLM53_CPU_CACHE_ASYNC_LOAD=1
    export LIGHTLLM_GLM53_FUSED_KDA_QKV_NORM=1
    export LIGHTLLM_GLM53_PREFILL_ALIGNED_BACKFILL=1
    export LIGHTLLM_GLM53_PREFILL_SHARED_EXPERT_OVERLAP=1
    export LIGHTLLM_GLM53_PREFILL_TOKEN_MAJOR_CONV=1
    export LIGHTLLM_GLM53_TUNED_ACT_QUANT=1
    export LIGHTLLM_GLM53_TUNED_GROUP_QUANT=1
    export LIGHTMEM_IDLE_WAIT=1
    export LIGHTMEM_PARALLEL_SHARD_READS=1
    ARGS+=(
      --use_dp_split_mode_connect_pd_master
      --batch_max_tokens "${P_BATCH_MAX_TOKENS:-8192}" --chunked_prefill_size "${P_CHUNKED_PREFILL_SIZE:-8192}"
      --short_prefill_token_threshold 4096 --running_max_req_size 128 --mem_fraction "${P_MEM_FRACTION:-0.73}"
      --disable_cudagraph --mtp_step 1 --ssm_state_mode native --linear_att_page_block_num 8
      --enable_cpu_cache --cpu_cache_storage_size 1024 --enable_disk_cache --disk_cache_storage_size 10240
      --disk_cache_dir "${DISK_CACHE_DIR:-/mnt/vqa/lightllm_disk_cache}"
    )
    CAPACITY="${P_MAX_TOTAL_TOKEN_NUM:-}"
    if [[ -z "$CAPACITY" && "$PD_TOPOLOGY" == ep16 ]]; then CAPACITY=2249892; fi
  else
    export LIGHTLLM_GLM53_DECODE_MASK_MOE_PADDING=1
    export LIGHTLLM_GLM53_DRAFT_GRAPH_MAX_ROWS=16
    export LIGHTLLM_GLM53_SHARED_EXPERT_OVERLAP=1
    export LIGHTLLM_USE_CUSTOM_FP8_SPARSE_DECODE=1
    ARGS+=(
      --batch_max_tokens 1024 --running_max_req_size 256 --mem_fraction "${D_MEM_FRACTION:-0.85}"
      --graph_max_batch_size 16 --dp_balancer bs_balancer --mtp_step 3 --ssm_state_mode replay
      --ssm_replay_config "$SSM_REPLAY_CONFIG"
    )
    CAPACITY="${D_MAX_TOTAL_TOKEN_NUM:-}"
    if [[ -z "$CAPACITY" && "$PD_TOPOLOGY" == ep16 ]]; then CAPACITY=2800000; fi
  fi
  if [[ -n "$CAPACITY" && "$CAPACITY" != 0 ]]; then ARGS+=(--max_total_token_num "$CAPACITY"); fi
fi
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ "$DRY_RUN" == 1 ]]; then
  printf '%q ' "$PYTHON_BIN" -u -m lightllm.server.api_server "${ARGS[@]}"
  printf '\n'
  exit 0
fi
exec "$PYTHON_BIN" -u -m lightllm.server.api_server "${ARGS[@]}"
