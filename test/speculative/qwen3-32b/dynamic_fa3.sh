MTP_STEP=3
PORT=8089
MEM_FRACTION=""
MAX_TOTAL_TOKEN_NUM=""
MAX_REQ_TOTAL_LEN=""
BATCH_MAX_TOKENS=""
export CUDA_VISIBLE_DEVICES=4,5
# 解析命名参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --mtp-step)
            MTP_STEP="$2"
            shift 2
            ;;
        --mem-fraction)
            MEM_FRACTION="$2"
            shift 2
            ;;
        --max-total-token-num)
            MAX_TOTAL_TOKEN_NUM="$2"
            shift 2
            ;;
        --max-req-total-len)
            MAX_REQ_TOTAL_LEN="$2"
            shift 2
            ;;
        --batch-max-tokens)
            BATCH_MAX_TOKENS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

MODEL_DIR=/mtc/models/qwen3-32b
DRAFT_MODEL_DIR=/mtc/models/qwen3-32b-eagle3

EXTRA_ARGS=()
if [[ -n "${MEM_FRACTION}" ]]; then
    EXTRA_ARGS+=(--mem_fraction "${MEM_FRACTION}")
fi
if [[ -n "${MAX_TOTAL_TOKEN_NUM}" ]]; then
    EXTRA_ARGS+=(--max_total_token_num "${MAX_TOTAL_TOKEN_NUM}")
fi
if [[ -n "${MAX_REQ_TOTAL_LEN}" ]]; then
    EXTRA_ARGS+=(--max_req_total_len "${MAX_REQ_TOTAL_LEN}")
fi
if [[ -n "${BATCH_MAX_TOKENS}" ]]; then
    EXTRA_ARGS+=(--batch_max_tokens "${BATCH_MAX_TOKENS}")
fi

LOADWORKER=18 python -m lightllm.server.api_server --port ${PORT} \
--tp 2 \
--model_dir ${MODEL_DIR} \
--mtp_mode eagle3 \
--mtp_draft_model_dir ${DRAFT_MODEL_DIR} \
--graph_grow_step_size 1 \
--mtp_step ${MTP_STEP}  \
--llm_decode_att_backend fa3 \
--mtp_dynamic_verify \
"${EXTRA_ARGS[@]}"
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_prefill_microbatch_overlap \
#--enable_decode_microbatch_overlap \
