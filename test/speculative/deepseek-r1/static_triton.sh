MTP_STEP=4
# 解析命名参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mtp-step)
            MTP_STEP="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

MODEL_DIR=/mtc/models/DeepSeek-R1
DRAFT_MODEL_DIR=/mtc/models/DeepSeek-R1-NextN

# H200 single node deepseek R1 tp mode
LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
--tp 8 \
--model_dir ${MODEL_DIR} \
--mtp_mode eagle_with_att \
--mtp_draft_model_dir ${DRAFT_MODEL_DIR} \
--mtp_step ${MTP_STEP}  \
--llm_decode_att_backend triton
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_prefill_microbatch_overlap \
#--enable_decode_microbatch_overlap \
