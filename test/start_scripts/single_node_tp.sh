# H200 single node deepseek R1 tp mode
LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
--model_dir /path/DeepSeek-R1 \
--tp 8 \
--llm_prefill_att_backend fa3 --llm_decode_att_backend fa3
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_prefill_microbatch_overlap \
#--enable_decode_microbatch_overlap \
