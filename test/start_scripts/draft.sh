# 使能 cpu cache 功能，扩大kv cache 复用的可能性。
LOADWORKER=18 python -m lightllm.server.api_server \
--model_dir /mtc/models/qwen3-8b --tp 2 --dp 1 --enable_cpu_cache  --cpu_cache_storage_size 66 --cpu_cache_token_page_size 128 \
--batch_max_tokens 4096 --chunked_prefill_size 2048 \
--max_total_token_num 20000 \
--llm_kv_type int8kv | tee log.txt


# 精度评测命令
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions \
--model_args '{"model":"Qwen/Qwen3-8B", "base_url":"http://localhost:8000/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code



# H200 single node deepseek R1 tp mode
LOADWORKER=18 python -m lightllm.server.api_server \
--model_dir /mtc/DeepSeek-R1 \
--tp 8 \
--llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 \
--batch_max_tokens 4096 --chunked_prefill_size 2048 \
--max_total_token_num 20000 \
--enable_cpu_cache  --cpu_cache_storage_size 66 --cpu_cache_token_page_size 128

# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_prefill_microbatch_overlap \
#--enable_decode_microbatch_overlap \
# 精度测试。
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"deepseek-ai/DeepSeek-R1", "base_url":"http://localhost:8000/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code