LOADWORKER=18 NUM_MAX_DISPATCH_TOKENS_PER_RANK=256  python -m lightllm.server.api_server  --enable_ep_moe --model_dir /mtc/models/DeepSeek-R1 --tp 8 --dp 8 --port 8089 --max_total_token_num 60000 --graph_max_batch_size 16 --batch_max_tokens 6000 --mtp_mode eagle_with_att --mtp_draft_model_dir /mtc/models/DeepSeek-R1-NextN --mtp_step 2

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"deepseek-ai/DeepSeek-R1", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 32 --confirm_run_unsafe_code


LOADWORKER=18 NUM_MAX_DISPATCH_TOKENS_PER_RANK=256  python -m lightllm.server.api_server  --enable_ep_moe --model_dir /mtc/models/DeepSeek-R1 --tp 8 --dp 8 --port 8089 --max_total_token_num 60000 --graph_max_batch_size 16 --batch_max_tokens 6000 --mtp_mode eagle_with_att --mtp_draft_model_dir /mtc/models/DeepSeek-R1-NextN --mtp_step 2 --enable_tpsp_mix_mode 

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"deepseek-ai/DeepSeek-R1", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 32 --confirm_run_unsafe_code


LOADWORKER=18 NUM_MAX_DISPATCH_TOKENS_PER_RANK=256  python -m lightllm.server.api_server  --enable_ep_moe --model_dir /mtc/models/DeepSeek-R1 --tp 8 --dp 8 --port 8089 --max_total_token_num 60000 --graph_max_batch_size 16 --batch_max_tokens 6000 --mtp_mode eagle_with_att --mtp_draft_model_dir /mtc/models/DeepSeek-R1-NextN --mtp_step 2 --enable_prefill_microbatch_overlap --enable_decode_microbatch_overlap

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"deepseek-ai/DeepSeek-R1", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 32 --confirm_run_unsafe_code


LOADWORKER=18 NUM_MAX_DISPATCH_TOKENS_PER_RANK=256  python -m lightllm.server.api_server  --enable_ep_moe --model_dir /mtc/models/DeepSeek-R1 --tp 8 --dp 8 --port 8089 --max_total_token_num 60000 --graph_max_batch_size 16 --batch_max_tokens 6000 --mtp_mode eagle_with_att --mtp_draft_model_dir /mtc/models/DeepSeek-R1-NextN --mtp_step 2 --enable_prefill_microbatch_overlap --enable_decode_microbatch_overlap --enable_dp_prefill_balance

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"deepseek-ai/DeepSeek-R1", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 32 --confirm_run_unsafe_code

# 帮我写一段提示词，告诉AI单独一个一个的进行上述测试的启动服务，然后再执行评测脚本，将结果写入out.txt 中，注意需要标记启动的参数和结果信息。不要用health 接口去判断服务是否启动，直接探测端口是否处于listen状态即可, 执行评测命令的时候，需要用no_proxy 将本地local ip 排除。
# 不要写额外的脚本统一执行，就是单独一个一个的启动服务，然后再执行评测脚本，然后注意等待服务启动完成，可以20s检测一次其控制台输出，看是否启动完成，还是启动报错。