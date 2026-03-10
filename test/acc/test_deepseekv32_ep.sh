LOADWORKER=14 python -m lightllm.server.api_server --model_dir /mtc/sufubao/DeepSeek-V3.2 --tp 8 --graph_max_batch_size 32 --tool_call_parser deepseekv32 --mem_fraction 0.8 --reasoning_parser deepseek-v3 --dp 4 --enable_ep_moe


HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"deepseek-ai/DeepSeek-V3.2", "base_url":"http://localhost:8000/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code

export no_proxy="localhost,127.0.0.1,::1"