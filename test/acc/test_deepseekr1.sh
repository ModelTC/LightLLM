LOADWORKER=18 python -m lightllm.server.api_server --batch_max_tokens 6000 --model_dir /mtc/models/DeepSeek-R1 --tp 8 --port 8089 --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3



HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"deepseek-ai/DeepSeek-R1", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code