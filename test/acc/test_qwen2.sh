# first
LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d --tp 2 --port 8089 --enable_fa3

# second
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"Qwen/Qwen2.5-Math-7B-Instruct", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code