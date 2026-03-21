# 先清理掉占用显卡和端口的进程

# first
LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/Qwen2.5-14B-Instruct --tp 2 --port 8089 --llm_kv_type fp8kv_sph --kv_quant_calibration_config_path /mtc/wzj/lightllm_dev/LightLLM/test/advanced_config/fp8_calibration_per_head/test_kv_cache_calib_per_head_qwen2.5_14b.json

# second
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"Qwen/Qwen2.5-14B-Instruct", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 64 --confirm_run_unsafe_code