# first
LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --tp 2 --port 8089

# second
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"qwen/qwen3-8b", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code


LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --tp 2 --port 8089 --enable_tpsp_mix_mode

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"qwen/qwen3-8b", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code


LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --tp 2 --dp 2 --port 8089 --enable_tpsp_mix_mode --enable_dp_prefill_balance

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"qwen/qwen3-8b", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code


# 驱动提示词
# 读取 test_qwen3.sh 中的三种测试，分别按照其中的启动命令启动服务，然后运行测试命令得到精度结果，将结果收集到out.txt 中，注意需要标记启动的类型和结果信息。不要用health 接口去判断服务是否启动，直接探测端口是否处于listen状态即可。