# first
LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --tp 2 --port 8089

# second
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"qwen/qwen3-8b", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code


LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --tp 2 --port 8089 --enable_tpsp_mix_mode

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"qwen/qwen3-8b", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code


LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --tp 2 --dp 2 --port 8089 --enable_tpsp_mix_mode --enable_dp_prefill_balance

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"qwen/qwen3-8b", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code


LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --tp 2 --dp 2 --port 8089 --max_total_token_num 200000 --enable_cpu_cache  --cpu_cache_storage_size 128 --cpu_cache_token_page_size 128

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"qwen/qwen3-8b", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code


LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --tp 2 --dp 2 --port 8089 --max_total_token_num 200000 --enable_cpu_cache  --cpu_cache_storage_size 128 --cpu_cache_token_page_size 128 --llm_kv_type int8kv

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"qwen/qwen3-8b", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code


LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH=128 python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --tp 2 --dp 2 --port 8089 --max_total_token_num 200000 --enable_cpu_cache  --cpu_cache_storage_size 64 --cpu_cache_token_page_size 128 --enable_disk_cache --disk_cache_storage_size 256 --disk_cache_dir /mtc/test/tmp/

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval --model local-completions --model_args '{"model":"qwen/qwen3-8b", "base_url":"http://localhost:8089/v1/completions", "max_length": 16384}' --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code

# 帮我写一段提示词，告诉AI单独一个一个的进行上述测试的启动服务，然后再执行评测脚本，将结果写入out.txt 中，注意需要标记启动的参数和结果信息。不要用health 接口去判断服务是否启动，直接探测端口是否处于listen状态即可, 执行评测命令的时候，需要用no_proxy 将本地local ip 排除。
# 不要写额外的脚本来启动服务，就是单独一个一个的按照上面的描述启动服务，然后再执行评测脚本，然后注意等待服务启动完成，可以20s检测一次其控制台输出，看是否启动完成，还是启动报错。
# 应该把server启动在后台，然后再去探测端口， 判断服务是否启动成功。最后需要总结下测试的结果。对于开启 cpu cache 和 disk cache的测试，需要跑两次，确认命中后的效率和精度。