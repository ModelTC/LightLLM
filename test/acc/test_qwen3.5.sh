/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17

# first 测试基础功能
LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server \
--model_dir /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
--tp 2 \
--port 8089

# second
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval  --model local-completions \
--model_args '{"model":"Qwen/Qwen3.5-27B","base_url":"http://127.0.0.1:8089/v1/completions","tokenizer_backend":"huggingface","tokenizer":"qwen/Qwen3.5-0.8B","max_length":16384,"max_gen_toks":2048,"num_concurrent":8,"max_retries":3}' \
--tasks gsm8k  \
--batch_size 64

# 测试
LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server \
--model_dir /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
--tp 2 \
--port 8089 \
--linear_att_cache_size 10 \
--linear_att_hash_page_size 256 \
--linear_att_page_block_num 2 \
--max_total_token_num 200000

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval  --model local-completions \
--model_args '{"model":"Qwen/Qwen3.5-27B","base_url":"http://127.0.0.1:8089/v1/completions","tokenizer_backend":"huggingface","tokenizer":"qwen/Qwen3.5-0.8B","max_length":16384,"max_gen_toks":2048,"num_concurrent":8,"max_retries":3}' \
--tasks gsm8k  \
--batch_size 64
# 测试 cpu cache 与 linear att 的配合是否正常
LOADWORKER=18 python -m lightllm.server.api_server \
--model_dir /mtc/models/Qwen3.5-27B \
--tp 8 \
--port 8089 \
--linear_att_cache_size 10 \
--linear_att_hash_page_size 256 \
--linear_att_page_block_num 4 \
--max_total_token_num 60000 \
--enable_cpu_cache  \
--cpu_cache_storage_size 128

export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval  --model local-completions \
--model_args '{"model":"Qwen/Qwen3.5-27B","base_url":"http://127.0.0.1:8089/v1/completions","tokenizer_backend":"huggingface","tokenizer":"qwen/Qwen3.5-0.8B","max_length":16384,"max_gen_toks":2048,"num_concurrent":8,"max_retries":3}' \
--tasks gsm8k  \
--batch_size 64


# 帮我写一段提示词，告诉AI单独一个一个的进行上述测试的启动服务，然后再执行评测脚本，将结果写入out.txt 中，注意需要标记启动的参数和结果信息。不要用health 接口去判断服务是否启动，直接探测端口是否处于listen状态即可, 执行评测命令的时候，需要用no_proxy 将本地local ip 排除。
# 不要写额外的脚本来启动服务，就是单独一个一个的按照上面的描述启动服务，然后再执行评测脚本，然后注意等待服务启动完成，可以20s检测一次其控制台输出，看是否启动完成，还是启动报错。
# 应该把server启动在后台，然后再去探测端口， 判断服务是否启动成功。最后需要总结下测试的结果。