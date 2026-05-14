$pd_master_ip 为本机的ip地址, 测试的时候，自己修改为对应的ip地址

# 启动pd_master节点
# 测试前关闭代理
export http_proxy=
export https_proxy=
python -m lightllm.server.api_server --model_dir /mtc/models/qwen3-8b --run_mode "pd_master" --host $pd_master_ip --port 8089

# 启动prefill 节点
$host 为本机的ip地址, 测试的时候，自己修改为对应的ip地址
$pd_master_ip 为pd_master的ip地址, 测试的时候，自己修改为对应的ip地址,在测试的时候为本机ip地址
# 测试前关闭代理
export http_proxy=
export https_proxy=
LOADWORKER=18 CUDA_VISIBLE_DEVICES=0,1 python -m lightllm.server.api_server \
--model_dir /mtc/models/qwen3-8b \
--run_mode "prefill" \
--tp 2 \
--dp 1 \
--host $host \
--port 8001 \
--disable_cudagraph \
--pd_master_ip $pd_master_ip \
--pd_master_port 8089

# 启动 decode 节点
# 测试前关闭代理
export http_proxy=
export https_proxy=
$host 为本机的ip地址, 测试的时候，自己修改为对应的ip地址
$pd_master_ip 为pd_master的ip地址, 测试的时候，自己修改为对应的ip地址,在测试的时候为本机ip地址
LOADWORKER=18 CUDA_VISIBLE_DEVICES=2,3 python -m lightllm.server.api_server \
--model_dir /mtc/models/qwen3-8b \
--run_mode "decode" \
--tp 2 \
--dp 1 \
--host $host \
--port 8002 \
--pd_master_ip $pd_master_ip \
--pd_master_port 8089

# 等待 prefill 和 decode 节点启动完成，并连上 pd master以后，执行测试脚本
# 测试前关闭代理
export http_proxy=
export https_proxy=
$pd_master_ip 为pd_master的ip地址, 测试的时候，自己修改为对应的ip地址
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 lm_eval \
--model local-completions --model_args \
'{"model":"qwen/qwen3-8b", "base_url":"http://$pd_master_ip:8089/v1/completions", "max_length": 16384, "tokenized_requests": false}' \
--tasks gsm8k --batch_size 500 --confirm_run_unsafe_code

# 1. 按顺序在不同的cmd中启动上面的程序，然后再执行评测脚本，将结果写入out.txt 中，注意需要标记启动的参数和结果信息。 
# 2. 执行评测命令的时候，需要用no_proxy 将本地local ip 排除。
# 3. 不要写额外的脚本来启动服务，就是单独一个一个的按照上面的描述启动服务，然后再执行评测脚本，然后注意等待服务启动完成，可以20s检测一次其控制台输出，看是否启动完成，还是启动报错。
# 4. 最后需要总结下测试的结果，并将结果输出到对话中。
# 5. 如果启动过程中出现错误，需要记录错误信息，并输出到对话中。
# 6. 测试完成后，关闭所有启动的进程。
# 7. lm_eval 的评测命令有时候需要利用代理去下载一些缓存，所以可以先不关闭代码，跑一次lm_eval对应的命令，等cache下载好了以后，再关闭代理，跑第二次评测命令。