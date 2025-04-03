# 单机 deepseek V3 ep 运行模式启动示例, 启动参数中的tp含义发生了变化，代表使用的所有卡数量，并不是tp推理。
# max_total_token_num 可以按照实际场景调节。
MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8088 --model_dir /dev/shm/DeepSeek-R1 \ 
--tp 8 \
--dp 8 \
--max_total_token_num 200000 \
--graph_max_batch_size 64 \
--batch_max_tokens 8192 \
--enable_flashinfer_prefill \
--enable_flashinfer_decode  \
--enable_prefill_microbatch_overlap \
--disable_aggressive_schedule

# H800 双机 deepseek V3 ep 运行模式启动实列
# 启动命令中的 nccl_host 和 nccl_port 两个节点的必须一致，一般nccl_host设置为 node 0的ip。
# max_total_token_num 最佳设置需要按照使用场景和显存情况配置。
# 启动后两个节点的8088端口都可以接收访问的请求
# node 0
MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8088 --model_dir /dev/shm/DeepSeek-R1 \ 
--tp 16 \
--dp 16 \
--max_total_token_num 200000 \
--graph_max_batch_size 64 \
--batch_max_tokens 8192 \
--enable_flashinfer_prefill \
--enable_flashinfer_decode  \
--enable_prefill_microbatch_overlap \
--nnodes 2 \
--node_rank 0 \
--nccl_host <node_0_ip> \
--nccl_port 2732
# node 1
MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8088 --model_dir /dev/shm/DeepSeek-R1 \ 
--tp 16 \
--dp 16 \
--max_total_token_num 200000 \
--graph_max_batch_size 64 \
--batch_max_tokens 8192 \
--enable_flashinfer_prefill \
--enable_flashinfer_decode  \
--enable_prefill_microbatch_overlap \
--nnodes 2 \
--node_rank 1 \
--nccl_host <node_0_ip> \
--nccl_port 2732

# pd 分离启动示列， 单机 做 P 和 D， 也支持多机组成的D和单机的P混合。
# 目前 P D 分离的 PD master可能存在并发处理问题，还需提升。

# pd master 启动
python -m lightllm.server.api_server --model_dir /dev/shm/DeepSeek-R1 --run_mode "pd_master" --host `hostname -i` --port 60011

# p 启动
nvidia-cuda-mps-control -d 
MOE_MODE=EP KV_TRANS_USE_P2P=1 LOADWORKER=18 python -m lightllm.server.api_server --model_dir /dev/shm/DeepSeek-R1 \
--run_mode "prefill" \
--tp 8 \
--dp 8 \
--host `hostname -i` \
--port 8019 \
--nccl_port 2732 \
--max_total_token_num 200000 \
--batch_max_tokens 8192 \
--enable_flashinfer_prefill \
--enable_flashinfer_decode  \
--enable_prefill_microbatch_overlap \
--use_dynamic_prompt_cache \
--disable_cudagraph \
--pd_master_ip <pd_master_ip> \
--pd_master_port 60011

# d 启动
nvidia-cuda-mps-control -d
MOE_MODE=EP KV_TRANS_USE_P2P=1 LOADWORKER=18 python -m lightllm.server.api_server --model_dir /dev/shm/DeepSeek-R1 \
--run_mode "decode" \
--tp 8 \
--dp 8 \
--host `hostname -i` \
--port 8121 \
--nccl_port 12322 \
--max_total_token_num 200000 \
--graph_max_batch_size 64 \
--enable_flashinfer_prefill \
--enable_flashinfer_decode  \
--enable_prefill_microbatch_overlap \
--use_dynamic_prompt_cache \
--pd_master_ip <pd_master_ip> \
--pd_master_port 60011

