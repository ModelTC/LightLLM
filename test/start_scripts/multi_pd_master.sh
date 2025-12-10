# 多 pd_master 节点部署示例
python -m lightllm.server.api_server --run_mode "config_server" --config_server_host 10.120.114.74 --config_server_port 60088

python -m lightllm.server.api_server --model_dir /mtc/models/DeepSeek-V2-Lite-Chat --run_mode "pd_master" --host 10.120.114.74 --port 60011 --config_server_host 10.120.114.74 --config_server_port 60088

python -m lightllm.server.api_server --model_dir /mtc/models/DeepSeek-V2-Lite-Chat --run_mode "pd_master" --host 10.120.114.74 --port 60012 --config_server_host 10.120.114.74 --config_server_port 60088

nvidia-cuda-mps-control -d 
CUDA_VISIBLE_DEVICES=0 LOADWORKER=1 python -m lightllm.server.api_server --model_dir /mtc/models/DeepSeek-V2-Lite-Chat \
--run_mode "prefill" \
--host 10.120.178.74 \
--port 8019 \
--tp 1 \
--nccl_port 2732 \
--max_total_token_num 40000 \
--tokenizer_mode fast \
--max_req_total_len 16000 \
--running_max_req_size 128 \
--disable_cudagraph \
--config_server_host 10.120.114.74 \
--config_server_port 60088

CUDA_VISIBLE_DEVICES=1 LOADWORKER=10 python -m lightllm.server.api_server --model_dir /mtc/models/DeepSeek-V2-Lite-Chat \
--run_mode "decode" \
--host 10.120.178.74 \
--port 8121 \
--nccl_port 12322 \
--tp 1 \
--max_total_token_num 40000 \
--graph_max_len_in_batch 2048 \
--graph_max_batch_size 16 \
--tokenizer_mode fast \
--config_server_host 10.120.114.74 \
--config_server_port 60088 