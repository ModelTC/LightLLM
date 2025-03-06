# pd start
python -m lightllm.server.api_server --model_dir /dev/shm/llama2-7b-chat --run_mode "pd_master" --host `hostname -i` --port 60011

nvidia-cuda-mps-control -d 
CUDA_VISIBLE_DEVICES=0,1,2,3 KV_TRANS_USE_P2P=1 LOADWORKER=1 python -m lightllm.server.api_server --model_dir /dev/shm/llama2-7b-chat \
--run_mode "prefill" \
--host `hostname -i` \
--port 8019 \
--tp 4 \
--nccl_port 2732 \
--max_total_token_num 400000 \
--tokenizer_mode fast \
--pd_master_ip `hostname -i` \
--pd_master_port 60011 \
--use_dynamic_prompt_cache \
--max_req_total_len 16000 \
--running_max_req_size 128 \
--disable_cudagraph

nvidia-cuda-mps-control -d
CUDA_VISIBLE_DEVICES=4,5,6,7 KV_TRANS_USE_P2P=1 LOADWORKER=10 python -m lightllm.server.api_server --model_dir /dev/shm/llama2-7b-chat \
--run_mode "decode" \
--host `hostname -i` \
--port 8121 \
--nccl_port 12322 \
--tp 4 \
--max_total_token_num 400000 \
--graph_max_len_in_batch 2048 \
--graph_max_batch_size 16 \
--tokenizer_mode fast \
--pd_master_ip `hostname -i` \
--pd_master_port 60011 \
--use_dynamic_prompt_cache

# pd start1
python -m lightllm.server.api_server --model_dir /dev/shm/llama2-7b-chat --run_mode "pd_master" --host `hostname -i` --port 60011

nvidia-cuda-mps-control -d 
CUDA_VISIBLE_DEVICES=0 KV_TRANS_USE_P2P=1 LOADWORKER=1 python -m lightllm.server.api_server --model_dir /dev/shm/llama2-7b-chat \
--run_mode "prefill" \
--host `hostname -i` \
--port 8019 \
--tp 1 \
--nccl_port 2732 \
--max_total_token_num 40000 \
--tokenizer_mode fast \
--pd_master_ip `hostname -i` \
--pd_master_port 60011 \
--use_dynamic_prompt_cache \
--max_req_total_len 16000 \
--running_max_req_size 128 \
--disable_cudagraph

nvidia-cuda-mps-control -d
CUDA_VISIBLE_DEVICES=1 KV_TRANS_USE_P2P=1 LOADWORKER=10 python -m lightllm.server.api_server --model_dir /dev/shm/llama2-7b-chat \
--run_mode "decode" \
--host `hostname -i` \
--port 8121 \
--nccl_port 12322 \
--tp 1 \
--max_total_token_num 40000 \
--graph_max_len_in_batch 2048 \
--graph_max_batch_size 16 \
--tokenizer_mode fast \
--pd_master_ip `hostname -i` \
--pd_master_port 60011 \
--use_dynamic_prompt_cache


# normal start
LOADWORKER=8 python -m lightllm.server.api_server --port 8018 --model_dir /dev/shm/llama2-7b-chat --tp 2 --graph_max_batch_size 16 --use_dynamic_prompt_cache


