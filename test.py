import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

data_o = torch.zeros((128 * 1024), dtype=torch.int32, device="cuda")
in_data = list(range(0, 1000))
in_datas = [list(range(0, 1000)) for _ in range(100)]
import time

cpu_tensor = torch.zeros((128 * 1024), dtype=torch.int32, device="cpu", pin_memory=False)
pin_mem_tensor = torch.zeros((128 * 1024), dtype=torch.int32, device="cpu", pin_memory=True)
gpu_tensor = torch.zeros((128 * 1024), dtype=torch.int32, device="cuda")

a = torch.arange(0, 10).cuda()
b = torch.arange(0, 10).cuda()

print((gpu_tensor == 1).dtype)
# max_data = tmp.max()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile/profile.file"),
) as prof:
    # gpu_tensor[:] = pin_mem_tensor
    # torch.cuda.current_stream().synchronize()
    # a = torch.tensor([1,3, 7], device="cuda")
    # gpu_tensor[:] = pin_mem_tensor
    for _ in range(100):
        cpu_tensor.cuda(non_blocking=True)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16), flush=True)


# CUDA_VISIBLE_DEVICES=4,5,6,7 LOADWORKER=16 python -m lightllm.server.api_server --port 8019 --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --tp 4 --dp 1 --diverse_mode | tee log.txt

# CUDA_VISIBLE_DEVICES=4,5,6,7 LOADWORKER=16 python -m lightllm.server.api_server --port 8019 --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --tp 1 --dp 1 --diverse_mode | tee log.txt 你试试这个


# CUDA_VISIBLE_DEVICES=4,5,6,7 LOADWORKER=16 python -m lightllm.server.api_server --port 8019 --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --tp 4 --dp 1 --output_constraint_mode xgrammar | tee log.txt


pin_mem_tensor.numpy()[0:10] = list(range(10))

print("ok")

# CUDA_VISIBLE_DEVICES=4,5,6,7 LOADWORKER=16 python -m lightllm.server.api_server --port 8019 --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --tp 8 --dp 8 | tee log.txt 你试试这个




# LOADWORKER=16 python -m lightllm.server.api_server --model_dir /mtc/DeepSeek-R1 --mtp_draft_model_dir /mtc/DeepSeek-R1-NextN/ --mtp_mode deepseekv3 --mtp_step 1 --enable_fa3 --graph_max_batch_size 64 --tp 8 --port 15001 | tee debug.txt


# LOADWORKER=16 python -m lightllm.server.api_server --model_dir /mtc/DeepSeek-R1 --mtp_draft_model_dir /mtc/DeepSeek-R1-NextN/ --mtp_mode deepseekv3 --mtp_step 1 --enable_fa3 --graph_max_batch_size 64 --tp 8 --port 15001 | tee debug.txt


# LOADWORKER=16 python -m lightllm.server.api_server --model_dir /mtc/DeepSeek-R1 --mtp_draft_model_dir /mtc/DeepSeek-R1-NextN/ --mtp_mode deepseekv3 --mtp_step 1 --enable_fa3 --graph_max_batch_size 64 --tp 8 --port 15001 | tee debug.txt


# MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8019 \
# --model_dir /mtc/DeepSeek-R1 \
# --tp 8 \
# --dp 8 \
# --enable_fa3 \
# --enable_prefill_microbatch_overlap \
# --enable_decode_microbatch_overlap \
# --mem_fraction 0.8 \
# --batch_max_tokens 4096


# MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8019 \
# --model_dir /mtc/DeepSeek-R1 \
# --tp 8 \
# --dp 8 \
# --enable_fa3 \
# --mem_fraction 0.8 \
# --batch_max_tokens 4096 \
# --mtp_draft_model_dir /mtc/DeepSeek-R1-NextN/ --mtp_mode deepseekv3 --mtp_step 1



# CUDA_VISIBLE_DEVICES=0,1 LOADWORKER=18 python -m lightllm.server.api_server --port 8019 \
# --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ \
# --tp 4 \
# --enable_fa3 \
# --nnodes 2 \
# --node_rank 0 \
# --nccl_host 127.0.0.1 \
# --nccl_port 2732

# CUDA_VISIBLE_DEVICES=2,3 LOADWORKER=18 python -m lightllm.server.api_server --port 8021 \
# --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ \
# --tp 4 \
# --enable_fa3 \
# --nnodes 2 \
# --node_rank 1 \
# --nccl_host 127.0.0.1 \
# --nccl_port 2732


# python -m lightllm.server.api_server --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --run_mode "pd_master" --host 127.0.0.1 --port 60011


# CUDA_VISIBLE_DEVICES=0,1 MOE_MODE=EP KV_TRANS_USE_P2P=1 LOADWORKER=18 python -m lightllm.server.api_server \
# --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ \
# --run_mode "prefill" \
# --tp 2 \
# --dp 1 \
# --host 0.0.0.0 \
# --port 8019 \
# --nccl_port 2732 \
# --enable_fa3 \
# --disable_cudagraph \
# --pd_master_ip 127.0.0.1 \
# --pd_master_port 60011 


# CUDA_VISIBLE_DEVICES=2,3 MOE_MODE=EP KV_TRANS_USE_P2P=1 LOADWORKER=18 python -m lightllm.server.api_server \
# --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ \
# --run_mode "decode" \
# --tp 2 \
# --dp 1 \
# --host 0.0.0.0 \
# --port 8121 \
# --nccl_port 27321 \
# --enable_fa3 \
# --pd_master_ip 127.0.0.1 \
# --pd_master_port 60011 


# CUDA_VISIBLE_DEVICES=0,1 LOADWORKER=16 python -m lightllm.server.api_server --port 8019 --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --tp 4 --dp 4 --nccl_port 27321  --node_rank 0 --nnodes 2 | tee log.txt

# CUDA_VISIBLE_DEVICES=2,3 LOADWORKER=16 python -m lightllm.server.api_server --port 8011 --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --tp 4 --dp 4 --nccl_port 27321  --node_rank 1 --nnodes 2


# LOADWORKER=16 python -m lightllm.server.api_server --port 8019 --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --tp 8 --dp 1 --enable_fa3



# lightllm                                               v1.0.1-4209c8c4-deepep


# docker run -itd  --gpus all --privileged=true --shm-size=128G -v /mtc:/mtc --name wzj 44feca8a0c86


# CUDA_VISIBLE_DEVICES=2,3 LOADWORKER=16 python -m lightllm.server.api_server --port 8011 --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --tp 4 --dp 4 --nccl_port 27321  --node_rank 1 --nnodes 2


# LOADWORKER=16 python -m lightllm.server.api_server --port 8011 --model_dir /mtc/niushengxiao/Qwen/Qwen2.5-14B-Instruct/ --tp 1 --dp 1 --nccl_port 27321  --enable_cpu_cache