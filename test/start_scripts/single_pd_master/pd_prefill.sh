# PD prefill mode for deepseek R1 (DP+EP) on H200
# host: the host of the current node
# pd_master_ip: the ip of the pd master
# sh pd_prefill.sh <host> <pd_master_ip>
export host=$1
export pd_master_ip=$2
nvidia-cuda-mps-control -d 
MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server \
--model_dir /path/DeepSeek-R1 \
--run_mode "prefill" \
--tp 8 \
--dp 8 \
--host $host \
--port 8019 \
--nccl_port 2732 \
--enable_fa3 \
--disable_cudagraph \
--pd_master_ip $pd_master_ip \
--pd_master_port 60011 
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_prefill_microbatch_overlap