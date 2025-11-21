# PD decode mode for deepseek R1 (DP+EP) on H200
# host: the host of the current node
# pd_master_ip: the ip of the pd master
# sh pd_decode.sh <host> <pd_master_ip>
export host=$1
export pd_master_ip=$2
nvidia-cuda-mps-control -d
MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server \
--model_dir /path/DeepSeek-R1 \
--run_mode "decode" \
--tp 8 \
--dp 8 \
--host $host \
--port 8121 \
--nccl_port 12322 \
--enable_fa3 \
--pd_master_ip $pd_master_ip \
--pd_master_port 60011 
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_decode_microbatch_overlap