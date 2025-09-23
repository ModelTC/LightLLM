# PD decode mode for deepseek R1 (DP+EP) on H200
# host: the host of the current node
# pd_master_ip: the ip of the pd master
# sh pd_decode.sh <host> <pd_master_ip>
export host=$1
export pd_master_ip=$2

export UCX_NET_DEVICES=$(ibv_devinfo | grep 'hca_id:' | grep -v -E 'mlx5_8|mlx5_9' | awk '{print $2":1"}' | paste -sd, -)
export UCX_LOG_LEVEL=info
export UCX_TLS=rc,cuda,gdr_copy

nvidia-cuda-mps-control -d
MOE_MODE=EP KV_TRANS_USE_P2P=1 LOADWORKER=18 python -m lightllm.server.api_server \
--model_dir /path/DeepSeek-R1 \
--run_mode "nixl_decode" \
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