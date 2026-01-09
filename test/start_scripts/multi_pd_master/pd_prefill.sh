# prefill 
# host: the host of the prefill server
# config_server_host: the host of the config server
# sh pd_prefill.sh <host> <config_server_host>
export host=$1
export config_server_host=$2
nvidia-cuda-mps-control -d
MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server \
--model_dir /path/DeepSeek-R1 \
--run_mode "prefill" \
--host $host \
--port 8019 \
--tp 8 \
--dp 8 \
--nccl_port 2732 \
--llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 \
--disable_cudagraph \
--config_server_host $config_server_host \
--config_server_port 60088
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_prefill_microbatch_overlap