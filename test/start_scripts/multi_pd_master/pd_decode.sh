# decode
# host: the host of the decode server
# config_server_host: the host of the config server
# sh decode.sh <host> <config_server_host>
export host=$1
export config_server_host=$2
nvidia-cuda-mps-control -d
MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server \
--model_dir /path/DeepSeek-R1 \
--run_mode "decode" \
--host $host \
--port 8121 \
--nccl_port 12322 \
--tp 8 \
--dp 8 \
--llm_prefill_att_backend fa3 --llm_decode_att_backend fa3
--config_server_host $config_server_host \
--config_server_port 60088
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_decode_microbatch_overlap