# H200 multi node deepseek R1 ep mode node 1
# nccl_host: the ip of the nccl host
# sh multi_node_ep_node1.sh <nccl_host>
export nccl_host=$1
MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
--model_dir /path/DeepSeek-R1 \
--tp 16 \
--dp 16 \
--llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 \
--nnodes 2 \
--node_rank 1 \
--nccl_host $nccl_host \
--nccl_port 2732 
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_prefill_microbatch_overlap
#--enable_decode_microbatch_overlap