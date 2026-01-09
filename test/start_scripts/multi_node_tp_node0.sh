# H200/H100 multi node deepseek R1 tp mode node 0
# nccl_host: the ip of the nccl host
# sh multi_node_tp_node0.sh <nccl_host>
export nccl_host=$1
LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
--model_dir /path/DeepSeek-R1 \
--tp 16 \
--llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 \
--nnodes 2 \
--node_rank 0 \
--nccl_host $nccl_host \
--nccl_port 2732