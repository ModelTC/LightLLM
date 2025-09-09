export CUDA_VISIBLE_DEVICES=2,3
export DISABLE_CHECK_MAX_LEN_INFER=1


python -m lightllm.server.api_server --model_dir /data/nvme0/models/llama3-8b-instruct  \
                                     --host 0.0.0.0                 \
                                     --port 8888                   \
                                     --tp 1                        \
                                     --nccl_port 65535                \
				                     --data_type bf16   \
				                     --trust_remote_code  \
			                         --graph_max_batch_size 32 \
				                     --output_constraint_mode pre3