 LIGHTLLM_USE_TRITON_FP8_SCALED_MM=1 LOADWORKER=4 python test/benchmark/static_inference/test_model.py --model_dir /data/Qwen3-Omni-30B-A3B-Instruct --quant_type vllm-fp8w8a8 --tp 1 --graph_max_batch_size 8 --max_total_token_num 40000 \
 --input_len 1024 \
 --output_len 128 \
 --torch_profile \
 --nccl_port 29988 \
 --data_type bf16