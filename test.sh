# export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
LIGHTLLM_USE_TRITON_FP8_SCALED_MM=1 LOADWORKER=1 python -m lightllm.server.api_server --model_dir /data/asr_model/SenseASR-Omni-30B-A3B-Instruct --quant_type vllm-fp8w8a8 --tp 1 --graph_max_batch_size 8 --enable_multimodal --enable_multimodal_audio --max_total_token_num 40000 --port 8100 --host 0.0.0.0
