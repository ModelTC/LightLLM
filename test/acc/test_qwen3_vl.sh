# first
LOADWORKER=18 CUDA_VISIBLE_DEVICES=6,7 python -m lightllm.server.api_server --model_dir /mtc/models/Qwen3-VL-8B-Instruct --tp 2 --port 8089

# second
python test_vlm_models.py