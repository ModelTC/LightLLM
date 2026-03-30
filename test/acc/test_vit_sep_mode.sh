# 启动 config_server
python -m lightllm.server.api_server \
--run_mode config_server \
--config_server_host 0.0.0.0 \
--config_server_port 8090 \
--config_server_visual_redis_port 6000


# 启动 visual_only 模式的推理服务, --visual_rpyc_port 8091 visual_only 模式需要设置这个参数，提供给其他服务调用本地视觉推理接口
# --config_server_host 应该是启动config_server 服务的ip, 这里因为测试是在同一台机器上，所以是0.0.0.0。
CUDA_VISIBLE_DEVICES=0 python -m lightllm.server.api_server \
--run_mode visual_only \
--host 0.0.0.0 \
--config_server_host 0.0.0.0 \
--config_server_port 8090 \
--config_server_visual_redis_port 6000 \
--model_dir /mtc/models/Qwen3-VL-8B-Instruct \
--visual_dp 1 \
--visual_tp 1 \
--afs_image_embed_dir /mtc/afs/vit_embed_dir \
--afs_embed_capacity 250000 \
--visual_rpyc_port 8091

