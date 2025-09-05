CUDA_VISIBLE_DEVICES=0 python \
    -m lightllm.server.api_server \
    --model_dir /mnt/youwei-data/zhuohang/model/opendatalab/MinerU2.0-2505-0.9B/ \
    --host 0.0.0.0 \
    --port 8081 \
    --enable_multimodal
