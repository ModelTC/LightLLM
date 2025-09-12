CUDA_VISIBLE_DEVICES=0 python \
    -m lightllm.server.api_server \
    --model_dir /mnt/youwei-data/zhuohang/model/opendatalab/MinerU2.0-2505-0.9B/ \
    --host 0.0.0.0 \
    --port 8081 \
    --enable_multimodal

curl http://127.0.0.1:8081/generate \
    -H "Content-Type: application/json" \
    -d '{
          "inputs": "What is AI?",
          "parameters":{
            "max_new_tokens":128,
            "frequency_penalty":1
          }
        }'

curl http://127.0.0.1:8081/health

