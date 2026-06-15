sudo docker run -d \
  --name junyi-lightllm \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user $(id -u):$(id -g) \
  -e HOME=/workspace \
  -v /data/nvme0/chenjunyi:/workspace \
  -v /mtc:/mtc \
  -w /workspace \
  junyi-lightllm:latest \
  sleep infinity