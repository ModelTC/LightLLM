docker run -d \
  --name junyi-lightllm \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user $(id -u):$(id -g) \
  -v /mtc:/mtc \
  -w /mtc/chenjunyi1/project/LightLLM \
  junyi-lightllm:latest \
  sleep infinity