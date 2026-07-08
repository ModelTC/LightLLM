MODEL_DIR=/mtc/models/qwen3-32b 
DRAFT_MODEL_DIR=/mtc/models/qwen3-32b-eagle3

PATH=/data/nvme0/chenjunyi/miniconda3/envs/lightllm/bin:$PATH

LOADWORKER=18 /data/nvme0/chenjunyi/miniconda3/envs/lightllm/bin/python -m lightllm.server.api_server --port 8088 \
--tp 2 \
--model_dir ${MODEL_DIR} \
--disable_dynamic_prompt_cache \
--graph_grow_step_size 1 \
--llm_decode_att_backend triton