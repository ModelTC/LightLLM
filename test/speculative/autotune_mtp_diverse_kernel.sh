# Autotune after set the corresponding static parameters
export PYTHONPATH=/data/nvme0/chenjunyi/project/lightllm 
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=2 
python /data/nvme0/chenjunyi/project/lightllm/lightllm/common/basemodel/triton_kernel/att/decode_att/gqa/mtp_diverse/stage1_single_token.py