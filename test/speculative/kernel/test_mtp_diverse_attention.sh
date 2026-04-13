#!/bin/bash
export PYTHONPATH=/data/nvme0/chenjunyi/project/lightllm
cd /data/nvme0/chenjunyi/project/lightllm
python unit_tests/common/basemodel/triton_kernel/att/decode_att/gqa/mtp_diverse/test_mtp_diverse.py
