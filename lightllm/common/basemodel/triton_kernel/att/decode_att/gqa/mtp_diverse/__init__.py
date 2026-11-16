"""
MTP Diverse Attention Module

MTP (Multi-Token Prediction) Diverse Attention 的实现。
"""
from lightllm.common.attention.backends.dense.triton.kernels.decode.gqa.mtp_diverse.mtp_diverse_attn import (
    token_decode_attention_mtp_diverse_single_token,
)
from lightllm.common.attention.backends.dense.triton.kernels.decode.gqa.mtp_diverse.stage1_single_token import (
    mtp_diverse_stage1_single_token,
)
from lightllm.common.attention.backends.dense.triton.kernels.decode.gqa.mtp_diverse.stage2_single_token import (
    mtp_diverse_stage2_single_token,
)

__all__ = [
    "token_decode_attention_mtp_diverse_single_token",
    "mtp_diverse_stage1_single_token",
    "mtp_diverse_stage2_single_token",
]
