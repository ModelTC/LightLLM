"""注意力资源的 State 命名工具。"""

from __future__ import annotations

import torch

MAX_DRAFT_MODELS = 36
MODEL_ROLES = ("main", *(f"draft{i}" for i in range(MAX_DRAFT_MODELS)))


# ========================================================================
# 一、State 命名与成员管理
# ========================================================================

def dtype_name(dtype: torch.dtype):
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
