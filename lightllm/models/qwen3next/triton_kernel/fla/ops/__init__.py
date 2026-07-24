# Compatibility re-export. Implementation lives in common.
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

__all__ = [
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
