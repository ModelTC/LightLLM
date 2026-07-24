# Compatibility re-export. Implementation lives in common.
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.fused_recurrent import *  # noqa: F401,F403
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)

__all__ = ["fused_recurrent_gated_delta_rule"]
