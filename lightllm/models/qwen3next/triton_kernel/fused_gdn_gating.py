# Compatibility re-export. Implementation lives in common.
from lightllm.common.basemodel.triton_kernel.linear_att.fused_gdn_gating import *  # noqa: F401,F403
from lightllm.common.basemodel.triton_kernel.linear_att.fused_gdn_gating import fused_gdn_gating

__all__ = ["fused_gdn_gating"]
