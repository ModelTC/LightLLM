# Compatibility re-export. Implementation lives in common.
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_spec import *  # noqa: F401,F403
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_spec import causal_conv1d_update

__all__ = ["causal_conv1d_update"]
