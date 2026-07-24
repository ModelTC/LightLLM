# Compatibility re-export. Implementation lives in common.
from lightllm.common.basemodel.triton_kernel.linear_att.gdn_decode_pack import *  # noqa: F401,F403
from lightllm.common.basemodel.triton_kernel.linear_att.gdn_decode_pack import (
    conv_pack_gdn_decode_inputs,
)

__all__ = ["conv_pack_gdn_decode_inputs"]
