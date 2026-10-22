from lightllm.common.basemodel.attention.linear.gdn import GDNLinearAttBackend
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import chunk_gated_delta_rule


class TritonLinearAttBackend(GDNLinearAttBackend):
    def get_prefill_kernel(self):
        return chunk_gated_delta_rule
