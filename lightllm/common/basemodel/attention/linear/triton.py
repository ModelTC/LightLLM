from lightllm.common.basemodel.attention.linear.gdn import LinearAttBackend
from lightllm.common.attention.kernels.gdn.third_party_fla.ops.chunk import chunk_gated_delta_rule


class TritonLinearAttBackend(LinearAttBackend):
    def get_prefill_kernel(self):
        return chunk_gated_delta_rule
