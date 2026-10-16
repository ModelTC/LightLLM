from lightllm.common.basemodel.attention.linear.gdn import LinearAttBackend

from lightllm.backends import get_operator_backend

_operator_backend = get_operator_backend()

chunk_gated_delta_rule = _operator_backend.resolve("common.basemodel.linear_att.fla.ops.chunk_gated_delta_rule")


class TritonLinearAttBackend(LinearAttBackend):
    def get_prefill_kernel(self):
        return chunk_gated_delta_rule
