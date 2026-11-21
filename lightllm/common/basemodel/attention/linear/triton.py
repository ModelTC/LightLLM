from lightllm.common.basemodel.attention.linear.gdn import LinearAttBackend


class TritonLinearAttBackend(LinearAttBackend):
    """Native GDN state, Triton chunked prefill."""
