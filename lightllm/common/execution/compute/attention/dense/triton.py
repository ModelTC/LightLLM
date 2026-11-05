from .backend import DenseAttentionBackend


class TritonAttentionBackend(DenseAttentionBackend):
    supports_auxiliary_states = False

    def __init__(self, phase):
        super().__init__("triton", phase)
