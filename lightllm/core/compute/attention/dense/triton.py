from .backend import DenseAttentionBackend


class TritonAttentionBackend(DenseAttentionBackend):
    def __init__(self, phase):
        super().__init__("triton", phase)
