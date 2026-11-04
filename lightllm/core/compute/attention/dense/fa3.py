from .backend import DenseAttentionBackend


class FA3AttentionBackend(DenseAttentionBackend):
    def __init__(self, phase):
        super().__init__("fa3", phase)
