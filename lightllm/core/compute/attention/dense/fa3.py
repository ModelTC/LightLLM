from .backend import DenseAttentionBackend


class FA3AttentionBackend(DenseAttentionBackend):
    supports_auxiliary_states = False

    def __init__(self, phase):
        super().__init__("fa3", phase)
