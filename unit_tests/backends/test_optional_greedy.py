from lightllm.backends.operators import OperatorBackend


class ExistingBackend(OperatorBackend):
    def resolve(self, name):
        raise NotImplementedError

    def selection_module(self, family):
        raise NotImplementedError

    def select_weight_forward(self, op):
        raise NotImplementedError

    def initialize_linear_attention(self):
        return None


def test_existing_backends_keep_the_original_sampler():
    backend = ExistingBackend()
    values = [object() for _ in range(7)]
    assert backend.greedy_sample(*values) is None
