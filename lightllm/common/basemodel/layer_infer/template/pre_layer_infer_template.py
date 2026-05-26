import torch
from lightllm.platform import get_backend
from ..pre_layer_infer import PreLayerInfer


class PreLayerInferTpl(PreLayerInfer):
    """ """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = 1e-5
        platform_backend = get_backend()
        self.target_device = platform_backend.runtime.target_device()
        return

    def _norm(self, input, infer_state, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")
