import torch
from ..pre_layer_infer import PreLayerInfer


class PreLayerInferTpl(PreLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.eps_ = 1e-5
        return

    def _norm(self, input, infer_state, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")
