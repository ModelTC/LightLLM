import torch
from lightllm.utils.device_utils import get_target_device
from ..pre_layer_infer import PreLayerInfer


class PreLayerInferTpl(PreLayerInfer):
    """ """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = 1e-5
        self.target_device = get_target_device()
        return

    def _norm(self, input, infer_state, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")
