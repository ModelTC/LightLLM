import torch
from ..pre_layer_infer import PreLayerInfer


class PreLayerInferTpl(PreLayerInfer):
    """
    """
    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = 1e-5
        self.vob_start_id_ = -1
        self.vob_end_id_ = -1
        return
    
    def _norm(self, input, infer_state, layer_weight)->torch.Tensor:
        raise Exception("need to impl")
