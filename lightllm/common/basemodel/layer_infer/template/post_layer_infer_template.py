import torch
from ..post_layer_infer import PostLayerInfer
from typing import Tuple

class PostLayerInferTpl(PostLayerInfer):
    """
    """
    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = 1e-5
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        return
    
    def _norm(self, input, infer_state, layer_weight)->torch.Tensor:
        raise Exception("need to impl")
    

    def _slice_get_last_input(self, input, infer_state)->Tuple[torch.Tensor, int]:
        raise Exception("need to impl")