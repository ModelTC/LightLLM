import torch
from ..post_layer_infer import PostLayerInfer


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