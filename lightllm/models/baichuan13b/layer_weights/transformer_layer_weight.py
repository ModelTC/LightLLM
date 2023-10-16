import torch
import math
import numpy as np

from lightllm.models.baichuan7b.layer_weights.transformer_layer_weight import BaiChuan7bTransformerLayerWeight
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight

class BaiChuan13bTransformerLayerWeight(BaiChuan7bTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return
    
    def init_static_params(self):
        return BloomTransformerLayerWeight.init_static_params(self)
    
    def _generate_alibi(self, n_head, dtype=torch.float16):
        return BloomTransformerLayerWeight._generate_alibi(self, n_head, dtype)
    
    def verify_load(self):
        super().verify_load()
        assert self.tp_alibi is not None, "load error"
        return


