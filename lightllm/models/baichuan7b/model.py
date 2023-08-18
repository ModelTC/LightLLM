import os
import json
import torch

from .layer_weights.transformer_layer_weight import BaiChuan7bTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel

class Baichuan7bTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = BaiChuan7bTransformerLayerWeight

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=""):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)
    

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        return 