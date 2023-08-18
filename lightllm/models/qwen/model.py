import os
import json
import torch

from .layer_infer.transformer_layer_infer import QwenTransformerLayerInfer
from .layer_weights.pre_and_post_layer_weight import QwenPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import QwenTransformerLayerWeight

from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.build_utils import repair_config


class QWenTpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = QwenPreAndPostLayerWeight
    transformer_weight_class = QwenTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = QwenTransformerLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=""):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)
    
    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        repair_config(self.config, same_names=["ffn_hidden_size", "intermediate_size"])
        repair_config(self.config, same_names=["rms_norm_eps", "layer_norm_epsilon"])
        return 
    