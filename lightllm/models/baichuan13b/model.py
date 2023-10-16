import os
import json
import torch

from .layer_weights.transformer_layer_weight import BaiChuan13bTransformerLayerWeight
from .layer_infer.transformer_layer_infer import Baichuan13bTransformerLayerInfer
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.model import LlamaTpPartModel


class Baichuan13bTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = BaiChuan13bTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Baichuan13bTransformerLayerInfer

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num,
                 load_way="HF", mode=[], weight_dict=None, finetune_config=None):
        super().__init__(
            tp_rank,
            world_size,
            weight_dir,
            max_total_token_num,
            load_way,
            mode,
            weight_dict,
            finetune_config)

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        return

    def _verify_params(self):
        assert self.load_way == "HF", "llama only support HF format to load Now!"
