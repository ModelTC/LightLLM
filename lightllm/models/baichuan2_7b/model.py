import os
import json
import torch

from lightllm.models.baichuan2_7b.layer_weights.pre_and_post_layer_weight import Baichuan2_7bPreAndPostLayerWeight
from lightllm.models.baichuan7b.model import Baichuan7bTpPartModel
from lightllm.models.baichuan2_7b.layer_infer.transformer_layer_infer import Baichuan2_7bTransformerLayerInfer


class Baichuan2_7bTpPartModel(Baichuan7bTpPartModel):
    # weight class
    pre_and_post_weight_class = Baichuan2_7bPreAndPostLayerWeight

    # infer class
    transformer_layer_infer_class = Baichuan2_7bTransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)