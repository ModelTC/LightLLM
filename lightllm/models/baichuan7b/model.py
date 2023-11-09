import os
import json
import torch

from .layer_weights.transformer_layer_weight import BaiChuan7bTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel

class Baichuan7bTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = BaiChuan7bTransformerLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)
    