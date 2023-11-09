import os
import json
import torch

from .layer_weights.transformer_layer_weight import YiTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel

class YiTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = YiTransformerLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)
    