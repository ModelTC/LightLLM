import os
import json
import torch
from lightllm.models.minicpm.layer_weights.transformer_layer_weight import MiniCPMTransformerLayerWeight
from lightllm.models.minicpm.layer_weights.pre_and_post_layer_weight import MiniCPMPreAndPostLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel


class MiniCPMTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = MiniCPMTransformerLayerWeight
    pre_and_post_weight_class = MiniCPMPreAndPostLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)
    