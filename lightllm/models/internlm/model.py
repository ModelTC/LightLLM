import os
import json
import torch

from lightllm.models.internlm.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel


class InternlmTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = InternlmTransformerLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)
