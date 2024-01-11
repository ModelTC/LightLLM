import os
import json
import torch

from lightllm.models.internlm2.layer_weights.transformer_layer_weight import Internlm2TransformerLayerWeight
from lightllm.models.internlm2.layer_weights.pre_and_post_layer_weight import Internlm2PreAndPostLayerWeight 
from lightllm.models.internlm.model import InternlmTpPartModel


class Internlm2TpPartModel(InternlmTpPartModel):
    # weight class
    pre_and_post_weight_class = Internlm2PreAndPostLayerWeight 
    transformer_weight_class = Internlm2TransformerLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)
    
