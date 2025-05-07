import os
import json
import torch
from lightllm.models.registry import ModelRegistry
from lightllm.models.phi3.layer_weights.transformer_layer_weight import Phi3TransformerLayerWeight
from lightllm.models.phi3.layer_infer.transformer_layer_infer import Phi3TransformerLayerInfer
from lightllm.models.llama.model import LlamaTpPartModel


@ModelRegistry("phi3")
class Phi3TpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = Phi3TransformerLayerWeight

    transformer_layer_infer_class = Phi3TransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
