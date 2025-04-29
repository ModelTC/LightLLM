import torch
from typing import final
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import Qwen3TransformerLayerInfer
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class Qwen3TpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = Qwen3TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Qwen3TransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
