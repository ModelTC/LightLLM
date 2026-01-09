from lightllm.models.registry import ModelRegistry
from lightllm.models.internlm.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel


@ModelRegistry("internlm")
class InternlmTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = InternlmTransformerLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)
