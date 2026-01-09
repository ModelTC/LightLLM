from lightllm.models.registry import ModelRegistry
from lightllm.models.stablelm.layer_infer.transformer_layer_infer import StablelmTransformerLayerInfer
from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer
from lightllm.models.stablelm.layer_weights.pre_and_post_layer_weight import StableLMPreAndPostLayerWeight
from lightllm.models.stablelm.layer_weights.transformer_layer_weight import StablelmTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.build_utils import repair_config


@ModelRegistry("stablelm")
class StablelmTpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = StableLMPreAndPostLayerWeight
    transformer_weight_class = StablelmTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = StablelmTransformerLayerInfer
    post_layer_infer_class = BloomPostLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _init_config(self):
        super()._init_config()
        repair_config(self.config, same_names=["rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon"])
        return
