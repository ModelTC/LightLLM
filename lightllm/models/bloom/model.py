from lightllm.models.registry import ModelRegistry
from lightllm.models.bloom.layer_infer.pre_layer_infer import BloomPreLayerInfer
from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer
from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.common.basemodel import InferStateInfo, TpPartBaseModel


@ModelRegistry("bloom")
class BloomTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = BloomPreAndPostLayerWeight
    transformer_weight_class = BloomTransformerLayerWeight

    # infer class
    pre_layer_infer_class = BloomPreLayerInfer
    post_layer_infer_class = BloomPostLayerInfer
    transformer_layer_infer_class = BloomTransformerLayerInfer

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        self._reset_num_key_value_heads()
        return

    def _reset_num_key_value_heads(self):
        self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return
