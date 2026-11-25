from lightllm.common.basemodel.attention_runtime import AttentionSpec
from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer


class StarcoderTransformerLayerInfer(BloomTransformerLayerInfer):
    """Bloom's layer with multi-query attention and no ALiBi."""

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        return

    def attention_spec(self, model) -> AttentionSpec:
        config = self._gqa_config(model)
        return AttentionSpec(prefill=config, decode=config)

    def _gqa_alibi_slopes(self, layer_weight):
        return None
