from lightllm.models.glm4_moe.layer_weights.transformer_layer_weight import Glm4MoeTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NoTpNormWeight


class Glm4MoeMTPTransformerLayerWeight(Glm4MoeTransformerLayerWeight):
    """
    Transformer layer weights for GLM-4.7 MoE MTP model.

    MTP layers only need FFN weights (no attention), so we override
    _init_weight to skip QKV and O projection initialization.
    """

    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _init_weight(self):
        """Initialize only FFN-related weights (no attention weights for MTP)."""
        self._init_norm()
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()

    def _init_norm(self):
        """Initialize only FFN normalization (no attention norm for MTP)."""
        self.ffn_norm_weight_ = NoTpNormWeight(
            self._ffn_norm_weight_name, self.data_type_, bias_name=self._ffn_norm_bias_name
        )
