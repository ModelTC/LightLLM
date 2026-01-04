from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NoTpNormWeight, TpHeadNormWeight


class CohereTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _parse_config(self):
        super()._parse_config()
        self.use_qk_norm = self.network_config_.get("use_qk_norm", False)

    def _init_norm(self):
        self.att_norm_weight_ = NoTpNormWeight(self._att_norm_weight_name, self.data_type_)

        if self.use_qk_norm:
            self.q_norm_weight_ = TpHeadNormWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_norm.weight", self.data_type_
            )
            self.k_norm_weight_ = TpHeadNormWeight(
                f"model.layers.{self.layer_num_}.self_attn.k_norm.weight", self.data_type_
            )

        return
