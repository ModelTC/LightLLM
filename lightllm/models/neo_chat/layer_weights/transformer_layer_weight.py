from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    QKRMSNORMWeight,
    RMSNormWeight,
    QKVROWNMMWeight,
)


class NeoChatTransformerLayerWeight(Qwen3TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_norm_hw_name = f"model.layers.{self.layer_num_}.self_attn.q_norm_hw.weight"
        self._k_norm_hw_name = f"model.layers.{self.layer_num_}.self_attn.k_norm_hw.weight"

    def _init_qkv(self):
        in_dim = self.n_embed
        self.qkv_proj = QKVROWNMMWeight(
            in_dim=in_dim,
            q_head_num=self.q_head_num_,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._q_weight_name, self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._q_bias_name, self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("qkv_proj"),
        )

    def _init_norm(self):
        hidden_size = self.network_config_["hidden_size"]
        self.att_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._att_norm_weight_name,
            data_type=self.data_type_,
        )
        self.ffn_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._ffn_norm_weight_name,
            data_type=self.data_type_,
        )

        self.qk_norm_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            q_weight_name=self._q_norm_name,
            k_weight_name=self._k_norm_name,
            data_type=self.data_type_,
        )

        self.qk_hw_norm_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            q_weight_name=self._q_norm_hw_name,
            k_weight_name=self._k_norm_hw_name,
            data_type=self.data_type_,
        )
