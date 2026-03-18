from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    QKRMSNORMWeight,
    ROWMMWeight,
    RMSNormWeight,
)


class NeoChatMOETransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        self._is_merge_kv = network_config.get("merge_kv", True)
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _init_weight_names(self):
        super()._init_weight_names()

        if self._is_merge_kv:
            self._q_norm_hw_name = f"model.layers.{self.layer_num_}.self_attn.q_norm_hw.weight"
            self._k_norm_hw_name = f"model.layers.{self.layer_num_}.self_attn.k_norm_hw.weight"
        else:
            self._q_norm_h_name = f"model.layers.{self.layer_num_}.self_attn.q_norm_h.weight"
            self._q_norm_w_name = f"model.layers.{self.layer_num_}.self_attn.q_norm_w.weight"

            self._k_norm_h_name = f"model.layers.{self.layer_num_}.self_attn.k_norm_h.weight"
            self._k_norm_w_name = f"model.layers.{self.layer_num_}.self_attn.k_norm_w.weight"

    def _init_qkv(self):
        super()._init_qkv()

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

        self.q_norm_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            weight_name=self._q_norm_name,
            data_type=self.data_type_,
        )
        self.k_norm_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            weight_name=self._k_norm_name,
            data_type=self.data_type_,
        )

        if self._is_merge_kv:
            self.q_norm_hw_weight_ = QKRMSNORMWeight(
                dim=self.head_dim // 2,
                weight_name=self._q_norm_hw_name,
                data_type=self.data_type_,
            )
            self.k_norm_hw_weight_ = QKRMSNORMWeight(
                dim=self.head_dim // 2,
                weight_name=self._k_norm_hw_name,
                data_type=self.data_type_,
            )
        else:
            self.q_norm_h_weight_ = QKRMSNORMWeight(
                dim=self.head_dim // 4,
                weight_name=self._q_norm_h_name,
                data_type=self.data_type_,
            )
            self.q_norm_w_weight_ = QKRMSNORMWeight(
                dim=self.head_dim // 4,
                weight_name=self._q_norm_w_name,
                data_type=self.data_type_,
            )
            self.k_norm_h_weight_ = QKRMSNORMWeight(
                dim=self.head_dim // 4,
                weight_name=self._k_norm_h_name,
                data_type=self.data_type_,
            )
            self.k_norm_w_weight_ = QKRMSNORMWeight(
                dim=self.head_dim // 4,
                weight_name=self._k_norm_w_name,
                data_type=self.data_type_,
            )
