from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    QKRMSNORMWeight,
    ROWMMWeight,
)


class NeoChatTransformerLayerWeight(Qwen3TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_weight_hw_name = f"model.layers.{self.layer_num_}.self_attn.q_proj_hw.weight"
        self._q_bias_hw_name = None
        self._k_weight_hw_name = f"model.layers.{self.layer_num_}.self_attn.k_proj_hw.weight"
        self._k_bias_hw_name = None

        self._q_norm_h_name = f"model.layers.{self.layer_num_}.self_attn.q_norm_h.weight"
        self._q_norm_w_name = f"model.layers.{self.layer_num_}.self_attn.q_norm_w.weight"

        self._k_norm_h_name = f"model.layers.{self.layer_num_}.self_attn.k_norm_h.weight"
        self._k_norm_w_name = f"model.layers.{self.layer_num_}.self_attn.k_norm_w.weight"

    def _init_qkv(self):
        super()._init_qkv()
        self.q_hw_proj = ROWMMWeight(
            in_dim=self.network_config_["hidden_size"],
            out_dims=[self.q_head_num_ * self.head_dim],
            weight_names=self._q_weight_hw_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_hw_name,
            quant_method=self.get_quant_method("q_hw_proj"),
        )
        self.k_hw_proj = ROWMMWeight(
            in_dim=self.network_config_["hidden_size"],
            out_dims=[self.k_head_num_ * self.head_dim],
            weight_names=self._k_weight_hw_name,
            data_type=self.data_type_,
            bias_names=self._k_bias_hw_name,
            quant_method=self.get_quant_method("k_hw_proj"),
        )

    def _init_norm(self):
        super()._init_norm()

        self.q_norm_h_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            weight_name=self._q_norm_h_name,
            data_type=self.data_type_,
        )
        self.q_norm_w_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            weight_name=self._q_norm_w_name,
            data_type=self.data_type_,
        )
        self.k_norm_h_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            weight_name=self._k_norm_h_name,
            data_type=self.data_type_,
        )
        self.k_norm_w_weight_ = QKRMSNORMWeight(
            dim=self.head_dim // 2,
            weight_name=self._k_norm_w_name,
            data_type=self.data_type_,
        )
