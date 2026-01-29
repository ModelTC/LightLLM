from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    NormWeight,
    ROWMMWeight,
)


class NeoChatMOETransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        self._is_merge_kv = network_config["merge_kv"]
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_weight_hw_name = f"model.layers.{self.layer_num_}.self_attn.q_proj_hw.weight"
        self._q_bias_hw_name = None
        self._k_weight_hw_name = f"model.layers.{self.layer_num_}.self_attn.k_proj_hw.weight"
        self._k_bias_hw_name = None

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
        self.q_hw_proj = ROWMMWeight(
            weight_names=self._q_weight_hw_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_hw_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="q_hw_proj",
        )
        self.k_hw_proj = ROWMMWeight(
            weight_names=self._k_weight_hw_name,
            data_type=self.data_type_,
            bias_names=self._k_bias_hw_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="k_hw_proj",
        )

    def _init_norm(self):
        super()._init_norm()
        if self._is_merge_kv:
            self.q_norm_hw_weight_ = NormWeight(weight_name=self._q_norm_hw_name, data_type=self.data_type_)
            self.k_norm_hw_weight_ = NormWeight(weight_name=self._k_norm_hw_name, data_type=self.data_type_)
        else:
            self.q_norm_h_weight_ = NormWeight(weight_name=self._q_norm_h_name, data_type=self.data_type_)
            self.q_norm_w_weight_ = NormWeight(weight_name=self._q_norm_w_name, data_type=self.data_type_)
            self.k_norm_h_weight_ = NormWeight(weight_name=self._k_norm_h_name, data_type=self.data_type_)
            self.k_norm_w_weight_ = NormWeight(weight_name=self._k_norm_w_name, data_type=self.data_type_)
