import torch
import numpy as np
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import ROWMMWeight
from lightllm.common.basemodel.layer_weights.meta_weights.norm_weight import NormWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Gemma3TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(
        self,
        layer_num,
        data_type,
        network_config,
        mode=[],
        quant_cfg=None,
    ):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        self._att_norm_weight_name = f"model.layers.{self.layer_num_}.input_layernorm.weight"
        self._k_norm_weight_name = f"model.layers.{self.layer_num_}.self_attn.k_norm.weight"
        self._q_norm_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._ffn_norm_weight_name = f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._pre_feedforward_layernorm_name = f"model.layers.{self.layer_num_}.pre_feedforward_layernorm.weight"
        self._post_feedforward_layernorm_name = f"model.layers.{self.layer_num_}.post_feedforward_layernorm.weight"

    def _init_ffn(self):
        self.gate_proj = ROWMMWeight(
            weight_name=self._gate_weight_name,
            data_type=self.data_type_,
            bias_name=self._gate_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="gate_proj",
        )
        self.up_proj = ROWMMWeight(
            weight_name=self._up_weight_name,
            data_type=self.data_type_,
            bias_name=self._up_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="up_proj",
        )
        super()._init_ffn()

    def _init_qkv(self):
        self.k_proj = ROWMMWeight(
            weight_name=self._k_weight_name,
            data_type=self.data_type_,
            bias_name=self._k_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="k_proj",
        )
        self.v_proj = ROWMMWeight(
            weight_name=self._v_weight_name,
            data_type=self.data_type_,
            bias_name=self._v_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="v_proj",
        )
        super()._init_qkv()

    def _init_norm(self):
        super()._init_norm()
        self.k_norm_weight_ = NormWeight(self._k_norm_weight_name, self.data_type_, bias_name=None)
        self.q_norm_weight_ = NormWeight(self._q_norm_weight_name, self.data_type_, bias_name=None)
        self.pre_feedforward_layernorm_weight_ = NormWeight(
            self._pre_feedforward_layernorm_name, self.data_type_, bias_name=None
        )
        self.post_feedforward_layernorm_weight_ = NormWeight(
            self._post_feedforward_layernorm_name, self.data_type_, bias_name=None
        )

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        return
