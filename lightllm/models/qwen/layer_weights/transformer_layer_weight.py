import torch
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class QwenTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

    def load_hf_weights(self, weights):
        qkv_weight_name = f"transformer.h.{self.layer_num_}.attn.c_attn.weight"
        if qkv_weight_name in weights:
            qkv_weight_ = weights[qkv_weight_name]
            split_size = qkv_weight_.shape[0] // 3
            q_weight_, k_weight_, v_weight_ = torch.split(qkv_weight_, split_size, dim=0)
            weights[self._q_weight_name] = q_weight_
            weights[self._k_weight_name] = k_weight_
            weights[self._v_weight_name] = v_weight_
            del weights[qkv_weight_name]

        qkv_bias_name = f"transformer.h.{self.layer_num_}.attn.c_attn.bias"
        if qkv_bias_name in weights:
            qkv_bias = weights[qkv_bias_name]
            split_size = qkv_bias.shape[0] // 3
            q_bias_, k_bias_, v_bias_ = torch.split(qkv_bias, split_size, dim=0)
            weights[self._q_bias_name] = q_bias_
            weights[self._k_bias_name] = k_bias_
            weights[self._v_bias_name] = v_bias_
            del weights[qkv_bias_name]

        super().load_hf_weights(weights)

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_weight_name = f"transformer.h.{self.layer_num_}.attn.q_proj.weight"
        self._q_bias_name = f"transformer.h.{self.layer_num_}.attn.q_proj.bias"
        self._k_weight_name = f"transformer.h.{self.layer_num_}.attn.k_proj.weight"
        self._k_bias_name = f"transformer.h.{self.layer_num_}.attn.k_proj.bias"
        self._v_weight_name = f"transformer.h.{self.layer_num_}.attn.v_proj.weight"
        self._v_bias_name = f"transformer.h.{self.layer_num_}.attn.v_proj.bias"

        self._o_weight_name = f"transformer.h.{self.layer_num_}.attn.c_proj.weight"
        self._gate_weight_name = f"transformer.h.{self.layer_num_}.mlp.w2.weight"
        self._up_weight_name = f"transformer.h.{self.layer_num_}.mlp.w1.weight"
        self._down_weight_name = f"transformer.h.{self.layer_num_}.mlp.c_proj.weight"
        self._att_norm_weight_name = f"transformer.h.{self.layer_num_}.ln_1.weight"
        self._ffn_norm_weight_name = f"transformer.h.{self.layer_num_}.ln_2.weight"
