from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight


class StarcoderTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg, layer_prefix="transformer.h")
        assert network_config["num_attention_heads"] % self.tp_world_size_ == 0

    def load_hf_weights(self, weights):
        n_embed = self.network_config_["hidden_size"]
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]

        qkv_weight_name = f"transformer.h.{self.layer_num_}.attn.c_attn.weight"
        if qkv_weight_name in weights:
            qkv_weight = weights[qkv_weight_name]
            weights[self._q_weight_name] = qkv_weight[:, :n_embed]
            weights[self._k_weight_name] = qkv_weight[:, n_embed : n_embed + head_dim]
            weights[self._v_weight_name] = qkv_weight[:, n_embed + head_dim : n_embed + 2 * head_dim]
            del weights[qkv_weight_name]

        qkv_bias_name = f"transformer.h.{self.layer_num_}.attn.c_attn.bias"
        if qkv_bias_name in weights:
            qkv_bias = weights[qkv_bias_name]
            weights[self._q_bias_name] = qkv_bias[:, :n_embed]
            weights[self._k_bias_name] = qkv_bias[:, n_embed : n_embed + head_dim]
            weights[self._v_bias_name] = qkv_bias[:, n_embed + head_dim : n_embed + 2 * head_dim]
            del weights[qkv_bias_name]
        super().load_hf_weights(weights)

    def _init_weight_names(self):
        self._q_weight_name = f"transformer.h.{self.layer_num_}.self_attn.q_proj.weight"
        self._k_weight_name = f"transformer.h.{self.layer_num_}.self_attn.k_proj.weight"
        self._v_weight_name = f"transformer.h.{self.layer_num_}.self_attn.v_proj.weight"
        self._q_bias_name = f"transformer.h.{self.layer_num_}.self_attn.q_proj.bias"
        self._k_bias_name = f"transformer.h.{self.layer_num_}.self_attn.k_proj.bias"
        self._v_bias_name = f"transformer.h.{self.layer_num_}.self_attn.v_proj.bias"

        self._o_weight_name = f"transformer.h.{self.layer_num_}.attn.c_proj.weight"
        self._o_bias_name = f"transformer.h.{self.layer_num_}.attn.c_proj.bias"

        self._gate_up_weight_name = f"transformer.h.{self.layer_num_}.mlp.c_fc.weight"
        self._gate_up_bias_name = f"transformer.h.{self.layer_num_}.mlp.c_fc.bias"
        self._down_weight_name = f"transformer.h.{self.layer_num_}.mlp.c_proj.weight"
        self._down_bias_name = f"transformer.h.{self.layer_num_}.mlp.c_proj.bias"

        self._att_norm_weight_name = f"transformer.h.{self.layer_num_}.ln_1.weight"
        self._att_norm_bias_name = f"transformer.h.{self.layer_num_}.ln_1.bias"
        self._ffn_norm_weight_name = f"transformer.h.{self.layer_num_}.ln_2.weight"
        self._ffn_norm_bias_name = f"transformer.h.{self.layer_num_}.ln_2.bias"

    def _init_ffn(self):
        self.gate_up_proj = ROWMMWeight(
            weight_names=self._gate_up_weight_name,
            data_type=self.data_type_,
            bias_names=self._gate_up_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="gate_up_proj",
        )
        self.down_proj = COLMMWeight(
            weight_names=self._down_weight_name,
            data_type=self.data_type_,
            bias_names=self._down_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="down_proj",
        )
