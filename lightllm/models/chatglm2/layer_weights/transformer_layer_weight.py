from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class ChatGLM2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(
            layer_num,
            data_type,
            network_config,
            mode,
            quant_cfg,
        )
        return

    def _preprocess_weight(self, weights):
        n_kv_embed = self.head_dim * self.n_kv_head
        qkv_weight_name = f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.weight"
        if qkv_weight_name in weights:
            qkv_weight_ = weights[qkv_weight_name]
            weights[self._q_weight_name] = qkv_weight_[: self.n_embed, :]
            weights[self._k_weight_name] = qkv_weight_[self.n_embed : self.n_embed + n_kv_embed, :]
            weights[self._v_weight_name] = qkv_weight_[self.n_embed + n_kv_embed : self.n_embed + 2 * n_kv_embed, :]
            del weights[qkv_weight_name]

        qkv_bias_name = f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.bias"
        if qkv_bias_name in weights:
            qkv_bias_ = weights[qkv_bias_name]
            weights[self._q_bias_name] = qkv_bias_[: self.n_embed]
            weights[self._k_bias_name] = qkv_bias_[self.n_embed : self.n_embed + n_kv_embed]
            weights[self._v_bias_name] = qkv_bias_[self.n_embed + n_kv_embed : self.n_embed + 2 * n_kv_embed]
            del weights[qkv_bias_name]

        gate_up_weight_name = f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_h_to_4h.weight"
        if gate_up_weight_name in weights:
            gate_up_weight_ = weights[gate_up_weight_name]
            weights[self._gate_weight_name] = gate_up_weight_[: self.n_inter, :]
            weights[self._up_weight_name] = gate_up_weight_[self.n_inter : 2 * self.n_inter, :]
            del weights[gate_up_weight_name]

    def _parse_config(self):
        self.n_embed = self.network_config_["hidden_size"]
        self.n_head = self.network_config_["num_attention_heads"]
        self.n_inter = self.network_config_["ffn_hidden_size"]
        self.n_kv_head = self.network_config_["multi_query_group_num"]
        self.head_dim = self.network_config_.get("head_dim", self.n_embed // self.n_head)

    def load_hf_weights(self, weights):
        self._preprocess_weight(weights)
        super().load_hf_weights(weights)
        return

    def _init_weight_names(self):
        self._q_weight_name = f"transformer.encoder.layers.{self.layer_num_}.self_attention.q_proj.weight"
        self._q_bias_name = f"transformer.encoder.layers.{self.layer_num_}.self_attention.q_proj.bias"
        self._k_weight_name = f"transformer.encoder.layers.{self.layer_num_}.self_attention.k_proj.weight"
        self._k_bias_name = f"transformer.encoder.layers.{self.layer_num_}.self_attention.k_proj.bias"
        self._v_weight_name = f"transformer.encoder.layers.{self.layer_num_}.self_attention.v_proj.weight"
        self._v_bias_name = f"transformer.encoder.layers.{self.layer_num_}.self_attention.v_proj.bias"
        self._o_weight_name = f"transformer.encoder.layers.{self.layer_num_}.self_attention.dense.weight"
        self._o_bias_name = None

        self._gate_weight_name = f"transformer.encoder.layers.{self.layer_num_}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"transformer.encoder.layers.{self.layer_num_}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._down_weight_name = f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_4h_to_h.weight"
        self._down_bias_name = None

        self._att_norm_weight_name = f"transformer.encoder.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"transformer.encoder.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None
