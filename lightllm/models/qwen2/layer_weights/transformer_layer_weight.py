from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Qwen2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"
        self._k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"
        self._v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"

    def _parse_config(self):
        self.tp_q_head_num_ = self.network_config_["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = max(self.network_config_["num_key_value_heads"] // self.tp_world_size_, 1)
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]
        self.head_dim = self.network_config_.get("head_dim", head_dim)
        assert (self.tp_k_head_num_ * self.tp_world_size_) % self.network_config_["num_key_value_heads"] == 0

    def _repeat_weight(self, name, weights):
        # for tp_world_size_ > num_key_value_heads
        if name not in weights:
            return

        tensor = weights[name]
        num_kv_heads = self.network_config_["num_key_value_heads"]
        repeat_size = (self.tp_k_head_num_ * self.tp_world_size_) // num_kv_heads

        if tensor.ndim == 1:
            # Bias (1D tensor)
            tensor = tensor.reshape(num_kv_heads, -1).unsqueeze(1).repeat(1, repeat_size, 1).reshape(-1)
        else:
            # Weight (2D tensor)
            tensor = (
                tensor.reshape(num_kv_heads, -1, tensor.shape[-1])
                .unsqueeze(1)
                .repeat(1, repeat_size, 1, 1)
                .reshape(-1, tensor.shape[-1])
            )
        weights[name] = tensor

    def load_hf_weights(self, weights):
        self._repeat_weight(self._k_weight_name, weights)
        self._repeat_weight(self._v_weight_name, weights)
        if self._k_bias_name is not None and self._v_bias_name is not None:
            self._repeat_weight(self._k_bias_name, weights)
            self._repeat_weight(self._v_bias_name, weights)
        return super().load_hf_weights(weights)
