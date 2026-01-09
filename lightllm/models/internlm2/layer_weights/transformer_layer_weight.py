from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Internlm2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def load_hf_weights(self, weights):
        qkv_weight_name = f"model.layers.{self.layer_num_}.attention.wqkv.weight"
        if qkv_weight_name in weights:
            qkv_weight_ = weights[qkv_weight_name]
            q_groups = self.n_head // self.n_kv_head
            qkv_weight_ = qkv_weight_.reshape(self.n_kv_head, q_groups + 2, self.head_dim, -1)
            q_weight_ = qkv_weight_[:, :q_groups, :, :].reshape(-1, qkv_weight_.shape[-1])
            k_weight_ = qkv_weight_[:, -2, :, :].reshape(-1, qkv_weight_.shape[-1])
            v_weight_ = qkv_weight_[:, -1, :, :].reshape(-1, qkv_weight_.shape[-1])
            weights[self._q_weight_name] = q_weight_
            weights[self._k_weight_name] = k_weight_
            weights[self._v_weight_name] = v_weight_
            del weights[qkv_weight_name]
        super().load_hf_weights(weights)

    def _init_weight_names(self):
        super()._init_weight_names()
        self._o_weight_name = f"model.layers.{self.layer_num_}.attention.wo.weight"

        self._gate_weight_name = f"model.layers.{self.layer_num_}.feed_forward.w1.weight"
        self._up_weight_name = f"model.layers.{self.layer_num_}.feed_forward.w3.weight"
        self._down_weight_name = f"model.layers.{self.layer_num_}.feed_forward.w2.weight"
        self._att_norm_weight_name = f"model.layers.{self.layer_num_}.attention_norm.weight"
        self._ffn_norm_weight_name = f"model.layers.{self.layer_num_}.ffn_norm.weight"
