from lightllm.models.internlm_wquant.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeightQuantized


class Internlm2TransformerLayerWeightQuantized(InternlmTransformerLayerWeightQuantized):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.attention_norm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.attention_norm.weight"])

        n_embed = self.network_config_["hidden_size"]
        q_split_n_embed = n_embed // self.world_size_
        kv_split_n_embed = (
            n_embed
            // self.network_config_["num_attention_heads"]
            * self.network_config_["num_key_value_heads"]
            // self.world_size_
        )
        head_dim = n_embed // self.network_config_["num_attention_heads"]
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.attention.wqkv.weight" in weights:
            qkv_weight_ = weights[f"model.layers.{self.layer_num_}.attention.wqkv.weight"]
            q_groups = self.network_config_["num_attention_heads"] // self.network_config_["num_key_value_heads"]
            qkv_weight_ = qkv_weight_.reshape(self.network_config_["num_key_value_heads"], q_groups + 2, head_dim, -1)
            q_weight_ = qkv_weight_[:, :q_groups, :, :].reshape(-1, qkv_weight_.shape[-1])
            q_weight_ = q_weight_[q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1) :].transpose(0, 1)
            self.q_weight_ = self.quantize_weight(q_weight_)

            k_weight_ = qkv_weight_[:, -2, :, :].reshape(-1, qkv_weight_.shape[-1])
            self.k_weight_ = k_weight_[
                kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1) :
            ].transpose(0, 1)
            v_weight_ = qkv_weight_[:, -1, :, :].reshape(-1, qkv_weight_.shape[-1])
            self.v_weight_ = v_weight_[
                kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1) :
            ].transpose(0, 1)

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1, handle_func=self.quantize_weight)

        # attention output dense params
        if f"model.layers.{self.layer_num_}.attention.wo.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.attention.wo.weight"]
            self.o_weight_ = self.o_weight_[:, q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self.quantize_weight(self.o_weight_.transpose(0, 1))
        if f"model.layers.{self.layer_num_}.attention.wo.bias" in weights:
            self.o_bias_ = weights[f"model.layers.{self.layer_num_}.attention.wo.bias"]
            self.o_bias_ = self._cuda(self.o_bias_)
        return

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.ffn_norm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.ffn_norm.weight"])

        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.feed_forward.w3.weight" in weights:
            up_proj = weights[f"model.layers.{self.layer_num_}.feed_forward.w3.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.up_proj = up_proj.transpose(0, 1)

        if f"model.layers.{self.layer_num_}.feed_forward.w1.weight" in weights:
            gate_proj = weights[f"model.layers.{self.layer_num_}.feed_forward.w1.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.gate_proj = gate_proj.transpose(0, 1)

        self._try_cat_to(["gate_proj", "up_proj"], "gate_up_proj", cat_dim=1, handle_func=self.quantize_weight)

        if f"model.layers.{self.layer_num_}.feed_forward.w2.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.feed_forward.w2.weight"][
                :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ]
            self.down_proj = self.quantize_weight(self.down_proj.transpose(0, 1))
        return
