from lightllm.common.basemodel.layer_weights.meta_weights import RMSNormWeight, ROWMMWeight, ParameterWeight
from lightllm.models.gemma4.layer_weights.transformer_layer_weight import Gemma4TransformerLayerWeight


class Gemma4MTPTransformerLayerWeight(Gemma4TransformerLayerWeight):
    """
    Gemma-4 assistant decoder-layer weights. Same block shape as the target's
    Gemma4TransformerLayerWeight, but:
      * checkpoint prefix is `model.layers.{i}` (the target uses
        `model.language_model.layers.{i}`),
      * attention is Q-projection only - the assistant has no k/v_proj or k_norm
        (it reads the target's KV cache),
      * never MoE / never PLE (the assistant trunk is always dense).
    `layer_num_` here is the assistant-local index (0..num_mtp_layers-1).
    """

    def _init_weight_names(self):
        prefix = f"model.layers.{self.layer_num_}"
        self._q_weight_name = f"{prefix}.self_attn.q_proj.weight"
        self._q_bias_name = None
        self._o_weight_name = f"{prefix}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._q_norm_weight_name = f"{prefix}.self_attn.q_norm.weight"

        self._gate_weight_name = f"{prefix}.mlp.gate_proj.weight"
        self._up_weight_name = f"{prefix}.mlp.up_proj.weight"
        self._down_weight_name = f"{prefix}.mlp.down_proj.weight"

        self._att_norm_weight_name = f"{prefix}.input_layernorm.weight"
        self._ffn_norm_weight_name = f"{prefix}.post_attention_layernorm.weight"
        self._pre_feedforward_layernorm_name = f"{prefix}.pre_feedforward_layernorm.weight"
        self._post_feedforward_layernorm_name = f"{prefix}.post_feedforward_layernorm.weight"

        self._layer_scalar_name = f"{prefix}.layer_scalar"

    def _init_qkv(self):
        # Q-projection only: the assistant reads K/V from the target's cache.
        self.q_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.q_head_num_ * self.head_dim],
            weight_names=self._q_weight_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_name,
            quant_method=self.get_quant_method("q_proj"),
        )

    def _init_norm(self):
        hidden_size = self.network_config_["hidden_size"]
        # Standard RMSNorm (not the gemma2/3 (1+w) variant). No k_norm: there is
        # no k_proj on the assistant.
        self.q_norm_weight_ = RMSNormWeight(
            dim=self._layer_head_dim,
            weight_name=self._q_norm_weight_name,
            data_type=self.data_type_,
        )
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
        self.pre_feedforward_layernorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._pre_feedforward_layernorm_name,
            data_type=self.data_type_,
        )
        self.post_feedforward_layernorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._post_feedforward_layernorm_name,
            data_type=self.data_type_,
        )
        self.layer_scalar_ = ParameterWeight(
            weight_name=self._layer_scalar_name,
            data_type=self.data_type_,
            weight_shape=(1,),
        )
