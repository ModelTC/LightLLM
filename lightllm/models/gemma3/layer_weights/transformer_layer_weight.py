from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight import ROWMMWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NoTpGEMMANormWeight


class Gemma3TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(
        self,
        layer_num,
        data_type,
        network_config,
        quant_cfg=None,
    ):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
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
            in_dim=self.n_embed,
            out_dims=[self.n_inter],
            weight_names=self._gate_weight_name,
            data_type=self.data_type_,
            bias_names=self._gate_bias_name,
            quant_method=self.get_quant_method("gate_proj"),
        )
        self.up_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_inter],
            weight_names=self._up_weight_name,
            data_type=self.data_type_,
            bias_names=self._up_bias_name,
            quant_method=self.get_quant_method("up_proj"),
        )
        super()._init_ffn()

    def _init_qkv(self):
        kv_out_dim = self.k_head_num_ * self.head_dim
        self.k_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[kv_out_dim],
            weight_names=self._k_weight_name,
            data_type=self.data_type_,
            bias_names=self._k_bias_name,
            quant_method=self.get_quant_method("k_proj"),
        )
        self.v_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[kv_out_dim],
            weight_names=self._v_weight_name,
            data_type=self.data_type_,
            bias_names=self._v_bias_name,
            quant_method=self.get_quant_method("v_proj"),
        )
        super()._init_qkv()

    def _init_norm(self):
        super()._init_norm()

        self.k_norm_weight_ = NoTpGEMMANormWeight(
            dim=self.head_dim_, weight_name=self._k_norm_weight_name, data_type=self.data_type_, bias_name=None
        )
        self.q_norm_weight_ = NoTpGEMMANormWeight(
            dim=self.head_dim_, weight_name=self._q_norm_weight_name, data_type=self.data_type_, bias_name=None
        )
        self.pre_feedforward_layernorm_weight_ = NoTpGEMMANormWeight(
            dim=self.n_embed,
            weight_name=self._pre_feedforward_layernorm_name,
            data_type=self.data_type_,
            bias_name=None,
        )
        self.post_feedforward_layernorm_weight_ = NoTpGEMMANormWeight(
            dim=self.n_embed,
            weight_name=self._post_feedforward_layernorm_name,
            data_type=self.data_type_,
            bias_name=None,
        )

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        return
