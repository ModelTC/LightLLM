from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight import ROWMMWeight, COLMMWeight
from lightllm.common.basemodel.layer_weights.meta_weights import RMSNormWeight, ParameterWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Gemma4TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(
        self,
        layer_num,
        data_type,
        network_config,
        quant_cfg=None,
    ):
        self._pre_parse_layer_shape(layer_num, network_config)
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _pre_parse_layer_shape(self, layer_num, network_config):
        layer_type = network_config["layer_types"][layer_num]
        self._is_sliding = layer_type == "sliding_attention"
        if self._is_sliding:
            self._layer_head_dim = network_config["head_dim"]
            self._layer_kv_head_num = network_config["num_key_value_heads"]
            self._layer_k_eq_v = False
        else:
            self._layer_head_dim = network_config["global_head_dim"]
            self._layer_kv_head_num = network_config["num_global_key_value_heads"]
            self._layer_k_eq_v = network_config.get("attention_k_eq_v", True)

    def _parse_config(self):
        self.n_head = self.network_config_["num_attention_heads"]
        self.q_head_num_ = self.network_config_["num_attention_heads"]
        self.k_head_num_ = self._layer_kv_head_num
        self.v_head_num_ = self._layer_kv_head_num
        self.o_head_num_ = self.q_head_num_
        self.head_dim = self._layer_head_dim
        self.n_embed = self.network_config_["hidden_size"]
        self.n_inter = self.network_config_["intermediate_size"]

    def _init_weight_names(self):
        prefix = f"model.language_model.layers.{self.layer_num_}"
        self._q_weight_name = f"{prefix}.self_attn.q_proj.weight"
        self._q_bias_name = None
        self._k_weight_name = f"{prefix}.self_attn.k_proj.weight"
        self._k_bias_name = None
        self._v_weight_name = f"{prefix}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._o_weight_name = f"{prefix}.self_attn.o_proj.weight"
        self._o_bias_name = None

        self._q_norm_weight_name = f"{prefix}.self_attn.q_norm.weight"
        self._k_norm_weight_name = f"{prefix}.self_attn.k_norm.weight"

        self._gate_weight_name = f"{prefix}.mlp.gate_proj.weight"
        self._up_weight_name = f"{prefix}.mlp.up_proj.weight"
        self._down_weight_name = f"{prefix}.mlp.down_proj.weight"

        self._att_norm_weight_name = f"{prefix}.input_layernorm.weight"
        self._ffn_norm_weight_name = f"{prefix}.post_attention_layernorm.weight"
        self._pre_feedforward_layernorm_name = f"{prefix}.pre_feedforward_layernorm.weight"
        self._post_feedforward_layernorm_name = f"{prefix}.post_feedforward_layernorm.weight"

        self._layer_scalar_name = f"{prefix}.layer_scalar"

    def _init_weight(self):
        self._init_qkv()
        self._init_o()
        self._init_ffn()
        self._init_norm()

    def _init_qkv(self):
        in_dim = self.n_embed
        q_out_dim = self.q_head_num_ * self.head_dim
        kv_out_dim = self.k_head_num_ * self.head_dim

        self.q_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim],
            weight_names=self._q_weight_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_name,
            quant_method=self.get_quant_method("q_proj"),
        )
        self.k_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[kv_out_dim],
            weight_names=self._k_weight_name,
            data_type=self.data_type_,
            bias_names=self._k_bias_name,
            quant_method=self.get_quant_method("k_proj"),
        )
        if not self._layer_k_eq_v:
            self.v_proj = ROWMMWeight(
                in_dim=in_dim,
                out_dims=[kv_out_dim],
                weight_names=self._v_weight_name,
                data_type=self.data_type_,
                bias_names=self._v_bias_name,
                quant_method=self.get_quant_method("v_proj"),
            )
        # For k_eq_v layers HF checkpoint has no v_proj weight; the inference
        # code aliases v = k at compute time, so no weight object is created.

    def _init_o(self):
        in_dim = self.o_head_num_ * self.head_dim
        out_dim = self.n_embed
        self.o_proj = COLMMWeight(
            in_dim=in_dim,
            out_dims=[out_dim],
            weight_names=self._o_weight_name,
            data_type=self.data_type_,
            bias_names=self._o_bias_name,
            quant_method=self.get_quant_method("o_proj"),
        )

    def _init_ffn(self):
        self.gate_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_inter],
            weight_names=self._gate_weight_name,
            data_type=self.data_type_,
            bias_names=None,
            quant_method=self.get_quant_method("gate_proj"),
        )
        self.up_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_inter],
            weight_names=self._up_weight_name,
            data_type=self.data_type_,
            bias_names=None,
            quant_method=self.get_quant_method("up_proj"),
        )
        self.down_proj = COLMMWeight(
            in_dim=self.n_inter,
            out_dims=[self.n_embed],
            weight_names=self._down_weight_name,
            data_type=self.data_type_,
            bias_names=None,
            quant_method=self.get_quant_method("down_proj"),
        )

    def _init_norm(self):
        hidden_size = self.network_config_["hidden_size"]
        # Gemma-4 uses *standard* RMSNorm (x * rsqrt(var+eps) * w), NOT the
        # gemma2/3 (1+w) variant. Using NoTpGEMMANormWeight here produces
        # nothing but high-frequency-token gibberish ("de la de...").
        self.q_norm_weight_ = RMSNormWeight(
            dim=self._layer_head_dim,
            weight_name=self._q_norm_weight_name,
            data_type=self.data_type_,
        )
        self.k_norm_weight_ = RMSNormWeight(
            dim=self._layer_head_dim,
            weight_name=self._k_norm_weight_name,
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
        # scalar multiplier applied to the attention output
        self.layer_scalar_ = ParameterWeight(
            weight_name=self._layer_scalar_name,
            data_type=self.data_type_,
            weight_shape=(1,),
        )
