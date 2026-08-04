from lightllm.common.basemodel.layer_weights.meta_weights import COLMMWeight, KVROWNMMWeight, RMSNormWeight, ROWMMWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Qwen3DFlashTransformerLayerWeight(LlamaTransformerLayerWeight):
    """DFlash decoder layer weights.

    The DSpark/DFlash Qwen3 layer uses Qwen3-style q/k RMSNorm, but its weight
    prefix is `layers.{i}` rather than `model.layers.{i}`.
    """

    def _init_weight_names(self):
        weight_prefix = f"layers.{self.layer_num_}"
        self._q_weight_name = f"{weight_prefix}.self_attn.q_proj.weight"
        self._q_bias_name = None
        self._k_weight_name = f"{weight_prefix}.self_attn.k_proj.weight"
        self._k_bias_name = None
        self._v_weight_name = f"{weight_prefix}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._o_weight_name = f"{weight_prefix}.self_attn.o_proj.weight"
        self._o_bias_name = None

        self._gate_weight_name = f"{weight_prefix}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"{weight_prefix}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._down_weight_name = f"{weight_prefix}.mlp.down_proj.weight"
        self._down_bias_name = None

        self._att_norm_weight_name = f"{weight_prefix}.input_layernorm.weight"
        self._ffn_norm_weight_name = f"{weight_prefix}.post_attention_layernorm.weight"
        self._q_norm_name = f"{weight_prefix}.self_attn.q_norm.weight"
        self._k_norm_name = f"{weight_prefix}.self_attn.k_norm.weight"
        return

    def _init_qkv(self):
        in_dim = self.n_embed
        q_out_dim = self.q_head_num_ * self.head_dim
        self.q_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim],
            weight_names=self._q_weight_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_name,
            quant_method=self.get_quant_method("q_proj"),
        )
        self.kv_proj = KVROWNMMWeight(
            in_dim=in_dim,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("kv_proj"),
        )
        return

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
        return

    def _init_ffn(self):
        self.gate_up_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_inter, self.n_inter],
            weight_names=[self._gate_weight_name, self._up_weight_name],
            data_type=self.data_type_,
            bias_names=[self._gate_bias_name, self._up_bias_name],
            quant_method=self.get_quant_method("gate_up_proj"),
        )
        self.down_proj = COLMMWeight(
            in_dim=self.n_inter,
            out_dims=[self.n_embed],
            weight_names=self._down_weight_name,
            data_type=self.data_type_,
            bias_names=self._down_bias_name,
            quant_method=self.get_quant_method("down_proj"),
        )
        return

    def _init_norm(self):
        super()._init_norm()
        self.q_norm_weight_ = RMSNormWeight(
            dim=self.head_dim,
            weight_name=self._q_norm_name,
            data_type=self.data_type_,
        )
        self.k_norm_weight_ = RMSNormWeight(
            dim=self.head_dim,
            weight_name=self._k_norm_name,
            data_type=self.data_type_,
        )
        return
