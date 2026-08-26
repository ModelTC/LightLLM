from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import KVROWNMMWeight, ROWMMWeight
from lightllm.common.basemodel.layer_weights.meta_weights.norm_weight import QKRMSNORMWeight, RMSNormWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Qwen3EagleTransformerLayerWeight(LlamaTransformerLayerWeight):
    def _init_weight_names(self):
        super()._init_weight_names()
        weight_prefix = f"layers.{self.layer_num_}"
        self._q_weight_name = f"{weight_prefix}.self_attn.q_proj.weight"
        self._k_weight_name = f"{weight_prefix}.self_attn.k_proj.weight"
        self._v_weight_name = f"{weight_prefix}.self_attn.v_proj.weight"
        self._kv_weight_name = f"{weight_prefix}.self_attn.kv_proj.weight"
        self._o_weight_name = f"{weight_prefix}.self_attn.o_proj.weight"

        self._gate_weight_name = f"{weight_prefix}.mlp.gate_proj.weight"
        self._up_weight_name = f"{weight_prefix}.mlp.up_proj.weight"
        self._down_weight_name = f"{weight_prefix}.mlp.down_proj.weight"
        self._gate_up_bias_name = None

        self._att_norm_weight_name = f"{weight_prefix}.input_layernorm.weight"
        self._ffn_norm_weight_name = f"{weight_prefix}.post_attention_layernorm.weight"
        self._hidden_norm_weight_name = f"{weight_prefix}.hidden_norm.weight"
        self._q_norm_name = f"{weight_prefix}.self_attn.q_norm.weight"
        self._k_norm_name = f"{weight_prefix}.self_attn.k_norm.weight"

    def _init_qkv(self):
        in_dim = self.n_embed * 2
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

    def _init_norm(self):
        super()._init_norm()
        hidden_size = self.network_config_["hidden_size"]
        self.hidden_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._hidden_norm_weight_name,
            data_type=self.data_type_,
        )
        self.qk_norm_weight_ = QKRMSNORMWeight(
            dim=self.head_dim,
            q_weight_name=self._q_norm_name,
            k_weight_name=self._k_norm_name,
            data_type=self.data_type_,
        )
