from __future__ import annotations

from lightllm.common.basemodel.layer_weights.meta_weights import (
    KVROWNMMWeight,
    NoTpGEMMANormWeight,
    QKVROWNMMWeight,
    ROWMMWeight,
)
from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import (
    Gemma_2bTransformerLayerWeight,
)


class Pi0VLMTransformerLayerWeight(Gemma_2bTransformerLayerWeight):
    """PaliGemma weights using the normal LightLLM Gemma/Llama layout."""

    def _init_weight_names(self):
        prefix = "paligemma_with_expert.paligemma.model.language_model.layers." f"{self.layer_num_}"
        self._q_weight_name = f"{prefix}.self_attn.q_proj.weight"
        self._q_bias_name = None
        self._k_weight_name = f"{prefix}.self_attn.k_proj.weight"
        self._k_bias_name = None
        self._v_weight_name = f"{prefix}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._o_weight_name = f"{prefix}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._gate_weight_name = f"{prefix}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"{prefix}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._down_weight_name = f"{prefix}.mlp.down_proj.weight"
        self._down_bias_name = None
        self._att_norm_weight_name = f"{prefix}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"{prefix}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

    def _init_qkv(self):
        self.q_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.q_head_num_ * self.head_dim],
            weight_names=self._q_weight_name,
            data_type=self.data_type_,
            bias_names=None,
            quant_method=self.get_quant_method("q_proj"),
        )
        # KVROWNMMWeight contains LightLLM's standard replicated-KV policy for
        # the Gemma single-KV-head / multi-TP case.
        self.kv_proj = KVROWNMMWeight(
            in_dim=self.n_embed,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[None, None],
            quant_method=self.get_quant_method("kv_proj"),
        )

    def _init_norm(self):
        self.att_norm_weight_ = NoTpGEMMANormWeight(
            dim=self.n_embed,
            weight_name=self._att_norm_weight_name,
            data_type=self.data_type_,
        )
        self.ffn_norm_weight_ = NoTpGEMMANormWeight(
            dim=self.n_embed,
            weight_name=self._ffn_norm_weight_name,
            data_type=self.data_type_,
        )


class Pi0ActionTransformerLayerWeight(Gemma_2bTransformerLayerWeight):
    """Action-expert weights backed entirely by quantizable meta weights."""

    def _init_weight_names(self):
        prefix = "paligemma_with_expert.gemma_expert.model.layers." f"{self.layer_num_}"
        self._q_weight_name = f"{prefix}.self_attn.q_proj.weight"
        self._k_weight_name = f"{prefix}.self_attn.k_proj.weight"
        self._v_weight_name = f"{prefix}.self_attn.v_proj.weight"
        self._o_weight_name = f"{prefix}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._gate_weight_name = f"{prefix}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"{prefix}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._down_weight_name = f"{prefix}.mlp.down_proj.weight"
        self._down_bias_name = None
        self._att_norm_weight_name = f"{prefix}.input_layernorm.weight"
        self._ffn_norm_weight_name = f"{prefix}.post_attention_layernorm.weight"
        self._att_norm_dense_weight_name = f"{prefix}.input_layernorm.dense.weight"
        self._att_norm_dense_bias_name = f"{prefix}.input_layernorm.dense.bias"
        self._ffn_norm_dense_weight_name = f"{prefix}.post_attention_layernorm.dense.weight"
        self._ffn_norm_dense_bias_name = f"{prefix}.post_attention_layernorm.dense.bias"

    def _init_qkv(self):
        self.qkv_proj = QKVROWNMMWeight(
            in_dim=self.n_embed,
            q_head_num=self.q_head_num_,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[
                self._q_weight_name,
                self._k_weight_name,
                self._v_weight_name,
            ],
            data_type=self.data_type_,
            bias_names=[None, None, None],
            quant_method=self.get_quant_method("qkv_proj"),
        )
        self.q_width, self.kv_width, value_width = self.qkv_proj.out_dims
        if self.kv_width != value_width:
            raise ValueError("pi0 key/value projection widths must match")
        self.local_q_heads = self.q_width // self.head_dim
        self.local_kv_heads = self.kv_width // self.head_dim

    def _init_norm(self):
        if self.network_config_["is_pi05"]:
            self.att_norm_weight_ = None
            self.ffn_norm_weight_ = None
            self.att_norm_dense = ROWMMWeight(
                in_dim=self.n_embed,
                out_dims=[3 * self.n_embed],
                weight_names=self._att_norm_dense_weight_name,
                data_type=self.data_type_,
                bias_names=self._att_norm_dense_bias_name,
                quant_method=self.get_quant_method("att_norm_dense"),
                tp_rank=0,
                tp_world_size=1,
            )
            self.ffn_norm_dense = ROWMMWeight(
                in_dim=self.n_embed,
                out_dims=[3 * self.n_embed],
                weight_names=self._ffn_norm_dense_weight_name,
                data_type=self.data_type_,
                bias_names=self._ffn_norm_dense_bias_name,
                quant_method=self.get_quant_method("ffn_norm_dense"),
                tp_rank=0,
                tp_world_size=1,
            )
        else:
            self.att_norm_dense = None
            self.ffn_norm_dense = None
            self.att_norm_weight_ = NoTpGEMMANormWeight(
                dim=self.n_embed,
                weight_name=self._att_norm_weight_name,
                data_type=self.data_type_,
            )
            self.ffn_norm_weight_ = NoTpGEMMANormWeight(
                dim=self.n_embed,
                weight_name=self._ffn_norm_weight_name,
                data_type=self.data_type_,
            )
