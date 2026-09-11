from __future__ import annotations

from lightllm.common.basemodel.layer_weights.meta_weights import (
    KVROWNMMWeight,
    QKVROWNMMWeight,
    ROWMMWeight,
)
from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import (
    Gemma_2bTransformerLayerWeight,
)


class Pi0VLMTransformerLayerWeight(Gemma_2bTransformerLayerWeight):
    """PaliGemma weights using the normal LightLLM Gemma/Llama layout."""

    checkpoint_layer_prefix = (
        "paligemma_with_expert.paligemma.model.language_model.layers"
    )

    def _init_weight_names(self):
        super()._init_weight_names()
        prefix = f"{self.checkpoint_layer_prefix}.{self.layer_num_}"
        self._q_weight_name = f"{prefix}.self_attn.q_proj.weight"
        self._k_weight_name = f"{prefix}.self_attn.k_proj.weight"
        self._v_weight_name = f"{prefix}.self_attn.v_proj.weight"
        self._o_weight_name = f"{prefix}.self_attn.o_proj.weight"
        self._gate_weight_name = f"{prefix}.mlp.gate_proj.weight"
        self._up_weight_name = f"{prefix}.mlp.up_proj.weight"
        self._down_weight_name = f"{prefix}.mlp.down_proj.weight"
        self._att_norm_weight_name = f"{prefix}.input_layernorm.weight"
        self._ffn_norm_weight_name = f"{prefix}.post_attention_layernorm.weight"

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


class Pi0ActionTransformerLayerWeight(Pi0VLMTransformerLayerWeight):
    """Action-expert weights backed entirely by quantizable meta weights."""

    checkpoint_layer_prefix = "paligemma_with_expert.gemma_expert.model.layers"

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
