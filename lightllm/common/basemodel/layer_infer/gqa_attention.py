from typing import Any, Optional

import torch

from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs
from lightllm.common.attention.layer import Attention
from lightllm.common.attention_state.full_att import FullAttentionState
from lightllm.common.basemodel.attention_groups import attention_is_causal, gqa_kv_quant_fields


class GQAAttentionLayerMixin:
    """Grouped-query attention of a transformer layer over its full-attention KV cache.

    The layer sets layer_num_, tp_q_head_num_ and head_dim_, and overrides the hooks below for its
    static attention properties (sliding window, sinks, ALiBi), its per-layer weights and the cache
    view it reads. init_attention creates the layer's Attention once the cache is allocated;
    _context_attention_kernel and _token_attention_kernel run it with the step args of its group.
    """

    # Whether prefill attends bidirectionally inside image spans (InferStateInfo.b_image_token_end).
    prefill_image_spans = False
    # False for layers that only write their KV cache in prefill and attend only in decode.
    attends_in_prefill = True
    attn: Optional[Attention] = None

    def _full_att_state(self, att_state_manager, att_state_role: str) -> FullAttentionState:
        return att_state_manager.get_state(
            model_role=att_state_role,
            state_type_filter="gqa",
            logical_layer_id=self.layer_num_,
            state_class=FullAttentionState,
        )

    def _gqa_kv_cache(self, state: FullAttentionState) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer's K and V cache views, [cache_slots, num_kv_heads, head_dim]."""
        return state.get_layer_view(logical_layer_id=self.layer_num_)

    def _gqa_layer_config_fields(self) -> dict[str, Any]:
        """GQALayerConfig fields of the layer's own attention, such as sliding_window, has_sinks and has_alibi."""
        return {}

    def _gqa_layer_weights(self, layer_weight) -> dict[str, Any]:
        """GQALayerInputs fields from the layer's weights, such as sinks and alibi_slopes."""
        return {}

    def init_attention(self, model) -> None:
        state = self._full_att_state(model.att_state_manager, model.att_state_role)
        k, _ = self._gqa_kv_cache(state)
        config = dict(
            num_heads=self.tp_q_head_num_,
            num_kv_heads=k.shape[1],
            head_dim=self.head_dim_,
            dtype=model.data_type,
            causal=attention_is_causal(model),
            **gqa_kv_quant_fields(model.args.llm_kv_type, state.get_layer_kv_scales(self.layer_num_), self.head_dim_),
            **self._gqa_layer_config_fields(),
        )
        prefill_config = None
        if self.attends_in_prefill:
            prefill_config = GQALayerConfig(**config, bidirectional_image_spans=self.prefill_image_spans)
        self.attn = model.att_groups.create_attention(prefill_config, GQALayerConfig(**config))

    def _gqa_attention(self, q: torch.Tensor, infer_state, layer_weight, *, prefill: bool) -> torch.Tensor:
        state = self._full_att_state(infer_state.att_state_manager, infer_state.att_state_role)
        k, v = self._gqa_kv_cache(state)
        inputs = GQALayerInputs(
            q=q.view(-1, self.tp_q_head_num_, self.head_dim_),
            k_cache=k,
            v_cache=v,
            kv_scales=state.get_layer_kv_scales(self.layer_num_),
            **self._gqa_layer_weights(layer_weight),
        )
        run = self.attn.prefill if prefill else self.attn.decode
        o_tensor = run(
            infer_state.att_step_args[self.attn.group],
            inputs,
            out=self.alloc_tensor(inputs.q.shape, dtype=inputs.q.dtype, device=inputs.q.device),
            alloc_func=self.alloc_tensor,
        )
        return o_tensor.view(q.shape)

    def _context_attention_kernel(self, q: torch.Tensor, kv, infer_state, layer_weight, out=None) -> torch.Tensor:
        return self._gqa_attention(q, infer_state, layer_weight, prefill=True)

    def _token_attention_kernel(self, q: torch.Tensor, infer_state, layer_weight, out=None) -> torch.Tensor:
        return self._gqa_attention(q, infer_state, layer_weight, prefill=False)
