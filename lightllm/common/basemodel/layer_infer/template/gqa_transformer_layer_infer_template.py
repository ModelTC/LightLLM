import torch

from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs, gqa_kv_quant_fields
from lightllm.common.attention_state.full_att import FullAttentionState
from lightllm.common.basemodel.attention_backend_manager import LayerAttentionConfig, attention_is_causal
from ...infer_struct import InferStateInfo
from .transformer_layer_infer_template import TransformerLayerInferTpl


class GQATransformerLayerInferTpl(TransformerLayerInferTpl):
    """A transformer layer whose attention is GQA over its full-attention KV cache.

    Prefill and decode use one GQA config by default. A model adds its own layer attributes through
    _gqa_config(model, **fields), per-layer weights through _gqa_sinks and _gqa_alibi_slopes, and a
    differently laid out cache view through _gqa_kv_cache.
    """

    def attention_config(self, model) -> LayerAttentionConfig:
        config = self._gqa_config(model)
        return LayerAttentionConfig(prefill=config, decode=config)

    def _gqa_cache_state(self, manager, model_role) -> FullAttentionState:
        return manager.get_state(
            model_role=model_role,
            state_type_filter="gqa",
            logical_layer_id=self.layer_num_,
            state_class=FullAttentionState,
        )

    def _gqa_kv_cache(self, state: FullAttentionState):
        """This layer's (keys, values) cache views."""
        return state.get_layer_view(logical_layer_id=self.layer_num_)

    def _gqa_config(self, model, **fields) -> GQALayerConfig:
        state = self._gqa_cache_state(model.att_state_manager, model.att_state_role)
        keys, _ = self._gqa_kv_cache(state)
        scales = state.get_layer_kv_scales(self.layer_num_)
        return GQALayerConfig(
            num_heads=self.tp_q_head_num_,
            num_kv_heads=keys.shape[1],
            head_dim=self.head_dim_,
            dtype=model.data_type,
            causal=attention_is_causal(model),
            **gqa_kv_quant_fields(model.args.llm_kv_type, scales, self.head_dim_),
            **fields,
        )

    def _gqa_sinks(self, layer_weight):
        return None

    def _gqa_alibi_slopes(self, layer_weight):
        return None

    def _gqa_attention(self, backend, q, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        state = self._gqa_cache_state(infer_state.att_state_manager, infer_state.att_state_role)
        keys, values = self._gqa_kv_cache(state)
        config = backend.layer_config
        query = q.view(-1, config.num_heads, config.head_dim)
        inputs = GQALayerInputs(
            q=query,
            k_cache=keys,
            v_cache=values,
            kv_scales=state.get_layer_kv_scales(self.layer_num_),
            sinks=self._gqa_sinks(layer_weight),
            alibi_slopes=self._gqa_alibi_slopes(layer_weight),
        )
        output = backend.forward(
            infer_state.att_states[backend.state_key],
            inputs,
            out=self.alloc_tensor(query.shape, dtype=query.dtype, device=query.device),
            alloc_func=self.alloc_tensor,
        )
        return output.view(q.shape)

    def _context_attention_kernel(self, q, kv, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        return self._gqa_attention(self.prefill_backend, q, infer_state, layer_weight)

    def _token_attention_kernel(self, q, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        return self._gqa_attention(self.decode_backend, q, infer_state, layer_weight)
