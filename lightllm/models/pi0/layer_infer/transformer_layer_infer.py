from __future__ import annotations

import torch

from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import (
    Gemma_2bTransformerLayerInfer,
)
from lightllm.models.gemma_2b.triton_kernel.gelu_and_mul import gelu_and_mul_fwd
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.pi0.infer_struct import Pi0ActionInferStateInfo
from lightllm.models.pi0.layer_weights.transformer_layer_weight import (
    Pi0ActionTransformerLayerWeight,
)


class Pi0ActionTransformerLayerInfer(Gemma_2bTransformerLayerInfer):
    """Pi0 expert layer with a continuous-state suffix and Gemma RMSNorm."""

    def __init__(self, layer_num: int, network_config: dict):
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]

    def _get_qkv(
        self,
        hidden_states: torch.Tensor,
        infer_state: Pi0ActionInferStateInfo,
        weight: Pi0ActionTransformerLayerWeight,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = weight.qkv_proj.mm(hidden_states.reshape(-1, self.embed_dim_))
        query, key, value = torch.split(
            qkv,
            [weight.q_width, weight.kv_width, weight.kv_width],
            dim=-1,
        )
        query = query.view(-1, weight.local_q_heads, weight.head_dim)
        key = key.view(-1, weight.local_kv_heads, weight.head_dim)
        value = value.view(-1, weight.local_kv_heads, weight.head_dim)
        rotary_emb_fwd(
            query,
            key,
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return query, key, value

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        infer_state: Pi0ActionInferStateInfo,
    ) -> torch.Tensor:
        prefix = infer_state.prefix_kv.layer(self.layer_num_)
        return infer_state.prefill_att_state.prefill_att(
            query,
            prefix.key,
            prefix.value,
            suffix_k=key,
            suffix_v=value,
            alloc_func=self.alloc_tensor,
        )

    def _ffn(
        self,
        hidden_states: torch.Tensor,
        infer_state: Pi0ActionInferStateInfo,
        weight: Pi0ActionTransformerLayerWeight,
    ) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.embed_dim_)
        hidden_states = self._tpsp_allgather(
            input=hidden_states,
            infer_state=infer_state,
        )
        gate_up = weight.gate_up_proj.mm(hidden_states)
        activated = self.alloc_tensor(
            (hidden_states.shape[0], gate_up.shape[1] // 2),
            hidden_states.dtype,
        )
        gelu_and_mul_fwd(gate_up, activated, block_m=32, block_n=64)
        output = weight.down_proj.mm(activated)
        return self._tpsp_reduce(input=output, infer_state=infer_state)

    def context_forward(
        self,
        hidden_states: torch.Tensor,
        infer_state: Pi0ActionInferStateInfo,
        weight: Pi0ActionTransformerLayerWeight,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self._att_norm(hidden_states, infer_state, weight)
        query, key, value = self._get_qkv(normalized, infer_state, weight)

        attention_output = self._attention(query, key, value, infer_state)

        projected = self._get_o(
            attention_output.view(-1, weight.q_width), infer_state, weight
        )
        hidden_states = residual + projected

        residual = hidden_states
        normalized = self._ffn_norm(hidden_states, infer_state, weight)
        mlp_output = self._ffn(normalized, infer_state, weight).view_as(residual)
        return residual + mlp_output
