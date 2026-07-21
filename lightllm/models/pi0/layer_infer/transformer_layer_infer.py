from __future__ import annotations

import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import (
    Gemma_2bTransformerLayerInfer,
)
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.pi0.infer_struct import Pi0ActionInferStateInfo
from lightllm.models.pi0.layer_weights.transformer_layer_weight import (
    Pi0ActionTransformerLayerWeight,
)


class Pi0ActionTransformerLayerInfer(Gemma_2bTransformerLayerInfer):
    """Gemma expert layer with π₀'s adaptive norm and block mask policy."""

    def __init__(self, layer_num: int, network_config: dict):
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]

    def _norm(
        self,
        hidden_states: torch.Tensor,
        weight: Pi0ActionTransformerLayerWeight,
        condition: torch.Tensor | None,
        *,
        post_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dense = weight.ffn_norm_dense if post_attention else weight.att_norm_dense
        if dense is not None:
            if condition is None:
                raise ValueError("pi0.5 expert layer requires timestep condition")
            normalized = rmsnorm_forward(
                hidden_states,
                None,
                self.eps_,
                out=self.alloc_tensor(hidden_states.shape, torch.float32),
            )
            modulation = dense.mm(condition)
            if hidden_states.ndim == 3:
                modulation = modulation[:, None, :]
            scale, shift, gate = modulation.chunk(3, dim=-1)
            output = normalized.float() * (1.0 + scale.float()) + shift.float()
            return output.to(hidden_states.dtype), gate.to(hidden_states.dtype)

        norm = weight.ffn_norm_weight_ if post_attention else weight.att_norm_weight_
        return norm(input=hidden_states, eps=self.eps_, alloc_func=self.alloc_tensor), None

    def _get_qkv(
        self,
        hidden_states: torch.Tensor,
        infer_state: Pi0ActionInferStateInfo,
        weight: Pi0ActionTransformerLayerWeight,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return query, torch.cat([key, value], dim=1)

    def _action_attention(
        self,
        query: torch.Tensor,
        infer_state: Pi0ActionInferStateInfo,
    ) -> torch.Tensor:
        key, value = infer_state.mem_manager.get_att_input_params(self.layer_num_)
        return infer_state.prefill_att_state.prefill_att(
            q=query.contiguous(),
            k=key,
            v=value,
            alloc_func=self.alloc_tensor,
        )

    def context_forward(
        self,
        hidden_states: torch.Tensor,
        infer_state: Pi0ActionInferStateInfo,
        weight: Pi0ActionTransformerLayerWeight,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized, attention_gate = self._norm(
            hidden_states,
            weight,
            infer_state.condition,
            post_attention=False,
        )
        query, cache_kv = self._get_qkv(
            normalized,
            infer_state,
            weight,
        )
        infer_state.mem_manager.operator.copy_kv_to_mem_manager(
            layer_index=self.layer_num_,
            mem_index=infer_state.mem_index,
            kv=cache_kv,
        )

        batch_size, suffix_length = hidden_states.shape[:2]
        query = query.view(
            batch_size,
            suffix_length,
            weight.local_q_heads,
            weight.head_dim,
        )
        if infer_state.state_infer_state is None:
            attention_output = self._action_attention(
                query.reshape(-1, weight.local_q_heads, weight.head_dim),
                infer_state,
            )
        else:
            state_output = self._action_attention(
                query[:, :1].reshape(-1, weight.local_q_heads, weight.head_dim),
                infer_state.state_infer_state,
            ).view(batch_size, 1, weight.local_q_heads, weight.head_dim)
            action_output = self._action_attention(
                query[:, 1:].reshape(-1, weight.local_q_heads, weight.head_dim),
                infer_state,
            ).view(
                batch_size,
                suffix_length - 1,
                weight.local_q_heads,
                weight.head_dim,
            )
            attention_output = torch.cat([state_output, action_output], dim=1)

        projected = self._get_o(
            attention_output.reshape(-1, weight.q_width),
            infer_state,
            weight,
        ).view_as(residual)
        hidden_states = residual + projected if attention_gate is None else residual + projected * attention_gate

        residual = hidden_states
        normalized, mlp_gate = self._norm(
            hidden_states,
            weight,
            infer_state.condition,
            post_attention=True,
        )
        mlp_output = self._ffn(normalized, infer_state, weight).view_as(residual)
        return residual + mlp_output if mlp_gate is None else residual + mlp_output * mlp_gate
