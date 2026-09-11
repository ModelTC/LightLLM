import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.pi0.infer_struct import Pi0ActionInferStateInfo
from lightllm.models.pi0.layer_infer.transformer_layer_infer import (
    Pi0ActionTransformerLayerInfer,
)
from lightllm.models.pi05.layer_weights import Pi05ActionTransformerLayerWeight


def adaptive_rmsnorm(
    hidden_states: torch.Tensor,
    condition: torch.Tensor,
    dense,
    *,
    eps: float,
    alloc_tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Pi0.5's shared scale/shift/gate RMSNorm modulation."""

    normalized = rmsnorm_forward(
        hidden_states,
        None,
        eps,
        out=alloc_tensor(hidden_states.shape, torch.float32),
    )
    scale, shift, gate = dense.mm(condition).chunk(3, dim=-1)
    batch_size = condition.shape[0]
    suffix_length = hidden_states.shape[0] // batch_size
    normalized = normalized.view(batch_size, suffix_length, -1)
    output = (
        normalized.float() * (1.0 + scale[:, None].float()) + shift[:, None].float()
    )
    return (
        output.to(hidden_states.dtype).view_as(hidden_states),
        gate[:, None].to(hidden_states.dtype),
    )


class Pi05ActionTransformerLayerInfer(Pi0ActionTransformerLayerInfer):
    """Pi0.5 expert layer with fixed timestep-conditioned AdaRMSNorm."""

    def context_forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        infer_state: Pi0ActionInferStateInfo,
        weight: Pi05ActionTransformerLayerWeight,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized, attention_gate = adaptive_rmsnorm(
            hidden_states,
            condition,
            weight.att_norm_dense,
            eps=self.eps_,
            alloc_tensor=self.alloc_tensor,
        )
        query, key, value = self._get_qkv(normalized, infer_state, weight)

        batch_size = infer_state.batch_size
        suffix_length = infer_state.suffix_length
        attention_output = self._attention(
            query,
            key,
            value,
            infer_state,
        )
        projected = self._get_o(
            attention_output.view(-1, weight.q_width),
            infer_state,
            weight,
        )
        projected = (
            projected.view(batch_size, suffix_length, -1) * attention_gate
        ).view_as(residual)
        hidden_states = residual + projected

        residual = hidden_states
        normalized, mlp_gate = adaptive_rmsnorm(
            hidden_states,
            condition,
            weight.ffn_norm_dense,
            eps=self.eps_,
            alloc_tensor=self.alloc_tensor,
        )
        mlp_output = self._ffn(normalized, infer_state, weight)
        mlp_output = (
            mlp_output.view(batch_size, suffix_length, -1) * mlp_gate
        ).view_as(residual)
        return residual + mlp_output
