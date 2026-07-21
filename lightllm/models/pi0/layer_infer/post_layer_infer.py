from __future__ import annotations

import torch

from lightllm.common.basemodel import PostLayerInferTpl
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.pi0.layer_weights.pre_and_post_layer_weight import (
    Pi0ActionPreAndPostLayerWeight,
)


class Pi0ActionPostLayerInfer(PostLayerInferTpl):
    def token_forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor | None,
        action_horizon: int,
        action_dim: int,
        weight: Pi0ActionPreAndPostLayerWeight,
    ) -> torch.Tensor:
        if weight.final_norm_dense is not None:
            normalized = rmsnorm_forward(
                hidden_states,
                None,
                1e-6,
                out=torch.empty_like(hidden_states, dtype=torch.float32),
            )
            modulation = weight.final_norm_dense.mm(condition)
            if hidden_states.ndim == 3:
                modulation = modulation[:, None, :]
            scale, shift, _ = modulation.chunk(3, dim=-1)
            hidden_states = (normalized.float() * (1.0 + scale.float()) + shift.float()).to(hidden_states.dtype)
        else:
            hidden_states = weight.final_norm_weight_(
                input=hidden_states,
                eps=1e-6,
                alloc_func=torch.empty,
            )
        action_hidden = hidden_states[:, -action_horizon:]
        actions = weight.action_out_proj.mm(action_hidden.reshape(-1, action_hidden.shape[-1])).view(
            *action_hidden.shape[:-1], -1
        )
        actions = actions.float()
        return actions[..., :action_dim]
