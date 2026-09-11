from __future__ import annotations

import torch

from lightllm.common.basemodel import PostLayerInferTpl
from lightllm.models.pi0.layer_weights.pre_and_post_layer_weight import (
    Pi0ActionPreAndPostLayerWeight,
)


class Pi0ActionPostLayerInfer(PostLayerInferTpl):
    def token_forward(
        self,
        hidden_states: torch.Tensor,
        action_horizon: int,
        weight: Pi0ActionPreAndPostLayerWeight,
    ) -> torch.Tensor:
        hidden_states = weight.final_norm_weight_(
            input=hidden_states,
            eps=1e-6,
            alloc_func=torch.empty,
        )
        batch_size = hidden_states.shape[0] // (action_horizon + 1)
        action_hidden = hidden_states.view(batch_size, action_horizon + 1, -1)[:, 1:]
        actions = weight.action_out_proj.mm(
            action_hidden.reshape(-1, action_hidden.shape[-1])
        ).view(*action_hidden.shape[:-1], -1)
        return actions.float()
